// TTS compute: token generation / sampling / embedding — compute layer: model/graph execution. No frame/queue/worker knowledge.
// ==========================================================================

#include "omni-impl.h"
#include "omni.h"
#include "omni_internal.h"
#include "token2wav/token2wav-impl.h"
#include "common/common.h"
#include "common/sampling.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

bool load_tts_weights_from_gguf(struct omni_context * ctx_omni, const char * tts_model_path) {

    auto ggml_tensor_to_f32 = [](const ggml_tensor * t, float * dst, int64_t n) {
        if (t->type == GGML_TYPE_F32) {
            memcpy(dst, t->data, (size_t)n * sizeof(float));
            return;
        }
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(t->data, dst, n);
        } else {
            LOG_ERR("TTS: no to_float converter for ggml type %d\n", (int)t->type);
        }
    };

    // Initialize GGUF context
    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_meta,
    };
    
    struct gguf_context * ctx_gguf = gguf_init_from_file(tts_model_path, params);
    if (!ctx_gguf) {
        LOG_ERR("TTS: Failed to load GGUF file: %s\n", tts_model_path);
        return false;
    }
    
    // Load emb_code.0.weight: (num_audio_tokens=6562, hidden_size=768)
    // This is used to convert audio token IDs to embeddings during decode phase
    const char * emb_code_name = "emb_code.0.weight";
    int64_t emb_code_idx = gguf_find_tensor(ctx_gguf, emb_code_name);
    if (emb_code_idx >= 0) {
        struct ggml_tensor * emb_code_tensor = ggml_get_tensor(ctx_meta, emb_code_name);
        if (emb_code_tensor) {
            // emb_code is Embedding(num_audio_tokens, hidden_size)
            // In PyTorch: weight shape is (num_audio_tokens, hidden_size) = [6562, 768]
            // In GGUF: stored as (hidden_size, num_audio_tokens) = [768, 6562] (transposed)
            int64_t dim0 = emb_code_tensor->ne[0];
            int64_t dim1 = emb_code_tensor->ne[1];
            
            // Determine which dimension is which based on expected values
            int64_t num_audio_tokens = 6562;
            int64_t hidden_size = 768;
            
            // GGUF stores as (hidden_size, num_audio_tokens) = [768, 6562]
            if (dim0 == hidden_size && dim1 == num_audio_tokens) {
                // Correct: stored as (hidden_size, num_audio_tokens)
            } else if (dim0 == num_audio_tokens && dim1 == hidden_size) {
                // Stored as (num_audio_tokens, hidden_size) - need to transpose
                num_audio_tokens = dim0;
                hidden_size = dim1;
            } else {
                LOG_ERR("TTS: emb_code.0.weight has unexpected shape [%ld, %ld], expected [768, 6562] or [6562, 768]\n", dim0, dim1);
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Allocate memory and copy data
            size_t emb_code_size = dim0 * dim1 * sizeof(float);
            ctx_omni->emb_code_weight = (float *)malloc(emb_code_size);
            if (!ctx_omni->emb_code_weight) {
                LOG_ERR("TTS: Failed to allocate memory for emb_code.0.weight\n");
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Copy/convert tensor data based on type
            enum ggml_type emb_code_type = emb_code_tensor->type;
            int64_t emb_code_elements = dim0 * dim1;
            
            ggml_tensor_to_f32(emb_code_tensor, ctx_omni->emb_code_weight, dim0 * dim1);
            
            ctx_omni->emb_code_vocab_size = num_audio_tokens;  // 6562
            ctx_omni->emb_code_hidden_size = hidden_size;     // 768
            // NOTE: GGUF data is stored in row-major order, matching NumPy/PyTorch
            // Even when metadata shape is [768, 6562], the actual data layout is (6562, 768) row-major
            // So we should access as: weight[token_idx * hidden_size + j]
            // This means: emb_code_stored_as_transposed should be FALSE
            ctx_omni->emb_code_stored_as_transposed = false; // Data is always (vocab_size, hidden_size) in memory
        } else {
            LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", emb_code_name);
            ggml_free(ctx_meta);
            gguf_free(ctx_gguf);
            return false;
        }
    } else {
        LOG_ERR("TTS: Tensor %s not found in GGUF file (this is OK if using token IDs)\n", emb_code_name);
        // Note: emb_code is optional if we use token IDs, but we prefer embeddings
    }
    
    // Load emb_text.weight: (vocab_size=152064, hidden_size=768)
    // PyTorch: nn.Embedding(vocab_size, hidden_size) -> weight shape is [vocab_size, hidden_size] = [152064, 768]
    // GGUF: may be stored as [hidden_size, vocab_size] = [768, 152064] (transposed)
    const char * emb_text_name = "emb_text.weight";
    int64_t emb_text_idx = gguf_find_tensor(ctx_gguf, emb_text_name);
    if (emb_text_idx >= 0) {
        struct ggml_tensor * emb_text_tensor = ggml_get_tensor(ctx_meta, emb_text_name);
        if (emb_text_tensor) {
            int64_t dim0 = emb_text_tensor->ne[0];
            int64_t dim1 = emb_text_tensor->ne[1];
            
            // Expected values
            int64_t expected_vocab_size = 152064;
            int64_t expected_hidden_size = 768;
            
            // GGML tensor 维度理解：
            // ne[0] = 最内层维度 (stride=1) = hidden_size = 768
            // ne[1] = 外层维度 = vocab_size = 152064
            // 内存布局是 row-major，即 [vocab_size][hidden_size]
            // 所以 ne[0]=768, ne[1]=152064 意味着数据已经是 [vocab_size, hidden_size] 格式
            // 不需要转置！
            
            int64_t vocab_size, hidden_size;
            
            if (dim0 == expected_hidden_size && dim1 == expected_vocab_size) {
                // GGML shape: ne[0]=768, ne[1]=152064
                // 这意味着内存布局是 [vocab_size=152064][hidden_size=768]
                // 不需要转置
                vocab_size = dim1;   // 152064
                hidden_size = dim0;  // 768
            } else if (dim0 == expected_vocab_size && dim1 == expected_hidden_size) {
                // GGML shape: ne[0]=152064, ne[1]=768 (unusual)
                // 这意味着内存布局是 [hidden_size=768][vocab_size=152064]
                // 这种情况需要转置
                vocab_size = dim0;   // 152064
                hidden_size = dim1;  // 768
                LOG_ERR("TTS: emb_text.weight has unusual GGML shape, not handled\n");
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            } else {
                LOG_ERR("TTS: emb_text.weight has unexpected shape [%ld, %ld], expected ne=[%ld, %ld]\n", 
                       dim0, dim1, expected_hidden_size, expected_vocab_size);
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Allocate memory for the weight
            size_t emb_text_size = vocab_size * hidden_size * sizeof(float);
            ctx_omni->emb_text_weight = (float *)malloc(emb_text_size);
            if (!ctx_omni->emb_text_weight) {
                LOG_ERR("TTS: Failed to allocate memory for emb_text.weight\n");
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Copy/convert tensor data based on type
            enum ggml_type emb_text_type = emb_text_tensor->type;
            int64_t emb_text_elements = vocab_size * hidden_size;
            
            ggml_tensor_to_f32(emb_text_tensor, ctx_omni->emb_text_weight, vocab_size * hidden_size);
            
            ctx_omni->emb_text_vocab_size = vocab_size;   // 152064
            ctx_omni->emb_text_hidden_size = hidden_size;  // 768
        } else {
            LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", emb_text_name);
            ggml_free(ctx_meta);
            gguf_free(ctx_gguf);
            return false;
        }
    } else {
        LOG_ERR("TTS: Tensor %s not found in GGUF file\n", emb_text_name);
        ggml_free(ctx_meta);
        gguf_free(ctx_gguf);
        return false;
    }
    
    // Load projector_semantic weights
    const char * projector_names[] = {
        "projector_semantic.linear1.weight",
        "projector_semantic.linear1.bias",
        "projector_semantic.linear2.weight",
        "projector_semantic.linear2.bias"
    };
    
    float ** projector_ptrs[] = {
        &ctx_omni->projector_semantic_linear1_weight,
        &ctx_omni->projector_semantic_linear1_bias,
        &ctx_omni->projector_semantic_linear2_weight,
        &ctx_omni->projector_semantic_linear2_bias
    };
    
    // PyTorch nn.Linear(in_features, out_features) weight shape is (out_features, in_features)
    // GGUF may store as (out_features, in_features) or (in_features, out_features)
    // We need to detect and handle both cases
    int64_t expected_shapes_pytorch[][2] = {
        {768, 4096},  // linear1.weight: PyTorch shape (out_features, in_features)
        {768, 0},     // linear1.bias (1D)
        {768, 768},   // linear2.weight: PyTorch shape (out_features, in_features)
        {768, 0}      // linear2.bias (1D)
    };
    
    int64_t expected_shapes_transposed[][2] = {
        {4096, 768},  // linear1.weight: transposed shape (in_features, out_features)
        {768, 0},     // linear1.bias (1D)
        {768, 768},   // linear2.weight: same for square matrix
        {768, 0}      // linear2.bias (1D)
    };
    
    // Track whether weights need transposition
    bool need_transpose[2] = {false, false};  // [linear1, linear2]
    
    for (int i = 0; i < 4; i++) {
        int64_t tensor_idx = gguf_find_tensor(ctx_gguf, projector_names[i]);
        if (tensor_idx >= 0) {
            struct ggml_tensor * tensor = ggml_get_tensor(ctx_meta, projector_names[i]);
            if (tensor) {
                int64_t dim0 = tensor->ne[0];
                int64_t dim1 = (ggml_n_dims(tensor) > 1) ? tensor->ne[1] : 0;
                
                if (i % 2 == 0) {  // weight (2D)
                    // Check if stored as PyTorch shape (out_features, in_features) or transposed
                    bool is_pytorch_shape = (dim0 == expected_shapes_pytorch[i][0] && 
                                           dim1 == expected_shapes_pytorch[i][1]);
                    bool is_transposed_shape = (dim0 == expected_shapes_transposed[i][0] && 
                                               dim1 == expected_shapes_transposed[i][1]);
                    
                    if (is_pytorch_shape) {
                        // Stored as PyTorch shape (out_features, in_features), need to transpose
                        need_transpose[i / 2] = true;
                    } else if (is_transposed_shape) {
                        // Already transposed, use directly
                        need_transpose[i / 2] = false;
                    } else {
                        LOG_ERR("TTS: %s has unexpected shape: [%ld, %ld], expected [%ld, %ld] or [%ld, %ld]\n",
                               projector_names[i], dim0, dim1, 
                               expected_shapes_pytorch[i][0], expected_shapes_pytorch[i][1],
                               expected_shapes_transposed[i][0], expected_shapes_transposed[i][1]);
                        // Try to continue, assume PyTorch shape
                        need_transpose[i / 2] = true;
                    }
                } else {  // bias (1D)
                    if (dim0 != expected_shapes_pytorch[i][0] || dim1 != 0) {
                        LOG_ERR("TTS: %s has wrong shape: [%ld, %ld], expected [%ld, 0]\n",
                               projector_names[i], dim0, dim1, expected_shapes_pytorch[i][0]);
                    }
                }
                
                // Check tensor type for F16 conversion
                enum ggml_type proj_tensor_type = tensor->type;
                
                if (i % 2 == 0 && need_transpose[i / 2]) {
                    // Weight needs transposition: allocate transposed size (always F32 output)
                    int64_t in_dim = expected_shapes_pytorch[i][1];  // PyTorch in_features
                    int64_t out_dim = expected_shapes_pytorch[i][0];  // PyTorch out_features
                    size_t transposed_size = in_dim * out_dim * sizeof(float);
                    *projector_ptrs[i] = (float *)malloc(transposed_size);
                    if (!*projector_ptrs[i]) {
                        LOG_ERR("TTS: Failed to allocate memory for transposed %s\n", projector_names[i]);
                        // Clean up
                        for (int j = 0; j < i; j++) {
                            if (*projector_ptrs[j]) {
                                free(*projector_ptrs[j]);
                                *projector_ptrs[j] = nullptr;
                            }
                        }
                        ggml_free(ctx_meta);
                        gguf_free(ctx_gguf);
                        return false;
                    }
                    
                    // Transpose: src[out_dim][in_dim] -> dst[in_dim][out_dim]
                    float * dst_data = *projector_ptrs[i];
                    if (proj_tensor_type == GGML_TYPE_F32) {
                        const float * src_data = (const float *)tensor->data;
                        for (int64_t out = 0; out < out_dim; out++) {
                            for (int64_t in = 0; in < in_dim; in++) {
                                dst_data[in * out_dim + out] = src_data[out * in_dim + in];
                            }
                        }
                    } else {
                        std::vector<float> tmp(dim0 * dim1);
                        ggml_tensor_to_f32(tensor, tmp.data(), dim0 * dim1);
                        for (int64_t out = 0; out < out_dim; out++) {
                            for (int64_t in = 0; in < in_dim; in++) {
                                dst_data[in * out_dim + out] = tmp[out * in_dim + in];
                            }
                        }
                    }
                } else {
                    int64_t num_elements = (dim1 > 0) ? dim0 * dim1 : dim0;
                    size_t output_size = num_elements * sizeof(float);
                    *projector_ptrs[i] = (float *)malloc(output_size);
                    if (!*projector_ptrs[i]) {
                        LOG_ERR("TTS: Failed to allocate memory for %s\n", projector_names[i]);
                        for (int j = 0; j < i; j++) { free(*projector_ptrs[j]); *projector_ptrs[j] = nullptr; }
                        ggml_free(ctx_meta); gguf_free(ctx_gguf); return false;
                    }
                    ggml_tensor_to_f32(tensor, *projector_ptrs[i], num_elements);
                }
            } else {
                LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", projector_names[i]);
                // Clean up
                for (int j = 0; j < i; j++) {
                    if (*projector_ptrs[j]) {
                        free(*projector_ptrs[j]);
                        *projector_ptrs[j] = nullptr;
                    }
                }
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
        } else {
            LOG_ERR("TTS: Tensor %s not found in GGUF file\n", projector_names[i]);
            // Clean up
            for (int j = 0; j < i; j++) {
                if (*projector_ptrs[j]) {
                    free(*projector_ptrs[j]);
                    *projector_ptrs[j] = nullptr;
                }
            }
            ggml_free(ctx_meta);
            gguf_free(ctx_gguf);
            return false;
        }
    }
    
    // Set projector dimensions
    ctx_omni->projector_semantic_input_dim = 4096;
    ctx_omni->projector_semantic_output_dim = 768;
    
    // Load head_code.weight: (hidden_size=768, num_audio_tokens=6562)
    // Note: num_vq=1, so we only load head_code.0.weight
    const char * head_code_name = "head_code.0.weight";
    int64_t head_code_idx = gguf_find_tensor(ctx_gguf, head_code_name);
    if (head_code_idx >= 0) {
        struct ggml_tensor * head_code_tensor = ggml_get_tensor(ctx_meta, head_code_name);
        if (head_code_tensor) {
            // head_code is Linear(hidden_size, num_audio_tokens, bias=False)
            // In PyTorch: weight shape is (num_audio_tokens, hidden_size) = [6562, 768]
            // In GGUF: stored as (hidden_size, num_audio_tokens) = [768, 6562] (already transposed)
            int64_t dim0 = head_code_tensor->ne[0];
            int64_t dim1 = (ggml_n_dims(head_code_tensor) > 1) ? head_code_tensor->ne[1] : 0;
            
            // Expected shape in GGUF: (hidden_size=768, num_audio_tokens=6562)
            int64_t expected_hidden_size = 768;
            int64_t expected_num_audio_tokens = 6562;
            
            // Allocate memory for weight: (hidden_size, num_audio_tokens) = [768, 6562]
            size_t head_code_size = expected_hidden_size * expected_num_audio_tokens * sizeof(float);
            ctx_omni->head_code_weight = (float *)malloc(head_code_size);
            if (!ctx_omni->head_code_weight) {
                LOG_ERR("TTS: Failed to allocate memory for head_code.0.weight\n");
                // Clean up already loaded weights
                if (ctx_omni->emb_text_weight) {
                    free(ctx_omni->emb_text_weight);
                    ctx_omni->emb_text_weight = nullptr;
                }
                for (int j = 0; j < 4; j++) {
                    if (*projector_ptrs[j]) {
                        free(*projector_ptrs[j]);
                        *projector_ptrs[j] = nullptr;
                    }
                }
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // 布局假设（由下方运行校验确认）：转换脚本总是把 head_code 存成 [768, 6562]
            //（即 Linear 权重已转置），即使 GGUF metadata shape 记录为 [6562, 768]。
            // 因此拷贝循环固定按 [768, 6562] 读源数据（hidden 慢轴），不做运行时转置判断。
            // 若日后更换转换脚本/模型导致布局变化，此假设失效——
            // 靠下方 head_code_weight[0] 与 Python weight[0,0] 的对照打印发现。
            if (!(dim0 == expected_hidden_size && dim1 == expected_num_audio_tokens) &&
                !(dim0 == expected_num_audio_tokens && dim1 == expected_hidden_size)) {
                LOG_ERR("TTS: head_code.0.weight has unexpected shape [%ld, %ld], expected [%d, %d] or [%d, %d]\n",
                       dim0, dim1, expected_hidden_size, expected_num_audio_tokens, expected_num_audio_tokens, expected_hidden_size);
                // Clean up
                free(ctx_omni->head_code_weight);
                ctx_omni->head_code_weight = nullptr;
                if (ctx_omni->emb_text_weight) {
                    free(ctx_omni->emb_text_weight);
                    ctx_omni->emb_text_weight = nullptr;
                }
                for (int j = 0; j < 4; j++) {
                    if (*projector_ptrs[j]) {
                        free(*projector_ptrs[j]);
                        *projector_ptrs[j] = nullptr;
                    }
                }
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Check tensor type and copy/convert accordingly
            // 优化：转置存储为 [6562, 768]，使每个output token的权重连续存储
            // 这样在计算logits时可以用高效的向量点积
            // 原始: weight[j * 6562 + i] = W[j, i]  (j=hidden, i=output)
            // 转置后: weight[i * 768 + j] = W[i, j] (i=output, j=hidden)
            enum ggml_type tensor_type = head_code_tensor->type;
            
            print_with_timestamp("TTS: head_code shape: dim0=%ld, dim1=%ld, will transpose to [%ld, %ld]\n", 
                                dim0, dim1, expected_num_audio_tokens, expected_hidden_size);
            
            // Transpose: [hidden, audio_tokens] -> [audio_tokens, hidden]
            if (tensor_type == GGML_TYPE_F32) {
                const float * src_data = (const float *)head_code_tensor->data;
                for (int64_t i = 0; i < expected_num_audio_tokens; ++i) {
                    for (int64_t j = 0; j < expected_hidden_size; ++j) {
                        ctx_omni->head_code_weight[i * expected_hidden_size + j] = src_data[j * expected_num_audio_tokens + i];
                    }
                }
            } else {
                std::vector<float> tmp(expected_hidden_size * expected_num_audio_tokens);
                ggml_tensor_to_f32(head_code_tensor, tmp.data(), expected_hidden_size * expected_num_audio_tokens);
                for (int64_t i = 0; i < expected_num_audio_tokens; ++i) {
                    for (int64_t j = 0; j < expected_hidden_size; ++j) {
                        ctx_omni->head_code_weight[i * expected_hidden_size + j] = tmp[j * expected_num_audio_tokens + i];
                    }
                }
            }
            
            ctx_omni->head_code_hidden_size = expected_hidden_size;
            ctx_omni->head_code_num_audio_tokens = expected_num_audio_tokens;
            
            // 布局校验（守护上面的 [768, 6562] 假设）：若 head_code_weight[0] 与
            // Python weight[0,0]≈0.01385498 明显对不上，说明布局假设失效，需回看
            // 转换脚本或改拷贝循环索引。量化模型允许小幅偏差（如 Q4 下 0.0136 vs 0.0139）。
            // Python weight[0, 0:5] = [0.01385498, -0.01647949, 0.0111084, -0.01367188, -0.01141357]
            print_with_timestamp("TTS: head_code loaded, verifying first few values:\n");
            print_with_timestamp("  head_code_weight[0] = %.8f (expect ~0.01385498)\n", ctx_omni->head_code_weight[0]);
            print_with_timestamp("  head_code_weight[1] = %.8f (expect ~-0.01647949)\n", ctx_omni->head_code_weight[1]);
            print_with_timestamp("  head_code_weight[768] = %.8f (this is weight[1, 0], expect ~0.04150391)\n", ctx_omni->head_code_weight[768]);
        } else {
            LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", head_code_name);
            // Clean up
            if (ctx_omni->emb_text_weight) {
                free(ctx_omni->emb_text_weight);
                ctx_omni->emb_text_weight = nullptr;
            }
            for (int j = 0; j < 4; j++) {
                if (*projector_ptrs[j]) {
                    free(*projector_ptrs[j]);
                    *projector_ptrs[j] = nullptr;
                }
            }
            ggml_free(ctx_meta);
            gguf_free(ctx_gguf);
            return false;
        }
    } else {
        LOG_ERR("TTS: Tensor %s not found in GGUF file\n", head_code_name);
        // Clean up
        if (ctx_omni->emb_text_weight) {
            free(ctx_omni->emb_text_weight);
            ctx_omni->emb_text_weight = nullptr;
        }
        for (int j = 0; j < 4; j++) {
            if (*projector_ptrs[j]) {
                free(*projector_ptrs[j]);
                *projector_ptrs[j] = nullptr;
            }
        }
        ggml_free(ctx_meta);
        gguf_free(ctx_gguf);
        return false;
    }
    
    ggml_free(ctx_meta);
    gguf_free(ctx_gguf);
    return true;
}

// TODO: 实现TTS的emb_text层调用
// 功能：将token ID转换为embedding
// 输入：token_id (llama_token), TTS模型上下文
// 输出：embedding向量 (float*, 大小为tts_n_embd)
// 实现方式：
//   1. 从TTS模型文件中加载emb_text权重（152064 x 768）
//      - 在omni_init中加载emb_text权重到ctx_omni中
//      - 权重名称：根据GGUF格式，可能是"tts.emb_text.weight"或类似
//   2. 查找：embedding = emb_text_weight[token_id]
//   3. 返回embedding向量
// 注意：需要确保token_id在[0, 152064)范围内
// 
// 实现步骤：
// 1. 在omni_context中添加emb_text_weight字段（float*, 152064 * 768）
// 2. 在omni_init中从TTS模型文件加载emb_text权重
// 3. 在这里实现查找逻辑
bool tts_emb_text(struct omni_context * ctx_omni, llama_token token_id, float * embedding_out, int tts_n_embd) {
    // Check if weights are loaded
    if (!ctx_omni->emb_text_weight) {
        LOG_ERR("TTS: emb_text_weight not loaded\n");
        return false;
    }
    
    // Check token_id range
    if (token_id < 0 || token_id >= ctx_omni->emb_text_vocab_size) {
        LOG_ERR("TTS: token_id %d out of range [0, %d)\n", token_id, ctx_omni->emb_text_vocab_size);
        return false;
    }
    
    // Check embedding dimension
    if (tts_n_embd != ctx_omni->emb_text_hidden_size) {
        LOG_ERR("TTS: tts_n_embd (%d) != emb_text_hidden_size (%d)\n", tts_n_embd, ctx_omni->emb_text_hidden_size);
        return false;
    }
    
    // Copy embedding: embedding_out = emb_text_weight[token_id]
    const float * src = ctx_omni->emb_text_weight + token_id * tts_n_embd;
    memcpy(embedding_out, src, tts_n_embd * sizeof(float));
    
    return true;
}

// TTS的projector_semantic层实现
// 功能：将LLM hidden states（4096维）投影到TTS hidden_dim（768维）
// 输入：llm_hidden_states (float*, n_tokens * 4096), n_tokens, llm_n_embd (4096)
// 输出：projected_hidden_states (float*, n_tokens * 768), tts_n_embd (768)
// 
// 权重存储格式：
//   - PyTorch nn.Linear(in_features, out_features)的权重形状是(out_features, in_features)
//   - GGUF可能存储为(out_features, in_features)或(in_features, out_features)
//   - 加载时会自动检测并转置为(in_features, out_features)格式以便矩阵乘法
//   - linear1_weight: 转置后存储为(4096, 768) - 存储在ctx_omni->projector_semantic_linear1_weight
//   - linear1_bias: (768,) - 存储在ctx_omni->projector_semantic_linear1_bias
//   - linear2_weight: (768, 768) - 存储在ctx_omni->projector_semantic_linear2_weight
//   - linear2_bias: (768,) - 存储在ctx_omni->projector_semantic_linear2_bias
// 
// 矩阵乘法实现：
//   对每个token的hidden state：
//     hidden_proj = ReLU(linear1(hidden) + bias1)  // (1, 4096) @ (4096, 768) = (1, 768)
//     hidden_proj = linear2(hidden_proj) + bias2    // (1, 768) @ (768, 768) = (1, 768)
//   归一化在调用者中完成（使用normalize_l2_per_token）
bool tts_projector_semantic(struct omni_context * ctx_omni, 
                                    const float * llm_hidden_states, int n_tokens, int llm_n_embd,
                                    float * projected_hidden_states, int tts_n_embd) {
    // 优先使用新的 ggml 实现 (精度验证版本)
    if (ctx_omni->projector.initialized) {
        // 检查维度
        if (llm_n_embd != ctx_omni->projector.hparams.in_dim) {
            LOG_ERR("TTS: llm_n_embd (%d) != projector in_dim (%d)\n", 
                    llm_n_embd, ctx_omni->projector.hparams.in_dim);
            return false;
        }
        if (tts_n_embd != ctx_omni->projector.hparams.out_dim) {
            LOG_ERR("TTS: tts_n_embd (%d) != projector out_dim (%d)\n", 
                    tts_n_embd, ctx_omni->projector.hparams.out_dim);
            return false;
        }
        
        // 使用 ggml 后端计算
        std::vector<float> result = projector_forward(ctx_omni->projector, llm_hidden_states, n_tokens);
        if (result.empty()) {
            LOG_ERR("TTS: projector_forward failed\n");
            return false;
        }
        
        // 复制结果
        memcpy(projected_hidden_states, result.data(), n_tokens * tts_n_embd * sizeof(float));
        return true;
    }
    
    // Fallback: 使用旧的 float* 权重实现
    // Check if weights are loaded
    if (!ctx_omni->projector_semantic_linear1_weight || 
        !ctx_omni->projector_semantic_linear1_bias ||
        !ctx_omni->projector_semantic_linear2_weight ||
        !ctx_omni->projector_semantic_linear2_bias) {
        LOG_ERR("TTS: projector_semantic weights not loaded (both ggml and legacy)\n");
        return false;
    }
    
    // Check dimensions
    if (llm_n_embd != ctx_omni->projector_semantic_input_dim) {
        LOG_ERR("TTS: llm_n_embd (%d) != projector_semantic_input_dim (%d)\n", 
                llm_n_embd, ctx_omni->projector_semantic_input_dim);
        return false;
    }
    
    if (tts_n_embd != ctx_omni->projector_semantic_output_dim) {
        LOG_ERR("TTS: tts_n_embd (%d) != projector_semantic_output_dim (%d)\n", 
                tts_n_embd, ctx_omni->projector_semantic_output_dim);
        return false;
    }
    
    const int input_dim = ctx_omni->projector_semantic_input_dim;   // 4096
    const int output_dim = ctx_omni->projector_semantic_output_dim; // 768
    
    // Process each token (legacy CPU implementation)
    for (int t = 0; t < n_tokens; t++) {
        const float * hidden = llm_hidden_states + t * input_dim;
        float * output = projected_hidden_states + t * output_dim;
        
        // Temporary buffer for intermediate results
        std::vector<float> temp(output_dim);
        
        // Step 1: linear1: temp = hidden @ linear1_weight + linear1_bias
        for (int j = 0; j < output_dim; j++) {
            float sum = ctx_omni->projector_semantic_linear1_bias[j];
            for (int i = 0; i < input_dim; i++) {
                sum += hidden[i] * ctx_omni->projector_semantic_linear1_weight[i * output_dim + j];
            }
            temp[j] = sum;
        }
        
        // Step 2: ReLU activation
        for (int j = 0; j < output_dim; j++) {
            temp[j] = (temp[j] > 0.0f) ? temp[j] : 0.0f;
        }
        
        // Step 3: linear2: output = temp @ linear2_weight + linear2_bias
        for (int j = 0; j < output_dim; j++) {
            float sum = ctx_omni->projector_semantic_linear2_bias[j];
            for (int i = 0; i < output_dim; i++) {
                sum += temp[i] * ctx_omni->projector_semantic_linear2_weight[i * output_dim + j];
            }
            output[j] = sum;
        }
    }
    
    return true;
}

// 辅助函数：L2归一化（对每个token的embedding分别归一化）
// 匹配Python的 F.normalize(hidden_embeds, p=2, dim=-1)
// 注意：PyTorch的F.normalize使用sqrt(sum(x^2) + eps)，然后除以norm
void normalize_l2_per_token(float * embeddings, int n_tokens, int n_embd, float eps) {
    for (int t = 0; t < n_tokens; t++) {
        float * vec = embeddings + t * n_embd;
        
        // Calculate L2 norm (matching PyTorch: sqrt(sum(x^2) + eps))
        float norm_sq = 0.0f;
        for (int i = 0; i < n_embd; i++) {
            float val = vec[i];
            norm_sq += val * val;
        }
        // PyTorch F.normalize: norm = sqrt(sum(x^2) + eps), then x = x / norm
        float norm = std::sqrt(norm_sq + eps);
        
        // Normalize: divide by norm (matching PyTorch F.normalize with p=2, dim=-1)
        // CRITICAL: Always normalize, even if norm is very small (eps ensures norm > 0)
        if (norm > 0.0f) {
            float inv_norm = 1.0f / norm;
            for (int i = 0; i < n_embd; i++) {
                vec[i] *= inv_norm;
            }
        } else {
            // If norm is zero (shouldn't happen), set to unit vector
            LOG_WRN("TTS: WARNING - zero norm detected for token %d, setting to unit vector\n", t);
            float inv_sqrt_n = 1.0f / std::sqrt((float)n_embd);
            for (int i = 0; i < n_embd; i++) {
                vec[i] = inv_sqrt_n;
            }
        }
        
        // Verify normalization (always check to catch bugs)
        float verify_norm_sq = 0.0f;
        for (int i = 0; i < n_embd; i++) {
            float val = vec[i];
            verify_norm_sq += val * val;
        }
        float verify_norm = std::sqrt(verify_norm_sq);
        if (std::abs(verify_norm - 1.0f) > 0.01f) {
            LOG_ERR("TTS: ERROR - normalization verification failed for token %d: norm=%.6f (expected ~1.0), norm_sq=%.6f\n", 
                    t, verify_norm, verify_norm_sq);
        }
    }
}
bool eval_tokens_tts(struct omni_context* ctx_omni, common_params* params, std::vector<llama_token> tokens, int n_batch, int * n_past_tts) {
    fflush(stdout);
    int N = (int) tokens.size();
    fflush(stdout);
    // Note: TTS model might need different KV cache management
    // For now, we'll use a simple approach similar to LLM
    fflush(stdout);
    fflush(stdout);
    for (int i = 0; i < N; i += n_batch) {
        fflush(stdout);
        int n_eval = (int) tokens.size() - i;
        fflush(stdout);
        if (n_eval > n_batch) {
            n_eval = n_batch;
            fflush(stdout);
        }
        if (n_eval == 0) {
            fflush(stdout);
            break;
        }
        fflush(stdout);
        
        // Use llama_batch_get_one and manually set pos
        // Note: llama_batch_get_one may return batch with nullptr pos, so we need to handle it
        llama_batch batch = llama_batch_get_one(&tokens[i], n_eval);
        fflush(stdout);
        
        // If batch.pos is nullptr, we need to allocate it
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            fflush(stdout);
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        
        // Set pos values to ensure correct KV cache position
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = *n_past_tts + j;
        }
        fflush(stdout);
        
        // Enable embeddings output for TTS model (needed for head_code logits calculation)
        llama_set_embeddings(ctx_omni->ctx_tts_llama, true);
        fflush(stdout);
        int decode_ret = llama_decode(ctx_omni->ctx_tts_llama, batch);
        
        // Keep embeddings enabled for sample_tts_token to use
        
        if (decode_ret != 0) {
            LOG_ERR("%s : failed to eval TTS tokens. token %d/%d (batch size %d, n_past %d), decode_ret=%d\n", 
                    __func__, i, N, n_batch, *n_past_tts, decode_ret);
            return false;
        }
        *n_past_tts += n_eval;
    }
    return true;
}

bool eval_string_tts(struct omni_context * ctx_omni, common_params* params, const char* str, int n_batch, int * n_past_tts, bool add_bos) {
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_omni->ctx_tts_llama, str2, add_bos, true);
    return eval_tokens_tts(ctx_omni, params, embd_inp, n_batch, n_past_tts);
}

// 使用embedding作为TTS输入的prefill函数（类似prefill_with_emb，但针对TTS模型）
bool prefill_with_emb_tts(struct omni_context* ctx_omni, common_params* params, float* embed, int n_pos, int n_batch, int* n_past_tts) {
    // 验证输入参数
    if (n_pos <= 0) {
        LOG_ERR("%s: invalid n_pos=%d, skipping\n", __func__, n_pos);
        return false;
    }
    if (n_pos > 10000) {
        LOG_ERR("%s: n_pos=%d seems too large, likely data corruption\n", __func__, n_pos);
        return false;
    }
    if (!ctx_omni->ctx_tts_llama || !ctx_omni->model_tts) {
        LOG_ERR("%s: TTS model not loaded\n", __func__);
        return false;
    }
    
    int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
    
    // 验证 n_embd 是合理值
    if (n_embd <= 0 || n_embd > 10000) {
        LOG_ERR("%s: invalid n_embd=%d from TTS model, likely model corruption\n", __func__, n_embd);
        return false;
    }
    
    // 检查乘法溢出
    if (n_pos > (INT_MAX / n_embd)) {
        LOG_ERR("%s: n_pos=%d * n_embd=%d would overflow\n", __func__, n_pos, n_embd);
        return false;
    }
    
    // Save condition embeddings for first audio token re-forward (if not already saved)
    // This is needed to match Python's behavior: first audio token re-forwards the condition
    if (!ctx_omni->tts_condition_saved && n_pos > 0) {
        ctx_omni->tts_condition_embeddings.resize(n_pos * n_embd);
        std::memcpy(ctx_omni->tts_condition_embeddings.data(), embed, n_pos * n_embd * sizeof(float));
        ctx_omni->tts_condition_length = n_pos;
        ctx_omni->tts_condition_n_embd = n_embd;
        ctx_omni->tts_condition_saved = true;
    }
    
    // Save the starting position before the loop
    int text_start_pos = *n_past_tts;
    
    // Check if we need to save all hidden states (for alignment testing)
    const char* save_hidden_states_dir = getenv("TTS_SAVE_HIDDEN_STATES_DIR");
    bool save_all_hidden_states = (save_hidden_states_dir != nullptr);
    
    for (int i = 0; i < n_pos; i += n_batch) {
        int n_eval = n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        
        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd = (embed + i * n_embd);  // 使用embedding作为输入
        
        // 设置pos值以确保正确的KV cache位置
        // Python: pos_ids = torch.arange(text_start_pos, text_start_pos + condition_length)
        // C++: batch.pos[j] = text_start_pos + i + j (where i is the offset within the current batch)
        std::vector<llama_pos> pos_vec(n_eval);
        batch.pos = pos_vec.data();
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = text_start_pos + i + j;  // Fix: use text_start_pos + i + j instead of *n_past_tts + j
        }
        
        // Enable embeddings output for TTS model (needed for head_code logits calculation)
        llama_set_embeddings(ctx_omni->ctx_tts_llama, true);
        
        if (llama_decode(ctx_omni->ctx_tts_llama, batch)) {
            LOG_ERR("%s : failed to eval TTS embeddings. pos %d/%d (batch size %d, n_past %d)\n", 
                    __func__, i, n_pos, n_batch, *n_past_tts);
            llama_set_embeddings(ctx_omni->ctx_tts_llama, false);
            return false;
        }
        
        // Save hidden states for each token in the batch (for alignment testing)
        // Note: llama_get_embeddings_ith uses negative indices relative to the end of the batch
        // For a batch of n_eval tokens: -1 is last, -2 is second-to-last, ..., -n_eval is first
        if (save_all_hidden_states) {
            for (int j = 0; j < n_eval; j++) {
                int token_idx = text_start_pos + i + j;
                // Get j-th token in current batch: j=0 -> -n_eval (first), j=n_eval-1 -> -1 (last)
                int llama_idx = j - n_eval;
                const float* hidden_state = llama_get_embeddings_ith(ctx_omni->ctx_tts_llama, llama_idx);
                if (hidden_state) {
                    char filepath[512];
                    snprintf(filepath, sizeof(filepath), "%s/hidden_states_%03d.bin", save_hidden_states_dir, token_idx);
                    FILE* f = fopen(filepath, "wb");
                    if (f) {
                        fwrite(&token_idx, sizeof(int32_t), 1, f);
                        fwrite(&n_embd, sizeof(int32_t), 1, f);
                        fwrite(hidden_state, sizeof(float), n_embd, f);
                        fclose(f);
                    }
                } else {
                    LOG_WRN("TTS: Failed to get hidden state for token %d (llama_idx=%d)\n", token_idx, llama_idx);
                }
            }
        }
        
        // Keep embeddings enabled for sample_tts_token to use
    }
    
    // Update n_past_tts after all tokens are processed
    *n_past_tts = text_start_pos + n_pos;
    
    return true;
}

// Save logits to file for Python comparison
static void save_logits_to_file(const char* filepath, const float* logits, int num_tokens, int token_index) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s/logits_%03d.bin", filepath, token_index);
    
    FILE* f = fopen(full_path, "wb");
    if (!f) {
        LOG_ERR("Failed to open logits file for writing: %s\n", full_path);
        return;
    }
    
    // Write: token_index (int32), num_tokens (int32), logits (float32 array)
    fwrite(&token_index, sizeof(int32_t), 1, f);
    fwrite(&num_tokens, sizeof(int32_t), 1, f);
    fwrite(logits, sizeof(float), num_tokens, f);
    fclose(f);
}

// Save hidden states to file for Python comparison
static void save_hidden_states_to_file(const char* filepath, const float* hidden_states, int hidden_size, int token_index) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s/hidden_states_%03d.bin", filepath, token_index);
    
    FILE* f = fopen(full_path, "wb");
    if (!f) {
        LOG_ERR("Failed to open hidden states file for writing: %s\n", full_path);
        return;
    }
    
    // Write: token_index (int32), hidden_size (int32), hidden_states (float32 array)
    fwrite(&token_index, sizeof(int32_t), 1, f);
    fwrite(&hidden_size, sizeof(int32_t), 1, f);
    fwrite(hidden_states, sizeof(float), hidden_size, f);
    fclose(f);
}

// ========== RAS (Repetition Aware Sampling) Implementation ==========
// Ported from Python: tts_streaming_generate.py
// Key idea: Use nucleus sampling (top_p + top_k), but if too many recent tokens repeat,
// fall back to random sampling to break the pattern.

// Nucleus sampling (top-p + top-k)
// Returns a sampled token index from the logits
static int nucleus_sampling_tts(const float* logits, int num_tokens, float top_p, int top_k, std::mt19937& rng) {
    // 1. Compute softmax probabilities
    float max_logit = logits[0];
    for (int i = 1; i < num_tokens; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    std::vector<float> probs(num_tokens);
    float sum = 0.0f;
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] /= sum;
    }
    
    // 2. Sort by probability descending
    std::vector<std::pair<float, int>> sorted_probs;
    sorted_probs.reserve(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        sorted_probs.push_back({probs[i], i});
    }
    std::sort(sorted_probs.begin(), sorted_probs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 3. Collect tokens within top_p and top_k
    std::vector<float> filtered_probs;
    std::vector<int> filtered_indices;
    float cum_prob = 0.0f;
    
    for (const auto& p : sorted_probs) {
        if (cum_prob < top_p && (int)filtered_probs.size() < top_k) {
            cum_prob += p.first;
            filtered_probs.push_back(p.first);
            filtered_indices.push_back(p.second);
        } else {
            break;
        }
    }
    
    // 4. Renormalize filtered probs
    float filtered_sum = 0.0f;
    for (float p : filtered_probs) {
        filtered_sum += p;
    }
    for (float& p : filtered_probs) {
        p /= filtered_sum;
    }
    
    // 5. Multinomial sampling
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (size_t i = 0; i < filtered_probs.size(); ++i) {
        cum += filtered_probs[i];
        if (r <= cum) {
            return filtered_indices[i];
        }
    }
    return filtered_indices.back();  // Fallback
}

// Random sampling (uniform multinomial from all tokens)
static int random_sampling_tts(const float* logits, int num_tokens, std::mt19937& rng) {
    // 1. Compute softmax probabilities
    float max_logit = logits[0];
    for (int i = 1; i < num_tokens; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    std::vector<float> probs(num_tokens);
    float sum = 0.0f;
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] /= sum;
    }
    
    // 2. Multinomial sampling
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (int i = 0; i < num_tokens; ++i) {
        cum += probs[i];
        if (r <= cum) {
            return i;
        }
    }
    return num_tokens - 1;  // Fallback
}

// RAS: Repetition Aware Sampling
// If the sampled token repeats too often in recent history, use random sampling instead
// 采样参数与 Python TTSSamplingParams 对齐
static int ras_sampling_tts(
    const float* logits, 
    int num_tokens, 
    const std::vector<int>& decoded_tokens,  // Recent tokens (relative indices)
    std::mt19937& rng,
    float top_p = 0.85f,   // TTSSamplingParams.top_p = 0.85
    int top_k = 25,        // TTSSamplingParams.top_k = 25
    int win_size = 16,     // TTSSamplingParams.win_size = 16
    float tau_r = 0.1f     // TTSSamplingParams.tau_r = 0.1
) {
    // 1. First, do nucleus sampling to get a candidate
    int top_id = nucleus_sampling_tts(logits, num_tokens, top_p, top_k, rng);
    
    // 2. Count how many times this token appears in the recent window
    int start_idx = std::max(0, (int)decoded_tokens.size() - win_size);
    int rep_num = 0;
    for (int i = start_idx; i < (int)decoded_tokens.size(); ++i) {
        if (decoded_tokens[i] == top_id) {
            rep_num++;
        }
    }
    
    // 3. If repetition exceeds threshold, switch to random sampling
    // Python: if rep_num >= win_size * tau_r
    if (rep_num >= (int)(win_size * tau_r)) {
        return random_sampling_tts(logits, num_tokens, rng);
    }
    
    return top_id;
}

// Apply repetition penalty to logits - matching Python's CustomRepetitionPenaltyLogitsProcessorRepeat
// Python implementation:
//   freq = F.one_hot(input_ids, num_tokens).sum(1)  # count frequency
//   alpha = torch.pow(penalty, freq)
//   inp = scores.multiply(alpha)  # for negative scores
//   oth = scores.divide(alpha)    # for positive scores
//   out = torch.where(scores < 0, inp, oth)
static void apply_repetition_penalty_tts(
    float* logits, 
    int num_tokens,
    const std::vector<int>& decoded_tokens,  // Recent tokens (relative indices)
    float penalty,
    int past_window = 16
) {
    if (decoded_tokens.empty() || penalty == 1.0f) {
        return;
    }
    
    // Get the window of recent tokens
    int start_idx = std::max(0, (int)decoded_tokens.size() - past_window);
    
    // Count frequency of each token in the window
    std::vector<int> freq(num_tokens, 0);
    for (int i = start_idx; i < (int)decoded_tokens.size(); ++i) {
        int tok = decoded_tokens[i];
        if (tok >= 0 && tok < num_tokens) {
            freq[tok]++;
        }
    }
    
    // Apply penalty: alpha = penalty ^ freq
    // For positive logits: divide by alpha (makes them smaller, lower probability)
    // For negative logits: multiply by alpha (makes them more negative, lower probability)
    for (int i = 0; i < num_tokens; ++i) {
        if (freq[i] > 0) {
            float alpha = powf(penalty, (float)freq[i]);
            if (logits[i] < 0) {
                logits[i] *= alpha;  // More negative
            } else {
                logits[i] /= alpha;  // Smaller positive
            }
        }
    }
}

// Nucleus sampling with min_tokens_to_keep (matching Python's TopPLogitsWarper and TopKLogitsWarper)
static int nucleus_sampling_with_min_keep_tts(
    const float* logits, 
    int num_tokens, 
    float top_p, 
    int top_k, 
    int min_tokens_to_keep,  // Python default: 3
    std::mt19937& rng
) {
    // 1. Compute softmax probabilities
    float max_logit = logits[0];
    for (int i = 1; i < num_tokens; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    std::vector<float> probs(num_tokens);
    float sum = 0.0f;
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] /= sum;
    }
    
    // 2. Sort by probability descending
    std::vector<std::pair<float, int>> sorted_probs;
    sorted_probs.reserve(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        sorted_probs.push_back({probs[i], i});
    }
    std::sort(sorted_probs.begin(), sorted_probs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 3. Collect tokens within top_p and top_k, but keep at least min_tokens_to_keep
    std::vector<float> filtered_probs;
    std::vector<int> filtered_indices;
    float cum_prob = 0.0f;
    
    for (const auto& p : sorted_probs) {
        // Always keep min_tokens_to_keep tokens
        if ((int)filtered_probs.size() < min_tokens_to_keep) {
            cum_prob += p.first;
            filtered_probs.push_back(p.first);
            filtered_indices.push_back(p.second);
        }
        // After min_tokens_to_keep, apply top_p and top_k
        else if (cum_prob < top_p && (int)filtered_probs.size() < top_k) {
            cum_prob += p.first;
            filtered_probs.push_back(p.first);
            filtered_indices.push_back(p.second);
        } else {
            break;
        }
    }
    
    // 4. Renormalize filtered probs
    float filtered_sum = 0.0f;
    for (float p : filtered_probs) {
        filtered_sum += p;
    }
    for (float& p : filtered_probs) {
        p /= filtered_sum;
    }
    
    // 5. Multinomial sampling
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (size_t i = 0; i < filtered_probs.size(); ++i) {
        cum += filtered_probs[i];
        if (r <= cum) {
            return filtered_indices[i];
        }
    }
    return filtered_indices.back();  // Fallback
}

// ==================== 单工版本的 sample_tts_token ====================
// 遗留：单工（非 duplex）模式专用。duplex 路径用 generate_audio_tokens_local / sample_tts_token（见上/omni_internal.h）。
// 自旧版单工实现复制，保证单工模式行为完全一致
// 添加 is_final_text_chunk 参数（与 Python 对齐）：
//    - 非 final chunk：采样到 EOS 时不 prefill，避免污染 KV cache
//    - final chunk：采样到 EOS 时正常 prefill
llama_token sample_tts_token_simplex(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past_tts, const std::vector<llama_token> * all_generated_tokens, int token_index_in_chunk, bool force_no_eos, bool is_final_text_chunk) {
    const char* logits_debug_dir = getenv("TTS_LOGITS_DEBUG_DIR");
    
    const int audio_bos_token_id = 151687;
    const int num_audio_tokens = 6562;
    const int eos_relative_idx = num_audio_tokens - 1;  // EOS token relative index: 6561
    
    // 单工版本：is_audio_bos 只有在整个生成过程的第一个 token 时才为 true
    bool is_audio_bos = (all_generated_tokens == nullptr || all_generated_tokens->empty()) && (token_index_in_chunk == 0);
    if (is_audio_bos) {
        print_with_timestamp("TTS simplex: is_audio_bos=true (first audio token)\n");
    }
    
    // Re-forward condition for first audio token
    if (is_audio_bos && ctx_omni->tts_condition_saved && ctx_omni->tts_condition_length > 0) {
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
        }
        int condition_n_past = 0;
        if (!prefill_with_emb_tts(ctx_omni, params, 
                                 ctx_omni->tts_condition_embeddings.data(), 
                                 ctx_omni->tts_condition_length, 
                                 params->n_batch, 
                                 &condition_n_past)) {
            LOG_ERR("TTS simplex: Failed to re-forward condition\n");
            return 0;
        }
        *n_past_tts = condition_n_past;
    }

#if OMNI_REPLACEMENT_TTS_T2W
    if (ctx_omni->tts_t2w_replacement) {
        llama_token replacement_token = 0;
        if (omni_tts_replacement_sample_token(
                smpl, ctx_omni, params, n_past_tts,
                all_generated_tokens, nullptr, token_index_in_chunk,
                force_no_eos, is_final_text_chunk, &replacement_token)) {
            return replacement_token;
        }
        LOG_WRN("TTS replacement: GPU head/lookup failed in simplex; using legacy CPU path for this token\n");
    }
#endif

    // 使用 head_code 层计算 audio logits
    const float * hidden_state = llama_get_embeddings_ith(ctx_omni->ctx_tts_llama, -1);
    if (hidden_state == nullptr) {
        LOG_ERR("TTS simplex: failed to get hidden state\n");
        return 0;
    }
    
    if (ctx_omni->head_code_weight == nullptr) {
        LOG_ERR("TTS simplex: head_code weight not loaded\n");
        return 0;
    }
    
    if (ctx_omni->head_code_hidden_size != 768 || ctx_omni->head_code_num_audio_tokens != num_audio_tokens) {
        LOG_ERR("TTS simplex: head_code dimensions mismatch\n");
        return 0;
    }
    
    std::vector<float> audio_logits(num_audio_tokens, 0.0f);
    const float * head_code_w = ctx_omni->head_code_weight;
    const int hidden_size = ctx_omni->head_code_hidden_size;
    
    for (int i = 0; i < num_audio_tokens; ++i) {
        const float * row = head_code_w + i * hidden_size;
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden_state[j] * row[j];
        }
        audio_logits[i] = sum;
    }
    
    std::mt19937* rng = get_sampler_rng(smpl);
    std::mt19937 local_rng;
    if (rng == nullptr) {
        local_rng = std::mt19937(std::random_device{}());
        rng = &local_rng;
    }
    
    // 单工版本：使用 all_generated_tokens 做 repetition penalty
    std::vector<int> decoded_tokens_relative;
    if (all_generated_tokens != nullptr) {
        for (llama_token tid : *all_generated_tokens) {
            int relative_idx = tid - audio_bos_token_id;
            if (relative_idx >= 0 && relative_idx < num_audio_tokens) {
                decoded_tokens_relative.push_back(relative_idx);
            }
        }
    }
    
    // TTS 采样参数
    // Python tts_streaming_generate.py 使用：
    // - temperature (从 TTSSamplingParams 传入，默认 0.8)
    // - repetition_penalty = 1.05
    // - window = 8 (recent_ids = logits_token[:, -8:])
    // - multinomial 采样 (无 top-p/top-k)
    float temperature = 0.8f;
    float repetition_penalty = 1.05f;
    int win_size = 8;  // Python: recent_ids = logits_token[:, -8:]
    
    bool use_argmax = (params->sampling.temp <= 0.0f);
    
    int selected_relative_idx;
    if (use_argmax) {
        float max_logit = audio_logits[0];
        selected_relative_idx = 0;
        for (int i = 1; i < num_audio_tokens; ++i) {
            if (audio_logits[i] > max_logit) {
                max_logit = audio_logits[i];
                selected_relative_idx = i;
            }
        }
    } else {
        // Step 1: 应用 temperature
        // Python: logits /= self.temperature
        for (int i = 0; i < num_audio_tokens; ++i) {
            audio_logits[i] /= temperature;
        }
        
        // Step 2: 只有 t > 0 时才应用 repetition penalty
        // Python: if t > 0: ... recent_ids = logits_token[:, -8:]
        if (!is_audio_bos && !decoded_tokens_relative.empty()) {
            // 获取最近 win_size 个 tokens
            int start_idx = std::max(0, (int)decoded_tokens_relative.size() - win_size);
            
            // 使用 bool mask 而不是频率计数
            // Python: occurred = F.one_hot(recent_ids, ...).sum(dim=1).bool()
            std::vector<bool> occurred(num_audio_tokens, false);
            for (int i = start_idx; i < (int)decoded_tokens_relative.size(); ++i) {
                int tok = decoded_tokens_relative[i];
                if (tok >= 0 && tok < num_audio_tokens) {
                    occurred[tok] = true;
                }
            }
            
            // 应用 repetition penalty
            // Python: logits = torch.where(occurred & (logits >= 0), logits / penalty, logits)
            //         logits = torch.where(occurred & (logits < 0), logits * penalty, logits)
            for (int i = 0; i < num_audio_tokens; ++i) {
                if (occurred[i]) {
                    if (audio_logits[i] >= 0) {
                        audio_logits[i] /= repetition_penalty;
                    } else {
                        audio_logits[i] *= repetition_penalty;
                    }
                }
            }
        }
        
        // Step 3: 如果 force_no_eos=true，将 EOS logit 设为 -inf
        if (force_no_eos) {
            audio_logits[eos_relative_idx] = -std::numeric_limits<float>::infinity();
        }
        
        // Step 4: 使用 multinomial 采样（无 top-p/top-k）
        // Python: scores = F.softmax(logits, dim=-1)
        //         next_token = torch.multinomial(scores, num_samples=1)
        selected_relative_idx = random_sampling_tts(audio_logits.data(), num_audio_tokens, *rng);
    }
    
    if (selected_relative_idx < 0 || selected_relative_idx >= num_audio_tokens) {
        selected_relative_idx = 0;
    }
    
    const llama_token id = audio_bos_token_id + selected_relative_idx;
    int relative_idx = selected_relative_idx;
    
    common_sampler_accept(smpl, id, true);
    
    // 检测是否采样到 EOS
    bool is_eos = (relative_idx == eos_relative_idx);
    
    // EOS prefill 逻辑：
    // Python 中 EOS token 不会被加入 all_generated_tokens，所以下一轮不会 forward 进 TTS
    // C++ 中需要显式控制：
    //   - 非 final chunk：采样到 EOS 时不 prefill，直接返回（让调用方知道已结束）
    //   - final chunk：整个生成结束，prefill EOS（如果需要的话，保持状态一致性）
    if (is_eos && !is_final_text_chunk) {
        // 非 final chunk 采样到 EOS，不 prefill，直接返回
        // 这样 TTS 模型的 KV cache 不会包含 EOS，下一个 chunk 可以继续使用
        return id;
    }
    
    // 使用 emb_code 获取 embedding，然后通过 prefill_with_emb_tts 输入模型
    if (ctx_omni->emb_code_weight != nullptr && relative_idx >= 0 && relative_idx < ctx_omni->emb_code_vocab_size) {
        const float * emb_code_w = ctx_omni->emb_code_weight;
        const int emb_code_hidden_size = ctx_omni->emb_code_hidden_size;
        const int emb_code_vocab_size = ctx_omni->emb_code_vocab_size;
        
        std::vector<float> audio_token_embedding(emb_code_hidden_size);
        
        if (ctx_omni->emb_code_stored_as_transposed) {
            for (int j = 0; j < emb_code_hidden_size; ++j) {
                audio_token_embedding[j] = emb_code_w[j * emb_code_vocab_size + relative_idx];
            }
        } else {
            for (int j = 0; j < emb_code_hidden_size; ++j) {
                audio_token_embedding[j] = emb_code_w[relative_idx * emb_code_hidden_size + j];
            }
        }
        
        if (!prefill_with_emb_tts(ctx_omni, params, audio_token_embedding.data(), 1, 1, n_past_tts)) {
            LOG_ERR("TTS simplex: failed to decode audio token embedding\n");
            return 0;
        }
    } else {
        LOG_ERR("TTS simplex: emb_code not available\n");
        return 0;
    }
    
    return id;
}

llama_token sample_tts_token(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past_tts, const std::vector<llama_token> * all_generated_tokens, const std::vector<llama_token> * chunk_generated_tokens, int token_index_in_chunk, bool force_no_eos, bool is_final_text_chunk) {

    // Debug: Save logits directory (set via environment variable)
    const char* logits_debug_dir = getenv("TTS_LOGITS_DEBUG_DIR");
    
    // TTS model constants
    const int audio_bos_token_id = 151687;
    const int num_audio_tokens = 6562;  // head_code output size
    
    // 分离两个概念，与 Python generate_chunk 对齐：
    // 1. is_first_token_overall: 整个生成过程的第一个 token（用于 re-forward condition）
    //    Python TTSStreamingGenerator: audio_bos = len(self.all_generated_tokens) == 0 and t == 0
    // 2. is_chunk_first_token: 当前 chunk 的第一个 token（用于跳过 sampling processors）
    //    Python generate_chunk: if t == 0: audio_bos = True
    //    Python generate_chunk 中，每个 chunk 的第一个 token 不应用 repetition penalty 和 warpers
    
    // is_first_token_overall: 用于控制是否 re-forward condition（清空 KV cache）
    bool is_first_token_overall = (all_generated_tokens == nullptr || all_generated_tokens->empty()) && (token_index_in_chunk == 0);
    
    // skip_processors: 控制是否跳过 sampling processors
    // - 双工模式：每个 chunk 的第一个 token 都跳过（与 Python generate_chunk 对齐）
    //   Python generate_chunk: if t == 0: audio_bos = True
    // - 单工模式：只有整个生成过程的第一个 token 跳过（与 Python TTSStreamingGenerator 对齐）
    //   Python: audio_bos = len(self.all_generated_tokens) == 0 and t == 0
    bool skip_processors;
    if (ctx_omni->duplex_mode) {
        // 双工模式：每个 chunk 的第一个 token 都跳过
        skip_processors = (token_index_in_chunk == 0);
    } else {
        // 单工模式：只有整个生成过程的第一个 token 跳过
        skip_processors = is_first_token_overall;
    }
    
    // 只在第一个token时打印
    if (is_first_token_overall) {
        print_with_timestamp("TTS sample: is_first_token_overall=true, duplex_mode=%d\n", ctx_omni->duplex_mode);
    }
    
    // CRITICAL FIX: For the first audio token of the ENTIRE generation, we need to re-forward the entire condition
    // This matches Python's behavior where past_key_values=None on the first forward
    // Python: outputs = self.tts.model(position_ids=pos_ids, past_key_values=None, inputs_embeds=inputs_embeds, use_cache=True)
    // 只在整个生成过程的第一个 token 时 re-forward，不是每个 chunk 的第一个
    if (is_first_token_overall && ctx_omni->tts_condition_saved && ctx_omni->tts_condition_length > 0) {
        // 验证 tts_condition_* 值是否合理
        int cond_len = ctx_omni->tts_condition_length;
        int cond_n_embd = ctx_omni->tts_condition_n_embd;
        size_t cond_emb_size = ctx_omni->tts_condition_embeddings.size();
        size_t expected_size = (size_t)cond_len * cond_n_embd;
        
        if (cond_len <= 0 || cond_len > 10000) {
            LOG_ERR("TTS sample: invalid tts_condition_length=%d\n", cond_len);
            return 0;
        }
        if (cond_n_embd <= 0 || cond_n_embd > 10000) {
            LOG_ERR("TTS sample: invalid tts_condition_n_embd=%d\n", cond_n_embd);
            return 0;
        }
        if (cond_emb_size != expected_size) {
            LOG_ERR("TTS sample: tts_condition_embeddings size mismatch: %zu != %zu (len=%d * n_embd=%d)\n",
                    cond_emb_size, expected_size, cond_len, cond_n_embd);
            return 0;
        }
        // Clear KV cache to match Python's past_key_values=None behavior
        // Use llama_memory_seq_rm to remove all tokens from sequence 0
        // seq_id=0, p0=0 (from beginning), p1=-1 (to end) removes all tokens
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
        } else {
            LOG_ERR("TTS: Failed to get memory for KV cache clear\n");
        }
        
        // Reset n_past_tts to 0 (like Python's text_start_pos=0)
        int condition_n_past = 0;
        
        // Re-forward the entire condition
        if (!prefill_with_emb_tts(ctx_omni, params, 
                                 ctx_omni->tts_condition_embeddings.data(), 
                                 ctx_omni->tts_condition_length, 
                                 params->n_batch, 
                                 &condition_n_past)) {
            LOG_ERR("TTS: Failed to re-forward condition for first audio token\n");
            return 0;
        }
        
        // Update n_past_tts to match the condition length
        *n_past_tts = condition_n_past;
    }

#if OMNI_REPLACEMENT_TTS_T2W
    if (ctx_omni->tts_t2w_replacement) {
        llama_token replacement_token = 0;
        if (omni_tts_replacement_sample_token(
                smpl, ctx_omni, params, n_past_tts,
                all_generated_tokens, chunk_generated_tokens,
                token_index_in_chunk, force_no_eos, is_final_text_chunk,
                &replacement_token)) {
            return replacement_token;
        }
        LOG_WRN("TTS replacement: GPU head/lookup failed; using legacy CPU path for this token\n");
    }
#endif

    // 1. 获取TTS模型的最后一个位置的hidden state
    // hidden_state shape: (hidden_size=768,)

    const float * hidden_state = llama_get_embeddings_ith(ctx_omni->ctx_tts_llama, -1);

    if (hidden_state == nullptr) {
        LOG_ERR("TTS: failed to get hidden state from TTS model\n");
        return 0;
    }
    
    // Debug: Save hidden states to file for Python comparison
    if (logits_debug_dir != nullptr) {
        int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
        save_hidden_states_to_file(logits_debug_dir, hidden_state, hidden_size, token_index_in_chunk);
    }
    
    // 2. 检查head_code权重是否已加载
    if (ctx_omni->head_code_weight == nullptr) {
        LOG_ERR("TTS: head_code weight not loaded\n");
        return 0;
    }
    
    if (ctx_omni->head_code_hidden_size != 768 || ctx_omni->head_code_num_audio_tokens != num_audio_tokens) {
        LOG_ERR("TTS: head_code dimensions mismatch: expected (768, 6562), got (%d, %d)\n",
                ctx_omni->head_code_hidden_size, ctx_omni->head_code_num_audio_tokens);
        return 0;
    }
    
    // 3. 使用head_code层计算audio token的logits
    // logits = hidden_state @ head_code_weight
    // hidden_state: (768,), head_code_weight: [6562, 768] (转置后存储) -> logits: (6562,)
    std::vector<float> audio_logits(num_audio_tokens, 0.0f);
    const float * head_code_w = ctx_omni->head_code_weight;
    const int hidden_size = ctx_omni->head_code_hidden_size;
    
    // 优化：head_code_weight 已转置为 [6562, 768]，每行连续存储
    // 连续内存访问，cache友好
    for (int i = 0; i < num_audio_tokens; ++i) {
        const float * row = head_code_w + i * hidden_size;
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden_state[j] * row[j];
        }
        audio_logits[i] = sum;
    }
    
    int eos_relative_idx = num_audio_tokens - 1;  // EOS token relative index: 6561
    
    // Debug: Save raw logits to file for Python comparison
    if (logits_debug_dir != nullptr) {
        save_logits_to_file(logits_debug_dir, audio_logits.data(), num_audio_tokens, token_index_in_chunk);
    }
    
    // Save hidden state and logits for comparison (before evaluation)
    const char* output_dir = getenv("TTS_OUTPUT_DIR");
    if (output_dir != nullptr) {
        // Save hidden state (for first token only)
        if (token_index_in_chunk == 0) {
            char hidden_state_path[512];
            snprintf(hidden_state_path, sizeof(hidden_state_path), "%s/cpp_first_hidden_state.bin", output_dir);
            FILE* f_hidden = fopen(hidden_state_path, "wb");
            if (f_hidden) {
                fwrite(hidden_state, sizeof(float), hidden_size, f_hidden);
                fclose(f_hidden);
            }
        }
        
        // Save logits for all tokens (for debugging)
        char logits_path[512];
        snprintf(logits_path, sizeof(logits_path), "%s/cpp_logits_token_%d.bin", output_dir, token_index_in_chunk);
        FILE* f_logits = fopen(logits_path, "wb");
        if (f_logits) {
            fwrite(audio_logits.data(), sizeof(float), num_audio_tokens, f_logits);
            fclose(f_logits);
        }
    }
    
    // 4. 采样流程 - 与 Python MiniCPMTTS.generate() 完全对齐
    // Python 采样流程:
    //   1. logits /= temperature (默认 0.8)
    //   2. if not audio_bos: apply repetition_penalty (penalty=1.05, past_window=16)
    //   3. if not audio_bos: apply TopP (0.85) + TopK (25) warper (min_tokens_to_keep=3)
    //   4. softmax + multinomial
    
    // Get RNG from sampler
    std::mt19937* rng = get_sampler_rng(smpl);
    std::mt19937 local_rng;
    if (rng == nullptr) {
        // Fallback to local RNG
        LOG_WRN("TTS: sampler RNG not available, using local RNG\n");
        local_rng = std::mt19937(std::random_device{}());
        rng = &local_rng;
    }
    
    // Collect decoded tokens (relative indices) for repetition penalty
    // - 双工模式：使用 chunk_generated_tokens（当前 chunk 内的 tokens）
    //   Python generate_chunk: input_ids_sliced = new_tokens[:, 0:t]
    // - 单工模式：使用 all_generated_tokens（所有生成的 tokens）
    //   Python TTSStreamingGenerator: self.all_generated_tokens
    std::vector<int> decoded_tokens_relative;
    const std::vector<llama_token> * tokens_for_penalty;
    if (ctx_omni->duplex_mode) {
        // 双工模式：优先使用当前 chunk 的 tokens
        tokens_for_penalty = chunk_generated_tokens ? chunk_generated_tokens : all_generated_tokens;
    } else {
        // 单工模式：使用所有生成的 tokens
        tokens_for_penalty = all_generated_tokens;
    }
    if (tokens_for_penalty != nullptr) {
        for (llama_token tid : *tokens_for_penalty) {
            // Convert absolute token ID to relative index
            int relative_idx = tid - audio_bos_token_id;
            if (relative_idx >= 0 && relative_idx < num_audio_tokens) {
                decoded_tokens_relative.push_back(relative_idx);
            }
        }
    }
    
    // 采样参数与 Python TTSSamplingParams 对齐
    float temperature = 0.8f;       // TTSSamplingParams.temperature = 0.8
    float top_p = 0.85f;            // TTSSamplingParams.top_p = 0.85
    int top_k = 25;                 // TTSSamplingParams.top_k = 25
    float repetition_penalty = 1.05f; // TTSSamplingParams.repetition_penalty = 1.05
    int win_size = 16;              // TTSSamplingParams.win_size = 16
    float tau_r = 0.1f;             // TTSSamplingParams.tau_r = 0.1
    int min_tokens_to_keep = 3;     // Python: TopPLogitsWarper/TopKLogitsWarper default
    
    // Get temperature from params if specified (for argmax mode)
    bool use_argmax = (params->sampling.temp <= 0.0f);
    
    int selected_relative_idx;
    if (use_argmax) {
        // Deterministic sampling: select argmax (highest logit)
        float max_logit = audio_logits[0];
        selected_relative_idx = 0;
        for (int i = 1; i < num_audio_tokens; ++i) {
            if (audio_logits[i] > max_logit) {
                max_logit = audio_logits[i];
                selected_relative_idx = i;
            }
        }
    } else {
        // Step 1: Apply temperature
        for (int i = 0; i < num_audio_tokens; ++i) {
            audio_logits[i] /= temperature;
        }
        
        // Step 2 & 3: Apply repetition penalty and TopP/TopK
        // - 双工模式：每个 chunk 的第一个 token 跳过
        // - 单工模式：只有整个生成过程的第一个 token 跳过
        if (!skip_processors && !decoded_tokens_relative.empty()) {
            // Apply repetition penalty (matching Python's CustomRepetitionPenaltyLogitsProcessorRepeat)
            apply_repetition_penalty_tts(audio_logits.data(), num_audio_tokens, 
                                        decoded_tokens_relative, repetition_penalty, win_size);
        }
        
        // 在采样前阻止 EOS token 被采样
        // Python generate_chunk: if force_no_stop or t < min_new_tokens: logits[:, eos_token] = -torch.inf
        // 这样可以确保在达到 min_new_tokens 之前不会生成 EOS
        if (ctx_omni->duplex_mode && force_no_eos) {
            int eos_relative_idx = num_audio_tokens - 1;  // EOS token relative index: 6561
            audio_logits[eos_relative_idx] = -std::numeric_limits<float>::infinity();
        }
        
        // Step 4: Nucleus sampling with min_tokens_to_keep (matching Python's warpers)
        selected_relative_idx = nucleus_sampling_with_min_keep_tts(
            audio_logits.data(), 
            num_audio_tokens, 
            top_p,
            top_k,
            min_tokens_to_keep,
            *rng
        );
    }
    
    // 5. 验证采样结果在有效范围内
    if (selected_relative_idx < 0 || selected_relative_idx >= num_audio_tokens) {
        LOG_ERR("TTS: invalid selected index %d, should be in [0, %d)\n", 
                selected_relative_idx, num_audio_tokens);
        // Fallback: use first token
        selected_relative_idx = 0;
    }
    
    // Convert relative index to absolute token ID
    const llama_token id = audio_bos_token_id + selected_relative_idx;
    int relative_idx = selected_relative_idx;
    
    common_sampler_accept(smpl, id, true);
    
    // 检测是否采样到 EOS
    int eos_relative_idx_check = num_audio_tokens - 1;  // EOS token relative index: 6561
    bool is_eos = (relative_idx == eos_relative_idx_check);
    
    // EOS prefill 逻辑：
    // Python 中 EOS token 不会被加入 all_generated_tokens，所以下一轮不会 forward 进 TTS
    // C++ 中需要显式控制：
    //   - 非 final chunk：采样到 EOS 时不 prefill，直接返回（让调用方知道已结束）
    //   - final chunk：整个生成结束，prefill EOS（如果需要的话，保持状态一致性）
    if (ctx_omni->duplex_mode && is_eos && !is_final_text_chunk) {
        // 非 final chunk 采样到 EOS，不 prefill，直接返回
        // 这样 TTS 模型的 KV cache 不会包含 EOS，下一个 chunk 可以继续使用
    
        return id;
    }
    
    // 9. 使用emb_code将audio token ID转换为embedding，然后使用prefill_with_emb_tts
    // 这样可以避免token ID超出词汇表范围的问题
    if (ctx_omni->emb_code_weight != nullptr && relative_idx >= 0 && relative_idx < ctx_omni->emb_code_vocab_size) {
        // Get embedding from emb_code
        const float * emb_code_w = ctx_omni->emb_code_weight;
        const int emb_code_hidden_size = ctx_omni->emb_code_hidden_size;
        const int emb_code_vocab_size = ctx_omni->emb_code_vocab_size;
        
        // Allocate embedding buffer
        std::vector<float> audio_token_embedding(emb_code_hidden_size);
        
        if (ctx_omni->emb_code_stored_as_transposed) {
            // Stored as (hidden_size, num_audio_tokens) = [768, 6562]
            // Access: embedding[j] = emb_code_weight[j * num_audio_tokens + relative_idx]
            for (int j = 0; j < emb_code_hidden_size; ++j) {
                audio_token_embedding[j] = emb_code_w[j * emb_code_vocab_size + relative_idx];
            }
        } else {
            // Stored as (num_audio_tokens, hidden_size) = [6562, 768]
            // Access: embedding[j] = emb_code_weight[relative_idx * hidden_size + j]
            for (int j = 0; j < emb_code_hidden_size; ++j) {
                audio_token_embedding[j] = emb_code_w[relative_idx * emb_code_hidden_size + j];
            }
        }
        
    
        // Use prefill_with_emb_tts to decode with embedding instead of token ID
        if (!prefill_with_emb_tts(ctx_omni, params, audio_token_embedding.data(), 1, 1, n_past_tts)) {
            LOG_ERR("TTS: failed to decode audio token embedding\n");
            return 0;
        }
    
    } else {
        // Fallback: use token IDs (may fail if token ID exceeds vocab size)
        LOG_ERR("TTS: emb_code not available, falling back to token IDs (may fail if token exceeds vocab)\n");
        std::vector<llama_token> tokens;
        tokens.push_back(id);
    
        if (!eval_tokens_tts(ctx_omni, params, tokens, 1, n_past_tts)) {
            LOG_ERR("TTS: failed to decode audio token ID (token may exceed vocab size)\n");
            return 0;
        }
    
    }
    
    return id;
}

// Check if a token is an audio token (based on config: num_audio_tokens = 6562)
// Audio tokens are typically in a specific range

extern const std::vector<llama_token> g_special_token_ids = {
    151667,  // <think>
    151668,  // </think>
    151704,  // <|tts_eos|> - TTS 结束标记
    151706,  // <|speak|> - Python 的 chunk_speak_token_ids 会过滤此 token
    151705,  // <|listen|> - 双工模式切换到听的标记
    151718,  // <|chunk_eos|> - chunk 结束标记
    151721,  // <|chunk_tts_eos|> - TTS chunk 结束标记
    151717,  // <|turn_eos|> - 轮次结束标记
    271,     // <reserved_182> (newline, appears near think tags)
};

// Known empty token IDs (from file_loader.py)
// WARNING: These tokens were incorrectly marked as "empty" but they actually decode to real text!
// 99692 = "好的", 104314 = "给你", 99526 = "讲"
// DO NOT add tokens here without verifying they actually decode to empty string!
extern const std::set<llama_token> g_known_empty_token_ids = {
    // Intentionally empty - no tokens confirmed to decode to truly empty strings
};

bool is_valid_tts_token(llama_token tid) {
    // Skip special tokens
    for (llama_token sid : g_special_token_ids) {
        if (tid == sid) {
            return false;
        }
    }
    
    // Skip tokens >= 150000 (usually special tokens like <|im_end|>, <|tts_bos|>, etc.)
    if (tid >= 150000) {
        return false;
    }
    
    // Skip known empty tokens
    if (g_known_empty_token_ids.find(tid) != g_known_empty_token_ids.end()) {
        return false;
    }
    
    return true;
}

// 遗留：单工（非 duplex）模式专用。duplex 路径用 generate_audio_tokens_local / sample_tts_token（见上/omni_internal.h）。
bool generate_audio_tokens_local_simplex(
    struct omni_context * ctx_omni,
    common_params * params,
    const std::vector<float>& merged_embeddings,
    int n_tokens,
    int tts_n_embd,
    int chunk_idx,
    std::vector<int32_t>& output_audio_tokens,
    const std::string& output_dir = "",
    bool is_final_text_chunk = false  // 是否是最后一个 text chunk
) {
    print_with_timestamp("TTS Simplex: generating audio tokens for chunk %d (n_tokens=%d, tts_n_embd=%d)\n",
                        chunk_idx, n_tokens, tts_n_embd);
    
    const int audio_bos_token_id = 151687;
    const int num_audio_tokens = 6562;
    // Python: max_new_token=500，每个 LLM condition 生成直到 EOS
    // 然后每 25 个 tokens yield 一次
    const int max_audio_tokens = 500;
    
    if (!ctx_omni->ctx_tts_llama || !ctx_omni->model_tts) {
        LOG_ERR("TTS Simplex: TTS model not loaded\n");
        return false;
    }
    
    if (!ctx_omni->head_code_weight || !ctx_omni->emb_code_weight) {
        LOG_ERR("TTS Simplex: TTS weights not loaded\n");
        return false;
    }
    
    if (merged_embeddings.size() != (size_t)(n_tokens * tts_n_embd)) {
        LOG_ERR("TTS Simplex: merged_embeddings size mismatch: %zu != %d * %d\n",
                merged_embeddings.size(), n_tokens, tts_n_embd);
        return false;
    }
    
    // 在 prefill 之前动态添加 audio_bos embedding
    // Python 中 audio_bos 是在 TTS 类内部（TTSStreamingGenerator.generate_with_buffer）添加的
    // 每个 chunk 都会加 audio_bos: condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
    std::vector<float> condition_with_bos = merged_embeddings;  // 复制一份
    int extra_tokens = 0;  // 额外添加的 tokens 数量
    
    // text_eos_embed 不再在 condition 构建阶段添加
    // 原因：一个 text chunk 会生成多个 audio chunk，text_eos_embed 放在 condition 中
    // 会导致所有 audio chunk 都受到影响。正确做法是在第一轮 audio 生成结束（EOS）后，
    // 再注入 text_eos_embed + audio_bos 做第二轮生成，这样只影响最后一批 audio tokens。
    const int text_eos_token_id = 151692;  // TTS 的 text_eos_token_id
    
    // 只添加 audio_bos（总是添加）
    // Python: condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
    std::vector<float> audio_bos_embed(tts_n_embd, 0.0f);
    if (tts_emb_text(ctx_omni, audio_bos_token_id, audio_bos_embed.data(), tts_n_embd)) {
        condition_with_bos.insert(condition_with_bos.end(), 
                                  audio_bos_embed.begin(), audio_bos_embed.end());
        extra_tokens++;
        print_with_timestamp("TTS Simplex: 在 prefill 前添加 audio_bos (chunk_idx=%d, new_size=%zu)\n", 
                            chunk_idx, condition_with_bos.size() / tts_n_embd);
    } else {
        LOG_ERR("TTS Simplex: failed to get audio_bos embedding\n");
    }
    int n_tokens_with_bos = n_tokens + extra_tokens;  // 包含 audio_bos
    
    // Save condition embeddings (包含 audio_bos)
    ctx_omni->tts_condition_embeddings = condition_with_bos;
    ctx_omni->tts_condition_length = n_tokens_with_bos;
    ctx_omni->tts_condition_n_embd = tts_n_embd;
    ctx_omni->tts_condition_saved = true;
    
    int n_past_tts = 0;
    if (chunk_idx == 0) {
        // 第一个 chunk：清空 KV cache
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
            print_with_timestamp("TTS Simplex: first chunk - cleared KV cache\n");
        }
        ctx_omni->tts_n_past_accumulated = 0;
        ctx_omni->tts_all_generated_tokens.clear();
        ctx_omni->tts_token_buffer.clear();  // 新轮对话清空 buffer
        print_with_timestamp("TTS Simplex: first chunk - cleared tts_all_generated_tokens and tts_token_buffer\n");
    } else {
        n_past_tts = ctx_omni->tts_n_past_accumulated;
        print_with_timestamp("TTS Simplex: chunk %d - keeping KV cache, n_past_tts=%d\n", chunk_idx, n_past_tts);
    }
    
    // 使用包含 audio_bos 的 condition 进行 prefill
    if (!prefill_with_emb_tts(ctx_omni, params, 
                              condition_with_bos.data(),
                              n_tokens_with_bos, params->n_batch, &n_past_tts)) {
        LOG_ERR("TTS Simplex: prefill_with_emb_tts failed\n");
        return false;
    }
    print_with_timestamp("TTS Simplex: prefill completed, n_past_tts=%d\n", n_past_tts);
    
    // Create sampler - matching Python TTSStreamingGenerator
    // Create sampler - matching Python TTSStreamingGenerator
    // Python TTSSamplingParams 默认 temperature=0.8
    common_params_sampling tts_sampling = params->sampling;
    tts_sampling.temp = ctx_omni->tts_temperature;  // [与 Python 对齐] TTSSamplingParams.temperature
    tts_sampling.top_k = 25;
    tts_sampling.penalty_repeat = 1.05f;
    tts_sampling.min_p = 0.01f;
    tts_sampling.penalty_last_n = 16;
    
    struct common_sampler * tts_sampler = common_sampler_init(ctx_omni->model_tts, tts_sampling);
    if (!tts_sampler) {
        LOG_ERR("TTS Simplex: failed to create sampler\n");
        return false;
    }
    print_with_timestamp("TTS Simplex: sampler created\n");
    
    output_audio_tokens.clear();
    
    // 统一使用 tts_token_buffer 管理，和 Python _token_buffer 一致
    // Python chunk_size=25，每凑够 25 个才 yield
    const int CHUNK_SIZE = 25;  // 与 Python TTSStreamingGenerator.chunk_size=25 对齐
    
    // 不强制最小生成数量，让 TTS 自然决定何时结束
    const int min_new_tokens = 0;
    
    // 两阶段生成：
    // Phase 1: 不带 text_eos_embed，正常生成 audio tokens 直到 EOS
    // Phase 2 (仅 is_final_text_chunk): 注入 text_eos_embed + audio_bos，生成最后一批 audio tokens
    // 这样 text_eos_embed 只影响最后一批 audio tokens，而非整个 text chunk 的所有 audio
    bool need_phase2 = false;  // 是否需要第二阶段生成
    
    // ===== Phase 1: 正常生成（不带 text_eos_embed） =====
    for (int t = 0; t < max_audio_tokens; ++t) {
        // 检测 break_event，立即停止 TTS 生成
        if (ctx_omni->break_event.load()) {
            print_with_timestamp("TTS Simplex: break_event detected at step %d, stopping immediately\n", t);
            break;
        }
        
        // 如果还没达到 min_new_tokens，阻止 EOS 被采样
        bool force_no_eos = (t < min_new_tokens);
        
        // Phase 1 始终使用 is_final_text_chunk=false：EOS 不 prefill，保持 KV cache 干净
        llama_token sampled_token_abs = sample_tts_token_simplex(
            tts_sampler,
            ctx_omni,
            params,
            &n_past_tts,
            &ctx_omni->tts_all_generated_tokens,
            t,
            force_no_eos,
            false  // Phase 1: 不 prefill EOS，留给 Phase 2 处理
        );
        
        if (sampled_token_abs == 0) {
            LOG_ERR("TTS Simplex: sample_tts_token failed at step %d\n", t);
            break;
        }
        
        int relative_idx = sampled_token_abs - audio_bos_token_id;
        if (relative_idx < 0 || relative_idx >= num_audio_tokens) {
            LOG_ERR("TTS Simplex: invalid token ID %d at step %d\n", sampled_token_abs, t);
            break;
        }
        
        output_audio_tokens.push_back(relative_idx);
        ctx_omni->tts_all_generated_tokens.push_back(sampled_token_abs);
        
        // EOS token 检测
        bool is_eos = (relative_idx == num_audio_tokens - 1);
        if (is_eos) {
            print_with_timestamp("TTS Simplex Phase1: EOS token at step %d\n", t + 1);
            output_audio_tokens.pop_back();
            ctx_omni->tts_all_generated_tokens.pop_back();
            // 如果是最后一个 text chunk，需要进入 Phase 2
            if (is_final_text_chunk) {
                need_phase2 = true;
                print_with_timestamp("TTS Simplex: is_final_text_chunk=true, will enter Phase 2 for text_eos_embed\n");
            }
        } else {
            // 非 EOS token 加入 tts_token_buffer
            ctx_omni->tts_token_buffer.push_back(relative_idx);
        }
        
        // 当 buffer >= chunk_size 时，yield 出 chunk_size 个
        while ((int)ctx_omni->tts_token_buffer.size() >= CHUNK_SIZE && ctx_omni->t2w_thread_info) {
            T2WOut *t2w_out = new T2WOut();
            t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(), 
                                         ctx_omni->tts_token_buffer.begin() + CHUNK_SIZE);
            t2w_out->is_final = false;
            t2w_out->round_idx = ctx_omni->simplex_round_idx;
            
            {
                std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                ctx_omni->t2w_thread_info->queue.push(t2w_out);
            }
            ctx_omni->t2w_thread_info->cv.notify_one();
            
            print_with_timestamp("TTS Simplex Phase1: yield %d tokens 到 T2W (step %d, buffer=%zu)\n", 
                                CHUNK_SIZE, t + 1, ctx_omni->tts_token_buffer.size());
            ctx_omni->tts_token_buffer.erase(ctx_omni->tts_token_buffer.begin(), 
                                              ctx_omni->tts_token_buffer.begin() + CHUNK_SIZE);
        }
        
        if (t < 5 || (t + 1) % 25 == 0) {
            print_with_timestamp("TTS Simplex Phase1: token %d/%d: rel_id=%d\n", t + 1, max_audio_tokens, relative_idx);
        }
        
        // 如果是 EOS，退出 Phase 1 循环
        if (is_eos) {
            break;
        }
    }
    
    // ===== Phase 2: 注入 text_eos_embed，生成最后一批 audio tokens =====
    // 仅在 is_final_text_chunk 且 Phase 1 正常结束（EOS）时执行
    if (need_phase2 && !ctx_omni->break_event.load()) {
        print_with_timestamp("TTS Simplex Phase2: injecting text_eos_embed + audio_bos at n_past=%d\n", n_past_tts);
        
        // 注入 text_eos_embed 到 TTS KV cache
        std::vector<float> text_eos_embed(tts_n_embd, 0.0f);
        bool inject_ok = false;
        if (tts_emb_text(ctx_omni, text_eos_token_id, text_eos_embed.data(), tts_n_embd)) {
            if (prefill_with_emb_tts(ctx_omni, params, text_eos_embed.data(), 1, 1, &n_past_tts)) {
                print_with_timestamp("TTS Simplex Phase2: text_eos_embed injected, n_past=%d\n", n_past_tts);
                
                // 再注入 audio_bos，开始新一轮 audio 生成
                std::vector<float> audio_bos_embed2(tts_n_embd, 0.0f);
                if (tts_emb_text(ctx_omni, audio_bos_token_id, audio_bos_embed2.data(), tts_n_embd)) {
                    if (prefill_with_emb_tts(ctx_omni, params, audio_bos_embed2.data(), 1, 1, &n_past_tts)) {
                        print_with_timestamp("TTS Simplex Phase2: audio_bos injected, n_past=%d, starting final generation\n", n_past_tts);
                        inject_ok = true;
                    }
                }
            }
        }
        
        if (inject_ok) {
            // Phase 2 生成循环：text_eos_embed 已注入，生成最后的 audio tokens
            for (int t2 = 0; t2 < max_audio_tokens; ++t2) {
                if (ctx_omni->break_event.load()) {
                    print_with_timestamp("TTS Simplex Phase2: break_event at step %d\n", t2);
                    break;
                }
                
                // Phase 2 使用 is_final_text_chunk=true：EOS 会被 prefill
                llama_token sampled_token_abs = sample_tts_token_simplex(
                    tts_sampler,
                    ctx_omni,
                    params,
                    &n_past_tts,
                    &ctx_omni->tts_all_generated_tokens,
                    t2,
                    false,  // 不强制阻止 EOS
                    true    // Phase 2: is_final，EOS 会被 prefill
                );
                
                if (sampled_token_abs == 0) {
                    LOG_ERR("TTS Simplex Phase2: sample failed at step %d\n", t2);
                    break;
                }
                
                int relative_idx = sampled_token_abs - audio_bos_token_id;
                if (relative_idx < 0 || relative_idx >= num_audio_tokens) {
                    LOG_ERR("TTS Simplex Phase2: invalid token %d at step %d\n", sampled_token_abs, t2);
                    break;
                }
                
                output_audio_tokens.push_back(relative_idx);
                ctx_omni->tts_all_generated_tokens.push_back(sampled_token_abs);
                
                bool is_eos = (relative_idx == num_audio_tokens - 1);
                if (is_eos) {
                    print_with_timestamp("TTS Simplex Phase2: EOS at step %d (final end)\n", t2 + 1);
                    output_audio_tokens.pop_back();
                    ctx_omni->tts_all_generated_tokens.pop_back();
                } else {
                    ctx_omni->tts_token_buffer.push_back(relative_idx);
                }
                
                // yield audio chunks
                while ((int)ctx_omni->tts_token_buffer.size() >= CHUNK_SIZE && ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(), 
                                                 ctx_omni->tts_token_buffer.begin() + CHUNK_SIZE);
                    t2w_out->is_final = false;
                    t2w_out->round_idx = ctx_omni->simplex_round_idx;
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                    print_with_timestamp("TTS Simplex Phase2: yield %d tokens 到 T2W\n", CHUNK_SIZE);
                    ctx_omni->tts_token_buffer.erase(ctx_omni->tts_token_buffer.begin(), 
                                                      ctx_omni->tts_token_buffer.begin() + CHUNK_SIZE);
                }
                
                if (is_eos) {
                    break;
                }
            }
        } else {
            LOG_ERR("TTS Simplex Phase2: failed to inject text_eos_embed/audio_bos\n");
        }
    }
    
    // chunk 结束时，tts_token_buffer 中可能有剩余的 tokens (< CHUNK_SIZE)
    // 这些 tokens 保留在 buffer 中，等下一个 text chunk 继续累积
    // 只有 is_final_text_chunk 时才 flush 所有剩余
    print_with_timestamp("TTS Simplex: chunk 结束，tts_token_buffer 剩余 %zu tokens (保留等下一个 chunk)\n", 
                        ctx_omni->tts_token_buffer.size());
    
    // 只有最后一个 text chunk 时才 flush 剩余的 buffer
    // Python: if finished.all() and text_finished: yield all remaining; buffer = []
    if (is_final_text_chunk && !ctx_omni->tts_token_buffer.empty() && ctx_omni->t2w_thread_info) {
        T2WOut *t2w_out = new T2WOut();
        t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(), 
                                     ctx_omni->tts_token_buffer.end());
        t2w_out->is_final = false;  // 注意：这里不设 is_final=true，is_final 由 tts_thread_func 在 llm_finish 时发送
        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 传递轮次索引
        
        {
            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
            ctx_omni->t2w_thread_info->queue.push(t2w_out);
        }
        ctx_omni->t2w_thread_info->cv.notify_one();
        
        print_with_timestamp("TTS Simplex: is_final_text_chunk=true, flush 剩余 %zu tokens 从 tts_token_buffer\n", 
                            ctx_omni->tts_token_buffer.size());
        ctx_omni->tts_token_buffer.clear();
    }
    
    // 只有 duplex_mode 时才发送 is_chunk_end 信号
    // 单工模式下，T2W 依赖滑动窗口机制，中间 chunk 不需要 flush
    if (ctx_omni->duplex_mode && ctx_omni->t2w_thread_info) {
        T2WOut *t2w_out = new T2WOut();
        t2w_out->audio_tokens.clear();  // 空的 token 列表
        t2w_out->is_final = false;  // Not final (turn not ended)
        t2w_out->is_chunk_end = true;  // 标记 chunk 结束，T2W 需要 flush buffer
        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 传递轮次索引
        
        {
            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
            ctx_omni->t2w_thread_info->queue.push(t2w_out);
        }
        ctx_omni->t2w_thread_info->cv.notify_one();
        print_with_timestamp("TTS Simplex: 发送 is_chunk_end=true 信号\n");
    }
    
    common_sampler_free(tts_sampler);
    
    print_with_timestamp("TTS Simplex: generated %zu audio tokens for chunk %d\n",
                        output_audio_tokens.size(), chunk_idx);
    
    // 输出前几个和后几个 audio tokens，帮助定位重复问题
    if (output_audio_tokens.size() > 0) {
        std::string first_tokens = "";
        std::string last_tokens = "";
        int show_count = std::min((int)output_audio_tokens.size(), 5);
        for (int i = 0; i < show_count; i++) {
            first_tokens += std::to_string(output_audio_tokens[i]) + " ";
        }
        for (int i = std::max(0, (int)output_audio_tokens.size() - 5); i < (int)output_audio_tokens.size(); i++) {
            last_tokens += std::to_string(output_audio_tokens[i]) + " ";
        }
        print_with_timestamp("TTS Simplex: chunk %d first tokens: [%s], last tokens: [%s]\n",
                            chunk_idx, first_tokens.c_str(), last_tokens.c_str());
    }
    
    ctx_omni->tts_n_past_accumulated = n_past_tts;
    print_with_timestamp("TTS Simplex: updated tts_n_past_accumulated=%d, total_generated_tokens=%zu\n", 
                        ctx_omni->tts_n_past_accumulated, ctx_omni->tts_all_generated_tokens.size());
    
    // Save tokens to file if output_dir is specified
    if (!output_dir.empty() && !output_audio_tokens.empty()) {
        std::string tokens_file = output_dir + "/audio_tokens_chunk_" + std::to_string(chunk_idx) + ".bin";
        FILE* f = fopen(tokens_file.c_str(), "wb");
        if (f) {
            fwrite(output_audio_tokens.data(), sizeof(int32_t), output_audio_tokens.size(), f);
            fclose(f);
        }
    }
    
    return !output_audio_tokens.empty();
}

// Uses the verified TTS model to generate audio tokens from merged_embeddings
// ==============================================================================
bool generate_audio_tokens_local(
    struct omni_context * ctx_omni,
    common_params * params,
    const std::vector<float>& merged_embeddings,
    int n_tokens,
    int tts_n_embd,
    int chunk_idx,
    std::vector<int32_t>& output_audio_tokens,
    bool is_end_of_turn,  // 是否是轮次结束（默认值见 omni.h 声明）
    const std::string& output_dir,
    std::chrono::steady_clock::time_point queue_enter,
    uint64_t frame_id,
    void (*segment_callback)(int, int, void *),
    void * segment_callback_user_data
) {
    print_with_timestamp("TTS Local: generating audio tokens for chunk %d (n_tokens=%d, tts_n_embd=%d, emb_size=%zu)\n",
                         chunk_idx, n_tokens, tts_n_embd, merged_embeddings.size());

    // 验证输入参数
    // 当 is_end_of_turn=true 时，允许 n_tokens=0
    // 因为我们会添加 text_eos_embed 和 audio_bos，让 TTS 生成最后的 audio tokens
    if (n_tokens < 0) {
        LOG_ERR("TTS Local: invalid n_tokens=%d\n", n_tokens);
        return false;
    }
    if (n_tokens == 0 && !is_end_of_turn) {
        LOG_ERR("TTS Local: n_tokens=0 but is_end_of_turn=false, nothing to generate\n");
        return false;
    }
    if (n_tokens > 10000) {
        LOG_ERR("TTS Local: n_tokens=%d seems too large, likely data corruption\n", n_tokens);
        return false;
    }
    if (tts_n_embd <= 0 || tts_n_embd > 10000) {
        LOG_ERR("TTS Local: invalid tts_n_embd=%d\n", tts_n_embd);
        return false;
    }
    
    // 记录特殊情况
    if (n_tokens == 0 && is_end_of_turn) {
        print_with_timestamp("TTS Local: n_tokens=0 but is_end_of_turn=true, will add text_eos_embed and generate final tokens\n");
    }
    
    // TTS model constants
    const int audio_bos_token_id = 151687;
    const int num_audio_tokens = 6562;
    // max_audio_tokens:
    // - 双工模式: 26 (与 Python 对齐，max_token_per_chunk = 25 + 1)
    // - 单工模式: 500 (允许更长的生成，靠 EOS 结束)
    const int max_audio_tokens = ctx_omni->duplex_mode ? 26 : 500;
    print_with_timestamp("TTS Local: mode=%s, max_audio_tokens=%d\n", 
                         ctx_omni->duplex_mode ? "duplex" : "simplex", max_audio_tokens);
    
    // Verify TTS model is loaded
    if (!ctx_omni->ctx_tts_llama || !ctx_omni->model_tts) {
        LOG_ERR("TTS Local: TTS model not loaded\n");
        return false;
    }
    
    // Verify TTS weights are loaded
    if (!ctx_omni->head_code_weight || !ctx_omni->emb_code_weight) {
        LOG_ERR("TTS Local: TTS weights not loaded (head_code or emb_code)\n");
        return false;
    }
    
    // Verify merged_embeddings size
    if (merged_embeddings.size() != (size_t)(n_tokens * tts_n_embd)) {
        LOG_ERR("TTS Local: merged_embeddings size mismatch: %zu != %d * %d\n",
                merged_embeddings.size(), n_tokens, tts_n_embd);
        return false;
    }

    // 在 prefill 之前动态添加 text_eos_embed（如果是轮次结束）和 audio_bos embedding
    // Python TTSStreamingGenerator.generate_with_buffer 逻辑：
    //   if text_finished: condition = torch.cat([condition, self.text_eos_embed], dim=1)
    //   condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
    std::vector<float> condition_with_bos = merged_embeddings;  // 复制一份
    int extra_tokens = 0;
    
    // 如果是轮次结束，先添加 text_eos_embed
    // Python: if text_finished: condition = torch.cat([condition, self.text_eos_embed], dim=1)
    const int text_eos_token_id = 151692;  // TTS 的 text_eos_token_id
    if (is_end_of_turn) {
        std::vector<float> text_eos_embed(tts_n_embd, 0.0f);
        if (tts_emb_text(ctx_omni, text_eos_token_id, text_eos_embed.data(), tts_n_embd)) {
            condition_with_bos.insert(condition_with_bos.end(), 
                                      text_eos_embed.begin(), text_eos_embed.end());
            extra_tokens++;
            print_with_timestamp("TTS Local: is_end_of_turn=true, 添加 text_eos_embed (chunk_idx=%d, new_size=%zu)\n", 
                                chunk_idx, condition_with_bos.size() / tts_n_embd);
        } else {
            LOG_WRN("TTS Local: failed to get text_eos embedding\n");
        }
    }
    
    // 然后添加 audio_bos（总是添加）
    // Python: condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
    std::vector<float> audio_bos_embed(tts_n_embd, 0.0f);
    if (tts_emb_text(ctx_omni, audio_bos_token_id, audio_bos_embed.data(), tts_n_embd)) {
        condition_with_bos.insert(condition_with_bos.end(), 
                                  audio_bos_embed.begin(), audio_bos_embed.end());
        extra_tokens++;
        print_with_timestamp("TTS Local: 在 prefill 前添加 audio_bos (chunk_idx=%d, new_size=%zu)\n", 
                            chunk_idx, condition_with_bos.size() / tts_n_embd);
    } else {
        LOG_ERR("TTS Local: failed to get audio_bos embedding\n");
    }
    int n_tokens_with_bos = n_tokens + extra_tokens;  // 包含 text_eos_embed（如果有）和 audio_bos
    
    // Save condition embeddings for re-forward in sample_tts_token (包含 audio_bos)
    ctx_omni->tts_condition_embeddings = condition_with_bos;
    ctx_omni->tts_condition_length = n_tokens_with_bos;
    ctx_omni->tts_condition_n_embd = tts_n_embd;
    ctx_omni->tts_condition_saved = true;
    // Python TTSStreamingGenerator 逻辑：
    // - idx == 0：清空 KV cache，从头开始，并拼接 spk_emb
    // - idx > 0：保持 past_key_values，继续生成
    // 这样 TTS 模型可以保持上下文连续性，避免提前生成 EOS
    int n_past_tts = 0;
    if (chunk_idx == 0) {
        // 第一个 chunk：清空 KV cache 和累积状态
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
            print_with_timestamp("TTS Local: first chunk - cleared KV cache\n");
        }
        ctx_omni->tts_n_past_accumulated = 0;  // 重置累计 n_past
        ctx_omni->tts_all_generated_tokens.clear();
        if (!ctx_omni->duplex_mode) {
            ctx_omni->tts_token_buffer.clear();  // 新轮对话清空 buffer（仅单工模式）
        }
        print_with_timestamp("TTS Local: first chunk - cleared tts_all_generated_tokens and tts_token_buffer\n");
    } else {
        // 后续 chunk：保持 KV cache，继续使用累计的 n_past
        n_past_tts = ctx_omni->tts_n_past_accumulated;
        print_with_timestamp("TTS Local: chunk %d - keeping KV cache, n_past_tts=%d\n", chunk_idx, n_past_tts);
    }
    // 使用包含 audio_bos 的 condition 进行 prefill
    if (!prefill_with_emb_tts(ctx_omni, params, 
                              condition_with_bos.data(),
                              n_tokens_with_bos, params->n_batch, &n_past_tts)) {
        LOG_ERR("TTS Local: prefill_with_emb_tts failed\n");
        return false;
    }
    print_with_timestamp("TTS Local: prefill completed, n_past_tts=%d\n", n_past_tts);

    // 2. Create sampler for TTS with correct TTS sampling parameters
    // TTS 流式采样参数 - 与 Python TTSStreamingGenerator 对齐
    // Python TTSSamplingParams 默认 temperature=0.8
    common_params_sampling tts_sampling = params->sampling;
    tts_sampling.temp = ctx_omni->tts_temperature;  // [与 Python 对齐] TTSSamplingParams.temperature
    tts_sampling.top_p = 0.85f;  // TTSSamplingParams.top_p=0.85
    tts_sampling.top_k = 25;               // top_k = 25
    tts_sampling.penalty_repeat = 1.05f;   // repetition_penalty = 1.05
    tts_sampling.min_p = 0.01f;            // min_p = 0.01
    tts_sampling.penalty_last_n = 16;      // past_window = 16 (与Python对齐)
    
    struct common_sampler * tts_sampler = common_sampler_init(ctx_omni->model_tts, tts_sampling);
    if (!tts_sampler) {
        LOG_ERR("TTS Local: failed to create sampler\n");
        return false;
    }
    
    // 3. Generate audio tokens with streaming to T2W queue
    output_audio_tokens.clear();
    // Python: self.all_generated_tokens 是类成员变量，跨 chunk 持续累积
    // 用于：1. 正确判断是否是整个生成过程的第一个 token（re-forward condition）
    
    // 当前 chunk 的 tokens，用于 repetition penalty
    // Python generate_chunk: input_ids_sliced = new_tokens[:, 0:t]  # 只用当前 chunk 内的 tokens
    std::vector<llama_token> chunk_generated_tokens;
    
    // min_new_tokens 逻辑
    // - 双工模式: 与 Python streaming_generate 对齐
    //   max_token_per_chunk = 25 + 1  # 26
    //   min_token_per_chunk = 25 + 1  # 26
    //   if end_of_turn: min_token_per_chunk = 0
    // - 单工模式: 设置最小 100 个 tokens，防止 TTS 过早生成 EOS
    //   10 个中文字约需要 100-150 个 audio tokens (每字 10-15 tokens)
    const int min_new_tokens = ctx_omni->duplex_mode ? 
        (is_end_of_turn ? 0 : 26) : 
        100;  // 单工模式：至少生成 100 个 tokens 防止过早 EOS
    
    // 流水线优化：Token2Wav需要28个tokens (25+3 lookahead)才能输出音频
    // 首批28个tokens后推送，后续每25个tokens推送一次
    const int STREAM_CHUNK_SIZE = 25;
    const int FIRST_CHUNK_SIZE = 28;  // 首批推送阈值，减少首响时间
    bool first_chunk_pushed = false;
    std::vector<int32_t> stream_buffer;

    // Token boundaries are reported after the TTS backend has synchronized.
    // The worker owns the lease; this callback only reports a safe boundary.
    int tts_group = 1;
    const char * tts_seg_env = getenv("OMNI_SM_TTS_SEG_GROUP");
    if (tts_seg_env != nullptr) tts_group = atoi(tts_seg_env);
    if (tts_group <= 0) tts_group = 0;
    const int tts_segment_count_hint = tts_group > 0
        ? (max_audio_tokens + tts_group - 1) / tts_group : 0;
    int tts_segment_id = 0, tts_segment_tokens = 0;

    for (int t = 0; t < max_audio_tokens; ++t) {
        // 计算 force_no_eos
        // Python generate_chunk: if force_no_stop or t < min_new_tokens: logits[:, eos_token] = -torch.inf
        // 如果还没达到 min_new_tokens，阻止 EOS 被采样（在采样前设置 EOS logit 为 -inf）
        bool force_no_eos = (t < min_new_tokens);
        
        // Sample next token
        // 传入：
        // - all_generated_tokens: 用于判断是否是整个生成过程的第一个 token
        // - chunk_generated_tokens: 用于 repetition penalty（只用当前 chunk 的 tokens）
        // - t: token_index_in_chunk，用于判断是否跳过 sampling processors
        // - force_no_eos: 是否阻止 EOS（用于 min_new_tokens 逻辑）
        // - is_end_of_turn: 是否是 final chunk（EOS 是否应加入 KV cache）
        llama_token sampled_token_abs = sample_tts_token(
            tts_sampler,
            ctx_omni,
            params,
            &n_past_tts,
            &ctx_omni->tts_all_generated_tokens,  // 用于判断 is_first_token_overall
            &chunk_generated_tokens,               // 用于 repetition penalty
            t,                                     // token_index_in_chunk
            force_no_eos,                          // 阻止 EOS
            is_end_of_turn                         // EOS 是否加入 KV cache
        );
        
        if (sampled_token_abs == 0) {
            LOG_ERR("TTS Local: sample_tts_token failed at step %d\n", t);
            break;
        }

        // Convert to relative index and check EOS
        int relative_idx = sampled_token_abs - audio_bos_token_id;
        if (relative_idx < 0 || relative_idx >= num_audio_tokens) {
            LOG_ERR("TTS Local: invalid token ID %d (relative_idx: %d) at step %d\n",
                    sampled_token_abs, relative_idx, t);
            break;
        }
        
        // Store token (as absolute ID for internal use, will convert to relative for output)
        output_audio_tokens.push_back(relative_idx);  // Store relative ID for token2wav
        stream_buffer.push_back(relative_idx);  // Also add to stream buffer
        ctx_omni->tts_all_generated_tokens.push_back(sampled_token_abs);
        // 添加到当前 chunk 的列表
        chunk_generated_tokens.push_back(sampled_token_abs);
        
        // EOS token 检测 - 必须在流式推送之前
        // Python: if next_id.eq(self.eos_token).any(): finished[:] = True; else: 添加到 buffer
        // 如果是 EOS，立即从 buffer 中移除，防止被推送到 T2W
        bool is_eos = (relative_idx == num_audio_tokens - 1);
        if (is_eos) {
            output_audio_tokens.pop_back();
            stream_buffer.pop_back();
            chunk_generated_tokens.pop_back();
            ctx_omni->tts_all_generated_tokens.pop_back();
            // 不 break，继续执行后续逻辑（包括可能的流式推送剩余 tokens）
        }
        
        // 流式推送，但保留 lookahead
        // Python _generate_waveform_from_tokens 逻辑：
        //   while len(buffer) >= CHUNK_SIZE + pre_lookahead:  # 28
        //       stream(buffer[:28])
        //       buffer = buffer[CHUNK_SIZE:]  # 只移除 25 个，保留 3 个 lookahead
        // 
        // 关键：我们推送所有可用的 tokens，但 T2W 端负责保留 lookahead
        // 这里我们每 25 个 tokens 推送一次，让 T2W 端的滑动窗口正确处理
        // is_end_of_turn 时不进行流式推送，让所有 tokens 在最后一起发送
        // 这样 T2W 才能一次性 flush 所有 buffer，输出完整音频
        int push_threshold = first_chunk_pushed ? STREAM_CHUNK_SIZE : FIRST_CHUNK_SIZE;
        if ((int)stream_buffer.size() >= push_threshold && ctx_omni->t2w_thread_info &&
                !is_end_of_turn && !ctx_omni->break_event.load()) {
            first_chunk_pushed = true;
            T2WOut *t2w_out = new T2WOut();
            t2w_out->audio_tokens.assign(stream_buffer.begin(), stream_buffer.end());
            t2w_out->is_final = false;
            t2w_out->is_chunk_end = false;  // 中间推送，不是 chunk 结束
            t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 传递轮次索引
            t2w_out->frame_id = frame_id;
            t2w_out->chunk_idx = chunk_idx;
            {
                std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                ctx_omni->t2w_thread_info->queue.push(t2w_out);
            }
            ctx_omni->t2w_thread_info->cv.notify_one();
            stream_buffer.clear();
        }
        
        if (t < 5 || (t + 1) % 25 == 0) {
        }
        
        // 如果是 EOS，在完成上述处理后退出循环
        if (is_eos) {
            break;
        }

        // EOS 已在上方 break，不会在末 token 让出。
        ++tts_segment_tokens;
        if (segment_callback != nullptr && tts_group > 0 && tts_segment_tokens >= tts_group &&
                tts_segment_id + 1 < tts_segment_count_hint) {
            llama_synchronize(ctx_omni->ctx_tts_llama);
            segment_callback(tts_segment_id, tts_segment_count_hint,
                             segment_callback_user_data);
            ++tts_segment_id;
            tts_segment_tokens = 0;
        }
    }

    // 问题：如果只推送 25 tokens，T2W buffer 从 3->28，处理后剩余 3 个 tokens
    //       这 3 个 tokens 是当前 chunk 的最后 3 个，要等到下一个 chunk 才被处理
    //       导致当前 chunk 的尾音被延迟，听起来"吞字"
    // 解决：在 chunk 结束时发送 is_chunk_end=true 信号，让 T2W flush 掉 buffer 中的剩余 tokens
    if (ctx_omni->t2w_thread_info &&
            !ctx_omni->break_event.load()) {
        T2WOut *t2w_out = new T2WOut();
        t2w_out->audio_tokens.assign(stream_buffer.begin(), stream_buffer.end());
        // is_end_of_turn 时传 is_final=true，让 T2W 立即 flush 所有 buffer
        // Python: token2wav.stream(..., last_chunk=is_last_chunk) 其中 is_last_chunk = end_of_turn
        t2w_out->is_final = is_end_of_turn;
        t2w_out->is_chunk_end = !is_end_of_turn;  // 非 turn 结束时才用 is_chunk_end
        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 传递轮次索引
        t2w_out->frame_id = frame_id;
        t2w_out->chunk_idx = chunk_idx;
        {
            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
            ctx_omni->t2w_thread_info->queue.push(t2w_out);
        }
        ctx_omni->t2w_thread_info->cv.notify_one();
    }
    
    // Cleanup sampler
    common_sampler_free(tts_sampler);
    
    // 更新累计 n_past，用于下一个 chunk 的 KV cache 位置
    // Python: self.text_start_pos += condition_length + len(chunk_generated_tokens)
    ctx_omni->tts_n_past_accumulated = n_past_tts;
    
    // Save tokens to file if output_dir is specified
    if (!output_dir.empty() && !output_audio_tokens.empty()) {
        std::string tokens_file = output_dir + "/audio_tokens_chunk_" + std::to_string(chunk_idx) + ".bin";
        FILE* f = fopen(tokens_file.c_str(), "wb");
        if (f) {
            fwrite(output_audio_tokens.data(), sizeof(int32_t), output_audio_tokens.size(), f);
            fclose(f);
        }
    }
    
    return !output_audio_tokens.empty();
}
