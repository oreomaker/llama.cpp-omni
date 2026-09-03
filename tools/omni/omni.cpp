#include "omni-impl.h"
#include "vision.h"
#include "audition.h"
#include "omni.h"
#include "omni_internal.h"
#include "duplex_core.h"
#include "omni_prewarm.h"
#include "token2wav/token2wav-impl.h"

// Implementation map:
//   omni.cpp         Lifecycle: init/free、模型加载、media adapters、prompt 配置、
//                    单工 LLM 线程（遗留路径）、滑窗状态。
//   omni_io.cpp      Shared logging, directory creation, and WAV input helpers.
//   omni_t2w.cpp     T2W worker（Python/C++）与 WAV 输出。
//   compute_llm.cpp  LLM 图执行（eval/prefill/sample/滑窗）。
//   compute_tts.cpp  TTS 生成（generate_audio_tokens_local*）。
//   compute_enc.cpp  VPM/APM 编码。
//   duplex_core.cpp  duplex 调度：管道线程、stream_* 路由、高层会话。
//   tts_worker.cpp   TTS 调度：tts_thread_func_duplex / tts_thread_func。
//   omni.h           Public API plus the current internal omni_context state.
//   omni_internal.h  Internal interface shared by the TUs.

#include "llama.h"
#include "common/common.h"
#include "common/sampling.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif


#include <iostream>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include <set>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <unordered_map>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <sstream>
#include <random>
#include <array>
#include <cstdarg>
#include <signal.h>

#ifdef _WIN32
    #include <windows.h>
    #include <direct.h>
    #include <io.h>
    #include <process.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    // Windows compatibility macros
    #define popen  _popen
    #define pclose _pclose
    #define unlink _unlink
    #define stat   _stat
    #define S_IFDIR _S_IFDIR
#else
    #include <sys/time.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <sys/wait.h>
    #include <unistd.h>
    #include <dirent.h>
#endif

// ==================== 特殊 Token 分类 ====================
// 
// 双工模式下，模型生成的特殊 token 分为以下几类：
// 
// 1. 状态切换 Token:
//    - <|speak|>: 模型决定开始说话
//    - <|listen|>: 模型决定开始听（双工）
// 
// 2. Chunk 终止 Token:
//    - <|chunk_eos|>: 语义 chunk 结束，仍在同一轮内
//    - <|chunk_tts_eos|>: TTS chunk 结束，触发 TTS 处理
// 
// 3. 轮次/序列终止 Token:
//    - <|turn_eos|>: 当前轮次结束，可切换回 listen
//    - <|tts_eos|>: 旧版 TTS 结束（单工模式）
//    - </s>: 序列完全结束
//
// Python 参考 (MiniCPMODuplex):
//   CHUNK_EOS_TOKEN_ID = 128261
//   CHUNK_TTS_EOS_TOKEN_ID = 128268
//   TURN_EOS_TOKEN_ID = 128260
//   LISTEN_TOKEN_ID = 128267
//   SPEAK_TOKEN_ID = 128266



//
// omni structure
//
struct unit_buffer {
    std::vector<float> buffer;
    std::string text;
    bool completed = false;
    int unit_n_past = 0;
    float duration;
};

struct omni_output {
    std::vector<unit_buffer *> output;
    int idx;
};

std::condition_variable g_decode_cv;
bool prefill_done = true;
std::mutex speek_mtx;
std::condition_variable speek_cv;
bool last_speek_done_flag = false;

// 让 thread 可以结束
std::atomic<bool> llm_thread_running(true);
std::atomic<bool> tts_thread_running(true);
std::atomic<bool> t2w_thread_running(true);

// 读取 omni_output 互斥
std::mutex buffer_mutex;


static struct llama_model * llama_init(common_params * params, std::string model_path) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = common_model_params_to_llama(*params);
    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n" , __func__);
        return NULL;
    }
    return model;
}

// TTS专用模型加载 - 支持独立的GPU层数设置
// 通过环境变量 TTS_GPU_LAYERS 控制，-1 表示使用与LLM相同的设置
static struct llama_model * llama_init_tts(common_params * params, std::string model_path, int n_gpu_layers_override = -1) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = common_model_params_to_llama(*params);
    model_params.partial_load = true;  // TTS GGUF contains extra tensors (emb_code, head_code, projector_*) beyond standard llama

    // 如果指定了override值(>=0)，使用它；否则保持与LLM相同的设置
    if (n_gpu_layers_override >= 0) {
        model_params.n_gpu_layers = n_gpu_layers_override;
    }

    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load TTS model\n" , __func__);
        return NULL;
    }
    return model;
}

struct omni_context * omni_init(struct common_params * params, int media_type, bool use_tts, std::string tts_bin_dir,
                                int tts_gpu_layers, const std::string & token2wav_device, bool duplex_mode,
                                llama_model * existing_model, llama_context * existing_ctx,
                                const std::string & base_output_dir) {
    // process the prompt
    print_with_timestamp("=== omni_init start\n");
    // if (params->prompt.empty() && params->interactive == false) {
    //     LOG_INF("prompt should be given or interactive mode should be on");
    //     return NULL;
    // }
    // auto ctx_omni = (struct omni_context *)malloc(sizeof(omni_context));
    auto ctx_omni = new omni_context();

    ctx_omni->params = params;
    ctx_omni->media_type = media_type;
    ctx_omni->use_tts = use_tts;
    ctx_omni->duplex_mode = duplex_mode;
    ctx_omni->base_output_dir = base_output_dir;  // 可配置的输出目录
    print_with_timestamp("media_type = %d, duplex_mode = %d, base_output_dir = %s\n", media_type, duplex_mode, base_output_dir.c_str());

    // prompt 格式（对齐 Python default_tts_chat_template）：
    //   template: '<|im_start|>' + role + '\n' + content + '<|im_end|>\n'
    //   generation: '<|im_start|>assistant\n' + think_str + '<|tts_bos|>'
    //
    // audio_assistant 模式的 system prompt：
    //   vc_prompt_prefix + <|audio_start|>[ref_audio_embed]<|audio_end|> + vc_prompt_suffix
    // 完整格式见下文各分支拼接的 voice_clone_prompt / assistant_prompt：
    //   voice_clone_prompt 是 system prompt 的 prefix（含 <|audio_start|>），
    //   assistant_prompt 是 suffix（含 <|audio_end|>...<|im_end|>）；
    //   stream_decode 会再添加实际的 assistant generation prompt。
    if (duplex_mode) {
        // 双工模式：嵌入参考音频；不需要 <|im_start|>user\n，用 <unit> 标记用户输入。
        // Audio 与 Omni 格式相同。
        ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|><|im_end|>\n";

        ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|><|im_end|>\n";
    } else {
        // 非双工 Audio 格式（audio_assistant 模式）：在 sys prompt 末尾直接加
        // <|im_start|>user\n，不在 stream_prefill 里动态添加（不依赖 Python 端 counter 重置）。
        ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。<|im_end|>\n<|im_start|>user\n";
        
        // Omni 模式（非双工）：与 Audio 模式类似，末尾也添加 <|im_start|>user\n
        ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。<|im_end|>\n<|im_start|>user\n";
    }

    llama_model * model = nullptr;
    llama_context * ctx_llama = nullptr;
    
    // 支持模型复用（单工模式常用）
    if (existing_model != nullptr && existing_ctx != nullptr) {
        print_with_timestamp("=== omni_init: reusing existing LLM model and context\n");
        model = existing_model;
        ctx_llama = existing_ctx;
        ctx_omni->owns_model = false;  // 不拥有模型，omni_free 时不释放
        
        // 模式切换时清理 LLM 的 KV cache，避免位置冲突。
        //
        // NOTE: 过去用 llama_memory_seq_rm(mem, 0, 0, -1)。新版 llama.cpp 的
        // llama_kv_cache::seq_rm 加了 GGML_ASSERT(seq_id == -1 || seq_id < seq_to_stream.size())，
        // 在 ISWA/hybrid 组合里部分内部 kv 的 seq_to_stream 未覆盖 seq_id=0 时会 abort，
        // 导致 llama-server 崩溃。这里语义只是整体重置 LLM KV cache，改用
        // llama_memory_clear 更安全——它按 n_stream 遍历重置 cells/heads，不依赖 seq_to_stream。
        llama_memory_t mem = llama_get_memory(ctx_llama);
        if (mem) {
            llama_memory_clear(mem, /*data=*/false);
            print_with_timestamp("=== omni_init: cleared LLM KV cache for mode switch\n");
        }
    } else {
        // 加载新模型
        print_with_timestamp("=== omni_init: loading new LLM model\n");
        model = llama_init(params, params->model.path);
        if (model == NULL) {
            return NULL;
        }
        llama_context_params ctx_params = common_context_params_to_llama(*params);
        ctx_params.n_ctx                = params->n_ctx;

        ctx_llama = llama_new_context_with_model(model, ctx_params);
        if (ctx_llama == NULL) {
            LOG_ERR("%s: error: failed to create the llama_context\n" , __func__);
            return NULL;
        }
        ctx_omni->owns_model = true;  // 拥有模型，omni_free 时需要释放
    }
    
    struct common_sampler * sampler = common_sampler_init(model, params->sampling);
    ctx_omni->ctx_llama = ctx_llama;
    ctx_omni->model = model;
    ctx_omni->ctx_sampler = sampler;

    if (use_tts && !params->tts_model.empty()) {
        print_with_timestamp("=== omni_init: loading TTS model\n");
        // 使用TTS专用的模型加载函数，支持独立的GPU层数设置
        // tts_gpu_layers 从 omni_init 参数传入，-1 表示使用与LLM相同的设置
        print_with_timestamp("TTS model: loading with n_gpu_layers=%d\n", tts_gpu_layers);
        llama_model * tts_model = llama_init_tts(params, params->tts_model, tts_gpu_layers);
        if (tts_model == NULL) {
            LOG_ERR("%s: error: failed to init TTS model from %s\n", __func__, params->tts_model.c_str());
            llama_free(ctx_llama);
            llama_free_model(model);
            common_sampler_free(sampler);
            delete ctx_omni;
            return NULL;
        }
        
        // TTS 模型使用独立的上下文参数
        // 注意：TTS 模型可能需要不同的上下文大小和批处理大小
        llama_context_params tts_ctx_params = common_context_params_to_llama(*params);
        // 如果 TTS 模型需要更小的上下文窗口，可以在这里调整
        // 例如：tts_ctx_params.n_ctx = std::min(params->n_ctx, 2048); // 限制 TTS 上下文大小
        tts_ctx_params.n_ctx = params->n_ctx;  // 暂时使用相同的 n_ctx，后续可以根据需要调整
        
        llama_context * ctx_tts_llama = llama_new_context_with_model(tts_model, tts_ctx_params);
        if (ctx_tts_llama == NULL) {
            LOG_ERR("%s: error: failed to create the TTS llama_context\n", __func__);
            llama_free_model(tts_model);
            // 清理已分配的资源
            llama_free(ctx_llama);
            llama_free_model(model);
            common_sampler_free(sampler);
            delete ctx_omni;
            return NULL;
        }
        
        // 创建 TTS 的采样器（参数对齐 Python ras_sampling / TTSSamplingParams）
        common_params_sampling tts_sampling = params->sampling;
        tts_sampling.temp = ctx_omni->tts_temperature;  // TTSSamplingParams.temperature（默认 0.8）
        tts_sampling.top_p = 0.85f;            // TTSSamplingParams.top_p
        tts_sampling.top_k = 25;               // ras_sampling top_k
        tts_sampling.penalty_repeat = 1.05f;   // repetition_penalty
        tts_sampling.min_p = 0.01f;            // min_p
        // Python: CustomRepetitionPenaltyLogitsProcessorRepeat(..., 16)
        tts_sampling.penalty_last_n = 16;      // past_window（与 Python 对齐）
        struct common_sampler * tts_sampler = common_sampler_init(tts_model, tts_sampling);
        print_with_timestamp("TTS sampler: temp=%.2f, top_p=%.2f, top_k=%d, rep_penalty=%.2f\n",
                            tts_sampling.temp, tts_sampling.top_p, tts_sampling.top_k, tts_sampling.penalty_repeat);
        
        ctx_omni->model_tts = tts_model;
        ctx_omni->ctx_tts_llama = ctx_tts_llama;
        ctx_omni->ctx_tts_sampler = tts_sampler;
        
        // Load TTS weights from GGUF file
        print_with_timestamp("TTS: loading weights from GGUF (emb_code, emb_text, projector_semantic, head_code)...\n");
        if (!load_tts_weights_from_gguf(ctx_omni, params->tts_model.c_str())) {
            LOG_ERR("%s: error: failed to load TTS weights from %s\n", __func__, params->tts_model.c_str());
            llama_free(ctx_tts_llama);
            llama_free_model(tts_model);
            common_sampler_free(tts_sampler);
            llama_free(ctx_llama);
            llama_free_model(model);
            common_sampler_free(sampler);
            delete ctx_omni;
            return NULL;
        }
        print_with_timestamp("TTS: weights loaded successfully\n");
        
        // Load Projector Semantic from GGUF file
        // 路径: {tts_bin_dir}/MiniCPM-o-4_5-projector-F16.gguf
        std::string projector_path = tts_bin_dir + "/MiniCPM-o-4_5-projector-F16.gguf";
        print_with_timestamp("Projector: loading from %s\n", projector_path.c_str());
        if (projector_init(ctx_omni->projector, projector_path, true)) {
            print_with_timestamp("Projector: loaded successfully\n");
        } else {
            print_with_timestamp("Projector: failed to load, will use fallback implementation\n");
        }
    }

    ctx_omni->omni_emb.resize((64 + 10 + 1) * 4096); // temp fix for omni embed
    ctx_omni->audio_emb.resize((10 + 1) * 4096); // temp fix for audio embed
    print_with_timestamp("=== omni_init: loading APM model\n");
    if (params->apm_model.empty()) {
        LOG_ERR("%s: error: apm_model path is empty\n", __func__);
        if (ctx_omni->use_tts) {
            llama_free(ctx_omni->ctx_tts_llama);
            llama_free_model(ctx_omni->model_tts);
            common_sampler_free(ctx_omni->ctx_tts_sampler);
        }
        llama_free(ctx_llama);
        llama_free_model(model);
        common_sampler_free(sampler);
        delete ctx_omni;
        return NULL;
    }
    ctx_omni->ctx_audio = audition_init(params->apm_model.c_str(), audition_context_params{true, GGML_LOG_LEVEL_INFO});
    print_with_timestamp("APM: init from %s\n", params->apm_model.c_str());
    if (ctx_omni->ctx_audio == nullptr) {
        LOG_ERR("%s: error: failed to init audition model from %s\n", __func__, params->apm_model.c_str());
        // 清理 TTS 模型资源（如果已加载）
        if (ctx_omni->use_tts) {
            llama_free(ctx_omni->ctx_tts_llama);
            llama_free_model(ctx_omni->model_tts);
            common_sampler_free(ctx_omni->ctx_tts_sampler);
        }
        llama_free(ctx_llama);
        llama_free_model(model);
        common_sampler_free(sampler);
        delete ctx_omni;
        return NULL;
    }

    ctx_omni->n_past = 0;
    
    if (media_type == 2) {
        LOG_INF("init vision....");
        const char * vision_path = ctx_omni->params->vpm_model.c_str();
        auto * ctx_vision = vision_init(vision_path, vision_context_params{true, GGML_LOG_LEVEL_INFO, nullptr});
        ctx_omni->ctx_vision = ctx_vision;

        // batch encode 开关由 common_params 控制（默认关闭）
        if (ctx_vision) {
            vision_set_batch_encode(ctx_vision, ctx_omni->params->vpm_batch_encode);
        }

        // Set CoreML model path if available (for vision ANE acceleration)
        // Note: .mlmodelc is a directory, not a file, so use stat instead of ifstream
        if (ctx_vision && !ctx_omni->params->vision_coreml_model_path.empty()) {
            struct stat coreml_stat;
            if (stat(ctx_omni->params->vision_coreml_model_path.c_str(), &coreml_stat) == 0) {
                vision_set_coreml_model_path(ctx_vision, ctx_omni->params->vision_coreml_model_path.c_str());
                LOG_INF("Vision CoreML model path set to: %s\n", ctx_omni->params->vision_coreml_model_path.c_str());
            } else {
                LOG_WRN("Vision CoreML model path does not exist: %s, skipping ANE\n", ctx_omni->params->vision_coreml_model_path.c_str());
            }
        }
    }
    
    ctx_omni->llm_thread_info = new LLMThreadInfo(1000);
    if (ctx_omni->use_tts) {
        LOG_INF("init tts....");
        ctx_omni->tts_thread_info = new TTSThreadInfo(1);
        ctx_omni->omni_output = new omni_output();
        ctx_omni->tts_bin_dir = tts_bin_dir;
        
        // Initialize T2W thread info
        LOG_INF("init t2w....");
        ctx_omni->t2w_thread_info = new T2WThreadInfo(25);  // Queue size of 10 chunks
        
        // Initialize C++ Token2Wav session
        // Try to load token2wav GGUF models from {model_dir}/token2wav-gguf/
        // Fallback to tools/omni/token2wav-gguf if not found
        ctx_omni->token2wav_initialized = false;
        
        // 使用 Python Token2Wav 时跳过 C++ 初始化以节省显存
        bool skip_cpp_token2wav = ctx_omni->use_python_token2wav;
        
        // Check if token2wav model files exist
        // 优先检查 HF 模型目录下的 token2wav-gguf (tts_bin_dir 的父目录)
        // 目录结构: {model_dir}/token2wav-gguf/
        std::string gguf_root_dir = tts_bin_dir;
        size_t last_slash = gguf_root_dir.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            gguf_root_dir = gguf_root_dir.substr(0, last_slash);  // 获取 tts 的父目录
        }
        ctx_omni->token2wav_model_dir = gguf_root_dir + "/token2wav-gguf";
        
        std::string encoder_test = ctx_omni->token2wav_model_dir + "/encoder.gguf";
        {
            std::ifstream f(encoder_test);
            if (!f.good()) {
                // 尝试备用路径 (本地开发用)
                ctx_omni->token2wav_model_dir = "tools/omni/token2wav-gguf";
                print_with_timestamp("Token2Wav: trying fallback path %s\n", ctx_omni->token2wav_model_dir.c_str());
            } else {
                print_with_timestamp("Token2Wav: found models in %s\n", ctx_omni->token2wav_model_dir.c_str());
            }
        }
        std::string encoder_gguf = ctx_omni->token2wav_model_dir + "/encoder.gguf";
        std::string flow_matching_gguf = ctx_omni->token2wav_model_dir + "/flow_matching.gguf";
        std::string flow_extra_gguf = ctx_omni->token2wav_model_dir + "/flow_extra.gguf";
        std::string vocoder_gguf = ctx_omni->token2wav_model_dir + "/hifigan2.gguf";
        std::string prompt_cache_gguf = ctx_omni->token2wav_model_dir + "/prompt_cache.gguf";
        
        // Check if all files exist
        bool all_files_exist = true;
        std::vector<std::string> gguf_files = {encoder_gguf, flow_matching_gguf, flow_extra_gguf, vocoder_gguf};
        for (const auto& file : gguf_files) {
            std::ifstream f(file);
            if (!f.good()) {
                all_files_exist = false;
                break;
            } else {
            }
        }
        
        if (all_files_exist && !skip_cpp_token2wav) {
            print_with_timestamp("Token2Wav: all model files found, initializing session...\n");
            ctx_omni->token2wav_session = std::make_unique<omni::flow::Token2WavSession>();
            
            // Device configuration - 使用 omni_init 传入的 token2wav_device 参数
            // 格式: "gpu", "gpu:0", "gpu:1", "cpu"
            std::string device_token2mel = token2wav_device;

            // Vocoder 设备策略：
            //   CUDA: vocoder 跟随 token2wav_device（GPU），因为 CUDA kernel launch 开销低
            //   Metal (macOS): vocoder 强制用 CPU，因为 vocoder 有大量小操作，
            //     Metal kernel dispatch 开销累积后远慢于 CPU 直接计算
            //   可通过 OMNI_VOC_DEVICE 环境变量覆盖
            const char * voc_dev_env = getenv("OMNI_VOC_DEVICE");
            std::string device_vocoder;
            if (voc_dev_env) {
                device_vocoder = voc_dev_env;
                print_with_timestamp("Token2Wav: vocoder device overridden by OMNI_VOC_DEVICE=%s\n", voc_dev_env);
            } else {
#ifdef GGML_USE_CUDA
                device_vocoder = token2wav_device;
                print_with_timestamp("Token2Wav: CUDA detected, vocoder using GPU (%s)\n", device_vocoder.c_str());
#else
                device_vocoder = "cpu";
                print_with_timestamp("Token2Wav: non-CUDA backend, vocoder using CPU for better performance\n");
#endif
            }
            
            // 优先用 prompt_cache.gguf；失败时 fallback 到 prompt_bundle（实时计算）
            std::string prompt_bundle_dir = "tools/omni/assets/default_ref_audio";
            std::string spk_file = prompt_bundle_dir + "/spk_f32.bin";
            std::string tokens_file = prompt_bundle_dir + "/prompt_tokens_i32.bin";
            std::string mel_file = prompt_bundle_dir + "/prompt_mel_btc_f32.bin";
            
            bool use_prompt_bundle = false;
            {
                std::ifstream f1(spk_file), f2(tokens_file), f3(mel_file);
                use_prompt_bundle = f1.good() && f2.good() && f3.good();
            }
            
            bool init_ok = false;
            // 优先级: prompt_cache.gguf > prompt_bundle (实时计算 fallback)
            print_with_timestamp("Token2Wav: using prompt_cache from %s\n", prompt_cache_gguf.c_str());

            // CoreML 后端选择：通过 params->token2wav_coreml_model_path 控制。
            // - 空字符串（默认）：纯 GPU/GGUF 路径，不启用 CoreML。
            // - "auto"：自动在 token2wav-gguf 目录查找 CoreML 模型。
            // - 具体路径：使用指定 CoreML 模型。
            std::string coreml_model_path;
            if (!ctx_omni->params->token2wav_coreml_model_path.empty()) {
                if (ctx_omni->params->token2wav_coreml_model_path == "auto") {
                    // 与 GGUF 文件同级目录下查找
                    std::string auto_pkg = ctx_omni->token2wav_model_dir + "/coreml_minicpmo45_t2w_dit.mlpackage";
                    std::ifstream probe(auto_pkg + "/Manifest.json");
                    if (probe.good()) {
                        coreml_model_path = auto_pkg;
                    } else {
                        // fallback: 尝试带完整文件名
                        auto_pkg = ctx_omni->token2wav_model_dir + "/coreml_minicpmo45_t2w_dit_f16_chunk56_cache600.mlpackage";
                        std::ifstream probe2(auto_pkg + "/Manifest.json");
                        if (probe2.good()) {
                            coreml_model_path = auto_pkg;
                        } else {
                            print_with_timestamp("Token2Wav: CoreML auto mode, but no CoreML model found in %s\n",
                                                 ctx_omni->token2wav_model_dir.c_str());
                        }
                    }
                } else {
                    coreml_model_path = ctx_omni->params->token2wav_coreml_model_path;
                }
            }
            if (!coreml_model_path.empty()) {
                print_with_timestamp("Token2Wav: CoreML model = %s\n", coreml_model_path.c_str());
            }

            // flow 的 diffusion 步数（默认 5），可用 OMNI_T2W_N_TIMESTEPS 覆盖。
            int n_ts = 5;
            if (const char * e = getenv("OMNI_T2W_N_TIMESTEPS")) { int v = atoi(e); if (v > 0) n_ts = v; }
            init_ok = ctx_omni->token2wav_session->init_from_prompt_cache_gguf(
                    encoder_gguf, flow_matching_gguf, flow_extra_gguf, prompt_cache_gguf,
                    vocoder_gguf, device_token2mel, device_vocoder, n_ts, 1.0f, coreml_model_path);
            if (!init_ok && use_prompt_bundle) {
                print_with_timestamp("Token2Wav: prompt_cache failed, fallback to prompt_bundle from %s\n", prompt_bundle_dir.c_str());
                init_ok = ctx_omni->token2wav_session->init_from_prompt_bundle(
                        encoder_gguf, flow_matching_gguf, flow_extra_gguf, prompt_bundle_dir,
                        vocoder_gguf, device_token2mel, device_vocoder, n_ts, 1.0f);
            }
            // Fallback to CPU
            if (!init_ok) {
                print_with_timestamp("Token2Wav: GPU init failed, trying CPU mode...\n");
                ctx_omni->token2wav_session.reset();
                ctx_omni->token2wav_session = std::make_unique<omni::flow::Token2WavSession>();
                init_ok = ctx_omni->token2wav_session->init_from_prompt_cache_gguf(
                        encoder_gguf, flow_matching_gguf, flow_extra_gguf, prompt_cache_gguf,
                        vocoder_gguf, "cpu", "cpu", n_ts, 1.0f);
            }
            
            if (init_ok) {
                ctx_omni->token2wav_initialized = true;
                // Initialize token2wav buffer with 3 silence tokens (4218) as Python does
                // Python: buffer = [4218] * 3  # 预先放入3个前缀静音token
                ctx_omni->token2wav_buffer.clear();
                ctx_omni->token2wav_buffer = {4218, 4218, 4218};
                ctx_omni->token2wav_wav_idx = 0;
                print_with_timestamp("Token2Wav: initialized successfully\n");
            } else {
                ctx_omni->token2wav_session.reset();
                print_with_timestamp("Token2Wav: initialization failed\n");
            }
        } else {
            print_with_timestamp("Token2Wav: model files not found in %s\n", ctx_omni->token2wav_model_dir.c_str());
        }
        
        // ==================== 初始化 Python Token2Wav ====================
        // 默认使用 Python Token2Wav（精度更高）。设置脚本目录与模型目录。
        // Python T2W 脚本目录：tools/omni/pyt2w/
        // Python T2W 模型目录：dependencies/token2wav/
        
        // 计算 Python T2W 脚本目录（相对于 tts_bin_dir）
        // tts_bin_dir 通常是 /xxx/tools/omni/convert/gguf/token2wav-gguf
        // 我们需要 /xxx/tools/omni/pyt2w
        std::string t2w_script_dir = tts_bin_dir;  // /xxx/tools/omni/convert/gguf/token2wav-gguf
        // 回退到 tools/omni/
        size_t convert_pos = t2w_script_dir.find("/convert/gguf/tts");
        if (convert_pos != std::string::npos) {
            t2w_script_dir = t2w_script_dir.substr(0, convert_pos) + "/pyt2w";
        } else if ((convert_pos = t2w_script_dir.find("/convert/gguf")) != std::string::npos) {
            t2w_script_dir = t2w_script_dir.substr(0, convert_pos) + "/pyt2w";
        } else {
            // 尝试从当前工作目录构建
            t2w_script_dir = "./tools/omni/pyt2w";
        }
        ctx_omni->python_t2w_script_dir = t2w_script_dir;
        
        // Python T2W 模型目录（stepaudio2 模型）
        // 默认路径：相对于 script_dir 的 token2wav 子目录
        ctx_omni->python_t2w_model_dir = t2w_script_dir + "/token2wav";
        
        // 参考音频路径
        std::string ref_audio_path = "tools/omni/assets/default_ref_audio/default_ref_audio.wav";
        
        print_with_timestamp("Python T2W: script_dir=%s, model_dir=%s\n", 
                             ctx_omni->python_t2w_script_dir.c_str(),
                             ctx_omni->python_t2w_model_dir.c_str());
        
        if (ctx_omni->use_python_token2wav) {
            print_with_timestamp("Python T2W: 使用 Python Token2Wav 实现\n");
            
            // Python T2W GPU 配置：C++ LLM+TTS 约 22GB、Python T2W 约 3.3GB，
            // 单卡 24GB 放不下，需独立 GPU。可用环境变量 PYTHON_T2W_GPU 指定。
            // 优先级：PYTHON_T2W_GPU > 外部 CUDA_VISIBLE_DEVICES > token2wav_device。
            const char* env_python_t2w_gpu = getenv("PYTHON_T2W_GPU");
            if (env_python_t2w_gpu && strlen(env_python_t2w_gpu) > 0) {
                ctx_omni->python_t2w_dedicated_gpu = env_python_t2w_gpu;
            }
            
            ctx_omni->python_t2w_gpu_id = "";
            
            if (!ctx_omni->python_t2w_dedicated_gpu.empty()) {
                // 使用配置的独立 GPU
                ctx_omni->python_t2w_gpu_id = ctx_omni->python_t2w_dedicated_gpu;
                print_with_timestamp("Python T2W: 使用独立 GPU %s (C++ 和 Python 分开)\n", ctx_omni->python_t2w_gpu_id.c_str());
            } else {
                const char* env_cuda_visible = getenv("CUDA_VISIBLE_DEVICES");
                if (env_cuda_visible && strlen(env_cuda_visible) > 0) {
                    // 外部已设置，Python 子进程会继承，不需要额外设置
                    print_with_timestamp("Python T2W: 继承外部 CUDA_VISIBLE_DEVICES=%s (与 C++ 共用)\n", env_cuda_visible);
                } else if (token2wav_device.find("gpu") != std::string::npos) {
                    // 外部未设置，从 token2wav_device 提取
                    size_t colon_pos = token2wav_device.find(':');
                    if (colon_pos != std::string::npos) {
                        ctx_omni->python_t2w_gpu_id = token2wav_device.substr(colon_pos + 1);
                    } else {
                        ctx_omni->python_t2w_gpu_id = "0";
                    }
                    print_with_timestamp("Python T2W: 设置 CUDA_VISIBLE_DEVICES=%s (与 C++ 共用)\n", ctx_omni->python_t2w_gpu_id.c_str());
                } else {
                    print_with_timestamp("Python T2W: CPU 模式\n");
                }
            }
            
            // 启动 Python 服务
            if (start_python_t2w_service(ctx_omni)) {
                // 初始化模型
                if (init_python_t2w_model(ctx_omni, token2wav_device)) {
                    // 设置参考音频
                    if (set_python_t2w_ref_audio(ctx_omni, ref_audio_path)) {
                        print_with_timestamp("Python T2W: 初始化成功\n");
                    } else {
                        print_with_timestamp("Python T2W: 设置参考音频失败\n");
                        ctx_omni->use_python_token2wav = false;
                    }
                } else {
                    print_with_timestamp("Python T2W: 初始化模型失败\n");
                    ctx_omni->use_python_token2wav = false;
                }
            } else {
                print_with_timestamp("Python T2W: 启动服务失败\n");
                ctx_omni->use_python_token2wav = false;
            }
            
            // 如果 Python 初始化失败，回退到 C++ 实现
            if (!ctx_omni->use_python_token2wav) {
                print_with_timestamp("Python T2W: 回退到 C++ 实现\n");
            }
        } else {
            print_with_timestamp("Token2Wav: 使用 C++ 实现\n");
        }
    }
    ctx_omni->async = true;
    
    // ==================== 初始化特殊 Token ID ====================
    // 从 LLM 词表中查找并缓存特殊 token ID
    // 这些 token 用于控制双工模式下的状态切换
    
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab) {
        // 使用 llama_tokenize 直接将字符串转换为 token ID
        // parse_special=true 确保特殊 token 被正确解析
        auto find_token = [&](const char * token_str) -> llama_token {
            llama_token tokens[4];  // 预留空间
            int n_tokens = llama_tokenize(vocab, token_str, strlen(token_str), tokens, 4, false, true);
            if (n_tokens == 1) {
                return tokens[0];
            }
            // 如果 tokenize 失败，尝试遍历词表查找（使用 special=true）
            int n_vocab = llama_vocab_n_tokens(vocab);
            for (int i = 0; i < n_vocab; i++) {
                char buf[128];
                int len = llama_token_to_piece(vocab, i, buf, sizeof(buf), 0, true);  // special=true
                if (len > 0 && len < (int)sizeof(buf)) {
                    buf[len] = '\0';
                    if (strcmp(buf, token_str) == 0) {
                        return i;
                    }
                }
            }
            return -1;
        };
        
        ctx_omni->special_token_speak = find_token("<|speak|>");
        ctx_omni->special_token_listen = find_token("<|listen|>");
        ctx_omni->special_token_chunk_eos = find_token("<|chunk_eos|>");
        ctx_omni->special_token_chunk_tts_eos = find_token("<|chunk_tts_eos|>");
        ctx_omni->special_token_turn_eos = find_token("<|turn_eos|>");
        ctx_omni->special_token_tts_eos = find_token("<|tts_eos|>");
        ctx_omni->special_token_eos = llama_vocab_eos(vocab);
        
        // 同时初始化 tts_bos_token_id（用于双工模式强制继续说话）
        llama_token tts_bos = find_token("<|tts_bos|>");
        if (tts_bos >= 0) {
            ctx_omni->tts_bos_token_id = tts_bos;
        }
        // 初始化 </unit> token（用于双工模式 chunk 边界标记）
        ctx_omni->special_token_unit_end = find_token("</unit>");
        
        // 初始化 <|tts_pad|> token（双工模式下禁止采样；对应 Python forbidden_token_ids）
        ctx_omni->special_token_tts_pad = find_token("<|tts_pad|>");
    }
        
    // ANE/CoreML warmup: pre-load models into NPU to avoid first-inference latency
    omni_warmup_ane(ctx_omni);

    if (ctx_omni->duplex_mode) {
        // Prepare the fused embedding cache without starting worker threads so
        // the first real session can reuse the same P topology.
        duplex_prepare(ctx_omni, params);
    }
#ifdef GGML_USE_CUDA
    // 段租约 SM 调度器装配（presets + backends）。OMNI_SM_SMS 未设时 no-op。
    // 在 duplex_prepare 之后（backends 已就绪）、prewarm 之前。
    sm_sched_init(ctx_omni);
    // 预热：代表输入 graph 预热 + vocoder 预热（纯 init，无条件跑）。
    omni_prewarm(ctx_omni);
#endif

    print_with_timestamp("=== omni_init success: ctx_llama = %p\n", (void*)ctx_omni->ctx_llama);
    return ctx_omni;
}

//
// ANE/CoreML warmup — pre-load models into NPU to avoid first-inference latency
//
void omni_warmup_ane(struct omni_context * ctx_omni) {
#if defined(__APPLE__)
    if (!ctx_omni) return;

    LOG_INF("%s: starting ANE/CoreML warmup...\n", __func__);

    // 1. Vision ANE warmup
    if (ctx_omni->ctx_vision) {
        vision_coreml_warmup(ctx_omni->ctx_vision);
    }

    // 2. Future: audio ANE warmup
    // if (ctx_omni->ctx_audio) {
    //     audition_coreml_warmup(ctx_omni->ctx_audio);
    // }

    // 3. Future: other module ANE warmup
    // ...

    LOG_INF("%s: ANE/CoreML warmup finished\n", __func__);
#else
    (void)ctx_omni;
#endif
}

bool omni_tts_queues_empty(struct omni_context * ctx_omni) {
    bool tts_empty = true, t2w_empty = true;
    if (ctx_omni->tts_thread_info) {
        std::lock_guard<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
        tts_empty = ctx_omni->tts_thread_info->queue.empty();
    }
    if (ctx_omni->t2w_thread_info) {
        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
        t2w_empty = ctx_omni->t2w_thread_info->queue.empty();
    }
    return tts_empty && t2w_empty;
}

// 停止所有线程（发送信号，不等待）
void omni_stop_threads(struct omni_context * ctx_omni) {
    // 发送停止信号
    llm_thread_running = false;
    tts_thread_running = false;
    t2w_thread_running = false;

    // 唤醒所有等待的线程
    if (ctx_omni->llm_thread_info) {
        ctx_omni->llm_thread_info->cv.notify_all();
    }
    if (ctx_omni->tts_thread_info) {
        ctx_omni->tts_thread_info->cv.notify_all();
    }
    if (ctx_omni->t2w_thread_info) {
        ctx_omni->t2w_thread_info->cv.notify_all();
    }

    // 停止 duplex 双线程
    if (ctx_omni->duplex != nullptr) {
        ctx_omni->duplex->running.store(false);
        ctx_omni->duplex->pending_cv.notify_all();
        ctx_omni->duplex->llm_cv.notify_all();
    }

    print_with_timestamp("omni_stop_threads: stop signals sent\n");
}

// Reset a context between backend-protocol sessions: end any duplex session,
// stop+join the LLM/TTS/token2wav worker threads and drain their queues, while
// keeping the loaded model alive so the next session can reuse this context.
void omni_prepare_for_reuse(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return;
    }

    if (ctx_omni->duplex_session) {
        omni_duplex_session_end(ctx_omni);
    }
    duplex_stop_threads(ctx_omni);

    llm_thread_running = false;
    if (ctx_omni->llm_thread_info) {
        ctx_omni->llm_thread_info->cv.notify_all();
    }
    if (ctx_omni->llm_thread.joinable()) {
        ctx_omni->llm_thread.join();
    }

    tts_thread_running = false;
    if (ctx_omni->tts_thread_info) {
        ctx_omni->tts_thread_info->cv.notify_all();
    }
    if (ctx_omni->tts_thread.joinable()) {
        ctx_omni->tts_thread.join();
    }

    t2w_thread_running = false;
    if (ctx_omni->t2w_thread_info) {
        ctx_omni->t2w_thread_info->cv.notify_all();
    }
    if (ctx_omni->t2w_thread.joinable()) {
        ctx_omni->t2w_thread.join();
    }

    if (ctx_omni->llm_thread_info) {
        std::lock_guard<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
        while (!ctx_omni->llm_thread_info->queue.empty()) {
            delete ctx_omni->llm_thread_info->queue.front();
            ctx_omni->llm_thread_info->queue.pop();
        }
    }
    if (ctx_omni->tts_thread_info) {
        std::lock_guard<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
        while (!ctx_omni->tts_thread_info->queue.empty()) {
            delete ctx_omni->tts_thread_info->queue.front();
            ctx_omni->tts_thread_info->queue.pop();
        }
    }
    if (ctx_omni->t2w_thread_info) {
        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
        while (!ctx_omni->t2w_thread_info->queue.empty()) {
            delete ctx_omni->t2w_thread_info->queue.front();
            ctx_omni->t2w_thread_info->queue.pop();
        }
    }

    print_with_timestamp("omni_prepare_for_reuse: inference threads stopped and queues cleared\n");
}

void omni_free(struct omni_context * ctx_omni) {

    // 高层 duplex session 必须在底层 duplex pipeline 之前结束：
    //   session_end 会尝试 drain 已 push 但未完成的 frame，
    //   这要求 duplex 内部线程依然运行。
    if (ctx_omni->duplex_session) {
        omni_duplex_session_end(ctx_omni);
    }

    // 先停 duplex（它可能正在调用 llama_decode）
    duplex_stop_threads(ctx_omni);

    // 等待 llm 和 tts thread 停止
    llm_thread_running = false; // Signal the thread to stop
    if (ctx_omni->llm_thread.joinable()) {
        ctx_omni->llm_thread_info->cv.notify_all(); // Wake up the thread if it's waiting
        ctx_omni->llm_thread.join(); // Wait for the thread to finish
    }
 
    if (ctx_omni->use_tts) {
        tts_thread_running = false; // Signal the thread to stop
        if (ctx_omni->tts_thread.joinable()) {
            ctx_omni->tts_thread_info->cv.notify_all(); // Wake up the thread if it's waiting
            ctx_omni->tts_thread.join(); // Wait for the thread to finish
        }
        
        // Stop T2W thread
        t2w_thread_running = false; // Signal the thread to stop
        if (ctx_omni->t2w_thread.joinable()) {
            ctx_omni->t2w_thread_info->cv.notify_all(); // Wake up the thread if it's waiting
            ctx_omni->t2w_thread.join(); // Wait for the thread to finish
        }
    }
    
    vision_free(ctx_omni->ctx_vision);
    audition_free(ctx_omni->ctx_audio);
    
    if (ctx_omni->use_tts) {
        llama_free(ctx_omni->ctx_tts_llama);
        llama_free_model(ctx_omni->model_tts);
        common_sampler_free(ctx_omni->ctx_tts_sampler);

        if (ctx_omni->tts_condition_graph.initialized) {
            tts_condition_graph_free(ctx_omni);
        }
        
        // Free TTS weights
        if (ctx_omni->emb_code_weight) {
            free(ctx_omni->emb_code_weight);
            ctx_omni->emb_code_weight = nullptr;
        }
        if (ctx_omni->emb_text_weight) {
            free(ctx_omni->emb_text_weight);
            ctx_omni->emb_text_weight = nullptr;
        }
        if (ctx_omni->projector_semantic_linear1_weight) {
            free(ctx_omni->projector_semantic_linear1_weight);
            ctx_omni->projector_semantic_linear1_weight = nullptr;
        }
        if (ctx_omni->projector_semantic_linear1_bias) {
            free(ctx_omni->projector_semantic_linear1_bias);
            ctx_omni->projector_semantic_linear1_bias = nullptr;
        }
        if (ctx_omni->projector_semantic_linear2_weight) {
            free(ctx_omni->projector_semantic_linear2_weight);
            ctx_omni->projector_semantic_linear2_weight = nullptr;
        }
        if (ctx_omni->projector_semantic_linear2_bias) {
            free(ctx_omni->projector_semantic_linear2_bias);
            ctx_omni->projector_semantic_linear2_bias = nullptr;
        }
        if (ctx_omni->head_code_weight) {
            free(ctx_omni->head_code_weight);
            ctx_omni->head_code_weight = nullptr;
        }
        
        // Free C++ Token2Wav session
        if (ctx_omni->token2wav_session) {
            ctx_omni->token2wav_session.reset();
            ctx_omni->token2wav_initialized = false;
            LOG_INF("Token2Wav (C++): session released\n");
        }
        
        // 停止 Python Token2Wav 服务
        if (ctx_omni->python_t2w_initialized) {
            stop_python_t2w_service(ctx_omni);
        }
        
        // Free ggml-based projector model
        if (ctx_omni->projector.initialized) {
            projector_free(ctx_omni->projector);
        }
    }
    
    // 仅在拥有模型时释放 LLM model 和 context；外部传入（模型复用）则不释放
    if (ctx_omni->owns_model) {
        llama_free(ctx_omni->ctx_llama);
        llama_free_model(ctx_omni->model);
    }
    common_sampler_free(ctx_omni->ctx_sampler);
    // delete ctx_omni->ctx_tts;
    delete ctx_omni->llm_thread_info;
    delete ctx_omni->audio_input_manager;
    
    // omni_output 还要把里面 output 的每个元素也 delete 下
    if (ctx_omni->use_tts) {
        delete ctx_omni->tts_thread_info;
        delete ctx_omni->t2w_thread_info;
        
        // omni_output 还要把里面 output 的每个元素也 delete 下
        if (ctx_omni->omni_output) {
            for (auto &buffer : ctx_omni->omni_output->output) {
                delete buffer;
            }
            ctx_omni->omni_output->output.clear(); // Clear the vector
            delete ctx_omni->omni_output;
        }
    }

    sm_sched_free(ctx_omni->sm_sched);

    // NOTE: do NOT call llama_backend_free() here — the backend/model is shared
    // and owned by the server; freeing it would break the still-running process.

    delete ctx_omni;
}

// ==================== 语言设置函数 ====================
// 设置语言并更新 system prompt（zh=中文，en=英文）
// 基于 Python MiniCPM-o-4_5 modeling_minicpmo.py 中的 audio_assistant 模式 prompt
void omni_set_language(struct omni_context * ctx_omni, const std::string & lang) {
    if (ctx_omni == nullptr) {
        print_with_timestamp("omni_set_language: ctx_omni is null\n");
        return;
    }
    
    ctx_omni->language = lang;
    print_with_timestamp("omni_set_language: setting language to '%s'\n", lang.c_str());
    
    if (ctx_omni->duplex_mode) {
        // 双工模式：prompt 固定使用英文（与 Python 对齐）
        ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|><|im_end|>\n";
        ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|><|im_end|>\n";
    } else {
        // 非双工模式（audio_assistant 模式）：根据语言设置 prompt
        if (lang == "en") {
            // 英文 prompt（来自 Python modeling_minicpmo.py）
            ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>";
            ctx_omni->audio_assistant_prompt = "<|audio_end|>Please assist users while maintaining this voice style. Please answer the user's questions seriously and in a high quality. Please chat with the user in a highly human-like and oral style. You are a helpful assistant developed by ModelBest: MiniCPM-Omni.<|im_end|>\n<|im_start|>user\n";
            
            ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>";
            ctx_omni->omni_assistant_prompt = "<|audio_end|>Please assist users while maintaining this voice style. Please answer the user's questions seriously and in a high quality. Please chat with the user in a highly human-like and oral style.<|im_end|>\n<|im_start|>user\n";
        } else {
            // 中文 prompt（默认，来自 Python modeling_minicpmo.py）
            ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
            ctx_omni->audio_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。<|im_end|>\n<|im_start|>user\n";
            
            ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
            ctx_omni->omni_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。<|im_end|>\n<|im_start|>user\n";
        }
    }
    
    // 重置 system_prompt_initialized，让下次 stream_prefill(index=0) 重新 prefill system prompt
    ctx_omni->system_prompt_initialized = false;
    
    print_with_timestamp("omni_set_language: prompts updated for language '%s', system_prompt_initialized reset to false\n", lang.c_str());
}

// 遗留：无活跃调用方。把音频 embedding 直接 eval 进 KV（早期单工路径使用）。
static void process_audio(struct omni_context * ctx_omni, struct omni_embed * embeds, common_params * params, bool save_spk_emb=false) {
    LOG_INF("%s: audio token past: %d\n", __func__, ctx_omni->n_past);
    omni_eval_embed(ctx_omni->ctx_llama, embeds, params->n_batch, &ctx_omni->n_past);
    LOG_INF("%s: audio token past after eval: %d\n", __func__, ctx_omni->n_past);
}

// 把 "<|im_start|>user\n" 前缀写入 KV。duplex 的 stream_prefill(index=0) 会调用。
void eval_prefix(struct omni_context* ctx_omni, common_params* params){
    std::string prefix = "<|im_start|>user\n";
    std::cout << "prefix : " << prefix << std::endl;
    eval_string(ctx_omni, params, prefix.c_str(), params->n_batch, &ctx_omni->n_past, false);
}

// 遗留：无活跃调用方。同 eval_prefix，但返回 hidden states。
void eval_prefix_with_hidden(struct omni_context* ctx_omni, common_params* params, float *& hidden_states){
    std::string prefix = "<|im_start|>user\n";
    std::cout << "prefix : " << prefix << std::endl;
    eval_string_with_hidden(ctx_omni, params, prefix.c_str(), params->n_batch, &ctx_omni->n_past, false, hidden_states);
}

/**
 * 遗留：单工（非 duplex）模式 LLM 线程。活跃路径是 duplex_llm_thread_func（duplex_core.cpp）。
 *
 * 负责多模态（视觉+音频）嵌入的前缀填充（prefill）：
 * - 主循环运行到 llm_thread_running=false；
 * - 等待条件：队列非空 OR need_speek=true OR 线程停止；
 * - 分支1：队列非空 → 处理嵌入数据的 prefill；
 * - 分支2：队列空且 need_speek=true → 通知解码线程可开始生成。
 */
void llm_thread_func(omni_context* ctx_omni, common_params* params){
    print_with_timestamp("LLM thread started\n");
    // 获取模型的隐藏层维度，用于计算token数量
    const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    
    // ========== 主循环：持续处理嵌入数据 ==========
    while(llm_thread_running){
        // 获取队列的互斥锁，保护共享资源
        std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
        auto& queue = ctx_omni->llm_thread_info->queue;

        // 打印当前状态（调试用）
        
        // ========== 等待条件满足 ==========
        // 等待以下任一条件满足：
        // 1. 队列不为空（有新嵌入数据需要处理）
        // 2. need_speek 为 true（需要开始生成文本）
        // 3. llm_thread_running 为 false（线程需要停止）
        ctx_omni->llm_thread_info->cv.wait(lock, [&] { 
            return !queue.empty() || ctx_omni->need_speek || !llm_thread_running; 
        });
        
        // 检查是否需要停止线程
        if (!llm_thread_running) {
            break;
        }

        // ========== 分支1：处理队列中的嵌入数据（前缀填充） ==========
        if (!queue.empty()){
            // 打印 prefill 开始时的 n_past
            print_with_timestamp("LLM thread: start prefill, n_past=%d, n_keep=%d, n_ctx=%d\n",
                                 ctx_omni->n_past, ctx_omni->n_keep, params->n_ctx);

            // prefill 阶段不清除 KV cache（无论单双工都累积用户输入）。
            // KV cache 仅在：1. 新会话开始（reset API）；2. 滑动窗口触发（context 满）。
            print_with_timestamp("LLM thread: prefill continuing, n_past=%d (no KV cache clear)\n", ctx_omni->n_past);
            
            // 标记前缀填充未完成，防止解码线程过早开始
            prefill_done = false;
            
            // 步骤1：批量取出队列中的所有嵌入数据
            // 这样可以一次性处理多个嵌入，提高效率
            std::vector<omni_embeds*> llm_embeds;
            llm_embeds.clear();
            while (!queue.empty()) {
                llm_embeds.push_back(queue.front());
                queue.pop();
            }
            
            // 释放锁，允许其他线程继续向队列添加数据
            lock.unlock();
            
            // 如果批量处理多个嵌入，打印日志
            print_with_timestamp("Batch processing %zu llm prefill\n", llm_embeds.size());
            if (llm_embeds.size() > 1)
            
            // 通知等待的生产者线程，队列有空间了
            ctx_omni->llm_thread_info->cv.notify_all();

            // 只有非双工模式才需要 <|im_start|>user\n：第一轮已含在 sys prompt 末尾，
            // 后续轮次在 stream_decode 结束时添加，这里不再动态添加。
            // 双工模式直接用 <unit> 标记用户输入开始。

            // 逐个处理嵌入数据，按模式添加特殊标记
            for (int il = 0; il < (int)llm_embeds.size(); ++il) {
                auto embeds = llm_embeds[il];

                // 注册 unit 开始（滑窗记账）
                if (ctx_omni->sliding_window_config.mode != "off") {
                    sliding_window_register_unit_start(ctx_omni);
                }

                // ========== 子分支1：处理包含视觉嵌入的数据 ==========
                // vision_embed 是二维 vector: [0]=overview, [1..n]=slices
                if (embeds->vision_embed.size() > 0){
                    int n_chunks = (int)embeds->vision_embed.size();
                    int tokens_per_chunk = (int)embeds->vision_embed[0].size() / hidden_size;
                    int n_audio_tokens = embeds->audio_embed.size() / hidden_size;
                    bool has_audio = (n_audio_tokens > 0);
                    bool has_slices = (n_chunks > 1);
                    
                    // 按模式决定是否添加 <unit> 前缀
                    if (ctx_omni->duplex_mode) {
                        eval_string(ctx_omni, params, "<unit><image>", params->n_batch, &ctx_omni->n_past, false);
                    } else {
                        eval_string(ctx_omni, params, "<image>", params->n_batch, &ctx_omni->n_past, false);
                    }
                    
                    // Prefill overview embedding (第一个 chunk)
                    prefill_with_emb(ctx_omni, params, embeds->vision_embed[0].data(), tokens_per_chunk, 
                                    params->n_batch, &ctx_omni->n_past);
                    eval_string(ctx_omni, params, "</image>", params->n_batch, &ctx_omni->n_past, false);
                    
                    // V2.6 schema：有 slices 时加 <slice> 标记
                    // 格式: <image>(overview)</image><slice>(slice1)</slice><slice>(slice2)</slice>\n
                    if (has_slices) {
                        for (int i = 1; i < n_chunks; i++) {
                            eval_string(ctx_omni, params, "<slice>", params->n_batch, &ctx_omni->n_past, false);
                            prefill_with_emb(ctx_omni, params, embeds->vision_embed[i].data(), tokens_per_chunk,
                                            params->n_batch, &ctx_omni->n_past);
                            eval_string(ctx_omni, params, "</slice>", params->n_batch, &ctx_omni->n_past, false);
                        }
                        // V2.6 格式在 slices 后添加换行
                        eval_string(ctx_omni, params, "\n", params->n_batch, &ctx_omni->n_past, false);
                    }
                    
                    print_with_timestamp("Omni模式: %d vision chunks (%d tokens each), %d audio tokens, has_slices=%d\n", 
                                        n_chunks, tokens_per_chunk, n_audio_tokens, has_slices);
                    
                    // 音频部分
                    if (has_audio) {
                        if (!ctx_omni->duplex_mode) {
                            // 单工格式：<|audio_start|> + audio + <|audio_end|>
                            eval_string(ctx_omni, params, "<|audio_start|>", params->n_batch, &ctx_omni->n_past, false);
                        }
                        prefill_with_emb(ctx_omni, params, embeds->audio_embed.data(), n_audio_tokens,
                                        params->n_batch, &ctx_omni->n_past);
                        if (!ctx_omni->duplex_mode) {
                            eval_string(ctx_omni, params, "<|audio_end|>", params->n_batch, &ctx_omni->n_past, false);
                        }
                    }
                }
                // ========== 子分支2：处理只有音频嵌入的数据（纯音频模式） ==========
                else if (!embeds->audio_embed.empty()) {
                    int n_audio_tokens = embeds->audio_embed.size() / hidden_size;
                    print_with_timestamp("用户语音: %d audio tokens\n", n_audio_tokens);
                    
                    // 按模式选择格式
                    if (ctx_omni->duplex_mode) {
                        // 双工格式：<unit> + audio_embedding（无 audio_start/end）
                        eval_string(ctx_omni, params, "<unit>", params->n_batch, &ctx_omni->n_past, false);
                    } else {
                        // 单工格式：<|audio_start|> + audio + <|audio_end|>
                        eval_string(ctx_omni, params, "<|audio_start|>", params->n_batch, &ctx_omni->n_past, false);
                    }
                    
                    // Prefill 音频 embedding
                    prefill_with_emb(ctx_omni, params, embeds->audio_embed.data(), n_audio_tokens,
                                    params->n_batch, &ctx_omni->n_past);
                    
                    // 单工格式需要 <|audio_end|>
                    if (!ctx_omni->duplex_mode) {
                        eval_string(ctx_omni, params, "<|audio_end|>", params->n_batch, &ctx_omni->n_past, false);
                    }
                }
                // 子分支3：纯文本输入（turn-based chat 文本对话）
                // 单工：assistant_prompt 末尾已带 <|im_start|>user\n，文本本身不需要
                //       任何 <|audio_start|>/<|audio_end|> 之类的 modality 包裹。
                // 双工：与音频对齐，先 <unit> 再写入文本。
                else if (!embeds->user_text.empty()) {
                    if (ctx_omni->duplex_mode) {
                        eval_string(ctx_omni, params, "<unit>", params->n_batch, &ctx_omni->n_past, false);
                    }
                    eval_string(ctx_omni, params, embeds->user_text.c_str(), params->n_batch, &ctx_omni->n_past, false);
                }
                
                // 注册 unit 结束（滑窗记账）
                if (ctx_omni->sliding_window_config.mode != "off") {
                    std::string input_type = embeds->vision_embed.size() > 0 ? "omni"
                                           : !embeds->audio_embed.empty()    ? "audio"
                                           :                                   "text";
                    sliding_window_register_unit_end(ctx_omni, input_type, {}, false);
                }
                
                // 释放嵌入数据的内存（由生产者线程分配）
                delete embeds;
            }
            
            // 打印 prefill 结束后的 n_past
            print_with_timestamp("LLM thread: prefill done, n_past=%d, n_keep=%d, 本次消耗 %d tokens, duplex_mode=%d\n",
                                 ctx_omni->n_past, ctx_omni->n_keep,
                                 ctx_omni->n_past - ctx_omni->n_keep,
                                 ctx_omni->duplex_mode);

            // prefill 完成后检查是否需要滑窗
            if (ctx_omni->sliding_window_config.mode != "off") {
                sliding_window_enforce(ctx_omni);
            }
        }
        
        // ========== 分支2：队列为空且需要开始生成文本 ==========
        // 这个分支在以下情况触发：
        // 1. 所有嵌入数据都已处理完成（队列为空）
        // 2. 解码线程设置了 need_speek = true，表示需要开始生成文本

        if (queue.empty() && ctx_omni->need_speek){
            // 标记前缀填充完成
            prefill_done = true;
            
            // 如果使用TTS，重置speek_done标志，允许TTS线程开始工作
            if (ctx_omni->use_tts && !ctx_omni->duplex_mode) {
                ctx_omni->speek_done = false;
            }
            
            // 重置need_speek标志
            ctx_omni->need_speek = false;
            
            // 通知等待的解码线程：前缀填充已完成，可以开始生成文本了
            g_decode_cv.notify_all();
        }
    }
}

// Special token IDs to filter (from file_loader.py)
// 对齐 Python MiniCPMODuplex.streaming_generate 的过滤逻辑：
//   1. chunk_speak_token_ids = [speak_token_id] 中的 token 不加入 TTS 条件
//   2. j != 0 条件跳过循环中的第一个 token
// 注意：这些 token 大多也会被 tid >= 150000 规则过滤，显式列出便于理解和调试
