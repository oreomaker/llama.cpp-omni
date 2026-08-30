// LLM compute: token classification / eval / prefill / sample / projector / sliding window — compute layer: model/graph execution. No frame/queue/worker knowledge.
// ==========================================================================

#include "omni-impl.h"
#include "omni.h"
#include "omni_internal.h"
#include "common/common.h"
#include "common/sampling.h"

#include <cstring>
#include <random>
// 获取 token 类型
OmniTokenType get_token_type(struct omni_context * ctx, llama_token token) {
    if (token == ctx->special_token_speak) {
        return OmniTokenType::SPEAK;
    } else if (token == ctx->special_token_listen) {
        return OmniTokenType::LISTEN;
    } else if (token == ctx->special_token_chunk_eos) {
        return OmniTokenType::CHUNK_EOS;
    } else if (token == ctx->special_token_chunk_tts_eos) {
        return OmniTokenType::CHUNK_TTS_EOS;
    } else if (token == ctx->special_token_turn_eos) {
        return OmniTokenType::TURN_EOS;
    } else if (token == ctx->special_token_tts_eos) {
        return OmniTokenType::TTS_EOS;
    } else if (token == ctx->special_token_eos) {
        return OmniTokenType::EOS;
    }
    return OmniTokenType::NORMAL;
}

// 检查是否是会话/轮次结束 token
bool is_end_token(struct omni_context * ctx, llama_token token) {
    OmniTokenType type = get_token_type(ctx, token);

    if (ctx->duplex_mode) {
        // 双工模式:
        // - chunk_eos/chunk_tts_eos: 结束当前 stream_decode 调用，Python server 管理多轮
        // - listen: 结束当前发言段
        // - turn_eos/tts_eos/eos: 在 stream_decode 内层循环中单独处理（设 llm_finish + is_end_of_turn）
        return 
            type == OmniTokenType::LISTEN ||      // 双工模式下 <|listen|> 结束当前发言
            type == OmniTokenType::CHUNK_EOS ||   // <|chunk_eos|> 结束当前 chunk
            type == OmniTokenType::CHUNK_TTS_EOS; // <|chunk_tts_eos|> 结束当前 TTS chunk
    } else {
        // 单工流式 TTS 模式: 
        // Python (ChunkPrefillChunkGenerate): terminators=["<|tts_eos|>", "<|im_end|>", "</s>"]
        // 单工模式下只检查主要终止 token，避免误判
        return type == OmniTokenType::TTS_EOS ||      // <|tts_eos|>
               type == OmniTokenType::EOS;            // </s> 或 <|endoftext|>
    }
}

// 检查是否是 chunk 结束 token (但不是会话结束)
bool is_chunk_end_token(struct omni_context * ctx, llama_token token) {
    OmniTokenType type = get_token_type(ctx, token);
    return type == OmniTokenType::CHUNK_EOS || 
           type == OmniTokenType::CHUNK_TTS_EOS;
}

// 获取 token 类型名称（用于日志）
const char * get_token_type_name(OmniTokenType type) {
    switch (type) {
        case OmniTokenType::NORMAL:        return "NORMAL";
        case OmniTokenType::SPEAK:         return "SPEAK";
        case OmniTokenType::LISTEN:        return "LISTEN";
        case OmniTokenType::CHUNK_EOS:     return "CHUNK_EOS";
        case OmniTokenType::CHUNK_TTS_EOS: return "CHUNK_TTS_EOS";
        case OmniTokenType::TURN_EOS:      return "TURN_EOS";
        case OmniTokenType::TTS_EOS:       return "TTS_EOS";
        case OmniTokenType::EOS:           return "EOS";
        default:                           return "UNKNOWN";
    }
}

// 前向声明
void kv_cache_slide_window(struct omni_context* ctx_omni, common_params* params, int chunk_size);

//
// omni mtmd embed
//
bool omni_eval_embed(llama_context * ctx_llama, const struct omni_embed * omni_embed, int n_batch, int * n_past) {
    int n_embd  = llama_n_embd(llama_get_model(ctx_llama));

    for (int i = 0; i < omni_embed->n_pos; i += n_batch) {
        int n_eval = omni_embed->n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd     = (omni_embed->embed + i*n_embd);
        std::vector<llama_pos> pos_vec(n_eval);
        for (int j = 0; j < n_eval; j++) {
            pos_vec[j] = *n_past + j;
        }
        batch.pos = pos_vec.data();
        
        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

bool prefill_with_emb(struct omni_context * ctx_omni, common_params * params, float* embed, int n_pos, int n_batch, int*n_past) {
    kv_cache_slide_window(ctx_omni, params, n_pos);
    
    int n_embd  = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    for (int i = 0; i < n_pos; i += n_batch) {
        int n_eval = n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd     = (embed + i*n_embd);
        std::vector<llama_pos> pos_vec(n_eval);
        for (int j = 0; j < n_eval; j++) {
            pos_vec[j] = *n_past + j;
        }
        batch.pos = pos_vec.data();
        
        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

// 与 prefill_with_emb 类似，但会将每次 decode 的 hidden_state 保存并拼接到 hidden_states 中
// hidden_states 由函数内部分配空间，大小为 n_pos * n_embd * sizeof(float)，调用者负责释放
bool prefill_emb_with_hidden(struct omni_context * ctx_omni, common_params * params, float* embed, int n_pos, int n_batch, int* n_past, float *& hidden_states) {
    if (n_pos == 0) {
        hidden_states = nullptr;
        return true;
    }

    kv_cache_slide_window(ctx_omni, params, n_pos);

    int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));

    // 在函数内部分配空间
    hidden_states = (float *)malloc(n_pos * n_embd * sizeof(float));
    if (hidden_states == nullptr) {
        LOG_ERR("%s : failed to allocate memory for hidden_states\n", __func__);
        return false;
    }

    int tokens_processed = 0;

    for (int i = 0; i < n_pos; i += n_batch) {
        int n_eval = n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }

        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd     = (embed + i * n_embd);
        std::vector<llama_pos> pos_vec(n_eval);
        for (int j = 0; j < n_eval; j++) {
            pos_vec[j] = *n_past + j;
        }
        batch.pos = pos_vec.data();

        // 启用 embeddings 输出
        llama_set_embeddings(ctx_omni->ctx_llama, true);

        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            llama_set_embeddings(ctx_omni->ctx_llama, false);
            free(hidden_states);
            hidden_states = nullptr;
            return false;
        }

        // 获取当前 batch 的 embeddings 并复制到 hidden_states
        float * emb = llama_get_embeddings(ctx_omni->ctx_llama);
        if (emb != nullptr) {
            memcpy(hidden_states + tokens_processed * n_embd, emb, n_eval * n_embd * sizeof(float));
        }

        llama_set_embeddings(ctx_omni->ctx_llama, false);

        *n_past += n_eval;
        tokens_processed += n_eval;
    }
    return true;
}

void kv_cache_slide_window(struct omni_context* ctx_omni, common_params* params, int chunk_size) {
    const int n_ctx = params->n_ctx;

    // ==================== 双工模式：按轮次边界智能清理 ====================
    if (ctx_omni->duplex_mode) {
        const int duplex_trigger = std::max(ctx_omni->n_keep, n_ctx - 2048);

        if (ctx_omni->n_past + chunk_size < duplex_trigger) {
            return;
        }

        // defer：生成中 / TTS 忙 / 正在 SPEAK 时不滑窗，除非快要撞上 n_ctx 硬限制
        bool generating    = ctx_omni->text_streaming;
        bool tts_busy      = !ctx_omni->speek_done;
        bool mid_speak     = !ctx_omni->slide_last_was_listen.load();
        bool force_slide   = (ctx_omni->n_past + chunk_size >= n_ctx - 512);

        if (!force_slide && (generating || tts_busy || mid_speak)) {
            return;
        }
        if (force_slide) {
            print_with_timestamp("⚠️ slide FORCE: n_past=%d approaching n_ctx=%d\n",
                                 ctx_omni->n_past, n_ctx);
        }

        int old_n_past = ctx_omni->n_past;
        const auto& rounds = ctx_omni->round_start_positions;
        const int total_rounds = (int)rounds.size();

        print_with_timestamp("⚠️ slide TRIGGERED: n_past=%d, chunk=%d, trigger=%d, n_ctx=%d, n_keep=%d, rounds=%d\n",
                             ctx_omni->n_past, chunk_size, duplex_trigger, n_ctx,
                             ctx_omni->n_keep, total_rounds);

        if (total_rounds < 2) {
            // 轮次边界不足（一轮超长对话）：不做全量清空，尽量保留最近一段 KV。
            // 原行为是丢掉 [n_keep, n_past)，模型立刻失忆 → 连续长对话下触发后模型会
            // 陷入长时间 LISTEN / 彻底失控（case 3、case 4）。
            // 改为按 tail window 保留最近 target_keep_tokens，通过左移 KV 对齐位置。
            const int target_keep_tokens = std::max(n_ctx / 4, 2048);
            int keep_from = ctx_omni->n_past - target_keep_tokens;
            keep_from = std::max(keep_from, ctx_omni->n_keep);
            int n_discard = keep_from - ctx_omni->n_keep;

            if (n_discard <= 0) {
                print_with_timestamp("⚠️ slide skipped (rounds<2): nothing to discard (n_past=%d, target_keep=%d)\n",
                                     ctx_omni->n_past, target_keep_tokens);
            } else {
                llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
                if (mem) {
                    llama_memory_seq_rm(mem, 0, ctx_omni->n_keep, keep_from);
                    llama_memory_seq_add(mem, 0, keep_from, ctx_omni->n_past, -n_discard);
                }
                ctx_omni->n_past -= n_discard;
                // 没有可信的 round boundary 了，清空以避免后续按错位的 boundary 切割
                ctx_omni->round_start_positions.clear();
                if (ctx_omni->ctx_tts_llama) {
                    llama_memory_t tts_mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                    if (tts_mem) llama_memory_seq_rm(tts_mem, 0, 0, -1);
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                }
                print_with_timestamp("⚠️ slide DONE: rounds<2 tail-keep: n_past %d→%d, freed %d (kept last %d), TTS KV cleared\n",
                                     old_n_past, ctx_omni->n_past, n_discard, target_keep_tokens);
            }
        } else {
            // 保留约 n_ctx/4 的上下文，按 round boundary 对齐截断
            const int target_keep_tokens = std::max(n_ctx / 4, 2048);
            const int min_keep_from_pos  = ctx_omni->n_past - target_keep_tokens;

            int keep_from = rounds[total_rounds - 2];
            for (int i = 0; i <= total_rounds - 2; i++) {
                if (rounds[i] >= min_keep_from_pos) {
                    keep_from = rounds[i];
                    break;
                }
            }
            keep_from = std::max(keep_from, ctx_omni->n_keep);

            int n_discard = keep_from - ctx_omni->n_keep;

            if (n_discard <= 0) {
                print_with_timestamp("⚠️ slide: nothing to discard (keep_from=%d <= n_keep=%d)\n",
                                     keep_from, ctx_omni->n_keep);
            } else {
                llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
                if (mem) {
                    llama_memory_seq_rm(mem, 0, ctx_omni->n_keep, keep_from);
                    llama_memory_seq_add(mem, 0, keep_from, ctx_omni->n_past, -n_discard);
                }
                ctx_omni->n_past -= n_discard;

                std::vector<int> new_rounds;
                for (int i = 0; i < total_rounds; i++) {
                    if (rounds[i] > keep_from) {
                        new_rounds.push_back(rounds[i] - n_discard);
                    }
                }
                ctx_omni->round_start_positions = new_rounds;

                if (ctx_omni->ctx_tts_llama) {
                    llama_memory_t tts_mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                    if (tts_mem) llama_memory_seq_rm(tts_mem, 0, 0, -1);
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                }

                print_with_timestamp("⚠️ slide DONE: n_past %d→%d, freed %d, boundaries_kept=%d, TTS KV cleared\n",
                                     old_n_past, ctx_omni->n_past, n_discard, (int)new_rounds.size());
            }
        }
        return;
    }

    // ==================== 非双工模式：原有滑窗逻辑 ====================
    if (ctx_omni->n_past + chunk_size < n_ctx) {
        return;
    }
    
    print_with_timestamp("⚠️ KV Cache 滑动窗口触发: n_past=%d, chunk_size=%d, n_ctx=%d, n_keep=%d, 轮次数=%zu\n",
                         ctx_omni->n_past, chunk_size, n_ctx, ctx_omni->n_keep, 
                         ctx_omni->round_start_positions.size());
    
    int n_discard = 0;
    int delete_end_pos = ctx_omni->n_keep;
    
    if (ctx_omni->max_preserved_context > 0 && ctx_omni->round_start_positions.size() >= 1) {
        const auto& rounds = ctx_omni->round_start_positions;
        int cumulative_length = 0;
        int keep_from_round = rounds.size();
        int total_rounds = rounds.size();
        
        for (int i = total_rounds - 1; i >= 0; --i) {
            int round_start = (i == 0) ? ctx_omni->n_keep : rounds[i - 1];
            int round_end = rounds[i];
            int round_length = round_end - round_start;
            
            if (cumulative_length + round_length > ctx_omni->max_preserved_context) {
                break;
            }
            
            cumulative_length += round_length;
            keep_from_round = i;
        }
        
        if (keep_from_round >= total_rounds) {
            keep_from_round = total_rounds - 1;
        }
        
        int delete_start = ctx_omni->n_keep;
        delete_end_pos = (keep_from_round == 0) ? ctx_omni->n_keep : rounds[keep_from_round - 1];
        
        if (delete_end_pos > delete_start) {
            n_discard = delete_end_pos - delete_start;
            
            std::vector<int> new_rounds;
            for (int i = keep_from_round; i < total_rounds; ++i) {
                new_rounds.push_back(rounds[i] - n_discard);
            }
            ctx_omni->round_start_positions = new_rounds;
        } else {
            n_discard = 0;
        }
    }
    
    if (n_discard == 0) {
        const int n_left = ctx_omni->n_past - ctx_omni->n_keep;
        n_discard = n_left / 2;
        delete_end_pos = ctx_omni->n_keep + n_discard;
        
        if (n_left <= 0 || n_discard <= 0) {
            return;
        }
        
        std::vector<int> new_rounds;
        for (int pos : ctx_omni->round_start_positions) {
            if (pos > delete_end_pos) {
                new_rounds.push_back(pos - n_discard);
            }
        }
        ctx_omni->round_start_positions = new_rounds;
    }
    
    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
    if (mem) {
        bool rm_ok = llama_memory_seq_rm(mem, 0, ctx_omni->n_keep, delete_end_pos);
        (void)rm_ok;
        llama_memory_seq_add(mem, 0, delete_end_pos, ctx_omni->n_past, -n_discard);
    }
    
    int old_n_past = ctx_omni->n_past;
    ctx_omni->n_past -= n_discard;
    print_with_timestamp("⚠️ KV Cache 滑动窗口完成: n_past %d→%d\n", old_n_past, ctx_omni->n_past);
}

bool eval_tokens(struct omni_context* ctx_omni, common_params* params, std::vector<llama_token> tokens, int n_batch, int * n_past, bool get_emb) {
    int N = (int) tokens.size();
    kv_cache_slide_window(ctx_omni, params, N);

    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (n_eval == 0)
            break;
        if (get_emb) {
            llama_set_embeddings(ctx_omni->ctx_llama, true);
        }
        // llama_batch_get_one 返回的 batch.pos 可能是 nullptr，需要手动设置
        llama_batch batch = llama_batch_get_one(&tokens[i], n_eval);
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = *n_past + j;  // 从当前 n_past 位置开始
        }
        
        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        if (get_emb) {
            llama_set_embeddings(ctx_omni->ctx_llama, false);
        }
        *n_past += n_eval;
    }
    return true;
}

// 与 eval_tokens 类似，但会将每次 decode 的 hidden_state 保存并拼接到 hidden_states 中
// hidden_states 由函数内部分配空间，大小为 N * n_embd * sizeof(float)，调用者负责释放
bool eval_tokens_with_hidden(struct omni_context* ctx_omni, common_params* params, std::vector<llama_token> tokens, int n_batch, int * n_past, float *& hidden_states) {
    int N = (int) tokens.size();
    if (N == 0) {
        hidden_states = nullptr;
        return true;
    }

    kv_cache_slide_window(ctx_omni, params, N);

    const int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));

    // 在函数内部分配空间
    hidden_states = (float *)malloc(N * n_embd * sizeof(float));
    if (hidden_states == nullptr) {
        LOG_ERR("%s : failed to allocate memory for hidden_states\n", __func__);
        return false;
    }

    int tokens_processed = 0;

    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (n_eval == 0)
            break;

        // 启用 embeddings 输出
        llama_set_embeddings(ctx_omni->ctx_llama, true);
        // llama_batch_get_one 返回的 batch.pos 可能是 nullptr，需要手动设置
        llama_batch batch = llama_batch_get_one(&tokens[i], n_eval);
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = *n_past + j;  // 从当前 n_past 位置开始
        }

        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            llama_set_embeddings(ctx_omni->ctx_llama, false);
            free(hidden_states);
            hidden_states = nullptr;
            return false;
        }

        // 获取当前 batch 的 embeddings 并复制到 hidden_states
        float * emb = llama_get_embeddings(ctx_omni->ctx_llama);
        if (emb != nullptr) {
            // 将当前 batch 的 embeddings 复制到 hidden_states 的对应位置
            memcpy(hidden_states + tokens_processed * n_embd, emb, n_eval * n_embd * sizeof(float));
        }

        llama_set_embeddings(ctx_omni->ctx_llama, false);

        *n_past += n_eval;
        tokens_processed += n_eval;
    }
    return true;
}

static bool eval_id(struct omni_context * ctx_omni, common_params* params, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_omni, params, tokens, 1, n_past);
}

static bool eval_id_with_hidden(struct omni_context * ctx_omni, common_params* params, int id, int * n_past, float *& hidden_states) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens_with_hidden(ctx_omni, params, tokens, 1, n_past, hidden_states);
}

bool eval_string(struct omni_context * ctx_omni, common_params* params, const char* str, int n_batch, int * n_past, bool add_bos, bool get_emb) {
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_omni->ctx_llama, str2, add_bos, true);
    return eval_tokens(ctx_omni, params, embd_inp, n_batch, n_past, get_emb);
}

bool eval_string_with_hidden(struct omni_context * ctx_omni, common_params* params, const char* str, int n_batch, int * n_past, bool add_bos, float *& hidden_states) {
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_omni->ctx_llama, str2, add_bos, true);
    return eval_tokens_with_hidden(ctx_omni, params, embd_inp, n_batch, n_past, hidden_states);
}

const char * sample(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past) {
    const llama_token id = common_sampler_sample(smpl, ctx_omni->ctx_llama, -1);
    common_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx_omni->ctx_llama)), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_omni->ctx_llama, id);
    }
    eval_id(ctx_omni, params, id, n_past);
    return ret.c_str();
}

static const char * sample_with_hidden(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past, float *& hidden_states) {
    const llama_token id = common_sampler_sample(smpl, ctx_omni->ctx_llama, -1);
    common_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx_omni->ctx_llama)), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_omni->ctx_llama, id);
    }
    eval_id_with_hidden(ctx_omni, params, id, n_past, hidden_states);
    return ret.c_str();
}

const char * llama_loop(struct omni_context * ctx_omni, common_params *params, struct common_sampler * smpl, int &n_past) {
    const char * tmp = sample(smpl, ctx_omni, params, &n_past);
    return tmp;
}

// 修改sample_with_hidden来返回token ID（通过引用参数）
// 双工模式：支持 listen_prob_scale 参数，增加 <|listen|> 的采样概率
// 双工模式：支持 forbidden_token_ids，禁止采样 <|tts_pad|> 等 token
const char * sample_with_hidden_and_token(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past, float *& hidden_states, llama_token & token_id) {
    float * logits = llama_get_logits_ith(ctx_omni->ctx_llama, -1);
    
    // 双工模式：在采样前调整 logits
    if (ctx_omni->duplex_mode) {
        if (logits != nullptr) {
            // 1. 调整 <|listen|> 的 logit（listen_prob_scale）
            // listen_prob_scale > 1.0 会增加 <|listen|> 的概率，让模型更倾向于先听
            if (ctx_omni->special_token_listen >= 0) {
                // 使用 listen_prob_scale 调整 <|listen|> 的 logit
                // 默认值 1.0 不改变，> 1.0 增加 listen 概率
                // 这里我们使用加法而不是乘法，因为 logit 可能是负数
                // 添加一个偏置值来增加 listen 的概率
                // listen_prob_bias = log(listen_prob_scale) ≈ (listen_prob_scale - 1.0) for small values
                float listen_bias = (ctx_omni->listen_prob_scale - 1.0f) * 2.0f;  // 放大效果
                logits[ctx_omni->special_token_listen] += listen_bias;
            }
            
            // 2. 禁止采样 <|tts_pad|> token
            // Python: self.forbidden_token_ids = [self.tts_pad_id] + list(bad_token_ids)
            //         logits[:, self.forbidden_token_ids] = float("-inf")
            // <|tts_pad|> 是填充 token，模型不应该主动生成它
            // 如果不禁止，模型可能生成 <|speak|> → <|tts_pad|> → <|chunk_eos|>，导致无有效输出
            if (ctx_omni->special_token_tts_pad >= 0) {
                logits[ctx_omni->special_token_tts_pad] = -INFINITY;
            }

            // 3. Duplex 模式下对 <|turn_eos|> 应用长度惩罚，让 Python 透传的配置真正生效
            if (ctx_omni->length_penalty != 1.0f && ctx_omni->special_token_turn_eos >= 0) {
                float eos_logit = logits[ctx_omni->special_token_turn_eos];
                if (eos_logit > 0) {
                    logits[ctx_omni->special_token_turn_eos] = eos_logit / ctx_omni->length_penalty;
                } else {
                    logits[ctx_omni->special_token_turn_eos] = eos_logit * ctx_omni->length_penalty;
                }
            }
        }
    }
    
    // Length Penalty：调整 EOS token 的 logit 值（单工模式）
    // length_penalty > 1.0 会降低 EOS 概率，让模型生成更长的输出
    if (!ctx_omni->duplex_mode && ctx_omni->length_penalty != 1.0f && ctx_omni->special_token_tts_eos >= 0) {
        if (logits != nullptr) {
            float eos_logit = logits[ctx_omni->special_token_tts_eos];
            if (eos_logit > 0) {
                // logit > 0 时，除以 length_penalty 来降低概率
                logits[ctx_omni->special_token_tts_eos] = eos_logit / ctx_omni->length_penalty;
            } else {
                // logit <= 0 时，乘以 length_penalty 来降低概率
                logits[ctx_omni->special_token_tts_eos] = eos_logit * ctx_omni->length_penalty;
            }
        }
    }
    
    const llama_token id = common_sampler_sample(smpl, ctx_omni->ctx_llama, -1);
    token_id = id;  // 保存token ID
    common_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx_omni->ctx_llama)), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_omni->ctx_llama, id);
    }
    eval_id_with_hidden(ctx_omni, params, id, n_past, hidden_states);
    return ret.c_str();
}

static const char * llama_loop_with_hidden(struct omni_context * ctx_omni, common_params *params, struct common_sampler * smpl, int &n_past, float *& hidden_states) {
    llama_token dummy_token;
    const char * tmp = sample_with_hidden_and_token(smpl, ctx_omni, params, &n_past, hidden_states, dummy_token);
    return tmp;
}

// 新增：返回token ID的版本
const char * llama_loop_with_hidden_and_token(struct omni_context * ctx_omni, common_params *params, struct common_sampler * smpl, int &n_past, float *& hidden_states, llama_token & token_id) {
    const char * tmp = sample_with_hidden_and_token(smpl, ctx_omni, params, &n_past, hidden_states, token_id);
    return tmp;
}

//
// TTS specific helper functions
//

// Helper function to get RNG from sampler chain for multinomial sampling
// Uses common_sampler_get_rng() from common/sampling.cpp
// This ensures audio_bos sampling uses the same RNG as non-audio_bos sampling
std::mt19937* get_sampler_rng(struct common_sampler * smpl) {
    void* rng_ptr = common_sampler_get_rng(smpl);
    return static_cast<std::mt19937*>(rng_ptr);
}

// ==============================================================================
// Projector Semantic 实现 (精度验证版本)
// 使用 ggml 后端进行计算，支持 CUDA 加速
// forward(x): relu(linear1(x)) -> linear2
// ==============================================================================
bool projector_init(projector_model & model, const std::string & fname, bool use_cuda) {
    
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };
    
    struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), params);
    if (!ctx_gguf) {
        LOG_ERR("Projector: failed to open '%s'\n", fname.c_str());
        return false;
    }
    
#ifdef GGML_USE_CUDA
    if (use_cuda) {
        model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
        if (!model.backend) {
            model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        }
    } else {
        model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
#else
    (void)use_cuda;
    model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
#endif

    if (!model.backend) {
        LOG_ERR("Projector: failed to init backend\n");
        gguf_free(ctx_gguf);
        return false;
    }
    
    model.buf_type = ggml_backend_get_default_buffer_type(model.backend);
    
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    
    size_t ctx_size = ggml_tensor_overhead() * n_tensors;
    struct ggml_init_params ctx_params = {
        /*.mem_size   = */ ctx_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    model.ctx_w = ggml_init(ctx_params);
    
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        enum ggml_type type = gguf_get_tensor_type(ctx_gguf, i);
        struct ggml_tensor * tensor = nullptr;
        
        if (strcmp(name, "linear1.weight") == 0) {
            tensor = ggml_new_tensor_2d(model.ctx_w, type, 4096, 768);
            model.layer.linear1_weight = tensor;
            model.hparams.in_dim = 4096;
            model.hparams.out_dim = 768;
        } else if (strcmp(name, "linear1.bias") == 0) {
            tensor = ggml_new_tensor_1d(model.ctx_w, type, 768);
            model.layer.linear1_bias = tensor;
        } else if (strcmp(name, "linear2.weight") == 0) {
            tensor = ggml_new_tensor_2d(model.ctx_w, type, 768, 768);
            model.layer.linear2_weight = tensor;
        } else if (strcmp(name, "linear2.bias") == 0) {
            tensor = ggml_new_tensor_1d(model.ctx_w, type, 768);
            model.layer.linear2_bias = tensor;
        } else {
            continue;
        }
        
        if (tensor) {
            ggml_set_name(tensor, name);
        }
    }
    
    model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    
    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        LOG_ERR("Projector: failed to open file for reading\n");
        return false;
    }
    
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model.ctx_w, name);
        if (!tensor) continue;
        
        size_t offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
        fseek(f, offset, SEEK_SET);
        
        size_t tensor_size = ggml_nbytes(tensor);
        void * data = malloc(tensor_size);
        if (fread(data, 1, tensor_size, f) != tensor_size) {
            LOG_ERR("Projector: failed to read tensor %s\n", name);
            free(data);
            fclose(f);
            return false;
        }
        
        ggml_backend_tensor_set(tensor, data, 0, tensor_size);
        free(data);
    }
    
    fclose(f);
    gguf_free(ctx_gguf);
    
    model.initialized = true;
    return true;
}

void projector_free(projector_model & model) {
    if (model.ctx_w) ggml_free(model.ctx_w);
    if (model.buf_w) ggml_backend_buffer_free(model.buf_w);
    if (model.backend) ggml_backend_free(model.backend);
    model.ctx_w = nullptr;
    model.buf_w = nullptr;
    model.backend = nullptr;
    model.initialized = false;
}

static struct ggml_cgraph * projector_build_graph(projector_model & model, struct ggml_context * ctx, struct ggml_tensor * input) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    
    // linear1 + relu
    struct ggml_tensor * hidden = ggml_mul_mat(ctx, model.layer.linear1_weight, input);
    hidden = ggml_add(ctx, hidden, model.layer.linear1_bias);
    hidden = ggml_relu(ctx, hidden);
    
    // linear2
    struct ggml_tensor * output = ggml_mul_mat(ctx, model.layer.linear2_weight, hidden);
    output = ggml_add(ctx, output, model.layer.linear2_bias);
    
    ggml_build_forward_expand(gf, output);
    return gf;
}

std::vector<float> projector_forward(projector_model & model, const float * input_data, int n_tokens) {
    const int in_dim = model.hparams.in_dim;
    const int out_dim = model.hparams.out_dim;
    
    // 验证参数
    if (n_tokens <= 0 || n_tokens > 10000) {
        LOG_ERR("projector_forward: invalid n_tokens=%d\n", n_tokens);
        return {};
    }
    if (in_dim <= 0 || in_dim > 10000 || out_dim <= 0 || out_dim > 10000) {
        LOG_ERR("projector_forward: invalid dimensions in_dim=%d, out_dim=%d\n", in_dim, out_dim);
        return {};
    }
    
    size_t ctx_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   = */ ctx_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, n_tokens);
    ggml_set_name(input, "input");
    ggml_set_input(input);
    
    struct ggml_cgraph * gf = projector_build_graph(model, ctx, input);
    
    ggml_backend_buffer_t buf_compute = ggml_backend_alloc_ctx_tensors(ctx, model.backend);
    if (!buf_compute) {
        LOG_ERR("Projector: failed to allocate compute buffer\n");
        ggml_free(ctx);
        return {};
    }
    
    ggml_backend_tensor_set(input, input_data, 0, n_tokens * in_dim * sizeof(float));
    
    enum ggml_status status = ggml_backend_graph_compute(model.backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("Projector: graph compute failed with status %d\n", (int)status);
        ggml_backend_buffer_free(buf_compute);
        ggml_free(ctx);
        return {};
    }
    
    struct ggml_tensor * output = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
    std::vector<float> result(n_tokens * out_dim);
    ggml_backend_tensor_get(output, result.data(), 0, n_tokens * out_dim * sizeof(float));
    
    ggml_backend_buffer_free(buf_compute);
    ggml_free(ctx);
    
    return result;
}
// ==============================================================================

// Load TTS weights from GGUF file
bool is_audio_token(llama_token token, int audio_bos_token_id, int num_audio_tokens) {
    // Audio tokens are typically in range [audio_bos_token_id, audio_bos_token_id + num_audio_tokens)
    // Check if token is in the audio token range
    return (token >= audio_bos_token_id && token < audio_bos_token_id + num_audio_tokens);
}

// Simple UTF-8 incomplete byte checker
// Returns the number of incomplete bytes at the end of the string
// Returns 0 if the string ends with a complete UTF-8 sequence
size_t findIncompleteUtf8(const std::string& str) {
    if (str.empty()) return 0;
    
    size_t len = str.length();
    
    // Check from the end backwards to find incomplete UTF-8 sequences
    size_t pos = len;
    int expected_continuation_bytes = 0;
    
    while (pos > 0) {
        unsigned char c = (unsigned char)str[pos - 1];
        
        if ((c & 0x80) == 0) {
            // ASCII character (0xxxxxxx), complete sequence
            break;
        } else if ((c & 0xC0) == 0x80) {
            // Continuation byte (10xxxxxx), part of a multi-byte sequence
            expected_continuation_bytes++;
            pos--;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence start (110xxxxx)
            // Should have exactly 1 continuation byte after it
            if (expected_continuation_bytes == 1) {
                // Complete 2-byte sequence
                break;
            } else {
                // Incomplete: missing continuation byte(s)
                return len - pos + 1;  // Return incomplete bytes including the start byte
            }
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence start (1110xxxx)
            // Should have exactly 2 continuation bytes after it
            if (expected_continuation_bytes == 2) {
                // Complete 3-byte sequence
                break;
            } else {
                // Incomplete: missing continuation byte(s)
                return len - pos + (3 - expected_continuation_bytes);
            }
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence start (11110xxx)
            // Should have exactly 3 continuation bytes after it
            if (expected_continuation_bytes == 3) {
                // Complete 4-byte sequence
                break;
            } else {
                // Incomplete: missing continuation byte(s)
                return len - pos + (4 - expected_continuation_bytes);
            }
        } else {
            // Invalid UTF-8 byte pattern (should not happen in valid UTF-8)
            // Treat as complete to avoid breaking valid text
            break;
        }
    }
    
    // If we've checked all bytes and still have expected continuation bytes,
    // the sequence is incomplete
    if (pos == 0 && expected_continuation_bytes > 0) {
        return len;
    }
    
    return 0;  // String ends with complete UTF-8 sequence
}

// ==================== 滑动窗口实现 (#39) ====================
// 基于 Python sliding_utils.py 和 stream_decoder.py

/**
 * 重置滑动窗口状态
 */
void sliding_window_reset(struct omni_context * ctx_omni) {
    if (!ctx_omni) return;
    
    int old_unit_count = ctx_omni->unit_history.size();
    
    ctx_omni->unit_history.clear();
    ctx_omni->next_unit_id = 0;
    ctx_omni->pending_unit_id = -1;
    ctx_omni->pending_unit_start_cache_len = 0;
    ctx_omni->system_preserve_length = 0;
    ctx_omni->position_offset = 0;
    ctx_omni->current_turn_id = 0;

    // 统计信息
    ctx_omni->sliding_event_count = 0;
    ctx_omni->total_dropped_tokens = 0;
    ctx_omni->total_dropped_units = 0;
    ctx_omni->total_dropped_turns = 0;
    ctx_omni->total_unit_fallbacks = 0;
    
    if (old_unit_count > 0) {
        print_with_timestamp("[SW] reset: cleared %d units, all sliding window state reset\n", old_unit_count);
    }
}

/**
 * 获取当前 KV cache 长度
 */
int get_cache_length(struct omni_context * ctx_omni) {
    if (!ctx_omni || !ctx_omni->ctx_llama) return 0;
    return ctx_omni->n_past;
}

/**
 * 注册 unit 开始
 * 在每个 unit（音频/视频/omni chunk）处理开始时调用
 * @return 返回分配的 unit_id
 */
int sliding_window_register_unit_start(struct omni_context * ctx_omni) {
    if (!ctx_omni) return -1;
    
    ctx_omni->pending_unit_id = ctx_omni->next_unit_id;
    ctx_omni->pending_unit_start_cache_len = get_cache_length(ctx_omni);
    
    print_with_timestamp("[SW] unit_start: pending_unit_id=%d, cache_len=%d, preserve=%d, units=%zu\n",
                        ctx_omni->pending_unit_id,
                        ctx_omni->pending_unit_start_cache_len,
                        ctx_omni->system_preserve_length,
                        ctx_omni->unit_history.size());
    
    return ctx_omni->pending_unit_id;
}

/**
 * 注册 unit 结束
 * 在每个 unit 处理完成后调用，记录该 unit 的信息
 */
void sliding_window_register_unit_end(struct omni_context * ctx_omni, 
                                      const std::string & input_type,
                                      const std::vector<llama_token> & generated_tokens,
                                      bool is_listen) {
    if (!ctx_omni) return;
    
    if (ctx_omni->pending_unit_id < 0) {
        print_with_timestamp("[SW] WARNING: register_unit_end called without register_unit_start\n");
        return;
    }
    
    int current_cache_len = get_cache_length(ctx_omni);
    int unit_len = current_cache_len - ctx_omni->pending_unit_start_cache_len;
    
    if (unit_len > 0) {
        UnitEntry entry;
        entry.unit_id = ctx_omni->pending_unit_id;
        entry.length = unit_len;
        entry.type = input_type;
        entry.generated_tokens = generated_tokens;
        entry.is_listen = is_listen;
        // 归属到当前正在构建的 turn；turn 边界处由 stream_decode 负责 current_turn_id++
        entry.turn_id = ctx_omni->current_turn_id;

        ctx_omni->unit_history.push_back(entry);

        print_with_timestamp("[SW] unit_end: unit_id=%d type=%s len=%d gen_tokens=%zu is_listen=%d turn_id=%d | cache=%d preserve=%d total_units=%zu\n",
                            entry.unit_id, entry.type.c_str(), entry.length,
                            entry.generated_tokens.size(), entry.is_listen, entry.turn_id,
                            current_cache_len, ctx_omni->system_preserve_length,
                            ctx_omni->unit_history.size());
    } else {
        print_with_timestamp("[SW] WARNING: unit_end: unit_id=%d has zero length (start=%d, current=%d), not recorded\n",
                            ctx_omni->pending_unit_id, ctx_omni->pending_unit_start_cache_len, current_cache_len);
    }
    
    ctx_omni->pending_unit_id = -1;
    ctx_omni->pending_unit_start_cache_len = 0;
    ctx_omni->next_unit_id++;
}

/**
 * 注册 system prompt
 * 在 system prompt prefill 完成后调用，记录保护长度
 */
void sliding_window_register_system_prompt(struct omni_context * ctx_omni) {
    if (!ctx_omni) return;
    
    ctx_omni->system_preserve_length = get_cache_length(ctx_omni);
    print_with_timestamp("[SW] system_prompt registered: preserve_length=%d (will be protected from sliding)\n",
                        ctx_omni->system_preserve_length);
}

/**
 * 从 KV cache 中删除指定数量的 tokens
 * 删除位于 [preserve, preserve + length) 区间的 tokens
 *
 * 实现要点：
 *   1. llama_memory_seq_rm 只是把指定区间的 KV 标记为空，
 *      并不会移动后面 token 的 position；
 *   2. 因此必须紧接着调用 llama_memory_seq_add(... , -length)，
 *      把 [preserve + length, cache_len_before) 范围内剩余 token 的 position
 *      整体向前平移 length，使 KV 中的 position 重新连续；
 *   3. 否则下一次 llama_decode 会报
 *      "inconsistent sequence positions: X = old_pmax, Y = n_past, requires Y = X + 1"
 *      这与 kv_cache_slide_window 里采用的策略保持一致。
 */
bool sliding_window_drop_tokens_from_cache(struct omni_context * ctx_omni, int length) {
    if (!ctx_omni || !ctx_omni->ctx_llama || length <= 0) {
        print_with_timestamp("[SW] drop_tokens: invalid params (length=%d)\n", length);
        return false;
    }
    
    int cache_len_before = get_cache_length(ctx_omni);
    int preserve = ctx_omni->system_preserve_length;
    
    if (cache_len_before <= preserve) {
        print_with_timestamp("[SW] drop_tokens: cache_len=%d <= preserve=%d, nothing to drop\n",
                            cache_len_before, preserve);
        return false;
    }
    
    int available = cache_len_before - preserve;
    if (available < length) {
        print_with_timestamp("[SW] drop_tokens: cannot drop %d tokens, only %d available (cache=%d, preserve=%d)\n",
                            length, available, cache_len_before, preserve);
        return false;
    }
    
    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
    if (!mem) {
        print_with_timestamp("[SW] drop_tokens: failed to get llama memory\n");
        return false;
    }
    
    // 1) 删除 [preserve, preserve + length) 区间的 KV
    bool success = llama_memory_seq_rm(mem, 0, preserve, preserve + length);
    
    if (success) {
        // 2) 把后段 [preserve + length, cache_len_before) 的 position
        //    整体往前平移 length，保证 KV 中 position 连续，下一次 decode 不会触发
        //    "inconsistent sequence positions" 错误。
        //    与 kv_cache_slide_window 的处理方式一致。
        if (cache_len_before > preserve + length) {
            llama_memory_seq_add(mem, 0, preserve + length, cache_len_before, -length);
        }

        ctx_omni->n_past = cache_len_before - length;

        // position_offset 字段已不再用于位置修正，仅保留累计值用于诊断/兼容旧日志
        ctx_omni->position_offset += length;

        print_with_timestamp("[SW] drop_tokens: SUCCESS, dropped %d tokens from [%d, %d), shifted [%d, %d) by -%d, cache %d -> %d\n",
                            length, preserve, preserve + length,
                            preserve + length, cache_len_before, length,
                            cache_len_before, ctx_omni->n_past);
    } else {
        print_with_timestamp("[SW] drop_tokens: FAILED to drop %d tokens\n", length);
    }
    
    return success;
}

/**
 * 删除指定的 unit
 */
bool sliding_window_drop_unit(struct omni_context * ctx_omni, int unit_id) {
    if (!ctx_omni) return false;
    
    // 查找 unit
    auto it = std::find_if(ctx_omni->unit_history.begin(), ctx_omni->unit_history.end(),
                          [unit_id](const UnitEntry& e) { return e.unit_id == unit_id; });
    
    if (it == ctx_omni->unit_history.end()) {
        print_with_timestamp("[SW] drop_unit: unit_id=%d not found\n", unit_id);
        return false;
    }
    
    int total_len = it->length;
    if (total_len <= 0) {
        print_with_timestamp("[SW] drop_unit: unit_id=%d has zero length, removing from history\n", unit_id);
        ctx_omni->unit_history.erase(it);
        return false;
    }
    
    int cache_before = get_cache_length(ctx_omni);
    if (!sliding_window_drop_tokens_from_cache(ctx_omni, total_len)) {
        print_with_timestamp("[SW] drop_unit: failed to drop %d tokens for unit_id=%d\n", total_len, unit_id);
        return false;
    }
    
    int cache_after = get_cache_length(ctx_omni);
    print_with_timestamp("[SW] 🗑️ DROPPED unit_id=%d type=%s len=%d gen_tokens=%zu | cache %d -> %d, offset=%d\n",
                        it->unit_id, it->type.c_str(), it->length, it->generated_tokens.size(),
                        cache_before, cache_after, ctx_omni->position_offset);
    
    ctx_omni->unit_history.erase(it);
    return true;
}

/**
 * 删除最早的一个已完成 turn（按 turn 粒度一次性丢）
 *
 * 行为：
 *   1. 在 unit_history 里找到第一条非 system unit 的 turn_id（记为 earliest）。
 *   2. 如果 earliest >= current_turn_id，说明 unit_history 里只剩下当前还没收尾的 turn，
 *      这种情况下函数返回 false，交给上层退化成按 unit 删。
 *   3. 否则收集"前缀"中所有非 system 且 turn_id == earliest 的 unit，把它们的 length
 *      累加起来一次性从 KV cache 中丢掉，并从 unit_history 里擦除对应条目。
 *
 * 设计依据：
 *   - UnitEntry 是按时间顺序 push 进 unit_history 的，turn_id 单调不降，
 *     所以最小 turn_id 的 unit 一定集中在 unit_history 的前缀段。
 *   - 用一次 sliding_window_drop_tokens_from_cache(total_len) 来移动 KV，
 *     比 per-unit 循环调用更省：只做一次 llama_memory_seq_rm + 一次 seq_add。
 *
 * @return true 成功丢掉一个完整 turn；false 表示当前没有"已完成且非 system"的 turn 可丢
 */
bool sliding_window_drop_next_turn(struct omni_context * ctx_omni) {
    if (!ctx_omni || ctx_omni->unit_history.empty()) {
        return false;
    }

    // 1) 找到最早一条非 system unit 的 turn_id
    int earliest_turn = -1;
    for (const auto & e : ctx_omni->unit_history) {
        if (e.type == "system") continue;
        earliest_turn = e.turn_id;
        break;
    }
    if (earliest_turn < 0) {
        // 只剩 system 条目，没什么可丢的
        return false;
    }

    // 2) 判断这个 turn 是否已经完结
    //    - current_turn_id 指向"正在构建中的 turn"，它 ≥ earliest_turn
    //    - 只有当 earliest_turn < current_turn_id 时，earliest_turn 才真正已经收尾，
    //      这时按 turn 一次性丢掉才安全（不会把活跃轮次从中间砍半）
    //    - 否则（只剩当前 turn）返回 false，让 enforce 退化到按 unit 丢
    if (earliest_turn >= ctx_omni->current_turn_id) {
        print_with_timestamp("[SW] drop_next_turn: only current turn (turn_id=%d) left in history, "
                             "fall back to unit-level drop\n", earliest_turn);
        return false;
    }

    // 3) 收集前缀里所有 turn_id == earliest_turn 的非 system unit
    //    （遇到 turn_id 更大的 unit 就停；system 单独跳过但不打断）
    int total_len = 0;
    int n_units   = 0;
    for (const auto & e : ctx_omni->unit_history) {
        if (e.type == "system") continue;
        if (e.turn_id != earliest_turn) break;
        total_len += e.length;
        n_units++;
    }

    if (total_len <= 0 || n_units == 0) {
        // 理论上不会走到这里（turn 内若全是零长度 unit 也没必要丢），保底
        print_with_timestamp("[SW] drop_next_turn: turn_id=%d has no droppable length (n_units=%d)\n",
                             earliest_turn, n_units);
        return false;
    }

    int cache_before = get_cache_length(ctx_omni);
    if (!sliding_window_drop_tokens_from_cache(ctx_omni, total_len)) {
        print_with_timestamp("[SW] drop_next_turn: drop_tokens_from_cache failed "
                             "(turn_id=%d, total_len=%d)\n", earliest_turn, total_len);
        return false;
    }

    // 4) 从 unit_history 里擦除刚刚丢掉的 n_units 条非 system unit（保留可能混入的 system 条目）
    int removed = 0;
    for (auto it = ctx_omni->unit_history.begin();
         it != ctx_omni->unit_history.end() && removed < n_units; ) {
        if (it->type == "system") { ++it; continue; }
        if (it->turn_id != earliest_turn) break;  // 防御性：不该跨 turn
        it = ctx_omni->unit_history.erase(it);
        removed++;
    }

    int cache_after = get_cache_length(ctx_omni);
    print_with_timestamp("[SW] 🗑️ DROPPED turn_id=%d: %d units, %d tokens | cache %d -> %d, "
                         "remaining_units=%zu, current_turn_id=%d\n",
                         earliest_turn, n_units, total_len,
                         cache_before, cache_after,
                         ctx_omni->unit_history.size(), ctx_omni->current_turn_id);
    return true;
}

/**
 * 删除最早的一个非 system unit
 */
bool sliding_window_drop_next_unit(struct omni_context * ctx_omni) {
    if (!ctx_omni) return false;
    
    for (const auto& entry : ctx_omni->unit_history) {
        // 跳过 system 类型
        if (entry.type == "system") {
            print_with_timestamp("[SW] drop_next_unit: skipping system unit_id=%d\n", entry.unit_id);
            continue;
        }
        
        print_with_timestamp("[SW] drop_next_unit: attempting to drop unit_id=%d\n", entry.unit_id);
        if (sliding_window_drop_unit(ctx_omni, entry.unit_id)) {
            return true;
        }
    }
    
    print_with_timestamp("[SW] drop_next_unit: no droppable unit found in %zu units\n", ctx_omni->unit_history.size());
    return false;
}

/**
 * 执行滑动窗口策略
 * 当 cache 长度超过高水位线时，循环移除最早的 unit，直到降到低水位线以下
 * 
 * @return true 如果执行了滑窗操作，false 如果没有需要滑窗
 */
bool sliding_window_enforce(struct omni_context * ctx_omni) {
    if (!ctx_omni) return false;
    
    const auto& cfg = ctx_omni->sliding_window_config;
    
    // 检查是否启用滑窗
    if (cfg.mode == "off") {
        return false;
    }
    
    int cache_len_before = get_cache_length(ctx_omni);
    
    // 检查是否超过高水位线
    if (cache_len_before <= cfg.high_water_tokens) {
        return false;  // 未超过高水位线，不触发
    }
    
    // 只有新增的 "turn" 模式才走 turn-first / unit-fallback 的新路径，
    //   其它任何已有模式（"basic"/"context"/未来扩展）继续按 unit 粒度丢，保证老行为一字不动。
    const bool turn_mode = (cfg.mode == "turn");

    if (turn_mode) {
        print_with_timestamp("[SW] ⚡ SLIDING TRIGGERED: mode=turn cache=%d > high_water=%d, target=low_water=%d, "
                            "current_turn_id=%d, units=%zu\n",
                            cache_len_before, cfg.high_water_tokens, cfg.low_water_tokens,
                            ctx_omni->current_turn_id, ctx_omni->unit_history.size());
    } else {
        // 老日志格式保持原样，避免已有分析脚本/对照测试被扰动
        print_with_timestamp("[SW] ⚡ SLIDING TRIGGERED: cache=%d > high_water=%d, target=low_water=%d\n",
                            cache_len_before, cfg.high_water_tokens, cfg.low_water_tokens);
    }

    // 命名约定（与 omni_context 字段对齐，避免 turn / 非 turn 两套含义混淆）：
    //   dropped_turns      : 本次 enforce 里"整 turn 丢"成功的次数
    //   dropped_units      : 本次 enforce 里被移除的 UnitEntry 总条目数
    //                        （turn 模式 = 整 turn 丢带走的 + fallback 丢的；非 turn 模式 = 按 unit 丢的）
    //   unit_fallbacks     : 仅 turn 模式 —— 整 turn 丢不动退化到按 unit 丢的次数
    int dropped_turns   = 0;
    int dropped_units   = 0;
    int unit_fallbacks  = 0;
    int cache_len = cache_len_before;

    while (cache_len > cfg.low_water_tokens) {
        bool dropped_one = false;

        if (turn_mode) {
            // turn 级滑窗 + unit 级兜底
            //   - 主策略：整 turn 丢（把"用户输入 + 模型回复"当一块记忆），语义连贯；
            //   - 兜底：当 unit_history 里只剩下当前还没收尾的 turn（drop_next_turn 返回 false）
            //     时退化到按 unit 丢，避免"一轮超长、整 turn 又丢不动"把长对话卡死。
            int units_before = (int)ctx_omni->unit_history.size();
            if (sliding_window_drop_next_turn(ctx_omni)) {
                dropped_turns++;
                dropped_units += units_before - (int)ctx_omni->unit_history.size();
                dropped_one = true;
            } else if (sliding_window_drop_next_unit(ctx_omni)) {
                unit_fallbacks++;   // fallback 计数（turn 模式专属）
                dropped_units++;    // 同时计入总量
                dropped_one = true;
            }
        } else {
            // 原有行为：按 unit 丢（basic / context / 其它未列模式）
            if (sliding_window_drop_next_unit(ctx_omni)) {
                dropped_units++;
                dropped_one = true;
            }
        }

        if (!dropped_one) {
            print_with_timestamp("[SW] enforce_window: no more %s to drop, stopping (cache=%d, units=%zu)\n",
                                turn_mode ? "turns/units" : "units",
                                cache_len, ctx_omni->unit_history.size());
            break;
        }
        cache_len = get_cache_length(ctx_omni);
    }

    // 兼容旧返回语义：是否发生过任意"丢"动作
    int dropped_actions = dropped_turns + (turn_mode ? unit_fallbacks : dropped_units);
    if (dropped_units > 0 || dropped_turns > 0) {
        // 更新统计
        ctx_omni->sliding_event_count++;
        ctx_omni->total_dropped_tokens    += cache_len_before - cache_len;
        ctx_omni->total_dropped_units     += dropped_units;     // 两种模式语义统一：被移除的 UnitEntry 总数
        ctx_omni->total_dropped_turns     += dropped_turns;     // 仅 turn 模式 > 0
        ctx_omni->total_unit_fallbacks    += unit_fallbacks;    // 仅 turn 模式 > 0

        // 一致性检查
        int expected = ctx_omni->system_preserve_length;
        for (const auto& u : ctx_omni->unit_history) {
            expected += u.length;
        }
        bool is_consistent = (expected == cache_len);

        if (turn_mode) {
            print_with_timestamp("[SW] ✅ SLIDING DONE: cache %d -> %d, "
                                "dropped %d turns + %d unit-fallbacks (total %d unit entries removed), "
                                "remaining %zu units | consistency: expected=%d actual=%d %s\n",
                                cache_len_before, cache_len,
                                dropped_turns, unit_fallbacks, dropped_units,
                                ctx_omni->unit_history.size(),
                                expected, cache_len, is_consistent ? "✓" : "✗ MISMATCH!");
        } else {
            // 保持老日志格式不变，方便已有分析脚本 / 对照测试
            print_with_timestamp("[SW] ✅ SLIDING DONE: cache %d -> %d, dropped %d units, remaining %zu units | "
                                "consistency: expected=%d actual=%d %s\n",
                                cache_len_before, cache_len, dropped_units,
                                ctx_omni->unit_history.size(),
                                expected, cache_len, is_consistent ? "✓" : "✗ MISMATCH!");
        }

        if (!is_consistent) {
            print_with_timestamp("[SW] ❌ CONSISTENCY ERROR! preserve=%d + sum(units)=%d != cache=%d, offset=%d\n",
                                ctx_omni->system_preserve_length,
                                expected - ctx_omni->system_preserve_length,
                                cache_len, ctx_omni->position_offset);
        }
    }

    return dropped_actions > 0;
}

//
// omni main
//
