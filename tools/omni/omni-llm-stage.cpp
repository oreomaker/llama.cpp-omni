#include "omni-llm-stage.h"

#include "common/common.h"
#include "common/sampling.h"
#include "omni-impl.h"
#include "omni-log.h"
#include "omni-session-state.h"
#include "omni-sliding-window.h"
#include "omni-token-protocol.h"
#include "omni-tts-stage.h"
#include "omni.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <utility>

namespace {
struct LlmDecodeRuntime {
    int  max_tgt_len             = 0;
    int  step_size               = 10;
    int  llm_n_embd              = 0;
    int  generated_decode_tokens = 0;
    int  current_chunk_tokens    = 0;
    bool llm_finish              = false;
    bool llm_first_token_logged  = false;
};

bool omni_llm_stage_eval_tokens_with_hidden(struct omni_context *    ctx_omni,
                                            struct common_params *   params,
                                            std::vector<llama_token> tokens,
                                            int                      n_batch,
                                            int *                    n_past,
                                            float *&                 hidden_states) {
    const int n_tokens = (int) tokens.size();
    if (n_tokens == 0) {
        hidden_states = nullptr;
        return true;
    }

    kv_cache_slide_window(ctx_omni, params, n_tokens);

    const int n_embd = llama_model_n_embd(llama_get_model(ctx_omni->ctx_llama));
    hidden_states    = (float *) malloc(n_tokens * n_embd * sizeof(float));
    if (hidden_states == nullptr) {
        LOG_ERR("%s : failed to allocate memory for hidden_states\n", __func__);
        return false;
    }

    int tokens_processed = 0;
    for (int i = 0; i < n_tokens; i += n_batch) {
        int n_eval = std::min(n_tokens - i, n_batch);
        if (n_eval == 0) {
            break;
        }

        llama_set_embeddings(ctx_omni->ctx_llama, true);
        llama_batch            batch = llama_batch_get_one(tokens.data() + i, n_eval);
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        for (int j = 0; j < n_eval; ++j) {
            batch.pos[j] = *n_past + j;
        }

        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, n_tokens, n_batch,
                    *n_past);
            llama_set_embeddings(ctx_omni->ctx_llama, false);
            free(hidden_states);
            hidden_states = nullptr;
            return false;
        }

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

bool omni_llm_stage_eval_id_with_hidden(struct omni_context * ctx_omni,
                                        struct common_params * params,
                                        llama_token            id,
                                        int *                  n_past,
                                        float *&               hidden_states) {
    std::vector<llama_token> tokens = { id };
    return omni_llm_stage_eval_tokens_with_hidden(ctx_omni, params, std::move(tokens), 1, n_past, hidden_states);
}

const char * omni_llm_stage_sample_with_hidden_and_token(struct common_sampler * smpl,
                                                         struct omni_context *   ctx_omni,
                                                         struct common_params *  params,
                                                         int *                   n_past,
                                                         float *&                hidden_states,
                                                         llama_token &           token_id) {
    float * logits = llama_get_logits_ith(ctx_omni->ctx_llama, -1);

    if (ctx_omni->duplex_mode && logits != nullptr) {
        if (ctx_omni->special_token_listen >= 0) {
            const float listen_bias = (ctx_omni->listen_prob_scale - 1.0f) * 2.0f;
            logits[ctx_omni->special_token_listen] += listen_bias;
        }
        if (ctx_omni->special_token_tts_pad >= 0) {
            logits[ctx_omni->special_token_tts_pad] = -INFINITY;
        }
    }

    if (!ctx_omni->duplex_mode && ctx_omni->length_penalty != 1.0f && ctx_omni->special_token_tts_eos >= 0 &&
        logits != nullptr) {
        const float eos_logit = logits[ctx_omni->special_token_tts_eos];
        if (eos_logit > 0) {
            logits[ctx_omni->special_token_tts_eos] = eos_logit / ctx_omni->length_penalty;
        } else {
            logits[ctx_omni->special_token_tts_eos] = eos_logit * ctx_omni->length_penalty;
        }
    }

    const llama_token id = common_sampler_sample(smpl, ctx_omni->ctx_llama, -1);
    token_id             = id;
    common_sampler_accept(smpl, id, true);

    static std::string ret;
    if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx_omni->ctx_llama)), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_omni->ctx_llama, id);
    }

    omni_llm_stage_eval_id_with_hidden(ctx_omni, params, id, n_past, hidden_states);
    return ret.c_str();
}

const char * omni_llm_stage_loop_with_hidden_and_token(struct omni_context *   ctx_omni,
                                                       struct common_params *  params,
                                                       struct common_sampler * smpl,
                                                       int &                   n_past,
                                                       float *&                hidden_states,
                                                       llama_token &           token_id) {
    return omni_llm_stage_sample_with_hidden_and_token(smpl, ctx_omni, params, &n_past, hidden_states, token_id);
}

std::string omni_llm_stage_build_decode_prefix(const struct omni_context * ctx_omni) {
    if (ctx_omni->duplex_mode) {
        return "";
    }
    if (ctx_omni->use_tts) {
        return "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>";
    }
    return "<|im_end|>\n<|im_start|>assistant\n";
}

void omni_llm_stage_apply_decode_prefix(struct omni_context * ctx_omni, const std::string & prompt) {
    if (prompt.empty()) {
        print_with_timestamp("stream_decode: 双工模式，跳过 assistant prompt\n");
        return;
    }

    if (ctx_omni->use_tts) {
        print_with_timestamp("📍 [单工TTS] 添加 assistant prompt: \"%s\", n_past=%d\n", prompt.c_str(),
                             ctx_omni->session.n_past);
    }

    omni_llm_stage_eval_string(ctx_omni, ctx_omni->params, prompt.c_str(), ctx_omni->params->n_batch,
                               &ctx_omni->session.n_past, false);

    if (ctx_omni->use_tts) {
        print_with_timestamp("📍 [单工TTS] assistant prompt 完成, n_past=%d\n", ctx_omni->session.n_past);
    }
}

LlmDecodeRuntime omni_llm_stage_init_decode_runtime(struct omni_context * ctx_omni) {
    LlmDecodeRuntime runtime;
    runtime.max_tgt_len = ctx_omni->params->n_predict < 0 ? ctx_omni->params->n_ctx : ctx_omni->params->n_predict;
    runtime.llm_n_embd  = llama_model_n_embd(llama_get_model(ctx_omni->ctx_llama));
    print_with_timestamp("LLM decode: max_tgt_len = %d, n_predict = %d, n_ctx = %d\n", runtime.max_tgt_len,
                         ctx_omni->params->n_predict, ctx_omni->params->n_ctx);
    return runtime;
}

void omni_llm_stage_mark_decode_turn_end(struct omni_context * ctx_omni,
                                         OmniTokenType         token_type,
                                         bool &                is_end_of_turn) {
    if (!ctx_omni->duplex_mode) {
        return;
    }

    if (token_type == OmniTokenType::TURN_EOS || token_type == OmniTokenType::TTS_EOS ||
        token_type == OmniTokenType::EOS) {
        is_end_of_turn               = true;
        ctx_omni->turn.current_turn_ended = true;
        print_with_timestamp(
            "LLM Duplex: turn_eos detected (type=%d), set is_end_of_turn=true (not breaking, wait for chunk_eos)\n",
            (int) token_type);
    }
}

void omni_llm_stage_handle_decode_end_token(struct omni_context * ctx_omni, OmniTokenType token_type) {
    if (!ctx_omni->duplex_mode) {
        ctx_omni->gate.llm_generation_done.store(true);
        print_with_timestamp("LLM: detected end token, set llm_generation_done=true\n");
    }

    if (token_type == OmniTokenType::TURN_EOS || token_type == OmniTokenType::TTS_EOS ||
        token_type == OmniTokenType::EOS) {
        ctx_omni->turn.current_turn_ended = true;
    }

    if (token_type == OmniTokenType::LISTEN && ctx_omni->duplex_mode) {
        ctx_omni->turn.ended_with_listen = true;

        if (ctx_omni->async) {
            std::lock_guard<std::mutex> lock(ctx_omni->text_mtx);
            ctx_omni->text_queue.push_back("__IS_LISTEN__");
            ctx_omni->text_cv.notify_all();
        }
    }
}

void omni_llm_stage_strip_decode_special_tokens(std::string & response) {
    static const std::vector<std::string> end_token_strings = {
        "<|tts_eos|>", "</s>", "<|listen|>", "<|turn_eos|>", "<|chunk_eos|>", "<|chunk_tts_eos|>",
    };

    for (const auto & delimiter : end_token_strings) {
        const size_t end = response.find(delimiter);
        if (end != std::string::npos) {
            response = response.substr(0, end);
        }
    }

    size_t speak_pos = response.find("<|speak|>");
    while (speak_pos != std::string::npos) {
        response.erase(speak_pos, std::string("<|speak|>").length());
        speak_pos = response.find("<|speak|>");
    }
}

void omni_llm_stage_publish_decode_response(struct omni_context * ctx_omni, const std::string & response) {
    if (response.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->text_mtx);
    ctx_omni->text_queue.push_back(response);
    ctx_omni->text_cv.notify_all();
}

int omni_llm_stage_active_duplex_chunk_idx(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    return ctx_omni->active_duplex_chunk_idx;
}

void omni_llm_stage_dispatch_decode_chunk_to_tts(struct omni_context *                 ctx_omni,
                                                 const OmniLlmStageDecodeRequest &     request,
                                                 const OmniLlmStageDecodeChunk &       chunk,
                                                 int                                   llm_n_embd) {
    if (!ctx_omni->async || !ctx_omni->use_tts || ctx_omni->tts_thread_info == nullptr ||
        (chunk.text.empty() && !chunk.llm_finish)) {
        return;
    }

    LLMOut * llm_out        = new LLMOut();
    llm_out->text           = chunk.text;
    llm_out->n_past         = ctx_omni->session.n_past;
    llm_out->llm_finish     = chunk.llm_finish;
    llm_out->debug_dir      = request.debug_dir;
    llm_out->round_meta     = omni_session_round_meta(ctx_omni);
    llm_out->token_ids      = chunk.token_ids;
    llm_out->hidden_states  = chunk.hidden_states;
    llm_out->n_embd         = llm_n_embd;
    llm_out->is_end_of_turn = chunk.is_end_of_turn;
    llm_out->duplex_chunk_idx = omni_llm_stage_active_duplex_chunk_idx(ctx_omni);

    std::string token_ids_str;
    for (size_t i = 0; i < chunk.token_ids.size() && i < 20; ++i) {
        token_ids_str += std::to_string(chunk.token_ids[i]);
        if (i + 1 < chunk.token_ids.size() && i < 19) {
            token_ids_str += " ";
        }
    }
    if (chunk.token_ids.size() > 20) {
        token_ids_str += "...";
    }

    print_with_timestamp("LLM->TTS: text='%s', n_tokens=%zu, hidden_size=%zu, n_embd=%d, token_ids=[%s]\n",
                         chunk.text.c_str(), chunk.token_ids.size(), chunk.hidden_states.size(), llm_n_embd,
                         token_ids_str.c_str());

    std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
    ctx_omni->tts_thread_info->cv.wait(lock, [&] {
        return ctx_omni->tts_thread_info->queue.size() <
               static_cast<size_t>(ctx_omni->tts_thread_info->MAX_QUEUE_SIZE);
    });

    if (!ctx_omni->gate.speech_ready || ctx_omni->duplex_mode) {
        ctx_omni->tts_thread_info->queue.push(llm_out);
        ctx_omni->tts_thread_info->cv.notify_all();
    } else {
        delete llm_out;
    }
}

void omni_llm_stage_finish_decode_text_stream(struct omni_context * ctx_omni) {
    std::lock_guard<std::mutex> lock(ctx_omni->text_mtx);
    if (!ctx_omni->duplex_mode || !ctx_omni->turn.ended_with_listen) {
        ctx_omni->text_queue.push_back("__END_OF_TURN__");
    }

    ctx_omni->gate.text_done = true;
    ctx_omni->text_cv.notify_all();
    ctx_omni->gate.text_streaming = false;
}
}  // namespace

bool omni_llm_stage_eval_tokens(struct omni_context *    ctx_omni,
                                struct common_params *   params,
                                std::vector<llama_token> tokens,
                                int                      n_batch,
                                int *                    n_past,
                                bool                     get_emb) {
    const int n_tokens = (int) tokens.size();
    kv_cache_slide_window(ctx_omni, params, n_tokens);

    for (int i = 0; i < n_tokens; i += n_batch) {
        const int n_eval = std::min(n_tokens - i, n_batch);
        if (n_eval == 0) {
            break;
        }

        if (get_emb) {
            llama_set_embeddings(ctx_omni->ctx_llama, true);
        }

        llama_batch            batch = llama_batch_get_one(tokens.data() + i, n_eval);
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        for (int j = 0; j < n_eval; ++j) {
            batch.pos[j] = *n_past + j;
        }

        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, n_tokens, n_batch,
                    *n_past);
            if (get_emb) {
                llama_set_embeddings(ctx_omni->ctx_llama, false);
            }
            return false;
        }

        if (get_emb) {
            llama_set_embeddings(ctx_omni->ctx_llama, false);
        }
        *n_past += n_eval;
    }

    return true;
}

bool omni_llm_stage_eval_string(struct omni_context * ctx_omni,
                                struct common_params * params,
                                const char *           str,
                                int                    n_batch,
                                int *                  n_past,
                                bool                   add_bos,
                                bool                   get_emb) {
    std::string              str_buf = str;
    std::vector<llama_token> tokens  = common_tokenize(ctx_omni->ctx_llama, str_buf, add_bos, true);
    return omni_llm_stage_eval_tokens(ctx_omni, params, std::move(tokens), n_batch, n_past, get_emb);
}

bool omni_llm_stage_eval_string_with_hidden(struct omni_context * ctx_omni,
                                            struct common_params * params,
                                            const char *           str,
                                            int                    n_batch,
                                            int *                  n_past,
                                            bool                   add_bos,
                                            float *&               hidden_states) {
    std::string              str_buf = str;
    std::vector<llama_token> tokens  = common_tokenize(ctx_omni->ctx_llama, str_buf, add_bos, true);
    return omni_llm_stage_eval_tokens_with_hidden(ctx_omni, params, std::move(tokens), n_batch, n_past,
                                                  hidden_states);
}

void omni_llm_stage_finalize_decode_round(struct omni_context * ctx_omni) {
    if (ctx_omni->duplex_mode) {
        return;
    }

    const int reserved_space = 1024;
    const int n_ctx          = ctx_omni->params->n_ctx;

    if (ctx_omni->session.n_past > n_ctx - reserved_space) {
        print_with_timestamp("⚠️ Decode 结束滑窗检查: n_past=%d > n_ctx-reserved=%d，需要滑窗\n", ctx_omni->session.n_past,
                             n_ctx - reserved_space);
        kv_cache_slide_window(ctx_omni, ctx_omni->params, reserved_space);
    } else {
        print_with_timestamp("📍 Decode 结束: n_past=%d, 剩余空间=%d, 无需滑窗\n", ctx_omni->session.n_past,
                             n_ctx - ctx_omni->session.n_past);
    }

    ctx_omni->session.round_start_positions.push_back(ctx_omni->session.n_past);
    print_with_timestamp("📍 轮次 %zu 结束，记录边界于 n_past=%d\n", ctx_omni->session.round_start_positions.size(),
                         ctx_omni->session.n_past);

    const bool prefix_ok = omni_llm_stage_eval_string(ctx_omni, ctx_omni->params, "<|im_end|>\n<|im_start|>user\n",
                                                      ctx_omni->params->n_batch, &ctx_omni->session.n_past, false);
    if (!prefix_ok) {
        print_with_timestamp("⚠️ 为下一轮准备 user 前缀失败，n_past=%d\n", ctx_omni->session.n_past);
        return;
    }

    print_with_timestamp("📍 为下一轮准备: eval <|im_end|>\\n<|im_start|>user\\n, n_past=%d\n", ctx_omni->session.n_past);
}

bool omni_llm_stage_decode_run(struct omni_context *             ctx_omni,
                               const OmniLlmStageDecodeRequest & request,
                               OmniLlmStageDecodeResult *        out_result) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || ctx_omni->params == nullptr) {
        return false;
    }

    omni_llm_stage_apply_decode_prefix(ctx_omni, omni_llm_stage_build_decode_prefix(ctx_omni));

    OmniLlmStageDecodeResult result;
    LlmDecodeRuntime         runtime = omni_llm_stage_init_decode_runtime(ctx_omni);

    for (; runtime.generated_decode_tokens < runtime.max_tgt_len;) {
        if (ctx_omni->gate.break_event.load()) {
            runtime.llm_finish = true;
            result.interrupted = true;
            break;
        }

        fflush(stdout);

        OmniLlmStageDecodeChunk chunk;
        int                     valid_chunk_tokens      = 0;
        int                     total_tokens_generated  = 0;
        const int               max_chunk_tokens        =
            ctx_omni->duplex_mode ? ctx_omni->max_new_speak_tokens_per_chunk : 0;
        bool chunk_limit_reached = max_chunk_tokens > 0 && runtime.current_chunk_tokens >= max_chunk_tokens;

        while (valid_chunk_tokens < runtime.step_size && !runtime.llm_finish && !ctx_omni->gate.break_event.load() &&
               !chunk_limit_reached) {
            const char * tmp           = nullptr;
            float *      hidden_states = nullptr;
            llama_token  sampled_token = 0;

            {
                std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
                tmp = omni_llm_stage_loop_with_hidden_and_token(ctx_omni, ctx_omni->params, ctx_omni->ctx_sampler,
                                                                ctx_omni->session.n_past, hidden_states, sampled_token);
            }

            total_tokens_generated++;

            if (tmp == nullptr) {
                free(hidden_states);
                LOG_ERR("llama_loop returned nullptr!");
                break;
            }

            if (hidden_states != nullptr && omni_tts_is_valid_token(sampled_token)) {
                chunk.token_ids.push_back(sampled_token);
                chunk.hidden_states.insert(chunk.hidden_states.end(), hidden_states, hidden_states + runtime.llm_n_embd);
                valid_chunk_tokens++;
                runtime.current_chunk_tokens++;

                if (max_chunk_tokens > 0 && runtime.current_chunk_tokens >= max_chunk_tokens) {
                    chunk_limit_reached = true;
                }
            }
            free(hidden_states);

            if (!runtime.llm_first_token_logged) {
                runtime.llm_first_token_logged = true;
            }

            const OmniTokenType token_type = omni_get_token_type(ctx_omni, sampled_token);
            omni_llm_stage_mark_decode_turn_end(ctx_omni, token_type, chunk.is_end_of_turn);

            if (omni_is_end_token(ctx_omni, sampled_token)) {
                runtime.llm_finish = true;
                omni_llm_stage_handle_decode_end_token(ctx_omni, token_type);
                break;
            }

            chunk.text += std::string(tmp);
            fflush(stdout);
        }

        if (chunk_limit_reached) {
            if (ctx_omni->special_token_chunk_eos >= 0) {
                std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
                std::vector<llama_token>    chunk_eos_tokens = { ctx_omni->special_token_chunk_eos };
                omni_llm_stage_eval_tokens(ctx_omni, ctx_omni->params, std::move(chunk_eos_tokens),
                                           ctx_omni->params->n_batch, &ctx_omni->session.n_past);
            }
            runtime.llm_finish           = true;
            runtime.current_chunk_tokens = 0;
        }

        if (ctx_omni->duplex_mode && ctx_omni->special_token_unit_end >= 0) {
            std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
            std::vector<llama_token>    unit_end_tokens = { ctx_omni->special_token_unit_end };
            omni_llm_stage_eval_tokens(ctx_omni, ctx_omni->params, std::move(unit_end_tokens), ctx_omni->params->n_batch,
                                       &ctx_omni->session.n_past);
        }

        runtime.generated_decode_tokens += total_tokens_generated;
        chunk.llm_finish = runtime.llm_finish;

        omni_llm_stage_strip_decode_special_tokens(chunk.text);
        omni_llm_stage_publish_decode_response(ctx_omni, chunk.text);
        omni_llm_stage_dispatch_decode_chunk_to_tts(ctx_omni, request, chunk, runtime.llm_n_embd);

        if (runtime.llm_finish) {
            break;
        }
    }

    omni_llm_stage_finish_decode_text_stream(ctx_omni);

    result.llm_finish              = runtime.llm_finish;
    result.ended_with_listen       = ctx_omni->turn.ended_with_listen.load();
    result.generated_decode_tokens = runtime.generated_decode_tokens;
    if (out_result != nullptr) {
        *out_result = result;
    }

    return true;
}
