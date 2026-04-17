#include "omni-llm-stage.h"

#include "common/common.h"
#include "common/sampling.h"
#include "omni-impl.h"
#include "omni-log.h"
#include "omni-session-state.h"
#include "omni-sliding-window.h"
#include "omni-token-protocol.h"
#include "omni-turn-coordinator.h"
#include "omni-tts-stage.h"
#include "omni.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <utility>

namespace {
constexpr int kLegacyDecodeChunkStepSize = 10;

double omni_llm_stage_timing_elapsed_ms(const std::chrono::high_resolution_clock::time_point & start,
                                        const std::chrono::high_resolution_clock::time_point & end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void omni_llm_stage_note_prefill_timing(struct omni_context * ctx_omni, int chunk_idx, double ms) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.llm_prefill_ms              = timing.llm_prefill_ms < 0.0 ? ms : timing.llm_prefill_ms + ms;
}

void omni_llm_stage_note_decode_timing(struct omni_context * ctx_omni, int chunk_idx, double ms) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.llm_decode_ms               = timing.llm_decode_ms < 0.0 ? ms : timing.llm_decode_ms + ms;
}

int omni_llm_stage_active_duplex_chunk_idx(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    return ctx_omni->active_duplex_chunk_idx;
}

std::vector<omni_embeds *> omni_llm_stage_drain_prefill_queue(struct LLMThreadInfo * llm_thread_info) {
    std::vector<omni_embeds *> llm_embeds;
    if (llm_thread_info == nullptr) {
        return llm_embeds;
    }

    auto & queue = llm_thread_info->queue;
    while (!queue.empty()) {
        llm_embeds.push_back(queue.front());
        queue.pop();
    }
    return llm_embeds;
}

bool omni_llm_stage_has_pending_prefill(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || ctx_omni->llm_thread_info == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
    return !ctx_omni->llm_thread_info->queue.empty();
}

omni_embeds * omni_llm_stage_peek_prefill_head(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || ctx_omni->llm_thread_info == nullptr) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
    return ctx_omni->llm_thread_info->queue.empty() ? nullptr : ctx_omni->llm_thread_info->queue.front();
}

omni_embeds * omni_llm_stage_pop_prefill_head(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || ctx_omni->llm_thread_info == nullptr) {
        return nullptr;
    }

    omni_embeds * result = nullptr;
    {
        std::lock_guard<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
        if (!ctx_omni->llm_thread_info->queue.empty()) {
            result = ctx_omni->llm_thread_info->queue.front();
            ctx_omni->llm_thread_info->queue.pop();
        }
    }

    if (result != nullptr) {
        ctx_omni->llm_thread_info->cv.notify_all();
    }
    return result;
}

bool omni_llm_stage_micro_batch_supported(const struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || ctx_omni->params == nullptr) {
        return false;
    }

    return ctx_omni->params->kv_unified && llama_n_seq_max(ctx_omni->ctx_llama) >= 2;
}

int omni_llm_stage_estimate_speculative_decode_tail(const struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return 0;
    }

    const int max_tgt_len =
        ctx_omni->params->n_predict < 0 ? ctx_omni->params->n_ctx : ctx_omni->params->n_predict;
    int remaining_decode_budget = std::max(0, max_tgt_len - ctx_omni->turn.generated_decode_tokens);

    if (ctx_omni->duplex_mode && ctx_omni->max_new_speak_tokens_per_chunk > 0) {
        remaining_decode_budget =
            std::max(0, ctx_omni->max_new_speak_tokens_per_chunk - ctx_omni->turn.current_chunk_tokens);
    }

    const int remaining_legacy_chunks =
        remaining_decode_budget > 0 ? (remaining_decode_budget + kLegacyDecodeChunkStepSize - 1) /
                                          kLegacyDecodeChunkStepSize :
                                      1;
    const int estimated_closure_tokens = ctx_omni->duplex_mode ? remaining_legacy_chunks + 2 : 0;
    return remaining_decode_budget + estimated_closure_tokens;
}

void omni_llm_stage_reset_staged_state(struct omni_context * ctx_omni,
                                       OmniLlmSchedulerState & state,
                                       bool                   clear_staging_seq) {
    if (clear_staging_seq && ctx_omni != nullptr && ctx_omni->ctx_llama != nullptr) {
        if (llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama)) {
            llama_memory_seq_rm(mem, state.staging_seq, 0, -1);
        }
    }

    state.staged_ready     = false;
    state.staged_chunk_idx = -1;
    state.branch_n_past    = 0;
    state.staged_begin_pos = 0;
    state.staged_n_past    = 0;
    state.spec_decode_tail = 0;
}

bool omni_llm_stage_close_decode_preemptively(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || ctx_omni->params == nullptr) {
        return false;
    }

    if (ctx_omni->special_token_chunk_eos < 0) {
        LOG_ERR("%s: missing <|chunk_eos|> token, cannot preemptively close decode\n", __func__);
        return false;
    }

    std::vector<llama_token> close_tokens = { ctx_omni->special_token_chunk_eos };
    if (ctx_omni->duplex_mode) {
        if (ctx_omni->special_token_unit_end < 0) {
            LOG_ERR("%s: missing </unit> token in duplex mode, cannot preemptively close decode\n", __func__);
            return false;
        }
        close_tokens.push_back(ctx_omni->special_token_unit_end);
    }

    bool close_ok = false;
    {
        std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
        close_ok = omni_llm_stage_eval_tokens(ctx_omni, ctx_omni->params, std::move(close_tokens),
                                              ctx_omni->params->n_batch, &ctx_omni->session.n_past);
    }

    if (!close_ok) {
        LOG_ERR("%s: failed to append preemptive closure tokens\n", __func__);
        return false;
    }

    ctx_omni->turn.current_chunk_tokens = 0;
    print_with_timestamp("LLM scheduler: preemptively closed current decode with pending prefill\n");
    return true;
}

OmniLlmStageDecodeRequest omni_llm_stage_get_pipeline_request(struct omni_context * ctx_omni) {
    OmniLlmStageDecodeRequest request;
    if (ctx_omni == nullptr) {
        return request;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->pipeline_request_mtx);
    request.debug_dir = ctx_omni->pipeline_debug_dir;
    request.round_idx = ctx_omni->pipeline_round_idx;
    return request;
}

void omni_llm_stage_post_pipeline_result(struct omni_context * ctx_omni,
                                         bool                  decode_ok,
                                         bool                  ended_with_listen) {
    if (ctx_omni == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(ctx_omni->pipeline_result_mtx);
        // Preserve FIFO pairing between stream_decode() callers and worker completions.
        ctx_omni->pipeline_result_queue.push({ ended_with_listen, decode_ok });
    }
    ctx_omni->pipeline_result_cv.notify_one();
}

void omni_llm_stage_set_active_duplex_chunk_idx(struct omni_context * ctx_omni, int chunk_idx) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    ctx_omni->active_duplex_chunk_idx = chunk_idx;
    ctx_omni->duplex_chunk_timings[chunk_idx];
}

int omni_llm_stage_decode_max_tgt_len(const struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || ctx_omni->params == nullptr) {
        return 0;
    }

    return ctx_omni->params->n_predict < 0 ? ctx_omni->params->n_ctx : ctx_omni->params->n_predict;
}

void omni_llm_stage_prepare_worker_decode(struct omni_context *             ctx_omni,
                                          const OmniLlmStageDecodeRequest & request) {
    if (request.round_idx >= 0 && !ctx_omni->duplex_mode && ctx_omni->session.current_round.round_idx != request.round_idx) {
        print_with_timestamp("📍 [轮次同步] 调用方指定 round_idx=%d，当前 simplex_round_idx=%d，强制同步\n",
                             request.round_idx, ctx_omni->session.current_round.round_idx);
        omni_session_set_round_index(ctx_omni, request.round_idx);
    } else {
        omni_session_sync_round_meta(ctx_omni);
    }

    ctx_omni->stream_decode_start_time = std::chrono::high_resolution_clock::now();
    print_with_timestamp("📍 stream_decode 开始: n_past=%d, n_keep=%d, n_ctx=%d, duplex_mode=%d\n",
                         ctx_omni->session.n_past, ctx_omni->session.prompt.n_keep, ctx_omni->params->n_ctx,
                         ctx_omni->duplex_mode);

    ctx_omni->turn.current_turn_ended = false;
    ctx_omni->turn.decode_prefix_applied = false;
    ctx_omni->turn.generated_decode_tokens = 0;
    ctx_omni->turn.current_chunk_tokens = 0;
    if (!ctx_omni->duplex_mode) {
        ctx_omni->gate.llm_generation_done.store(false);
    }
    ctx_omni->turn.ended_with_listen.store(false);

    if (ctx_omni->duplex_mode && ctx_omni->gate.break_event.load()) {
        ctx_omni->gate.break_event.store(false);
        print_with_timestamp("📍 stream_decode: reset break_event from true to false\n");
    }

    {
        std::lock_guard<std::mutex> lock(ctx_omni->text_mtx);
        ctx_omni->text_queue.clear();
        ctx_omni->gate.text_done = false;
        ctx_omni->gate.text_streaming = true;
    }

    if (ctx_omni->use_tts) {
        ctx_omni->gate.speech_ready = false;
    }
}

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

void omni_llm_stage_dispatch_decode_chunk_to_tts(struct omni_context *             ctx_omni,
                                                 const OmniLlmStageDecodeRequest & request,
                                                 const OmniLlmDecodeSliceResult &  slice,
                                                 int                               llm_n_embd) {
    if (!ctx_omni->async || !ctx_omni->use_tts || ctx_omni->tts_thread_info == nullptr ||
        (slice.text.empty() && !slice.llm_finish)) {
        return;
    }

    LLMOut * llm_out        = new LLMOut();
    llm_out->text           = slice.text;
    llm_out->n_past         = ctx_omni->session.n_past;
    llm_out->llm_finish     = slice.llm_finish;
    llm_out->debug_dir      = request.debug_dir;
    llm_out->round_meta     = omni_session_round_meta(ctx_omni);
    llm_out->token_ids      = slice.token_ids;
    llm_out->hidden_states  = slice.hidden_states;
    llm_out->n_embd         = llm_n_embd;
    llm_out->is_end_of_turn = slice.is_end_of_turn;
    llm_out->duplex_chunk_idx = omni_llm_stage_active_duplex_chunk_idx(ctx_omni);

    std::string token_ids_str;
    for (size_t i = 0; i < slice.token_ids.size() && i < 20; ++i) {
        token_ids_str += std::to_string(slice.token_ids[i]);
        if (i + 1 < slice.token_ids.size() && i < 19) {
            token_ids_str += " ";
        }
    }
    if (slice.token_ids.size() > 20) {
        token_ids_str += "...";
    }

    print_with_timestamp("LLM->TTS: text='%s', n_tokens=%zu, hidden_size=%zu, n_embd=%d, token_ids=[%s]\n",
                         slice.text.c_str(), slice.token_ids.size(), slice.hidden_states.size(), llm_n_embd,
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

void omni_llm_stage_publish_decode_slice(struct omni_context *             ctx_omni,
                                         const OmniLlmStageDecodeRequest & request,
                                         OmniLlmDecodeSliceResult *        slice) {
    if (ctx_omni == nullptr || slice == nullptr) {
        return;
    }

    omni_llm_stage_strip_decode_special_tokens(slice->text);
    omni_llm_stage_publish_decode_response(ctx_omni, slice->text);
    omni_llm_stage_dispatch_decode_chunk_to_tts(ctx_omni, request, *slice,
                                                llama_model_n_embd(llama_get_model(ctx_omni->ctx_llama)));
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

static bool omni_llm_stage_eval_embeds_seq(struct omni_context *  ctx_omni,
                                           struct common_params * params,
                                           const float *          embed,
                                           int                    n_pos,
                                           int                    n_batch,
                                           int *                  n_past,
                                           llama_seq_id           seq_id) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || params == nullptr || n_past == nullptr) {
        return false;
    }

    if (n_pos <= 0 || embed == nullptr) {
        return true;
    }

    if (seq_id == 0) {
        kv_cache_slide_window(ctx_omni, params, n_pos);
    }

    const int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    for (int i = 0; i < n_pos; i += n_batch) {
        const int n_eval = std::min(n_pos - i, n_batch);
        if (n_eval == 0) {
            break;
        }

        llama_batch batch = llama_batch_init(n_eval, n_embd, 1);
        std::memcpy(batch.embd, embed + i * n_embd, sizeof(float) * n_eval * n_embd);
        for (int j = 0; j < n_eval; ++j) {
            batch.pos[j]      = *n_past + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = seq_id;
            batch.logits[j]   = (j + 1 == n_eval);
        }
        batch.n_tokens = n_eval;

        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval embeds. token %d/%d (batch size %d, seq_id %d, n_past %d)\n", __func__, i,
                    n_pos, n_batch, seq_id, *n_past);
            llama_batch_free(batch);
            return false;
        }

        llama_batch_free(batch);
        *n_past += n_eval;
    }

    return true;
}

bool omni_llm_stage_eval_tokens_seq(struct omni_context *    ctx_omni,
                                    struct common_params *   params,
                                    std::vector<llama_token> tokens,
                                    int                      n_batch,
                                    int *                    n_past,
                                    llama_seq_id             seq_id,
                                    bool                     get_emb) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || params == nullptr || n_past == nullptr) {
        return false;
    }

    const int n_tokens = (int) tokens.size();
    if (n_tokens == 0) {
        return true;
    }

    if (seq_id == 0) {
        kv_cache_slide_window(ctx_omni, params, n_tokens);
    }

    for (int i = 0; i < n_tokens; i += n_batch) {
        const int n_eval = std::min(n_tokens - i, n_batch);
        if (n_eval == 0) {
            break;
        }

        if (get_emb) {
            llama_set_embeddings(ctx_omni->ctx_llama, true);
        }

        llama_batch batch = llama_batch_init(n_eval, 0, 1);
        for (int j = 0; j < n_eval; ++j) {
            common_batch_add(batch, tokens[i + j], *n_past + j, { seq_id }, get_emb || (j + 1 == n_eval));
        }

        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, seq_id %d, n_past %d)\n", __func__, i, n_tokens,
                    n_batch, seq_id, *n_past);
            llama_batch_free(batch);
            if (get_emb) {
                llama_set_embeddings(ctx_omni->ctx_llama, false);
            }
            return false;
        }

        llama_batch_free(batch);
        if (get_emb) {
            llama_set_embeddings(ctx_omni->ctx_llama, false);
        }
        *n_past += n_eval;
    }

    return true;
}

bool omni_llm_stage_eval_tokens(struct omni_context *    ctx_omni,
                                struct common_params *   params,
                                std::vector<llama_token> tokens,
                                int                      n_batch,
                                int *                    n_past,
                                bool                     get_emb) {
    return omni_llm_stage_eval_tokens_seq(ctx_omni, params, std::move(tokens), n_batch, n_past, /*seq_id=*/0,
                                          get_emb);
}

bool omni_llm_stage_eval_string_seq(struct omni_context * ctx_omni,
                                    struct common_params * params,
                                    const char *           str,
                                    int                    n_batch,
                                    int *                  n_past,
                                    llama_seq_id           seq_id,
                                    bool                   add_bos,
                                    bool                   get_emb) {
    std::string              str_buf = str;
    std::vector<llama_token> tokens  = common_tokenize(ctx_omni->ctx_llama, str_buf, add_bos, true);
    return omni_llm_stage_eval_tokens_seq(ctx_omni, params, std::move(tokens), n_batch, n_past, seq_id, get_emb);
}

bool omni_llm_stage_eval_string(struct omni_context * ctx_omni,
                                struct common_params * params,
                                const char *           str,
                                int                    n_batch,
                                int *                  n_past,
                                bool                   add_bos,
                                bool                   get_emb) {
    return omni_llm_stage_eval_string_seq(ctx_omni, params, str, n_batch, n_past, /*seq_id=*/0, add_bos, get_emb);
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

bool omni_llm_stage_prefill_apply_seq(struct omni_context *      ctx_omni,
                                      struct common_params *     params,
                                      const struct omni_embeds & embeds,
                                      llama_seq_id               seq_id,
                                      int *                      n_past) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || params == nullptr || n_past == nullptr) {
        return false;
    }

    const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    const bool track_sliding_window = seq_id == 0 && ctx_omni->session.sliding_window_config.mode != "off";

    if (track_sliding_window) {
        sliding_window_register_unit_start(ctx_omni);
    }

    if (!embeds.vision_embed.empty()) {
        const int  n_chunks         = (int) embeds.vision_embed.size();
        const int  tokens_per_chunk = (int) embeds.vision_embed[0].size() / hidden_size;
        const int  n_audio_tokens   = embeds.audio_embed.size() / hidden_size;
        const bool has_audio        = n_audio_tokens > 0;
        const bool has_slices       = n_chunks > 1;

        if (ctx_omni->duplex_mode) {
            if (!omni_llm_stage_eval_string_seq(ctx_omni, params, "<unit><image>", params->n_batch, n_past, seq_id,
                                                false)) {
                goto fail;
            }
        } else {
            if (!omni_llm_stage_eval_string_seq(ctx_omni, params, "<image>", params->n_batch, n_past, seq_id, false)) {
                goto fail;
            }
        }

        if (!omni_llm_stage_eval_embeds_seq(ctx_omni, params, embeds.vision_embed[0].data(), tokens_per_chunk,
                                            params->n_batch, n_past, seq_id)) {
            goto fail;
        }
        if (!omni_llm_stage_eval_string_seq(ctx_omni, params, "</image>", params->n_batch, n_past, seq_id, false)) {
            goto fail;
        }

        if (has_slices) {
            for (int i = 1; i < n_chunks; ++i) {
                if (!omni_llm_stage_eval_string_seq(ctx_omni, params, "<slice>", params->n_batch, n_past, seq_id,
                                                    false)) {
                    goto fail;
                }
                if (!omni_llm_stage_eval_embeds_seq(ctx_omni, params, embeds.vision_embed[i].data(), tokens_per_chunk,
                                                    params->n_batch, n_past, seq_id)) {
                    goto fail;
                }
                if (!omni_llm_stage_eval_string_seq(ctx_omni, params, "</slice>", params->n_batch, n_past, seq_id,
                                                    false)) {
                    goto fail;
                }
            }
            if (!omni_llm_stage_eval_string_seq(ctx_omni, params, "\n", params->n_batch, n_past, seq_id, false)) {
                goto fail;
            }
        }

        print_with_timestamp("Omni模式: %d vision chunks (%d tokens each), %d audio tokens, has_slices=%d\n", n_chunks,
                             tokens_per_chunk, n_audio_tokens, has_slices);

        if (has_audio) {
            if (!ctx_omni->duplex_mode) {
                if (!omni_llm_stage_eval_string_seq(ctx_omni, params, "<|audio_start|>", params->n_batch, n_past,
                                                    seq_id, false)) {
                    goto fail;
                }
            }
            if (!omni_llm_stage_eval_embeds_seq(ctx_omni, params, embeds.audio_embed.data(), n_audio_tokens,
                                                params->n_batch, n_past, seq_id)) {
                goto fail;
            }
            if (!ctx_omni->duplex_mode) {
                if (!omni_llm_stage_eval_string_seq(ctx_omni, params, "<|audio_end|>", params->n_batch, n_past,
                                                    seq_id, false)) {
                    goto fail;
                }
            }
        }
    } else {
        const int n_audio_tokens = embeds.audio_embed.size() / hidden_size;
        print_with_timestamp("用户语音: %d audio tokens\n", n_audio_tokens);

        if (ctx_omni->duplex_mode) {
            if (!omni_llm_stage_eval_string_seq(ctx_omni, params, "<unit>", params->n_batch, n_past, seq_id, false)) {
                goto fail;
            }
        } else {
            if (!omni_llm_stage_eval_string_seq(ctx_omni, params, "<|audio_start|>", params->n_batch, n_past, seq_id,
                                                false)) {
                goto fail;
            }
        }

        if (n_audio_tokens > 0) {
            if (!omni_llm_stage_eval_embeds_seq(ctx_omni, params, embeds.audio_embed.data(), n_audio_tokens,
                                                params->n_batch, n_past, seq_id)) {
                goto fail;
            }
        }

        if (!ctx_omni->duplex_mode) {
            if (!omni_llm_stage_eval_string_seq(ctx_omni, params, "<|audio_end|>", params->n_batch, n_past, seq_id,
                                                false)) {
                goto fail;
            }
        }
    }

    if (track_sliding_window) {
        const std::string input_type = embeds.vision_embed.empty() ? "audio" : "omni";
        sliding_window_register_unit_end(ctx_omni, input_type, {}, false);
    }

    return true;

fail:
    if (track_sliding_window) {
        ctx_omni->session.pending_unit_id              = -1;
        ctx_omni->session.pending_unit_start_cache_len = 0;
    }
    return false;
}

bool omni_llm_stage_prefill_apply(struct omni_context *      ctx_omni,
                                  struct common_params *     params,
                                  const struct omni_embeds & embeds) {
    return omni_llm_stage_prefill_apply_seq(ctx_omni, params, embeds, /*seq_id=*/0, &ctx_omni->session.n_past);
}

void omni_llm_stage_finalize_prefill(struct omni_context * ctx_omni) {
    if (ctx_omni->session.sliding_window_config.mode != "off") {
        sliding_window_enforce(ctx_omni);
    }
}

static bool omni_llm_stage_stage_prefill_speculatively(struct omni_context *    ctx_omni,
                                                       struct common_params *   params,
                                                       OmniLlmSchedulerState &  state,
                                                       struct omni_embeds **    out_staged_prefill) {
    if (out_staged_prefill == nullptr) {
        return false;
    }
    *out_staged_prefill = nullptr;

    if (!omni_llm_stage_micro_batch_supported(ctx_omni)) {
        return false;
    }

    omni_embeds * pending = omni_llm_stage_peek_prefill_head(ctx_omni);
    if (pending == nullptr || pending->encode_failed) {
        return false;
    }

    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
    if (mem == nullptr) {
        LOG_ERR("%s: failed to get llama memory for staging\n", __func__);
        return false;
    }

    const auto prefill_start = std::chrono::high_resolution_clock::now();
    state.branch_n_past      = ctx_omni->session.n_past;
    state.spec_decode_tail   = std::max(0, omni_llm_stage_estimate_speculative_decode_tail(ctx_omni));
    // llama.cpp requires each sequence to be fed with consecutive positions. We therefore stage
    // the speculative prefill right after the committed prefix, and only shift it forward after
    // the active decode tail length becomes known.
    state.staged_begin_pos   = state.branch_n_past;
    state.staged_n_past      = state.staged_begin_pos;
    state.staged_chunk_idx   = pending->index;

    llama_memory_seq_rm(mem, state.staging_seq, 0, -1);
    llama_memory_seq_cp(mem, state.active_seq, state.staging_seq, 0, state.branch_n_past);

    if (!omni_llm_stage_prefill_apply_seq(ctx_omni, params, *pending, state.staging_seq, &state.staged_n_past)) {
        LOG_WRN("%s: speculative staging failed for chunk %d, fallback to serial prefill\n", __func__, pending->index);
        omni_llm_stage_reset_staged_state(ctx_omni, state, /*clear_staging_seq=*/true);
        return false;
    }

    omni_embeds * popped = omni_llm_stage_pop_prefill_head(ctx_omni);
    if (popped == nullptr || popped != pending) {
        LOG_ERR("%s: speculative staging desynced with prefill queue head\n", __func__);
        if (popped != nullptr && popped != pending) {
            delete popped;
        }
        omni_llm_stage_reset_staged_state(ctx_omni, state, /*clear_staging_seq=*/true);
        return false;
    }

    *out_staged_prefill = popped;
    state.staged_ready  = true;
    omni_llm_stage_note_prefill_timing(
        ctx_omni, pending->index,
        omni_llm_stage_timing_elapsed_ms(prefill_start, std::chrono::high_resolution_clock::now()));
    print_with_timestamp(
        "LLM scheduler: staged chunk %d on seq=%d (branch=%d, staged_begin=%d, staged_n_past=%d, predicted_tail=%d)\n",
        pending->index, state.staging_seq, state.branch_n_past, state.staged_begin_pos, state.staged_n_past,
        state.spec_decode_tail);
    return true;
}

static bool omni_llm_stage_promote_staged(struct omni_context * ctx_omni, OmniLlmSchedulerState & state) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || !state.staged_ready) {
        return false;
    }

    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
    if (mem == nullptr) {
        LOG_ERR("%s: failed to get llama memory for promote\n", __func__);
        return false;
    }

    if (state.staged_n_past < state.staged_begin_pos || state.branch_n_past < 0) {
        LOG_ERR("%s: invalid staged state (branch=%d, begin=%d, n_past=%d)\n", __func__, state.branch_n_past,
                state.staged_begin_pos, state.staged_n_past);
        return false;
    }

    const int actual_decode_tail = ctx_omni->session.n_past - state.branch_n_past;
    const int delta              = actual_decode_tail;

    if (delta != 0) {
        llama_memory_seq_add(mem, state.staging_seq, state.staged_begin_pos, state.staged_n_past, delta);
    }

    if (!llama_memory_seq_rm(mem, state.active_seq, 0, -1)) {
        LOG_ERR("%s: failed to clear active sequence %d before promote\n", __func__, state.active_seq);
        return false;
    }

    llama_memory_seq_cp(mem, state.staging_seq, state.active_seq, 0, -1);
    if (!llama_memory_seq_rm(mem, state.staging_seq, 0, -1)) {
        LOG_WRN("%s: failed to clear staging sequence %d after promote, continuing with promoted active seq\n",
                __func__, state.staging_seq);
    }

    ctx_omni->session.n_past = state.staged_n_past + delta;
    print_with_timestamp(
        "LLM scheduler: promoted staged chunk %d (actual_tail=%d, predicted_tail=%d, shift=%d, n_past=%d)\n",
        state.staged_chunk_idx, actual_decode_tail, state.spec_decode_tail, delta, ctx_omni->session.n_past);
    omni_llm_stage_reset_staged_state(ctx_omni, state, /*clear_staging_seq=*/false);
    return true;
}

static bool omni_llm_stage_apply_serial_prefill_fallback(struct omni_context *     ctx_omni,
                                                         struct common_params *    params,
                                                         const struct omni_embeds * staged_prefill) {
    if (ctx_omni == nullptr || params == nullptr || staged_prefill == nullptr) {
        return false;
    }

    const auto prefill_start = std::chrono::high_resolution_clock::now();
    const bool ok            = omni_llm_stage_prefill_apply(ctx_omni, params, *staged_prefill);
    if (ok) {
        omni_llm_stage_finalize_prefill(ctx_omni);
        omni_llm_stage_note_prefill_timing(
            ctx_omni, staged_prefill->index,
            omni_llm_stage_timing_elapsed_ms(prefill_start, std::chrono::high_resolution_clock::now()));
    }
    return ok;
}

void omni_llm_stage_worker_loop(struct omni_context * ctx_omni, struct common_params * params) {
    if (ctx_omni == nullptr || ctx_omni->llm_thread_info == nullptr) {
        return;
    }

    print_with_timestamp("LLM thread started\n");
    OmniLlmSchedulerState                        scheduler;
    OmniLlmStageDecodeRequest                   active_request;
    std::chrono::high_resolution_clock::time_point decode_start;
    bool                                        policy_fallback_logged = false;
    omni_embeds *                               staged_prefill         = nullptr;

    while (ctx_omni->workers.llm_thread_running) {
        if (!scheduler.decode_active) {
            std::vector<omni_embeds *> simplex_prefills;
            omni_embeds *              duplex_prefill  = nullptr;
            bool                       decode_requested = false;
            bool                       serial_prefill_fallback = false;

            if (ctx_omni->duplex_mode &&
                ctx_omni->llm_schedule_policy == OmniLlmSchedulePolicy::MICRO_BATCH &&
                scheduler.staged_ready) {
                const int staged_chunk_idx = scheduler.staged_chunk_idx;
                if (omni_llm_stage_promote_staged(ctx_omni, scheduler)) {
                    delete staged_prefill;
                    staged_prefill = nullptr;

                    omni_llm_stage_set_active_duplex_chunk_idx(ctx_omni, staged_chunk_idx);
                    scheduler.active_chunk_idx = staged_chunk_idx;
                    decode_requested           = true;
                } else {
                    LOG_WRN("%s: promote failed for chunk %d, fallback to serial prefill\n", __func__,
                            staged_chunk_idx);
                    omni_llm_stage_reset_staged_state(ctx_omni, scheduler, /*clear_staging_seq=*/true);
                    duplex_prefill          = staged_prefill;
                    staged_prefill          = nullptr;
                    serial_prefill_fallback = duplex_prefill != nullptr;
                }
            }

            if (!decode_requested && duplex_prefill == nullptr) {
                std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
                auto *                       llm_thread_info = ctx_omni->llm_thread_info;
                llm_thread_info->cv.wait(lock, [&] {
                    return !llm_thread_info->queue.empty() || llm_thread_info->decode_requested.load() ||
                           !ctx_omni->workers.llm_thread_running;
                });

                if (!ctx_omni->workers.llm_thread_running) {
                    break;
                }

                decode_requested = llm_thread_info->decode_requested.exchange(false);
                if (ctx_omni->duplex_mode) {
                    if (!llm_thread_info->queue.empty()) {
                        duplex_prefill = llm_thread_info->queue.front();
                        llm_thread_info->queue.pop();
                    }
                } else {
                    simplex_prefills = omni_llm_stage_drain_prefill_queue(llm_thread_info);
                }
            }

            ctx_omni->llm_thread_info->cv.notify_all();

            if (ctx_omni->duplex_mode) {
                if (ctx_omni->llm_schedule_policy == OmniLlmSchedulePolicy::MICRO_BATCH &&
                    !omni_llm_stage_micro_batch_supported(ctx_omni) && !policy_fallback_logged) {
                    print_with_timestamp(
                        "LLM scheduler: MICRO_BATCH requires kv_unified=true and n_seq_max>=2, fallback to "
                        "DECODE_FIRST for this context\n");
                    policy_fallback_logged = true;
                }

                if (duplex_prefill != nullptr) {
                    if (duplex_prefill->encode_failed) {
                        print_with_timestamp("LLM thread: skip chunk %d because encode stage failed\n",
                                             duplex_prefill->index);
                        delete duplex_prefill;
                        omni_llm_stage_post_pipeline_result(ctx_omni, false, false);
                        continue;
                    }

                    omni_llm_stage_set_active_duplex_chunk_idx(ctx_omni, duplex_prefill->index);
                    scheduler.active_chunk_idx = duplex_prefill->index;

                    bool prefill_ok = false;
                    if (serial_prefill_fallback) {
                        prefill_ok = omni_llm_stage_apply_serial_prefill_fallback(ctx_omni, params, duplex_prefill);
                    } else {
                        const auto prefill_start = std::chrono::high_resolution_clock::now();
                        prefill_ok              = omni_llm_stage_prefill_apply(ctx_omni, params, *duplex_prefill);
                        if (prefill_ok) {
                            omni_llm_stage_finalize_prefill(ctx_omni);
                            omni_llm_stage_note_prefill_timing(
                                ctx_omni, duplex_prefill->index,
                                omni_llm_stage_timing_elapsed_ms(prefill_start,
                                                                 std::chrono::high_resolution_clock::now()));
                        }
                    }

                    if (!prefill_ok) {
                        LOG_ERR("%s: failed to apply duplex prefill for chunk %d\n", __func__, duplex_prefill->index);
                        delete duplex_prefill;
                        omni_llm_stage_post_pipeline_result(ctx_omni, false, false);
                        continue;
                    }

                    delete duplex_prefill;

                    print_with_timestamp(
                        "LLM thread: prefill done, n_past=%d, n_keep=%d, 本次消耗 %d tokens, duplex_mode=%d\n",
                        ctx_omni->session.n_past, ctx_omni->session.prompt.n_keep,
                        ctx_omni->session.n_past - ctx_omni->session.prompt.n_keep, ctx_omni->duplex_mode);
                    decode_requested = true;
                } else {
                    scheduler.active_chunk_idx = omni_llm_stage_active_duplex_chunk_idx(ctx_omni);
                }
            } else {
                bool simplex_prefill_failed = false;
                if (!simplex_prefills.empty()) {
                    print_with_timestamp("LLM thread: start prefill, n_past=%d, n_keep=%d, n_ctx=%d\n",
                                         ctx_omni->session.n_past, ctx_omni->session.prompt.n_keep, params->n_ctx);
                    print_with_timestamp("LLM thread: prefill continuing, n_past=%d (no KV cache clear)\n",
                                         ctx_omni->session.n_past);
                }

                for (auto *& embeds : simplex_prefills) {
                    if (embeds->encode_failed) {
                        print_with_timestamp("LLM thread: skip simplex prefill because encode stage failed\n");
                        delete embeds;
                        embeds = nullptr;
                        omni_llm_stage_post_pipeline_result(ctx_omni, false, false);
                        continue;
                    }

                    if (!omni_llm_stage_prefill_apply(ctx_omni, params, *embeds)) {
                        delete embeds;
                        embeds = nullptr;
                        omni_llm_stage_post_pipeline_result(ctx_omni, false, false);
                        for (auto * pending_embeds : simplex_prefills) {
                            if (pending_embeds != nullptr) {
                                delete pending_embeds;
                            }
                        }
                        simplex_prefills.clear();
                        simplex_prefill_failed = true;
                        decode_requested = false;
                        break;
                    }
                    delete embeds;
                    embeds = nullptr;
                }

                if (!simplex_prefill_failed && !simplex_prefills.empty()) {
                    print_with_timestamp(
                        "LLM thread: prefill done, n_past=%d, n_keep=%d, 本次消耗 %d tokens, duplex_mode=%d\n",
                        ctx_omni->session.n_past, ctx_omni->session.prompt.n_keep,
                        ctx_omni->session.n_past - ctx_omni->session.prompt.n_keep, ctx_omni->duplex_mode);
                    omni_llm_stage_finalize_prefill(ctx_omni);
                }
            }

            if (!decode_requested) {
                continue;
            }

            active_request = omni_llm_stage_get_pipeline_request(ctx_omni);
            omni_llm_stage_prepare_worker_decode(ctx_omni, active_request);

            LOG_INF("<user>%s\n", ctx_omni->params->prompt.c_str());
            LOG_INF("<assistant>");

            decode_start            = std::chrono::high_resolution_clock::now();
            scheduler.decode_active = true;
        }

        if (ctx_omni->duplex_mode &&
            ctx_omni->llm_schedule_policy == OmniLlmSchedulePolicy::MICRO_BATCH &&
            !scheduler.staged_ready &&
            staged_prefill == nullptr &&
            omni_llm_stage_has_pending_prefill(ctx_omni) &&
            omni_llm_stage_micro_batch_supported(ctx_omni)) {
            omni_llm_stage_stage_prefill_speculatively(ctx_omni, params, scheduler, &staged_prefill);
        }

        OmniLlmDecodeSliceResult slice;
        if (!omni_llm_stage_decode_slice(ctx_omni, active_request, ctx_omni->reschedule_interval_tokens, &slice)) {
            omni_llm_stage_finish_decode_text_stream(ctx_omni);
            omni_llm_stage_post_pipeline_result(ctx_omni, false, false);
            break;
        }

        if (ctx_omni->duplex_mode &&
            ctx_omni->llm_schedule_policy == OmniLlmSchedulePolicy::PREFILL_PREEMPTIVE &&
            !slice.finished && !slice.interrupted && omni_llm_stage_has_pending_prefill(ctx_omni)) {
            if (!omni_llm_stage_close_decode_preemptively(ctx_omni)) {
                omni_llm_stage_finish_decode_text_stream(ctx_omni);
                omni_llm_stage_post_pipeline_result(ctx_omni, false, false);
                break;
            }

            slice.interrupted = true;
            slice.preempted   = true;
            slice.llm_finish  = true;
        }

        omni_llm_stage_publish_decode_slice(ctx_omni, active_request, &slice);

        if (!slice.finished && !slice.interrupted) {
            continue;
        }

        omni_llm_stage_finish_decode_text_stream(ctx_omni);
        omni_llm_stage_note_decode_timing(
            ctx_omni, scheduler.active_chunk_idx,
            omni_llm_stage_timing_elapsed_ms(decode_start, std::chrono::high_resolution_clock::now()));
        const OmniTurnCloseKind close_kind = slice.preempted ? OmniTurnCloseKind::preempt :
                                             slice.interrupted ? OmniTurnCloseKind::abort :
                                                                 OmniTurnCloseKind::finish;

        if ((slice.interrupted || slice.preempted) && staged_prefill != nullptr) {
            omni_llm_stage_reset_staged_state(ctx_omni, scheduler, /*clear_staging_seq=*/true);
            delete staged_prefill;
            staged_prefill = nullptr;
        }

        omni_turn_coordinator_close(ctx_omni, close_kind, slice.preempted ? "pending-prefill" : nullptr);
        omni_llm_stage_post_pipeline_result(ctx_omni, !slice.interrupted || slice.preempted,
                                            slice.ended_with_listen);

        scheduler.decode_active    = false;
        scheduler.active_chunk_idx = -1;
    }

    if (staged_prefill != nullptr) {
        omni_llm_stage_reset_staged_state(ctx_omni, scheduler, /*clear_staging_seq=*/true);
        delete staged_prefill;
    }

    print_with_timestamp("LLM thread stopped\n");
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

bool omni_llm_stage_decode_slice(struct omni_context *             ctx_omni,
                                 const OmniLlmStageDecodeRequest & request,
                                 int                               max_tokens,
                                 OmniLlmDecodeSliceResult *        out_result) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || ctx_omni->params == nullptr) {
        return false;
    }

    (void) request;

    if (!ctx_omni->turn.decode_prefix_applied) {
        omni_llm_stage_apply_decode_prefix(ctx_omni, omni_llm_stage_build_decode_prefix(ctx_omni));
        ctx_omni->turn.decode_prefix_applied = true;
        print_with_timestamp("LLM decode: max_tgt_len = %d, n_predict = %d, n_ctx = %d\n",
                             omni_llm_stage_decode_max_tgt_len(ctx_omni), ctx_omni->params->n_predict,
                             ctx_omni->params->n_ctx);
    }

    OmniLlmDecodeSliceResult result;
    const int                llm_n_embd      = llama_model_n_embd(llama_get_model(ctx_omni->ctx_llama));
    const int                max_tgt_len     = omni_llm_stage_decode_max_tgt_len(ctx_omni);
    const int                schedule_budget = max_tokens > 0 ? max_tokens : 1;

    while (!result.finished && !result.interrupted) {
        if (ctx_omni->gate.break_event.load() && result.generated_tokens == 0) {
            result.interrupted = true;
            break;
        }

        if (ctx_omni->turn.generated_decode_tokens >= max_tgt_len) {
            result.finished = true;
            break;
        }

        fflush(stdout);

        std::string              chunk_text;
        std::vector<llama_token> chunk_token_ids;
        std::vector<float>       chunk_hidden_states;
        int                      valid_chunk_tokens     = 0;
        int                      total_tokens_generated = 0;
        bool                     chunk_is_end_of_turn   = false;
        bool                     chunk_llm_finish       = false;
        const int                max_chunk_tokens       =
            ctx_omni->duplex_mode ? ctx_omni->max_new_speak_tokens_per_chunk : 0;
        bool                     chunk_limit_reached =
            max_chunk_tokens > 0 && ctx_omni->turn.current_chunk_tokens >= max_chunk_tokens;

        while (valid_chunk_tokens < kLegacyDecodeChunkStepSize &&
               !chunk_llm_finish &&
               !ctx_omni->gate.break_event.load() &&
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
                return false;
            }

            if (hidden_states != nullptr && omni_tts_is_valid_token(sampled_token)) {
                chunk_token_ids.push_back(sampled_token);
                chunk_hidden_states.insert(chunk_hidden_states.end(), hidden_states, hidden_states + llm_n_embd);
                valid_chunk_tokens++;
                ctx_omni->turn.current_chunk_tokens++;

                if (max_chunk_tokens > 0 && ctx_omni->turn.current_chunk_tokens >= max_chunk_tokens) {
                    chunk_limit_reached = true;
                }
            }
            free(hidden_states);

            const OmniTokenType token_type = omni_get_token_type(ctx_omni, sampled_token);
            omni_llm_stage_mark_decode_turn_end(ctx_omni, token_type, chunk_is_end_of_turn);

            if (omni_is_end_token(ctx_omni, sampled_token)) {
                chunk_llm_finish = true;
                omni_llm_stage_handle_decode_end_token(ctx_omni, token_type);
                break;
            }

            chunk_text += std::string(tmp);
            fflush(stdout);
        }

        if (chunk_limit_reached) {
            if (ctx_omni->special_token_chunk_eos >= 0) {
                std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
                std::vector<llama_token>    chunk_eos_tokens = { ctx_omni->special_token_chunk_eos };
                omni_llm_stage_eval_tokens(ctx_omni, ctx_omni->params, std::move(chunk_eos_tokens),
                                           ctx_omni->params->n_batch, &ctx_omni->session.n_past);
            }
            chunk_llm_finish                    = true;
            ctx_omni->turn.current_chunk_tokens = 0;
        }

        if (ctx_omni->duplex_mode && ctx_omni->special_token_unit_end >= 0) {
            std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
            std::vector<llama_token>    unit_end_tokens = { ctx_omni->special_token_unit_end };
            omni_llm_stage_eval_tokens(ctx_omni, ctx_omni->params, std::move(unit_end_tokens), ctx_omni->params->n_batch,
                                       &ctx_omni->session.n_past);
        }

        ctx_omni->turn.generated_decode_tokens += total_tokens_generated;
        result.generated_tokens += total_tokens_generated;
        result.text += std::move(chunk_text);
        result.token_ids.insert(result.token_ids.end(), chunk_token_ids.begin(), chunk_token_ids.end());
        result.hidden_states.insert(result.hidden_states.end(), chunk_hidden_states.begin(), chunk_hidden_states.end());
        result.is_end_of_turn = result.is_end_of_turn || chunk_is_end_of_turn;
        result.ended_with_listen = ctx_omni->turn.ended_with_listen.load();

        if (chunk_llm_finish) {
            result.finished   = true;
            result.llm_finish = true;
            break;
        }

        if (ctx_omni->turn.generated_decode_tokens >= max_tgt_len) {
            result.finished = true;
            break;
        }

        if (ctx_omni->gate.break_event.load()) {
            result.interrupted = true;
            break;
        }

        if (result.generated_tokens >= schedule_budget) {
            break;
        }
    }

    result.ended_with_listen = ctx_omni->turn.ended_with_listen.load();
    if (out_result != nullptr) {
        *out_result = result;
    }

    return true;
}

bool omni_llm_stage_decode_run(struct omni_context *             ctx_omni,
                               const OmniLlmStageDecodeRequest & request,
                               OmniLlmStageDecodeResult *        out_result) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || ctx_omni->params == nullptr) {
        return false;
    }

    OmniLlmStageDecodeResult result;
    while (!result.interrupted) {
        OmniLlmDecodeSliceResult slice;
        if (!omni_llm_stage_decode_slice(ctx_omni, request, ctx_omni->reschedule_interval_tokens, &slice)) {
            return false;
        }

        omni_llm_stage_publish_decode_slice(ctx_omni, request, &slice);
        result.llm_finish        = result.llm_finish || slice.llm_finish;
        result.preempted         = result.preempted || slice.preempted;
        result.ended_with_listen = slice.ended_with_listen;

        if (slice.interrupted) {
            result.interrupted = true;
            break;
        }
        if (slice.finished) {
            break;
        }
    }

    omni_llm_stage_finish_decode_text_stream(ctx_omni);

    result.generated_decode_tokens = ctx_omni->turn.generated_decode_tokens;
    if (out_result != nullptr) {
        *out_result = result;
    }

    return true;
}
