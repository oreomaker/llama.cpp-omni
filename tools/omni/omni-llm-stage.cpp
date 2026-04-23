#include "omni-llm-stage.h"

#include "common/common.h"
#include "common/sampling.h"
#include "omni-impl.h"
#include "omni-log.h"
#include "omni-session-state.h"
#include "omni-sliding-window.h"
#include "omni-token-protocol.h"
#include "omni-tts-stage.h"
#include "omni-turn-coordinator.h"
#include "omni.h"
#include "src/llama-model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

namespace {
// =============================================================================
// Timing / Async Pipeline Bookkeeping
// Internal helpers for duplex timing, async request/result pairing, and worker
// decode preparation. These are not model-forward helpers.
// =============================================================================

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

void omni_llm_stage_note_decode_step_timing(struct omni_context *             ctx_omni,
                                            int                               chunk_idx,
                                            double                            ms,
                                            int                               sampled_token_count,
                                            int                               valid_token_count,
                                            bool                              chunk_limit_reached,
                                            bool                              ended_with_listen,
                                            bool                              llm_finish,
                                            bool                              interrupted) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.llm_decode_steps.emplace_back();
    auto & step = timing.llm_decode_steps.back();
    step.elapsed_ms                    = ms;
    step.sampled_token_count           = sampled_token_count;
    step.valid_token_count             = valid_token_count;
    step.chunk_limit_reached           = chunk_limit_reached;
    step.ended_with_listen             = ended_with_listen;
    step.llm_finish                    = llm_finish;
    step.interrupted                   = interrupted;
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

void omni_llm_stage_post_pipeline_result(struct omni_context * ctx_omni, bool decode_ok, bool ended_with_listen) {
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

void omni_llm_stage_prepare_decode_cycle(struct omni_context * ctx_omni, const OmniLlmStageDecodeRequest & request) {
    if (request.round_idx >= 0 && !ctx_omni->duplex_mode &&
        ctx_omni->session.current_round.round_idx != request.round_idx) {
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
        ctx_omni->gate.text_done      = false;
        ctx_omni->gate.text_streaming = true;
    }

    if (ctx_omni->use_tts) {
        ctx_omni->gate.speech_ready = false;
    }
}

// =============================================================================
// LLM Forward Internals
// Low-level token/hidden-state forward helpers and decode protocol helpers.
// These functions directly interact with llama forward/sampling state.
// =============================================================================

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

bool omni_llm_stage_eval_id_with_hidden(struct omni_context *  ctx_omni,
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

int omni_llm_stage_max_chunk_tokens(const struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || !ctx_omni->duplex_mode) {
        return 0;
    }
    return ctx_omni->max_new_speak_tokens_per_chunk;
}

void omni_llm_stage_eval_chunk_eos_token(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || ctx_omni->special_token_chunk_eos < 0) {
        return;
    }

    std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
    std::vector<llama_token>    chunk_eos_tokens = { ctx_omni->special_token_chunk_eos };
    omni_llm_stage_eval_tokens(ctx_omni, ctx_omni->params, std::move(chunk_eos_tokens), ctx_omni->params->n_batch,
                               &ctx_omni->session.n_past);
}

void omni_llm_stage_eval_unit_end_token(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || !ctx_omni->duplex_mode || ctx_omni->special_token_unit_end < 0) {
        return;
    }

    std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
    std::vector<llama_token>    unit_end_tokens = { ctx_omni->special_token_unit_end };
    omni_llm_stage_eval_tokens(ctx_omni, ctx_omni->params, std::move(unit_end_tokens), ctx_omni->params->n_batch,
                               &ctx_omni->session.n_past);
}

void omni_llm_stage_mark_decode_turn_end(struct omni_context * ctx_omni,
                                         OmniTokenType         token_type,
                                         bool &                is_end_of_turn) {
    if (!ctx_omni->duplex_mode) {
        return;
    }

    if (token_type == OmniTokenType::TURN_EOS || token_type == OmniTokenType::TTS_EOS ||
        token_type == OmniTokenType::EOS) {
        is_end_of_turn                    = true;
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
                                                 const OmniLlmStageDecodeChunk &   chunk,
                                                 int                               llm_n_embd) {
    if (!ctx_omni->async || !ctx_omni->use_tts || ctx_omni->tts_thread_info == nullptr ||
        (chunk.text.empty() && !chunk.llm_finish)) {
        return;
    }

    LLMOut * llm_out          = new LLMOut();
    llm_out->text             = chunk.text;
    llm_out->n_past           = ctx_omni->session.n_past;
    llm_out->llm_finish       = chunk.llm_finish;
    llm_out->debug_dir        = request.debug_dir;
    llm_out->round_meta       = omni_session_round_meta(ctx_omni);
    llm_out->token_ids        = chunk.token_ids;
    llm_out->hidden_states    = chunk.hidden_states;
    llm_out->n_embd           = llm_n_embd;
    llm_out->is_end_of_turn   = chunk.is_end_of_turn;
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
        return ctx_omni->tts_thread_info->queue.size() < static_cast<size_t>(ctx_omni->tts_thread_info->MAX_QUEUE_SIZE);
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

// =============================================================================
// Public LLM Forward API
// Shared by prefill and decode paths for plain token/string evaluation.
// =============================================================================

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

bool omni_llm_stage_eval_string(struct omni_context *  ctx_omni,
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

bool omni_llm_stage_eval_string_with_hidden(struct omni_context *  ctx_omni,
                                            struct common_params * params,
                                            const char *           str,
                                            int                    n_batch,
                                            int *                  n_past,
                                            bool                   add_bos,
                                            float *&               hidden_states) {
    std::string              str_buf = str;
    std::vector<llama_token> tokens  = common_tokenize(ctx_omni->ctx_llama, str_buf, add_bos, true);
    return omni_llm_stage_eval_tokens_with_hidden(ctx_omni, params, std::move(tokens), n_batch, n_past, hidden_states);
}

// =============================================================================
// Public Prefill Logic
// Applies one encoded multimodal input unit onto the main LLM sequence.
// =============================================================================

static bool omni_llm_stage_append_prefill_token_embeddings(struct omni_context * ctx_omni,
                                                           const std::string &   segment,
                                                           int                   hidden_size,
                                                           std::vector<float> &  merged_embeddings) {
    if (ctx_omni == nullptr || ctx_omni->model == nullptr || ctx_omni->model->tok_embd == nullptr) {
        LOG_ERR("%s: missing LLM token embeddings\n", __func__);
        return false;
    }

    if (segment.empty()) {
        return true;
    }

    std::string              segment_buf = segment;
    std::vector<llama_token> tokens      = common_tokenize(ctx_omni->ctx_llama, segment_buf, false, true);
    if (tokens.empty()) {
        return true;
    }

    struct ggml_tensor * tok_embd = ctx_omni->model->tok_embd;
    if (tok_embd->ne[0] != hidden_size) {
        LOG_ERR("%s: token embedding width mismatch: got %lld expected %d\n", __func__, (long long) tok_embd->ne[0],
                hidden_size);
        return false;
    }

    const int64_t vocab_size = tok_embd->ne[1];
    const size_t  row_stride = tok_embd->nb[1];
    const size_t  row_size   = ggml_row_size(tok_embd->type, tok_embd->ne[0]);
    if (row_stride < row_size) {
        LOG_ERR("%s: token embedding row stride too small: stride=%zu row_size=%zu\n", __func__, row_stride, row_size);
        return false;
    }

    const struct ggml_type_traits * type_traits = ggml_get_type_traits(tok_embd->type);
    if (type_traits == nullptr) {
        LOG_ERR("%s: failed to get ggml type traits for token embeddings\n", __func__);
        return false;
    }

    if (ggml_is_quantized(tok_embd->type)) {
        if (type_traits->to_float == nullptr) {
            LOG_ERR("%s: token embedding type %s does not support dequantize\n", __func__,
                    ggml_type_name(tok_embd->type));
            return false;
        }
        ggml_quantize_init(tok_embd->type);
    } else if (tok_embd->type != GGML_TYPE_F32 && tok_embd->type != GGML_TYPE_F16) {
        LOG_ERR("%s: unsupported token embedding type %s\n", __func__, ggml_type_name(tok_embd->type));
        return false;
    }

    std::vector<uint8_t> raw_row(row_size);
    std::vector<float>   f32_row(hidden_size);
    merged_embeddings.reserve(merged_embeddings.size() + tokens.size() * (size_t) hidden_size);

    for (llama_token token_id : tokens) {
        if (token_id < 0 || token_id >= vocab_size) {
            LOG_ERR("%s: token id %d out of token embedding range [0, %lld)\n", __func__, token_id,
                    (long long) vocab_size);
            return false;
        }

        ggml_backend_tensor_get(tok_embd, raw_row.data(), (size_t) token_id * row_stride, row_size);

        switch (tok_embd->type) {
            case GGML_TYPE_F32:
                memcpy(f32_row.data(), raw_row.data(), hidden_size * sizeof(float));
                break;
            case GGML_TYPE_F16:
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw_row.data(), f32_row.data(), hidden_size);
                break;
            default:
                type_traits->to_float(raw_row.data(), f32_row.data(), hidden_size);
                break;
        }

        merged_embeddings.insert(merged_embeddings.end(), f32_row.begin(), f32_row.end());
    }

    return true;
}

static bool omni_llm_stage_append_prefill_raw_embeddings(const std::vector<float> & source,
                                                         int                        hidden_size,
                                                         std::vector<float> &       merged_embeddings) {
    if (source.empty()) {
        return true;
    }

    if (hidden_size <= 0 || source.size() % (size_t) hidden_size != 0) {
        LOG_ERR("%s: invalid embedding payload size=%zu hidden_size=%d\n", __func__, source.size(), hidden_size);
        return false;
    }

    merged_embeddings.insert(merged_embeddings.end(), source.begin(), source.end());
    return true;
}

static bool omni_llm_stage_build_prefill_embeddings(struct omni_context *      ctx_omni,
                                                    const struct omni_embeds & embeds,
                                                    int                        hidden_size,
                                                    std::vector<float> &       merged_embeddings) {
    if (ctx_omni == nullptr) {
        return false;
    }

    if (!embeds.vision_embed.empty()) {
        const bool has_slices = embeds.vision_embed.size() > 1;

        if (!omni_llm_stage_append_prefill_token_embeddings(
                ctx_omni, ctx_omni->duplex_mode ? "<unit><image>" : "<image>", hidden_size, merged_embeddings) ||
            !omni_llm_stage_append_prefill_raw_embeddings(embeds.vision_embed[0], hidden_size, merged_embeddings) ||
            !omni_llm_stage_append_prefill_token_embeddings(ctx_omni, "</image>", hidden_size, merged_embeddings)) {
            return false;
        }

        if (has_slices) {
            for (size_t i = 1; i < embeds.vision_embed.size(); ++i) {
                if (!omni_llm_stage_append_prefill_token_embeddings(ctx_omni, "<slice>", hidden_size,
                                                                    merged_embeddings) ||
                    !omni_llm_stage_append_prefill_raw_embeddings(embeds.vision_embed[i], hidden_size,
                                                                  merged_embeddings) ||
                    !omni_llm_stage_append_prefill_token_embeddings(ctx_omni, "</slice>", hidden_size,
                                                                    merged_embeddings)) {
                    return false;
                }
            }

            if (!omni_llm_stage_append_prefill_token_embeddings(ctx_omni, "\n", hidden_size, merged_embeddings)) {
                return false;
            }
        }

        if (!embeds.audio_embed.empty()) {
            if (!ctx_omni->duplex_mode && !omni_llm_stage_append_prefill_token_embeddings(
                                              ctx_omni, "<|audio_start|>", hidden_size, merged_embeddings)) {
                return false;
            }

            if (!omni_llm_stage_append_prefill_raw_embeddings(embeds.audio_embed, hidden_size, merged_embeddings)) {
                return false;
            }

            if (!ctx_omni->duplex_mode && !omni_llm_stage_append_prefill_token_embeddings(
                                              ctx_omni, "<|audio_end|>", hidden_size, merged_embeddings)) {
                return false;
            }
        }

        return true;
    }

    if (!omni_llm_stage_append_prefill_token_embeddings(ctx_omni, ctx_omni->duplex_mode ? "<unit>" : "<|audio_start|>",
                                                        hidden_size, merged_embeddings) ||
        !omni_llm_stage_append_prefill_raw_embeddings(embeds.audio_embed, hidden_size, merged_embeddings)) {
        return false;
    }

    if (!ctx_omni->duplex_mode &&
        !omni_llm_stage_append_prefill_token_embeddings(ctx_omni, "<|audio_end|>", hidden_size, merged_embeddings)) {
        return false;
    }

    return true;
}

static void omni_llm_stage_prefill_apply_legacy(struct omni_context *      ctx_omni,
                                                struct common_params *     params,
                                                const struct omni_embeds & embeds,
                                                int                        hidden_size) {
    LOG_WRN("%s: using legacy segmented prefill path (multiple decode passes)\n", __func__);

    if (!embeds.vision_embed.empty()) {
        const int  n_chunks         = (int) embeds.vision_embed.size();
        const int  tokens_per_chunk = (int) embeds.vision_embed[0].size() / hidden_size;
        const int  n_audio_tokens   = embeds.audio_embed.size() / hidden_size;
        const bool has_audio        = n_audio_tokens > 0;
        const bool has_slices       = n_chunks > 1;

        if (ctx_omni->duplex_mode) {
            omni_llm_stage_eval_string(ctx_omni, params, "<unit><image>", params->n_batch, &ctx_omni->session.n_past,
                                       false);
        } else {
            omni_llm_stage_eval_string(ctx_omni, params, "<image>", params->n_batch, &ctx_omni->session.n_past, false);
        }

        prefill_with_emb(ctx_omni, params, const_cast<float *>(embeds.vision_embed[0].data()), tokens_per_chunk,
                         params->n_batch, &ctx_omni->session.n_past);
        omni_llm_stage_eval_string(ctx_omni, params, "</image>", params->n_batch, &ctx_omni->session.n_past, false);

        if (has_slices) {
            for (int i = 1; i < n_chunks; ++i) {
                omni_llm_stage_eval_string(ctx_omni, params, "<slice>", params->n_batch, &ctx_omni->session.n_past,
                                           false);
                prefill_with_emb(ctx_omni, params, const_cast<float *>(embeds.vision_embed[i].data()), tokens_per_chunk,
                                 params->n_batch, &ctx_omni->session.n_past);
                omni_llm_stage_eval_string(ctx_omni, params, "</slice>", params->n_batch, &ctx_omni->session.n_past,
                                           false);
            }
            omni_llm_stage_eval_string(ctx_omni, params, "\n", params->n_batch, &ctx_omni->session.n_past, false);
        }

        if (has_audio) {
            if (!ctx_omni->duplex_mode) {
                omni_llm_stage_eval_string(ctx_omni, params, "<|audio_start|>", params->n_batch,
                                           &ctx_omni->session.n_past, false);
            }
            prefill_with_emb(ctx_omni, params, const_cast<float *>(embeds.audio_embed.data()), n_audio_tokens,
                             params->n_batch, &ctx_omni->session.n_past);
            if (!ctx_omni->duplex_mode) {
                omni_llm_stage_eval_string(ctx_omni, params, "<|audio_end|>", params->n_batch,
                                           &ctx_omni->session.n_past, false);
            }
        }
    } else {
        const int n_audio_tokens = embeds.audio_embed.size() / hidden_size;

        if (ctx_omni->duplex_mode) {
            omni_llm_stage_eval_string(ctx_omni, params, "<unit>", params->n_batch, &ctx_omni->session.n_past, false);
        } else {
            omni_llm_stage_eval_string(ctx_omni, params, "<|audio_start|>", params->n_batch, &ctx_omni->session.n_past,
                                       false);
        }

        if (n_audio_tokens > 0) {
            prefill_with_emb(ctx_omni, params, const_cast<float *>(embeds.audio_embed.data()), n_audio_tokens,
                             params->n_batch, &ctx_omni->session.n_past);
        }

        if (!ctx_omni->duplex_mode) {
            omni_llm_stage_eval_string(ctx_omni, params, "<|audio_end|>", params->n_batch, &ctx_omni->session.n_past,
                                       false);
        }
    }
}

void omni_llm_stage_prefill_apply(struct omni_context *      ctx_omni,
                                  struct common_params *     params,
                                  const struct omni_embeds & embeds) {
    const int hidden_size = llama_model_n_embd(llama_get_model(ctx_omni->ctx_llama));

    if (ctx_omni->session.sliding_window_config.mode != "off") {
        sliding_window_register_unit_start(ctx_omni);
    }

    if (!embeds.vision_embed.empty()) {
        const int  n_chunks         = (int) embeds.vision_embed.size();
        const int  tokens_per_chunk = (int) embeds.vision_embed[0].size() / hidden_size;
        const int  n_audio_tokens   = embeds.audio_embed.size() / hidden_size;
        const bool has_slices       = n_chunks > 1;

        print_with_timestamp("Omni模式: %d vision chunks (%d tokens each), %d audio tokens, has_slices=%d\n", n_chunks,
                             tokens_per_chunk, n_audio_tokens, has_slices);
    } else {
        const int n_audio_tokens = embeds.audio_embed.size() / hidden_size;
        print_with_timestamp("用户语音: %d audio tokens\n", n_audio_tokens);
    }

    const bool lora_active = params != nullptr && !params->lora_init_without_apply && !params->lora_adapters.empty();

    std::vector<float> merged_embeddings;
    if (lora_active) {
        LOG_WRN("%s: active LoRA detected, using legacy segmented prefill to preserve token embedding deltas\n",
                __func__);
        omni_llm_stage_prefill_apply_legacy(ctx_omni, params, embeds, hidden_size);
    } else if (!omni_llm_stage_build_prefill_embeddings(ctx_omni, embeds, hidden_size, merged_embeddings)) {
        LOG_WRN("%s: merged prefill build failed, falling back to legacy segmented prefill\n", __func__);
        omni_llm_stage_prefill_apply_legacy(ctx_omni, params, embeds, hidden_size);
    } else {
        const int total_tokens = merged_embeddings.empty() ? 0 : (int) (merged_embeddings.size() / hidden_size);
        if (total_tokens > 0) {
            const int prefill_batch = params->n_batch > 0 ? std::min(total_tokens, params->n_batch) : total_tokens;
            if (!prefill_with_emb(ctx_omni, params, merged_embeddings.data(), total_tokens, prefill_batch,
                                  &ctx_omni->session.n_past)) {
                LOG_ERR("%s: merged prefill decode failed\n", __func__);
            }
        }
    }

    if (ctx_omni->session.sliding_window_config.mode != "off") {
        const std::string input_type = embeds.vision_embed.empty() ? "audio" : "omni";
        sliding_window_register_unit_end(ctx_omni, input_type, {}, false);
    }
}

void omni_llm_stage_finalize_prefill(struct omni_context * ctx_omni) {
    if (ctx_omni->session.sliding_window_config.mode != "off") {
        sliding_window_enforce(ctx_omni);
    }
}

namespace {
// =============================================================================
// Scheduler Planning and Execution
// The worker loop only waits for pending work; this section turns that work
// into an explicit scheduler plan while preserving the current policy:
// - simplex: drain all queued prefills, then run one decode on explicit request
// - duplex: every drained prefill unit is followed by one decode cycle
// - duplex: an explicit decode request with no queued prefill still runs once
// =============================================================================

enum class OmniLlmStageSchedulerDecodeMode {
    none,
    once_after_prefill_drain,
    after_each_prefill,
};

struct OmniLlmStageSchedulerWorkset {
    std::vector<omni_embeds *> prefills;
    bool                       decode_requested = false;
};

struct OmniLlmStageSchedulerPlan {
    std::vector<omni_embeds *>      prefills;
    OmniLlmStageSchedulerDecodeMode decode_mode = OmniLlmStageSchedulerDecodeMode::none;
};

void omni_llm_stage_scheduler_release_prefills(std::vector<omni_embeds *> & prefills, size_t begin_idx = 0) {
    for (size_t i = begin_idx; i < prefills.size(); ++i) {
        delete prefills[i];
    }
}

OmniLlmStageSchedulerWorkset omni_llm_stage_scheduler_wait_work(struct omni_context * ctx_omni, bool * should_stop) {
    OmniLlmStageSchedulerWorkset workset;
    if (should_stop != nullptr) {
        *should_stop = true;
    }
    if (ctx_omni == nullptr || ctx_omni->llm_thread_info == nullptr) {
        return workset;
    }

    std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
    auto *                       llm_thread_info = ctx_omni->llm_thread_info;
    llm_thread_info->cv.wait(lock, [&] {
        return !llm_thread_info->queue.empty() || llm_thread_info->decode_requested.load() ||
               !ctx_omni->workers.llm_thread_running;
    });

    if (!ctx_omni->workers.llm_thread_running) {
        return workset;
    }

    workset.prefills         = omni_llm_stage_drain_prefill_queue(llm_thread_info);
    workset.decode_requested = llm_thread_info->decode_requested.exchange(false);
    if (ctx_omni->duplex_mode && !workset.prefills.empty()) {
        workset.decode_requested = true;
    }

    if (should_stop != nullptr) {
        *should_stop = false;
    }
    return workset;
}

OmniLlmStageSchedulerPlan omni_llm_stage_scheduler_build_plan(struct omni_context *        ctx_omni,
                                                              OmniLlmStageSchedulerWorkset workset) {
    OmniLlmStageSchedulerPlan plan;
    plan.prefills = std::move(workset.prefills);

    if (ctx_omni == nullptr) {
        return plan;
    }

    if (ctx_omni->duplex_mode) {
        if (!plan.prefills.empty() || workset.decode_requested) {
            plan.decode_mode = OmniLlmStageSchedulerDecodeMode::after_each_prefill;
        }
    } else if (workset.decode_requested) {
        plan.decode_mode = OmniLlmStageSchedulerDecodeMode::once_after_prefill_drain;
    }

    return plan;
}

bool omni_llm_stage_scheduler_has_work(const OmniLlmStageSchedulerPlan & plan) {
    return !plan.prefills.empty() || plan.decode_mode != OmniLlmStageSchedulerDecodeMode::none;
}

void omni_llm_stage_log_prefill_activity(struct omni_context * ctx_omni, struct common_params * params) {
    if (ctx_omni == nullptr || params == nullptr) {
        return;
    }

    print_with_timestamp("LLM thread: start prefill, n_past=%d, n_keep=%d, n_ctx=%d\n", ctx_omni->session.n_past,
                         ctx_omni->session.prompt.n_keep, params->n_ctx);
    print_with_timestamp("LLM thread: prefill continuing, n_past=%d (no KV cache clear)\n", ctx_omni->session.n_past);
}

void omni_llm_stage_log_prefill_done(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return;
    }

    print_with_timestamp("LLM thread: prefill done, n_past=%d, n_keep=%d, 本次消耗 %d tokens, duplex_mode=%d\n",
                         ctx_omni->session.n_past, ctx_omni->session.prompt.n_keep,
                         ctx_omni->session.n_past - ctx_omni->session.prompt.n_keep, ctx_omni->duplex_mode);
}

// Current design:
// - This absorbs the old worker_decode_once() responsibility.
// - The scheduler owns one full async decode cycle: fetch request metadata,
//   prepare decode state, call decode_run(), close the turn, and post result.
// - decode_run() remains the inner decode execution wrapper only.
bool omni_llm_stage_scheduler_run_decode_cycle(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return false;
    }

    const OmniLlmStageDecodeRequest request = omni_llm_stage_get_pipeline_request(ctx_omni);
    omni_llm_stage_prepare_decode_cycle(ctx_omni, request);

    LOG_INF("<user>%s\n", ctx_omni->params->prompt.c_str());
    LOG_INF("<assistant>");

    const int                chunk_idx    = omni_llm_stage_active_duplex_chunk_idx(ctx_omni);
    const auto               decode_start = std::chrono::high_resolution_clock::now();
    OmniLlmStageDecodeResult decode_result;
    const bool               decode_ok = omni_llm_stage_decode_run(ctx_omni, request, &decode_result);
    omni_llm_stage_note_decode_timing(
        ctx_omni, chunk_idx, omni_llm_stage_timing_elapsed_ms(decode_start, std::chrono::high_resolution_clock::now()));
    if (decode_ok) {
        omni_turn_coordinator_close(ctx_omni,
                                    decode_result.interrupted ? OmniTurnCloseKind::abort : OmniTurnCloseKind::finish);
    }

    omni_llm_stage_post_pipeline_result(ctx_omni, decode_ok && !decode_result.interrupted,
                                        decode_result.ended_with_listen);
    return decode_ok;
}

bool omni_llm_stage_scheduler_apply_simplex_prefills(struct omni_context *        ctx_omni,
                                                     struct common_params *       params,
                                                     std::vector<omni_embeds *> & prefills) {
    if (ctx_omni == nullptr || params == nullptr) {
        return false;
    }

    bool applied_prefill = false;
    for (auto * embeds : prefills) {
        if (embeds == nullptr) {
            continue;
        }

        if (embeds->encode_failed) {
            print_with_timestamp("LLM thread: skip chunk %d because encode stage failed\n", embeds->index);
            delete embeds;
            omni_llm_stage_post_pipeline_result(ctx_omni, false, false);
            continue;
        }

        omni_llm_stage_prefill_apply(ctx_omni, params, *embeds);
        applied_prefill = true;
        delete embeds;
    }

    if (applied_prefill) {
        omni_llm_stage_log_prefill_done(ctx_omni);
        omni_llm_stage_finalize_prefill(ctx_omni);
    }

    return true;
}

bool omni_llm_stage_scheduler_apply_duplex_prefill(struct omni_context *  ctx_omni,
                                                   struct common_params * params,
                                                   omni_embeds *          embeds,
                                                   bool *                 out_prefill_applied) {
    if (out_prefill_applied != nullptr) {
        *out_prefill_applied = false;
    }
    if (ctx_omni == nullptr || params == nullptr || embeds == nullptr) {
        return false;
    }

    if (embeds->encode_failed) {
        print_with_timestamp("LLM thread: skip chunk %d because encode stage failed\n", embeds->index);
        delete embeds;
        omni_llm_stage_post_pipeline_result(ctx_omni, false, false);
        return true;
    }

    omni_llm_stage_set_active_duplex_chunk_idx(ctx_omni, embeds->index);
    const auto prefill_start = std::chrono::high_resolution_clock::now();
    omni_llm_stage_prefill_apply(ctx_omni, params, *embeds);
    omni_llm_stage_finalize_prefill(ctx_omni);
    omni_llm_stage_note_prefill_timing(
        ctx_omni, embeds->index,
        omni_llm_stage_timing_elapsed_ms(prefill_start, std::chrono::high_resolution_clock::now()));
    delete embeds;

    omni_llm_stage_log_prefill_done(ctx_omni);
    if (out_prefill_applied != nullptr) {
        *out_prefill_applied = true;
    }
    return true;
}

bool omni_llm_stage_scheduler_run_simplex_plan(struct omni_context *       ctx_omni,
                                               struct common_params *      params,
                                               OmniLlmStageSchedulerPlan & plan) {
    if (ctx_omni == nullptr || params == nullptr) {
        omni_llm_stage_scheduler_release_prefills(plan.prefills);
        return false;
    }

    if (!plan.prefills.empty()) {
        omni_llm_stage_log_prefill_activity(ctx_omni, params);
        if (!omni_llm_stage_scheduler_apply_simplex_prefills(ctx_omni, params, plan.prefills)) {
            return false;
        }
    }

    if (plan.decode_mode == OmniLlmStageSchedulerDecodeMode::once_after_prefill_drain) {
        return omni_llm_stage_scheduler_run_decode_cycle(ctx_omni);
    }

    return true;
}

bool omni_llm_stage_scheduler_run_duplex_plan(struct omni_context *       ctx_omni,
                                              struct common_params *      params,
                                              OmniLlmStageSchedulerPlan & plan) {
    if (ctx_omni == nullptr || params == nullptr) {
        omni_llm_stage_scheduler_release_prefills(plan.prefills);
        return false;
    }

    // Empty plan means there is no prefill work to consume.
    // In that case, only an explicit decode request can make progress.
    if (plan.prefills.empty()) {
        return plan.decode_mode == OmniLlmStageSchedulerDecodeMode::none ?
                   true :
                   omni_llm_stage_scheduler_run_decode_cycle(ctx_omni);
    }

    omni_llm_stage_log_prefill_activity(ctx_omni, params);

    for (size_t i = 0; i < plan.prefills.size(); ++i) {
        bool prefill_applied = false;
        if (!omni_llm_stage_scheduler_apply_duplex_prefill(ctx_omni, params, plan.prefills[i], &prefill_applied)) {
            omni_llm_stage_scheduler_release_prefills(plan.prefills, i + 1);
            return false;
        }

        // Duplex policy: after each successfully applied prefill, optionally run one
        // decode cycle before moving on. This keeps prefill/decode interleaving tight
        // and prevents one side from monopolizing the worker.
        if (plan.decode_mode == OmniLlmStageSchedulerDecodeMode::after_each_prefill && prefill_applied &&
            !omni_llm_stage_scheduler_run_decode_cycle(ctx_omni)) {
            omni_llm_stage_scheduler_release_prefills(plan.prefills, i + 1);
            return false;
        }
    }

    return true;
}

bool omni_llm_stage_scheduler_run_plan(struct omni_context *       ctx_omni,
                                       struct common_params *      params,
                                       OmniLlmStageSchedulerPlan & plan) {
    if (!omni_llm_stage_scheduler_has_work(plan)) {
        return true;
    }

    return ctx_omni->duplex_mode ? omni_llm_stage_scheduler_run_duplex_plan(ctx_omni, params, plan) :
                                   omni_llm_stage_scheduler_run_simplex_plan(ctx_omni, params, plan);
}
}  // namespace

// =============================================================================
// Worker Loop Entry
// The worker loop now only waits for pending work and hands it to the
// scheduler planning/execution layer above.
// =============================================================================

void omni_llm_stage_worker_loop(struct omni_context * ctx_omni, struct common_params * params) {
    if (ctx_omni == nullptr || ctx_omni->llm_thread_info == nullptr) {
        return;
    }

    print_with_timestamp("LLM thread started\n");

    while (ctx_omni->workers.llm_thread_running) {
        bool                         should_stop = false;
        OmniLlmStageSchedulerWorkset workset     = omni_llm_stage_scheduler_wait_work(ctx_omni, &should_stop);
        if (should_stop) {
            break;
        }

        OmniLlmStageSchedulerPlan plan = omni_llm_stage_scheduler_build_plan(ctx_omni, std::move(workset));
        if (!omni_llm_stage_scheduler_has_work(plan)) {
            continue;
        }

        ctx_omni->llm_thread_info->cv.notify_all();
        if (!omni_llm_stage_scheduler_run_plan(ctx_omni, params, plan)) {
            break;
        }
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
        print_with_timestamp("⚠️ Decode 结束滑窗检查: n_past=%d > n_ctx-reserved=%d，需要滑窗\n",
                             ctx_omni->session.n_past, n_ctx - reserved_space);
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

    print_with_timestamp("📍 为下一轮准备: eval <|im_end|>\\n<|im_start|>user\\n, n_past=%d\n",
                         ctx_omni->session.n_past);
}

// =============================================================================
// Decode Lifecycle Primitives
// These functions are the decode-side base for a future scheduler: explicit
// begin/slice/complete/publish/finish boundaries.
// =============================================================================

void omni_llm_stage_decode_begin(struct omni_context * ctx_omni, OmniLlmStageDecodeState * state) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || ctx_omni->params == nullptr || state == nullptr) {
        return;
    }

    *state = {};
    omni_llm_stage_apply_decode_prefix(ctx_omni, omni_llm_stage_build_decode_prefix(ctx_omni));

    state->max_tgt_len = ctx_omni->params->n_predict < 0 ? ctx_omni->params->n_ctx : ctx_omni->params->n_predict;
    state->llm_n_embd  = llama_model_n_embd(llama_get_model(ctx_omni->ctx_llama));

    print_with_timestamp("LLM decode: max_tgt_len = %d, n_predict = %d, n_ctx = %d\n", state->max_tgt_len,
                         ctx_omni->params->n_predict, ctx_omni->params->n_ctx);
}

bool omni_llm_stage_decode_slice(struct omni_context *     ctx_omni,
                                 OmniLlmStageDecodeState * state,
                                 OmniLlmStageDecodeSlice * out_slice) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || ctx_omni->params == nullptr || state == nullptr ||
        out_slice == nullptr) {
        return false;
    }

    *out_slice = {};

    if (state->llm_finish || state->generated_decode_tokens >= state->max_tgt_len) {
        return true;
    }

    if (ctx_omni->gate.break_event.load()) {
        state->llm_finish      = true;
        state->interrupted     = true;
        out_slice->interrupted = true;
        return true;
    }

    fflush(stdout);

    int       valid_chunk_count   = 0;
    const int max_chunk_tokens    = omni_llm_stage_max_chunk_tokens(ctx_omni);
    bool &    chunk_limit_reached = out_slice->chunk_limit_reached;
    chunk_limit_reached           = max_chunk_tokens > 0 && state->current_chunk_tokens >= max_chunk_tokens;

    while (valid_chunk_count < state->step_size && !state->llm_finish && !ctx_omni->gate.break_event.load() &&
           !chunk_limit_reached) {
        const char * tmp           = nullptr;
        float *      hidden_states = nullptr;
        llama_token  sampled_token = 0;
        const auto   step_start    = std::chrono::high_resolution_clock::now();

        {
            std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
            tmp = omni_llm_stage_loop_with_hidden_and_token(ctx_omni, ctx_omni->params, ctx_omni->ctx_sampler,
                                                            ctx_omni->session.n_past, hidden_states, sampled_token);
        }

        out_slice->total_tokens_generated++;
        const double step_ms = omni_llm_stage_timing_elapsed_ms(step_start, std::chrono::high_resolution_clock::now());

        if (tmp == nullptr) {
            free(hidden_states);
            LOG_ERR("llama_loop returned nullptr!");
            break;
        }

        int valid_token_count = 0;
        if (hidden_states != nullptr && omni_tts_is_valid_token(sampled_token)) {
            out_slice->chunk.token_ids.push_back(sampled_token);
            out_slice->chunk.hidden_states.insert(out_slice->chunk.hidden_states.end(), hidden_states,
                                                  hidden_states + state->llm_n_embd);
            valid_chunk_count++;
            valid_token_count = 1;
            state->current_chunk_tokens++;

            if (max_chunk_tokens > 0 && state->current_chunk_tokens >= max_chunk_tokens) {
                chunk_limit_reached = true;
            }
        }
        free(hidden_states);

        if (!state->llm_first_token_logged) {
            state->llm_first_token_logged = true;
        }

        const OmniTokenType token_type = omni_get_token_type(ctx_omni, sampled_token);
        omni_llm_stage_mark_decode_turn_end(ctx_omni, token_type, out_slice->chunk.is_end_of_turn);

        if (omni_is_end_token(ctx_omni, sampled_token)) {
            state->llm_finish = true;
            omni_llm_stage_handle_decode_end_token(ctx_omni, token_type);
            omni_llm_stage_note_decode_step_timing(ctx_omni, omni_llm_stage_active_duplex_chunk_idx(ctx_omni), step_ms,
                                                   1, valid_token_count, chunk_limit_reached,
                                                   ctx_omni->turn.ended_with_listen.load(), state->llm_finish, false);
            break;
        }

        out_slice->chunk.text += std::string(tmp);
        omni_llm_stage_note_decode_step_timing(ctx_omni, omni_llm_stage_active_duplex_chunk_idx(ctx_omni), step_ms, 1,
                                               valid_token_count, chunk_limit_reached,
                                               ctx_omni->turn.ended_with_listen.load(), state->llm_finish, false);
        fflush(stdout);
    }

    return true;
}

void omni_llm_stage_complete_decode_slice(struct omni_context *     ctx_omni,
                                          OmniLlmStageDecodeState * state,
                                          OmniLlmStageDecodeSlice * slice) {
    if (ctx_omni == nullptr || state == nullptr || slice == nullptr) {
        return;
    }

    if (slice->chunk_limit_reached) {
        omni_llm_stage_eval_chunk_eos_token(ctx_omni);
        state->llm_finish           = true;
        state->current_chunk_tokens = 0;
    }

    omni_llm_stage_eval_unit_end_token(ctx_omni);

    state->generated_decode_tokens += slice->total_tokens_generated;
    slice->chunk.llm_finish = state->llm_finish;
}

void omni_llm_stage_publish_decode_slice(struct omni_context *             ctx_omni,
                                         const OmniLlmStageDecodeRequest & request,
                                         const OmniLlmStageDecodeState &   state,
                                         const OmniLlmStageDecodeSlice &   slice) {
    if (ctx_omni == nullptr) {
        return;
    }

    OmniLlmStageDecodeChunk publish_chunk = slice.chunk;
    publish_chunk.llm_finish              = state.llm_finish;

    omni_llm_stage_strip_decode_special_tokens(publish_chunk.text);
    omni_llm_stage_publish_decode_response(ctx_omni, publish_chunk.text);
    omni_llm_stage_dispatch_decode_chunk_to_tts(ctx_omni, request, publish_chunk, state.llm_n_embd);
}

void omni_llm_stage_finish_decode(struct omni_context *           ctx_omni,
                                  const OmniLlmStageDecodeState & state,
                                  OmniLlmStageDecodeResult *      out_result) {
    if (ctx_omni == nullptr) {
        return;
    }

    omni_llm_stage_finish_decode_text_stream(ctx_omni);

    if (out_result != nullptr) {
        out_result->llm_finish              = state.llm_finish;
        out_result->interrupted             = state.interrupted;
        out_result->ended_with_listen       = ctx_omni->turn.ended_with_listen.load();
        out_result->generated_decode_tokens = state.generated_decode_tokens;
    }
}

// Current design:
// - This is still a compatibility wrapper that runs a full decode round by
//   looping over decode_slice() until finish/interruption.
// - It owns decode execution only; scheduler-side lifecycle concerns remain in
//   scheduler_run_decode_cycle().
// - Future attach-prefill scheduling should treat this as a legacy wrapper and
//   directly orchestrate decode_begin/decode_slice/complete/publish/finish.
bool omni_llm_stage_decode_run(struct omni_context *             ctx_omni,
                               const OmniLlmStageDecodeRequest & request,
                               OmniLlmStageDecodeResult *        out_result) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || ctx_omni->params == nullptr) {
        return false;
    }

    OmniLlmStageDecodeState state;
    omni_llm_stage_decode_begin(ctx_omni, &state);

    while (!state.llm_finish && state.generated_decode_tokens < state.max_tgt_len) {
        OmniLlmStageDecodeSlice slice;
        if (!omni_llm_stage_decode_slice(ctx_omni, &state, &slice)) {
            return false;
        }

        if (slice.interrupted) {
            break;
        }

        omni_llm_stage_complete_decode_slice(ctx_omni, &state, &slice);
        omni_llm_stage_publish_decode_slice(ctx_omni, request, state, slice);
    }

    omni_llm_stage_finish_decode(ctx_omni, state, out_result);
    return true;
}
