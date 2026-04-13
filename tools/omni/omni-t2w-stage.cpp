#include "omni-t2w-stage.h"

#include "omni-impl.h"
#include "omni-output.h"
#include "omni-python-t2w.h"
#include "omni-session-state.h"
#include "omni.h"
#include "token2wav/token2wav-impl.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

void print_with_timestamp(const char * format, ...);

namespace {

constexpr int32_t OMNI_T2W_CHUNK_SIZE    = 25;
constexpr int32_t OMNI_T2W_PRE_LOOKAHEAD = 3;
constexpr int32_t OMNI_T2W_WINDOW_SIZE   = OMNI_T2W_CHUNK_SIZE + OMNI_T2W_PRE_LOOKAHEAD;

struct OmniT2WStageState {
    std::vector<int32_t> token_buffer = { 4218, 4218, 4218 };
    OmniRoundMeta        active_round_meta;
    int                  last_round_idx = 0;
    std::string          tts_wav_output_dir;
    int                  wav_idx = 0;
};

struct OmniT2WBatch {
    std::vector<llama_token> new_tokens;
    bool                     is_final       = false;
    bool                     is_chunk_end   = false;
    bool                     has_round_meta = false;
    OmniRoundMeta            round_meta;
    int                      duplex_chunk_idx = -1;
};

struct OmniT2WBackendHooks {
    const char *                                                    log_prefix                       = "T2W";
    bool                                                            preserve_duplex_chunk_boundaries = false;
    bool                                                            reset_buffer_on_round_change     = false;
    bool                                                            note_duplex_timing               = false;
    bool                                                            log_done_flag                    = false;
    std::function<void(struct omni_context *, OmniT2WStageState &)> on_break;
    std::function<void(struct omni_context *, OmniT2WStageState &)> on_empty_final;
    std::function<void(struct omni_context *, OmniT2WStageState &)> on_final_flush;
    std::function<bool(struct omni_context *)>                      is_ready;
    std::function<
        bool(struct omni_context *, const OmniT2WStageState &, const std::vector<int32_t> &, bool, double &, double &)>
        process_window;
};

void omni_t2w_note_timing(struct omni_context * ctx_omni, int chunk_idx, double ms, int window_count, bool done) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.token2wav_ms                = timing.token2wav_ms < 0.0 ? ms : timing.token2wav_ms + ms;
    timing.token2wav_window_count += window_count;
    if (done) {
        timing.token2wav_done = true;
    }
}

std::vector<int32_t> omni_t2w_initial_buffer() {
    return { 4218, 4218, 4218 };
}

void omni_t2w_reset_local_state(OmniT2WStageState & state) {
    state.token_buffer = omni_t2w_initial_buffer();
    state.wav_idx      = 0;
}

bool omni_t2w_round_meta_changed(const OmniRoundMeta & lhs, const OmniRoundMeta & rhs) {
    return lhs.round_idx != rhs.round_idx || lhs.wav_turn_base != rhs.wav_turn_base ||
           lhs.duplex_mode != rhs.duplex_mode;
}

void omni_t2w_clear_queue(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || ctx_omni->t2w_thread_info == nullptr) {
        return;
    }

    auto &                      queue = ctx_omni->t2w_thread_info->queue;
    auto &                      mtx   = ctx_omni->t2w_thread_info->mtx;
    std::lock_guard<std::mutex> lock(mtx);
    while (!queue.empty()) {
        T2WOut * t2w_out = queue.front();
        queue.pop();
        delete t2w_out;
    }
}

void omni_t2w_init_state(struct omni_context * ctx_omni, OmniT2WStageState & state) {
    state.active_round_meta = omni_session_round_meta(ctx_omni);
    state.last_round_idx    = state.active_round_meta.round_idx;
    omni_ensure_round_tts_wav_output_dir(ctx_omni->base_output_dir, state.active_round_meta, &state.tts_wav_output_dir);
}

bool omni_t2w_handle_break(struct omni_context *       ctx_omni,
                           OmniT2WStageState &         state,
                           const OmniT2WBackendHooks & hooks) {
    if (ctx_omni == nullptr || !ctx_omni->break_event.load()) {
        return false;
    }

    omni_t2w_clear_queue(ctx_omni);
    ctx_omni->break_event.store(false);
    omni_t2w_reset_local_state(state);
    if (hooks.on_break) {
        hooks.on_break(ctx_omni, state);
    }
    return true;
}

bool omni_t2w_collect_batch(struct omni_context * ctx_omni,
                            bool                  preserve_duplex_chunk_boundaries,
                            OmniT2WBatch &        batch) {
    if (ctx_omni == nullptr || ctx_omni->t2w_thread_info == nullptr) {
        return false;
    }

    auto & queue = ctx_omni->t2w_thread_info->queue;
    auto & mtx   = ctx_omni->t2w_thread_info->mtx;
    auto & cv    = ctx_omni->t2w_thread_info->cv;

    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock,
            [&] { return !queue.empty() || !ctx_omni->workers.t2w_thread_running || ctx_omni->break_event.load(); });

    if (!ctx_omni->workers.t2w_thread_running && queue.empty()) {
        return false;
    }

    if (ctx_omni->break_event.load()) {
        return true;
    }

    while (!queue.empty()) {
        T2WOut * t2w_out = queue.front();
        if (preserve_duplex_chunk_boundaries && ctx_omni->duplex_mode && batch.duplex_chunk_idx >= 0 &&
            t2w_out->duplex_chunk_idx >= 0 && t2w_out->duplex_chunk_idx != batch.duplex_chunk_idx) {
            break;
        }

        queue.pop();
        if (batch.duplex_chunk_idx < 0 && t2w_out->duplex_chunk_idx >= 0) {
            batch.duplex_chunk_idx = t2w_out->duplex_chunk_idx;
        }

        batch.new_tokens.insert(batch.new_tokens.end(), t2w_out->audio_tokens.begin(), t2w_out->audio_tokens.end());
        batch.is_final       = batch.is_final || t2w_out->is_final;
        batch.is_chunk_end   = batch.is_chunk_end || t2w_out->is_chunk_end;
        batch.round_meta     = t2w_out->round_meta;
        batch.has_round_meta = true;
        delete t2w_out;
    }

    return true;
}

void omni_t2w_update_round_state(struct omni_context *       ctx_omni,
                                 OmniT2WStageState &         state,
                                 const OmniT2WBatch &        batch,
                                 const OmniT2WBackendHooks & hooks) {
    if (ctx_omni == nullptr) {
        return;
    }

    if (batch.has_round_meta && omni_t2w_round_meta_changed(batch.round_meta, state.active_round_meta)) {
        state.active_round_meta = batch.round_meta;
    }

    if (!state.active_round_meta.duplex_mode && state.active_round_meta.round_idx != state.last_round_idx) {
        print_with_timestamp("%s: round_idx 变化 %d -> %d（来自T2WOut），更新输出目录\n", hooks.log_prefix,
                             state.last_round_idx, state.active_round_meta.round_idx);
        state.last_round_idx = state.active_round_meta.round_idx;
        omni_ensure_round_tts_wav_output_dir(ctx_omni->base_output_dir, state.active_round_meta,
                                             &state.tts_wav_output_dir);
        state.wav_idx = 0;
        if (hooks.reset_buffer_on_round_change) {
            state.token_buffer = omni_t2w_initial_buffer();
        }
    }
}

bool omni_t2w_need_flush(const struct omni_context * ctx_omni, const OmniT2WBatch & batch) {
    if (ctx_omni == nullptr) {
        return false;
    }

    if (!ctx_omni->duplex_mode) {
        return batch.is_final || batch.is_chunk_end;
    }

    return batch.is_final;
}

std::string omni_t2w_next_wav_path(const OmniT2WStageState & state) {
    return state.tts_wav_output_dir + "/wav_" + std::to_string(state.active_round_meta.wav_turn_base + state.wav_idx) +
           ".wav";
}

void omni_t2w_log_wav_result(struct omni_context *       ctx_omni,
                             const OmniT2WBackendHooks & hooks,
                             const OmniT2WStageState &   state,
                             double                      inference_time_ms,
                             double                      audio_duration) {
    if (ctx_omni == nullptr || audio_duration <= 0.0) {
        return;
    }

    const auto wav_complete_time = std::chrono::high_resolution_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(wav_complete_time - ctx_omni->stream_decode_start_time)
            .count();

    if (state.wav_idx == 0) {
        print_with_timestamp("🎉 首响时间 (First Audio Response): %lldms\n", (long long) elapsed_ms);
    }

    const float rtf = (float) (inference_time_ms / 1000.0) / (float) audio_duration;
    print_with_timestamp("%s: wav_%d.wav | %.2fs audio | %.1fms inference | RTF=%.2f | t=%lldms\n", hooks.log_prefix,
                         state.active_round_meta.wav_turn_base + state.wav_idx, audio_duration, inference_time_ms, rtf,
                         (long long) elapsed_ms);
}

void omni_t2w_slide_token_buffer(struct omni_context *  ctx_omni,
                                 std::vector<int32_t> & token_buffer,
                                 bool                   is_last_window) {
    size_t slide_amount = 0;

    if (ctx_omni == nullptr) {
        token_buffer.clear();
        return;
    }

    if (!ctx_omni->duplex_mode) {
        if (token_buffer.size() > (size_t) OMNI_T2W_CHUNK_SIZE) {
            slide_amount = OMNI_T2W_CHUNK_SIZE;
        } else {
            token_buffer.clear();
            return;
        }
    } else if (is_last_window) {
        slide_amount = token_buffer.size();
    } else if (token_buffer.size() > (size_t) OMNI_T2W_CHUNK_SIZE) {
        slide_amount = OMNI_T2W_CHUNK_SIZE;
    } else if (token_buffer.size() > (size_t) OMNI_T2W_PRE_LOOKAHEAD) {
        slide_amount = token_buffer.size() - OMNI_T2W_PRE_LOOKAHEAD;
    }

    if (slide_amount == 0) {
        return;
    }

    if (slide_amount >= token_buffer.size()) {
        token_buffer.clear();
        return;
    }

    token_buffer.erase(token_buffer.begin(), token_buffer.begin() + slide_amount);
}

void omni_t2w_run_backend(struct omni_context * ctx_omni, const OmniT2WBackendHooks & hooks) {
    OmniT2WStageState state;
    omni_t2w_init_state(ctx_omni, state);

    while (ctx_omni->workers.t2w_thread_running) {
        if (omni_t2w_handle_break(ctx_omni, state, hooks)) {
            continue;
        }

        OmniT2WBatch batch;
        if (!omni_t2w_collect_batch(ctx_omni, hooks.preserve_duplex_chunk_boundaries, batch)) {
            break;
        }

        if (ctx_omni->break_event.load()) {
            continue;
        }

        if (batch.new_tokens.empty() && !batch.is_chunk_end && !batch.is_final) {
            continue;
        }

        omni_t2w_update_round_state(ctx_omni, state, batch, hooks);
        state.token_buffer.insert(state.token_buffer.end(), batch.new_tokens.begin(), batch.new_tokens.end());

        if (!hooks.is_ready || !hooks.is_ready(ctx_omni)) {
            if (hooks.note_duplex_timing) {
                omni_t2w_note_timing(ctx_omni, batch.duplex_chunk_idx, 0.0, 0, batch.is_final || batch.is_chunk_end);
            }
            continue;
        }

        const bool need_flush = omni_t2w_need_flush(ctx_omni, batch);
        if (batch.is_final && state.token_buffer.empty()) {
            if (hooks.on_empty_final) {
                hooks.on_empty_final(ctx_omni, state);
            }
            if (hooks.note_duplex_timing) {
                omni_t2w_note_timing(ctx_omni, batch.duplex_chunk_idx, 0.0, 0, true);
            }
            continue;
        }

        double t2w_ms_total     = 0.0;
        int    t2w_window_count = 0;

        while (state.token_buffer.size() >= (size_t) OMNI_T2W_WINDOW_SIZE ||
               (need_flush && !state.token_buffer.empty())) {
            const size_t process_size   = std::min(state.token_buffer.size(), (size_t) OMNI_T2W_WINDOW_SIZE);
            const bool   is_last_window = need_flush && (state.token_buffer.size() <= (size_t) OMNI_T2W_WINDOW_SIZE);

            std::vector<int32_t> window(state.token_buffer.begin(), state.token_buffer.begin() + process_size);
            double               inference_time_ms = 0.0;
            double               audio_duration    = 0.0;

            if (hooks.process_window &&
                hooks.process_window(ctx_omni, state, window, is_last_window, inference_time_ms, audio_duration)) {
                t2w_ms_total += inference_time_ms;
                t2w_window_count++;
                if (audio_duration > 0.0) {
                    omni_t2w_log_wav_result(ctx_omni, hooks, state, inference_time_ms, audio_duration);
                    state.wav_idx++;
                }
            } else {
                LOG_ERR("%s: process 失败\n", hooks.log_prefix);
            }

            omni_t2w_slide_token_buffer(ctx_omni, state.token_buffer, is_last_window);

            if (is_last_window) {
                if (batch.is_final) {
                    const int last_wav_idx =
                        state.wav_idx > 0 ? state.active_round_meta.wav_turn_base + state.wav_idx - 1 : 0;
                    if (omni_write_generation_done_flag(state.tts_wav_output_dir, last_wav_idx) &&
                        hooks.log_done_flag) {
                        print_with_timestamp("%s: 写入结束标记 %s/generation_done.flag (last_wav=%d)\n",
                                             hooks.log_prefix, state.tts_wav_output_dir.c_str(), last_wav_idx);
                    }
                    omni_t2w_reset_local_state(state);
                    if (hooks.on_final_flush) {
                        hooks.on_final_flush(ctx_omni, state);
                    }
                }
                break;
            }
        }

        if (hooks.note_duplex_timing) {
            omni_t2w_note_timing(ctx_omni, batch.duplex_chunk_idx, t2w_ms_total, t2w_window_count,
                                 batch.is_final || batch.is_chunk_end);
        }
    }
}

bool omni_t2w_process_python_window(struct omni_context *        ctx_omni,
                                    const OmniT2WStageState &    state,
                                    const std::vector<int32_t> & window,
                                    bool                         is_last_window,
                                    double &                     inference_time_ms,
                                    double &                     audio_duration) {
    return omni_process_python_t2w_tokens(ctx_omni, window, is_last_window, omni_t2w_next_wav_path(state),
                                          inference_time_ms, audio_duration);
}

bool omni_t2w_process_cpp_window(struct omni_context *        ctx_omni,
                                 const OmniT2WStageState &    state,
                                 const std::vector<int32_t> & window,
                                 bool                         is_last_window,
                                 double &                     inference_time_ms,
                                 double &                     audio_duration) {
    if (ctx_omni == nullptr || !ctx_omni->token2wav_session) {
        return false;
    }

    const auto         t2w_start = std::chrono::high_resolution_clock::now();
    std::vector<float> chunk_wav;
    if (!ctx_omni->token2wav_session->feed_window(window, is_last_window, chunk_wav)) {
        return false;
    }

    const auto t2w_end = std::chrono::high_resolution_clock::now();
    inference_time_ms  = std::chrono::duration<double, std::milli>(t2w_end - t2w_start).count();

    if (chunk_wav.empty()) {
        audio_duration = 0.0;
        return true;
    }

    if (!omni_write_wav_file_f32_mono_s16(omni_t2w_next_wav_path(state), chunk_wav,
                                          omni::flow::Token2Wav::kSampleRate)) {
        audio_duration = 0.0;
        return true;
    }

    audio_duration = chunk_wav.size() / (double) omni::flow::Token2Wav::kSampleRate;
    return true;
}

void t2w_thread_func_python(struct omni_context * ctx_omni, struct common_params * params) {
    (void) params;
    print_with_timestamp("T2W thread (Python) started\n");
    fflush(stdout);

    OmniT2WBackendHooks hooks;
    hooks.log_prefix                       = "T2W(Python)";
    hooks.preserve_duplex_chunk_boundaries = false;
    hooks.reset_buffer_on_round_change     = false;
    hooks.note_duplex_timing               = false;
    hooks.log_done_flag                    = false;
    hooks.on_break                         = [](struct omni_context * ctx, OmniT2WStageState &) {
        omni_reset_python_t2w_cache(ctx);
    };
    hooks.on_empty_final = [](struct omni_context * ctx, OmniT2WStageState &) {
        omni_reset_python_t2w_cache(ctx);
    };
    hooks.on_final_flush = [](struct omni_context * ctx, OmniT2WStageState &) {
        omni_reset_python_t2w_cache(ctx);
    };
    hooks.is_ready = [](struct omni_context * ctx) {
        return ctx != nullptr && ctx->python_t2w_initialized;
    };
    hooks.process_window = omni_t2w_process_python_window;

    omni_t2w_run_backend(ctx_omni, hooks);

    print_with_timestamp("T2W(Python) 线程: 停止\n");
    fflush(stdout);
}

void t2w_thread_func_cpp(struct omni_context * ctx_omni, struct common_params * params) {
    (void) params;
    print_with_timestamp("T2W thread (C++) started\n");
    fflush(stdout);

    OmniT2WBackendHooks hooks;
    hooks.log_prefix                       = "T2W线程";
    hooks.preserve_duplex_chunk_boundaries = true;
    hooks.reset_buffer_on_round_change     = true;
    hooks.note_duplex_timing               = true;
    hooks.log_done_flag                    = true;
    hooks.is_ready                         = [](struct omni_context * ctx) {
        return ctx != nullptr && ctx->token2wav_initialized && ctx->token2wav_session != nullptr;
    };
    hooks.process_window = omni_t2w_process_cpp_window;

    omni_t2w_run_backend(ctx_omni, hooks);

    print_with_timestamp("T2W(C++) 线程: 停止\n");
    fflush(stdout);
}

}  // namespace

void t2w_thread_func(struct omni_context * ctx_omni, struct common_params * params) {
    if (ctx_omni != nullptr && ctx_omni->use_python_token2wav) {
        t2w_thread_func_python(ctx_omni, params);
        return;
    }

    t2w_thread_func_cpp(ctx_omni, params);
}
