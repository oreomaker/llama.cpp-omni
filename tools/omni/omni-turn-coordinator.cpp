#include "omni-turn-coordinator.h"

#include "common/common.h"
#include "omni-llm-stage.h"
#include "omni-log.h"
#include "omni-session-state.h"
#include "omni.h"

#include <chrono>

static const char * omni_turn_close_kind_name(OmniTurnCloseKind kind) {
    switch (kind) {
        case OmniTurnCloseKind::finish:
            return "finish";
        case OmniTurnCloseKind::preempt:
            return "preempt";
        case OmniTurnCloseKind::abort:
            return "abort";
    }

    return "unknown";
}

static OmniTurnCloseResult omni_turn_coordinator_close_simplex_turn(struct omni_context * ctx_omni,
                                                                    OmniTurnCloseKind     kind) {
    OmniTurnCloseResult result;
    if (ctx_omni == nullptr) {
        return result;
    }

    result.completed_round = omni_session_round_meta(ctx_omni);
    result.active_round    = result.completed_round;
    result.interrupted     = kind != OmniTurnCloseKind::finish;

    if (kind == OmniTurnCloseKind::preempt) {
        result.turn_closed = false;
        return result;
    }

    if (ctx_omni->duplex_mode || ctx_omni->turn.current_turn_ended) {
        result.turn_closed = ctx_omni->turn.current_turn_ended;
        return result;
    }

    omni_llm_stage_finalize_decode_round(ctx_omni);
    ctx_omni->turn.current_turn_ended = true;

    result.turn_closed  = true;
    result.active_round = omni_session_make_round_meta(ctx_omni, result.completed_round.round_idx + 1);
    omni_session_set_round_meta(ctx_omni, result.active_round);

    print_with_timestamp("TurnCoordinator: %s simplex turn round_idx=%d -> %d\n", omni_turn_close_kind_name(kind),
                         result.completed_round.round_idx, result.active_round.round_idx);
    return result;
}

OmniPrefillSetup omni_turn_coordinator_prepare_prefill(struct omni_context * ctx_omni, int index) {
    OmniPrefillSetup setup;
    if (ctx_omni == nullptr) {
        return setup;
    }

    setup.need_bootstrap       = index == 0 && !ctx_omni->session.prompt.system_prompt_initialized;
    setup.should_start_workers = setup.need_bootstrap && ctx_omni->async;

    if (!(ctx_omni->use_tts && index == 0 && !ctx_omni->duplex_mode)) {
        return setup;
    }

    if (ctx_omni->warmup_done.load()) {
        if (ctx_omni->gate.break_event.load()) {
            print_with_timestamp("TTS: break_event active, skipping wait for previous round\n");
            ctx_omni->gate.speech_ready = true;
            ctx_omni->gate.break_event.store(false);
            ctx_omni->workers.speek_cv.notify_all();
        }

        print_with_timestamp("TTS: 等待上一轮语音生成完成\n");
        std::unique_lock<std::mutex> lock(ctx_omni->workers.speek_mtx);
        const bool                   wait_ok = ctx_omni->workers.speek_cv.wait_for(
            lock, std::chrono::seconds(5),
            [&] { return ctx_omni->gate.speech_ready || ctx_omni->gate.break_event.load(); });
        if (!wait_ok) {
            ctx_omni->gate.speech_ready = true;
        }
    }

    ctx_omni->gate.speech_ready = false;
    if (ctx_omni->tts_thread_info != nullptr) {
        omni_clear_tts_queue(ctx_omni,
                             ctx_omni->warmup_done.load() ? "stream_prefill: cleared TTS queue for new turn" : nullptr);
    }

    return setup;
}

void omni_turn_coordinator_prepare_decode(struct omni_context *       ctx_omni,
                                          int                         round_idx) {
    if (ctx_omni == nullptr) {
        return;
    }

    GGML_ASSERT(!ctx_omni->async);

    if (round_idx >= 0 && !ctx_omni->duplex_mode && ctx_omni->session.current_round.round_idx != round_idx) {
        print_with_timestamp("📍 [轮次同步] 调用方指定 round_idx=%d，当前 simplex_round_idx=%d，强制同步\n", round_idx,
                             ctx_omni->session.current_round.round_idx);
        omni_session_set_round_index(ctx_omni, round_idx);
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

OmniTurnCloseResult omni_turn_coordinator_close(struct omni_context * ctx_omni,
                                                OmniTurnCloseKind     kind,
                                                const char *          reason) {
    OmniTurnCloseResult result;
    if (ctx_omni == nullptr) {
        return result;
    }

    const bool turn_was_ended = ctx_omni->turn.current_turn_ended;

    if (kind == OmniTurnCloseKind::abort) {
        ctx_omni->gate.break_event.store(true);
        ctx_omni->turn.ended_with_listen.store(false);
        ctx_omni->gate.llm_generation_done.store(false);
        ctx_omni->gate.speech_ready = true;
        ctx_omni->workers.speek_cv.notify_all();

        omni_clear_tts_queue(ctx_omni);
        if (ctx_omni->tts_thread_info != nullptr) {
            ctx_omni->tts_thread_info->cv.notify_all();
        }
        if (ctx_omni->t2w_thread_info != nullptr) {
            ctx_omni->t2w_thread_info->cv.notify_all();
        }
    }

    if (!ctx_omni->duplex_mode && !turn_was_ended) {
        result = omni_turn_coordinator_close_simplex_turn(ctx_omni, kind);
    } else {
        result.completed_round = omni_session_round_meta(ctx_omni);
        result.active_round    = result.completed_round;
        result.interrupted     = kind != OmniTurnCloseKind::finish;
        if (kind == OmniTurnCloseKind::abort) {
            result.turn_closed           = true;
            ctx_omni->turn.current_turn_ended = true;
        } else if (kind == OmniTurnCloseKind::preempt) {
            result.turn_closed = false;
        } else {
            result.turn_closed = ctx_omni->turn.current_turn_ended;

            // Duplex also needs a per-turn round advance so downstream WAV naming
            // does not restart from wav_0 and overwrite the previous turn's audio.
            if (ctx_omni->duplex_mode && result.turn_closed) {
                result.active_round = omni_session_make_round_meta(ctx_omni, result.completed_round.round_idx + 1);
                omni_session_set_round_meta(ctx_omni, result.active_round);

                print_with_timestamp("TurnCoordinator: %s duplex turn round_idx=%d -> %d\n",
                                     omni_turn_close_kind_name(kind), result.completed_round.round_idx,
                                     result.active_round.round_idx);
            }
        }
    }

    print_with_timestamp(
        "TurnCoordinator: close(kind=%s, reason=%s, duplex_mode=%d, turn_was_ended=%d, round_idx=%d, "
        "next_round_idx=%d)\n",
        omni_turn_close_kind_name(kind), reason != nullptr ? reason : "none", ctx_omni->duplex_mode ? 1 : 0,
        turn_was_ended ? 1 : 0, result.completed_round.round_idx, result.active_round.round_idx);
    return result;
}
