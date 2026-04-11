#include "omni-turn-coordinator.h"

#include "omni-session-state.h"
#include "omni.h"
#include "common/common.h"

#include <chrono>

void print_with_timestamp(const char * format, ...);

OmniPrefillTurnDecision omni_turn_coordinator_begin_prefill(struct omni_context * ctx_omni, int index) {
    OmniPrefillTurnDecision decision;
    if (ctx_omni == nullptr) {
        return decision;
    }

    decision.need_bootstrap = index == 0 && !ctx_omni->system_prompt_initialized;
    decision.should_start_workers = decision.need_bootstrap && ctx_omni->async;

    if (!(ctx_omni->use_tts && index == 0 && !ctx_omni->duplex_mode)) {
        return decision;
    }

    decision.transition.should_wait_for_tts = true;
    decision.transition.should_clear_tts_queue = true;

    if (ctx_omni->warmup_done.load()) {
        if (ctx_omni->break_event.load()) {
            print_with_timestamp("TTS: break_event active, skipping wait for previous round\n");
            ctx_omni->speek_done = true;
            ctx_omni->break_event.store(false);
            ctx_omni->workers.speek_cv.notify_all();
            decision.transition.interrupted = true;
        }

        print_with_timestamp("TTS: 等待上一轮语音生成完成\n");
        std::unique_lock<std::mutex> lock(ctx_omni->workers.speek_mtx);
        const bool wait_ok = ctx_omni->workers.speek_cv.wait_for(lock, std::chrono::seconds(5), [&] {
            return ctx_omni->speek_done || ctx_omni->break_event.load();
        });
        if (!wait_ok) {
            ctx_omni->speek_done = true;
        }
    }

    ctx_omni->speek_done = false;
    if (ctx_omni->tts_thread_info != nullptr) {
        omni_clear_tts_queue(
            ctx_omni,
            ctx_omni->warmup_done.load() ? "stream_prefill: cleared TTS queue for new turn" : nullptr);
    }

    return decision;
}

void omni_turn_coordinator_prepare_decode(
        struct omni_context * ctx_omni,
        int round_idx,
        const OmniWorkerThreadFns & worker_fns) {
    if (ctx_omni == nullptr) {
        return;
    }

    if (round_idx >= 0 && !ctx_omni->duplex_mode && ctx_omni->simplex_round_idx != round_idx) {
        print_with_timestamp("📍 [轮次同步] 调用方指定 round_idx=%d，当前 simplex_round_idx=%d，强制同步\n",
                             round_idx, ctx_omni->simplex_round_idx);
        omni_session_set_round_index(ctx_omni, round_idx);
    } else {
        omni_session_sync_round_meta(ctx_omni);
    }

    ctx_omni->stream_decode_start_time = std::chrono::high_resolution_clock::now();
    print_with_timestamp("📍 stream_decode 开始: n_past=%d, n_keep=%d, n_ctx=%d, duplex_mode=%d\n",
                         ctx_omni->n_past, ctx_omni->n_keep, ctx_omni->params->n_ctx, ctx_omni->duplex_mode);

    if (!ctx_omni->duplex_mode) {
        ctx_omni->llm_generation_done.store(false);
    }
    ctx_omni->ended_with_listen.store(false);

    if (ctx_omni->duplex_mode && ctx_omni->break_event.load()) {
        ctx_omni->break_event.store(false);
        print_with_timestamp("📍 stream_decode: reset break_event from true to false\n");
    }

    {
        std::lock_guard<std::mutex> lock(ctx_omni->text_mtx);
        ctx_omni->text_queue.clear();
        ctx_omni->text_done_flag = false;
        ctx_omni->text_streaming = true;
    }

    if (ctx_omni->async) {
        omni_ensure_decode_workers_started(ctx_omni, worker_fns);
        omni_request_prefill(ctx_omni);
        print_with_timestamp("wait prefill done\n");
        omni_wait_for_prefill_completion(ctx_omni);
        omni_reset_prefill_completion(ctx_omni);
    }

    if (ctx_omni->use_tts) {
        ctx_omni->speek_done = false;
    }
}
