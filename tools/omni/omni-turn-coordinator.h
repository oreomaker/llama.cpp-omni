#pragma once

#include "omni-worker-coordinator.h"

struct omni_context;

struct OmniTurnTransition {
    bool should_wait_for_tts = false;
    bool should_clear_tts_queue = false;
    bool should_request_prefill = false;
    bool should_wait_prefill_done = false;
    bool should_finalize_turn = false;
    bool interrupted = false;
};

struct OmniPrefillTurnDecision {
    OmniTurnTransition transition;
    bool need_bootstrap = false;
    bool should_start_workers = false;
};

OmniPrefillTurnDecision omni_turn_coordinator_begin_prefill(struct omni_context * ctx_omni, int index);
void omni_turn_coordinator_prepare_decode(
        struct omni_context * ctx_omni,
        int round_idx,
        const OmniWorkerThreadFns & worker_fns);
