#pragma once

#include "omni-session-state.h"
#include "omni-worker-coordinator.h"

struct omni_context;

// Carries the setup decision before starting a prefill step.
struct OmniPrefillSetup {
    bool need_bootstrap       = false;
    bool should_start_workers = false;
};

enum class OmniTurnCloseKind {
    finish,
    abort,
};

// Reports the resulting turn/round state after a close operation.
struct OmniTurnCloseResult {
    bool          turn_closed = false;
    bool          interrupted = false;
    OmniRoundMeta completed_round;
    OmniRoundMeta active_round;
};

OmniPrefillSetup    omni_turn_coordinator_prepare_prefill(struct omni_context * ctx_omni, int index);
void                omni_turn_coordinator_prepare_decode(struct omni_context *       ctx_omni,
                                                         int                         round_idx,
                                                         const OmniWorkerThreadFns & worker_fns);
OmniTurnCloseResult omni_turn_coordinator_close(struct omni_context * ctx_omni,
                                                OmniTurnCloseKind     kind,
                                                const char *          reason = nullptr);
