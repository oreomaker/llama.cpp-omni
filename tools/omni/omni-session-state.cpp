#include "omni-session-state.h"

#include "omni.h"

void omni_session_sync_round_meta(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return;
    }

    ctx_omni->session.current_round.duplex_mode = ctx_omni->duplex_mode;
}

void omni_session_set_round_index(struct omni_context * ctx_omni, int round_idx) {
    if (ctx_omni == nullptr) {
        return;
    }

    ctx_omni->simplex_round_idx = round_idx;
    ctx_omni->wav_turn_base = round_idx * 1000;
    omni_session_sync_round_meta(ctx_omni);
}

OmniRoundMeta omni_session_round_meta(const struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return {};
    }

    OmniRoundMeta round = ctx_omni->session.current_round;
    round.duplex_mode = ctx_omni->duplex_mode;
    return round;
}
