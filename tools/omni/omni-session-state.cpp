#include "omni-session-state.h"

#include "omni.h"

OmniRoundMeta omni_session_make_round_meta(const struct omni_context * ctx_omni, int round_idx) {
    OmniRoundMeta round;
    round.round_idx = round_idx;
    round.wav_turn_base = round_idx * 1000;
    round.duplex_mode = ctx_omni != nullptr ? ctx_omni->duplex_mode : false;
    return round;
}

void omni_session_sync_round_meta(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return;
    }

    ctx_omni->session.current_round.wav_turn_base = ctx_omni->session.current_round.round_idx * 1000;
    ctx_omni->session.current_round.duplex_mode = ctx_omni->duplex_mode;
}

void omni_session_set_round_meta(struct omni_context * ctx_omni, const OmniRoundMeta & round_meta) {
    if (ctx_omni == nullptr) {
        return;
    }

    ctx_omni->session.current_round = round_meta;
    omni_session_sync_round_meta(ctx_omni);
}

void omni_session_set_round_index(struct omni_context * ctx_omni, int round_idx) {
    if (ctx_omni == nullptr) {
        return;
    }

    omni_session_set_round_meta(ctx_omni, omni_session_make_round_meta(ctx_omni, round_idx));
}

OmniRoundMeta omni_session_round_meta(const struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return {};
    }

    OmniRoundMeta round = ctx_omni->session.current_round;
    round.wav_turn_base = round.round_idx * 1000;
    round.duplex_mode = ctx_omni->duplex_mode;
    return round;
}
