#include "common.h"
#include "omni-session-state.h"
#include "omni-turn-coordinator.h"
#include "omni.h"

#include <cstdio>

#undef NDEBUG
#include <cassert>

namespace {

struct OmniTestFixture {
    common_params params;
    omni_context  ctx;

    OmniTestFixture() {
        params.n_ctx = 4096;
        ctx.params   = &params;
    }
};

void test_prepare_decode_resets_duplex_state() {
    OmniTestFixture fixture;

    fixture.ctx.async = false;
    fixture.ctx.duplex_mode = true;
    fixture.ctx.use_tts = true;
    fixture.ctx.turn.current_turn_ended = true;
    fixture.ctx.turn.ended_with_listen.store(true);
    fixture.ctx.turn.decode_prefix_applied = true;
    fixture.ctx.turn.generated_decode_tokens = 23;
    fixture.ctx.turn.current_chunk_tokens = 9;
    fixture.ctx.gate.break_event.store(true);
    fixture.ctx.gate.text_done = true;
    fixture.ctx.gate.text_streaming = false;
    fixture.ctx.gate.speech_ready = true;
    fixture.ctx.text_queue.push_back("stale");
    omni_session_set_round_index(&fixture.ctx, 2);

    omni_turn_coordinator_prepare_decode(&fixture.ctx, -1);

    assert(fixture.ctx.session.current_round.round_idx == 2);
    assert(fixture.ctx.turn.current_turn_ended == false);
    assert(fixture.ctx.turn.ended_with_listen.load() == false);
    assert(fixture.ctx.turn.decode_prefix_applied == false);
    assert(fixture.ctx.turn.generated_decode_tokens == 0);
    assert(fixture.ctx.turn.current_chunk_tokens == 0);
    assert(fixture.ctx.gate.break_event.load() == false);
    assert(fixture.ctx.gate.text_done == false);
    assert(fixture.ctx.gate.text_streaming == true);
    assert(fixture.ctx.gate.speech_ready == false);
    assert(fixture.ctx.text_queue.empty());
}

void test_prepare_decode_syncs_simplex_round_index() {
    OmniTestFixture fixture;

    fixture.ctx.async = false;
    fixture.ctx.duplex_mode = false;
    fixture.ctx.use_tts = false;
    omni_session_set_round_index(&fixture.ctx, 1);

    omni_turn_coordinator_prepare_decode(&fixture.ctx, 7);

    assert(fixture.ctx.session.current_round.round_idx == 7);
    assert(fixture.ctx.session.current_round.wav_turn_base == 7000);
    assert(fixture.ctx.session.current_round.duplex_mode == false);
    assert(fixture.ctx.turn.decode_prefix_applied == false);
    assert(fixture.ctx.turn.generated_decode_tokens == 0);
}

void test_close_duplex_finish_advances_round() {
    OmniTestFixture fixture;

    fixture.ctx.duplex_mode = true;
    omni_session_set_round_index(&fixture.ctx, 3);
    fixture.ctx.turn.current_turn_ended = true;

    const OmniTurnCloseResult result =
        omni_turn_coordinator_close(&fixture.ctx, OmniTurnCloseKind::finish, "unit-test");

    assert(result.interrupted == false);
    assert(result.turn_closed == true);
    assert(result.completed_round.round_idx == 3);
    assert(result.active_round.round_idx == 4);
    assert(fixture.ctx.session.current_round.round_idx == 4);
    assert(fixture.ctx.session.current_round.wav_turn_base == 4000);
}

void test_close_duplex_abort_marks_break_and_closes_turn() {
    OmniTestFixture fixture;

    fixture.ctx.duplex_mode = true;
    fixture.ctx.turn.current_turn_ended = false;
    fixture.ctx.turn.ended_with_listen.store(true);
    fixture.ctx.gate.break_event.store(false);
    fixture.ctx.gate.llm_generation_done.store(true);
    fixture.ctx.gate.speech_ready = false;

    const OmniTurnCloseResult result =
        omni_turn_coordinator_close(&fixture.ctx, OmniTurnCloseKind::abort, "unit-test");

    assert(result.interrupted == true);
    assert(result.turn_closed == true);
    assert(fixture.ctx.turn.current_turn_ended == true);
    assert(fixture.ctx.turn.ended_with_listen.load() == false);
    assert(fixture.ctx.gate.break_event.load() == true);
    assert(fixture.ctx.gate.llm_generation_done.load() == false);
    assert(fixture.ctx.gate.speech_ready == true);
}

void test_close_duplex_preempt_keeps_turn_open() {
    OmniTestFixture fixture;

    fixture.ctx.duplex_mode = true;
    fixture.ctx.turn.current_turn_ended = false;
    fixture.ctx.turn.ended_with_listen.store(true);
    fixture.ctx.gate.break_event.store(false);
    fixture.ctx.gate.llm_generation_done.store(true);
    fixture.ctx.gate.speech_ready = false;
    omni_session_set_round_index(&fixture.ctx, 9);

    const OmniTurnCloseResult result =
        omni_turn_coordinator_close(&fixture.ctx, OmniTurnCloseKind::preempt, "unit-test");

    assert(result.interrupted == true);
    assert(result.turn_closed == false);
    assert(result.completed_round.round_idx == 9);
    assert(result.active_round.round_idx == 9);
    assert(fixture.ctx.turn.current_turn_ended == false);
    assert(fixture.ctx.turn.ended_with_listen.load() == true);
    assert(fixture.ctx.gate.break_event.load() == false);
    assert(fixture.ctx.gate.llm_generation_done.load() == true);
    assert(fixture.ctx.gate.speech_ready == false);
    assert(fixture.ctx.session.current_round.round_idx == 9);
}

void test_session_round_meta_helpers_keep_wav_base_in_sync() {
    OmniTestFixture fixture;

    fixture.ctx.duplex_mode = true;
    omni_session_set_round_index(&fixture.ctx, 5);

    const OmniRoundMeta round = omni_session_round_meta(&fixture.ctx);
    assert(round.round_idx == 5);
    assert(round.wav_turn_base == 5000);
    assert(round.duplex_mode == true);
}

}  // namespace

int main() {
    test_prepare_decode_resets_duplex_state();
    test_prepare_decode_syncs_simplex_round_index();
    test_close_duplex_finish_advances_round();
    test_close_duplex_abort_marks_break_and_closes_turn();
    test_close_duplex_preempt_keeps_turn_open();
    test_session_round_meta_helpers_keep_wav_base_in_sync();

    std::printf("test-turn-coordinator: all tests passed\n");
    return 0;
}
