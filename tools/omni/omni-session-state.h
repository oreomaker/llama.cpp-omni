#pragma once

#include "llama.h"

#include <atomic>
#include <string>
#include <vector>

struct omni_context;

struct SlidingWindowConfig {
    // 滑窗模式: "off" / "basic" / "context"
    // - "off": 禁用滑窗
    // - "basic": 基础滑窗（按 cache 长度触发）
    // - "context": 带 context 的滑窗（保留生成文本到 previous）
    std::string mode = "off";

    // 基础滑窗参数
    int high_water_tokens = 4000;  // 高水位线：超过此值触发滑窗
    int low_water_tokens = 3500;   // 低水位线：滑窗后保留到此值

    // RoPE 参数
    float rope_theta = 10000.0f;   // RoPE base frequency
};

struct UnitEntry {
    int unit_id = -1;
    int length = 0;
    std::string type;
    std::vector<llama_token> generated_tokens;
    bool is_listen = false;
};

struct OmniRoundMeta {
    int round_idx = 0;
    int wav_turn_base = 0;
    bool duplex_mode = false;
};

struct OmniPromptState {
    int n_keep = 0;
    bool system_prompt_initialized = false;
    int system_preserve_length = 0;
};

struct OmniSessionState {
    int n_past = 0;
    OmniPromptState prompt;
    std::vector<int> round_start_positions;
    int max_preserved_context = 2048;
    SlidingWindowConfig sliding_window_config;
    std::vector<UnitEntry> unit_history;
    int next_unit_id = 0;
    int pending_unit_id = -1;
    int pending_unit_start_cache_len = 0;
    int position_offset = 0;
    int sliding_event_count = 0;
    int total_dropped_tokens = 0;
    int total_dropped_units = 0;
    OmniRoundMeta current_round;
};

struct OmniTurnState {
    bool current_turn_ended = true;
    std::atomic<bool> ended_with_listen{false};
    bool decode_prefix_applied = false;
    int  generated_decode_tokens = 0;
    int  current_chunk_tokens = 0;
};

struct OmniSessionGate {
    volatile bool speech_ready = true;
    bool text_streaming = false;
    bool text_done = false;
    std::atomic<bool> break_event{false};
    std::atomic<bool> session_stop_event{false};
    std::atomic<bool> llm_generation_done{false};
};

OmniRoundMeta omni_session_make_round_meta(const struct omni_context * ctx_omni, int round_idx);
void omni_session_sync_round_meta(struct omni_context * ctx_omni);
void omni_session_set_round_meta(struct omni_context * ctx_omni, const OmniRoundMeta & round_meta);
void omni_session_set_round_index(struct omni_context * ctx_omni, int round_idx);
OmniRoundMeta omni_session_round_meta(const struct omni_context * ctx_omni);
