#pragma once

// Duplex scheduling layer: frame orchestration, worker threads, and the
// high-level session.  This module owns "when and in what order" work runs;
// the actual model/graph execution lives in the compute layer (omni.cpp's
// LLM/TTS/encoder helpers).  Nothing here calls llama_decode directly except
// through the compute-layer functions declared in omni_internal.h.

#include "omni.h"
#include "omni_internal.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Pipeline structures (frames flow: encoder_queue -> prefill_queue -> decode)
// ---------------------------------------------------------------------------

struct DuplexEncodeReq {
    std::string aud_fname;
    std::string img_fname;
    int         index;
    int         max_slice_nums;  // -1 = 使用全局
};

struct DuplexPrefillPacket;

struct DuplexPrefillPacket {
    std::vector<std::vector<float>> vision_embed;  // [0]=overview, [1..]=slices
    std::vector<float>              audio_embed;
    int                             index = 0;
};

struct DuplexDecodeReq {
    std::string        debug_dir;
    int                round_idx = -1;
    std::atomic<bool>  done{false};
    std::atomic<bool>  ok{false};
    // 本帧 LLM 首个输出 token 的绝对 wall 时间（epoch ms）。由 duplex_llm_thread_func
    // 在采样循环首个有效 token 处设置，在 done 置 true 前写入（done 是 atomic，提供 happens-before）。
    double             t_first_token_ms = 0;
};

// 特殊 token 的 embedding 查表（fused prefill 用）。
// 在 duplex 路径启动后懒加载；加载失败时 initialized=false，
// fused prefill 路径回退到 per-chunk 多次 llama_decode 的老逻辑。
struct DuplexTokenEmbCache {
    bool initialized = false;
    int  hidden_size = 0;
    // text → embedding buffer（length = n_tokens * hidden_size）。
    // key 是 duplex prefill 里用到的固定字符串："<unit><image>" / "</image>" / "<slice>" / "</slice>" / "\n" / "<unit>"
    std::unordered_map<std::string, std::vector<float>> text_emb;
    std::unordered_map<std::string, int>                text_ntok;
};

struct DuplexPipeline {
    // ---------- control ----------
    std::atomic<bool> running{false};

    // ---------- encoder ----------
    std::thread encoder_thread;
    std::queue<DuplexEncodeReq *> encoder_queue;
    std::mutex  encoder_mtx;
    std::condition_variable encoder_cv;
    static constexpr size_t ENCODER_QUEUE_CAP = 16;

    // ---------- llm ----------
    std::thread llm_thread;
    std::queue<DuplexPrefillPacket *> prefill_queue;
    std::mutex  llm_mtx;
    std::condition_variable llm_cv;
    static constexpr size_t PREFILL_QUEUE_CAP = 32;

    DuplexDecodeReq * pending_decode = nullptr;
    std::condition_variable decode_done_cv;

    // 特殊 token embedding 查表（懒加载）
    DuplexTokenEmbCache emb_cache;

    // 已提交但尚未写入 ctx_llama KV 的 prefill 数量。
    // duplex_prefill 入口 +1；duplex_llm_thread_func 消费一个 packet 后 -1 并 notify。
    // duplex_decode 入口会等该计数归零，保证 decode 开始时上下文已完整。
    // 业务上 prefill/decode 严格交替（每秒 1 对），不会出现 race。
    std::atomic<int> in_flight_prefill{0};
    std::condition_variable in_flight_cv;
};

// High-level session: pending -> prefill_worker -> decode_pending -> decode_worker -> done_results
struct OmniDuplexPendingFrame {
    OmniDuplexFrame frame;
    int64_t frame_id;
    std::chrono::high_resolution_clock::time_point t_push;
    std::chrono::steady_clock::time_point t_push_steady;
};

struct OmniDuplexInflightFrame {
    int64_t frame_id;
    int64_t user_seq;
    std::chrono::high_resolution_clock::time_point t_push;
    std::chrono::steady_clock::time_point t_push_steady;
    std::chrono::high_resolution_clock::time_point t_prefilled;
    bool prefill_failed;
};

struct DuplexSession {
    std::atomic<bool> running{false};
    std::string debug_dir;

    // pending: push_frame -> prefill_worker
    std::queue<OmniDuplexPendingFrame> pending_frames;
    std::mutex pending_mtx;
    std::condition_variable pending_cv;
    static constexpr size_t PENDING_MAX = 64;

    // decode_pending: prefill_worker -> decode_worker
    std::queue<OmniDuplexInflightFrame> decode_pending;
    std::mutex decode_mtx;
    std::condition_variable decode_cv;

    // done: decode_worker -> wait_next_frame
    std::queue<OmniDuplexFrameResult> done_results;
    std::mutex done_mtx;
    std::condition_variable done_cv;

    std::atomic<int64_t> frame_id_counter{0};
    std::atomic<int>     in_flight{0};   // push 但还未出 done_results 的帧数

    std::thread prefill_worker;
    std::thread decode_worker;
};

// ---------------------------------------------------------------------------
// Duplex scheduling API (internal; called by omni.cpp init/free and the
// public stream_prefill / stream_decode routing).
// ---------------------------------------------------------------------------

bool duplex_prepare(struct omni_context * ctx_omni, common_params * params);
void duplex_start_threads(struct omni_context * ctx_omni, common_params * params);
void duplex_stop_threads(struct omni_context * ctx_omni);

bool duplex_prefill(struct omni_context * ctx_omni,
                    const std::string & aud_fname,
                    const std::string & img_fname,
                    int index, int max_slice_nums);
bool duplex_decode(struct omni_context * ctx_omni,
                   const std::string & debug_dir,
                   int round_idx,
                   double * out_t_first_token_ms = nullptr);

// Public routing entry points (declared in omni.h, implemented here).
bool stream_prefill(struct omni_context * ctx_omni, std::string aud_fname, std::string img_fname,
                    int index, int max_slice_nums, std::string text);
bool stream_decode(struct omni_context * ctx_omni, std::string debug_dir, int round_idx,
                   double * out_t_first_token_ms = nullptr);
