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
// Frame model
// ---------------------------------------------------------------------------
// 一帧从 push 到 DONE 的全部状态挂在单个 DuplexFrame 上；阶段推进 = 帧在队列间
// 移动时同步更新 stage。取代旧的一帧散在多个 req 结构 + 隐式计数/栈指针。
//
//    push / stream_prefill → pending → [encoder_worker] → prefill → [llm_owner] → done
//
// 计算所有者只有两个：encoder_worker（VPM/APM 编码）、llm_owner（唯一写
// ctx_llama / n_past 的线程，严格一帧一次 prefill + 一次 decode）。
// TTS/T2W worker 不动，消费 LLMOut（带 frame_id）。
enum class FrameStage {
    QUEUED,     // 已入 pending，等待编码
    ENCODING,   // encoder_worker 正在 VPM/APM 编码
    PREFILLED,  // 编码完成，等待 llm_owner prefill+decode
    DECODING,   // llm_owner 采样中
    DONE,       // decode 完成，result 已填
    FAILED,     // encode/decode 失败，帧自我完成
};

// 单一帧记录。取代 OmniDuplexPendingFrame / DuplexEncodeReq /
// DuplexPrefillPacket / DuplexDecodeReq / OmniDuplexInflightFrame。
// 帧由 push_frame / stream_prefill 分配；session 帧完成后被推入 done_results 由
// wait_next_frame 消费并删除，stream 帧完成后由 duplex_decode 消费并删除。
struct DuplexFrame {
    uint64_t frame_id = 0;              // 内部递增 id
    int      index = 0;                 // 外部 prefill index（encoder/合同日志用）
    int      round_idx = -1;            // decode 的 round_idx（round sync + LLMOut.frame_id）
    OmniDuplexFrame input;              // aud_fname / img_fname / max_slice_nums / user_seq
    std::string debug_dir;              // LLMOut.debug_dir（TTS chunk 输出目录）
    FrameStage stage = FrameStage::QUEUED;

    // 编码结果（原 DuplexPrefillPacket）
    bool encode_ok = false;
    std::vector<std::vector<float>> vision_embed;   // [0]=overview, [1..]=slices
    std::vector<float> audio_embed;

    // 解码结果
    bool decode_ok = false;
    bool is_speak = false;              // 原 ended_with_listen 反推
    std::string text;                   // 本帧 SPEAK 文本（剔除控制 token）
    double t_first_token_ms = 0;        // TTFT：本帧首个 LLM 输出 token 的 epoch ms

    // 计时
    std::chrono::high_resolution_clock::time_point t_push;      // push_frame / stream_prefill
    std::chrono::high_resolution_clock::time_point t_prefilled; // prefill 提交到 encoder（≈push 后立即）
    double ms_decode = 0;               // llm_owner prefill+decode 耗时

    // 取消代数：帧创建时快照当前代；llm_owner 只处理代数匹配的帧。
    // 取代旧 break_event 的"丢弃当前帧"（当前代码没有任何调用方设置 break_event，
    // 该字段是给未来打断 API 预留的帧归属钩子）。
    uint64_t cancel_generation = 0;

    // 流式路径（server 直接调 stream_prefill/stream_decode）用：是否已允许 decode。
    // session 帧创建即置 true（llm_owner 立即消费）；stream 帧由 duplex_decode 置 true。
    std::atomic<bool> decode_requested{false};

    // 帧级完成句柄：stream_decode 等 done 时用；session 帧走 done_results 队列。
    std::atomic<bool> done{false};
    std::atomic<bool> ok{false};
    std::mutex done_mtx;
    std::condition_variable done_cv;

    // 该帧属于 session（push_frame 创建）还是流式路径（stream_prefill 创建）。
    bool session_frame = false;

    OmniDuplexFrameResult result;       // 最终 public 结果（DONE 时填充）
};

// ---------------------------------------------------------------------------
// Pipeline structures (frames flow: pending -> prefill -> decode)
// ---------------------------------------------------------------------------

// 帧管道：pending → (encoder_thread) → prefill → (llm_thread) → done。
// 队列里的元素都是 DuplexFrame*（阶段推进 = 帧换队列 + 更新 stage）。
// pending/prefill 队列随管道存活，由 duplex_stop_threads 在 join 线程后统一清理，
// 因此 encoder/llm 线程不依赖 session 生命周期，session_end 不会踩到它们。
struct DuplexPipeline {
    // ---------- control ----------
    std::atomic<bool> running{false};

    // ---------- pending: push_frame / stream_prefill → encoder_worker ----------
    std::queue<DuplexFrame *> pending_frames;
    std::mutex  pending_mtx;
    std::condition_variable pending_cv;
    static constexpr size_t PENDING_CAP = 64;

    // ---------- prefill: encoder_worker → llm_owner ----------
    std::queue<DuplexFrame *> prefill_queue;
    std::mutex  llm_mtx;
    std::condition_variable llm_cv;
    static constexpr size_t PREFILL_QUEUE_CAP = 32;

    // stream 路径的 decode 槽位：duplex_decode 从 prefill_queue 取下队头帧放入此槽，
    // llm_owner 取走处理。同一时刻最多一帧（配合 stream_decode_mtx 串行认领），
    // 替代旧 pending_decode 栈指针。受 llm_mtx 保护。
    DuplexFrame * pending_decode = nullptr;

    // ---------- workers ----------
    std::thread encoder_thread;
    std::thread llm_thread;

    // 流式路径（无 session）的 frame_id 来源；session push 用自身的 frame_id_counter。
    std::atomic<uint64_t> frame_id_counter{0};

    // 取消代数：帧创建时快照；外部打断时递增使旧帧失效（当前无打断 API，字段为钩子）。
    std::atomic<uint64_t> cancel_generation{0};

    // 串行化 stream_decode 的帧认领（同一时刻只处理一帧），替代旧 pending_decode 单槽。
    std::mutex stream_decode_mtx;
};

// 高层会话：push_frame → pending（在 pipeline）→ ... → done_results → wait_next_frame。
// 只保留结果队列、frame_id 计数与 in_flight；旧 prefill_worker/decode_worker 已删除，
// 管道由 encoder_worker + llm_owner 直接构成。会话内不跑线程，session_end 只排空等待。
struct DuplexSession {
    std::atomic<bool> running{false};
    std::string debug_dir;

    // done: llm_owner → wait_next_frame（持有完成的 DuplexFrame*，由 wait 侧 delete）
    std::queue<DuplexFrame *> done_results;
    std::mutex  done_mtx;
    std::condition_variable done_cv;

    std::atomic<int64_t> frame_id_counter{0};
    std::atomic<int>     in_flight{0};   // push 但未出 done_results 的帧数
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
// 签名与 omni.h 一致；内部 duplex 路径需要首 token 时刻时经 duplex_decode 的
// out_t_first_token_ms 传递，stream_decode 本身不暴露该参数。
bool stream_prefill(struct omni_context * ctx_omni, std::string aud_fname, std::string img_fname,
                    int index, int max_slice_nums, std::string text);
bool stream_decode(struct omni_context * ctx_omni, std::string debug_dir, int round_idx);
