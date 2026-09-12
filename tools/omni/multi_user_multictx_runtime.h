#pragma once

// Multi-user orchestration for independent LLM contexts or one shared
// continuous-batching engine. It does not use the duplex global queues or the
// process-wide SM scheduler. Model weights are owned by OmniSharedModelBundle.

#include "common.h"
#include "llama.h"
#include "omni.h"
#include "vision.h"
#include "audition.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct OmniSharedWeightInfo {
    std::string name;
    // Opaque owner address.  The public llama API does not expose individual
    // tensor buffer addresses, so this points at the shared model handle.
    const void * address = nullptr;
    size_t bytes = 0;
    std::string device;
};

// Shared component owners are optional. They are initialized only when the
// runtime configuration enables the corresponding stage.
class OmniSharedModelBundle {
public:
    static std::shared_ptr<OmniSharedModelBundle> load(const common_params & params,
                                                       std::string * error = nullptr);
    static std::shared_ptr<OmniSharedModelBundle> from_model(llama_model * model,
                                                             bool take_ownership = false);

    ~OmniSharedModelBundle();

    OmniSharedModelBundle(const OmniSharedModelBundle &) = delete;
    OmniSharedModelBundle & operator=(const OmniSharedModelBundle &) = delete;

    llama_model * llm_model() const { return model_; }
    struct vision_shared_model * vpm_model() const { return vpm_; }
    struct audition_shared_model * apm_model() const { return apm_; }
    llama_model * tts_model() const { return tts_model_; }
    int32_t tts_n_embd() const { return tts_model_ ? llama_model_n_embd(tts_model_) : 0; }
    bool has_vpm() const { return vpm_ != nullptr; }
    bool has_apm() const { return apm_ != nullptr; }
    bool has_tts() const { return tts_model_ != nullptr; }
    const struct omni_context * tts_weights() const { return tts_template_; }
    bool initialize_components(const common_params & params, const std::string & tts_bin_dir,
                               bool need_vpm, bool need_apm, bool need_tts,
                               std::string * error = nullptr);
    uint64_t model_bytes() const { return model_bytes_; }
    const std::vector<OmniSharedWeightInfo> & weights() const { return weights_; }

private:
    explicit OmniSharedModelBundle(llama_model * model, bool owns_model);

    llama_model * model_ = nullptr;
    bool owns_model_ = false;
    uint64_t model_bytes_ = 0;
    std::vector<OmniSharedWeightInfo> weights_;
    struct vision_shared_model * vpm_ = nullptr;
    struct audition_shared_model * apm_ = nullptr;
    llama_model * tts_model_ = nullptr;
    struct omni_context * tts_template_ = nullptr;
    std::string tts_bin_dir_;
};

enum class OmniLlmExecutionPolicy {
    PER_USER_CONTEXT,
    CONTINUOUS_BATCH,
};

struct OmniMultiUserConfig {
    uint32_t max_users = 1;
    uint32_t max_pending_frames_per_user = 8;
    uint32_t max_total_pending_frames = 0; // 0 = max_users * per-user limit
    bool enable_vpm = false;
    bool enable_apm = false;
    bool enable_tts = false;
    bool enable_t2w = false;
    std::string tts_bin_dir;
    bool frame_boundary_only = true;

    OmniLlmExecutionPolicy llm_execution_policy = OmniLlmExecutionPolicy::PER_USER_CONTEXT;
    uint32_t llm_max_batch_size = 0; // 0 = max_users
    uint32_t llm_max_batch_delay_us = 1000;
    uint32_t llm_sliding_window_high_tokens = 0; // 0 = derived from n_ctx_seq
    uint32_t llm_sliding_window_low_tokens = 0;  // 0 = derived from high water mark

    // Optional conservative admission budget.  Zero disables the check.
    uint64_t max_memory_bytes = 0;
    uint64_t per_user_memory_bytes = 0;

    // Maximum number of valid SPEAK text tokens per frame. Control tokens do
    // not consume this budget, matching duplex max_new_speak_tokens_per_chunk.
    uint32_t default_max_new_tokens = 26;
    uint32_t duplex_force_listen_count = 3;
};

enum class OmniSubmitStatus {
    ACCEPTED,
    USER_NOT_FOUND,
    USER_CLOSING,
    DUPLICATE_FRAME,
    OUT_OF_ORDER_FRAME,
    USER_QUEUE_FULL,
    RUNTIME_QUEUE_FULL,
    RESOURCE_EXHAUSTED,
    INVALID_TASK,
    RUNTIME_STOPPED,
};

enum class OmniUserCloseMode {
    DRAIN,
    CANCEL,
};

enum class OmniMultiUserEventKind {
    DECISION,
    TEXT,
    AUDIO_WINDOW,
    FRAME_DRAINED,
    FAILED,
    CANCELLED,
};

struct OmniUserSessionConfig {
    std::string voice_audio;
    std::string output_dir;
    uint64_t deadline_ns = 0;
};

struct OmniUserTask {
    uint64_t user_id = 0;
    uint64_t frame_id = 0;
    OmniDuplexFrame input;
    uint64_t deadline_ns = 0;
    uint64_t cancel_generation = 0; // filled by runtime; ignored from callers
    uint64_t submit_ns = 0;         // filled by runtime; ignored from callers

    // LLM-only baseline input.  Supplying tokens avoids tokenizer policy
    // differences in replay tests; text is tokenized when tokens is empty.
    std::vector<llama_token> tokens;
    std::vector<llama_token> replay_decode_tokens;
    std::string text;
    uint32_t max_new_tokens = 0; // 0 = runtime default

    // Preprocessed media for the shared LLM engine. The worker that owns the
    // per-user VPM/APM state fills these fields before enqueueing LLM prefill.
    std::vector<std::vector<float>> llm_vision_embeddings;
    std::vector<float> llm_audio_embeddings;
    int32_t llm_embedding_dim = 0;
    bool run_tts = false;
    bool run_t2w = false;
    std::vector<float> tts_embeddings;
    int32_t tts_embedding_tokens = 0;
    int32_t tts_embedding_dim = 0;
    std::vector<int32_t> audio_tokens;
    bool audio_tokens_final = true;
};

struct OmniMultiUserFrameTimings {
    double vpm_ms = 0.0;
    double apm_ms = 0.0;
    double llm_ms = 0.0;
    double tts_ms = 0.0;
    double t2w_ms = 0.0;
    double total_ms = 0.0;
    double frame_queue_wait_ms = 0.0;
    double llm_queue_wait_ms = 0.0;
    double llm_batch_wait_ms = 0.0;
    double llm_prefill_batch_wait_ms = 0.0;
    double llm_prefill_ms = 0.0;
    double llm_decode_ms = 0.0;
    double first_token_ms = 0.0;
    double first_audio_ms = 0.0;
};

struct OmniStageTimingStats {
    uint64_t count = 0;
    double total_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;

    double average_ms() const { return count ? total_ms / static_cast<double>(count) : 0.0; }
};

struct OmniMultiUserStageStats {
    uint64_t user_id = 0;
    OmniStageTimingStats vpm;
    OmniStageTimingStats apm;
    OmniStageTimingStats llm;
    OmniStageTimingStats tts;
    OmniStageTimingStats t2w;
    uint64_t completed_frames = 0;
    uint64_t speak_frames = 0;
    uint64_t listen_frames = 0;
};

enum class OmniLlmResultStatus {
    COMPLETE,
    CANCELLED,
    FAILED,
};

enum class OmniLlmFinishReason {
    NONE,
    LISTEN,
    CHUNK_EOS,
    CHUNK_TTS_EOS,
    TURN_EOS,
    TTS_EOS,
    EOS,
    MAX_NEW_TOKENS,
    MAX_DECODE_STEPS,
    DEADLINE,
    CANCELLED,
    ERROR,
};

struct OmniMultiUserEvent {
    uint64_t user_id = 0;
    uint64_t frame_id = 0;
    int32_t chunk_idx = -1;
    uint64_t cancel_generation = 0;
    OmniMultiUserEventKind kind = OmniMultiUserEventKind::FAILED;

    std::vector<llama_token> token_ids;
    std::string text;
    std::string error;
    std::string stage;
    std::vector<float> embedding;
    std::vector<int32_t> audio_token_ids;
    std::vector<float> pcm;
    bool is_speak = false;
    bool silent = false;
    bool is_end_of_turn = false;
    OmniLlmFinishReason finish_reason = OmniLlmFinishReason::NONE;
    OmniMultiUserFrameTimings timings;
    int32_t n_past_before = 0;
    int32_t n_past_after = 0;
    uint64_t owner_thread_id = 0;
    std::chrono::steady_clock::time_point service_begin{};
    std::chrono::steady_clock::time_point service_end{};
};

struct OmniMultiContextUserStats {
    uint64_t user_id = 0;
    const void * context_address = nullptr;
    const void * vpm_context_address = nullptr;
    const void * apm_context_address = nullptr;
    const void * tts_context_address = nullptr;
    const void * t2w_executor_address = nullptr;
    uint64_t owner_thread_id = 0;
    uint32_t n_ctx = 0;
    uint32_t n_ctx_seq = 0;
    uint32_t n_seq_max = 0;
    int32_t llm_seq_id = -1;
    int32_t n_past = 0;
    uint64_t kv_bytes_estimate = 0;
    uint64_t kv_bytes_used_estimate = 0;
    uint64_t sliding_event_count = 0;
    uint64_t sliding_dropped_tokens = 0;
    size_t llm_unit_count = 0;
    int32_t llm_current_turn = 0;
    uint64_t apm_kv_bytes = 0;
    bool has_vpm = false;
    bool has_apm = false;
    bool has_tts = false;
    bool has_t2w = false;
    size_t queue_depth = 0;
    bool active = false;
    bool closing = false;
};

struct OmniLlmRequestKey {
    uint64_t user_id = 0;
    uint64_t frame_id = 0;
    uint64_t cancel_generation = 0;
};

struct OmniLlmEngineRequest {
    OmniLlmRequestKey key;
    OmniUserTask task;
    std::chrono::steady_clock::time_point queue_enter{};
};

struct OmniLlmEngineResult {
    OmniLlmRequestKey key;
    OmniLlmResultStatus status = OmniLlmResultStatus::FAILED;
    OmniLlmFinishReason finish_reason = OmniLlmFinishReason::ERROR;
    std::vector<llama_token> token_ids;
    std::vector<float> hidden_states;
    std::string text;
    std::string error;
    std::string stage;
    bool is_speak = false;
    bool is_listen = false;
    // TURN_EOS/TTS_EOS/EOS ended the semantic turn, but not the current
    // duplex frame.  The final chunk uses this to flush TTS state.
    bool is_end_of_turn = false;
    int32_t n_past_before = 0;
    int32_t n_past_after = 0;
    uint64_t owner_thread_id = 0;
    uint32_t decode_steps = 0;
    uint32_t prefill_batch_size = 0;
    bool prefill_fallback = false;
    uint32_t decode_batch_count = 0;
    uint32_t decode_batch_size_max = 0;
    OmniMultiUserFrameTimings timings;
};

struct OmniLlmBatchStats {
    uint64_t prefill_count = 0;
    uint64_t prefill_dispatch_count = 0;
    uint64_t prefill_item_count = 0;
    uint64_t prefill_fallback_count = 0;
    uint64_t prefill_heterogeneous_fallback_count = 0;
    uint64_t prefill_input_tokens = 0;
    // Packed ragged input does not materialize padding. Kept for report compatibility.
    uint64_t prefill_padding_tokens = 0;
    uint32_t prefill_batch_size_max = 0;
    uint64_t decode_dispatch_count = 0;
    uint64_t decode_item_count = 0;
    uint64_t decode_fallback_count = 0;
    uint32_t decode_batch_size_max = 0;
    uint64_t mixed_dispatch_count = 0;
    uint64_t ubatch_count = 0;
    double prefill_service_ms = 0.0;
    double decode_service_ms = 0.0;
    double prefill_batch_wait_ms = 0.0;
    double prefill_batch_wait_max_ms = 0.0;
    double batch_wait_ms = 0.0;
    double batch_wait_max_ms = 0.0;
    uint64_t graph_hits = 0;
    uint64_t graph_misses = 0;
    uint64_t graph_captures = 0;
    uint64_t graph_property_misses = 0;
    bool graph_stats_available = false;
    uint64_t kv_tokens_used = 0;
    uint64_t kv_tokens_peak = 0;
    uint64_t kv_bytes_used_estimate = 0;
    uint64_t kv_bytes_peak_estimate = 0;
    uint64_t sliding_event_count = 0;
    uint64_t sliding_dropped_tokens = 0;

    double average_batch_size() const {
        return decode_dispatch_count
            ? static_cast<double>(decode_item_count) / static_cast<double>(decode_dispatch_count)
            : 0.0;
    }

    double average_prefill_batch_size() const {
        return prefill_dispatch_count
            ? static_cast<double>(prefill_item_count) / static_cast<double>(prefill_dispatch_count)
            : 0.0;
    }

    double prefill_padding_ratio() const {
        const uint64_t total = prefill_input_tokens + prefill_padding_tokens;
        return total ? static_cast<double>(prefill_padding_tokens) / static_cast<double>(total) : 0.0;
    }
};

class OmniLlmEngine {
public:
    virtual ~OmniLlmEngine() = default;

    virtual bool valid() const = 0;
    virtual const std::string & error() const = 0;
    virtual bool open_user(uint64_t user_id, const OmniUserSessionConfig & config) = 0;
    virtual bool prefill(const OmniLlmEngineRequest & request) = 0;
    virtual bool decode(const OmniLlmRequestKey & key) = 0;
    virtual bool poll(const OmniLlmRequestKey & key, OmniLlmEngineResult & result,
                      int timeout_ms = -1) = 0;
    virtual void cancel(uint64_t user_id, uint64_t cancel_generation) = 0;
    virtual bool drain(uint64_t user_id, int timeout_ms = -1) = 0;
    virtual void close_user(uint64_t user_id) = 0;
    virtual void shutdown() = 0;

    virtual const void * context_address() const = 0;
    virtual int32_t sequence_id(uint64_t user_id) const = 0;
    virtual int32_t n_past(uint64_t user_id) const = 0;
    virtual uint32_t n_ctx() const = 0;
    virtual uint32_t n_ctx_seq() const = 0;
    virtual uint32_t n_seq_max() const = 0;
    virtual uint64_t owner_thread_id() const = 0;
    virtual OmniLlmBatchStats batch_stats() const = 0;
    virtual bool sequence_stats(uint64_t, uint64_t & kv_bytes_used,
                                uint64_t & sliding_events, uint64_t & dropped_tokens,
                                size_t & unit_count, int32_t & current_turn) const {
        kv_bytes_used = 0;
        sliding_events = 0;
        dropped_tokens = 0;
        unit_count = 0;
        current_turn = 0;
        return false;
    }
};

class OmniMultiContextUserWorker;

class OmniMultiContextRuntime {
public:
    OmniMultiContextRuntime(std::shared_ptr<OmniSharedModelBundle> bundle,
                            const common_params & params,
                            OmniMultiUserConfig config = {},
                            std::shared_ptr<OmniLlmEngine> llm_engine = nullptr);
    ~OmniMultiContextRuntime();

    OmniMultiContextRuntime(const OmniMultiContextRuntime &) = delete;
    OmniMultiContextRuntime & operator=(const OmniMultiContextRuntime &) = delete;

    bool valid() const;
    const std::string & error() const { return error_; }

    bool open_user(uint64_t user_id, const OmniUserSessionConfig & config = {});
    OmniSubmitStatus submit(const OmniUserTask & task);
    bool wait_event(uint64_t user_id, OmniMultiUserEvent & event, int timeout_ms = -1);
    bool cancel_user(uint64_t user_id);
    bool close_user(uint64_t user_id, OmniUserCloseMode mode, int timeout_ms = -1);
    void shutdown();

    bool get_user_stats(uint64_t user_id, OmniMultiContextUserStats & stats) const;
    bool get_user_stage_stats(uint64_t user_id, OmniMultiUserStageStats & stats) const;
    OmniLlmBatchStats get_llm_batch_stats() const;
    uint32_t user_count() const;
    uint64_t model_bytes() const;
    uint64_t estimated_user_bytes() const;

private:
    struct UserState;

    void publish_event(const std::shared_ptr<UserState> & user, OmniMultiUserEvent event);
    void on_task_finished(const std::shared_ptr<UserState> & user);
    bool wait_user_idle(const std::shared_ptr<UserState> & user, int timeout_ms);

    std::shared_ptr<OmniSharedModelBundle> bundle_;
    common_params params_;
    OmniMultiUserConfig config_;
    std::shared_ptr<OmniLlmEngine> llm_engine_;
    mutable std::mutex mtx_;
    std::unordered_map<uint64_t, std::shared_ptr<UserState>> users_;
    std::atomic<size_t> total_pending_{0};
    std::atomic<bool> running_{true};
    std::string error_;
};

// The planning document uses both names while the API settles.  Keep the
// descriptive context name as the implementation and provide the shorter
// spelling for internal callers.
using OmniMultiUserRuntime = OmniMultiContextRuntime;
