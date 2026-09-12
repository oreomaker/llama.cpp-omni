#include "multi_user_multictx_runtime.h"
#include "multi_user_llm_engine.h"

#include "common.h"
#include "duplex_core.h"
#include "../../src/llama-ext.h"
#include "omni_internal.h"
#include "sampling.h"
#include "token2wav/token2wav.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <unordered_set>
#include <utility>

namespace {

uint64_t thread_id_value() {
    return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

uint64_t steady_now_ns() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

double elapsed_ms_from_ns(uint64_t begin_ns) {
    if (begin_ns == 0) {
        return 0.0;
    }
    const uint64_t now_ns = steady_now_ns();
    return now_ns >= begin_ns ? static_cast<double>(now_ns - begin_ns) / 1.0e6 : 0.0;
}

bool deadline_expired(uint64_t deadline_ns) {
    if (deadline_ns == 0) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    return now_ns >= deadline_ns;
}

llama_token find_special_token(const llama_vocab * vocab, const char * text) {
    if (!vocab) {
        return -1;
    }
    llama_token one[4] = {};
    const int n = llama_tokenize(vocab, text, static_cast<int32_t>(std::strlen(text)),
                                 one, 4, false, true);
    if (n == 1) {
        return one[0];
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < n_vocab; ++i) {
        char piece[128] = {};
        const int len = llama_token_to_piece(vocab, i, piece, sizeof(piece), 0, true);
        if (len > 0 && len < static_cast<int>(sizeof(piece)) &&
                std::strncmp(piece, text, static_cast<size_t>(len)) == 0 &&
                text[static_cast<size_t>(len)] == '\0') {
            return i;
        }
    }
    return -1;
}

void initialize_duplex_tokens(omni_context * ctx, llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    ctx->special_token_speak         = find_special_token(vocab, "<|speak|>");
    ctx->special_token_listen        = find_special_token(vocab, "<|listen|>");
    ctx->special_token_chunk_eos     = find_special_token(vocab, "<|chunk_eos|>");
    ctx->special_token_chunk_tts_eos = find_special_token(vocab, "<|chunk_tts_eos|>");
    ctx->special_token_turn_eos      = find_special_token(vocab, "<|turn_eos|>");
    ctx->special_token_tts_eos       = find_special_token(vocab, "<|tts_eos|>");
    ctx->special_token_eos           = llama_vocab_eos(vocab);
    ctx->tts_bos_token_id            = find_special_token(vocab, "<|tts_bos|>");
    ctx->special_token_unit_end      = find_special_token(vocab, "</unit>");
    ctx->special_token_tts_pad       = find_special_token(vocab, "<|tts_pad|>");
    ctx->slide_last_was_listen       = true;
    ctx->current_turn_ended          = true;
}

OmniLlmFinishReason finish_reason_for_token(OmniTokenType type) {
    switch (type) {
        case OmniTokenType::LISTEN:        return OmniLlmFinishReason::LISTEN;
        case OmniTokenType::CHUNK_EOS:     return OmniLlmFinishReason::CHUNK_EOS;
        case OmniTokenType::CHUNK_TTS_EOS: return OmniLlmFinishReason::CHUNK_TTS_EOS;
        case OmniTokenType::TURN_EOS:      return OmniLlmFinishReason::TURN_EOS;
        case OmniTokenType::TTS_EOS:       return OmniLlmFinishReason::TTS_EOS;
        case OmniTokenType::EOS:           return OmniLlmFinishReason::EOS;
        default:                           return OmniLlmFinishReason::NONE;
    }
}

} // namespace

class OmniContinuousBatchLlmEngine final : public OmniLlmEngine {
public:
    OmniContinuousBatchLlmEngine(std::shared_ptr<OmniSharedModelBundle> bundle,
                                 const common_params & params,
                                 const OmniMultiUserConfig & config)
        : bundle_(std::move(bundle)), params_(params), config_(config) {
        if (!bundle_ || !bundle_->llm_model()) {
            error_ = "shared LLM model is unavailable";
            return;
        }

        llama_context_params cparams = common_context_params_to_llama(params_);
        const uint32_t n_ctx_per_user = cparams.n_ctx > 0
            ? cparams.n_ctx
            : static_cast<uint32_t>(std::max(1, llama_model_n_ctx_train(bundle_->llm_model())));
        if (n_ctx_per_user > std::numeric_limits<uint32_t>::max() / config_.max_users) {
            error_ = "shared LLM context size overflows uint32_t";
            return;
        }
        n_ctx_per_user_ = n_ctx_per_user;
        cparams.n_ctx = n_ctx_per_user * config_.max_users;
        cparams.n_seq_max = config_.max_users;
        cparams.n_batch = std::max<uint32_t>(cparams.n_batch, max_batch_size());
        cparams.n_ubatch = std::min(cparams.n_batch,
                                    std::max<uint32_t>(1, cparams.n_ubatch));
        cparams.kv_unified = true;
        ctx_ = llama_init_from_model(bundle_->llm_model(), cparams);
        if (!ctx_) {
            error_ = "failed to create shared llama_context";
            return;
        }
        n_ctx_ = llama_n_ctx(ctx_);
        n_ctx_seq_ = n_ctx_per_user_;
        n_seq_max_ = llama_n_seq_max(ctx_);
        const uint64_t n_embd = static_cast<uint64_t>(
            std::max(1, llama_model_n_embd(bundle_->llm_model())));
        const uint64_t n_layer = static_cast<uint64_t>(
            std::max(1, llama_model_n_layer(bundle_->llm_model())));
        kv_bytes_per_token_ = n_embd * n_layer * 2u * sizeof(float);
        if (n_seq_max_ < config_.max_users) {
            error_ = "shared llama_context has fewer sequence slots than max_users";
            llama_free(ctx_);
            ctx_ = nullptr;
            return;
        }
        embedding_ctx_ = std::make_unique<omni_context>();
        embedding_ctx_->ctx_llama = ctx_;
        embedding_ctx_->model = bundle_->llm_model();
        embedding_ctx_->params = &params_;
        if (!duplex_init_emb_cache_for_runtime(embedding_ctx_.get(), &params_)) {
            error_ = "failed to initialize shared token embedding lookup";
            embedding_ctx_.reset();
            llama_free(ctx_);
            ctx_ = nullptr;
            return;
        }
        for (uint32_t seq_id = 0; seq_id < config_.max_users; ++seq_id) {
            free_seq_ids_.push_back(static_cast<llama_seq_id>(seq_id));
        }
        valid_ = true;
        thread_ = std::thread(&OmniContinuousBatchLlmEngine::run, this);
    }

    ~OmniContinuousBatchLlmEngine() override {
        shutdown();
        for (auto & entry : users_) {
            if (entry.second->sampler) {
                common_sampler_free(entry.second->sampler);
            }
        }
        users_.clear();
        embedding_ctx_.reset();
        if (ctx_) {
            llama_free(ctx_);
            ctx_ = nullptr;
        }
    }

    bool valid() const override {
        return valid_ && ctx_ && !stopped_.load(std::memory_order_acquire);
    }

    const std::string & error() const override {
        return error_;
    }

    bool open_user(uint64_t user_id, const OmniUserSessionConfig & session_config) override {
        if (!valid() || user_id == 0 || !session_config.voice_audio.empty()) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mtx_);
        if (users_.find(user_id) != users_.end() || free_seq_ids_.empty()) {
            return false;
        }
        auto state = std::make_unique<SequenceState>();
        state->user_id = user_id;
        state->seq_id = free_seq_ids_.front();
        free_seq_ids_.pop_front();
        state->sampler = common_sampler_init(bundle_->llm_model(), params_.sampling);
        if (!state->sampler) {
            free_seq_ids_.push_front(state->seq_id);
            return false;
        }
        state->duplex = std::make_unique<omni_context>();
        state->duplex->ctx_llama = ctx_;
        state->duplex->model = bundle_->llm_model();
        state->duplex->ctx_sampler = state->sampler;
        state->duplex->params = &params_;
        state->duplex->duplex_mode = true;
        state->duplex->async = false;
        initialize_duplex_tokens(state->duplex.get(), bundle_->llm_model());
        state->duplex->force_listen_count = static_cast<int>(config_.duplex_force_listen_count);
        state->duplex->sliding_window_config.mode = "turn";
        state->duplex->sliding_window_config.high_water_tokens = static_cast<int>(sliding_high_water());
        state->duplex->sliding_window_config.low_water_tokens = static_cast<int>(sliding_low_water());
        users_.emplace(user_id, std::move(state));
        return true;
    }

    bool prefill(const OmniLlmEngineRequest & request) override {
        if (!valid() || request.key.user_id == 0 || request.key.frame_id == 0) {
            return false;
        }
        auto job = std::make_shared<Job>();
        job->request = request;
        job->result.key = request.key;
        job->result.status = OmniLlmResultStatus::FAILED;
        job->result.finish_reason = OmniLlmFinishReason::ERROR;
        job->queue_enter = request.queue_enter.time_since_epoch().count() == 0
            ? std::chrono::steady_clock::now() : request.queue_enter;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            auto it = users_.find(request.key.user_id);
            if (it == users_.end() || it->second->job || it->second->failed ||
                    close_requested_.count(request.key.user_id)) {
                return false;
            }
            it->second->job = job;
            prefill_queue_.push_back(job);
        }
        cv_.notify_all();
        return true;
    }

    bool decode(const OmniLlmRequestKey & key) override {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = users_.find(key.user_id);
        if (it == users_.end() || !it->second->job || !same_key(it->second->job->request.key, key)) {
            return false;
        }
        it->second->job->decode_requested = true;
        if (it->second->job->prefilled && !it->second->job->completed) {
            activate_locked(*it->second);
        }
        cv_.notify_all();
        return true;
    }

    bool poll(const OmniLlmRequestKey & key, OmniLlmEngineResult & result,
              int timeout_ms) override {
        std::unique_lock<std::mutex> lock(mtx_);
        auto find_job = [&]() -> std::shared_ptr<Job> {
            auto it = users_.find(key.user_id);
            if (it == users_.end() || !it->second->job ||
                    !same_key(it->second->job->request.key, key)) {
                return nullptr;
            }
            return it->second->job;
        };
        std::shared_ptr<Job> job = find_job();
        if (!job) {
            return false;
        }
        auto ready = [&] { return job->completed || stopped_.load(std::memory_order_acquire); };
        if (timeout_ms < 0) {
            cv_.wait(lock, ready);
        } else if (timeout_ms == 0) {
            if (!ready()) {
                return false;
            }
        } else if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready)) {
            return false;
        }
        if (!job->completed) {
            return false;
        }
        result = std::move(job->result);
        auto it = users_.find(key.user_id);
        if (it != users_.end() && it->second->job == job) {
            it->second->job.reset();
        }
        cv_.notify_all();
        return true;
    }

    void cancel(uint64_t user_id, uint64_t cancel_generation) override {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = users_.find(user_id);
        if (it == users_.end()) {
            return;
        }
        it->second->cancel_generation = cancel_generation;
        if (it->second->job && !it->second->job->completed) {
            it->second->job->cancelled = true;
        }
        cv_.notify_all();
    }

    bool drain(uint64_t user_id, int timeout_ms) override {
        std::unique_lock<std::mutex> lock(mtx_);
        auto ready = [&] {
            auto it = users_.find(user_id);
            return it == users_.end() || !it->second->job;
        };
        if (timeout_ms < 0) {
            cv_.wait(lock, ready);
            return true;
        }
        return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready);
    }

    void close_user(uint64_t user_id) override {
        std::unique_lock<std::mutex> lock(mtx_);
        auto it = users_.find(user_id);
        if (it == users_.end()) {
            return;
        }
        close_requested_.insert(user_id);
        cv_.notify_all();
        cv_.wait(lock, [&] { return users_.find(user_id) == users_.end(); });
    }

    void shutdown() override {
        if (stopped_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    const void * context_address() const override { return ctx_; }

    int32_t sequence_id(uint64_t user_id) const override {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = users_.find(user_id);
        return it == users_.end() ? -1 : it->second->seq_id;
    }

    int32_t n_past(uint64_t user_id) const override {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = users_.find(user_id);
        return it == users_.end() ? 0 : it->second->n_past.load(std::memory_order_acquire);
    }

    uint32_t n_ctx() const override { return n_ctx_; }
    uint32_t n_ctx_seq() const override { return n_ctx_seq_; }
    uint32_t n_seq_max() const override { return n_seq_max_; }
    uint64_t owner_thread_id() const override {
        return owner_thread_id_.load(std::memory_order_acquire);
    }

    OmniLlmBatchStats batch_stats() const override {
        std::lock_guard<std::mutex> lock(stats_mtx_);
        return batch_stats_;
    }

    bool sequence_stats(uint64_t user_id, uint64_t & kv_bytes_used,
                        uint64_t & sliding_events, uint64_t & dropped_tokens,
                        size_t & unit_count, int32_t & current_turn) const override {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = users_.find(user_id);
        if (it == users_.end()) {
            return false;
        }
        const SequenceState & state = *it->second;
        kv_bytes_used = static_cast<uint64_t>(state.n_past.load(std::memory_order_acquire)) *
                        kv_bytes_per_token_;
        sliding_events = static_cast<uint64_t>(state.duplex->sliding_event_count);
        dropped_tokens = static_cast<uint64_t>(state.duplex->total_dropped_tokens);
        unit_count = state.duplex->unit_history.size();
        current_turn = state.duplex->current_turn_id;
        return true;
    }

private:
    struct PrefillPlan {
        std::vector<float> embeddings;
        size_t token_count = 0;
        std::string input_type;
    };

    struct Job {
        OmniLlmEngineRequest request;
        OmniLlmEngineResult result;
        std::chrono::steady_clock::time_point queue_enter{};
        std::chrono::steady_clock::time_point token_ready{};
        llama_token pending_token = LLAMA_TOKEN_NULL;
        uint32_t valid_tokens = 0;
        size_t replay_index = 0;
        uint32_t max_tokens = 0;
        uint32_t max_decode_steps = 0;
        bool decode_requested = false;
        bool prefilled = false;
        bool active = false;
        bool cancelled = false;
        bool completed = false;
        bool pending_is_unit_end = false;
        bool saw_speak = false;
        common_sampler * sampler_checkpoint = nullptr;
        int32_t rollback_n_past = 0;
        int32_t rollback_n_keep = 0;
        int32_t force_listen_used_before = 0;
        size_t rollback_unit_history_size = 0;
        size_t rollback_round_count = 0;
        int32_t rollback_next_unit_id = 0;
        int32_t rollback_current_turn_id = 0;
        bool rollback_session_initialized = false;
        bool slide_last_was_listen_before = true;
        bool current_turn_ended_before = true;
        bool ended_with_listen_before = false;
        int32_t frame_start_n_past = 0;
        int32_t decode_start_n_past = 0;
    };

    struct SequenceState {
        uint64_t user_id = 0;
        llama_seq_id seq_id = -1;
        common_sampler * sampler = nullptr;
        std::unique_ptr<omni_context> duplex;
        std::shared_ptr<Job> job;
        std::atomic<int32_t> n_past{0};
        int32_t n_keep = 0;
        uint64_t cancel_generation = 0;
        bool session_initialized = false;
        bool failed = false;
    };

    struct BatchItem {
        SequenceState * state = nullptr;
        std::shared_ptr<Job> job;
        std::vector<float> embeddings;
        std::string input_type;
        int32_t position = 0;
        int32_t batch_last = -1;
        bool prefill = false;
    };

    static bool same_key(const OmniLlmRequestKey & lhs, const OmniLlmRequestKey & rhs) {
        return lhs.user_id == rhs.user_id && lhs.frame_id == rhs.frame_id &&
               lhs.cancel_generation == rhs.cancel_generation;
    }

    uint32_t max_batch_size() const {
        const uint32_t configured = config_.llm_max_batch_size
            ? config_.llm_max_batch_size : config_.max_users;
        return std::max<uint32_t>(1, std::min(configured, config_.max_users));
    }

    uint32_t sliding_high_water() const {
        if (config_.llm_sliding_window_high_tokens > 0) {
            return std::min(config_.llm_sliding_window_high_tokens, n_ctx_seq_);
        }
        const uint32_t reserve = std::min<uint32_t>(512, std::max<uint32_t>(32, n_ctx_seq_ / 4));
        return std::max<uint32_t>(1, n_ctx_seq_ - reserve);
    }

    uint32_t sliding_low_water() const {
        const uint32_t high = sliding_high_water();
        if (config_.llm_sliding_window_low_tokens > 0) {
            return std::min(config_.llm_sliding_window_low_tokens, high);
        }
        return high > 1 ? std::max<uint32_t>(1, high * 3 / 4) : high;
    }

    void update_kv_stats_locked() {
        uint64_t tokens = 0;
        for (const auto & entry : users_) {
            tokens += static_cast<uint64_t>(
                std::max<int32_t>(0, entry.second->n_past.load(std::memory_order_acquire)));
        }
        std::lock_guard<std::mutex> stats_lock(stats_mtx_);
        batch_stats_.kv_tokens_used = tokens;
        batch_stats_.kv_tokens_peak = std::max(batch_stats_.kv_tokens_peak, tokens);
        batch_stats_.kv_bytes_used_estimate = tokens * kv_bytes_per_token_;
        batch_stats_.kv_bytes_peak_estimate = std::max(
            batch_stats_.kv_bytes_peak_estimate, batch_stats_.kv_bytes_used_estimate);
    }

    void update_kv_stats() {
        std::lock_guard<std::mutex> lock(mtx_);
        update_kv_stats_locked();
    }

    struct GraphStatsSnapshot {
        bool valid = false;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t captures = 0;
        uint64_t property_misses = 0;
    };

    GraphStatsSnapshot graph_stats_snapshot() const {
        GraphStatsSnapshot result;
#ifdef GGML_USE_CUDA
        ggml_backend_t backend = llama_get_gpu_backend(ctx_);
        ggml_backend_cuda_graph_stats stats = {};
        if (backend && ggml_backend_is_cuda(backend) &&
                ggml_backend_cuda_get_graph_stats(backend, &stats)) {
            result.valid = true;
            result.hits = stats.hits;
            result.misses = stats.misses;
            result.captures = stats.captures;
            result.property_misses = stats.property_misses;
        }
#endif
        return result;
    }

    void record_graph_stats(const GraphStatsSnapshot & before,
                            const GraphStatsSnapshot & after) {
        if (!before.valid || !after.valid) {
            return;
        }
        std::lock_guard<std::mutex> lock(stats_mtx_);
        batch_stats_.graph_stats_available = true;
        batch_stats_.graph_hits += after.hits - before.hits;
        batch_stats_.graph_misses += after.misses - before.misses;
        batch_stats_.graph_captures += after.captures - before.captures;
        batch_stats_.graph_property_misses += after.property_misses - before.property_misses;
    }

    bool job_cancelled(const SequenceState & state, const Job & job) const {
        return job.cancelled || state.cancel_generation != job.request.key.cancel_generation;
    }

    void activate_locked(SequenceState & state) {
        if (!state.job || state.job->active || state.job->completed) {
            return;
        }
        state.job->active = true;
        active_users_.push_back(state.user_id);
    }

    void remove_active_locked(uint64_t user_id) {
        active_users_.erase(std::remove(active_users_.begin(), active_users_.end(), user_id),
                            active_users_.end());
    }

    bool restore_sequence_position(SequenceState & state, int32_t position) {
        llama_memory_t memory = llama_get_memory(ctx_);
        if (!memory) {
            return false;
        }
        const llama_pos current = llama_memory_seq_pos_max(memory, state.seq_id) + 1;
        if (current < position) {
            return false;
        }
        if (current > position && !llama_memory_seq_rm(memory, state.seq_id, position, -1)) {
            return false;
        }
        const llama_pos restored = llama_memory_seq_pos_max(memory, state.seq_id) + 1;
        if (restored != position) {
            return false;
        }
        state.n_past = position;
        return true;
    }

    void complete_locked(SequenceState & state, OmniLlmResultStatus status,
                         OmniLlmFinishReason reason, const std::string & error = {}) {
        if (!state.job || state.job->completed) {
            return;
        }
        Job & job = *state.job;
        std::string result_error = error;
        if (status != OmniLlmResultStatus::COMPLETE && job.sampler_checkpoint) {
            const bool restored = restore_sequence_position(state, job.rollback_n_past);
            if (!restored) {
                status = OmniLlmResultStatus::FAILED;
                reason = OmniLlmFinishReason::ERROR;
                result_error = "failed to restore shared LLM sequence after: " + error;
                state.failed = true;
            } else {
                state.n_keep = job.rollback_n_keep;
                state.session_initialized = job.rollback_session_initialized;
                state.duplex->force_listen_used = job.force_listen_used_before;
                state.duplex->slide_last_was_listen = job.slide_last_was_listen_before;
                state.duplex->current_turn_ended = job.current_turn_ended_before;
                state.duplex->ended_with_listen = job.ended_with_listen_before;
                state.duplex->next_unit_id = job.rollback_next_unit_id;
                state.duplex->current_turn_id = job.rollback_current_turn_id;
                state.duplex->pending_unit_id = -1;
                state.duplex->pending_unit_start_cache_len = 0;
                if (state.duplex->unit_history.size() > job.rollback_unit_history_size) {
                    state.duplex->unit_history.resize(job.rollback_unit_history_size);
                }
                if (state.duplex->round_start_positions.size() > job.rollback_round_count) {
                    state.duplex->round_start_positions.resize(job.rollback_round_count);
                }
            }
            common_sampler_free(state.sampler);
            state.sampler = job.sampler_checkpoint;
            state.duplex->ctx_sampler = state.sampler;
            job.sampler_checkpoint = nullptr;
        } else if (job.sampler_checkpoint) {
            common_sampler_free(job.sampler_checkpoint);
            job.sampler_checkpoint = nullptr;
        }
        job.result.status = status;
        job.result.finish_reason = reason;
        job.result.error = result_error;
        job.result.n_past_after = state.n_past;
        job.result.owner_thread_id = owner_thread_id_.load(std::memory_order_acquire);
        job.result.timings.llm_ms = job.result.timings.llm_prefill_ms +
                                   job.result.timings.llm_decode_ms;
        job.completed = true;
        job.active = false;
        remove_active_locked(state.user_id);
        update_kv_stats_locked();
        cv_.notify_all();
    }

    bool decode_tokens(SequenceState & state, const std::vector<llama_token> & tokens,
                       bool output_last_logits) {
        size_t offset = 0;
        const size_t batch_limit = std::max<size_t>(1, llama_n_batch(ctx_));
        while (offset < tokens.size()) {
            const size_t n = std::min(batch_limit, tokens.size() - offset);
            if (state.n_past + static_cast<int32_t>(n) > static_cast<int32_t>(n_ctx_seq_)) {
                return false;
            }
            llama_batch batch = llama_batch_init(static_cast<int32_t>(n), 0, 1);
            if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
                llama_batch_free(batch);
                return false;
            }
            batch.n_tokens = static_cast<int32_t>(n);
            for (size_t i = 0; i < n; ++i) {
                batch.token[i] = tokens[offset + i];
                batch.pos[i] = state.n_past + static_cast<int32_t>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = state.seq_id;
                batch.logits[i] = output_last_logits && offset + i + 1 == tokens.size();
            }
            const GraphStatsSnapshot graph_before = graph_stats_snapshot();
            const int rc = llama_decode(ctx_, batch);
            const GraphStatsSnapshot graph_after = graph_stats_snapshot();
            record_graph_stats(graph_before, graph_after);
            llama_batch_free(batch);
            if (rc != 0) {
                return false;
            }
            state.n_past += static_cast<int32_t>(n);
            offset += n;
        }
        update_kv_stats();
        return true;
    }

    void append_tokenized(std::vector<llama_token> & output, const std::string & text) const {
        std::vector<llama_token> tokens = common_tokenize(ctx_, text, false, true);
        output.insert(output.end(), tokens.begin(), tokens.end());
    }

    bool build_prefill_plan(const OmniUserTask & task, PrefillPlan & plan) const {
        const bool has_vision = !task.llm_vision_embeddings.empty();
        const bool has_audio = !task.llm_audio_embeddings.empty();
        const int32_t n_embd = llama_model_n_embd(bundle_->llm_model());
        if (!embedding_ctx_ ||
                (task.llm_embedding_dim != 0 && task.llm_embedding_dim != n_embd)) {
            return false;
        }
        int fused_tokens = 0;
        if (!llm_build_fused_embeddings(embedding_ctx_.get(), task.llm_vision_embeddings,
                                        task.llm_audio_embeddings, task.text, task.tokens,
                                        n_embd, plan.embeddings, &fused_tokens)) {
            return false;
        }
        plan.token_count = static_cast<size_t>(fused_tokens);
        plan.input_type = has_vision ? (has_audio ? "omni" : "video")
                                     : (has_audio ? "audio" : "text");
        return plan.token_count > 0;
    }

    bool drop_sequence_prefix(SequenceState & state, int32_t length) {
        if (length <= 0 || state.n_past <= state.n_keep ||
                length > state.n_past - state.n_keep) {
            return false;
        }
        llama_memory_t mem = llama_get_memory(ctx_);
        if (!mem || !llama_memory_seq_rm(mem, state.seq_id, state.n_keep,
                                         state.n_keep + length)) {
            return false;
        }
        if (state.n_past > state.n_keep + length) {
            llama_memory_seq_add(mem, state.seq_id, state.n_keep + length,
                                 state.n_past, -length);
        }
        state.n_past -= length;
        std::vector<int> round_positions;
        for (int position : state.duplex->round_start_positions) {
            if (position > state.n_keep + length) {
                round_positions.push_back(position - length);
            }
        }
        state.duplex->round_start_positions = std::move(round_positions);
        state.duplex->position_offset += length;
        ++state.duplex->sliding_event_count;
        state.duplex->total_dropped_tokens += length;
        {
            std::lock_guard<std::mutex> stats_lock(stats_mtx_);
            ++batch_stats_.sliding_event_count;
            batch_stats_.sliding_dropped_tokens += static_cast<uint64_t>(length);
        }
        return true;
    }

    bool drop_oldest_turn(SequenceState & state) {
        int32_t turn_id = -1;
        int32_t length = 0;
        size_t count = 0;
        for (const UnitEntry & entry : state.duplex->unit_history) {
            if (entry.type == "system") {
                continue;
            }
            if (turn_id < 0) {
                turn_id = entry.turn_id;
            }
            if (entry.turn_id != turn_id) {
                break;
            }
            length += entry.length;
            ++count;
        }
        if (turn_id < 0 || turn_id >= state.duplex->current_turn_id || length <= 0 || count == 0) {
            return false;
        }
        if (!drop_sequence_prefix(state, length)) {
            return false;
        }
        state.duplex->unit_history.erase(state.duplex->unit_history.begin(),
                                         state.duplex->unit_history.begin() + count);
        state.duplex->total_dropped_units += static_cast<int>(count);
        ++state.duplex->total_dropped_turns;
        return true;
    }

    bool drop_oldest_unit(SequenceState & state) {
        for (auto it = state.duplex->unit_history.begin();
             it != state.duplex->unit_history.end(); ++it) {
            if (it->type == "system" || it->length <= 0) {
                continue;
            }
            const int32_t length = it->length;
            if (!drop_sequence_prefix(state, length)) {
                return false;
            }
            state.duplex->unit_history.erase(it);
            ++state.duplex->total_dropped_units;
            ++state.duplex->total_unit_fallbacks;
            return true;
        }
        return false;
    }

    bool ensure_sequence_capacity(SequenceState & state, size_t required_tokens) {
        const uint32_t high = std::max<uint32_t>(state.n_keep + 1, sliding_high_water());
        const uint32_t low = std::max<uint32_t>(state.n_keep, sliding_low_water());
        if (required_tokens > n_ctx_seq_ - std::min<uint32_t>(n_ctx_seq_, state.n_keep)) {
            return false;
        }
        if (static_cast<uint64_t>(state.n_past) + required_tokens <= high) {
            return true;
        }
        while (state.n_past > static_cast<int32_t>(low) ||
               static_cast<uint64_t>(state.n_past) + required_tokens > n_ctx_seq_) {
            if (!drop_oldest_turn(state) && !drop_oldest_unit(state)) {
                break;
            }
        }
        update_kv_stats();
        return static_cast<uint64_t>(state.n_past) + required_tokens <= n_ctx_seq_;
    }

    llama_token sample_next(SequenceState & state, int32_t output_index) {
        float * logits = llama_get_logits_ith(ctx_, output_index);
        omni_context * duplex = state.duplex.get();
        if (!logits) {
            return LLAMA_TOKEN_NULL;
        }
        if (duplex->special_token_listen >= 0) {
            logits[duplex->special_token_listen] += (duplex->listen_prob_scale - 1.0f) * 2.0f;
        }
        if (duplex->special_token_tts_pad >= 0) {
            logits[duplex->special_token_tts_pad] = -INFINITY;
        }
        if (duplex->length_penalty != 1.0f && duplex->special_token_turn_eos >= 0) {
            float & eos_logit = logits[duplex->special_token_turn_eos];
            eos_logit = eos_logit > 0 ? eos_logit / duplex->length_penalty
                                      : eos_logit * duplex->length_penalty;
        }
        const llama_token token = common_sampler_sample(state.sampler, ctx_, output_index);
        common_sampler_accept(state.sampler, token, true);
        return token;
    }

    bool initialize_session(SequenceState & state) {
        if (state.session_initialized) {
            return true;
        }
        std::vector<llama_token> session_tokens;
        append_tokenized(session_tokens,
                         "<|im_start|>system\nStreaming Duplex Conversation! "
                         "You are a helpful assistant.\n<|audio_start|>");
        append_tokenized(session_tokens, "<|audio_end|><|im_end|>\n");
        append_tokenized(session_tokens, "<|im_start|>user\n");
        if (!decode_tokens(state, session_tokens, false)) {
            return false;
        }
        state.session_initialized = true;
        state.n_keep = state.n_past;
        state.duplex->n_keep = state.n_keep;
        state.duplex->system_preserve_length = state.n_keep;
        return true;
    }

    void checkpoint_job(SequenceState & state, Job & job) {
        job.result.n_past_before = state.n_past;
        job.rollback_n_past = state.n_past;
        job.rollback_n_keep = state.n_keep;
        job.rollback_session_initialized = state.session_initialized;
        job.force_listen_used_before = state.duplex->force_listen_used;
        job.slide_last_was_listen_before = state.duplex->slide_last_was_listen;
        job.current_turn_ended_before = state.duplex->current_turn_ended;
        job.ended_with_listen_before = state.duplex->ended_with_listen;
        job.rollback_unit_history_size = state.duplex->unit_history.size();
        job.rollback_round_count = state.duplex->round_start_positions.size();
        job.rollback_next_unit_id = state.duplex->next_unit_id;
        job.rollback_current_turn_id = state.duplex->current_turn_id;
        job.frame_start_n_past = state.n_past;
        job.sampler_checkpoint = common_sampler_clone(state.sampler);
    }

    bool rollback_batch_items(const std::vector<BatchItem *> & items) {
        bool ok = true;
        for (BatchItem * item : items) {
            ok = restore_sequence_position(*item->state, item->position) && ok;
        }
        update_kv_stats();
        return ok;
    }

    void fail_batch_items(const std::vector<BatchItem *> & items,
                          const std::string & error) {
        std::lock_guard<std::mutex> lock(mtx_);
        for (BatchItem * item : items) {
            if (item->state->job == item->job && !item->job->completed) {
                complete_locked(*item->state, OmniLlmResultStatus::FAILED,
                                OmniLlmFinishReason::ERROR, error);
            }
        }
    }

    void register_input_unit(SequenceState & state, Job & job,
                             const std::string & input_type) {
        const int32_t length = state.n_past - job.frame_start_n_past;
        if (length <= 0) {
            return;
        }
        UnitEntry entry;
        entry.unit_id = state.duplex->next_unit_id++;
        entry.length = length;
        entry.type = input_type;
        entry.turn_id = state.duplex->current_turn_id;
        state.duplex->unit_history.push_back(std::move(entry));
    }

    void finish_success_locked(SequenceState & state, Job & job,
                               OmniLlmFinishReason reason) {
        const int32_t response_length = state.n_past - job.decode_start_n_past;
        if (response_length > 0) {
            UnitEntry entry;
            entry.unit_id = state.duplex->next_unit_id++;
            entry.length = response_length;
            entry.type = "response";
            entry.generated_tokens = job.result.token_ids;
            entry.is_listen = reason == OmniLlmFinishReason::LISTEN;
            entry.turn_id = state.duplex->current_turn_id;
            state.duplex->unit_history.push_back(std::move(entry));
        }
        state.duplex->ended_with_listen = reason == OmniLlmFinishReason::LISTEN;
        if (reason == OmniLlmFinishReason::LISTEN) {
            state.duplex->slide_last_was_listen = true;
            // LISTEN closes the frame.  Preserve a TURN_EOS/TTS_EOS/EOS
            // observed earlier in this frame so the TTS final flush is kept.
            state.duplex->current_turn_ended = job.result.is_end_of_turn;
            state.duplex->round_start_positions.push_back(state.n_past);
            ++state.duplex->current_turn_id;
        } else if (job.result.is_end_of_turn) {
            state.duplex->current_turn_ended = true;
            state.duplex->round_start_positions.push_back(state.n_past);
            ++state.duplex->current_turn_id;
        } else if (job.saw_speak) {
            state.duplex->current_turn_ended = false;
        }
        complete_locked(state, OmniLlmResultStatus::COMPLETE, reason);
    }

    void start_decode_locked(SequenceState & state, Job & job, int32_t output_index) {
        job.max_tokens = job.request.task.max_new_tokens
            ? job.request.task.max_new_tokens : config_.default_max_new_tokens;
        job.max_decode_steps = std::max<uint32_t>(job.max_tokens * 4 + 32, job.max_tokens);
        job.decode_start_n_past = state.n_past;
        if (state.duplex->force_listen_used < state.duplex->force_listen_count) {
            ++state.duplex->force_listen_used;
            job.result.is_listen = true;
            job.result.finish_reason = OmniLlmFinishReason::LISTEN;
            job.pending_token = state.duplex->special_token_unit_end;
            job.pending_is_unit_end = true;
        } else if (!job.request.task.replay_decode_tokens.empty()) {
            job.pending_token = job.request.task.replay_decode_tokens[0];
            job.replay_index = 1;
            common_sampler_accept(state.sampler, job.pending_token, true);
        } else {
            job.pending_token = sample_next(state, output_index);
        }
        if (job.pending_token == LLAMA_TOKEN_NULL) {
            complete_locked(state, OmniLlmResultStatus::FAILED, OmniLlmFinishReason::ERROR,
                            "shared LLM sampling failed after prefill");
            return;
        }
        job.prefilled = true;
        job.token_ready = std::chrono::steady_clock::now();
        if (job.pending_token < 0) {
            finish_success_locked(state, job, job.result.finish_reason);
        } else if (job.decode_requested) {
            activate_locked(state);
        }
    }

    void schedule_finish_locked(SequenceState & state, Job & job,
                                OmniLlmFinishReason reason) {
        job.result.is_speak = job.saw_speak;
        if (job.saw_speak) {
            state.duplex->slide_last_was_listen = false;
        }
        job.result.finish_reason = reason;
        job.pending_token = state.duplex->special_token_unit_end;
        job.pending_is_unit_end = true;
        job.token_ready = std::chrono::steady_clock::now();
        if (job.pending_token < 0) {
            finish_success_locked(state, job, reason);
        } else {
            job.active = false;
            activate_locked(state);
        }
    }

    void process_decode_output_locked(BatchItem & item,
                                      const std::vector<float> & hidden,
                                      int32_t output_index,
                                      const std::chrono::steady_clock::time_point & now,
                                      const std::chrono::steady_clock::time_point & service_begin,
                                      double service_ms,
                                      uint32_t decode_batch_size) {
        SequenceState & state = *item.state;
        Job & job = *item.job;
        ++state.n_past;
        job.result.timings.llm_decode_ms += service_ms;
        job.result.timings.llm_batch_wait_ms +=
            std::chrono::duration<double, std::milli>(service_begin - job.token_ready).count();
        ++job.result.decode_batch_count;
        job.result.decode_batch_size_max = std::max(job.result.decode_batch_size_max,
                                                    decode_batch_size);
        if (job.pending_is_unit_end) {
            finish_success_locked(state, job, job.result.finish_reason);
            return;
        }

        ++job.result.decode_steps;
        if (job.result.decode_steps == 1) {
            job.result.timings.first_token_ms = job.request.task.submit_ns
                ? elapsed_ms_from_ns(job.request.task.submit_ns)
                : std::chrono::duration<double, std::milli>(now - job.queue_enter).count();
        }
        const llama_token token = job.pending_token;
        const OmniTokenType type = get_token_type(state.duplex.get(), token);
        if (type == OmniTokenType::SPEAK) {
            job.saw_speak = true;
            job.result.is_speak = true;
        } else if (type == OmniTokenType::LISTEN) {
            job.result.is_listen = true;
            state.duplex->slide_last_was_listen = true;
            schedule_finish_locked(state, job, OmniLlmFinishReason::LISTEN);
            return;
        } else if (type == OmniTokenType::TURN_EOS ||
                   type == OmniTokenType::TTS_EOS ||
                   type == OmniTokenType::EOS) {
            // These tokens close the semantic turn.  Duplex generation must
            // continue until LISTEN or a chunk terminator closes this frame.
            job.result.is_end_of_turn = true;
            state.duplex->current_turn_ended = true;
        } else if (type != OmniTokenType::NORMAL) {
            schedule_finish_locked(state, job, finish_reason_for_token(type));
            return;
        } else if (is_valid_tts_token(token)) {
            job.result.token_ids.push_back(token);
            job.result.hidden_states.insert(job.result.hidden_states.end(),
                                            hidden.begin(), hidden.end());
            job.result.text += common_token_to_piece(ctx_, token, false);
            ++job.valid_tokens;
        }

        job.result.is_speak = job.saw_speak;
        if (job.valid_tokens >= job.max_tokens) {
            if (job.saw_speak) {
                state.duplex->slide_last_was_listen = false;
            }
            schedule_finish_locked(state, job, OmniLlmFinishReason::MAX_NEW_TOKENS);
            return;
        }
        if (job.result.decode_steps >= job.max_decode_steps) {
            if (job.saw_speak) {
                state.duplex->slide_last_was_listen = false;
            }
            schedule_finish_locked(state, job, OmniLlmFinishReason::MAX_DECODE_STEPS);
            return;
        }
        if (job.replay_index < job.request.task.replay_decode_tokens.size()) {
            job.pending_token = job.request.task.replay_decode_tokens[job.replay_index++];
            common_sampler_accept(state.sampler, job.pending_token, true);
        } else if (!job.request.task.replay_decode_tokens.empty()) {
            complete_locked(state, OmniLlmResultStatus::FAILED, OmniLlmFinishReason::ERROR,
                            "decode token replay ended without a terminal token");
            return;
        } else {
            job.pending_token = sample_next(state, output_index);
        }
        if (job.pending_token == LLAMA_TOKEN_NULL) {
            complete_locked(state, OmniLlmResultStatus::FAILED, OmniLlmFinishReason::ERROR,
                            "shared LLM sampling failed");
            return;
        }
        job.token_ready = now;
        job.active = false;
        activate_locked(state);
    }

    bool execute_batch_items(const std::vector<BatchItem *> & items, bool retrying) {
        if (items.empty()) {
            return true;
        }
        const int32_t n_embd = llama_model_n_embd(bundle_->llm_model());
        size_t total_tokens = 0;
        uint32_t prefill_items = 0;
        uint32_t decode_items = 0;
        for (const BatchItem * item : items) {
            total_tokens += item->embeddings.size() / static_cast<size_t>(n_embd);
            if (item->prefill) {
                ++prefill_items;
            } else {
                ++decode_items;
            }
        }
        if (total_tokens == 0 || total_tokens > llama_n_batch(ctx_) ||
                total_tokens > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            fail_batch_items(items, "invalid shared LLM packed batch size");
            return false;
        }

        llama_batch batch = llama_batch_init(static_cast<int32_t>(total_tokens), n_embd, 1);
        if (!batch.embd || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            fail_batch_items(items, "failed to allocate shared LLM packed batch");
            return false;
        }
        batch.n_tokens = static_cast<int32_t>(total_tokens);
        int32_t row = 0;
        for (BatchItem * item : items) {
            const size_t count = item->embeddings.size() / static_cast<size_t>(n_embd);
            std::memcpy(batch.embd + static_cast<size_t>(row) * n_embd,
                        item->embeddings.data(), item->embeddings.size() * sizeof(float));
            for (size_t i = 0; i < count; ++i) {
                const int32_t batch_row = row + static_cast<int32_t>(i);
                batch.pos[batch_row] = item->position + static_cast<int32_t>(i);
                batch.n_seq_id[batch_row] = 1;
                batch.seq_id[batch_row][0] = item->state->seq_id;
                batch.logits[batch_row] = i + 1 == count ? 1 : 0;
            }
            item->batch_last = row + static_cast<int32_t>(count) - 1;
            row += static_cast<int32_t>(count);
        }

        if (!llama_set_embeddings_masked(ctx_, true)) {
            llama_batch_free(batch);
            fail_batch_items(items, "failed to enable selective hidden state output");
            return false;
        }
        const auto service_begin = std::chrono::steady_clock::now();
        const GraphStatsSnapshot graph_before = graph_stats_snapshot();
        const int rc = llama_decode(ctx_, batch);
        if (rc == 0) {
            llama_synchronize(ctx_);
        }
        const GraphStatsSnapshot graph_after = graph_stats_snapshot();
        record_graph_stats(graph_before, graph_after);
        const auto service_end = std::chrono::steady_clock::now();
        const double service_ms = std::chrono::duration<double, std::milli>(
            service_end - service_begin).count();
        const uint32_t n_ubatch = std::max<uint32_t>(1, llama_n_ubatch(ctx_));
        const uint64_t ubatch_count = (total_tokens + n_ubatch - 1) / n_ubatch;

        {
            std::lock_guard<std::mutex> stats_lock(stats_mtx_);
            batch_stats_.ubatch_count += ubatch_count;
            if (prefill_items > 0) {
                ++batch_stats_.prefill_dispatch_count;
                batch_stats_.prefill_item_count += prefill_items;
                batch_stats_.prefill_batch_size_max = std::max(
                    batch_stats_.prefill_batch_size_max, prefill_items);
                batch_stats_.prefill_service_ms += service_ms;
            }
            if (decode_items > 0) {
                ++batch_stats_.decode_dispatch_count;
                batch_stats_.decode_item_count += decode_items;
                batch_stats_.decode_batch_size_max = std::max(
                    batch_stats_.decode_batch_size_max, decode_items);
                batch_stats_.decode_service_ms += service_ms;
            }
            if (prefill_items > 0 && decode_items > 0) {
                ++batch_stats_.mixed_dispatch_count;
            }
        }

        if (rc != 0) {
            llama_set_embeddings_masked(ctx_, false);
            llama_batch_free(batch);
            const bool rollback_ok = rollback_batch_items(items);
            if (rc > 0 && rollback_ok && items.size() > 1 && !retrying) {
                {
                    std::lock_guard<std::mutex> stats_lock(stats_mtx_);
                    batch_stats_.prefill_fallback_count += prefill_items;
                    batch_stats_.decode_fallback_count += decode_items;
                }
                bool ok = true;
                for (BatchItem * item : items) {
                    item->job->result.prefill_fallback |= item->prefill;
                    ok = execute_batch_items({item}, true) && ok;
                }
                return ok;
            }
            fail_batch_items(items, rollback_ok ? "shared LLM batch failed"
                                                : "shared LLM batch failed and rollback was incomplete");
            return false;
        }

        std::vector<std::vector<float>> hidden(items.size());
        bool outputs_ok = true;
        for (size_t i = 0; i < items.size(); ++i) {
            if (!llama_get_logits_ith(ctx_, items[i]->batch_last)) {
                outputs_ok = false;
            }
            if (!items[i]->prefill) {
                const float * values = llama_get_embeddings_ith(ctx_, items[i]->batch_last);
                if (!values) {
                    outputs_ok = false;
                } else {
                    hidden[i].assign(values, values + n_embd);
                }
            }
        }
        if (!outputs_ok) {
            llama_set_embeddings_masked(ctx_, false);
            llama_batch_free(batch);
            const bool rollback_ok = rollback_batch_items(items);
            fail_batch_items(items, rollback_ok ? "shared LLM output mapping failed"
                                                : "shared LLM output mapping and rollback failed");
            return false;
        }

        const auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for (size_t i = 0; i < items.size(); ++i) {
                BatchItem & item = *items[i];
                SequenceState & state = *item.state;
                Job & job = *item.job;
                if (state.job != item.job || job.completed) {
                    continue;
                }
                if (job_cancelled(state, job)) {
                    complete_locked(state, OmniLlmResultStatus::CANCELLED,
                                    OmniLlmFinishReason::CANCELLED, "cancelled after LLM batch");
                    continue;
                }
                if (item.prefill) {
                    state.n_past += static_cast<int32_t>(
                        item.embeddings.size() / static_cast<size_t>(n_embd));
                    job.result.timings.llm_prefill_ms += service_ms;
                    job.result.timings.llm_queue_wait_ms =
                        std::chrono::duration<double, std::milli>(
                            service_begin - job.queue_enter).count();
                    job.result.timings.llm_prefill_batch_wait_ms =
                        job.result.timings.llm_queue_wait_ms;
                    job.result.prefill_batch_size = prefill_items;
                    register_input_unit(state, job, item.input_type);
                    start_decode_locked(state, job, item.batch_last);
                } else {
                    process_decode_output_locked(item, hidden[i], item.batch_last, now,
                                                 service_begin, service_ms, decode_items);
                }
            }
            update_kv_stats_locked();
            cv_.notify_all();
        }
        llama_set_embeddings_masked(ctx_, false);
        llama_batch_free(batch);
        return true;
    }

    void requeue_prefill(const std::vector<std::shared_ptr<Job>> & jobs, size_t begin) {
        if (begin >= jobs.size()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mtx_);
        for (size_t i = jobs.size(); i-- > begin; ) {
            if (!jobs[i]->completed) {
                prefill_queue_.push_front(jobs[i]);
            }
        }
        cv_.notify_all();
    }

    void process_work_batch(const std::vector<std::shared_ptr<Job>> & selected_prefill,
                            const std::vector<uint64_t> & selected_decode) {
        const int32_t n_embd = llama_model_n_embd(bundle_->llm_model());
        const size_t token_limit = std::max<size_t>(1, llama_n_batch(ctx_));
        std::vector<BatchItem> storage;
        storage.reserve(selected_prefill.size() + selected_decode.size());
        size_t total_tokens = 0;

        for (uint64_t user_id : selected_decode) {
            SequenceState * state = nullptr;
            std::shared_ptr<Job> job;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                auto it = users_.find(user_id);
                if (it == users_.end() || !it->second->job || it->second->job->completed) {
                    continue;
                }
                state = it->second.get();
                job = state->job;
                if (job_cancelled(*state, *job)) {
                    complete_locked(*state, OmniLlmResultStatus::CANCELLED,
                                    OmniLlmFinishReason::CANCELLED, "cancelled during LLM decode");
                    continue;
                }
                if (deadline_expired(job->request.task.deadline_ns)) {
                    complete_locked(*state, OmniLlmResultStatus::FAILED,
                                    OmniLlmFinishReason::DEADLINE, "deadline expired during LLM decode");
                    continue;
                }
                if (state->n_past >= static_cast<int32_t>(n_ctx_seq_)) {
                    complete_locked(*state, OmniLlmResultStatus::FAILED,
                                    OmniLlmFinishReason::ERROR, "shared LLM sequence context is full");
                    continue;
                }
            }
            BatchItem item;
            item.state = state;
            item.job = job;
            item.position = state->n_past;
            if (!duplex_tokens_to_embeddings(embedding_ctx_.get(), {job->pending_token},
                                             item.embeddings)) {
                std::lock_guard<std::mutex> lock(mtx_);
                complete_locked(*state, OmniLlmResultStatus::FAILED,
                                OmniLlmFinishReason::ERROR,
                                "failed to look up decode token embedding");
                continue;
            }
            storage.push_back(std::move(item));
            ++total_tokens;
        }

        size_t accepted_prefill = 0;
        for (; accepted_prefill < selected_prefill.size(); ++accepted_prefill) {
            const std::shared_ptr<Job> & job = selected_prefill[accepted_prefill];
            SequenceState * state = nullptr;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                auto it = users_.find(job->request.key.user_id);
                if (it == users_.end() || it->second->job != job || job->completed) {
                    continue;
                }
                state = it->second.get();
                if (job_cancelled(*state, *job)) {
                    complete_locked(*state, OmniLlmResultStatus::CANCELLED,
                                    OmniLlmFinishReason::CANCELLED, "cancelled before LLM prefill");
                    continue;
                }
                if (deadline_expired(job->request.task.deadline_ns)) {
                    complete_locked(*state, OmniLlmResultStatus::FAILED,
                                    OmniLlmFinishReason::DEADLINE, "deadline expired before LLM prefill");
                    continue;
                }
            }

            PrefillPlan plan;
            if (!build_prefill_plan(job->request.task, plan)) {
                std::lock_guard<std::mutex> lock(mtx_);
                complete_locked(*state, OmniLlmResultStatus::FAILED, OmniLlmFinishReason::ERROR,
                                "invalid shared LLM fused prefill input");
                continue;
            }
            if (plan.token_count > token_limit) {
                std::lock_guard<std::mutex> lock(mtx_);
                complete_locked(*state, OmniLlmResultStatus::FAILED, OmniLlmFinishReason::ERROR,
                                "shared LLM fused prefill exceeds n_batch");
                continue;
            }
            if (total_tokens + plan.token_count > token_limit) {
                break;
            }

            const bool needs_initialization = !state->session_initialized;
            if (needs_initialization) {
                checkpoint_job(*state, *job);
                if (!job->sampler_checkpoint || !initialize_session(*state)) {
                    std::lock_guard<std::mutex> lock(mtx_);
                    complete_locked(*state, OmniLlmResultStatus::FAILED,
                                    OmniLlmFinishReason::ERROR,
                                    "shared LLM session initialization failed");
                    continue;
                }
            }
            const uint32_t max_tokens = job->request.task.max_new_tokens
                ? job->request.task.max_new_tokens : config_.default_max_new_tokens;
            const size_t reserve = static_cast<size_t>(max_tokens) * 4 + 33;
            if (!ensure_sequence_capacity(*state, plan.token_count + reserve)) {
                std::lock_guard<std::mutex> lock(mtx_);
                complete_locked(*state, OmniLlmResultStatus::FAILED, OmniLlmFinishReason::ERROR,
                                "shared LLM sequence context is full after sliding");
                continue;
            }
            if (!needs_initialization) {
                checkpoint_job(*state, *job);
                if (!job->sampler_checkpoint) {
                    std::lock_guard<std::mutex> lock(mtx_);
                    complete_locked(*state, OmniLlmResultStatus::FAILED,
                                    OmniLlmFinishReason::ERROR,
                                    "failed to checkpoint shared LLM sampler");
                    continue;
                }
            }
            job->frame_start_n_past = state->n_past;
            BatchItem item;
            item.state = state;
            item.job = job;
            item.embeddings = std::move(plan.embeddings);
            item.input_type = std::move(plan.input_type);
            item.position = state->n_past;
            item.prefill = true;
            total_tokens += plan.token_count;
            storage.push_back(std::move(item));
        }
        requeue_prefill(selected_prefill, accepted_prefill);

        if (storage.empty()) {
            return;
        }
        {
            std::lock_guard<std::mutex> stats_lock(stats_mtx_);
            for (const BatchItem & item : storage) {
                if (item.prefill) {
                    ++batch_stats_.prefill_count;
                    batch_stats_.prefill_input_tokens +=
                        item.embeddings.size() / static_cast<size_t>(n_embd);
                }
            }
        }
        std::vector<BatchItem *> items;
        items.reserve(storage.size());
        for (BatchItem & item : storage) {
            items.push_back(&item);
        }
        execute_batch_items(items, false);
    }

    void take_work_batch(std::vector<std::shared_ptr<Job>> & prefill,
                         std::vector<uint64_t> & decode) {
        std::unique_lock<std::mutex> lock(mtx_);
        const uint32_t limit = max_batch_size();
        if (active_users_.size() + prefill_queue_.size() < limit &&
                config_.llm_max_batch_delay_us > 0) {
            cv_.wait_for(lock, std::chrono::microseconds(config_.llm_max_batch_delay_us), [&] {
                return stopped_.load(std::memory_order_acquire) ||
                       active_users_.size() + prefill_queue_.size() >= limit ||
                       !close_requested_.empty();
            });
        }
        while (!active_users_.empty() && decode.size() < limit) {
            const uint64_t user_id = active_users_.front();
            active_users_.pop_front();
            auto it = users_.find(user_id);
            if (it == users_.end() || !it->second->job || it->second->job->completed) {
                continue;
            }
            it->second->job->active = false;
            decode.push_back(user_id);
        }
        while (!prefill_queue_.empty() && decode.size() + prefill.size() < limit) {
            prefill.push_back(prefill_queue_.front());
            prefill_queue_.pop_front();
        }
    }

    void close_ready_users() {
        std::vector<uint64_t> closed;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for (uint64_t user_id : close_requested_) {
                auto it = users_.find(user_id);
                if (it != users_.end() && !it->second->job) {
                    closed.push_back(user_id);
                }
            }
        }
        for (uint64_t user_id : closed) {
            std::unique_ptr<SequenceState> state;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                auto it = users_.find(user_id);
                if (it == users_.end() || it->second->job) {
                    continue;
                }
                state = std::move(it->second);
                users_.erase(it);
                close_requested_.erase(user_id);
                remove_active_locked(user_id);
            }
            const bool cleared = llama_memory_seq_rm(
                llama_get_memory(ctx_), state->seq_id, -1, -1);
            if (!cleared) {
                std::fprintf(stderr, "[MULTICTX] failed to clear shared LLM sequence %d\n",
                             state->seq_id);
            }
            common_sampler_free(state->sampler);
            {
                std::lock_guard<std::mutex> lock(mtx_);
                if (cleared) {
                    free_seq_ids_.push_back(state->seq_id);
                } else {
                    error_ = "failed to clear a closed shared LLM sequence";
                }
                cv_.notify_all();
            }
            update_kv_stats();
        }
    }

    void run() {
        owner_thread_id_.store(thread_id_value(), std::memory_order_release);
        while (!stopped_.load(std::memory_order_acquire)) {
            close_ready_users();
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [&] {
                    return stopped_.load(std::memory_order_acquire) || !prefill_queue_.empty() ||
                           !active_users_.empty() || !close_requested_.empty();
                });
                if (stopped_.load(std::memory_order_acquire)) {
                    break;
                }
            }
            std::vector<std::shared_ptr<Job>> prefill;
            std::vector<uint64_t> decode;
            take_work_batch(prefill, decode);
            if (!prefill.empty() || !decode.empty()) {
                process_work_batch(prefill, decode);
            }
        }

        std::lock_guard<std::mutex> lock(mtx_);
        for (auto & entry : users_) {
            if (entry.second->job && !entry.second->job->completed) {
                complete_locked(*entry.second, OmniLlmResultStatus::CANCELLED,
                                OmniLlmFinishReason::CANCELLED, "LLM engine stopped");
            }
        }
        cv_.notify_all();
    }

    std::shared_ptr<OmniSharedModelBundle> bundle_;
    common_params params_;
    OmniMultiUserConfig config_;
    llama_context * ctx_ = nullptr;
    std::unique_ptr<omni_context> embedding_ctx_;
    uint32_t n_ctx_ = 0;
    uint32_t n_ctx_seq_ = 0;
    uint32_t n_ctx_per_user_ = 0;
    uint32_t n_seq_max_ = 0;
    uint64_t kv_bytes_per_token_ = 0;
    bool valid_ = false;
    std::string error_;
    std::atomic<bool> stopped_{false};
    std::atomic<uint64_t> owner_thread_id_{0};
    std::thread thread_;

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::unordered_map<uint64_t, std::unique_ptr<SequenceState>> users_;
    std::deque<llama_seq_id> free_seq_ids_;
    std::deque<std::shared_ptr<Job>> prefill_queue_;
    std::deque<uint64_t> active_users_;
    std::unordered_set<uint64_t> close_requested_;

    mutable std::mutex stats_mtx_;
    OmniLlmBatchStats batch_stats_;
};

class OmniPerUserLlmEngineAdapter final : public OmniLlmEngine {
public:
    using PrefillFn = std::function<bool(const OmniLlmEngineRequest &)>;
    using DecodeFn = std::function<bool(const OmniLlmRequestKey &)>;
    using PollFn = std::function<bool(const OmniLlmRequestKey &, OmniLlmEngineResult &, int)>;
    using DrainFn = std::function<bool(int)>;

    OmniPerUserLlmEngineAdapter(uint64_t user_id, llama_context * ctx,
                                std::atomic<int32_t> * n_past,
                                std::atomic<uint64_t> * owner_thread_id,
                                PrefillFn prefill_fn, DecodeFn decode_fn,
                                PollFn poll_fn, DrainFn drain_fn)
        : user_id_(user_id), ctx_(ctx), n_past_(n_past), owner_thread_id_(owner_thread_id),
          prefill_fn_(std::move(prefill_fn)), decode_fn_(std::move(decode_fn)),
          poll_fn_(std::move(poll_fn)), drain_fn_(std::move(drain_fn)) {
    }

    bool valid() const override { return ctx_ != nullptr; }
    const std::string & error() const override { return error_; }
    bool open_user(uint64_t user_id, const OmniUserSessionConfig &) override {
        return user_id == user_id_;
    }
    bool prefill(const OmniLlmEngineRequest & request) override { return prefill_fn_(request); }
    bool decode(const OmniLlmRequestKey & key) override { return decode_fn_(key); }
    bool poll(const OmniLlmRequestKey & key, OmniLlmEngineResult & result,
              int timeout_ms) override {
        return poll_fn_(key, result, timeout_ms);
    }
    void cancel(uint64_t, uint64_t) override {}
    bool drain(uint64_t user_id, int timeout_ms) override {
        return user_id == user_id_ && drain_fn_(timeout_ms);
    }
    void close_user(uint64_t) override {}
    void shutdown() override {}
    const void * context_address() const override { return ctx_; }
    int32_t sequence_id(uint64_t user_id) const override { return user_id == user_id_ ? 0 : -1; }
    int32_t n_past(uint64_t user_id) const override {
        return user_id == user_id_ ? n_past_->load(std::memory_order_acquire) : 0;
    }
    uint32_t n_ctx() const override { return llama_n_ctx(ctx_); }
    uint32_t n_ctx_seq() const override { return llama_n_ctx_seq(ctx_); }
    uint32_t n_seq_max() const override { return llama_n_seq_max(ctx_); }
    uint64_t owner_thread_id() const override {
        return owner_thread_id_->load(std::memory_order_acquire);
    }
    OmniLlmBatchStats batch_stats() const override { return {}; }
    bool sequence_stats(uint64_t user_id, uint64_t & kv_bytes_used,
                        uint64_t & sliding_events, uint64_t & dropped_tokens,
                        size_t & unit_count, int32_t & current_turn) const override {
        if (user_id != user_id_) {
            return false;
        }
        kv_bytes_used = 0;
        sliding_events = 0;
        dropped_tokens = 0;
        unit_count = 0;
        current_turn = 0;
        return true;
    }

private:
    uint64_t user_id_ = 0;
    llama_context * ctx_ = nullptr;
    std::atomic<int32_t> * n_past_ = nullptr;
    std::atomic<uint64_t> * owner_thread_id_ = nullptr;
    PrefillFn prefill_fn_;
    DecodeFn decode_fn_;
    PollFn poll_fn_;
    DrainFn drain_fn_;
    std::string error_;
};


std::shared_ptr<OmniLlmEngine> omni_create_continuous_batch_llm_engine(
    std::shared_ptr<OmniSharedModelBundle> bundle,
    const common_params & params,
    const OmniMultiUserConfig & config) {
    return std::make_shared<OmniContinuousBatchLlmEngine>(std::move(bundle), params, config);
}

std::unique_ptr<OmniLlmEngine> omni_create_per_user_llm_engine_adapter(
    uint64_t user_id,
    llama_context * ctx,
    std::atomic<int32_t> * n_past,
    std::atomic<uint64_t> * owner_thread_id,
    OmniLlmPrefillFn prefill_fn,
    OmniLlmDecodeFn decode_fn,
    OmniLlmPollFn poll_fn,
    OmniLlmDrainFn drain_fn) {
    return std::make_unique<OmniPerUserLlmEngineAdapter>(
        user_id, ctx, n_past, owner_thread_id, std::move(prefill_fn),
        std::move(decode_fn), std::move(poll_fn), std::move(drain_fn));
}
