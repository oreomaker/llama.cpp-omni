#include "multi_user_multictx_runtime.h"
#include "multi_user_llm_engine.h"

#include "common.h"
#include "duplex_core.h"
#include "omni_internal.h"
#include "sampling.h"
#include "token2wav/token2wav.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
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

class OmniMultiContextUserWorker {
public:
    using EventSink = std::function<void(OmniMultiUserEvent)>;
    using FinishSink = std::function<void()>;
    using GenerationSource = std::function<uint64_t()>;

    OmniMultiContextUserWorker(uint64_t user_id,
                               std::shared_ptr<OmniSharedModelBundle> bundle,
                               const common_params & params,
                               std::shared_ptr<OmniLlmEngine> shared_llm_engine,
                               EventSink event_sink,
                               FinishSink finish_sink,
                               GenerationSource generation_source,
                               uint32_t default_max_new_tokens,
                               uint32_t force_listen_count,
                               const OmniUserSessionConfig & session_config,
                               bool enable_vpm, bool enable_apm, bool enable_tts, bool enable_t2w,
                               const std::string & tts_bin_dir)
        : user_id_(user_id),
          bundle_(std::move(bundle)),
          params_(params),
          shared_llm_engine_(std::move(shared_llm_engine)),
          event_sink_(std::move(event_sink)),
          finish_sink_(std::move(finish_sink)),
          generation_source_(std::move(generation_source)),
          default_max_new_tokens_(default_max_new_tokens ? default_max_new_tokens : 26) {
        session_config_ = session_config;
        stage_stats_.user_id = user_id_;
        if (!bundle_ || !bundle_->llm_model()) {
            error_ = "shared LLM model is unavailable";
            return;
        }
        tts_enabled_ = enable_tts;
        t2w_enabled_ = enable_t2w;

        if (shared_llm_engine_) {
            if (!shared_llm_engine_->valid()) {
                error_ = shared_llm_engine_->error();
                return;
            }
            if (!initialize_auxiliary_contexts(enable_vpm, enable_apm, enable_tts, enable_t2w,
                                               tts_bin_dir)) {
                return;
            }
            n_ctx_ = shared_llm_engine_->n_ctx();
            n_ctx_seq_ = shared_llm_engine_->n_ctx_seq();
            n_seq_max_ = shared_llm_engine_->n_seq_max();
            kv_bytes_estimate_ = estimate_kv_bytes(bundle_->llm_model(), n_ctx_seq_);
            active_llm_engine_ = shared_llm_engine_.get();
            valid_ = true;
            return;
        }

        llama_context_params cparams = common_context_params_to_llama(params_);
        cparams.n_seq_max = 1;
        if (cparams.n_batch <= 0) {
            cparams.n_batch = 512;
        }
        if (cparams.n_ubatch <= 0) {
            cparams.n_ubatch = cparams.n_batch;
        }

        ctx_ = llama_init_from_model(bundle_->llm_model(), cparams);
        if (!ctx_) {
            error_ = "failed to create per-user llama_context";
            return;
        }

        common_params_sampling sampler_params = params_.sampling;
        sampler_ = common_sampler_init(bundle_->llm_model(), sampler_params);
        if (!sampler_) {
            error_ = "failed to create per-user sampler";
            llama_free(ctx_);
            ctx_ = nullptr;
            return;
        }

        duplex_ctx_ = std::make_unique<omni_context>();
        duplex_ctx_->ctx_llama = ctx_;
        duplex_ctx_->model = bundle_->llm_model();
        duplex_ctx_->ctx_sampler = sampler_;
        duplex_ctx_->params = &params_;
        duplex_ctx_->duplex_mode = true;
        duplex_ctx_->async = false;
        duplex_ctx_->media_type = (enable_vpm ? 2 : 0);
        duplex_ctx_->use_tts = enable_tts;
        duplex_ctx_->ctx_vision = nullptr;
        duplex_ctx_->ctx_audio = nullptr;
        duplex_ctx_->n_past = 0;
        duplex_ctx_->n_keep = 0;
        initialize_duplex_tokens(duplex_ctx_.get(), bundle_->llm_model());
        duplex_ctx_->force_listen_count = static_cast<int>(force_listen_count);
        duplex_ctx_->ctx_vision = vpm_ctx_;
        duplex_ctx_->ctx_audio = apm_ctx_;
        const char * segmented_debug = std::getenv("OMNI_DEBUG_SEGMENTED_PREFILL");
        segmented_prefill_debug_ = segmented_debug && std::atoi(segmented_debug) != 0;
        fused_prefill_available_ = duplex_init_emb_cache_for_runtime(duplex_ctx_.get(), &params_);
        if (!fused_prefill_available_ && !segmented_prefill_debug_) {
            error_ = "failed to initialize fused LLM prefill";
            return;
        }
        if (!initialize_auxiliary_contexts(enable_vpm, enable_apm, enable_tts, enable_t2w,
                                           tts_bin_dir)) {
            return;
        }

        n_ctx_ = llama_n_ctx(ctx_);
        n_ctx_seq_ = llama_n_ctx_seq(ctx_);
        n_seq_max_ = llama_n_seq_max(ctx_);
        kv_bytes_estimate_ = estimate_kv_bytes(bundle_->llm_model(), n_ctx_);
        per_user_llm_engine_ = omni_create_per_user_llm_engine_adapter(
            user_id_, ctx_, &n_past_, &owner_thread_id_,
            [this](const OmniLlmEngineRequest & request) {
                return per_user_llm_prefill(request);
            },
            [this](const OmniLlmRequestKey & key) {
                return per_user_llm_decode(key);
            },
            [this](const OmniLlmRequestKey & key, OmniLlmEngineResult & result, int timeout_ms) {
                return per_user_llm_poll(key, result, timeout_ms);
            },
            [this](int timeout_ms) {
                return wait_idle(timeout_ms);
            });
        active_llm_engine_ = per_user_llm_engine_.get();
        valid_ = true;
    }

    ~OmniMultiContextUserWorker() {
        stop(true);
        if (thread_.joinable()) {
            thread_.join();
        }
        if (sampler_) {
            common_sampler_free(sampler_);
            sampler_ = nullptr;
        }
        if (ctx_) {
            llama_free(ctx_);
            ctx_ = nullptr;
        }
        if (tts_ctx_) {
            common_sampler_free(tts_ctx_->ctx_tts_sampler);
            tts_ctx_->ctx_tts_sampler = nullptr;
            llama_free(tts_ctx_->ctx_tts_llama);
            tts_ctx_->ctx_tts_llama = nullptr;
            delete tts_ctx_;
            tts_ctx_ = nullptr;
        }
        duplex_ctx_.reset();
        if (vpm_ctx_) {
            vision_free(vpm_ctx_);
            vpm_ctx_ = nullptr;
        }
        if (apm_ctx_) {
            audition_free(apm_ctx_);
            apm_ctx_ = nullptr;
        }
    }

    OmniMultiContextUserWorker(const OmniMultiContextUserWorker &) = delete;
    OmniMultiContextUserWorker & operator=(const OmniMultiContextUserWorker &) = delete;

    bool valid() const { return valid_; }
    const std::string & error() const { return error_; }

    void start() {
        if (!valid_ || thread_.joinable()) {
            return;
        }
        thread_ = std::thread(&OmniMultiContextUserWorker::run, this);
    }

    bool enqueue(OmniUserTask task) {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        if (stop_requested_) {
            return false;
        }
        queue_.push_back(std::move(task));
        queue_cv_.notify_one();
        return true;
    }

    void cancel_pending(uint64_t generation) {
        if (shared_llm_engine_) {
            shared_llm_engine_->cancel(user_id_, generation);
        }
        std::deque<OmniUserTask> cancelled;
        bool defer = false;
        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            cancelled.swap(queue_);
            defer = active_.load(std::memory_order_acquire);
            if (defer) {
                for (OmniUserTask & task : cancelled) {
                    cancelled_queue_.push_back({std::move(task), generation});
                }
            }
        }
        if (defer) {
            return;
        }
        for (const OmniUserTask & task : cancelled) {
            OmniMultiUserEvent event;
            event.user_id = user_id_;
            event.frame_id = task.frame_id;
            event.cancel_generation = generation;
            event.kind = OmniMultiUserEventKind::CANCELLED;
            event.stage = "runtime";
            event.owner_thread_id = owner_thread_id_.load(std::memory_order_acquire);
            event.error = "cancelled before execution";
            event_sink_(std::move(event));
            finish_sink_();
        }
        idle_cv_.notify_all();
    }

    void stop(bool cancel) {
        if (cancel) {
            cancel_pending(generation_source_ ? generation_source_() : 0);
        }
        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            stop_requested_ = true;
        }
        queue_cv_.notify_all();
        idle_cv_.notify_all();
    }

    bool idle() const {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        return queue_.empty() && cancelled_queue_.empty() &&
               !active_.load(std::memory_order_acquire);
    }

    bool wait_idle(int timeout_ms) {
        std::unique_lock<std::mutex> lock(queue_mtx_);
        auto ready = [&] {
            return queue_.empty() && cancelled_queue_.empty() &&
                   !active_.load(std::memory_order_acquire);
        };
        if (timeout_ms < 0) {
            idle_cv_.wait(lock, ready);
            return true;
        }
        return idle_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready);
    }

    void join() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    OmniMultiContextUserStats stats() const {
        OmniMultiContextUserStats result;
        result.user_id = user_id_;
        result.context_address = shared_llm_engine_
            ? shared_llm_engine_->context_address() : ctx_;
        result.vpm_context_address = vpm_ctx_;
        result.apm_context_address = apm_ctx_;
        result.tts_context_address = tts_ctx_ ? tts_ctx_->ctx_tts_llama : nullptr;
        result.t2w_executor_address = t2w_session_.get();
        result.owner_thread_id = shared_llm_engine_
            ? shared_llm_engine_->owner_thread_id()
            : owner_thread_id_.load(std::memory_order_acquire);
        result.n_ctx = n_ctx_;
        result.n_ctx_seq = n_ctx_seq_;
        result.n_seq_max = n_seq_max_;
        result.llm_seq_id = shared_llm_engine_
            ? shared_llm_engine_->sequence_id(user_id_) : 0;
        result.n_past = shared_llm_engine_
            ? shared_llm_engine_->n_past(user_id_)
            : n_past_.load(std::memory_order_acquire);
        result.kv_bytes_estimate = kv_bytes_estimate_;
        if (shared_llm_engine_) {
            shared_llm_engine_->sequence_stats(user_id_, result.kv_bytes_used_estimate,
                                               result.sliding_event_count,
                                               result.sliding_dropped_tokens,
                                               result.llm_unit_count,
                                               result.llm_current_turn);
        }
        result.apm_kv_bytes = audition_stream_byte_size(apm_ctx_);
        result.has_vpm = vpm_ctx_ != nullptr;
        result.has_apm = apm_ctx_ != nullptr;
        result.has_tts = tts_ctx_ != nullptr;
        result.has_t2w = t2w_session_ != nullptr;
        result.queue_depth = queue_depth();
        result.active = active_.load(std::memory_order_acquire);
        return result;
    }

    OmniMultiUserStageStats stage_stats() const {
        std::lock_guard<std::mutex> lock(stage_stats_mtx_);
        return stage_stats_;
    }

    size_t queue_depth() const {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        return queue_.size() + cancelled_queue_.size();
    }

private:
    bool initialize_auxiliary_contexts(bool enable_vpm, bool enable_apm,
                                       bool enable_tts, bool enable_t2w,
                                       const std::string & tts_bin_dir) {
        if (enable_vpm) {
            vpm_ctx_ = vision_shared_clone(bundle_->vpm_model());
            if (!vpm_ctx_) {
                error_ = "failed to create per-user VPM executor";
                return false;
            }
            vision_set_batch_encode(vpm_ctx_, params_.vpm_batch_encode);
        }
        if (enable_apm) {
            apm_ctx_ = audition_shared_clone(bundle_->apm_model());
            if (!apm_ctx_) {
                error_ = "failed to create per-user APM executor";
                return false;
            }
        }
        if (enable_tts) {
            const omni_context * shared_tts = bundle_->tts_weights();
            if (!bundle_->tts_model() || !shared_tts) {
                error_ = "shared TTS model is unavailable";
                return false;
            }
            tts_ctx_ = new omni_context();
            tts_ctx_->duplex_mode = true;
            tts_ctx_->tts_temperature = 0.8f;
            tts_ctx_->params = &params_;
            tts_ctx_->model_tts = bundle_->tts_model();
            tts_ctx_->emb_code_weight = shared_tts->emb_code_weight;
            tts_ctx_->emb_code_vocab_size = shared_tts->emb_code_vocab_size;
            tts_ctx_->emb_code_hidden_size = shared_tts->emb_code_hidden_size;
            tts_ctx_->emb_code_stored_as_transposed = shared_tts->emb_code_stored_as_transposed;
            tts_ctx_->emb_text_weight = shared_tts->emb_text_weight;
            tts_ctx_->emb_text_vocab_size = shared_tts->emb_text_vocab_size;
            tts_ctx_->emb_text_hidden_size = shared_tts->emb_text_hidden_size;
            tts_ctx_->projector_semantic_linear1_weight = shared_tts->projector_semantic_linear1_weight;
            tts_ctx_->projector_semantic_linear1_bias = shared_tts->projector_semantic_linear1_bias;
            tts_ctx_->projector_semantic_linear2_weight = shared_tts->projector_semantic_linear2_weight;
            tts_ctx_->projector_semantic_linear2_bias = shared_tts->projector_semantic_linear2_bias;
            tts_ctx_->projector_semantic_input_dim = shared_tts->projector_semantic_input_dim;
            tts_ctx_->projector_semantic_output_dim = shared_tts->projector_semantic_output_dim;
            tts_ctx_->head_code_weight = shared_tts->head_code_weight;
            tts_ctx_->head_code_hidden_size = shared_tts->head_code_hidden_size;
            tts_ctx_->head_code_num_audio_tokens = shared_tts->head_code_num_audio_tokens;
            llama_context_params tts_params = common_context_params_to_llama(params_);
            tts_params.n_ctx = std::min<uint32_t>(
                tts_params.n_ctx > 0 ? tts_params.n_ctx : 2048u, 2048u);
            tts_params.n_seq_max = 1;
            tts_ctx_->ctx_tts_llama = llama_init_from_model(bundle_->tts_model(), tts_params);
            if (!tts_ctx_->ctx_tts_llama) {
                error_ = "failed to create per-user TTS context";
                return false;
            }
            tts_ctx_->ctx_tts_sampler = common_sampler_init(bundle_->tts_model(), params_.sampling);
            if (!tts_ctx_->ctx_tts_sampler) {
                error_ = "failed to create per-user TTS sampler";
                return false;
            }
        }
        if (enable_t2w) {
            if (tts_bin_dir.empty()) {
                error_ = "T2W model directory is empty";
                return false;
            }
            t2w_session_ = std::make_unique<omni::flow::Token2WavSession>();
            const std::string encoder = tts_bin_dir + "/encoder.gguf";
            const std::string flow_matching = tts_bin_dir + "/flow_matching.gguf";
            const std::string flow_extra = tts_bin_dir + "/flow_extra.gguf";
            const std::string vocoder = tts_bin_dir + "/hifigan2.gguf";
            const std::string prompt_cache = tts_bin_dir + "/prompt_cache.gguf";
            if (!t2w_session_->init_from_prompt_cache_gguf(
                    encoder, flow_matching, flow_extra, prompt_cache, vocoder,
                    "gpu", "gpu", 5, 1.0f)) {
                error_ = "failed to create per-user T2W executor";
                return false;
            }
        }
        return true;
    }

    static double elapsed_ms(const std::chrono::steady_clock::time_point & begin,
                             const std::chrono::steady_clock::time_point & end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    }

    void record_stage(OmniStageTimingStats & stage, double duration_ms) {
        std::lock_guard<std::mutex> lock(stage_stats_mtx_);
        if (stage.count == 0) {
            stage.min_ms = duration_ms;
            stage.max_ms = duration_ms;
        } else {
            stage.min_ms = std::min(stage.min_ms, duration_ms);
            stage.max_ms = std::max(stage.max_ms, duration_ms);
        }
        ++stage.count;
        stage.total_ms += duration_ms;
    }

    void record_frame_result(bool is_speak) {
        std::lock_guard<std::mutex> lock(stage_stats_mtx_);
        ++stage_stats_.completed_frames;
        if (is_speak) {
            ++stage_stats_.speak_frames;
        } else {
            ++stage_stats_.listen_frames;
        }
    }

    bool duplex_prefill_frame(const OmniUserTask & task, uint64_t generation) {
        const bool has_image = !task.input.img_fname.empty();
        const bool has_audio = !task.input.aud_fname.empty();
        if (!has_image && !has_audio && task.text.empty() && task.tokens.empty()) {
            return false;
        }
        std::vector<std::vector<float>> vision_chunks;
        if (has_image) {
            if (vpm_ctx_ && task.input.max_slice_nums >= 1) {
                vision_set_max_slice_nums(vpm_ctx_, task.input.max_slice_nums);
            }
            const auto vpm_begin = std::chrono::steady_clock::now();
            if (!vpm_ctx_ || !omni_image_embed_make_chunks_with_filename(
                    vpm_ctx_, params_.cpuparams.n_threads, task.input.img_fname, vision_chunks)) {
                return false;
            }
            frame_timings_.vpm_ms = elapsed_ms(vpm_begin, std::chrono::steady_clock::now());
            record_stage(stage_stats_.vpm, frame_timings_.vpm_ms);
        }
        std::vector<float> audio_embed;
        if (has_audio) {
            if (!apm_ctx_) {
                return false;
            }
            const auto apm_begin = std::chrono::steady_clock::now();
            std::unique_ptr<omni_embed, void (*)(omni_embed *)> embed(
                omni_audio_embed_make_with_filename(apm_ctx_, params_.cpuparams.n_threads,
                                                    task.input.aud_fname), omni_embed_free);
            if (!embed) {
                return false;
            }
            const int hidden = llama_model_n_embd(bundle_->llm_model());
            const int apm_hidden = audition_n_mmproj_embd(apm_ctx_);
            if (apm_hidden != hidden) {
                return false;
            }
            audio_embed.assign(embed->embed,
                               embed->embed + static_cast<size_t>(embed->n_pos) * hidden);
            frame_timings_.apm_ms = elapsed_ms(apm_begin, std::chrono::steady_clock::now());
            record_stage(stage_stats_.apm, frame_timings_.apm_ms);
        }
        const auto llm_prefill_begin = std::chrono::steady_clock::now();
        const int32_t prefill_start = duplex_ctx_->n_past;
        bool fused_prefilled = false;
        if (fused_prefill_available_) {
            const int hidden = llama_model_n_embd(bundle_->llm_model());
            fused_prefilled = llm_fused_prefill(duplex_ctx_.get(), &params_, vision_chunks,
                                                audio_embed, hidden, nullptr,
                                                task.text, task.tokens);
            if (fused_prefilled) {
                std::fprintf(stderr, "[MULTICTX PREFILL] user=%llu frame=%llu topology=fused\n",
                             static_cast<unsigned long long>(user_id_),
                             static_cast<unsigned long long>(task.frame_id));
            }
        }
        if (!fused_prefilled && !segmented_prefill_debug_) {
            const bool rollback_ok = llama_memory_seq_rm(
                llama_get_memory(duplex_ctx_->ctx_llama), 0, prefill_start, -1);
            if (!rollback_ok) {
                error_ = "failed to rollback fused LLM prefill";
            } else {
                duplex_ctx_->n_past = prefill_start;
            }
            return false;
        }
        if (!fused_prefilled && !llama_memory_seq_rm(llama_get_memory(duplex_ctx_->ctx_llama), 0,
                                                     prefill_start, -1)) {
            return false;
        }
        if (!fused_prefilled) {
            duplex_ctx_->n_past = prefill_start;
        }
        if (!fused_prefilled && !vision_chunks.empty()) {
            const int hidden = llama_model_n_embd(bundle_->llm_model());
            const int vpm_hidden = vision_n_mmproj_embd(vpm_ctx_);
            if (vpm_hidden != hidden || vision_chunks[0].empty()) {
                return false;
            }
            const int tokens_per_chunk = static_cast<int>(vision_chunks[0].size() / hidden);
            if (!eval_string(duplex_ctx_.get(), &params_, "<unit><image>", params_.n_batch,
                             &duplex_ctx_->n_past, false)) return false;
            if (!prefill_with_emb(duplex_ctx_.get(), &params_, vision_chunks[0].data(),
                                  tokens_per_chunk, params_.n_batch, &duplex_ctx_->n_past)) return false;
            if (!eval_string(duplex_ctx_.get(), &params_, "</image>", params_.n_batch,
                             &duplex_ctx_->n_past, false)) return false;
            for (size_t i = 1; i < vision_chunks.size(); ++i) {
                if (!eval_string(duplex_ctx_.get(), &params_, "<slice>", params_.n_batch,
                                 &duplex_ctx_->n_past, false) ||
                    !prefill_with_emb(duplex_ctx_.get(), &params_, vision_chunks[i].data(),
                                      tokens_per_chunk, params_.n_batch, &duplex_ctx_->n_past) ||
                    !eval_string(duplex_ctx_.get(), &params_, "</slice>", params_.n_batch,
                                 &duplex_ctx_->n_past, false)) return false;
            }
            if (vision_chunks.size() > 1 && !eval_string(duplex_ctx_.get(), &params_, "\n",
                                                           params_.n_batch, &duplex_ctx_->n_past, false)) return false;
        } else if (!fused_prefilled && !eval_string(duplex_ctx_.get(), &params_, "<unit>", params_.n_batch,
                                &duplex_ctx_->n_past, false)) {
            return false;
        }
        if (!fused_prefilled && !audio_embed.empty() && !prefill_with_emb(duplex_ctx_.get(), &params_, audio_embed.data(),
                                                       static_cast<int>(audio_embed.size() /
                                                                        static_cast<size_t>(llama_model_n_embd(bundle_->llm_model()))),
                                                       params_.n_batch, &duplex_ctx_->n_past)) {
            return false;
        }
        if (!fused_prefilled && !vision_chunks.empty()) {
            std::fprintf(stderr, "[MULTICTX PREFILL] user=%llu frame=%llu topology=segmented-debug\n",
                         static_cast<unsigned long long>(user_id_),
                         static_cast<unsigned long long>(task.frame_id));
        }
        if (!fused_prefilled && !task.text.empty()) {
            if (!eval_string(duplex_ctx_.get(), &params_, task.text.c_str(), params_.n_batch,
                             &duplex_ctx_->n_past, false)) return false;
        }
        if (!fused_prefilled && !task.tokens.empty()) {
            if (!eval_tokens(duplex_ctx_.get(), &params_, task.tokens, params_.n_batch,
                             &duplex_ctx_->n_past)) return false;
        }
        frame_timings_.llm_ms += elapsed_ms(llm_prefill_begin, std::chrono::steady_clock::now());
        return !cancelled(generation);
    }

    bool initialize_duplex_session() {
        if (session_initialized_) {
            return true;
        }
        const std::string prefix = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        const std::string suffix = "<|audio_end|><|im_end|>\n";
        if (!eval_string(duplex_ctx_.get(), &params_, prefix.c_str(), params_.n_batch,
                         &duplex_ctx_->n_past, false)) {
            return false;
        }
        if (!session_config_.voice_audio.empty() && apm_ctx_) {
            std::unique_ptr<omni_embed, void (*)(omni_embed *)> ref(
                omni_audio_embed_make_with_filename(apm_ctx_, params_.cpuparams.n_threads,
                                                    session_config_.voice_audio), omni_embed_free);
            if (!ref || ref->n_pos <= 0 || !prefill_with_emb(duplex_ctx_.get(), &params_, ref->embed,
                                                              ref->n_pos, params_.n_batch, &duplex_ctx_->n_past)) {
                return false;
            }
        }
        if (!eval_string(duplex_ctx_.get(), &params_, suffix.c_str(), params_.n_batch,
                         &duplex_ctx_->n_past, false)) {
            return false;
        }
        eval_prefix(duplex_ctx_.get(), &params_);
        session_initialized_ = true;
        duplex_ctx_->n_keep = duplex_ctx_->n_past;
        n_past_.store(duplex_ctx_->n_past, std::memory_order_release);
        return true;
    }

    bool duplex_decode_frame(const OmniUserTask & task, uint64_t generation,
                             std::vector<llama_token> & generated,
                             std::vector<float> & hidden_states, std::string & text,
                             bool & is_speak, bool & is_listen,
                             OmniLlmFinishReason & finish_reason) {
        const int max_tokens = task.max_new_tokens ? task.max_new_tokens : default_max_new_tokens_;
        const int llm_hidden = llama_model_n_embd(bundle_->llm_model());
        bool saw_speak = false;
        bool is_end_of_turn = false;
        int valid_tokens = 0;
        int decode_steps = 0;
        const int max_decode_steps = std::max(max_tokens * 4 + 32, max_tokens);
        while (valid_tokens < max_tokens && decode_steps++ < max_decode_steps) {
            if (cancelled(generation)) return false;
            float * hidden = nullptr;
            llama_token token = 0;
            const char * piece = llama_loop_with_hidden_and_token(
                duplex_ctx_.get(), &params_, sampler_, duplex_ctx_->n_past, hidden, token);
            if (!piece || !hidden) return false;
            const OmniTokenType type = get_token_type(duplex_ctx_.get(), token);
            if (type == OmniTokenType::SPEAK) {
                saw_speak = true;
                is_speak = true;
                continue;
            }
            if (type == OmniTokenType::LISTEN) {
                is_listen = true;
                duplex_ctx_->slide_last_was_listen = true;
                finish_reason = OmniLlmFinishReason::LISTEN;
                break;
            }
            if (type == OmniTokenType::TURN_EOS || type == OmniTokenType::TTS_EOS ||
                    type == OmniTokenType::EOS) {
                is_end_of_turn = true;
                duplex_ctx_->current_turn_ended = true;
                continue;
            }
            if (type == OmniTokenType::CHUNK_EOS || type == OmniTokenType::CHUNK_TTS_EOS) {
                finish_reason = finish_reason_for_token(type);
                break;
            }
            if (is_valid_tts_token(token)) {
                generated.push_back(token);
                hidden_states.insert(hidden_states.end(), hidden, hidden + llm_hidden);
                text += common_token_to_piece(duplex_ctx_->ctx_llama, token, false);
                ++valid_tokens;
            }
        }
        is_speak = saw_speak;
        per_user_llm_result_.is_end_of_turn = is_end_of_turn;
        if (is_speak) {
            duplex_ctx_->slide_last_was_listen = false;
        }
        if (finish_reason == OmniLlmFinishReason::NONE) {
            finish_reason = valid_tokens >= max_tokens
                ? OmniLlmFinishReason::MAX_NEW_TOKENS
                : OmniLlmFinishReason::MAX_DECODE_STEPS;
        }
        return true;
    }

    bool duplex_replay_frame(const OmniUserTask & task, uint64_t generation,
                             std::vector<llama_token> & generated,
                             std::vector<float> & hidden_states, std::string & text,
                             bool & is_speak, bool & is_listen,
                             OmniLlmFinishReason & finish_reason) {
        const uint32_t max_tokens = task.max_new_tokens
            ? task.max_new_tokens : default_max_new_tokens_;
        const uint32_t max_decode_steps = std::max<uint32_t>(max_tokens * 4 + 32, max_tokens);
        const int32_t n_embd = llama_model_n_embd(bundle_->llm_model());
        uint32_t valid_tokens = 0;
        uint32_t decode_steps = 0;
        bool saw_speak = false;
        bool is_end_of_turn = false;
        for (llama_token token : task.replay_decode_tokens) {
            if (cancelled(generation)) {
                return false;
            }
            if (decode_steps >= max_decode_steps || valid_tokens >= max_tokens) {
                break;
            }
            common_sampler_accept(sampler_, token, true);
            float * hidden = nullptr;
            std::vector<llama_token> one{token};
            if (!eval_tokens_with_hidden(duplex_ctx_.get(), &params_, one, 1,
                                         &duplex_ctx_->n_past, hidden) || !hidden) {
                free(hidden);
                return false;
            }
            ++decode_steps;
            const OmniTokenType type = get_token_type(duplex_ctx_.get(), token);
            if (type == OmniTokenType::SPEAK) {
                saw_speak = true;
                free(hidden);
                continue;
            }
            if (type == OmniTokenType::LISTEN) {
                is_listen = true;
                duplex_ctx_->slide_last_was_listen = true;
                finish_reason = OmniLlmFinishReason::LISTEN;
                free(hidden);
                break;
            }
            if (type == OmniTokenType::TURN_EOS || type == OmniTokenType::TTS_EOS ||
                    type == OmniTokenType::EOS) {
                is_end_of_turn = true;
                duplex_ctx_->current_turn_ended = true;
                free(hidden);
                continue;
            }
            if (type == OmniTokenType::CHUNK_EOS || type == OmniTokenType::CHUNK_TTS_EOS) {
                finish_reason = finish_reason_for_token(type);
                free(hidden);
                break;
            }
            if (is_valid_tts_token(token)) {
                generated.push_back(token);
                hidden_states.insert(hidden_states.end(), hidden, hidden + n_embd);
                text += common_token_to_piece(duplex_ctx_->ctx_llama, token, false);
                ++valid_tokens;
            }
            free(hidden);
        }
        is_speak = saw_speak;
        per_user_llm_result_.is_end_of_turn = is_end_of_turn;
        if (is_speak) {
            duplex_ctx_->slide_last_was_listen = false;
        }
        if (finish_reason == OmniLlmFinishReason::NONE) {
            if (valid_tokens >= max_tokens) {
                finish_reason = OmniLlmFinishReason::MAX_NEW_TOKENS;
            } else if (decode_steps >= max_decode_steps) {
                finish_reason = OmniLlmFinishReason::MAX_DECODE_STEPS;
            } else {
                return false;
            }
        }
        return true;
    }

    bool per_user_llm_prefill(const OmniLlmEngineRequest & request) {
        if (shared_llm_engine_ || per_user_llm_pending_) {
            return false;
        }
        frame_timings_ = {};
        frame_timings_.frame_queue_wait_ms = elapsed_ms_from_ns(request.task.submit_ns);
        per_user_llm_result_ = {};
        per_user_llm_result_.key = request.key;
        per_user_llm_result_.n_past_before = duplex_ctx_->n_past;
        per_user_llm_result_.owner_thread_id = owner_thread_id_.load(std::memory_order_acquire);
        per_user_llm_task_ = request.task;
        per_user_llm_pending_ = true;
        per_user_llm_ready_ = false;
        const auto prefill_begin = std::chrono::steady_clock::now();
        if (request.queue_enter.time_since_epoch().count() != 0) {
            per_user_llm_result_.timings.llm_queue_wait_ms = elapsed_ms(request.queue_enter, prefill_begin);
        }
        if (cancelled(request.key.cancel_generation)) {
            per_user_llm_result_.status = OmniLlmResultStatus::CANCELLED;
            per_user_llm_result_.finish_reason = OmniLlmFinishReason::CANCELLED;
            per_user_llm_result_.error = "cancelled before LLM prefill";
            per_user_llm_result_.stage = "LLM";
            per_user_llm_ready_ = true;
            return true;
        }
        if (deadline_expired(request.task.deadline_ns)) {
            per_user_llm_result_.status = OmniLlmResultStatus::FAILED;
            per_user_llm_result_.finish_reason = OmniLlmFinishReason::DEADLINE;
            per_user_llm_result_.error = "deadline expired before LLM prefill";
            per_user_llm_result_.stage = "LLM";
            per_user_llm_ready_ = true;
            return true;
        }
        if (!initialize_duplex_session()) {
            per_user_llm_result_.status = OmniLlmResultStatus::FAILED;
            per_user_llm_result_.finish_reason = OmniLlmFinishReason::ERROR;
            per_user_llm_result_.error = "duplex session initialization failed";
            per_user_llm_result_.stage = "SESSION_INIT";
            per_user_llm_ready_ = true;
            return true;
        }
        if (!duplex_prefill_frame(request.task, request.key.cancel_generation)) {
            const bool was_cancelled = cancelled(request.key.cancel_generation);
            per_user_llm_result_.status = was_cancelled
                ? OmniLlmResultStatus::CANCELLED : OmniLlmResultStatus::FAILED;
            per_user_llm_result_.finish_reason = was_cancelled
                ? OmniLlmFinishReason::CANCELLED : OmniLlmFinishReason::ERROR;
            per_user_llm_result_.error = was_cancelled
                ? "cancelled during LLM prefill" : "duplex prefill failed";
            per_user_llm_result_.stage = "PREFILL";
            per_user_llm_ready_ = true;
            return true;
        }
        per_user_llm_result_.timings = frame_timings_;
        per_user_llm_result_.timings.llm_prefill_ms = frame_timings_.llm_ms;
        if (duplex_ctx_->force_listen_used < duplex_ctx_->force_listen_count) {
            const auto decode_begin = std::chrono::steady_clock::now();
            ++duplex_ctx_->force_listen_used;
            if (duplex_ctx_->special_token_unit_end >= 0) {
                std::vector<llama_token> end{duplex_ctx_->special_token_unit_end};
                if (!eval_tokens(duplex_ctx_.get(), &params_, end, params_.n_batch,
                                 &duplex_ctx_->n_past)) {
                    per_user_llm_result_.status = OmniLlmResultStatus::FAILED;
                    per_user_llm_result_.finish_reason = OmniLlmFinishReason::ERROR;
                    per_user_llm_result_.error = "forced LISTEN unit terminator failed";
                    per_user_llm_result_.stage = "DECODE";
                    per_user_llm_ready_ = true;
                    return true;
                }
            }
            per_user_llm_result_.timings.llm_decode_ms =
                elapsed_ms(decode_begin, std::chrono::steady_clock::now());
            per_user_llm_result_.timings.llm_ms =
                per_user_llm_result_.timings.llm_prefill_ms +
                per_user_llm_result_.timings.llm_decode_ms;
            per_user_llm_result_.status = OmniLlmResultStatus::COMPLETE;
            per_user_llm_result_.finish_reason = OmniLlmFinishReason::LISTEN;
            per_user_llm_result_.stage = "DECODE";
            per_user_llm_result_.is_listen = true;
            per_user_llm_result_.n_past_after = duplex_ctx_->n_past;
            per_user_llm_ready_ = true;
        }
        return true;
    }

    bool per_user_llm_decode(const OmniLlmRequestKey & key) {
        if (!per_user_llm_pending_ || key.user_id != user_id_ ||
                key.frame_id != per_user_llm_result_.key.frame_id ||
                key.cancel_generation != per_user_llm_result_.key.cancel_generation) {
            return false;
        }
        if (per_user_llm_ready_) {
            return true;
        }
        const auto decode_begin = std::chrono::steady_clock::now();
        OmniLlmFinishReason finish_reason = OmniLlmFinishReason::NONE;
        const bool decode_ok = per_user_llm_task_.replay_decode_tokens.empty()
            ? duplex_decode_frame(per_user_llm_task_, key.cancel_generation,
                                  per_user_llm_result_.token_ids,
                                  per_user_llm_result_.hidden_states,
                                  per_user_llm_result_.text,
                                  per_user_llm_result_.is_speak,
                                  per_user_llm_result_.is_listen,
                                  finish_reason)
            : duplex_replay_frame(per_user_llm_task_, key.cancel_generation,
                                  per_user_llm_result_.token_ids,
                                  per_user_llm_result_.hidden_states,
                                  per_user_llm_result_.text,
                                  per_user_llm_result_.is_speak,
                                  per_user_llm_result_.is_listen,
                                  finish_reason);
        if (!decode_ok) {
            const bool was_cancelled = cancelled(key.cancel_generation);
            per_user_llm_result_.status = was_cancelled
                ? OmniLlmResultStatus::CANCELLED : OmniLlmResultStatus::FAILED;
            per_user_llm_result_.finish_reason = was_cancelled
                ? OmniLlmFinishReason::CANCELLED : OmniLlmFinishReason::ERROR;
            per_user_llm_result_.error = was_cancelled
                ? "cancelled during LLM decode" : "duplex decode failed";
            per_user_llm_result_.stage = "DECODE";
            per_user_llm_result_.n_past_after = duplex_ctx_->n_past;
            per_user_llm_ready_ = true;
            return true;
        }
        if (duplex_ctx_->special_token_unit_end >= 0) {
            std::vector<llama_token> end{duplex_ctx_->special_token_unit_end};
            if (!eval_tokens(duplex_ctx_.get(), &params_, end, params_.n_batch,
                             &duplex_ctx_->n_past)) {
                per_user_llm_result_.status = OmniLlmResultStatus::FAILED;
                per_user_llm_result_.finish_reason = OmniLlmFinishReason::ERROR;
                per_user_llm_result_.error = "duplex unit terminator failed";
                per_user_llm_result_.stage = "DECODE";
                per_user_llm_ready_ = true;
                return true;
            }
        }
        per_user_llm_result_.timings.llm_decode_ms =
            elapsed_ms(decode_begin, std::chrono::steady_clock::now());
        per_user_llm_result_.timings.llm_ms =
            per_user_llm_result_.timings.llm_prefill_ms +
            per_user_llm_result_.timings.llm_decode_ms;
        per_user_llm_result_.status = OmniLlmResultStatus::COMPLETE;
        per_user_llm_result_.finish_reason = finish_reason;
        per_user_llm_result_.stage = "DECODE";
        per_user_llm_result_.n_past_after = duplex_ctx_->n_past;
        per_user_llm_ready_ = true;
        return true;
    }

    bool per_user_llm_poll(const OmniLlmRequestKey & key, OmniLlmEngineResult & result, int) {
        if (!per_user_llm_pending_ || !per_user_llm_ready_ || key.user_id != user_id_ ||
                key.frame_id != per_user_llm_result_.key.frame_id ||
                key.cancel_generation != per_user_llm_result_.key.cancel_generation) {
            return false;
        }
        result = std::move(per_user_llm_result_);
        per_user_llm_pending_ = false;
        per_user_llm_ready_ = false;
        return true;
    }

    bool run_shared_llm_frame(OmniUserTask & task, uint64_t generation) {
        frame_timings_ = {};
        frame_timings_.frame_queue_wait_ms = elapsed_ms_from_ns(task.submit_ns);
        if (!run_media(task, generation)) {
            return false;
        }
        OmniLlmEngineRequest request;
        request.key.user_id = user_id_;
        request.key.frame_id = task.frame_id;
        request.key.cancel_generation = generation;
        request.task = task;
        request.queue_enter = std::chrono::steady_clock::now();
        if (!shared_llm_engine_->prefill(request) || !shared_llm_engine_->decode(request.key)) {
            fail(task, generation, "shared LLM request rejected", n_past_.load(), n_past_.load(), "LLM");
            return false;
        }

        OmniLlmEngineResult result;
        if (!shared_llm_engine_->poll(request.key, result, -1)) {
            fail(task, generation, "shared LLM result unavailable", n_past_.load(), n_past_.load(), "LLM");
            return false;
        }
        frame_timings_.llm_ms = result.timings.llm_ms;
        frame_timings_.llm_queue_wait_ms = result.timings.llm_queue_wait_ms;
        frame_timings_.llm_batch_wait_ms = result.timings.llm_batch_wait_ms;
        frame_timings_.llm_prefill_batch_wait_ms = result.timings.llm_prefill_batch_wait_ms;
        frame_timings_.llm_prefill_ms = result.timings.llm_prefill_ms;
        frame_timings_.llm_decode_ms = result.timings.llm_decode_ms;
        frame_timings_.first_token_ms = result.timings.first_token_ms;
        n_past_.store(result.n_past_after, std::memory_order_release);
        record_stage(stage_stats_.llm, frame_timings_.llm_ms);
        if (result.status == OmniLlmResultStatus::CANCELLED) {
            OmniMultiUserEvent event;
            event.user_id = user_id_;
            event.frame_id = task.frame_id;
            event.cancel_generation = generation_source_ ? generation_source_() : generation;
            event.kind = OmniMultiUserEventKind::CANCELLED;
            event.stage = "LLM";
            event.error = result.error;
            event.finish_reason = result.finish_reason;
            event.n_past_before = result.n_past_before;
            event.n_past_after = result.n_past_after;
            event.owner_thread_id = result.owner_thread_id;
            event.service_begin = service_begin_;
            event.service_end = std::chrono::steady_clock::now();
            publish(std::move(event));
            return false;
        }
        if (result.status != OmniLlmResultStatus::COMPLETE) {
            fail(task, generation, result.error.empty() ? "shared LLM execution failed" : result.error,
                 result.n_past_before, result.n_past_after, "LLM");
            return false;
        }

        std::vector<llama_token> & generated = result.token_ids;
        std::vector<float> & hidden_states = result.hidden_states;
        const bool is_speak = result.is_speak;
        const bool is_end_of_turn = result.is_end_of_turn;
        task.run_tts = is_speak && (!generated.empty() || is_end_of_turn) && tts_enabled_;
        task.run_t2w = is_speak && (!generated.empty() || is_end_of_turn) && t2w_enabled_;
        task.tts_embeddings.clear();
        task.tts_embedding_tokens = 0;
        task.tts_embedding_dim = 0;
        const auto tts_begin = std::chrono::steady_clock::now();
        if (is_speak && tts_enabled_ && (!generated.empty() || is_end_of_turn)) {
            const int tts_hidden = llama_model_n_embd(bundle_->tts_model());
            std::vector<float> projected(generated.size() * static_cast<size_t>(tts_hidden));
            if (!generated.empty() &&
                    (hidden_states.size() != generated.size() *
                     static_cast<size_t>(llama_model_n_embd(bundle_->llm_model())) ||
                     !tts_projector_semantic(tts_ctx_, hidden_states.data(),
                                             static_cast<int>(generated.size()),
                                             llama_model_n_embd(bundle_->llm_model()),
                                             projected.data(), tts_hidden))) {
                fail(task, generation, "TTS condition projection failed",
                     result.n_past_before, result.n_past_after, "TTS");
                return false;
            }
            task.tts_embeddings.resize(projected.size());
            for (size_t i = 0; i < generated.size(); ++i) {
                std::vector<float> emb(static_cast<size_t>(tts_hidden));
                if (!tts_emb_text(tts_ctx_, generated[i], emb.data(), tts_hidden)) {
                    fail(task, generation, "TTS text embedding failed",
                         result.n_past_before, result.n_past_after, "TTS");
                    return false;
                }
                normalize_l2_per_token(projected.data() + i * static_cast<size_t>(tts_hidden),
                                       1, tts_hidden);
                for (int j = 0; j < tts_hidden; ++j) {
                    task.tts_embeddings[i * static_cast<size_t>(tts_hidden) + static_cast<size_t>(j)] =
                        emb[static_cast<size_t>(j)] +
                        projected[i * static_cast<size_t>(tts_hidden) + static_cast<size_t>(j)];
                }
            }
            task.tts_embedding_tokens = static_cast<int32_t>(generated.size());
            task.tts_embedding_dim = tts_hidden;
        }

        OmniMultiUserEvent decision;
        decision.user_id = user_id_;
        decision.frame_id = task.frame_id;
        decision.cancel_generation = generation;
        decision.kind = OmniMultiUserEventKind::DECISION;
        decision.stage = "DECODE";
        decision.token_ids = result.token_ids;
        decision.text = result.text;
        decision.is_speak = result.is_speak;
        decision.silent = !result.is_speak;
        decision.is_end_of_turn = result.is_end_of_turn;
        decision.finish_reason = result.finish_reason;
        decision.timings = frame_timings_;
        decision.n_past_before = result.n_past_before;
        decision.n_past_after = result.n_past_after;
        decision.owner_thread_id = result.owner_thread_id;
        decision.service_begin = service_begin_;
        decision.service_end = std::chrono::steady_clock::now();
        publish(std::move(decision));
        if (result.is_speak && !result.text.empty()) {
            OmniMultiUserEvent text_event;
            text_event.user_id = user_id_;
            text_event.frame_id = task.frame_id;
            text_event.cancel_generation = generation;
            text_event.kind = OmniMultiUserEventKind::TEXT;
            text_event.stage = "DECODE";
            text_event.token_ids = result.token_ids;
            text_event.text = result.text;
            text_event.is_speak = true;
            text_event.finish_reason = result.finish_reason;
            text_event.n_past_before = result.n_past_before;
            text_event.n_past_after = result.n_past_after;
            text_event.owner_thread_id = result.owner_thread_id;
            text_event.service_begin = service_begin_;
            text_event.service_end = std::chrono::steady_clock::now();
            publish(std::move(text_event));
        }
        if (is_speak) {
            if (task.run_tts && !run_tts(task, generation, is_end_of_turn, tts_begin)) {
                return false;
            }
            if (task.run_t2w) {
                const auto t2w_begin = std::chrono::steady_clock::now();
                if (!run_t2w(task, generation, is_end_of_turn, t2w_begin)) {
                    return false;
                }
            }
        }

        OmniMultiUserEvent drained;
        drained.user_id = user_id_;
        drained.frame_id = task.frame_id;
        drained.cancel_generation = generation;
        drained.kind = OmniMultiUserEventKind::FRAME_DRAINED;
        drained.stage = "runtime";
        drained.is_speak = result.is_speak;
        drained.silent = !result.is_speak;
        drained.is_end_of_turn = result.is_end_of_turn;
        drained.finish_reason = result.finish_reason;
        frame_timings_.total_ms = elapsed_ms_from_ns(task.submit_ns);
        drained.timings = frame_timings_;
        drained.n_past_before = result.n_past_before;
        drained.n_past_after = result.n_past_after;
        drained.owner_thread_id = result.owner_thread_id;
        drained.service_begin = service_begin_;
        drained.service_end = std::chrono::steady_clock::now();
        publish(std::move(drained));
        record_frame_result(result.is_speak);
        return true;
    }

    bool run_duplex_frame(OmniUserTask & task, uint64_t generation) {
        if (shared_llm_engine_) {
            return run_shared_llm_frame(task, generation);
        }
        OmniLlmEngineRequest request;
        request.key.user_id = user_id_;
        request.key.frame_id = task.frame_id;
        request.key.cancel_generation = generation;
        request.task = task;
        request.queue_enter = service_begin_;
        if (!active_llm_engine_->prefill(request) || !active_llm_engine_->decode(request.key)) {
            fail(task, generation, "per-user LLM request rejected",
                 n_past_.load(), n_past_.load(), "LLM");
            return false;
        }
        OmniLlmEngineResult result;
        if (!active_llm_engine_->poll(request.key, result, -1)) {
            fail(task, generation, "per-user LLM result unavailable",
                 n_past_.load(), n_past_.load(), "LLM");
            return false;
        }
        if (result.status == OmniLlmResultStatus::CANCELLED) {
            OmniMultiUserEvent event;
            event.user_id = user_id_;
            event.frame_id = task.frame_id;
            event.cancel_generation = generation_source_ ? generation_source_() : generation;
            event.kind = OmniMultiUserEventKind::CANCELLED;
            event.stage = "LLM";
            event.error = result.error;
            event.finish_reason = result.finish_reason;
            event.n_past_before = result.n_past_before;
            event.n_past_after = result.n_past_after;
            event.service_begin = service_begin_;
            event.service_end = std::chrono::steady_clock::now();
            publish(std::move(event));
            return false;
        }
        if (result.status != OmniLlmResultStatus::COMPLETE) {
            fail(task, generation, result.error.empty() ? "per-user LLM execution failed" : result.error,
                 result.n_past_before, result.n_past_after,
                 result.stage.empty() ? "LLM" : result.stage.c_str());
            return false;
        }
        frame_timings_ = result.timings;
        record_stage(stage_stats_.llm, frame_timings_.llm_ms);
        n_past_.store(result.n_past_after, std::memory_order_release);
        std::vector<llama_token> & generated = result.token_ids;
        std::vector<float> & hidden_states = result.hidden_states;
        std::string & text = result.text;
        const bool is_speak = result.is_speak;
        const bool is_end_of_turn = result.is_end_of_turn;
        task.run_tts = is_speak && (!generated.empty() || is_end_of_turn) && tts_enabled_;
        task.run_t2w = is_speak && (!generated.empty() || is_end_of_turn) && t2w_enabled_;
        task.tts_embeddings.clear();
        task.tts_embedding_tokens = 0;
        task.tts_embedding_dim = 0;
        const auto tts_begin = std::chrono::steady_clock::now();
        if (is_speak && tts_enabled_ && (!generated.empty() || is_end_of_turn)) {
            const int tts_hidden = llama_model_n_embd(bundle_->tts_model());
            std::vector<float> projected(generated.size() * static_cast<size_t>(tts_hidden));
            if (!generated.empty() &&
                    !tts_projector_semantic(tts_ctx_, hidden_states.data(), static_cast<int>(generated.size()),
                                            llama_model_n_embd(bundle_->llm_model()), projected.data(), tts_hidden)) {
                fail(task, generation, "TTS condition projection failed", n_past_.load(), n_past_.load(), "TTS");
                return false;
            }
            task.tts_embeddings.resize(projected.size());
            for (size_t i = 0; i < generated.size(); ++i) {
                std::vector<float> emb(static_cast<size_t>(tts_hidden));
                if (!tts_emb_text(tts_ctx_, generated[i], emb.data(), tts_hidden)) return false;
                normalize_l2_per_token(projected.data() + i * static_cast<size_t>(tts_hidden), 1, tts_hidden);
                for (int j = 0; j < tts_hidden; ++j) {
                    task.tts_embeddings[i * static_cast<size_t>(tts_hidden) + static_cast<size_t>(j)] =
                        emb[static_cast<size_t>(j)] + projected[i * static_cast<size_t>(tts_hidden) + static_cast<size_t>(j)];
                }
            }
            task.tts_embedding_tokens = static_cast<int32_t>(generated.size());
            task.tts_embedding_dim = tts_hidden;
        }
        OmniMultiUserEvent decision;
        decision.user_id = user_id_;
        decision.frame_id = task.frame_id;
        decision.cancel_generation = generation;
        decision.kind = OmniMultiUserEventKind::DECISION;
        decision.stage = "DECODE";
        decision.token_ids = generated;
        decision.text = text;
        decision.is_speak = is_speak;
        decision.silent = !is_speak;
        decision.is_end_of_turn = result.is_end_of_turn;
        decision.finish_reason = result.finish_reason;
        decision.timings = frame_timings_;
        decision.n_past_before = result.n_past_before;
        decision.n_past_after = result.n_past_after;
        decision.service_begin = service_begin_;
        decision.service_end = std::chrono::steady_clock::now();
        publish(std::move(decision));
        if (is_speak && !text.empty()) {
            OmniMultiUserEvent text_event;
            text_event.user_id = user_id_;
            text_event.frame_id = task.frame_id;
            text_event.cancel_generation = generation;
            text_event.kind = OmniMultiUserEventKind::TEXT;
            text_event.stage = "DECODE";
            text_event.token_ids = generated;
            text_event.text = text;
            text_event.is_speak = true;
            text_event.finish_reason = result.finish_reason;
            text_event.n_past_before = result.n_past_before;
            text_event.n_past_after = result.n_past_after;
            text_event.service_begin = service_begin_;
            text_event.service_end = std::chrono::steady_clock::now();
            publish(std::move(text_event));
        }
        if (is_speak) {
            if (task.run_tts && !run_tts(task, generation, is_end_of_turn, tts_begin)) return false;
            if (task.run_t2w) {
                const auto t2w_begin = std::chrono::steady_clock::now();
                if (!run_t2w(task, generation, is_end_of_turn, t2w_begin)) return false;
            }
        }
        OmniMultiUserEvent drained;
        drained.user_id = user_id_;
        drained.frame_id = task.frame_id;
        drained.cancel_generation = generation;
        drained.kind = OmniMultiUserEventKind::FRAME_DRAINED;
        drained.stage = "runtime";
        drained.is_speak = is_speak;
        drained.silent = !is_speak;
        drained.is_end_of_turn = result.is_end_of_turn;
        drained.finish_reason = result.finish_reason;
        frame_timings_.total_ms = elapsed_ms_from_ns(task.submit_ns);
        drained.timings = frame_timings_;
        drained.n_past_before = result.n_past_before;
        drained.n_past_after = result.n_past_after;
        drained.service_begin = service_begin_;
        drained.service_end = std::chrono::steady_clock::now();
        publish(std::move(drained));
        record_frame_result(is_speak);
        return true;
    }

    bool publish_component(const OmniUserTask & task, uint64_t generation,
                           const char * stage, int32_t chunk_idx,
                           std::vector<float> embedding) {
        OmniMultiUserEvent event;
        event.user_id = user_id_;
        event.frame_id = task.frame_id;
        event.cancel_generation = generation;
        event.kind = OmniMultiUserEventKind::AUDIO_WINDOW;
        event.stage = stage;
        event.embedding = std::move(embedding);
        event.chunk_idx = chunk_idx;
        event.service_begin = service_begin_;
        event.service_end = std::chrono::steady_clock::now();
        publish(std::move(event));
        return !cancelled(generation);
    }

    bool run_media(OmniUserTask & task, uint64_t generation) {
        pending_media_embeddings_.clear();
        if (vpm_ctx_ && !task.input.img_fname.empty()) {
            const auto stage_begin = std::chrono::steady_clock::now();
            if (task.input.max_slice_nums >= 1) {
                vision_set_max_slice_nums(vpm_ctx_, task.input.max_slice_nums);
            }
            task.llm_vision_embeddings.clear();
            if (!omni_image_embed_make_chunks_with_filename(
                    vpm_ctx_, params_.cpuparams.n_threads, task.input.img_fname,
                    task.llm_vision_embeddings)) {
                fail(task, generation, "VPM encode failed", n_past_.load(), n_past_.load(), "VPM");
                return false;
            }
            const int32_t hidden = vision_n_mmproj_embd(vpm_ctx_);
            if (hidden != llama_model_n_embd(bundle_->llm_model())) {
                fail(task, generation, "VPM embedding dimension mismatch",
                     n_past_.load(), n_past_.load(), "VPM");
                return false;
            }
            task.llm_embedding_dim = hidden;
            frame_timings_.vpm_ms = elapsed_ms(stage_begin, std::chrono::steady_clock::now());
            record_stage(stage_stats_.vpm, frame_timings_.vpm_ms);
            for (size_t i = 0; i < task.llm_vision_embeddings.size(); ++i) {
                if (!publish_component(task, generation, "VPM", static_cast<int32_t>(i),
                                       task.llm_vision_embeddings[i])) {
                    return false;
                }
            }
        }
        if (apm_ctx_ && !task.input.aud_fname.empty()) {
            const auto stage_begin = std::chrono::steady_clock::now();
            std::unique_ptr<omni_embed, void (*)(omni_embed *)> embed(
                omni_audio_embed_make_with_filename(apm_ctx_, params_.cpuparams.n_threads,
                                                    task.input.aud_fname), omni_embed_free);
            if (!embed) {
                fail(task, generation, "APM encode failed", n_past_.load(), n_past_.load(), "APM");
                return false;
            }
            const int32_t hidden = audition_n_mmproj_embd(apm_ctx_);
            if (hidden != llama_model_n_embd(bundle_->llm_model())) {
                fail(task, generation, "APM embedding dimension mismatch",
                     n_past_.load(), n_past_.load(), "APM");
                return false;
            }
            task.llm_embedding_dim = hidden;
            task.llm_audio_embeddings.assign(
                embed->embed, embed->embed + static_cast<size_t>(embed->n_pos) * hidden);
            frame_timings_.apm_ms = elapsed_ms(stage_begin, std::chrono::steady_clock::now());
            record_stage(stage_stats_.apm, frame_timings_.apm_ms);
            if (!publish_component(task, generation, "APM", 0, task.llm_audio_embeddings)) {
                return false;
            }
        }
        if ((!task.llm_vision_embeddings.empty() || !task.llm_audio_embeddings.empty()) &&
                task.llm_embedding_dim != llama_model_n_embd(bundle_->llm_model())) {
            fail(task, generation, "shared LLM embedding dimension mismatch",
                 n_past_.load(), n_past_.load(), "PREFILL");
            return false;
        }
        return true;
    }

    bool run_tts(const OmniUserTask & task, uint64_t generation, bool is_end_of_turn,
                 const std::chrono::steady_clock::time_point & stage_begin) {
        if (!task.run_tts) {
            return true;
        }
        if (!tts_ctx_ || task.tts_embedding_tokens < 0 ||
                (task.tts_embedding_tokens > 0 && task.tts_embeddings.empty()) ||
                task.tts_embedding_dim <= 0) {
            fail(task, generation, "TTS requires per-user context and embeddings", n_past_.load(), n_past_.load(), "TTS");
            return false;
        }
        std::vector<int32_t> audio_tokens;
        if (!generate_audio_tokens_local(tts_ctx_, &params_, task.tts_embeddings,
                                         task.tts_embedding_tokens, task.tts_embedding_dim, 0,
                                         audio_tokens, is_end_of_turn, "", {}, task.frame_id)) {
            fail(task, generation, "TTS generation failed", n_past_.load(), n_past_.load(), "TTS");
            return false;
        }
        frame_timings_.tts_ms = elapsed_ms(stage_begin, std::chrono::steady_clock::now());
        if (!task.run_t2w) {
            frame_timings_.first_audio_ms = elapsed_ms_from_ns(task.submit_ns);
        }
        record_stage(stage_stats_.tts, frame_timings_.tts_ms);
        tts_audio_tokens_ = audio_tokens;
        OmniMultiUserEvent event;
        event.user_id = user_id_;
        event.frame_id = task.frame_id;
        event.cancel_generation = generation;
        event.kind = OmniMultiUserEventKind::AUDIO_WINDOW;
        event.stage = "TTS";
        event.chunk_idx = 0;
        event.audio_token_ids = std::move(audio_tokens);
        event.timings = frame_timings_;
        event.service_begin = service_begin_;
        event.service_end = std::chrono::steady_clock::now();
        publish(std::move(event));
        return !cancelled(generation);
    }

    bool run_t2w(const OmniUserTask & task, uint64_t generation, bool is_end_of_turn,
                 const std::chrono::steady_clock::time_point & stage_begin) {
        if (!task.run_t2w) {
            return true;
        }
        const std::vector<int32_t> & audio_tokens = task.audio_tokens.empty()
            ? tts_audio_tokens_ : task.audio_tokens;
        if (!t2w_session_ || audio_tokens.empty()) {
            fail(task, generation, "T2W requires per-user executor and audio tokens",
                 n_past_.load(), n_past_.load(), "T2W");
            return false;
        }
        std::vector<float> pcm;
        if (!t2w_session_->feed_window(audio_tokens, task.audio_tokens_final || is_end_of_turn, pcm)) {
            fail(task, generation, "T2W window failed", n_past_.load(), n_past_.load(), "T2W");
            return false;
        }
        frame_timings_.t2w_ms = elapsed_ms(stage_begin, std::chrono::steady_clock::now());
        frame_timings_.first_audio_ms = elapsed_ms_from_ns(task.submit_ns);
        record_stage(stage_stats_.t2w, frame_timings_.t2w_ms);
        OmniMultiUserEvent event;
        event.user_id = user_id_;
        event.frame_id = task.frame_id;
        event.cancel_generation = generation;
        event.kind = OmniMultiUserEventKind::AUDIO_WINDOW;
        event.stage = "T2W";
        event.chunk_idx = 0;
        event.pcm = std::move(pcm);
        event.timings = frame_timings_;
        event.service_begin = service_begin_;
        event.service_end = std::chrono::steady_clock::now();
        publish(std::move(event));
        return !cancelled(generation);
    }

    static uint64_t estimate_kv_bytes(const llama_model * model, uint32_t n_ctx) {
        if (!model || n_ctx == 0) {
            return 0;
        }
        const uint64_t embd = static_cast<uint64_t>(std::max(1, llama_model_n_embd(model)));
        const uint64_t layers = static_cast<uint64_t>(std::max(1, llama_model_n_layer(model)));
        // F32 K/V covers the common F16/F32 cache choices.  Backend graph and
        // allocator overhead are not included; this remains an estimate for
        // admission, while the context reports the actual limits.
        const uint64_t per_token = embd * layers * 2u * sizeof(float);
        if (per_token > std::numeric_limits<uint64_t>::max() / n_ctx) {
            return std::numeric_limits<uint64_t>::max();
        }
        return per_token * n_ctx;
    }

    bool cancelled(uint64_t generation) const {
        return generation_source_ && generation_source_() != generation;
    }

    void publish(OmniMultiUserEvent event) {
        if (event.owner_thread_id == 0) {
            event.owner_thread_id = owner_thread_id_.load(std::memory_order_acquire);
        }
        event_sink_(std::move(event));
    }

    bool decode_batch(const llama_token * tokens, size_t count, int32_t & n_past,
                     bool output_last_logits) {
#ifndef NDEBUG
        assert(owner_thread_id_.load(std::memory_order_acquire) == thread_id_value());
#endif
        if (count == 0) {
            return true;
        }

        const size_t batch_limit = std::max<size_t>(1, llama_n_batch(ctx_));
        size_t offset = 0;
        while (offset < count) {
            const size_t n = std::min(batch_limit, count - offset);
            llama_batch batch = llama_batch_init(static_cast<int32_t>(n), 0, 1);
            if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
                llama_batch_free(batch);
                return false;
            }
            batch.n_tokens = static_cast<int32_t>(n);
            for (size_t i = 0; i < n; ++i) {
                batch.token[i] = tokens[offset + i];
                batch.pos[i] = n_past + static_cast<int32_t>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (output_last_logits && offset + i + 1 == count) ? 1 : 0;
            }
            const int rc = llama_decode(ctx_, batch);
            llama_batch_free(batch);
            if (rc < 0) {
                return false;
            }
            n_past += static_cast<int32_t>(n);
            offset += n;
        }
        return true;
    }

    bool decode_one(llama_token token, int32_t & n_past, bool output_logits) {
        return decode_batch(&token, 1, n_past, output_logits);
    }

    bool decode_embeddings(const std::vector<float> & embeddings, int32_t & n_past) {
        const int32_t n_embd = llama_model_n_embd(bundle_->llm_model());
        if (n_embd <= 0 || embeddings.empty() || embeddings.size() % static_cast<size_t>(n_embd) != 0) {
            return false;
        }
        const size_t count = embeddings.size() / static_cast<size_t>(n_embd);
        const size_t batch_limit = std::max<size_t>(1, llama_n_batch(ctx_));
        for (size_t offset = 0; offset < count; ) {
            const size_t n = std::min(batch_limit, count - offset);
            llama_batch batch = llama_batch_init(static_cast<int32_t>(n), n_embd, 1);
            if (!batch.embd || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
                llama_batch_free(batch);
                return false;
            }
            batch.n_tokens = static_cast<int32_t>(n);
            std::memcpy(batch.embd, embeddings.data() + offset * static_cast<size_t>(n_embd),
                        n * static_cast<size_t>(n_embd) * sizeof(float));
            for (size_t i = 0; i < n; ++i) {
                batch.pos[i] = n_past + static_cast<int32_t>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (offset + i + 1 == count) ? 1 : 0;
            }
            const int rc = llama_decode(ctx_, batch);
            llama_batch_free(batch);
            if (rc < 0) {
                return false;
            }
            n_past += static_cast<int32_t>(n);
            offset += n;
        }
        return true;
    }

    void fail(const OmniUserTask & task, uint64_t generation, const std::string & message,
              int32_t n_past_before, int32_t n_past_after, const char * stage = "runtime") {
        OmniMultiUserEvent event;
        event.user_id = user_id_;
        event.frame_id = task.frame_id;
        event.cancel_generation = generation;
        event.kind = OmniMultiUserEventKind::FAILED;
        event.stage = stage;
        event.error = message;
        event.n_past_before = n_past_before;
        event.n_past_after = n_past_after;
        event.service_end = std::chrono::steady_clock::now();
        publish(std::move(event));
    }

    void process(OmniUserTask task) {
        const uint64_t generation = task.cancel_generation;
        const auto service_begin = std::chrono::steady_clock::now();
        service_begin_ = service_begin;
        int32_t n_past_before = n_past_.load(std::memory_order_acquire);
        int32_t n_past = n_past_before;

        if (cancelled(generation)) {
            OmniMultiUserEvent event;
            event.user_id = user_id_;
            event.frame_id = task.frame_id;
            event.cancel_generation = generation_source_ ? generation_source_() : generation;
            event.kind = OmniMultiUserEventKind::CANCELLED;
            event.stage = "LLM";
            event.error = "cancelled before execution";
            publish(std::move(event));
            finish_sink_();
            return;
        }
        if (deadline_expired(task.deadline_ns)) {
            fail(task, generation, "deadline expired before execution", n_past_before, n_past);
            finish_sink_();
            return;
        }

        // A multi-user frame follows the duplex transaction: build the unit,
        // prefill media/text, decode control tokens, then run SPEAK-only TTS/T2W.
        // LISTEN frames stop after the decision and are intentionally silent.
        if (!run_duplex_frame(task, generation)) {
            finish_sink_();
            return;
        }
        finish_sink_();
    }

    void run() {
        owner_thread_id_.store(thread_id_value(), std::memory_order_release);
        for (;;) {
            OmniUserTask task;
            {
                std::unique_lock<std::mutex> lock(queue_mtx_);
                queue_cv_.wait(lock, [&] { return stop_requested_ || !queue_.empty(); });
                if (stop_requested_ && queue_.empty()) {
                    break;
                }
                if (queue_.empty()) {
                    continue;
                }
                task = std::move(queue_.front());
                queue_.pop_front();
                active_.store(true, std::memory_order_release);
            }
            try {
                process(task);
            } catch (const std::exception & e) {
                fail(task, task.cancel_generation, std::string("worker exception: ") + e.what(),
                     n_past_.load(std::memory_order_acquire), n_past_.load(std::memory_order_acquire));
                finish_sink_();
            } catch (...) {
                fail(task, task.cancel_generation, "worker exception", n_past_.load(std::memory_order_acquire),
                     n_past_.load(std::memory_order_acquire));
                finish_sink_();
            }
            active_.store(false, std::memory_order_release);
            publish_deferred_cancellations();
            idle_cv_.notify_all();
        }
        active_.store(false, std::memory_order_release);
        publish_deferred_cancellations();
        idle_cv_.notify_all();
    }

    void publish_deferred_cancellations() {
        std::deque<std::pair<OmniUserTask, uint64_t>> cancelled;
        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            cancelled.swap(cancelled_queue_);
        }
        for (const auto & entry : cancelled) {
            OmniMultiUserEvent event;
            event.user_id = user_id_;
            event.frame_id = entry.first.frame_id;
            event.cancel_generation = entry.second;
            event.kind = OmniMultiUserEventKind::CANCELLED;
            event.stage = "runtime";
            event.error = "cancelled before execution";
            publish(std::move(event));
            finish_sink_();
        }
    }

    uint64_t user_id_ = 0;
    std::shared_ptr<OmniSharedModelBundle> bundle_;
    common_params params_;
    std::shared_ptr<OmniLlmEngine> shared_llm_engine_;
    std::unique_ptr<OmniLlmEngine> per_user_llm_engine_;
    OmniLlmEngine * active_llm_engine_ = nullptr;
    EventSink event_sink_;
    FinishSink finish_sink_;
    GenerationSource generation_source_;
    uint32_t default_max_new_tokens_ = 26;

    llama_context * ctx_ = nullptr;
    common_sampler * sampler_ = nullptr;
    vision_ctx * vpm_ctx_ = nullptr;
    audition_ctx * apm_ctx_ = nullptr;
    omni_context * tts_ctx_ = nullptr;
    std::unique_ptr<omni::flow::Token2WavSession> t2w_session_;
    std::unique_ptr<omni_context> duplex_ctx_;
    OmniUserSessionConfig session_config_;
    bool tts_enabled_ = false;
    bool t2w_enabled_ = false;
    bool session_initialized_ = false;
    bool fused_prefill_available_ = false;
    bool segmented_prefill_debug_ = false;
    std::chrono::steady_clock::time_point service_begin_{};
    OmniMultiUserFrameTimings frame_timings_{};
    OmniUserTask per_user_llm_task_;
    OmniLlmEngineResult per_user_llm_result_;
    bool per_user_llm_pending_ = false;
    bool per_user_llm_ready_ = false;
    OmniMultiUserStageStats stage_stats_{};
    mutable std::mutex stage_stats_mtx_;
    std::vector<int32_t> tts_audio_tokens_;
    std::vector<float> pending_media_embeddings_;
    bool valid_ = false;
    std::string error_;
    uint32_t n_ctx_ = 0;
    uint32_t n_ctx_seq_ = 0;
    uint32_t n_seq_max_ = 0;
    uint64_t kv_bytes_estimate_ = 0;
    std::atomic<int32_t> n_past_{0};
    std::atomic<uint64_t> owner_thread_id_{0};
    std::atomic<bool> active_{false};

    mutable std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    std::condition_variable idle_cv_;
    std::deque<OmniUserTask> queue_;
    std::deque<std::pair<OmniUserTask, uint64_t>> cancelled_queue_;
    bool stop_requested_ = false;
    std::thread thread_;
};

struct OmniMultiContextRuntime::UserState {
    uint64_t user_id = 0;
    OmniUserSessionConfig config;
    std::atomic<uint64_t> cancel_generation{0};
    uint64_t last_frame_id = 0;
    bool closing = false;
    mutable std::mutex mtx;
    std::condition_variable cv;
    std::deque<OmniMultiUserEvent> events;
    std::unique_ptr<OmniMultiContextUserWorker> worker;
};

OmniSharedModelBundle::OmniSharedModelBundle(llama_model * model, bool owns_model)
    : model_(model), owns_model_(owns_model) {
    if (model_) {
        model_bytes_ = llama_model_size(model_);
        weights_.push_back({"llm", model_, static_cast<size_t>(model_bytes_), "model"});
    }
}

OmniSharedModelBundle::~OmniSharedModelBundle() {
    if (vpm_) {
        vision_shared_free(vpm_);
        vpm_ = nullptr;
    }
    if (apm_) {
        audition_shared_free(apm_);
        apm_ = nullptr;
    }
    if (tts_template_) {
        free(tts_template_->emb_code_weight);
        free(tts_template_->emb_text_weight);
        free(tts_template_->projector_semantic_linear1_weight);
        free(tts_template_->projector_semantic_linear1_bias);
        free(tts_template_->projector_semantic_linear2_weight);
        free(tts_template_->projector_semantic_linear2_bias);
        free(tts_template_->head_code_weight);
        delete tts_template_;
        tts_template_ = nullptr;
    }
    if (tts_model_) {
        llama_model_free(tts_model_);
        tts_model_ = nullptr;
    }
    if (owns_model_ && model_) {
        llama_model_free(model_);
    }
}

bool OmniSharedModelBundle::initialize_components(const common_params & params,
                                                   const std::string & tts_bin_dir,
                                                   bool need_vpm, bool need_apm, bool need_tts,
                                                   std::string * error) {
    auto fail = [&](const std::string & message) {
        if (error) {
            *error = message;
        }
        return false;
    };
    try {
        if (need_vpm && !vpm_) {
            if (params.vpm_model.empty()) {
                return fail("VPM model path is empty");
            }
            vpm_ = vision_shared_init(params.vpm_model.c_str(),
                                      vision_context_params{true, GGML_LOG_LEVEL_WARN, nullptr});
            if (!vpm_) {
                return fail("failed to load shared VPM model: " + params.vpm_model);
            }
            vision_shared_set_batch_encode(vpm_, params.vpm_batch_encode);
            if (!params.vision_coreml_model_path.empty()) {
                vision_shared_set_coreml_model_path(vpm_, params.vision_coreml_model_path.c_str());
            }
            weights_.push_back({"vpm", vpm_, 0, "shared"});
        }
        if (need_apm && !apm_) {
            if (params.apm_model.empty()) {
                return fail("APM model path is empty");
            }
            apm_ = audition_shared_init(params.apm_model.c_str(),
                                        audition_context_params{true, GGML_LOG_LEVEL_WARN});
            if (!apm_) {
                return fail("failed to load shared APM model: " + params.apm_model);
            }
            weights_.push_back({"apm", apm_, 0, "shared"});
        }
        if (need_tts && !tts_model_) {
            if (params.tts_model.empty()) {
                return fail("TTS model path is empty");
            }
            common_params mutable_params = params;
            llama_model_params model_params = common_model_params_to_llama(mutable_params);
            model_params.partial_load = true;
            tts_model_ = llama_model_load_from_file(params.tts_model.c_str(), model_params);
            if (!tts_model_) {
                return fail("failed to load shared TTS model: " + params.tts_model);
            }
            tts_template_ = new omni_context();
            tts_template_->model_tts = tts_model_;
            tts_template_->duplex_mode = true;
            tts_template_->params = const_cast<common_params *>(&params);
            if (!load_tts_weights_from_gguf(tts_template_, params.tts_model.c_str())) {
                return fail("failed to load shared TTS projection weights: " + params.tts_model);
            }
            tts_bin_dir_ = tts_bin_dir;
            weights_.push_back({"tts", tts_model_, llama_model_size(tts_model_), "model"});
        }
    } catch (const std::exception & e) {
        return fail(e.what());
    }
    return true;
}

std::shared_ptr<OmniSharedModelBundle> OmniSharedModelBundle::from_model(llama_model * model,
                                                                          bool take_ownership) {
    if (!model) {
        return nullptr;
    }
    return std::shared_ptr<OmniSharedModelBundle>(
        new OmniSharedModelBundle(model, take_ownership));
}

std::shared_ptr<OmniSharedModelBundle> OmniSharedModelBundle::load(const common_params & params,
                                                                   std::string * error) {
    if (params.model.path.empty()) {
        if (error) {
            *error = "model path is empty";
        }
        return nullptr;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    common_params mutable_params = params;
    llama_model_params model_params = common_model_params_to_llama(mutable_params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        if (error) {
            *error = "failed to load shared LLM model: " + params.model.path;
        }
        return nullptr;
    }
    return std::shared_ptr<OmniSharedModelBundle>(new OmniSharedModelBundle(model, true));
}

OmniMultiContextRuntime::OmniMultiContextRuntime(std::shared_ptr<OmniSharedModelBundle> bundle,
                                                 const common_params & params,
                                                 OmniMultiUserConfig config,
                                                 std::shared_ptr<OmniLlmEngine> llm_engine)
    : bundle_(std::move(bundle)), params_(params), config_(config),
      llm_engine_(std::move(llm_engine)) {
    if (!bundle_ || !bundle_->llm_model()) {
        error_ = "shared model bundle is unavailable";
        running_.store(false, std::memory_order_release);
        return;
    }
    if (config_.max_users == 0 || config_.max_pending_frames_per_user == 0) {
        error_ = "max_users and max_pending_frames_per_user must be positive";
        running_.store(false, std::memory_order_release);
        return;
    }
    if (config_.enable_t2w && config_.tts_bin_dir.empty()) {
        error_ = "T2W is enabled but tts_bin_dir is empty";
        running_.store(false, std::memory_order_release);
        return;
    }
    if (llm_engine_ && config_.llm_execution_policy != OmniLlmExecutionPolicy::CONTINUOUS_BATCH) {
        error_ = "an injected LLM engine requires CONTINUOUS_BATCH policy";
        running_.store(false, std::memory_order_release);
        return;
    }
    std::string component_error;
    if (!bundle_->initialize_components(params_, config_.tts_bin_dir,
                                        config_.enable_vpm, config_.enable_apm,
                                        config_.enable_tts,
                                        &component_error)) {
        error_ = component_error;
        running_.store(false, std::memory_order_release);
        return;
    }
    if (config_.max_total_pending_frames == 0) {
        const uint64_t total = static_cast<uint64_t>(config_.max_users) *
                               config_.max_pending_frames_per_user;
        config_.max_total_pending_frames = total > std::numeric_limits<uint32_t>::max()
            ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(total);
    }
    if (config_.llm_execution_policy == OmniLlmExecutionPolicy::CONTINUOUS_BATCH) {
        if (!llm_engine_) {
            llm_engine_ = omni_create_continuous_batch_llm_engine(bundle_, params_, config_);
        }
        if (!llm_engine_->valid()) {
            error_ = llm_engine_->error();
            llm_engine_.reset();
            running_.store(false, std::memory_order_release);
        }
    }
}

OmniMultiContextRuntime::~OmniMultiContextRuntime() {
    shutdown();
}

bool OmniMultiContextRuntime::valid() const {
    return running_.load(std::memory_order_acquire) && bundle_ && bundle_->llm_model();
}

uint64_t OmniMultiContextRuntime::estimated_user_bytes() const {
    if (!bundle_ || !bundle_->llm_model()) {
        return 0;
    }
    const uint32_t n_ctx = params_.n_ctx > 0
        ? static_cast<uint32_t>(params_.n_ctx)
        : static_cast<uint32_t>(std::max(1, llama_model_n_ctx_train(bundle_->llm_model())));
    const uint64_t embd = static_cast<uint64_t>(std::max(1, llama_model_n_embd(bundle_->llm_model())));
    const uint64_t layers = static_cast<uint64_t>(std::max(1, llama_model_n_layer(bundle_->llm_model())));
    const uint64_t kv = static_cast<uint64_t>(n_ctx) * embd * layers * 2u * sizeof(float);
    const uint64_t compute = static_cast<uint64_t>(std::max(1, params_.n_batch)) * embd * sizeof(float) * 2u;
    uint64_t total = kv > std::numeric_limits<uint64_t>::max() - compute
        ? std::numeric_limits<uint64_t>::max() : kv + compute;
    if (config_.enable_tts && bundle_->tts_model()) {
        const uint64_t tts_embd = static_cast<uint64_t>(std::max(1, llama_model_n_embd(bundle_->tts_model())));
        const uint64_t tts_kv = static_cast<uint64_t>(n_ctx) * tts_embd * layers * 2u * sizeof(float);
        total = tts_kv > std::numeric_limits<uint64_t>::max() - total
            ? std::numeric_limits<uint64_t>::max() : total + tts_kv;
    }
    if (config_.enable_apm && bundle_->apm_model()) {
        // Whisper K/V is F16 and sized by the audition model audio context.
        const uint64_t apm_state = static_cast<uint64_t>(1500) * embd * layers * 2u * sizeof(uint16_t);
        total = apm_state > std::numeric_limits<uint64_t>::max() - total
            ? std::numeric_limits<uint64_t>::max() : total + apm_state;
    }
    return total;
}

bool OmniMultiContextRuntime::open_user(uint64_t user_id, const OmniUserSessionConfig & session_config) {
    if (!valid() || user_id == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    if (!running_.load(std::memory_order_acquire)) {
        error_ = "runtime is stopped";
        return false;
    }
    if (users_.find(user_id) != users_.end()) {
        error_ = "user is already open";
        return false;
    }
    if (users_.size() >= config_.max_users) {
        error_ = "maximum user count reached";
        return false;
    }
    const uint64_t per_user = config_.per_user_memory_bytes
        ? config_.per_user_memory_bytes : estimated_user_bytes();
    if (config_.max_memory_bytes) {
        if (bundle_->model_bytes() > config_.max_memory_bytes) {
            error_ = "shared model exceeds admission memory budget";
            return false;
        }
        const uint64_t remaining = config_.max_memory_bytes - bundle_->model_bytes();
        if (per_user > remaining / (users_.size() + 1)) {
            error_ = "per-user memory estimate exceeds admission budget";
            return false;
        }
    }

    auto user = std::make_shared<UserState>();
    user->user_id = user_id;
    user->config = session_config;
    if (llm_engine_ && !llm_engine_->open_user(user_id, session_config)) {
        error_ = session_config.voice_audio.empty()
            ? "failed to allocate shared LLM sequence"
            : "continuous batching does not support voice reference embedding yet";
        return false;
    }
    std::weak_ptr<UserState> weak_user = user;
    user->worker = std::make_unique<OmniMultiContextUserWorker>(
        user_id, bundle_, params_, llm_engine_,
        [this, weak_user](OmniMultiUserEvent event) {
            if (auto locked = weak_user.lock()) {
                publish_event(locked, std::move(event));
            }
        },
        [this, weak_user]() {
            if (auto locked = weak_user.lock()) {
                on_task_finished(locked);
            }
        },
        [weak_user]() -> uint64_t {
            if (auto locked = weak_user.lock()) {
                return locked->cancel_generation.load(std::memory_order_acquire);
            }
            return 0;
        },
                               config_.default_max_new_tokens,
                               config_.duplex_force_listen_count,
                               session_config,
                               config_.enable_vpm, config_.enable_apm,
                               config_.enable_tts, config_.enable_t2w,
                               config_.tts_bin_dir);
    if (!user->worker->valid()) {
        error_ = user->worker->error();
        if (llm_engine_) {
            llm_engine_->close_user(user_id);
        }
        return false;
    }
    user->worker->start();
    users_.emplace(user_id, std::move(user));
    return true;
}

OmniSubmitStatus OmniMultiContextRuntime::submit(const OmniUserTask & input_task) {
    if (!valid()) {
        return OmniSubmitStatus::RUNTIME_STOPPED;
    }
    if (input_task.tokens.empty() && input_task.text.empty() &&
            input_task.llm_vision_embeddings.empty() && input_task.llm_audio_embeddings.empty() &&
            input_task.input.img_fname.empty() && input_task.input.aud_fname.empty() &&
            !input_task.run_tts && !input_task.run_t2w) {
        return OmniSubmitStatus::INVALID_TASK;
    }
    if ((!input_task.input.img_fname.empty() && !config_.enable_vpm) ||
            (!input_task.input.aud_fname.empty() && !config_.enable_apm) ||
            (input_task.run_tts && !config_.enable_tts) ||
            (input_task.run_t2w && !config_.enable_t2w)) {
        return OmniSubmitStatus::INVALID_TASK;
    }

    std::shared_ptr<UserState> user;
    OmniUserTask task = input_task;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = users_.find(input_task.user_id);
        if (it == users_.end()) {
            return OmniSubmitStatus::USER_NOT_FOUND;
        }
        user = it->second;
        std::lock_guard<std::mutex> user_lock(user->mtx);
        if (user->closing) {
            return OmniSubmitStatus::USER_CLOSING;
        }
        if (input_task.frame_id == 0) {
            task.frame_id = user->last_frame_id + 1;
        } else if (input_task.frame_id <= user->last_frame_id) {
            return input_task.frame_id == user->last_frame_id
                ? OmniSubmitStatus::DUPLICATE_FRAME : OmniSubmitStatus::OUT_OF_ORDER_FRAME;
        } else if (input_task.frame_id != user->last_frame_id + 1) {
            return OmniSubmitStatus::OUT_OF_ORDER_FRAME;
        }

        const size_t user_depth = user->worker->queue_depth() +
                                  (user->worker->stats().active ? 1u : 0u);
        if (user_depth >= config_.max_pending_frames_per_user) {
            return OmniSubmitStatus::USER_QUEUE_FULL;
        }
        if (total_pending_.load(std::memory_order_acquire) >= config_.max_total_pending_frames) {
            return OmniSubmitStatus::RUNTIME_QUEUE_FULL;
        }
        task.cancel_generation = user->cancel_generation.load(std::memory_order_acquire);
        task.submit_ns = steady_now_ns();
        task.deadline_ns = task.deadline_ns ? task.deadline_ns : user->config.deadline_ns;
        task.user_id = input_task.user_id;
        user->last_frame_id = task.frame_id;
        total_pending_.fetch_add(1, std::memory_order_acq_rel);
    }

    if (!user->worker->enqueue(std::move(task))) {
        total_pending_.fetch_sub(1, std::memory_order_acq_rel);
        return OmniSubmitStatus::USER_CLOSING;
    }
    return OmniSubmitStatus::ACCEPTED;
}

void OmniMultiContextRuntime::publish_event(const std::shared_ptr<UserState> & user,
                                            OmniMultiUserEvent event) {
    if (!user) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(user->mtx);
        // Re-check while holding the event lock.  A cancellation can race the
        // first generation read; stale output must never enter the queue after
        // the cancellation becomes visible.
        const uint64_t current_generation = user->cancel_generation.load(std::memory_order_acquire);
        if (event.kind != OmniMultiUserEventKind::CANCELLED &&
                event.cancel_generation != current_generation) {
            return;
        }
        user->events.push_back(std::move(event));
    }
    user->cv.notify_all();
}

void OmniMultiContextRuntime::on_task_finished(const std::shared_ptr<UserState> & user) {
    total_pending_.fetch_sub(1, std::memory_order_acq_rel);
    if (user) {
        user->cv.notify_all();
    }
}

bool OmniMultiContextRuntime::wait_event(uint64_t user_id, OmniMultiUserEvent & event, int timeout_ms) {
    std::shared_ptr<UserState> user;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = users_.find(user_id);
        if (it == users_.end()) {
            return false;
        }
        user = it->second;
    }

    std::unique_lock<std::mutex> lock(user->mtx);
    auto ready = [&] { return !user->events.empty() || (!running_ && user->events.empty()); };
    if (timeout_ms < 0) {
        user->cv.wait(lock, ready);
    } else if (timeout_ms > 0 && !user->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready)) {
        return false;
    } else if (timeout_ms == 0 && !ready()) {
        return false;
    }
    if (user->events.empty()) {
        return false;
    }
    event = std::move(user->events.front());
    user->events.pop_front();
    return true;
}

bool OmniMultiContextRuntime::cancel_user(uint64_t user_id) {
    std::shared_ptr<UserState> user;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = users_.find(user_id);
        if (it == users_.end()) {
            return false;
        }
        user = it->second;
    }
    const uint64_t generation = user->cancel_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
    user->worker->cancel_pending(generation);
    return true;
}

bool OmniMultiContextRuntime::wait_user_idle(const std::shared_ptr<UserState> & user, int timeout_ms) {
    if (!user) {
        return false;
    }
    return user->worker->wait_idle(timeout_ms);
}

bool OmniMultiContextRuntime::close_user(uint64_t user_id, OmniUserCloseMode mode, int timeout_ms) {
    std::shared_ptr<UserState> user;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = users_.find(user_id);
        if (it == users_.end()) {
            return false;
        }
        user = it->second;
        {
            std::lock_guard<std::mutex> user_lock(user->mtx);
            user->closing = true;
        }
    }

    if (mode == OmniUserCloseMode::CANCEL) {
        cancel_user(user_id);
    }
    if (!wait_user_idle(user, timeout_ms)) {
        return false;
    }
    user->worker->stop(mode == OmniUserCloseMode::CANCEL);
    user->worker->join();
    if (llm_engine_) {
        llm_engine_->drain(user_id, timeout_ms);
        llm_engine_->close_user(user_id);
    }

    std::lock_guard<std::mutex> lock(mtx_);
    users_.erase(user_id);
    return true;
}

void OmniMultiContextRuntime::shutdown() {
    bool was_running = running_.exchange(false, std::memory_order_acq_rel);
    std::vector<std::shared_ptr<UserState>> users;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!was_running && users_.empty() && !llm_engine_) {
            return;
        }
        for (auto & entry : users_) {
            {
                std::lock_guard<std::mutex> user_lock(entry.second->mtx);
                entry.second->closing = true;
            }
            users.push_back(entry.second);
        }
    }
    for (const auto & user : users) {
        user->cancel_generation.fetch_add(1, std::memory_order_acq_rel);
        user->worker->stop(true);
    }
    for (const auto & user : users) {
        user->worker->join();
    }
    if (llm_engine_) {
        for (const auto & user : users) {
            llm_engine_->drain(user->user_id, -1);
            llm_engine_->close_user(user->user_id);
        }
        llm_engine_->shutdown();
    }
    {
        std::lock_guard<std::mutex> lock(mtx_);
        users_.clear();
    }
}

bool OmniMultiContextRuntime::get_user_stats(uint64_t user_id,
                                             OmniMultiContextUserStats & stats) const {
    std::shared_ptr<UserState> user;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = users_.find(user_id);
        if (it == users_.end()) {
            return false;
        }
        user = it->second;
    }
    stats = user->worker->stats();
    {
        std::lock_guard<std::mutex> lock(user->mtx);
        stats.closing = user->closing;
    }
    return true;
}

bool OmniMultiContextRuntime::get_user_stage_stats(uint64_t user_id,
                                                   OmniMultiUserStageStats & stats) const {
    std::shared_ptr<UserState> user;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = users_.find(user_id);
        if (it == users_.end()) {
            return false;
        }
        user = it->second;
    }
    stats = user->worker->stage_stats();
    return true;
}

OmniLlmBatchStats OmniMultiContextRuntime::get_llm_batch_stats() const {
    return llm_engine_ ? llm_engine_->batch_stats() : OmniLlmBatchStats{};
}

uint32_t OmniMultiContextRuntime::user_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return static_cast<uint32_t>(users_.size());
}

uint64_t OmniMultiContextRuntime::model_bytes() const {
    return bundle_ ? bundle_->model_bytes() : 0;
}
