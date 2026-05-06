#include "omni-encode-stage.h"

#include "audition.h"
#include "common/common.h"
#include "omni-impl.h"
#include "omni-llm-stage.h"
#include "omni-log.h"
#include "omni.h"
#include "vision.h"

#include "ggml-backend.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>

namespace {

constexpr const char * kOmniBackendProfileTag = "TAG=OMNI_BACKEND_PROFILE";

double omni_encode_timing_elapsed_ms(const std::chrono::high_resolution_clock::time_point & start,
                                     const std::chrono::high_resolution_clock::time_point & end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool omni_encode_backend_profile_enabled() {
    static const bool enabled = std::getenv("OMNI_BACKEND_PROFILE") != nullptr;
    return enabled;
}

void omni_encode_backend_profile_discard_span(OmniBackendProfileSpan & span) {
    for (auto & event_pair : span.events) {
        ggml_backend_event_free(event_pair.start);
        ggml_backend_event_free(event_pair.end);
        event_pair.start   = nullptr;
        event_pair.end     = nullptr;
        event_pair.backend = nullptr;
    }
    span.events.clear();
}

bool omni_encode_backend_profile_begin_span(ggml_backend_sched_t      sched,
                                           OmniBackendProfileStage   stage,
                                           int                       chunk_idx,
                                           int                       n_tokens,
                                           OmniBackendProfileSpan &  out_span) {
    out_span          = {};
    out_span.stage     = stage;
    out_span.chunk_idx = chunk_idx;
    out_span.n_tokens  = n_tokens;

    if (!omni_encode_backend_profile_enabled() || sched == nullptr) {
        return false;
    }

    const int n_backends = ggml_backend_sched_get_n_backends(sched);
    out_span.events.reserve(n_backends);

    for (int i = 0; i < n_backends; ++i) {
        ggml_backend_t     backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t device  = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
        if (backend == nullptr || device == nullptr || !ggml_backend_dev_supports_timed_events(device)) {
            continue;
        }

        ggml_backend_event_t start = ggml_backend_event_new_timed(device);
        ggml_backend_event_t end   = ggml_backend_event_new_timed(device);
        if (start == nullptr || end == nullptr) {
            ggml_backend_event_free(start);
            ggml_backend_event_free(end);
            continue;
        }

        ggml_backend_event_record(start, backend);
        out_span.events.push_back({
            /* .backend = */ backend,
            /* .start   = */ start,
            /* .end     = */ end,
            /* .backend_name = */ ggml_backend_dev_name(device),
        });
    }

    return !out_span.events.empty();
}

void omni_encode_backend_profile_end_span(OmniBackendProfileSpan & span, double submit_ms) {
    span.submit_ms = submit_ms;

    for (auto & event_pair : span.events) {
        if (event_pair.backend != nullptr && event_pair.end != nullptr) {
            ggml_backend_event_record(event_pair.end, event_pair.backend);
        }
    }
}

void omni_encode_backend_profile_log_span(const OmniBackendProfileSpan & span, double device_ms,
                                         const std::string & backend_breakdown) {
    const char * stage_name = "unknown";
    switch (span.stage) {
        case OmniBackendProfileStage::encode_vit_embed:   stage_name = "encode_vit_embed_device"; break;
        case OmniBackendProfileStage::encode_audio_embed: stage_name = "encode_audio_embed_device"; break;
        default: break;
    }

    const double queue_wait_ms = (device_ms >= 0.0 && span.submit_ms >= 0.0) ? span.submit_ms - device_ms : -1.0;
    print_with_timestamp(
        "%s stage=%s chunk=%d submit_ms=%.3f device_ms=%.3f queue_wait_ms=%.3f n_tokens=%d backends=%s\n",
        kOmniBackendProfileTag, stage_name, span.chunk_idx, span.submit_ms, device_ms, queue_wait_ms, span.n_tokens,
        backend_breakdown.c_str());
}

void omni_encode_backend_profile_finalize_and_log(OmniBackendProfileSpan & span) {
    double      device_ms         = -1.0;
    std::string backend_breakdown;

    bool first = true;
    for (auto & event_pair : span.events) {
        ggml_backend_event_synchronize(event_pair.start);
        ggml_backend_event_synchronize(event_pair.end);
        const float backend_ms = ggml_backend_event_elapsed_ms(event_pair.start, event_pair.end);
        if (backend_ms >= 0.0f) {
            device_ms = device_ms < 0.0 ? (double) backend_ms : std::max(device_ms, (double) backend_ms);
            if (!first) {
                backend_breakdown += ",";
            }
            first = false;

            char ms_buf[32];
            snprintf(ms_buf, sizeof(ms_buf), "%.3f", backend_ms);
            backend_breakdown += event_pair.backend_name;
            backend_breakdown += "=";
            backend_breakdown += ms_buf;
        }

        ggml_backend_event_free(event_pair.start);
        ggml_backend_event_free(event_pair.end);
        event_pair.start   = nullptr;
        event_pair.end     = nullptr;
        event_pair.backend = nullptr;
    }
    span.events.clear();

    if (device_ms >= 0.0) {
        omni_encode_backend_profile_log_span(span, device_ms, backend_breakdown);
    }
}

void omni_encode_stage_note_vit(struct omni_context * ctx_omni, int chunk_idx, double ms) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.vit_embedding_ms            = timing.vit_embedding_ms < 0.0 ? ms : timing.vit_embedding_ms + ms;
}

void omni_encode_stage_note_audio(struct omni_context * ctx_omni, int chunk_idx, double ms) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.audio_embedding_ms          = timing.audio_embedding_ms < 0.0 ? ms : timing.audio_embedding_ms + ms;
}

static void omni_encode_stage_note_audio_sub(struct omni_context * ctx_omni, int chunk_idx,
                                             double file_io_ms, double preprocess_ms, double encode_ms) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.audio_file_io_ms    = file_io_ms;
    timing.audio_preprocess_ms = preprocess_ms;
    timing.audio_encode_ms     = encode_ms;
}

void omni_encode_stage_post_failure(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || !ctx_omni->async) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(ctx_omni->pipeline_result_mtx);
        ctx_omni->pipeline_result_queue.push({});
    }
    ctx_omni->pipeline_result_cv.notify_one();
}

}  // namespace

bool omni_encode_prefill_input(struct omni_context * ctx_omni,
                               const std::string &   aud_fname,
                               const std::string &   img_fname,
                               int                   index,
                               int                   max_slice_nums,
                               struct omni_embeds &  encoded) {
    const int hidden_size = llama_model_n_embd(llama_get_model(ctx_omni->ctx_llama));
    encoded.index         = index;

    if (!img_fname.empty()) {
        if (max_slice_nums >= 1 && ctx_omni->ctx_vision != nullptr) {
            vision_set_max_slice_nums(ctx_omni->ctx_vision, max_slice_nums);
            LOG_INF("%s: [临时] max_slice_nums=%d for this prefill\n", __func__, max_slice_nums);
        }

        OmniBackendProfileSpan vit_profile_span;
        const bool             vit_profile_active = omni_encode_backend_profile_begin_span(
            vision_get_sched(ctx_omni->ctx_vision), OmniBackendProfileStage::encode_vit_embed, index, -1, vit_profile_span);
        auto       vit_embed_start = std::chrono::high_resolution_clock::now();
        const bool image_embed_ok  = omni_image_embed_make_chunks_with_filename(
            ctx_omni->ctx_vision, ctx_omni->params->cpuparams.n_threads, img_fname, encoded.vision_embed);
        const double vit_submit_ms =
            omni_encode_timing_elapsed_ms(vit_embed_start, std::chrono::high_resolution_clock::now());
        omni_encode_stage_note_vit(ctx_omni, index, vit_submit_ms);
        if (vit_profile_active) {
            omni_encode_backend_profile_end_span(vit_profile_span, vit_submit_ms);
            omni_encode_backend_profile_finalize_and_log(vit_profile_span);
        }
        if (!image_embed_ok) {
            LOG_ERR("%s: failed to create vision embeddings for %s\n", __func__, img_fname.c_str());
            return false;
        }
        LOG_INF("%s: vision_embed has %d chunks\n", __func__, (int) encoded.vision_embed.size());
    }

    if (!aud_fname.empty()) {
        print_with_timestamp("stream_prefill(index=%d): processing user audio: %s\n", index, aud_fname.c_str());
        OmniBackendProfileSpan audio_profile_span;
        const bool             audio_profile_active = omni_encode_backend_profile_begin_span(
            audition_get_sched(ctx_omni->ctx_audio), OmniBackendProfileStage::encode_audio_embed, index, -1, audio_profile_span);
        auto                          audio_embed_start = std::chrono::high_resolution_clock::now();
        omni_audio_sub_step_timing    sub_timing;
        auto * audio_embeds =
            omni_audio_embed_make_with_filename_timed(ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads,
                                                      aud_fname, &sub_timing);
        const double audio_submit_ms =
            omni_encode_timing_elapsed_ms(audio_embed_start, std::chrono::high_resolution_clock::now());
        omni_encode_stage_note_audio(ctx_omni, index, audio_submit_ms);
        omni_encode_stage_note_audio_sub(ctx_omni, index, sub_timing.file_io_ms, sub_timing.preprocess_ms,
                                         sub_timing.encode_ms);
        if (audio_profile_active) {
            omni_encode_backend_profile_end_span(audio_profile_span, audio_submit_ms);
            omni_encode_backend_profile_finalize_and_log(audio_profile_span);
        }
        if (audio_embeds != nullptr && audio_embeds->n_pos > 0) {
            print_with_timestamp("stream_prefill(index=%d): user audio embedding: n_pos=%d\n", index,
                                 audio_embeds->n_pos);
            encoded.audio_embed.resize(audio_embeds->n_pos * hidden_size);
            std::memcpy(encoded.audio_embed.data(), audio_embeds->embed, encoded.audio_embed.size() * sizeof(float));
            omni_embed_free(audio_embeds);
        } else {
            LOG_WRN("%s: audio encoding failed, skipping audio for this frame: %s\n", __func__, aud_fname.c_str());
        }
    }

    return true;
}

bool omni_submit_llm_prefill(struct omni_context * ctx_omni, std::unique_ptr<struct omni_embeds> encoded) {
    if (ctx_omni == nullptr || encoded == nullptr) {
        return false;
    }

    if (!ctx_omni->async) {
        omni_llm_stage_prefill_apply(ctx_omni, ctx_omni->params, *encoded);
        omni_llm_stage_finalize_prefill(ctx_omni);
        return true;
    }

    if (ctx_omni->llm_thread_info == nullptr) {
        return false;
    }

    std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
    ctx_omni->llm_thread_info->cv.wait(lock, [&] {
        return ctx_omni->llm_thread_info->queue.size() <
                   static_cast<size_t>(ctx_omni->llm_thread_info->MAX_QUEUE_SIZE) ||
               !ctx_omni->workers.llm_thread_running;
    });
    if (!ctx_omni->workers.llm_thread_running) {
        return false;
    }

    // Record queue depth and timestamp before pushing
    encoded->submit_time          = std::chrono::steady_clock::now();
    encoded->queue_depth_at_submit = static_cast<int>(ctx_omni->llm_thread_info->queue.size());
    ctx_omni->llm_thread_info->queue.push(encoded.release());
    lock.unlock();
    ctx_omni->llm_thread_info->cv.notify_all();
    return true;
}

void omni_encode_worker_loop(struct omni_context * ctx_omni, struct common_params * params) {
    (void) params;
    if (ctx_omni == nullptr || ctx_omni->encode_thread_info == nullptr) {
        return;
    }

    print_with_timestamp("Encode thread started\n");

    while (ctx_omni->workers.encode_thread_running) {
        OmniEncodeRequest request;
        {
            std::unique_lock<std::mutex> lock(ctx_omni->encode_thread_info->mtx);
            ctx_omni->encode_thread_info->cv.wait(lock, [&] {
                return !ctx_omni->encode_thread_info->queue.empty() || !ctx_omni->workers.encode_thread_running;
            });

            if (!ctx_omni->workers.encode_thread_running) {
                break;
            }

            request = std::move(ctx_omni->encode_thread_info->queue.front());
            ctx_omni->encode_thread_info->queue.pop();
        }
        ctx_omni->encode_thread_info->cv.notify_all();

        auto encoded = std::make_unique<struct omni_embeds>();
        if (!omni_encode_prefill_input(ctx_omni, request.aud_fname, request.img_fname, request.index,
                                       request.max_slice_nums, *encoded)) {
            LOG_ERR("%s: encode failed for chunk %d\n", __func__, request.index);
            auto failed = std::make_unique<struct omni_embeds>();
            failed->index         = request.index;
            failed->encode_failed = true;
            if (!omni_submit_llm_prefill(ctx_omni, std::move(failed)) && !ctx_omni->workers.llm_thread_running) {
                break;
            }
            continue;
        }

        if (!omni_submit_llm_prefill(ctx_omni, std::move(encoded))) {
            if (!ctx_omni->workers.llm_thread_running) {
                break;
            }
            LOG_ERR("%s: failed to submit encoded chunk %d to llm worker\n", __func__, request.index);
            omni_encode_stage_post_failure(ctx_omni);
        }
    }

    print_with_timestamp("Encode thread stopped\n");
}
