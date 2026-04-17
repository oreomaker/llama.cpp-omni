#include "omni-encode-stage.h"

#include "audition.h"
#include "common/common.h"
#include "omni-impl.h"
#include "omni-llm-stage.h"
#include "omni-log.h"
#include "omni.h"
#include "vision.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>

namespace {

double omni_encode_timing_elapsed_ms(const std::chrono::high_resolution_clock::time_point & start,
                                     const std::chrono::high_resolution_clock::time_point & end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
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

        auto       vit_embed_start = std::chrono::high_resolution_clock::now();
        const bool image_embed_ok  = omni_image_embed_make_chunks_with_filename(
            ctx_omni->ctx_vision, ctx_omni->params->cpuparams.n_threads, img_fname, encoded.vision_embed);
        omni_encode_stage_note_vit(ctx_omni, index,
                                   omni_encode_timing_elapsed_ms(vit_embed_start, std::chrono::high_resolution_clock::now()));
        if (!image_embed_ok) {
            LOG_ERR("%s: failed to create vision embeddings for %s\n", __func__, img_fname.c_str());
            return false;
        }
        LOG_INF("%s: vision_embed has %d chunks\n", __func__, (int) encoded.vision_embed.size());
    }

    if (!aud_fname.empty()) {
        print_with_timestamp("stream_prefill(index=%d): processing user audio: %s\n", index, aud_fname.c_str());
        auto   audio_embed_start = std::chrono::high_resolution_clock::now();
        auto * audio_embeds =
            omni_audio_embed_make_with_filename(ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, aud_fname);
        omni_encode_stage_note_audio(
            ctx_omni, index,
            omni_encode_timing_elapsed_ms(audio_embed_start, std::chrono::high_resolution_clock::now()));
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
