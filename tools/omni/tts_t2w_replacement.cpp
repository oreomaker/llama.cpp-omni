// Replacement TTS/T2W wiring.
//
// The runtime owns only the small TTS lookup/head graphs. Transformer and
// Token2Wav state stay owned by omni_context and the existing compute/worker
// modules. Sampling remains on the host for the small GPU top-k candidate set
// until a multinomial kernel is available in ggml.

#include "omni_internal.h"
#include "omni-impl.h"
#include "token2wav/token2wav-impl.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "common/common.h"
#include "common/sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

struct OmniTtsT2wReplacement {
    ~OmniTtsT2wReplacement();

    ggml_backend_t backend = nullptr;

    ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_tensor * emb_code_weight = nullptr;
    ggml_tensor * head_code_weight = nullptr;

    ggml_context * head_ctx = nullptr;
    ggml_gallocr_t head_alloc = nullptr;
    ggml_cgraph * head_graph = nullptr;
    ggml_tensor * head_hidden = nullptr;
    ggml_tensor * head_indices = nullptr;
    ggml_tensor * head_values = nullptr;

    ggml_context * lookup_ctx = nullptr;
    ggml_gallocr_t lookup_alloc = nullptr;
    ggml_cgraph * lookup_graph = nullptr;
    ggml_tensor * lookup_token = nullptr;
    ggml_tensor * lookup_embedding = nullptr;

    bool initialized = false;
    bool warned_fallback = false;
};

namespace {

constexpr int TTS_AUDIO_VOCAB = 6562;
constexpr int TTS_HIDDEN = 768;
constexpr int TTS_TOP_K = 25;
constexpr llama_token TTS_AUDIO_BOS = 151687;

void replacement_free(OmniTtsT2wReplacement & r) {
    if (r.head_alloc) {
        ggml_gallocr_free(r.head_alloc);
        r.head_alloc = nullptr;
    }
    if (r.lookup_alloc) {
        ggml_gallocr_free(r.lookup_alloc);
        r.lookup_alloc = nullptr;
    }
    if (r.head_ctx) {
        ggml_free(r.head_ctx);
        r.head_ctx = nullptr;
    }
    if (r.lookup_ctx) {
        ggml_free(r.lookup_ctx);
        r.lookup_ctx = nullptr;
    }
    r.head_graph = nullptr;
    r.lookup_graph = nullptr;
    r.head_hidden = nullptr;
    r.head_indices = nullptr;
    r.head_values = nullptr;
    r.lookup_token = nullptr;
    r.lookup_embedding = nullptr;

    if (r.weight_buffer) {
        ggml_backend_buffer_free(r.weight_buffer);
        r.weight_buffer = nullptr;
    }
    if (r.weight_ctx) {
        ggml_free(r.weight_ctx);
        r.weight_ctx = nullptr;
    }
    r.emb_code_weight = nullptr;
    r.head_code_weight = nullptr;
    r.backend = nullptr;
    r.initialized = false;
}

bool make_head_graph(OmniTtsT2wReplacement & r) {
    r.head_ctx = ggml_init({
        /*.mem_size   =*/ ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(64, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    });
    if (!r.head_ctx) {
        return false;
    }

    r.head_hidden = ggml_new_tensor_2d(r.head_ctx, GGML_TYPE_F32, TTS_HIDDEN, 1);
    ggml_set_name(r.head_hidden, "replacement_tts_hidden");
    ggml_set_input(r.head_hidden);

    ggml_tensor * logits = ggml_mul_mat(r.head_ctx, r.head_code_weight, r.head_hidden);
    ggml_set_name(logits, "replacement_tts_logits");
    ggml_tensor * indices = ggml_top_k(r.head_ctx, logits, TTS_TOP_K);
    ggml_set_name(indices, "replacement_tts_topk_indices");

    ggml_tensor * logits_rows = ggml_reshape_2d(r.head_ctx, logits, 1, TTS_AUDIO_VOCAB);
    ggml_tensor * values_rows = ggml_get_rows(r.head_ctx, logits_rows, indices);
    r.head_values = ggml_reshape_1d(r.head_ctx, values_rows, TTS_TOP_K);
    r.head_indices = ggml_reshape_1d(r.head_ctx, indices, TTS_TOP_K);
    ggml_set_name(r.head_values, "replacement_tts_topk_values");
    ggml_set_name(r.head_indices, "replacement_tts_topk_indices_flat");
    ggml_set_output(r.head_indices);
    ggml_set_output(r.head_values);

    r.head_graph = ggml_new_graph_custom(r.head_ctx, 64, false);
    ggml_build_forward_expand(r.head_graph, r.head_values);
    ggml_build_forward_expand(r.head_graph, r.head_indices);
    r.head_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(r.backend));
    return r.head_alloc && ggml_gallocr_alloc_graph(r.head_alloc, r.head_graph);
}

bool make_lookup_graph(OmniTtsT2wReplacement & r) {
    r.lookup_ctx = ggml_init({
        /*.mem_size   =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(32, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    });
    if (!r.lookup_ctx) {
        return false;
    }

    r.lookup_token = ggml_new_tensor_1d(r.lookup_ctx, GGML_TYPE_I32, 1);
    ggml_set_name(r.lookup_token, "replacement_tts_audio_token");
    ggml_set_input(r.lookup_token);
    r.lookup_embedding = ggml_cont(r.lookup_ctx,
                                   ggml_get_rows(r.lookup_ctx, r.emb_code_weight, r.lookup_token));
    ggml_set_name(r.lookup_embedding, "replacement_tts_audio_embedding");
    ggml_set_output(r.lookup_embedding);

    r.lookup_graph = ggml_new_graph_custom(r.lookup_ctx, 32, false);
    ggml_build_forward_expand(r.lookup_graph, r.lookup_embedding);
    r.lookup_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(r.backend));
    return r.lookup_alloc && ggml_gallocr_alloc_graph(r.lookup_alloc, r.lookup_graph);
}

bool finite_candidate(int id, float value) {
    return id >= 0 && id < TTS_AUDIO_VOCAB && std::isfinite(value);
}

int sample_candidates(const std::vector<std::pair<float, int>> & raw,
                      std::mt19937 & rng,
                      bool skip_processors,
                      bool force_no_eos,
                      const std::vector<llama_token> * history,
                      float temperature,
                      float top_p,
                      int repetition_window,
                      bool argmax) {
    std::vector<std::pair<float, int>> candidates = raw;
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [](const auto & p) { return !finite_candidate(p.second, p.first); }),
                     candidates.end());
    if (candidates.empty()) {
        return 0;
    }

    // ggml top_k returns an unordered set. Sorting only this small candidate
    // array keeps the expensive 6562-element ordering on the GPU.
    std::sort(candidates.begin(), candidates.end(),
              [](const auto & a, const auto & b) { return a.first > b.first; });

    if (argmax) {
        return candidates.front().second;
    }

    const int eos = TTS_AUDIO_VOCAB - 1;
    if (force_no_eos) {
        for (auto & p : candidates) {
            if (p.second == eos) {
                p.first = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // The first token of each duplex chunk intentionally skips processors in
    // the legacy implementation. It still samples from the GPU top-k set;
    // this is the documented transition until full-vocabulary multinomial is
    // implemented on CUDA.
    if (!skip_processors && history != nullptr) {
        std::vector<int> recent;
        const int begin = std::max(0, (int) history->size() - repetition_window);
        recent.reserve(history->size() - begin);
        for (int i = begin; i < (int) history->size(); ++i) {
            const int rel = (*history)[i] - TTS_AUDIO_BOS;
            if (rel >= 0 && rel < TTS_AUDIO_VOCAB) {
                recent.push_back(rel);
            }
        }
        for (auto & p : candidates) {
            if (std::find(recent.begin(), recent.end(), p.second) != recent.end()) {
                p.first = p.first < 0.0f ? p.first * 1.05f : p.first / 1.05f;
            }
        }
    }

    const float temp = temperature > 0.0f ? temperature : 1.0f;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (const auto & p : candidates) {
        max_logit = std::max(max_logit, p.first / temp);
    }
    std::vector<float> probs(candidates.size(), 0.0f);
    float total = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = std::exp(candidates[i].first / temp - max_logit);
        total += probs[i];
    }
    if (!(total > 0.0f) || !std::isfinite(total)) {
        return candidates.front().second;
    }
    for (float & p : probs) {
        p /= total;
    }

    // Apply the same top-p=0.85, top-k=25 policy to the GPU candidate set.
    // Keeping at least three entries matches the Python warper defaults.
    size_t keep = 0;
    float cumulative = 0.0f;
    for (; keep < probs.size() && (keep < 3 || cumulative < top_p); ++keep) {
        cumulative += probs[keep];
    }
    keep = std::min(keep, (size_t) TTS_TOP_K);
    float keep_total = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        keep_total += probs[i];
    }
    std::uniform_real_distribution<float> dist(0.0f, keep_total);
    const float draw = dist(rng);
    float acc = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        acc += probs[i];
        if (draw <= acc) {
            return candidates[i].second;
        }
    }
    return candidates[keep - 1].second;
}

} // namespace

OmniTtsT2wReplacement::~OmniTtsT2wReplacement() {
    replacement_free(*this);
}

bool omni_tts_t2w_replacement_init(struct omni_context * ctx) {
#if !OMNI_REPLACEMENT_TTS_T2W
    (void) ctx;
    return true;
#else
    if (!ctx || !ctx->ctx_tts_llama || !ctx->emb_code_weight || !ctx->head_code_weight ||
            !ctx->token2wav_initialized || !ctx->token2wav_session || ctx->use_python_token2wav) {
        LOG_ERR("TTS replacement: TTS weights or C++ Token2Wav CUDA session are unavailable\n");
        return false;
    }
    const ggml_backend_t t2m_backend = ctx->token2wav_session->t2w.t2m_backend();
    const ggml_backend_t voc_backend = ctx->token2wav_session->t2w.vocoder_backend();
    if (!t2m_backend || !voc_backend ||
            ggml_backend_dev_type(ggml_backend_get_device(t2m_backend)) != GGML_BACKEND_DEVICE_TYPE_GPU ||
            ggml_backend_dev_type(ggml_backend_get_device(voc_backend)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        LOG_ERR("TTS replacement: Token2Wav flow/vocoder must use GPU backends\n");
        return false;
    }
    if (ctx->emb_code_vocab_size != TTS_AUDIO_VOCAB ||
            ctx->emb_code_hidden_size != TTS_HIDDEN ||
            ctx->head_code_num_audio_tokens != TTS_AUDIO_VOCAB ||
            ctx->head_code_hidden_size != TTS_HIDDEN) {
        LOG_ERR("TTS replacement: unexpected lookup/head dimensions\n");
        return false;
    }

    ggml_backend_t backend = llama_get_gpu_backend(ctx->ctx_tts_llama);
    if (!backend) {
        LOG_ERR("TTS replacement: TTS context has no GPU backend\n");
        return false;
    }

    auto runtime = std::make_shared<OmniTtsT2wReplacement>();
    runtime->backend = backend;
    runtime->weight_ctx = ggml_init({
        /*.mem_size   =*/ ggml_tensor_overhead() * 4,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    });
    if (!runtime->weight_ctx) {
        LOG_ERR("TTS replacement: failed to create weight context\n");
        return false;
    }
    runtime->emb_code_weight = ggml_new_tensor_2d(runtime->weight_ctx, GGML_TYPE_F32,
                                                   TTS_HIDDEN, TTS_AUDIO_VOCAB);
    runtime->head_code_weight = ggml_new_tensor_2d(runtime->weight_ctx, GGML_TYPE_F32,
                                                    TTS_HIDDEN, TTS_AUDIO_VOCAB);
    ggml_set_name(runtime->emb_code_weight, "replacement_emb_code_weight");
    ggml_set_name(runtime->head_code_weight, "replacement_head_code_weight");
    runtime->weight_buffer = ggml_backend_alloc_ctx_tensors(runtime->weight_ctx, backend);
    if (!runtime->weight_buffer) {
        LOG_ERR("TTS replacement: failed to allocate GPU weight buffer\n");
        replacement_free(*runtime);
        return false;
    }
    const size_t weight_bytes = (size_t) TTS_HIDDEN * TTS_AUDIO_VOCAB * sizeof(float);
    ggml_backend_tensor_set(runtime->emb_code_weight, ctx->emb_code_weight, 0, weight_bytes);
    ggml_backend_tensor_set(runtime->head_code_weight, ctx->head_code_weight, 0, weight_bytes);

    if (!make_head_graph(*runtime) || !make_lookup_graph(*runtime)) {
        LOG_ERR("TTS replacement: failed to create lookup/head graphs\n");
        replacement_free(*runtime);
        return false;
    }

    std::vector<float> zeros(TTS_HIDDEN, 0.0f);
    int32_t token = 0;
    ggml_backend_tensor_set(runtime->head_hidden, zeros.data(), 0, weight_bytes / TTS_AUDIO_VOCAB);
    ggml_backend_tensor_set(runtime->lookup_token, &token, 0, sizeof(token));
    if (ggml_backend_graph_compute(backend, runtime->head_graph) != GGML_STATUS_SUCCESS ||
            ggml_backend_graph_compute(backend, runtime->lookup_graph) != GGML_STATUS_SUCCESS) {
        LOG_ERR("TTS replacement: graph warmup failed\n");
        replacement_free(*runtime);
        return false;
    }

    runtime->initialized = true;
    ctx->tts_t2w_replacement = std::move(runtime);
    LOG_INF("TTS replacement: GPU-resident emb_code/head_code initialized; sampling candidates=%d on CPU\n",
            TTS_TOP_K);
    return true;
#endif
}

void omni_tts_t2w_replacement_free(struct omni_context * ctx) {
    if (ctx) {
        ctx->tts_t2w_replacement.reset();
    }
}

bool omni_tts_replacement_sample_token(struct common_sampler * smpl,
                                       struct omni_context * ctx,
                                       common_params * params,
                                       int * n_past_tts,
                                       const std::vector<llama_token> * all_generated_tokens,
                                       const std::vector<llama_token> * chunk_generated_tokens,
                                       int token_index_in_chunk, bool force_no_eos,
                                       bool is_final_text_chunk,
                                       llama_token * out_token) {
#if !OMNI_REPLACEMENT_TTS_T2W
    (void) smpl; (void) ctx; (void) params; (void) n_past_tts;
    (void) all_generated_tokens; (void) chunk_generated_tokens;
    (void) token_index_in_chunk; (void) force_no_eos;
    (void) is_final_text_chunk; (void) out_token;
    return false;
#else
    (void) params;
    const bool profile = std::getenv("OMNI_TTS_PROFILE") != nullptr;
    const auto token_begin = std::chrono::steady_clock::now();
    if (!ctx || !ctx->tts_t2w_replacement || !ctx->tts_t2w_replacement->initialized ||
            !ctx->ctx_tts_llama || !n_past_tts || !out_token) {
        return false;
    }
    auto & r = *ctx->tts_t2w_replacement;
    ggml_tensor * hidden = llama_get_embeddings_ith_device(ctx->ctx_tts_llama, -1);
    if (!hidden || hidden->ne[0] != TTS_HIDDEN || hidden->ne[1] != 1) {
        return false;
    }
    ggml_backend_tensor_copy_async(r.backend, r.backend, hidden, r.head_hidden);
    const auto head_begin = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(r.backend, r.head_graph) != GGML_STATUS_SUCCESS) {
        return false;
    }

    std::vector<int32_t> indices(TTS_TOP_K);
    std::vector<float> values(TTS_TOP_K);
    ggml_backend_tensor_get(r.head_indices, indices.data(), 0, indices.size() * sizeof(int32_t));
    ggml_backend_tensor_get(r.head_values, values.data(), 0, values.size() * sizeof(float));
    const auto head_end = std::chrono::steady_clock::now();

    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(TTS_TOP_K);
    for (int i = 0; i < TTS_TOP_K; ++i) {
        candidates.emplace_back(values[i], indices[i]);
    }

    std::mt19937 * rng = get_sampler_rng(smpl);
    std::mt19937 local_rng(42);
    if (!rng) {
        rng = &local_rng;
    }
    const auto * history = ctx->duplex_mode ? chunk_generated_tokens : all_generated_tokens;
    const bool skip_processors = ctx->duplex_mode ? token_index_in_chunk == 0
                                                   : (all_generated_tokens == nullptr || all_generated_tokens->empty());
    const int relative = sample_candidates(candidates, *rng, skip_processors, force_no_eos,
                                           history, ctx->tts_temperature,
                                           ctx->duplex_mode ? 0.85f : 1.0f,
                                           ctx->duplex_mode ? 16 : 8,
                                           params != nullptr && params->sampling.temp <= 0.0f);
    const auto sample_end = std::chrono::steady_clock::now();
    if (relative < 0 || relative >= TTS_AUDIO_VOCAB) {
        return false;
    }

    const llama_token token = TTS_AUDIO_BOS + relative;
    common_sampler_accept(smpl, token, true);
    const bool eos = relative == TTS_AUDIO_VOCAB - 1;
    if (eos && !is_final_text_chunk) {
        *out_token = token;
        return true;
    }

    ggml_backend_tensor_set(r.lookup_token, &relative, 0, sizeof(relative));
    const auto lookup_begin = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(r.backend, r.lookup_graph) != GGML_STATUS_SUCCESS ||
            llama_decode_device_embd(ctx->ctx_tts_llama, r.lookup_embedding, *n_past_tts) != 0) {
        return false;
    }
    ++*n_past_tts;
    if (profile) {
        const auto lookup_end = std::chrono::steady_clock::now();
        const auto ms = [](const auto & a, const auto & b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::fprintf(stderr,
                     "[tts.replacement.prof] token=%d gpu_head_topk=%.3f candidate_sample=%.3f gpu_lookup_decode=%.3f host_total=%.3f\n",
                     token_index_in_chunk, ms(head_begin, head_end),
                     ms(head_end, sample_end), ms(lookup_begin, lookup_end),
                     ms(token_begin, lookup_end));
    }
    *out_token = token;
    return true;
#endif
}

void omni_tts_replacement_thread_func(struct omni_context * ctx, common_params * params) {
    if (!ctx || !params) {
        return;
    }
    print_with_timestamp("TTS replacement worker started\n");
    if (ctx->duplex_mode) {
        tts_thread_func_duplex(ctx, params);
    } else {
        tts_thread_func(ctx, params);
    }
}

void omni_t2w_replacement_thread_func(struct omni_context * ctx, common_params * params) {
    if (!ctx || !params) {
        return;
    }
    if (ctx->use_python_token2wav || !ctx->token2wav_initialized || !ctx->token2wav_session) {
        LOG_ERR("T2W replacement: C++ Token2Wav session is unavailable\n");
        return;
    }
    print_with_timestamp("T2W replacement worker started (C++ CUDA session)\n");
    t2w_thread_func(ctx, params);
}
