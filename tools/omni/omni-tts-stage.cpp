#include "omni-tts-stage.h"

#include "common/common.h"
#include "common/sampling.h"
#include "omni-impl.h"
#include "omni-log.h"
#include "omni-output.h"
#include "omni-worker-coordinator.h"
#include "omni.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <sys/stat.h>
#    include <sys/types.h>
#    ifndef S_IFDIR
#        define S_IFDIR _S_IFDIR
#    endif
#    ifndef stat
#        define stat _stat
#    endif
#else
#    include <sys/stat.h>
#    include <sys/types.h>
#endif

namespace {

void duplex_timing_mark_tts_done(struct omni_context * ctx_omni, int chunk_idx) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.tts_audio_token_ms          = std::max(timing.tts_audio_token_ms, 0.0);
    timing.tts_done                    = true;
}

size_t findIncompleteUtf8(const std::string & str) {
    if (str.empty()) {
        return 0;
    }

    size_t len = str.length();

    // Check from the end backwards to find incomplete UTF-8 sequences
    size_t pos                         = len;
    int    expected_continuation_bytes = 0;

    while (pos > 0) {
        unsigned char c = (unsigned char) str[pos - 1];

        if ((c & 0x80) == 0) {
            // ASCII character (0xxxxxxx), complete sequence
            break;
        } else if ((c & 0xC0) == 0x80) {
            // Continuation byte (10xxxxxx), part of a multi-byte sequence
            expected_continuation_bytes++;
            pos--;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence start (110xxxxx)
            if (expected_continuation_bytes == 1) {
                break;
            } else {
                return len - pos + 1;
            }
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence start (1110xxxx)
            if (expected_continuation_bytes == 2) {
                break;
            } else {
                return len - pos + (3 - expected_continuation_bytes);
            }
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence start (11110xxx)
            if (expected_continuation_bytes == 3) {
                break;
            } else {
                return len - pos + (4 - expected_continuation_bytes);
            }
        } else {
            break;
        }
    }

    if (pos == 0 && expected_continuation_bytes > 0) {
        return len;
    }

    return 0;
}

constexpr int OMNI_TTS_AUDIO_BOS_TOKEN_ID      = 151687;
constexpr int OMNI_TTS_TEXT_EOS_TOKEN_ID       = 151692;
constexpr int OMNI_TTS_NUM_AUDIO_TOKENS        = 6562;
constexpr int OMNI_TTS_AUDIO_EOS_RELATIVE_IDX  = OMNI_TTS_NUM_AUDIO_TOKENS - 1;
constexpr int OMNI_TTS_STREAM_CHUNK_SIZE       = 25;
constexpr int OMNI_TTS_FIRST_STREAM_CHUNK_SIZE = 28;

struct OmniTtsComputationDebugData {
    std::vector<float> llm_embeds;
    std::vector<float> projected_hidden_before_norm;
    std::vector<float> projected_hidden_after_norm;
};

double omni_tts_timing_elapsed_ms(const std::chrono::high_resolution_clock::time_point & start,
                                  const std::chrono::high_resolution_clock::time_point & end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void omni_tts_duplex_timing_note(struct omni_context * ctx_omni, int chunk_idx, double ms, int audio_token_count) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.tts_audio_token_ms          = timing.tts_audio_token_ms < 0.0 ? ms : timing.tts_audio_token_ms + ms;
    timing.tts_audio_token_count += audio_token_count;
    timing.tts_done = true;
}

std::mt19937 * omni_tts_get_sampler_rng(struct common_sampler * smpl) {
    void * rng_ptr = common_sampler_get_rng(smpl);
    return static_cast<std::mt19937 *>(rng_ptr);
}

bool omni_tts_eval_tokens(struct omni_context *    ctx_omni,
                          common_params *          params,
                          std::vector<llama_token> tokens,
                          int                      n_batch,
                          int *                    n_past_tts) {
    (void) params;
    const int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        n_eval     = std::min(n_eval, n_batch);
        if (n_eval == 0) {
            break;
        }

        llama_batch            batch = llama_batch_get_one(&tokens[i], n_eval);
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }

        for (int j = 0; j < n_eval; ++j) {
            batch.pos[j] = *n_past_tts + j;
        }

        llama_set_embeddings(ctx_omni->ctx_tts_llama, true);
        const int decode_ret = llama_decode(ctx_omni->ctx_tts_llama, batch);
        if (decode_ret != 0) {
            LOG_ERR("%s : failed to eval TTS tokens. token %d/%d (batch size %d, n_past %d), decode_ret=%d\n", __func__,
                    i, N, n_batch, *n_past_tts, decode_ret);
            return false;
        }

        *n_past_tts += n_eval;
    }

    return true;
}

void omni_tts_save_logits_to_file(const char * filepath, const float * logits, int num_tokens, int token_index) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s/logits_%03d.bin", filepath, token_index);

    FILE * f = fopen(full_path, "wb");
    if (f == nullptr) {
        LOG_ERR("Failed to open logits file for writing: %s\n", full_path);
        return;
    }

    fwrite(&token_index, sizeof(int32_t), 1, f);
    fwrite(&num_tokens, sizeof(int32_t), 1, f);
    fwrite(logits, sizeof(float), num_tokens, f);
    fclose(f);
}

void omni_tts_save_hidden_states_to_file(const char *  filepath,
                                         const float * hidden_states,
                                         int           hidden_size,
                                         int           token_index) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s/hidden_states_%03d.bin", filepath, token_index);

    FILE * f = fopen(full_path, "wb");
    if (f == nullptr) {
        LOG_ERR("Failed to open hidden states file for writing: %s\n", full_path);
        return;
    }

    fwrite(&token_index, sizeof(int32_t), 1, f);
    fwrite(&hidden_size, sizeof(int32_t), 1, f);
    fwrite(hidden_states, sizeof(float), hidden_size, f);
    fclose(f);
}

int omni_tts_random_sampling(const float * logits, int num_tokens, std::mt19937 & rng) {
    float max_logit = logits[0];
    for (int i = 1; i < num_tokens; ++i) {
        max_logit = std::max(logits[i], max_logit);
    }

    std::vector<float> probs(num_tokens);
    float              sum = 0.0f;
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] /= sum;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float                           r   = dist(rng);
    float                                 cum = 0.0f;
    for (int i = 0; i < num_tokens; ++i) {
        cum += probs[i];
        if (r <= cum) {
            return i;
        }
    }

    return num_tokens - 1;
}

void omni_tts_apply_repetition_penalty(float *                  logits,
                                       int                      num_tokens,
                                       const std::vector<int> & decoded_tokens,
                                       float                    penalty,
                                       int                      past_window = 16) {
    if (decoded_tokens.empty() || penalty == 1.0f) {
        return;
    }

    const int        start_idx = std::max(0, (int) decoded_tokens.size() - past_window);
    std::vector<int> freq(num_tokens, 0);
    for (int i = start_idx; i < (int) decoded_tokens.size(); ++i) {
        const int tok = decoded_tokens[i];
        if (tok >= 0 && tok < num_tokens) {
            freq[tok]++;
        }
    }

    for (int i = 0; i < num_tokens; ++i) {
        if (freq[i] > 0) {
            const float alpha = powf(penalty, (float) freq[i]);
            if (logits[i] < 0) {
                logits[i] *= alpha;
            } else {
                logits[i] /= alpha;
            }
        }
    }
}

int omni_tts_nucleus_sampling_with_min_keep(const float *  logits,
                                            int            num_tokens,
                                            float          top_p,
                                            int            top_k,
                                            int            min_tokens_to_keep,
                                            std::mt19937 & rng) {
    float max_logit = logits[0];
    for (int i = 1; i < num_tokens; ++i) {
        max_logit = std::max(logits[i], max_logit);
    }

    std::vector<float> probs(num_tokens);
    float              sum = 0.0f;
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < num_tokens; ++i) {
        probs[i] /= sum;
    }

    std::vector<std::pair<float, int>> sorted_probs;
    sorted_probs.reserve(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        sorted_probs.push_back({ probs[i], i });
    }
    std::sort(sorted_probs.begin(), sorted_probs.end(),
              [](const auto & lhs, const auto & rhs) { return lhs.first > rhs.first; });

    std::vector<float> filtered_probs;
    std::vector<int>   filtered_indices;
    float              cum_prob = 0.0f;

    for (const auto & p : sorted_probs) {
        if ((int) filtered_probs.size() < min_tokens_to_keep) {
            cum_prob += p.first;
            filtered_probs.push_back(p.first);
            filtered_indices.push_back(p.second);
        } else if (cum_prob < top_p && (int) filtered_probs.size() < top_k) {
            cum_prob += p.first;
            filtered_probs.push_back(p.first);
            filtered_indices.push_back(p.second);
        } else {
            break;
        }
    }

    float filtered_sum = 0.0f;
    for (float p : filtered_probs) {
        filtered_sum += p;
    }
    for (float & p : filtered_probs) {
        p /= filtered_sum;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float                           r   = dist(rng);
    float                                 cum = 0.0f;
    for (size_t i = 0; i < filtered_probs.size(); ++i) {
        cum += filtered_probs[i];
        if (r <= cum) {
            return filtered_indices[i];
        }
    }

    return filtered_indices.back();
}

bool omni_tts_emb_text(struct omni_context * ctx_omni, llama_token token_id, float * embedding_out, int tts_n_embd) {
    OmniTTSAuxWeights & tts_aux = ctx_omni->tts_aux;

    if (!tts_aux.emb_text_weight) {
        LOG_ERR("TTS: emb_text_weight not loaded\n");
        return false;
    }

    if (token_id < 0 || token_id >= tts_aux.emb_text_vocab_size) {
        LOG_ERR("TTS: token_id %d out of range [0, %d)\n", token_id, tts_aux.emb_text_vocab_size);
        return false;
    }

    if (tts_n_embd != tts_aux.emb_text_hidden_size) {
        LOG_ERR("TTS: tts_n_embd (%d) != emb_text_hidden_size (%d)\n", tts_n_embd, tts_aux.emb_text_hidden_size);
        return false;
    }

    const float * src = tts_aux.emb_text_weight + token_id * tts_n_embd;
    memcpy(embedding_out, src, tts_n_embd * sizeof(float));
    return true;
}

bool omni_tts_projector_semantic(struct omni_context * ctx_omni,
                                 const float *         llm_hidden_states,
                                 int                   n_tokens,
                                 int                   llm_n_embd,
                                 float *               projected_hidden_states,
                                 int                   tts_n_embd) {
    OmniTTSAuxWeights &       tts_aux       = ctx_omni->tts_aux;
    OmniTTSProjectorRuntime & tts_projector = ctx_omni->tts_projector;

    if (tts_projector.projector.initialized) {
        if (llm_n_embd != tts_projector.projector.hparams.in_dim) {
            LOG_ERR("TTS: llm_n_embd (%d) != projector in_dim (%d)\n", llm_n_embd,
                    tts_projector.projector.hparams.in_dim);
            return false;
        }
        if (tts_n_embd != tts_projector.projector.hparams.out_dim) {
            LOG_ERR("TTS: tts_n_embd (%d) != projector out_dim (%d)\n", tts_n_embd,
                    tts_projector.projector.hparams.out_dim);
            return false;
        }

        std::vector<float> result = projector_forward(tts_projector.projector, llm_hidden_states, n_tokens);
        if (result.empty()) {
            LOG_ERR("TTS: projector_forward failed\n");
            return false;
        }

        memcpy(projected_hidden_states, result.data(), n_tokens * tts_n_embd * sizeof(float));
        return true;
    }

    if (!tts_aux.projector_semantic_linear1_weight || !tts_aux.projector_semantic_linear1_bias ||
        !tts_aux.projector_semantic_linear2_weight || !tts_aux.projector_semantic_linear2_bias) {
        LOG_ERR("TTS: projector_semantic weights not loaded (both ggml and legacy)\n");
        return false;
    }

    if (llm_n_embd != tts_aux.projector_semantic_input_dim) {
        LOG_ERR("TTS: llm_n_embd (%d) != projector_semantic_input_dim (%d)\n", llm_n_embd,
                tts_aux.projector_semantic_input_dim);
        return false;
    }

    if (tts_n_embd != tts_aux.projector_semantic_output_dim) {
        LOG_ERR("TTS: tts_n_embd (%d) != projector_semantic_output_dim (%d)\n", tts_n_embd,
                tts_aux.projector_semantic_output_dim);
        return false;
    }

    const int input_dim  = tts_aux.projector_semantic_input_dim;
    const int output_dim = tts_aux.projector_semantic_output_dim;

    for (int t = 0; t < n_tokens; ++t) {
        const float *      hidden = llm_hidden_states + t * input_dim;
        float *            output = projected_hidden_states + t * output_dim;
        std::vector<float> temp(output_dim);

        for (int j = 0; j < output_dim; ++j) {
            float sum = tts_aux.projector_semantic_linear1_bias[j];
            for (int i = 0; i < input_dim; ++i) {
                sum += hidden[i] * tts_aux.projector_semantic_linear1_weight[i * output_dim + j];
            }
            temp[j] = sum;
        }

        for (int j = 0; j < output_dim; ++j) {
            temp[j] = temp[j] > 0.0f ? temp[j] : 0.0f;
        }

        for (int j = 0; j < output_dim; ++j) {
            float sum = tts_aux.projector_semantic_linear2_bias[j];
            for (int i = 0; i < output_dim; ++i) {
                sum += temp[i] * tts_aux.projector_semantic_linear2_weight[i * output_dim + j];
            }
            output[j] = sum;
        }
    }

    return true;
}

void omni_tts_normalize_l2_per_token(float * embeddings, int n_tokens, int n_embd, float eps = 1e-8f) {
    for (int t = 0; t < n_tokens; ++t) {
        float * vec     = embeddings + t * n_embd;
        float   norm_sq = 0.0f;
        for (int i = 0; i < n_embd; ++i) {
            const float val = vec[i];
            norm_sq += val * val;
        }

        const float norm = std::sqrt(norm_sq + eps);
        if (norm > 0.0f) {
            const float inv_norm = 1.0f / norm;
            for (int i = 0; i < n_embd; ++i) {
                vec[i] *= inv_norm;
            }
        } else {
            LOG_WRN("TTS: WARNING - zero norm detected for token %d, setting to unit vector\n", t);
            const float inv_sqrt_n = 1.0f / std::sqrt((float) n_embd);
            for (int i = 0; i < n_embd; ++i) {
                vec[i] = inv_sqrt_n;
            }
        }

        float verify_norm_sq = 0.0f;
        for (int i = 0; i < n_embd; ++i) {
            const float val = vec[i];
            verify_norm_sq += val * val;
        }
        const float verify_norm = std::sqrt(verify_norm_sq);
        if (std::abs(verify_norm - 1.0f) > 0.01f) {
            LOG_ERR(
                "TTS: ERROR - normalization verification failed for token %d: norm=%.6f (expected ~1.0), "
                "norm_sq=%.6f\n",
                t, verify_norm, verify_norm_sq);
        }
    }
}

const std::vector<llama_token> OMNI_TTS_SPECIAL_TOKEN_IDS = {
    151667, 151668, 151704, 151706, 151705, 151718, 151721, 151717, 271,
};

const std::set<llama_token> OMNI_TTS_KNOWN_EMPTY_TOKEN_IDS = {};

bool omni_tts_is_valid_token_internal(llama_token tid) {
    for (llama_token sid : OMNI_TTS_SPECIAL_TOKEN_IDS) {
        if (tid == sid) {
            return false;
        }
    }

    if (tid >= 150000) {
        return false;
    }

    return OMNI_TTS_KNOWN_EMPTY_TOKEN_IDS.find(tid) == OMNI_TTS_KNOWN_EMPTY_TOKEN_IDS.end();
}

void omni_tts_filter_special_tokens(std::vector<llama_token> & token_ids,
                                    std::vector<float> &       hidden_states,
                                    int                        n_embd) {
    if (hidden_states.size() != token_ids.size() * (size_t) n_embd) {
        LOG_ERR("filter_special_tokens: hidden_states size (%zu) != token_ids.size() * n_embd (%zu * %d)\n",
                hidden_states.size(), token_ids.size(), n_embd);
        return;
    }

    std::vector<llama_token> filtered_token_ids;
    std::vector<float>       filtered_hidden_states;

    for (size_t i = 0; i < token_ids.size(); ++i) {
        const llama_token tid = token_ids[i];
        if (!omni_tts_is_valid_token_internal(tid)) {
            continue;
        }

        filtered_token_ids.push_back(tid);
        for (int j = 0; j < n_embd; ++j) {
            filtered_hidden_states.push_back(hidden_states[i * n_embd + j]);
        }
    }

    size_t start_idx = 0;
    for (size_t i = 0; i < filtered_token_ids.size(); ++i) {
        if (OMNI_TTS_KNOWN_EMPTY_TOKEN_IDS.find(filtered_token_ids[i]) == OMNI_TTS_KNOWN_EMPTY_TOKEN_IDS.end()) {
            start_idx = i;
            break;
        }
    }

    if (start_idx > 0) {
        filtered_token_ids.erase(filtered_token_ids.begin(), filtered_token_ids.begin() + start_idx);
        filtered_hidden_states.erase(filtered_hidden_states.begin(),
                                     filtered_hidden_states.begin() + start_idx * n_embd);
    }

    if (filtered_hidden_states.size() != filtered_token_ids.size() * (size_t) n_embd) {
        LOG_ERR(
            "filter_special_tokens: alignment error after filtering! token_ids.size()=%zu, hidden_states.size()=%zu, "
            "n_embd=%d\n",
            filtered_token_ids.size(), filtered_hidden_states.size(), n_embd);
        return;
    }

    token_ids     = std::move(filtered_token_ids);
    hidden_states = std::move(filtered_hidden_states);
}

bool omni_tts_write_binary_chunk(const std::string & path,
                                 const void *        data,
                                 size_t              elem_size,
                                 size_t              elem_count,
                                 const int32_t *     header       = nullptr,
                                 size_t              header_count = 0) {
    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    if (header != nullptr && header_count > 0) {
        fwrite(header, sizeof(int32_t), header_count, f);
    }
    if (data != nullptr && elem_count > 0) {
        fwrite(data, elem_size, elem_count, f);
    }
    fclose(f);
    return true;
}

std::string omni_tts_clean_duplex_debug_text(const std::string & llm_text) {
    std::string clean_text = llm_text;
    size_t      pos        = 0;
    while ((pos = clean_text.find("[[")) != std::string::npos) {
        const size_t end_pos = clean_text.find("]]", pos);
        if (end_pos == std::string::npos) {
            break;
        }
        clean_text.erase(pos, end_pos - pos + 2);
    }
    while (clean_text.find("  ") != std::string::npos) {
        clean_text.erase(clean_text.find("  "), 1);
    }
    while (!clean_text.empty() && clean_text.front() == ' ') {
        clean_text.erase(clean_text.begin());
    }
    while (!clean_text.empty() && clean_text.back() == ' ') {
        clean_text.pop_back();
    }
    return clean_text;
}

bool omni_tts_compute_merged_embeddings(struct omni_context *            ctx_omni,
                                        const std::vector<llama_token> & filtered_token_ids,
                                        const std::vector<float> &       filtered_hidden_states,
                                        int                              current_chunk_n_embd,
                                        bool                             append_audio_bos,
                                        const char *                     missing_weight_log,
                                        OmniTtsPreparedChunk &           prepared_chunk,
                                        OmniTtsComputationDebugData *    debug_data) {
    OmniTTSAuxWeights & tts_aux = ctx_omni->tts_aux;

    prepared_chunk.tts_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
    if (!tts_aux.emb_text_weight || !tts_aux.projector_semantic_linear1_weight) {
        print_with_timestamp("%s\n", missing_weight_log);
        return true;
    }

    std::vector<float> llm_embeds(prepared_chunk.n_tokens_filtered * prepared_chunk.tts_n_embd, 0.0f);
    for (int i = 0; i < prepared_chunk.n_tokens_filtered; ++i) {
        if (!omni_tts_emb_text(ctx_omni, filtered_token_ids[i], llm_embeds.data() + i * prepared_chunk.tts_n_embd,
                               prepared_chunk.tts_n_embd)) {
            return true;
        }
    }

    std::vector<float> projected_hidden(prepared_chunk.n_tokens_filtered * prepared_chunk.tts_n_embd, 0.0f);
    if (!omni_tts_projector_semantic(ctx_omni, filtered_hidden_states.data(), prepared_chunk.n_tokens_filtered,
                                     current_chunk_n_embd, projected_hidden.data(), prepared_chunk.tts_n_embd)) {
        print_with_timestamp("TTS: WARNING - projector_semantic failed, skipping merged embedding save\n");
        return true;
    }

    if (debug_data != nullptr) {
        debug_data->llm_embeds                   = llm_embeds;
        debug_data->projected_hidden_before_norm = projected_hidden;
    }

    omni_tts_normalize_l2_per_token(projected_hidden.data(), prepared_chunk.n_tokens_filtered,
                                    prepared_chunk.tts_n_embd);

    if (debug_data != nullptr) {
        debug_data->projected_hidden_after_norm = projected_hidden;
    }

    const size_t merge_size = (size_t) prepared_chunk.n_tokens_filtered * prepared_chunk.tts_n_embd;
    if (prepared_chunk.n_tokens_filtered <= 0 || prepared_chunk.n_tokens_filtered > 10000 ||
        prepared_chunk.tts_n_embd <= 0 || prepared_chunk.tts_n_embd > 10000 || merge_size > 100000000) {
        LOG_ERR("TTS: invalid merge size: n_tokens_filtered=%d, tts_n_embd=%d, merge_size=%zu\n",
                prepared_chunk.n_tokens_filtered, prepared_chunk.tts_n_embd, merge_size);
        return false;
    }

    prepared_chunk.merged_embeddings.resize(merge_size);
    for (size_t i = 0; i < merge_size; ++i) {
        prepared_chunk.merged_embeddings[i] = llm_embeds[i] + projected_hidden[i];
    }

    if (append_audio_bos) {
        std::vector<float> audio_bos_embed(prepared_chunk.tts_n_embd, 0.0f);
        if (omni_tts_emb_text(ctx_omni, OMNI_TTS_AUDIO_BOS_TOKEN_ID, audio_bos_embed.data(),
                              prepared_chunk.tts_n_embd)) {
            prepared_chunk.merged_embeddings.insert(prepared_chunk.merged_embeddings.end(), audio_bos_embed.begin(),
                                                    audio_bos_embed.end());
            prepared_chunk.n_tokens_filtered += 1;
        }
    }

    prepared_chunk.merged_success = true;
    return true;
}

void omni_tts_write_duplex_debug_dump(const std::string &              llm_debug_output_dir,
                                      int                              current_chunk_idx,
                                      const std::string &              llm_text,
                                      const std::vector<llama_token> & current_chunk_token_ids,
                                      const std::vector<float> &       current_chunk_hidden_states,
                                      int                              current_chunk_n_embd,
                                      const OmniTtsPreparedChunk &     prepared_chunk) {
    const std::string clean_text = omni_tts_clean_duplex_debug_text(llm_text);
    if (!clean_text.empty()) {
        const std::string text_file = llm_debug_output_dir + "/llm_text.txt";
        FILE *            f_text    = fopen(text_file.c_str(), "a");
        if (f_text != nullptr) {
            fprintf(f_text, "%s\n", clean_text.c_str());
            fclose(f_text);
        }
    }

    const std::string token_ids_file = llm_debug_output_dir + "/llm_token_ids.txt";
    FILE *            f_tokens       = fopen(token_ids_file.c_str(), "a");
    if (f_tokens != nullptr) {
        fprintf(f_tokens, "[chunk_%d] ", current_chunk_idx);
        for (size_t i = 0; i < current_chunk_token_ids.size(); ++i) {
            fprintf(f_tokens, "%d", current_chunk_token_ids[i]);
            if (i + 1 < current_chunk_token_ids.size()) {
                fprintf(f_tokens, " ");
            }
        }
        fprintf(f_tokens, "\n");
        fclose(f_tokens);
    }

    const std::string hidden_txt_file = llm_debug_output_dir + "/llm_hidden_states.txt";
    FILE *            f_hidden_txt    = fopen(hidden_txt_file.c_str(), "a");
    if (f_hidden_txt != nullptr) {
        fprintf(f_hidden_txt, "[chunk_%d] Hidden States (shape: [%d, %d]):\n", current_chunk_idx,
                prepared_chunk.n_tokens_orig, current_chunk_n_embd);
        for (int i = 0; i < prepared_chunk.n_tokens_orig; ++i) {
            fprintf(f_hidden_txt, "  Token %d: %.6f %.6f %.6f ... (first 3 values)\n", i,
                    current_chunk_hidden_states[i * current_chunk_n_embd + 0],
                    current_chunk_hidden_states[i * current_chunk_n_embd + 1],
                    current_chunk_hidden_states[i * current_chunk_n_embd + 2]);
        }
        fclose(f_hidden_txt);
    }

    if (prepared_chunk.merged_success && !prepared_chunk.merged_embeddings.empty()) {
        const std::string merged_txt_file = llm_debug_output_dir + "/merged_embeddings.txt";
        FILE *            f_merged_txt    = fopen(merged_txt_file.c_str(), "a");
        if (f_merged_txt != nullptr) {
            fprintf(f_merged_txt, "[chunk_%d] Merged Embeddings (shape: [%d, %d]):\n", current_chunk_idx,
                    prepared_chunk.n_tokens_filtered, prepared_chunk.tts_n_embd);
            for (int i = 0; i < prepared_chunk.n_tokens_filtered; ++i) {
                fprintf(f_merged_txt, "  Token %d: %.6f %.6f %.6f ... (first 3 values)\n", i,
                        prepared_chunk.merged_embeddings[i * prepared_chunk.tts_n_embd + 0],
                        prepared_chunk.merged_embeddings[i * prepared_chunk.tts_n_embd + 1],
                        prepared_chunk.merged_embeddings[i * prepared_chunk.tts_n_embd + 2]);
            }
            fclose(f_merged_txt);
        }
    }
}

void omni_tts_write_simplex_debug_dump(const std::string &                 llm_debug_output_dir,
                                       int                                 current_chunk_idx,
                                       const std::string &                 response,
                                       const std::vector<llama_token> &    current_chunk_token_ids,
                                       const std::vector<float> &          current_chunk_hidden_states,
                                       int                                 current_chunk_n_embd,
                                       const OmniTtsPreparedChunk &        prepared_chunk,
                                       const OmniTtsComputationDebugData & debug_data) {
    const std::string chunk_dir = llm_debug_output_dir + "/chunk_" + std::to_string(current_chunk_idx);
    omni_ensure_directory(chunk_dir);

    const std::string text_file = chunk_dir + "/llm_text.txt";
    FILE *            f_text    = fopen(text_file.c_str(), "w");
    if (f_text != nullptr) {
        fprintf(f_text, "%s", response.c_str());
        fclose(f_text);
    }

    const std::string token_ids_file = chunk_dir + "/llm_token_ids.txt";
    FILE *            f_tokens       = fopen(token_ids_file.c_str(), "w");
    if (f_tokens != nullptr) {
        for (size_t i = 0; i < current_chunk_token_ids.size(); ++i) {
            fprintf(f_tokens, "%d", current_chunk_token_ids[i]);
            if (i + 1 < current_chunk_token_ids.size()) {
                fprintf(f_tokens, " ");
            }
        }
        fprintf(f_tokens, "\n");
        fclose(f_tokens);
    }

    const std::string hidden_file      = chunk_dir + "/llm_hidden_states.bin";
    int32_t           hidden_header[2] = { prepared_chunk.n_tokens_orig, current_chunk_n_embd };
    omni_tts_write_binary_chunk(hidden_file, current_chunk_hidden_states.data(), sizeof(float),
                                current_chunk_hidden_states.size(), hidden_header, 2);

    const std::string hidden_txt_file = chunk_dir + "/llm_hidden_states.txt";
    FILE *            f_hidden_txt    = fopen(hidden_txt_file.c_str(), "w");
    if (f_hidden_txt != nullptr) {
        fprintf(f_hidden_txt, "Hidden States (shape: [%d, %d]):\n", prepared_chunk.n_tokens_orig, current_chunk_n_embd);
        for (int i = 0; i < prepared_chunk.n_tokens_orig; ++i) {
            fprintf(f_hidden_txt, "Token %d: ", i);
            for (int j = 0; j < current_chunk_n_embd; ++j) {
                fprintf(f_hidden_txt, "%.6f", current_chunk_hidden_states[i * current_chunk_n_embd + j]);
                if (j + 1 < current_chunk_n_embd) {
                    fprintf(f_hidden_txt, " ");
                }
            }
            fprintf(f_hidden_txt, "\n");
        }
        fclose(f_hidden_txt);
    }

    if (prepared_chunk.merged_success && !prepared_chunk.merged_embeddings.empty()) {
        const std::string merged_file      = chunk_dir + "/merged_embeddings.bin";
        int32_t           merged_header[2] = { prepared_chunk.n_tokens_filtered, prepared_chunk.tts_n_embd };
        omni_tts_write_binary_chunk(merged_file, prepared_chunk.merged_embeddings.data(), sizeof(float),
                                    prepared_chunk.merged_embeddings.size(), merged_header, 2);

        const std::string merged_txt_file = chunk_dir + "/merged_embeddings.txt";
        FILE *            f_merged_txt    = fopen(merged_txt_file.c_str(), "w");
        if (f_merged_txt != nullptr) {
            fprintf(f_merged_txt, "Merged Embeddings (shape: [%d, %d]):\n", prepared_chunk.n_tokens_filtered,
                    prepared_chunk.tts_n_embd);
            fprintf(f_merged_txt,
                    "# Formula: merged_embeds = emb_text(filtered_token_ids) + "
                    "normalize(projector_semantic(filtered_hidden_states))\n");
            fprintf(f_merged_txt,
                    "# Note: Special tokens have been filtered before computation (matching Python behavior)\n");
            for (int i = 0; i < prepared_chunk.n_tokens_filtered; ++i) {
                fprintf(f_merged_txt, "Token %d: ", i);
                for (int j = 0; j < prepared_chunk.tts_n_embd; ++j) {
                    fprintf(f_merged_txt, "%.6f", prepared_chunk.merged_embeddings[i * prepared_chunk.tts_n_embd + j]);
                    if (j + 1 < prepared_chunk.tts_n_embd) {
                        fprintf(f_merged_txt, " ");
                    }
                }
                fprintf(f_merged_txt, "\n");
            }
            fclose(f_merged_txt);
            print_with_timestamp("TTS: saved merged embeddings (text) to %s\n", merged_txt_file.c_str());
        }
    } else {
        print_with_timestamp("TTS: skipped saving merged embeddings (computation failed or weights not available)\n");
    }

    if (!debug_data.llm_embeds.empty()) {
        const std::string llm_embeds_file = chunk_dir + "/llm_embeds_cpp.txt";
        FILE *            f_llm_embeds    = fopen(llm_embeds_file.c_str(), "w");
        if (f_llm_embeds != nullptr) {
            fprintf(f_llm_embeds, "LLM Embeddings from emb_text (C++ computed, shape: [%d, %d]):\n",
                    prepared_chunk.n_tokens_filtered, prepared_chunk.tts_n_embd);
            for (int i = 0; i < prepared_chunk.n_tokens_filtered; ++i) {
                fprintf(f_llm_embeds, "Token %d: ", i);
                for (int j = 0; j < prepared_chunk.tts_n_embd; ++j) {
                    fprintf(f_llm_embeds, "%.6f", debug_data.llm_embeds[i * prepared_chunk.tts_n_embd + j]);
                    if (j + 1 < prepared_chunk.tts_n_embd) {
                        fprintf(f_llm_embeds, " ");
                    }
                }
                fprintf(f_llm_embeds, "\n");
            }
            fclose(f_llm_embeds);
        }
    }

    if (!debug_data.projected_hidden_before_norm.empty()) {
        const std::string projected_file = chunk_dir + "/projected_hidden_before_norm_cpp.txt";
        FILE *            f_projected    = fopen(projected_file.c_str(), "w");
        if (f_projected != nullptr) {
            fprintf(f_projected, "Projected Hidden States BEFORE normalization (C++ computed, shape: [%d, %d]):\n",
                    prepared_chunk.n_tokens_filtered, prepared_chunk.tts_n_embd);
            for (int i = 0; i < prepared_chunk.n_tokens_filtered; ++i) {
                fprintf(f_projected, "Token %d: ", i);
                for (int j = 0; j < prepared_chunk.tts_n_embd; ++j) {
                    fprintf(f_projected, "%.6f",
                            debug_data.projected_hidden_before_norm[i * prepared_chunk.tts_n_embd + j]);
                    if (j + 1 < prepared_chunk.tts_n_embd) {
                        fprintf(f_projected, " ");
                    }
                }
                fprintf(f_projected, "\n");
            }
            fclose(f_projected);
        }
    }

    if (!debug_data.projected_hidden_after_norm.empty()) {
        const std::string projected_file = chunk_dir + "/projected_hidden_after_norm_cpp.txt";
        FILE *            f_projected    = fopen(projected_file.c_str(), "w");
        if (f_projected != nullptr) {
            fprintf(f_projected, "Projected Hidden States AFTER normalization (C++ computed, shape: [%d, %d]):\n",
                    prepared_chunk.n_tokens_filtered, prepared_chunk.tts_n_embd);
            for (int i = 0; i < prepared_chunk.n_tokens_filtered; ++i) {
                fprintf(f_projected, "Token %d: ", i);
                for (int j = 0; j < prepared_chunk.tts_n_embd; ++j) {
                    fprintf(f_projected, "%.6f",
                            debug_data.projected_hidden_after_norm[i * prepared_chunk.tts_n_embd + j]);
                    if (j + 1 < prepared_chunk.tts_n_embd) {
                        fprintf(f_projected, " ");
                    }
                }
                fprintf(f_projected, "\n");
            }
            fclose(f_projected);
        }
    }
}

llama_token omni_tts_sample_token_simplex_internal(struct common_sampler *          smpl,
                                                   struct omni_context *            ctx_omni,
                                                   common_params *                  params,
                                                   int *                            n_past_tts,
                                                   const std::vector<llama_token> * all_generated_tokens,
                                                   int                              token_index_in_chunk,
                                                   bool                             force_no_eos        = false,
                                                   bool                             is_final_text_chunk = false) {
    const bool is_audio_bos =
        (all_generated_tokens == nullptr || all_generated_tokens->empty()) && (token_index_in_chunk == 0);
    if (is_audio_bos) {
        print_with_timestamp("TTS simplex: is_audio_bos=true (first audio token)\n");
    }

    if (is_audio_bos && ctx_omni->tts_condition_saved && ctx_omni->tts_condition_length > 0) {
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
        }
        int condition_n_past = 0;
        if (!prefill_with_emb_tts(ctx_omni, params, ctx_omni->tts_condition_embeddings.data(),
                                  ctx_omni->tts_condition_length, params->n_batch, &condition_n_past)) {
            LOG_ERR("TTS simplex: Failed to re-forward condition\n");
            return 0;
        }
        *n_past_tts = condition_n_past;
    }

    const float * hidden_state = llama_get_embeddings_ith(ctx_omni->ctx_tts_llama, -1);
    if (hidden_state == nullptr) {
        LOG_ERR("TTS simplex: failed to get hidden state\n");
        return 0;
    }

    if (ctx_omni->tts_aux.head_code_weight == nullptr) {
        LOG_ERR("TTS simplex: head_code weight not loaded\n");
        return 0;
    }

    if (ctx_omni->tts_aux.head_code_hidden_size != 768 ||
        ctx_omni->tts_aux.head_code_num_audio_tokens != OMNI_TTS_NUM_AUDIO_TOKENS) {
        LOG_ERR("TTS simplex: head_code dimensions mismatch\n");
        return 0;
    }

    std::vector<float> audio_logits(OMNI_TTS_NUM_AUDIO_TOKENS, 0.0f);
    const float *      head_code_w = ctx_omni->tts_aux.head_code_weight;
    const int          hidden_size = ctx_omni->tts_aux.head_code_hidden_size;
    for (int i = 0; i < OMNI_TTS_NUM_AUDIO_TOKENS; ++i) {
        const float * row = head_code_w + i * hidden_size;
        float         sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden_state[j] * row[j];
        }
        audio_logits[i] = sum;
    }

    std::mt19937 * rng = omni_tts_get_sampler_rng(smpl);
    std::mt19937   local_rng;
    if (rng == nullptr) {
        local_rng = std::mt19937(std::random_device{}());
        rng       = &local_rng;
    }

    std::vector<int> decoded_tokens_relative;
    if (all_generated_tokens != nullptr) {
        for (llama_token tid : *all_generated_tokens) {
            const int relative_idx = tid - OMNI_TTS_AUDIO_BOS_TOKEN_ID;
            if (relative_idx >= 0 && relative_idx < OMNI_TTS_NUM_AUDIO_TOKENS) {
                decoded_tokens_relative.push_back(relative_idx);
            }
        }
    }

    constexpr float temperature        = 0.8f;
    constexpr float repetition_penalty = 1.05f;
    constexpr int   win_size           = 8;

    const bool use_argmax            = params->sampling.temp <= 0.0f;
    int        selected_relative_idx = 0;
    if (use_argmax) {
        float max_logit = audio_logits[0];
        for (int i = 1; i < OMNI_TTS_NUM_AUDIO_TOKENS; ++i) {
            if (audio_logits[i] > max_logit) {
                max_logit             = audio_logits[i];
                selected_relative_idx = i;
            }
        }
    } else {
        for (int i = 0; i < OMNI_TTS_NUM_AUDIO_TOKENS; ++i) {
            audio_logits[i] /= temperature;
        }

        if (!is_audio_bos && !decoded_tokens_relative.empty()) {
            const int         start_idx = std::max(0, (int) decoded_tokens_relative.size() - win_size);
            std::vector<bool> occurred(OMNI_TTS_NUM_AUDIO_TOKENS, false);
            for (int i = start_idx; i < (int) decoded_tokens_relative.size(); ++i) {
                const int tok = decoded_tokens_relative[i];
                if (tok >= 0 && tok < OMNI_TTS_NUM_AUDIO_TOKENS) {
                    occurred[tok] = true;
                }
            }
            for (int i = 0; i < OMNI_TTS_NUM_AUDIO_TOKENS; ++i) {
                if (occurred[i]) {
                    if (audio_logits[i] >= 0) {
                        audio_logits[i] /= repetition_penalty;
                    } else {
                        audio_logits[i] *= repetition_penalty;
                    }
                }
            }
        }

        if (force_no_eos) {
            audio_logits[OMNI_TTS_AUDIO_EOS_RELATIVE_IDX] = -std::numeric_limits<float>::infinity();
        }

        selected_relative_idx = omni_tts_random_sampling(audio_logits.data(), OMNI_TTS_NUM_AUDIO_TOKENS, *rng);
    }

    if (selected_relative_idx < 0 || selected_relative_idx >= OMNI_TTS_NUM_AUDIO_TOKENS) {
        selected_relative_idx = 0;
    }

    const llama_token id = OMNI_TTS_AUDIO_BOS_TOKEN_ID + selected_relative_idx;
    common_sampler_accept(smpl, id, true);

    const bool is_eos = selected_relative_idx == OMNI_TTS_AUDIO_EOS_RELATIVE_IDX;
    if (is_eos && !is_final_text_chunk) {
        return id;
    }

    if (ctx_omni->tts_aux.emb_code_weight != nullptr && selected_relative_idx >= 0 &&
        selected_relative_idx < ctx_omni->tts_aux.emb_code_vocab_size) {
        const float * emb_code_w           = ctx_omni->tts_aux.emb_code_weight;
        const int     emb_code_hidden_size = ctx_omni->tts_aux.emb_code_hidden_size;
        const int     emb_code_vocab_size  = ctx_omni->tts_aux.emb_code_vocab_size;

        std::vector<float> audio_token_embedding(emb_code_hidden_size);
        if (ctx_omni->tts_aux.emb_code_stored_as_transposed) {
            for (int j = 0; j < emb_code_hidden_size; ++j) {
                audio_token_embedding[j] = emb_code_w[j * emb_code_vocab_size + selected_relative_idx];
            }
        } else {
            for (int j = 0; j < emb_code_hidden_size; ++j) {
                audio_token_embedding[j] = emb_code_w[selected_relative_idx * emb_code_hidden_size + j];
            }
        }

        if (!prefill_with_emb_tts(ctx_omni, params, audio_token_embedding.data(), 1, 1, n_past_tts)) {
            LOG_ERR("TTS simplex: failed to decode audio token embedding\n");
            return 0;
        }
    } else {
        LOG_ERR("TTS simplex: emb_code not available\n");
        return 0;
    }

    return id;
}

llama_token omni_tts_sample_token_internal(struct common_sampler *          smpl,
                                           struct omni_context *            ctx_omni,
                                           common_params *                  params,
                                           int *                            n_past_tts,
                                           const std::vector<llama_token> * all_generated_tokens,
                                           const std::vector<llama_token> * chunk_generated_tokens,
                                           int                              token_index_in_chunk,
                                           bool                             force_no_eos,
                                           bool                             is_final_text_chunk = false) {
    const char * logits_debug_dir = getenv("TTS_LOGITS_DEBUG_DIR");

    const bool is_first_token_overall =
        (all_generated_tokens == nullptr || all_generated_tokens->empty()) && (token_index_in_chunk == 0);
    const bool skip_processors = ctx_omni->duplex_mode ? token_index_in_chunk == 0 : is_first_token_overall;

    if (is_first_token_overall) {
        print_with_timestamp("TTS sample: is_first_token_overall=true, duplex_mode=%d\n", ctx_omni->duplex_mode);
    }

    if (is_first_token_overall && ctx_omni->tts_condition_saved && ctx_omni->tts_condition_length > 0) {
        const int    cond_len      = ctx_omni->tts_condition_length;
        const int    cond_n_embd   = ctx_omni->tts_condition_n_embd;
        const size_t cond_emb_size = ctx_omni->tts_condition_embeddings.size();
        const size_t expected_size = (size_t) cond_len * cond_n_embd;

        if (cond_len <= 0 || cond_len > 10000) {
            LOG_ERR("TTS sample: invalid tts_condition_length=%d\n", cond_len);
            return 0;
        }
        if (cond_n_embd <= 0 || cond_n_embd > 10000) {
            LOG_ERR("TTS sample: invalid tts_condition_n_embd=%d\n", cond_n_embd);
            return 0;
        }
        if (cond_emb_size != expected_size) {
            LOG_ERR("TTS sample: tts_condition_embeddings size mismatch: %zu != %zu (len=%d * n_embd=%d)\n",
                    cond_emb_size, expected_size, cond_len, cond_n_embd);
            return 0;
        }

        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
        } else {
            LOG_ERR("TTS: Failed to get memory for KV cache clear\n");
        }

        int condition_n_past = 0;
        if (!prefill_with_emb_tts(ctx_omni, params, ctx_omni->tts_condition_embeddings.data(),
                                  ctx_omni->tts_condition_length, params->n_batch, &condition_n_past)) {
            LOG_ERR("TTS: Failed to re-forward condition for first audio token\n");
            return 0;
        }

        *n_past_tts = condition_n_past;
    }

    const float * hidden_state = llama_get_embeddings_ith(ctx_omni->ctx_tts_llama, -1);
    if (hidden_state == nullptr) {
        LOG_ERR("TTS: failed to get hidden state from TTS model\n");
        return 0;
    }

    if (logits_debug_dir != nullptr) {
        const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
        omni_tts_save_hidden_states_to_file(logits_debug_dir, hidden_state, hidden_size, token_index_in_chunk);
    }

    if (ctx_omni->tts_aux.head_code_weight == nullptr) {
        LOG_ERR("TTS: head_code weight not loaded\n");
        return 0;
    }

    if (ctx_omni->tts_aux.head_code_hidden_size != 768 ||
        ctx_omni->tts_aux.head_code_num_audio_tokens != OMNI_TTS_NUM_AUDIO_TOKENS) {
        LOG_ERR("TTS: head_code dimensions mismatch: expected (768, 6562), got (%d, %d)\n",
                ctx_omni->tts_aux.head_code_hidden_size, ctx_omni->tts_aux.head_code_num_audio_tokens);
        return 0;
    }

    std::vector<float> audio_logits(OMNI_TTS_NUM_AUDIO_TOKENS, 0.0f);
    const float *      head_code_w = ctx_omni->tts_aux.head_code_weight;
    const int          hidden_size = ctx_omni->tts_aux.head_code_hidden_size;
    for (int i = 0; i < OMNI_TTS_NUM_AUDIO_TOKENS; ++i) {
        const float * row = head_code_w + i * hidden_size;
        float         sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden_state[j] * row[j];
        }
        audio_logits[i] = sum;
    }

    if (logits_debug_dir != nullptr) {
        omni_tts_save_logits_to_file(logits_debug_dir, audio_logits.data(), OMNI_TTS_NUM_AUDIO_TOKENS,
                                     token_index_in_chunk);
    }

    const char * output_dir = getenv("TTS_OUTPUT_DIR");
    if (output_dir != nullptr) {
        if (token_index_in_chunk == 0) {
            char hidden_state_path[512];
            snprintf(hidden_state_path, sizeof(hidden_state_path), "%s/cpp_first_hidden_state.bin", output_dir);
            FILE * f_hidden = fopen(hidden_state_path, "wb");
            if (f_hidden != nullptr) {
                fwrite(hidden_state, sizeof(float), hidden_size, f_hidden);
                fclose(f_hidden);
            }
        }

        char logits_path[512];
        snprintf(logits_path, sizeof(logits_path), "%s/cpp_logits_token_%d.bin", output_dir, token_index_in_chunk);
        FILE * f_logits = fopen(logits_path, "wb");
        if (f_logits != nullptr) {
            fwrite(audio_logits.data(), sizeof(float), OMNI_TTS_NUM_AUDIO_TOKENS, f_logits);
            fclose(f_logits);
        }
    }

    std::mt19937 * rng = omni_tts_get_sampler_rng(smpl);
    std::mt19937   local_rng;
    if (rng == nullptr) {
        LOG_WRN("TTS: sampler RNG not available, using local RNG\n");
        local_rng = std::mt19937(std::random_device{}());
        rng       = &local_rng;
    }

    std::vector<int>                 decoded_tokens_relative;
    const std::vector<llama_token> * tokens_for_penalty =
        ctx_omni->duplex_mode ? (chunk_generated_tokens != nullptr ? chunk_generated_tokens : all_generated_tokens) :
                                all_generated_tokens;
    if (tokens_for_penalty != nullptr) {
        for (llama_token tid : *tokens_for_penalty) {
            const int relative_idx = tid - OMNI_TTS_AUDIO_BOS_TOKEN_ID;
            if (relative_idx >= 0 && relative_idx < OMNI_TTS_NUM_AUDIO_TOKENS) {
                decoded_tokens_relative.push_back(relative_idx);
            }
        }
    }

    constexpr float temperature        = 0.8f;
    constexpr float top_p              = 0.85f;
    constexpr int   top_k              = 25;
    constexpr float repetition_penalty = 1.05f;
    constexpr int   win_size           = 16;
    constexpr int   min_tokens_to_keep = 3;

    const bool use_argmax            = params->sampling.temp <= 0.0f;
    int        selected_relative_idx = 0;
    if (use_argmax) {
        float max_logit = audio_logits[0];
        for (int i = 1; i < OMNI_TTS_NUM_AUDIO_TOKENS; ++i) {
            if (audio_logits[i] > max_logit) {
                max_logit             = audio_logits[i];
                selected_relative_idx = i;
            }
        }
    } else {
        for (int i = 0; i < OMNI_TTS_NUM_AUDIO_TOKENS; ++i) {
            audio_logits[i] /= temperature;
        }

        if (!skip_processors && !decoded_tokens_relative.empty()) {
            omni_tts_apply_repetition_penalty(audio_logits.data(), OMNI_TTS_NUM_AUDIO_TOKENS, decoded_tokens_relative,
                                              repetition_penalty, win_size);
        }

        if (ctx_omni->duplex_mode && force_no_eos) {
            audio_logits[OMNI_TTS_AUDIO_EOS_RELATIVE_IDX] = -std::numeric_limits<float>::infinity();
        }

        selected_relative_idx = omni_tts_nucleus_sampling_with_min_keep(audio_logits.data(), OMNI_TTS_NUM_AUDIO_TOKENS,
                                                                        top_p, top_k, min_tokens_to_keep, *rng);
    }

    if (selected_relative_idx < 0 || selected_relative_idx >= OMNI_TTS_NUM_AUDIO_TOKENS) {
        LOG_ERR("TTS: invalid selected index %d, should be in [0, %d)\n", selected_relative_idx,
                OMNI_TTS_NUM_AUDIO_TOKENS);
        selected_relative_idx = 0;
    }

    const llama_token id = OMNI_TTS_AUDIO_BOS_TOKEN_ID + selected_relative_idx;
    common_sampler_accept(smpl, id, true);

    const bool is_eos = selected_relative_idx == OMNI_TTS_AUDIO_EOS_RELATIVE_IDX;
    if (ctx_omni->duplex_mode && is_eos && !is_final_text_chunk) {
        return id;
    }

    if (ctx_omni->tts_aux.emb_code_weight != nullptr && selected_relative_idx >= 0 &&
        selected_relative_idx < ctx_omni->tts_aux.emb_code_vocab_size) {
        const float * emb_code_w           = ctx_omni->tts_aux.emb_code_weight;
        const int     emb_code_hidden_size = ctx_omni->tts_aux.emb_code_hidden_size;
        const int     emb_code_vocab_size  = ctx_omni->tts_aux.emb_code_vocab_size;

        std::vector<float> audio_token_embedding(emb_code_hidden_size);
        if (ctx_omni->tts_aux.emb_code_stored_as_transposed) {
            for (int j = 0; j < emb_code_hidden_size; ++j) {
                audio_token_embedding[j] = emb_code_w[j * emb_code_vocab_size + selected_relative_idx];
            }
        } else {
            for (int j = 0; j < emb_code_hidden_size; ++j) {
                audio_token_embedding[j] = emb_code_w[selected_relative_idx * emb_code_hidden_size + j];
            }
        }

        if (!prefill_with_emb_tts(ctx_omni, params, audio_token_embedding.data(), 1, 1, n_past_tts)) {
            LOG_ERR("TTS: failed to decode audio token embedding\n");
            return 0;
        }
    } else {
        LOG_ERR("TTS: emb_code not available, falling back to token IDs (may fail if token exceeds vocab)\n");
        std::vector<llama_token> tokens;
        tokens.push_back(id);
        if (!omni_tts_eval_tokens(ctx_omni, params, tokens, 1, n_past_tts)) {
            LOG_ERR("TTS: failed to decode audio token ID (token may exceed vocab size)\n");
            return 0;
        }
    }

    return id;
}

}  // namespace

bool omni_tts_is_valid_token(llama_token tid) {
    return omni_tts_is_valid_token_internal(tid);
}

bool omni_tts_prepare_duplex_chunk(struct omni_context *            ctx_omni,
                                   const std::string &              llm_debug_output_dir,
                                   int                              current_chunk_idx,
                                   const std::string &              llm_text,
                                   const std::vector<llama_token> & current_chunk_token_ids,
                                   const std::vector<float> &       current_chunk_hidden_states,
                                   int                              current_chunk_n_embd,
                                   OmniTtsPreparedChunk &           prepared_chunk) {
    prepared_chunk = {};

    if (current_chunk_n_embd <= 0 || current_chunk_n_embd > 16384) {
        LOG_ERR("TTS Duplex: invalid current_chunk_n_embd=%d\n", current_chunk_n_embd);
        return false;
    }

    const size_t expected_hidden_size = current_chunk_token_ids.size() * (size_t) current_chunk_n_embd;
    if (current_chunk_hidden_states.size() != expected_hidden_size) {
        LOG_ERR("TTS Duplex: hidden_states size mismatch\n");
        return false;
    }

    prepared_chunk.n_tokens_orig = (int) (current_chunk_hidden_states.size() / current_chunk_n_embd);

    std::vector<llama_token> filtered_token_ids     = current_chunk_token_ids;
    std::vector<float>       filtered_hidden_states = current_chunk_hidden_states;
    omni_tts_filter_special_tokens(filtered_token_ids, filtered_hidden_states, current_chunk_n_embd);
    prepared_chunk.n_tokens_filtered = (int) (filtered_hidden_states.size() / current_chunk_n_embd);
    if (prepared_chunk.n_tokens_filtered <= 0) {
        return false;
    }

    if (!omni_tts_compute_merged_embeddings(
            ctx_omni, filtered_token_ids, filtered_hidden_states, current_chunk_n_embd, true,
            "TTS: WARNING - TTS weights not loaded, skipping merged embedding computation", prepared_chunk, nullptr)) {
        return false;
    }

    omni_tts_write_duplex_debug_dump(llm_debug_output_dir, current_chunk_idx, llm_text, current_chunk_token_ids,
                                     current_chunk_hidden_states, current_chunk_n_embd, prepared_chunk);

    return true;
}

bool omni_tts_prepare_simplex_chunk(struct omni_context *            ctx_omni,
                                    const std::string &              llm_debug_output_dir,
                                    int                              current_chunk_idx,
                                    const std::string &              response,
                                    const std::vector<llama_token> & current_chunk_token_ids,
                                    const std::vector<float> &       current_chunk_hidden_states,
                                    int                              current_chunk_n_embd,
                                    OmniTtsPreparedChunk &           prepared_chunk) {
    prepared_chunk = {};

    if (current_chunk_n_embd <= 0 || current_chunk_n_embd > 16384) {
        LOG_ERR("TTS: invalid current_chunk_n_embd=%d, skipping chunk %d\n", current_chunk_n_embd, current_chunk_idx);
        return false;
    }

    if (current_chunk_hidden_states.size() > 100000000) {
        LOG_ERR("TTS: hidden_states size too large (%zu), possible corruption, skipping chunk %d\n",
                current_chunk_hidden_states.size(), current_chunk_idx);
        return false;
    }

    const size_t expected_hidden_size = current_chunk_token_ids.size() * (size_t) current_chunk_n_embd;
    if (current_chunk_hidden_states.size() != expected_hidden_size) {
        LOG_ERR("TTS: hidden_states size mismatch: got %zu, expected %zu (tokens=%zu * n_embd=%d), skipping chunk %d\n",
                current_chunk_hidden_states.size(), expected_hidden_size, current_chunk_token_ids.size(),
                current_chunk_n_embd, current_chunk_idx);
        return false;
    }

    prepared_chunk.n_tokens_orig = (int) (current_chunk_hidden_states.size() / current_chunk_n_embd);

    std::vector<llama_token> filtered_token_ids     = current_chunk_token_ids;
    std::vector<float>       filtered_hidden_states = current_chunk_hidden_states;
    omni_tts_filter_special_tokens(filtered_token_ids, filtered_hidden_states, current_chunk_n_embd);
    prepared_chunk.n_tokens_filtered = (int) (filtered_hidden_states.size() / current_chunk_n_embd);

    print_with_timestamp("TTS: n_tokens_orig=%d, n_tokens_filtered=%d (filtered %d special tokens)\n",
                         prepared_chunk.n_tokens_orig, prepared_chunk.n_tokens_filtered,
                         prepared_chunk.n_tokens_orig - prepared_chunk.n_tokens_filtered);

    if (prepared_chunk.n_tokens_filtered <= 0) {
        LOG_WRN("TTS: all tokens filtered out, skipping chunk %d\n", current_chunk_idx);
        return false;
    }

    OmniTtsComputationDebugData debug_data;
    if (!omni_tts_compute_merged_embeddings(
            ctx_omni, filtered_token_ids, filtered_hidden_states, current_chunk_n_embd, false,
            "TTS: WARNING - TTS weights not loaded, skipping merged embedding computation", prepared_chunk,
            &debug_data)) {
        return false;
    }

    omni_tts_write_simplex_debug_dump(llm_debug_output_dir, current_chunk_idx, response, current_chunk_token_ids,
                                      current_chunk_hidden_states, current_chunk_n_embd, prepared_chunk, debug_data);

    return true;
}

bool omni_tts_generate_audio_tokens_local_simplex(struct omni_context *      ctx_omni,
                                                  common_params *            params,
                                                  const std::vector<float> & merged_embeddings,
                                                  int                        n_tokens,
                                                  int                        tts_n_embd,
                                                  int                        chunk_idx,
                                                  std::vector<int32_t> &     output_audio_tokens,
                                                  const OmniRoundMeta &      round_meta,
                                                  const std::string &        output_dir,
                                                  bool                       is_final_text_chunk) {
    print_with_timestamp("TTS Simplex: generating audio tokens for chunk %d (n_tokens=%d, tts_n_embd=%d)\n", chunk_idx,
                         n_tokens, tts_n_embd);

    const int max_audio_tokens = 500;

    if (!ctx_omni->ctx_tts_llama || !ctx_omni->model_tts) {
        LOG_ERR("TTS Simplex: TTS model not loaded\n");
        return false;
    }

    if (!ctx_omni->tts_aux.head_code_weight || !ctx_omni->tts_aux.emb_code_weight) {
        LOG_ERR("TTS Simplex: TTS weights not loaded\n");
        return false;
    }

    if (merged_embeddings.size() != (size_t) (n_tokens * tts_n_embd)) {
        LOG_ERR("TTS Simplex: merged_embeddings size mismatch: %zu != %d * %d\n", merged_embeddings.size(), n_tokens,
                tts_n_embd);
        return false;
    }

    std::vector<float> condition_with_bos = merged_embeddings;
    int                extra_tokens       = 0;

    std::vector<float> audio_bos_embed(tts_n_embd, 0.0f);
    if (omni_tts_emb_text(ctx_omni, OMNI_TTS_AUDIO_BOS_TOKEN_ID, audio_bos_embed.data(), tts_n_embd)) {
        condition_with_bos.insert(condition_with_bos.end(), audio_bos_embed.begin(), audio_bos_embed.end());
        extra_tokens++;
        print_with_timestamp("TTS Simplex: 在 prefill 前添加 audio_bos (chunk_idx=%d, new_size=%zu)\n", chunk_idx,
                             condition_with_bos.size() / tts_n_embd);
    } else {
        LOG_ERR("TTS Simplex: failed to get audio_bos embedding\n");
    }
    const int n_tokens_with_bos = n_tokens + extra_tokens;

    ctx_omni->tts_condition_embeddings = condition_with_bos;
    ctx_omni->tts_condition_length     = n_tokens_with_bos;
    ctx_omni->tts_condition_n_embd     = tts_n_embd;
    ctx_omni->tts_condition_saved      = true;

    int n_past_tts = 0;
    if (chunk_idx == 0) {
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
            print_with_timestamp("TTS Simplex: first chunk - cleared KV cache\n");
        }
        ctx_omni->tts_n_past_accumulated = 0;
        ctx_omni->tts_all_generated_tokens.clear();
        ctx_omni->tts_token_buffer.clear();
        print_with_timestamp("TTS Simplex: first chunk - cleared tts_all_generated_tokens and tts_token_buffer\n");
    } else {
        n_past_tts = ctx_omni->tts_n_past_accumulated;
        print_with_timestamp("TTS Simplex: chunk %d - keeping KV cache, n_past_tts=%d\n", chunk_idx, n_past_tts);
    }

    if (!prefill_with_emb_tts(ctx_omni, params, condition_with_bos.data(), n_tokens_with_bos, params->n_batch,
                              &n_past_tts)) {
        LOG_ERR("TTS Simplex: prefill_with_emb_tts failed\n");
        return false;
    }
    print_with_timestamp("TTS Simplex: prefill completed, n_past_tts=%d\n", n_past_tts);

    common_params_sampling tts_sampling = params->sampling;
    tts_sampling.temp                   = 0.8f;
    tts_sampling.top_p                  = 0.85f;
    tts_sampling.top_k                  = 25;
    tts_sampling.penalty_repeat         = 1.05f;
    tts_sampling.min_p                  = 0.01f;
    tts_sampling.penalty_last_n         = 16;

    struct common_sampler * tts_sampler = common_sampler_init(ctx_omni->model_tts, tts_sampling);
    if (tts_sampler == nullptr) {
        LOG_ERR("TTS Simplex: failed to create sampler\n");
        return false;
    }
    print_with_timestamp("TTS Simplex: sampler created\n");

    output_audio_tokens.clear();
    constexpr int min_new_tokens = 0;
    bool          need_phase2    = false;

    for (int t = 0; t < max_audio_tokens; ++t) {
        if (ctx_omni->gate.break_event.load()) {
            print_with_timestamp("TTS Simplex: break_event detected at step %d, stopping immediately\n", t);
            break;
        }

        const bool        force_no_eos      = t < min_new_tokens;
        const llama_token sampled_token_abs = omni_tts_sample_token_simplex_internal(
            tts_sampler, ctx_omni, params, &n_past_tts, &ctx_omni->tts_all_generated_tokens, t, force_no_eos, false);

        if (sampled_token_abs == 0) {
            LOG_ERR("TTS Simplex: sample_tts_token failed at step %d\n", t);
            break;
        }

        const int relative_idx = sampled_token_abs - OMNI_TTS_AUDIO_BOS_TOKEN_ID;
        if (relative_idx < 0 || relative_idx >= OMNI_TTS_NUM_AUDIO_TOKENS) {
            LOG_ERR("TTS Simplex: invalid token ID %d at step %d\n", sampled_token_abs, t);
            break;
        }

        output_audio_tokens.push_back(relative_idx);
        ctx_omni->tts_all_generated_tokens.push_back(sampled_token_abs);

        const bool is_eos = relative_idx == OMNI_TTS_AUDIO_EOS_RELATIVE_IDX;
        if (is_eos) {
            print_with_timestamp("TTS Simplex Phase1: EOS token at step %d\n", t + 1);
            output_audio_tokens.pop_back();
            ctx_omni->tts_all_generated_tokens.pop_back();
            if (is_final_text_chunk) {
                need_phase2 = true;
                print_with_timestamp("TTS Simplex: is_final_text_chunk=true, will enter Phase 2 for text_eos_embed\n");
            }
        } else {
            ctx_omni->tts_token_buffer.push_back(relative_idx);
        }

        while ((int) ctx_omni->tts_token_buffer.size() >= OMNI_TTS_STREAM_CHUNK_SIZE && ctx_omni->t2w_thread_info) {
            T2WOut * t2w_out = new T2WOut();
            t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(),
                                         ctx_omni->tts_token_buffer.begin() + OMNI_TTS_STREAM_CHUNK_SIZE);
            t2w_out->is_final   = false;
            t2w_out->round_meta = round_meta;

            {
                std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                ctx_omni->t2w_thread_info->queue.push(t2w_out);
            }
            ctx_omni->t2w_thread_info->cv.notify_one();

            print_with_timestamp("TTS Simplex Phase1: yield %d tokens 到 T2W (step %d, buffer=%zu)\n",
                                 OMNI_TTS_STREAM_CHUNK_SIZE, t + 1, ctx_omni->tts_token_buffer.size());
            ctx_omni->tts_token_buffer.erase(ctx_omni->tts_token_buffer.begin(),
                                             ctx_omni->tts_token_buffer.begin() + OMNI_TTS_STREAM_CHUNK_SIZE);
        }

        if (t < 5 || (t + 1) % 25 == 0) {
            print_with_timestamp("TTS Simplex Phase1: token %d/%d: rel_id=%d\n", t + 1, max_audio_tokens, relative_idx);
        }

        if (is_eos) {
            break;
        }
    }

    if (need_phase2 && !ctx_omni->gate.break_event.load()) {
        print_with_timestamp("TTS Simplex Phase2: injecting text_eos_embed + audio_bos at n_past=%d\n", n_past_tts);

        std::vector<float> text_eos_embed(tts_n_embd, 0.0f);
        bool               inject_ok = false;
        if (omni_tts_emb_text(ctx_omni, OMNI_TTS_TEXT_EOS_TOKEN_ID, text_eos_embed.data(), tts_n_embd)) {
            if (prefill_with_emb_tts(ctx_omni, params, text_eos_embed.data(), 1, 1, &n_past_tts)) {
                print_with_timestamp("TTS Simplex Phase2: text_eos_embed injected, n_past=%d\n", n_past_tts);

                std::vector<float> audio_bos_embed2(tts_n_embd, 0.0f);
                if (omni_tts_emb_text(ctx_omni, OMNI_TTS_AUDIO_BOS_TOKEN_ID, audio_bos_embed2.data(), tts_n_embd)) {
                    if (prefill_with_emb_tts(ctx_omni, params, audio_bos_embed2.data(), 1, 1, &n_past_tts)) {
                        print_with_timestamp(
                            "TTS Simplex Phase2: audio_bos injected, n_past=%d, starting final generation\n",
                            n_past_tts);
                        inject_ok = true;
                    }
                }
            }
        }

        if (inject_ok) {
            for (int t2 = 0; t2 < max_audio_tokens; ++t2) {
                if (ctx_omni->gate.break_event.load()) {
                    print_with_timestamp("TTS Simplex Phase2: break_event at step %d\n", t2);
                    break;
                }

                const llama_token sampled_token_abs = omni_tts_sample_token_simplex_internal(
                    tts_sampler, ctx_omni, params, &n_past_tts, &ctx_omni->tts_all_generated_tokens, t2, false, true);

                if (sampled_token_abs == 0) {
                    LOG_ERR("TTS Simplex Phase2: sample failed at step %d\n", t2);
                    break;
                }

                const int relative_idx = sampled_token_abs - OMNI_TTS_AUDIO_BOS_TOKEN_ID;
                if (relative_idx < 0 || relative_idx >= OMNI_TTS_NUM_AUDIO_TOKENS) {
                    LOG_ERR("TTS Simplex Phase2: invalid token %d at step %d\n", sampled_token_abs, t2);
                    break;
                }

                output_audio_tokens.push_back(relative_idx);
                ctx_omni->tts_all_generated_tokens.push_back(sampled_token_abs);

                const bool is_eos = relative_idx == OMNI_TTS_AUDIO_EOS_RELATIVE_IDX;
                if (is_eos) {
                    print_with_timestamp("TTS Simplex Phase2: EOS at step %d (final end)\n", t2 + 1);
                    output_audio_tokens.pop_back();
                    ctx_omni->tts_all_generated_tokens.pop_back();
                } else {
                    ctx_omni->tts_token_buffer.push_back(relative_idx);
                }

                while ((int) ctx_omni->tts_token_buffer.size() >= OMNI_TTS_STREAM_CHUNK_SIZE &&
                       ctx_omni->t2w_thread_info) {
                    T2WOut * t2w_out = new T2WOut();
                    t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(),
                                                 ctx_omni->tts_token_buffer.begin() + OMNI_TTS_STREAM_CHUNK_SIZE);
                    t2w_out->is_final   = false;
                    t2w_out->round_meta = round_meta;
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                    print_with_timestamp("TTS Simplex Phase2: yield %d tokens 到 T2W\n", OMNI_TTS_STREAM_CHUNK_SIZE);
                    ctx_omni->tts_token_buffer.erase(ctx_omni->tts_token_buffer.begin(),
                                                     ctx_omni->tts_token_buffer.begin() + OMNI_TTS_STREAM_CHUNK_SIZE);
                }

                if (is_eos) {
                    break;
                }
            }
        } else {
            LOG_ERR("TTS Simplex Phase2: failed to inject text_eos_embed/audio_bos\n");
        }
    }

    print_with_timestamp("TTS Simplex: chunk 结束，tts_token_buffer 剩余 %zu tokens (保留等下一个 chunk)\n",
                         ctx_omni->tts_token_buffer.size());

    if (is_final_text_chunk && !ctx_omni->tts_token_buffer.empty() && ctx_omni->t2w_thread_info) {
        T2WOut * t2w_out = new T2WOut();
        t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(), ctx_omni->tts_token_buffer.end());
        t2w_out->is_final   = false;
        t2w_out->round_meta = round_meta;

        {
            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
            ctx_omni->t2w_thread_info->queue.push(t2w_out);
        }
        ctx_omni->t2w_thread_info->cv.notify_one();

        print_with_timestamp("TTS Simplex: is_final_text_chunk=true, flush 剩余 %zu tokens 从 tts_token_buffer\n",
                             ctx_omni->tts_token_buffer.size());
        ctx_omni->tts_token_buffer.clear();
    }

    if (ctx_omni->duplex_mode && ctx_omni->t2w_thread_info) {
        T2WOut * t2w_out = new T2WOut();
        t2w_out->audio_tokens.clear();
        t2w_out->is_final     = false;
        t2w_out->is_chunk_end = true;
        t2w_out->round_meta   = round_meta;

        {
            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
            ctx_omni->t2w_thread_info->queue.push(t2w_out);
        }
        ctx_omni->t2w_thread_info->cv.notify_one();
        print_with_timestamp("TTS Simplex: 发送 is_chunk_end=true 信号\n");
    }

    common_sampler_free(tts_sampler);

    print_with_timestamp("TTS Simplex: generated %zu audio tokens for chunk %d\n", output_audio_tokens.size(),
                         chunk_idx);

    if (!output_audio_tokens.empty()) {
        std::string first_tokens;
        std::string last_tokens;
        const int   show_count = std::min((int) output_audio_tokens.size(), 5);
        for (int i = 0; i < show_count; ++i) {
            first_tokens += std::to_string(output_audio_tokens[i]) + " ";
        }
        for (int i = std::max(0, (int) output_audio_tokens.size() - 5); i < (int) output_audio_tokens.size(); ++i) {
            last_tokens += std::to_string(output_audio_tokens[i]) + " ";
        }
        print_with_timestamp("TTS Simplex: chunk %d first tokens: [%s], last tokens: [%s]\n", chunk_idx,
                             first_tokens.c_str(), last_tokens.c_str());
    }

    ctx_omni->tts_n_past_accumulated = n_past_tts;
    print_with_timestamp("TTS Simplex: updated tts_n_past_accumulated=%d, total_generated_tokens=%zu\n",
                         ctx_omni->tts_n_past_accumulated, ctx_omni->tts_all_generated_tokens.size());

    if (!output_dir.empty() && !output_audio_tokens.empty()) {
        const std::string tokens_file = output_dir + "/audio_tokens_chunk_" + std::to_string(chunk_idx) + ".bin";
        FILE *            f           = fopen(tokens_file.c_str(), "wb");
        if (f != nullptr) {
            fwrite(output_audio_tokens.data(), sizeof(int32_t), output_audio_tokens.size(), f);
            fclose(f);
        }
    }

    return !output_audio_tokens.empty();
}

bool omni_tts_generate_audio_tokens_local(struct omni_context *      ctx_omni,
                                          struct common_params *     params,
                                          const std::vector<float> & merged_embeddings,
                                          int                        n_tokens,
                                          int                        tts_n_embd,
                                          int                        chunk_idx,
                                          int                        duplex_chunk_idx,
                                          std::vector<int32_t> &     output_audio_tokens,
                                          bool                       is_end_of_turn,
                                          const OmniRoundMeta &      round_meta,
                                          const std::string &        output_dir) {
    print_with_timestamp("TTS Local: generating audio tokens for chunk %d (n_tokens=%d, tts_n_embd=%d, emb_size=%zu)\n",
                         chunk_idx, n_tokens, tts_n_embd, merged_embeddings.size());
    const auto tts_stage_start  = std::chrono::high_resolution_clock::now();
    auto       finish_tts_stage = [&](bool ok) {
        omni_tts_duplex_timing_note(
            ctx_omni, duplex_chunk_idx,
            omni_tts_timing_elapsed_ms(tts_stage_start, std::chrono::high_resolution_clock::now()),
            (int) output_audio_tokens.size());
        return ok;
    };

    if (n_tokens < 0) {
        LOG_ERR("TTS Local: invalid n_tokens=%d\n", n_tokens);
        return finish_tts_stage(false);
    }
    if (n_tokens == 0 && !is_end_of_turn) {
        LOG_ERR("TTS Local: n_tokens=0 but is_end_of_turn=false, nothing to generate\n");
        return finish_tts_stage(false);
    }
    if (n_tokens > 10000) {
        LOG_ERR("TTS Local: n_tokens=%d seems too large, likely data corruption\n", n_tokens);
        return finish_tts_stage(false);
    }
    if (tts_n_embd <= 0 || tts_n_embd > 10000) {
        LOG_ERR("TTS Local: invalid tts_n_embd=%d\n", tts_n_embd);
        return finish_tts_stage(false);
    }

    if (n_tokens == 0 && is_end_of_turn) {
        print_with_timestamp(
            "TTS Local: n_tokens=0 but is_end_of_turn=true, will add text_eos_embed and generate final tokens\n");
    }

    const int max_audio_tokens = ctx_omni->duplex_mode ? 26 : 500;
    print_with_timestamp("TTS Local: mode=%s, max_audio_tokens=%d\n", ctx_omni->duplex_mode ? "duplex" : "simplex",
                         max_audio_tokens);

    if (!ctx_omni->ctx_tts_llama || !ctx_omni->model_tts) {
        LOG_ERR("TTS Local: TTS model not loaded\n");
        return finish_tts_stage(false);
    }

    if (!ctx_omni->tts_aux.head_code_weight || !ctx_omni->tts_aux.emb_code_weight) {
        LOG_ERR("TTS Local: TTS weights not loaded (head_code or emb_code)\n");
        return finish_tts_stage(false);
    }

    if (merged_embeddings.size() != (size_t) (n_tokens * tts_n_embd)) {
        LOG_ERR("TTS Local: merged_embeddings size mismatch: %zu != %d * %d\n", merged_embeddings.size(), n_tokens,
                tts_n_embd);
        return finish_tts_stage(false);
    }

    std::vector<float> condition_with_bos = merged_embeddings;
    int                extra_tokens       = 0;

    if (is_end_of_turn) {
        std::vector<float> text_eos_embed(tts_n_embd, 0.0f);
        if (omni_tts_emb_text(ctx_omni, OMNI_TTS_TEXT_EOS_TOKEN_ID, text_eos_embed.data(), tts_n_embd)) {
            condition_with_bos.insert(condition_with_bos.end(), text_eos_embed.begin(), text_eos_embed.end());
            extra_tokens++;
            print_with_timestamp("TTS Local: is_end_of_turn=true, 添加 text_eos_embed (chunk_idx=%d, new_size=%zu)\n",
                                 chunk_idx, condition_with_bos.size() / tts_n_embd);
        } else {
            LOG_WRN("TTS Local: failed to get text_eos embedding\n");
        }
    }

    std::vector<float> audio_bos_embed(tts_n_embd, 0.0f);
    if (omni_tts_emb_text(ctx_omni, OMNI_TTS_AUDIO_BOS_TOKEN_ID, audio_bos_embed.data(), tts_n_embd)) {
        condition_with_bos.insert(condition_with_bos.end(), audio_bos_embed.begin(), audio_bos_embed.end());
        extra_tokens++;
        print_with_timestamp("TTS Local: 在 prefill 前添加 audio_bos (chunk_idx=%d, new_size=%zu)\n", chunk_idx,
                             condition_with_bos.size() / tts_n_embd);
    } else {
        LOG_ERR("TTS Local: failed to get audio_bos embedding\n");
    }
    const int n_tokens_with_bos = n_tokens + extra_tokens;

    ctx_omni->tts_condition_embeddings = condition_with_bos;
    ctx_omni->tts_condition_length     = n_tokens_with_bos;
    ctx_omni->tts_condition_n_embd     = tts_n_embd;
    ctx_omni->tts_condition_saved      = true;

    int n_past_tts = 0;
    if (chunk_idx == 0) {
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
            print_with_timestamp("TTS Local: first chunk - cleared KV cache\n");
        }
        ctx_omni->tts_n_past_accumulated = 0;
        ctx_omni->tts_all_generated_tokens.clear();
        if (!ctx_omni->duplex_mode) {
            ctx_omni->tts_token_buffer.clear();
        }
        print_with_timestamp("TTS Local: first chunk - cleared tts_all_generated_tokens and tts_token_buffer\n");
    } else {
        n_past_tts = ctx_omni->tts_n_past_accumulated;
        print_with_timestamp("TTS Local: chunk %d - keeping KV cache, n_past_tts=%d\n", chunk_idx, n_past_tts);
    }

    if (!prefill_with_emb_tts(ctx_omni, params, condition_with_bos.data(), n_tokens_with_bos, params->n_batch,
                              &n_past_tts)) {
        LOG_ERR("TTS Local: prefill_with_emb_tts failed\n");
        return finish_tts_stage(false);
    }
    print_with_timestamp("TTS Local: prefill completed, n_past_tts=%d\n", n_past_tts);

    common_params_sampling tts_sampling = params->sampling;
    tts_sampling.temp                   = 0.8f;
    tts_sampling.top_p                  = 0.85f;
    tts_sampling.top_k                  = 25;
    tts_sampling.penalty_repeat         = 1.05f;
    tts_sampling.min_p                  = 0.01f;
    tts_sampling.penalty_last_n         = 16;

    struct common_sampler * tts_sampler = common_sampler_init(ctx_omni->model_tts, tts_sampling);
    if (tts_sampler == nullptr) {
        LOG_ERR("TTS Local: failed to create sampler\n");
        return finish_tts_stage(false);
    }

    output_audio_tokens.clear();
    std::vector<llama_token> chunk_generated_tokens;

    const int min_new_tokens = ctx_omni->duplex_mode ? (is_end_of_turn ? 0 : 26) : 100;

    bool                 first_chunk_pushed = false;
    std::vector<int32_t> stream_buffer;

    for (int t = 0; t < max_audio_tokens; ++t) {
        const bool        force_no_eos      = t < min_new_tokens;
        const llama_token sampled_token_abs = omni_tts_sample_token_internal(
            tts_sampler, ctx_omni, params, &n_past_tts, &ctx_omni->tts_all_generated_tokens, &chunk_generated_tokens, t,
            force_no_eos, is_end_of_turn);

        if (sampled_token_abs == 0) {
            LOG_ERR("TTS Local: sample_tts_token failed at step %d\n", t);
            break;
        }

        const int relative_idx = sampled_token_abs - OMNI_TTS_AUDIO_BOS_TOKEN_ID;
        if (relative_idx < 0 || relative_idx >= OMNI_TTS_NUM_AUDIO_TOKENS) {
            LOG_ERR("TTS Local: invalid token ID %d (relative_idx: %d) at step %d\n", sampled_token_abs, relative_idx,
                    t);
            break;
        }

        output_audio_tokens.push_back(relative_idx);
        stream_buffer.push_back(relative_idx);
        ctx_omni->tts_all_generated_tokens.push_back(sampled_token_abs);
        chunk_generated_tokens.push_back(sampled_token_abs);

        const bool is_eos = relative_idx == OMNI_TTS_AUDIO_EOS_RELATIVE_IDX;
        if (is_eos) {
            output_audio_tokens.pop_back();
            stream_buffer.pop_back();
            chunk_generated_tokens.pop_back();
            ctx_omni->tts_all_generated_tokens.pop_back();
        }

        const int push_threshold = first_chunk_pushed ? OMNI_TTS_STREAM_CHUNK_SIZE : OMNI_TTS_FIRST_STREAM_CHUNK_SIZE;
        if ((int) stream_buffer.size() >= push_threshold && ctx_omni->t2w_thread_info && !is_end_of_turn) {
            first_chunk_pushed = true;
            T2WOut * t2w_out   = new T2WOut();
            t2w_out->audio_tokens.assign(stream_buffer.begin(), stream_buffer.end());
            t2w_out->is_final         = false;
            t2w_out->is_chunk_end     = false;
            t2w_out->round_meta       = round_meta;
            t2w_out->duplex_chunk_idx = duplex_chunk_idx;

            {
                std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                ctx_omni->t2w_thread_info->queue.push(t2w_out);
            }
            ctx_omni->t2w_thread_info->cv.notify_one();
            stream_buffer.clear();
        }

        if (is_eos) {
            break;
        }
    }

    if (ctx_omni->t2w_thread_info) {
        T2WOut * t2w_out = new T2WOut();
        t2w_out->audio_tokens.assign(stream_buffer.begin(), stream_buffer.end());
        t2w_out->is_final         = is_end_of_turn;
        t2w_out->is_chunk_end     = !is_end_of_turn;
        t2w_out->round_meta       = round_meta;
        t2w_out->duplex_chunk_idx = duplex_chunk_idx;

        {
            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
            ctx_omni->t2w_thread_info->queue.push(t2w_out);
        }
        ctx_omni->t2w_thread_info->cv.notify_one();
    }

    common_sampler_free(tts_sampler);
    ctx_omni->tts_n_past_accumulated = n_past_tts;

    if (!output_dir.empty() && !output_audio_tokens.empty()) {
        const std::string tokens_file = output_dir + "/audio_tokens_chunk_" + std::to_string(chunk_idx) + ".bin";
        FILE *            f           = fopen(tokens_file.c_str(), "wb");
        if (f != nullptr) {
            fwrite(output_audio_tokens.data(), sizeof(int32_t), output_audio_tokens.size(), f);
            fclose(f);
        }
    }

    return finish_tts_stage(!output_audio_tokens.empty());
}

// ==============================================================================
// TTS Worker Main Loops
// ==============================================================================

void omni_tts_worker_loop_duplex(struct omni_context * ctx_omni, struct common_params * params) {
    // TTS model state
    int                      tts_n_past = 0;
    std::vector<llama_token> audio_tokens;
    std::vector<llama_token> all_audio_tokens;
    std::string              debug_dir  = "";
    bool                     tts_finish = false;
    bool                     llm_finish = false;
    int                      chunk_idx  = 0;
    std::string              incomplete_bytes;

    // 双工模式：固定输出目录（不使用 round_XXX 子目录）
    const std::string & base_output_dir      = ctx_omni->base_output_dir;
    const std::string   tts_output_dir       = base_output_dir + "/tts_txt";
    const std::string   llm_debug_output_dir = base_output_dir + "/llm_debug";
    const std::string   tts_wav_output_dir   = base_output_dir + "/tts_wav";
    OmniRoundMeta       active_round_meta    = omni_session_round_meta(ctx_omni);

    auto create_dir = [](const std::string & dir_path) {
        if (!cross_platform_mkdir_p(dir_path)) {
            LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    };

    create_dir(tts_output_dir);
    create_dir(llm_debug_output_dir);
    create_dir(tts_wav_output_dir);

    // 标志位：当前 turn 是否已经执行过 turn_eos flush
    bool turn_eos_flushed = false;

    print_with_timestamp("TTS thread (duplex mode) started\n");

    // Multi Round Persistent Loop
    while (ctx_omni->workers.tts_thread_running) {
        if (!ctx_omni->workers.tts_thread_running) {
            break;
        }

        // 双工模式打断检测
        if (ctx_omni->gate.break_event.load()) {
            omni_clear_tts_queue(ctx_omni);
            llm_finish = false;
            tts_finish = false;
            chunk_idx  = 0;
            tts_n_past = 0;
            audio_tokens.clear();
            all_audio_tokens.clear();
            incomplete_bytes.clear();
            llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
            if (mem) {
                llama_memory_seq_rm(mem, 0, 0, -1);
                print_with_timestamp("TTS Duplex: break_event - cleared TTS KV cache\n");
            }
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            continue;
        }

        std::string              llm_text = "";
        std::vector<llama_token> current_chunk_token_ids;
        std::vector<float>       current_chunk_hidden_states;
        int                      current_chunk_n_embd   = 0;
        int                      current_chunk_perf_idx = -1;

        bool accumulated_is_end_of_turn = false;

        // Wait for queue
        if (!llm_finish || (llm_finish && llm_text.empty())) {
            std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
            auto &                       queue = ctx_omni->tts_thread_info->queue;
            ctx_omni->tts_thread_info->cv.wait(lock, [&] {
                return !queue.empty() || !ctx_omni->workers.tts_thread_running || ctx_omni->gate.break_event.load();
            });

            if (ctx_omni->gate.break_event.load()) {
                lock.unlock();
                continue;
            }

            if (!ctx_omni->workers.tts_thread_running) {
                break;
            }

            // 清空 current_chunk 数据
            current_chunk_token_ids.clear();
            current_chunk_hidden_states.clear();
            current_chunk_n_embd       = 0;
            accumulated_is_end_of_turn = false;
            current_chunk_perf_idx     = -1;

            // 累积所有队列中的数据
            while (!queue.empty()) {
                LLMOut * llm_out = queue.front();
                if (ctx_omni->duplex_mode && current_chunk_perf_idx >= 0 && llm_out->duplex_chunk_idx >= 0 &&
                    llm_out->duplex_chunk_idx != current_chunk_perf_idx) {
                    break;
                }
                if (current_chunk_perf_idx < 0 && llm_out->duplex_chunk_idx >= 0) {
                    current_chunk_perf_idx = llm_out->duplex_chunk_idx;
                }
                queue.pop();
                llm_finish |= llm_out->llm_finish;
                accumulated_is_end_of_turn |= llm_out->is_end_of_turn;
                active_round_meta = llm_out->round_meta;

                if (!ctx_omni->gate.speech_ready || ctx_omni->duplex_mode) {
                    llm_text += llm_out->text;
                    debug_dir = llm_out->debug_dir;
                }

                if (!llm_out->token_ids.empty() && !llm_out->hidden_states.empty()) {
                    current_chunk_token_ids.insert(current_chunk_token_ids.end(), llm_out->token_ids.begin(),
                                                   llm_out->token_ids.end());
                    current_chunk_hidden_states.insert(current_chunk_hidden_states.end(),
                                                       llm_out->hidden_states.begin(), llm_out->hidden_states.end());
                    current_chunk_n_embd = llm_out->n_embd;
                }
                delete llm_out;
            }
            lock.unlock();
            ctx_omni->tts_thread_info->cv.notify_all();

            print_with_timestamp(
                "TTS Duplex: after queue - speek_done=%d, llm_finish=%d, token_ids.size=%zu, is_end_of_turn=%d, "
                "llm_text.len=%zu\n",
                ctx_omni->gate.speech_ready ? 1 : 0, llm_finish ? 1 : 0, current_chunk_token_ids.size(),
                accumulated_is_end_of_turn ? 1 : 0, llm_text.size());

            if (ctx_omni->gate.speech_ready && llm_finish) {
                if (ctx_omni->duplex_mode && !current_chunk_token_ids.empty()) {
                    ctx_omni->gate.speech_ready = false;
                } else if (ctx_omni->duplex_mode && accumulated_is_end_of_turn) {
                    ctx_omni->gate.speech_ready = false;
                    print_with_timestamp("TTS Duplex: is_end_of_turn=true, will call TTS to flush buffer\n");
                } else {
                    // LISTEN/CHUNK_EOS 且没有实际文本
                    llm_finish = false;
                    llm_text.clear();

                    if (ctx_omni->t2w_thread_info) {
                        T2WOut * t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final         = false;
                        t2w_out->is_chunk_end     = true;
                        t2w_out->round_meta       = active_round_meta;
                        t2w_out->duplex_chunk_idx = current_chunk_perf_idx;
                        {
                            std::lock_guard<std::mutex> lock2(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    duplex_timing_mark_tts_done(ctx_omni, current_chunk_perf_idx);
                    continue;
                }
            }
        }

        std::string & response = llm_text;
        // 处理不完整的 UTF-8 字节
        if (!incomplete_bytes.empty()) {
            response = incomplete_bytes + response;
            incomplete_bytes.clear();
        }
        size_t incomplete_len = findIncompleteUtf8(response);
        if (incomplete_len > 0) {
            incomplete_bytes = response.substr(response.size() - incomplete_len, incomplete_len);
            response         = response.substr(0, response.size() - incomplete_len);
        }

        // Skip empty responses
        if (response.empty() && !llm_finish) {
            if (ctx_omni->gate.speech_ready) {
                llm_finish = false;
                if (ctx_omni->duplex_mode && !accumulated_is_end_of_turn) {
                    // 保持状态
                } else {
                    chunk_idx  = 0;
                    tts_n_past = 0;
                    audio_tokens.clear();
                    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                    if (mem) {
                        llama_memory_seq_rm(mem, 0, 0, -1);
                    }
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                }
            }
            continue;
        }

        // Tokenize text input
        std::vector<llama_token> text_tokens = common_tokenize(ctx_omni->ctx_tts_llama, response, false, true);

        if (text_tokens.empty() && !llm_finish) {
            continue;
        }

        // Handle empty final chunk
        if (text_tokens.empty() && response.empty() && llm_finish) {
            ctx_omni->gate.speech_ready  = true;
            ctx_omni->warmup_done = true;
            ctx_omni->workers.speek_cv.notify_all();

            if (ctx_omni->duplex_mode && !accumulated_is_end_of_turn) {
                // LISTEN/CHUNK_EOS
                if (ctx_omni->t2w_thread_info) {
                    T2WOut * t2w_out = new T2WOut();
                    t2w_out->audio_tokens.clear();
                    t2w_out->is_final         = false;
                    t2w_out->is_chunk_end     = true;
                    t2w_out->round_meta       = active_round_meta;
                    t2w_out->duplex_chunk_idx = current_chunk_perf_idx;
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                }
                duplex_timing_mark_tts_done(ctx_omni, current_chunk_perf_idx);
                tts_finish = false;
                llm_finish = false;
                continue;
            }

            if (ctx_omni->duplex_mode && accumulated_is_end_of_turn) {
                print_with_timestamp(
                    "TTS Duplex: empty final chunk but is_end_of_turn=true, will call TTS to flush buffer\n");
            } else {
                // 非双工模式
                if (ctx_omni->t2w_thread_info) {
                    T2WOut * t2w_out = new T2WOut();
                    t2w_out->audio_tokens.clear();
                    t2w_out->is_final     = true;
                    t2w_out->is_chunk_end = false;
                    t2w_out->round_meta   = active_round_meta;
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                }
                tts_finish = false;
                llm_finish = false;
                chunk_idx  = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                if (mem && !ctx_omni->duplex_mode) {
                    llama_memory_seq_rm(mem, 0, 0, -1);
                }
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                continue;
            }
        }

        // Check for LLM data
        bool has_llm_data =
            (!current_chunk_token_ids.empty() && !current_chunk_hidden_states.empty() && current_chunk_n_embd > 0);

        if (has_llm_data) {
            int current_chunk_idx = chunk_idx;

            if (turn_eos_flushed) {
                turn_eos_flushed = false;
            }

            print_with_timestamp("TTS Duplex: processing chunk_idx=%d, n_tokens=%zu, is_end_of_turn=%d\n", chunk_idx,
                                 current_chunk_token_ids.size(), accumulated_is_end_of_turn ? 1 : 0);

            OmniTtsPreparedChunk prepared_chunk;
            if (!omni_tts_prepare_duplex_chunk(ctx_omni, llm_debug_output_dir, current_chunk_idx, llm_text,
                                               current_chunk_token_ids, current_chunk_hidden_states,
                                               current_chunk_n_embd, prepared_chunk)) {
                duplex_timing_mark_tts_done(ctx_omni, current_chunk_perf_idx);
                continue;
            }

            int                  n_tokens_filtered = prepared_chunk.n_tokens_filtered;
            const int            tts_n_embd        = prepared_chunk.tts_n_embd;
            std::vector<float> & merged_embeddings = prepared_chunk.merged_embeddings;
            const bool           merged_success    = prepared_chunk.merged_success;

            bool should_call_tts =
                (merged_success && !merged_embeddings.empty()) || (accumulated_is_end_of_turn && ctx_omni->duplex_mode);

            if (should_call_tts) {
                std::vector<int32_t> audio_tokens_out;
                bool                 is_end_of_turn = accumulated_is_end_of_turn;

                if (merged_embeddings.empty() && is_end_of_turn) {
                    print_with_timestamp(
                        "TTS Duplex: is_end_of_turn=true with empty embeddings, calling TTS to flush\n");
                    n_tokens_filtered = 0;
                }

                bool tts_gen_success = omni_tts_generate_audio_tokens_local(
                    ctx_omni, params, merged_embeddings, n_tokens_filtered, tts_n_embd, current_chunk_idx,
                    current_chunk_perf_idx, audio_tokens_out, is_end_of_turn, active_round_meta, tts_wav_output_dir);

                if (tts_gen_success) {
                    all_audio_tokens.insert(all_audio_tokens.end(), audio_tokens_out.begin(), audio_tokens_out.end());
                    if (is_end_of_turn && ctx_omni->duplex_mode) {
                        turn_eos_flushed = true;
                    }
                }
            } else {
                duplex_timing_mark_tts_done(ctx_omni, current_chunk_perf_idx);
            }

            ++chunk_idx;
            llm_text.clear();
            response.clear();

            // Handle final chunk
            if (llm_finish) {
                tts_finish            = true;
                ctx_omni->gate.speech_ready  = true;
                ctx_omni->warmup_done = true;
                ctx_omni->workers.speek_cv.notify_all();

                if (ctx_omni->duplex_mode && !accumulated_is_end_of_turn) {
                    // LISTEN/CHUNK_EOS: 保持 TTS 状态
                    if (ctx_omni->t2w_thread_info) {
                        T2WOut * t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final         = false;
                        t2w_out->is_chunk_end     = true;
                        t2w_out->round_meta       = active_round_meta;
                        t2w_out->duplex_chunk_idx = current_chunk_perf_idx;
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    llm_finish = false;
                    tts_finish = false;
                } else {
                    // 真正的轮次结束
                    if (ctx_omni->t2w_thread_info && !turn_eos_flushed) {
                        T2WOut * t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final         = true;
                        t2w_out->round_meta       = active_round_meta;
                        t2w_out->duplex_chunk_idx = current_chunk_perf_idx;
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                    if (mem) {
                        llama_memory_seq_rm(mem, 0, 0, -1);
                    }
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                    tts_n_past                    = 0;
                    audio_tokens.clear();
                    all_audio_tokens.clear();
                    llm_finish = false;
                    tts_finish = false;
                }
            }
            continue;
        }
        if (ctx_omni->duplex_mode && accumulated_is_end_of_turn && llm_finish) {
            // turn 结束但没有新数据，调用 TTS flush buffer
            if (turn_eos_flushed) {
                print_with_timestamp("TTS Duplex: turn_eos already flushed, skipping TTS generation\n");
            } else {
                print_with_timestamp("TTS Duplex: no LLM data but is_end_of_turn=true, calling TTS to flush buffer\n");

                const int            tts_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
                std::vector<float>   empty_embeddings;
                std::vector<int32_t> audio_tokens_out;
                int                  n_tokens_for_tts  = 0;
                int                  current_chunk_idx = chunk_idx;

                bool tts_gen_success = omni_tts_generate_audio_tokens_local(
                    ctx_omni, params, empty_embeddings, n_tokens_for_tts, tts_n_embd, current_chunk_idx,
                    current_chunk_perf_idx, audio_tokens_out, true, active_round_meta, tts_wav_output_dir);

                if (tts_gen_success) {
                    all_audio_tokens.insert(all_audio_tokens.end(), audio_tokens_out.begin(), audio_tokens_out.end());
                } else {
                    if (ctx_omni->t2w_thread_info) {
                        T2WOut * t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final         = true;
                        t2w_out->is_chunk_end     = false;
                        t2w_out->round_meta       = active_round_meta;
                        t2w_out->duplex_chunk_idx = current_chunk_perf_idx;
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                }

                turn_eos_flushed = true;
            }

            llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
            if (mem) {
                llama_memory_seq_rm(mem, 0, 0, -1);
            }
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            tts_n_past                    = 0;
            audio_tokens.clear();
            all_audio_tokens.clear();
            llm_finish            = false;
            tts_finish            = false;
            ctx_omni->gate.speech_ready  = true;
            ctx_omni->warmup_done = true;
            ctx_omni->workers.speek_cv.notify_all();

            llm_text.clear();
            response.clear();
            continue;
        }
        llm_text.clear();
        response.clear();
    }

    print_with_timestamp("TTS thread (duplex mode) stopped\n");
}

void omni_tts_worker_loop_simplex(struct omni_context * ctx_omni, struct common_params * params) {
    // TTS model state
    int                      tts_n_past = 0;
    std::vector<llama_token> audio_tokens;
    std::vector<llama_token> all_audio_tokens;
    std::string              debug_dir  = "";
    bool                     tts_finish = false;
    bool                     llm_finish = false;
    int                      chunk_idx  = 0;
    std::string              incomplete_bytes;

    const std::string & base_output_dir   = ctx_omni->base_output_dir;
    OmniRoundMeta       active_round_meta = omni_session_round_meta(ctx_omni);

    // Helper function to create directory
    auto create_dir = [](const std::string & dir_path) {
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0) {
            if (!cross_platform_mkdir_p(dir_path)) {
                LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
                return false;
            }
            return true;
        }
        if (!(info.st_mode & S_IFDIR)) {
            LOG_ERR("Output path exists but is not a directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    };

    auto get_round_output_dir = [&](const OmniRoundMeta & round_meta) -> std::string {
        return omni_round_output_dir(base_output_dir, round_meta);
    };

    std::string current_round_dir    = get_round_output_dir(active_round_meta);
    std::string tts_output_dir       = current_round_dir + "/tts_txt";
    std::string llm_debug_output_dir = current_round_dir + "/llm_debug";
    std::string tts_wav_output_dir   = current_round_dir + "/tts_wav";

    int last_created_round_idx = -1;

    auto update_output_dirs = [&]() {
        current_round_dir    = get_round_output_dir(active_round_meta);
        tts_output_dir       = current_round_dir + "/tts_txt";
        llm_debug_output_dir = current_round_dir + "/llm_debug";
        tts_wav_output_dir   = current_round_dir + "/tts_wav";

        if (active_round_meta.round_idx != last_created_round_idx || active_round_meta.duplex_mode) {
            create_dir(tts_output_dir);
            create_dir(llm_debug_output_dir);
            create_dir(tts_wav_output_dir);
            last_created_round_idx = active_round_meta.round_idx;

            if (!active_round_meta.duplex_mode) {
                print_with_timestamp("TTS: 创建单工模式输出目录: %s\n", current_round_dir.c_str());
            }
        }
    };

    update_output_dirs();

    auto create_wav_timing_file = [&]() {
        std::string wav_timing_file = tts_wav_output_dir + "/wav_timing.txt";
        FILE *      f_timing        = fopen(wav_timing_file.c_str(), "w");
        if (f_timing) {
            fprintf(f_timing, "# WAV file generation timing log\n");
            fprintf(f_timing,
                    "# Format: chunk_index, elapsed_time_ms (since stream_decode start), file_size_bytes, "
                    "request_duration_ms\n");
            fprintf(f_timing, "# Time 0 is when stream_decode() function starts\n");
            fclose(f_timing);
        }
    };
    create_wav_timing_file();

    print_with_timestamp("TTS thread started\n");

    // Multi Round Persistent Loop
    while (ctx_omni->workers.tts_thread_running) {
        if (!ctx_omni->workers.tts_thread_running) {
            break;
        }

        // 打断检测：清空队列、重置状态
        if (ctx_omni->gate.break_event.load()) {
            omni_clear_tts_queue(ctx_omni);
            llm_finish = false;
            tts_finish = false;
            chunk_idx  = 0;
            tts_n_past = 0;
            audio_tokens.clear();
            all_audio_tokens.clear();
            incomplete_bytes.clear();
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            continue;
        }

        std::string              llm_text = "";
        std::vector<llama_token> current_chunk_token_ids;
        std::vector<float>       current_chunk_hidden_states;
        int                      current_chunk_n_embd = 0;

        // Always wait for queue if not finished, or if finished but need to reset state
        if (!llm_finish || (llm_finish && llm_text.empty())) {
            std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
            auto &                       queue = ctx_omni->tts_thread_info->queue;
            ctx_omni->tts_thread_info->cv.wait(lock, [&] {
                return !queue.empty() || !ctx_omni->workers.tts_thread_running || ctx_omni->gate.break_event.load();
            });

            if (ctx_omni->gate.break_event.load()) {
                lock.unlock();
                continue;
            }

            if (!ctx_omni->workers.tts_thread_running) {
                break;
            }

            // 每次只处理一个 chunk（与 Python 对齐）
            if (!queue.empty()) {
                LLMOut * llm_out = queue.front();
                llm_finish |= llm_out->llm_finish;
                active_round_meta = llm_out->round_meta;
                if (!ctx_omni->gate.speech_ready || ctx_omni->duplex_mode) {
                    llm_text  = llm_out->text;
                    debug_dir = llm_out->debug_dir;
                }
                if (!llm_out->token_ids.empty() && !llm_out->hidden_states.empty()) {
                    current_chunk_token_ids     = llm_out->token_ids;
                    current_chunk_hidden_states = llm_out->hidden_states;
                    current_chunk_n_embd        = llm_out->n_embd;

                    std::string token_ids_str = "";
                    for (size_t i = 0; i < current_chunk_token_ids.size() && i < 20; i++) {
                        token_ids_str += std::to_string(current_chunk_token_ids[i]);
                        if (i < current_chunk_token_ids.size() - 1 && i < 19) {
                            token_ids_str += " ";
                        }
                    }
                    if (current_chunk_token_ids.size() > 20) {
                        token_ids_str += "...";
                    }

                    print_with_timestamp(
                        "TTS<-LLM: chunk_idx=%d, text='%s', n_tokens=%zu, hidden_size=%zu, token_ids=[%s]\n", chunk_idx,
                        llm_text.c_str(), current_chunk_token_ids.size(), current_chunk_hidden_states.size(),
                        token_ids_str.c_str());
                }
                delete llm_out;
                queue.pop();
            }
            lock.unlock();
            ctx_omni->tts_thread_info->cv.notify_all();
            update_output_dirs();

            print_with_timestamp(
                "TTS: after queue pop - speek_done=%d, llm_finish=%d, llm_text.empty=%d, token_ids.size=%zu\n",
                ctx_omni->gate.speech_ready, llm_finish, llm_text.empty(), current_chunk_token_ids.size());

            if (ctx_omni->gate.speech_ready && llm_finish) {
                print_with_timestamp("TTS: speek_done=true and llm_finish=true, resetting state for next round\n");
                if (ctx_omni->duplex_mode && !current_chunk_token_ids.empty()) {
                    ctx_omni->gate.speech_ready = false;
                } else {
                    llm_finish = false;
                    llm_text.clear();
                    chunk_idx  = 0;
                    tts_n_past = 0;
                    audio_tokens.clear();
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                    continue;
                }
            }
        }

        std::string & response = llm_text;
        if (!incomplete_bytes.empty()) {
            print_with_timestamp("TTS: prepending incomplete_bytes (len=%zu) to response (len=%zu)\n",
                                 incomplete_bytes.length(), response.length());
            response = incomplete_bytes + response;
            incomplete_bytes.clear();
        }
        size_t incomplete_len = findIncompleteUtf8(response);
        if (incomplete_len > 0) {
            print_with_timestamp(
                "TTS: detected incomplete UTF-8 sequence at end: incomplete_len=%zu, response_len=%zu\n",
                incomplete_len, response.length());
            incomplete_bytes = response.substr(response.size() - incomplete_len, incomplete_len);
            response         = response.substr(0, response.size() - incomplete_len);
            print_with_timestamp("TTS: after truncation: response_len=%zu, incomplete_bytes_len=%zu\n",
                                 response.length(), incomplete_bytes.length());
        } else {
            incomplete_bytes.clear();
        }

        print_with_timestamp("TTS: before empty check - response.empty=%d, response='%s', llm_finish=%d\n",
                             response.empty(), response.substr(0, 50).c_str(), llm_finish);

        if (response.empty() && !llm_finish) {
            if (ctx_omni->gate.speech_ready) {
                print_with_timestamp(
                    "TTS: speek_done=true with empty response, keeping state (waiting for stream_prefill)\n");
                llm_finish = false;
                chunk_idx  = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
            }
            continue;
        }
        fflush(stdout);

        if (response.empty() && llm_finish) {
            print_with_timestamp("TTS: received llm_finish=true with no data, finalizing (tts_token_buffer=%zu)\n",
                                 ctx_omni->tts_token_buffer.size());

            if (ctx_omni->t2w_thread_info && !ctx_omni->tts_token_buffer.empty()) {
                T2WOut * t2w_out = new T2WOut();
                t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(), ctx_omni->tts_token_buffer.end());
                t2w_out->is_final   = false;
                t2w_out->round_meta = active_round_meta;
                {
                    std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    ctx_omni->t2w_thread_info->queue.push(t2w_out);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
                print_with_timestamp("TTS: flushed %zu remaining tokens from tts_token_buffer\n",
                                     ctx_omni->tts_token_buffer.size());
                ctx_omni->tts_token_buffer.clear();
            }

            if (ctx_omni->t2w_thread_info) {
                const OmniRoundMeta completed_round_meta = active_round_meta;
                T2WOut *            t2w_final            = new T2WOut();
                t2w_final->audio_tokens.clear();
                t2w_final->is_final   = true;
                t2w_final->round_meta = completed_round_meta;
                {
                    std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    ctx_omni->t2w_thread_info->queue.push(t2w_final);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
                print_with_timestamp("TTS: sent is_final=true to T2W (llm_finish with no data)\n");
            }

            ctx_omni->gate.speech_ready  = true;
            ctx_omni->warmup_done = true;
            ctx_omni->workers.speek_cv.notify_all();
            print_with_timestamp("TTS: finished processing all chunks (llm_finish with no data path)\n");
            tts_finish = false;
            llm_finish = false;
            chunk_idx  = 0;
            tts_n_past = 0;
            audio_tokens.clear();
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            continue;
        }

        fflush(stdout);
        print_with_timestamp("TTS: DEBUG before has_llm_data - token_ids.size=%zu, hidden_states.size=%zu, n_embd=%d\n",
                             current_chunk_token_ids.size(), current_chunk_hidden_states.size(), current_chunk_n_embd);

        bool has_llm_data =
            (!current_chunk_token_ids.empty() && !current_chunk_hidden_states.empty() && current_chunk_n_embd > 0);

        print_with_timestamp("TTS: has_llm_data=%d, token_ids=%zu, hidden_states=%zu, n_embd=%d, llm_finish=%d\n",
                             has_llm_data, current_chunk_token_ids.size(), current_chunk_hidden_states.size(),
                             current_chunk_n_embd, llm_finish);

        if (has_llm_data) {
            if (chunk_idx == 0 && !ctx_omni->duplex_mode) {
                update_output_dirs();
                create_wav_timing_file();
            }

            int current_chunk_idx = chunk_idx;

            OmniTtsPreparedChunk prepared_chunk;
            if (!omni_tts_prepare_simplex_chunk(ctx_omni, llm_debug_output_dir, current_chunk_idx, response,
                                                current_chunk_token_ids, current_chunk_hidden_states,
                                                current_chunk_n_embd, prepared_chunk)) {
                continue;
            }

            int                  n_tokens_filtered = prepared_chunk.n_tokens_filtered;
            const int            tts_n_embd        = prepared_chunk.tts_n_embd;
            std::vector<float> & merged_embeddings = prepared_chunk.merged_embeddings;
            const bool           merged_success    = prepared_chunk.merged_success;

            bool is_final_text_chunk = llm_finish;

            if (merged_success && !merged_embeddings.empty()) {
                std::vector<int32_t> audio_tokens_chunk;
                bool                 tts_gen_success = omni_tts_generate_audio_tokens_local_simplex(
                    ctx_omni, params, merged_embeddings, n_tokens_filtered, tts_n_embd, current_chunk_idx,
                    audio_tokens_chunk, active_round_meta, tts_wav_output_dir, is_final_text_chunk);
                if (tts_gen_success) {
                    std::string tokens_txt_file =
                        tts_wav_output_dir + "/audio_tokens_chunk_" + std::to_string(current_chunk_idx) + ".txt";
                    FILE * f_tokens = fopen(tokens_txt_file.c_str(), "w");
                    if (f_tokens) {
                        for (size_t i = 0; i < audio_tokens_chunk.size(); ++i) {
                            fprintf(f_tokens, "%d", audio_tokens_chunk[i]);
                            if (i < audio_tokens_chunk.size() - 1) {
                                fprintf(f_tokens, ",");
                            }
                        }
                        fprintf(f_tokens, "\n");
                        fclose(f_tokens);
                    }
                    all_audio_tokens.insert(all_audio_tokens.end(), audio_tokens_chunk.begin(),
                                            audio_tokens_chunk.end());
                } else {
                    LOG_ERR("TTS Local: failed for chunk %d\n", current_chunk_idx);
                }
            }

            ++chunk_idx;
            llm_text.clear();
            response.clear();

            if (llm_finish) {
                if (!ctx_omni->tts_token_buffer.empty() && ctx_omni->t2w_thread_info) {
                    print_with_timestamp("TTS: llm_finish=true, flushing remaining %zu tokens from tts_token_buffer\n",
                                         ctx_omni->tts_token_buffer.size());
                    T2WOut * t2w_out = new T2WOut();
                    if (t2w_out) {
                        t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(),
                                                     ctx_omni->tts_token_buffer.end());
                        t2w_out->is_final   = false;
                        t2w_out->round_meta = active_round_meta;
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    ctx_omni->tts_token_buffer.clear();
                }

                tts_finish            = true;
                ctx_omni->gate.speech_ready  = true;
                ctx_omni->warmup_done = true;
                ctx_omni->workers.speek_cv.notify_all();
                print_with_timestamp("TTS: finished processing all chunks\n");

                const OmniRoundMeta completed_round_meta = active_round_meta;

                if (ctx_omni->t2w_thread_info) {
                    T2WOut * t2w_out = new T2WOut();
                    t2w_out->audio_tokens.clear();
                    t2w_out->is_final   = true;
                    t2w_out->round_meta = completed_round_meta;
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                    print_with_timestamp("TTS: sent is_final=true to T2W queue (turn end)\n");
                }
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                if (ctx_omni->duplex_mode) {
                    ctx_omni->tts_condition_saved = false;
                }

                chunk_idx  = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                all_audio_tokens.clear();
                llm_finish = false;
                tts_finish = false;
            }

            continue;
        }
        llm_text.clear();
        response.clear();
    }

    print_with_timestamp("TTS thread stopped\n");
}
