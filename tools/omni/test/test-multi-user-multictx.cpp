// Shared-model multi-user LLM policy probe.
//
// The probe submits token, embedding, or media tasks directly to the internal
// runtime. It does not start the legacy duplex/TTS global queues and does not
// fork one process per user.

#include "multi_user_multictx_runtime.h"
#include "duplex_core.h"
#include "token2wav/token2wav.h"
#include "../../../src/llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::string model;
    bool help = false;
    uint32_t users = 2;
    uint32_t frames = 2;
    uint32_t max_new_tokens = 26;
    uint32_t interval_ms = 0;
    uint32_t user_stagger_ms = 0;
    int32_t n_ctx = 2048;
    int32_t n_batch = 256;
    int32_t n_ubatch = 0;
    int32_t n_gpu_layers = -1;
    std::string vpm_model;
    std::string apm_model;
    std::string tts_model;
    std::string t2w_dir;
    std::string image;
    std::string audio;
    std::string input_dir;
    std::string input_dirs;
    std::string arrival_trace;
    std::string ref_audio;
    bool tts = false;
    bool t2w = false;
    OmniLlmExecutionPolicy llm_policy = OmniLlmExecutionPolicy::PER_USER_CONTEXT;
    uint32_t llm_max_batch_size = 0;
    uint32_t llm_max_batch_delay_us = 1000;
    uint32_t llm_sliding_high_tokens = 0;
    uint32_t llm_sliding_low_tokens = 0;
    uint32_t embedding_tokens = 0;
    uint32_t embedding_token_step = 0;
    uint32_t expect_prefill_batch_min = 0;
    uint64_t expect_prefill_fallback_min = 0;
    uint64_t expect_mixed_min = 0;
    uint64_t expect_sliding_min = 0;
    bool verify_token_embeddings = false;
    bool token_replay = false;
    bool decode_token_replay = false;
    uint32_t force_listen_count = 3;
    uint32_t cancel_user = 0;
    uint32_t cancel_after_ms = 0;
};

void usage(const char * name) {
    std::fprintf(stderr,
        "usage: %s -m MODEL [--users N] [--frames N] [--max-new-tokens N] [--interval-ms N] "
        "[--ctx-size N] [--batch N] [--ubatch N] [-ngl N] [--user-stagger-ms N] [--vpm-model PATH --image PATH] "
        "[--apm-model PATH --audio PATH] [--input-dir DIR | --input-dirs DIR1,DIR2,...] "
        "[--arrival-trace CSV] [--tts-model PATH --tts] "
        "[--t2w-dir DIR --t2w] [--ref-audio PATH] "
        "[--llm-policy per-user|continuous] [--llm-max-batch-size N] "
        "[--llm-max-batch-delay-us N] [--llm-sliding-high-tokens N] "
        "[--llm-sliding-low-tokens N] [--embedding-tokens N] [--embedding-token-step N] "
        "[--expect-prefill-batch-min N] [--expect-prefill-fallback-min N] "
        "[--expect-mixed-min N] [--expect-sliding-min N] "
        "[--verify-token-embeddings] [--token-replay] [--decode-token-replay] "
        "[--force-listen-count N] "
        "[--cancel-user N --cancel-after-ms N]\n", name);
}

bool next_arg(int argc, char ** argv, int & i, const char * name, std::string & value) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        return false;
    }
    value = argv[++i];
    return true;
}

bool parse_args(int argc, char ** argv, Options & options) {
    for (int i = 1; i < argc; ++i) {
        std::string value;
        if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            usage(argv[0]);
            options.help = true;
            return false;
        } else if (!std::strcmp(argv[i], "-m")) {
            if (!next_arg(argc, argv, i, "-m", options.model)) return false;
        } else if (!std::strcmp(argv[i], "--users")) {
            if (!next_arg(argc, argv, i, "--users", value)) return false;
            options.users = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--frames")) {
            if (!next_arg(argc, argv, i, "--frames", value)) return false;
            options.frames = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--max-new-tokens")) {
            if (!next_arg(argc, argv, i, "--max-new-tokens", value)) return false;
            options.max_new_tokens = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--interval-ms")) {
            if (!next_arg(argc, argv, i, "--interval-ms", value)) return false;
            options.interval_ms = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--user-stagger-ms")) {
            if (!next_arg(argc, argv, i, "--user-stagger-ms", value)) return false;
            options.user_stagger_ms = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--ctx-size")) {
            if (!next_arg(argc, argv, i, "--ctx-size", value)) return false;
            options.n_ctx = std::atoi(value.c_str());
        } else if (!std::strcmp(argv[i], "--batch")) {
            if (!next_arg(argc, argv, i, "--batch", value)) return false;
            options.n_batch = std::atoi(value.c_str());
        } else if (!std::strcmp(argv[i], "--ubatch")) {
            if (!next_arg(argc, argv, i, "--ubatch", value)) return false;
            options.n_ubatch = std::atoi(value.c_str());
        } else if (!std::strcmp(argv[i], "-ngl")) {
            if (!next_arg(argc, argv, i, "-ngl", value)) return false;
            options.n_gpu_layers = std::atoi(value.c_str());
        } else if (!std::strcmp(argv[i], "--vpm-model")) {
            if (!next_arg(argc, argv, i, "--vpm-model", options.vpm_model)) return false;
        } else if (!std::strcmp(argv[i], "--apm-model")) {
            if (!next_arg(argc, argv, i, "--apm-model", options.apm_model)) return false;
        } else if (!std::strcmp(argv[i], "--tts-model")) {
            if (!next_arg(argc, argv, i, "--tts-model", options.tts_model)) return false;
        } else if (!std::strcmp(argv[i], "--t2w-dir")) {
            if (!next_arg(argc, argv, i, "--t2w-dir", options.t2w_dir)) return false;
        } else if (!std::strcmp(argv[i], "--image")) {
            if (!next_arg(argc, argv, i, "--image", options.image)) return false;
        } else if (!std::strcmp(argv[i], "--audio")) {
            if (!next_arg(argc, argv, i, "--audio", options.audio)) return false;
        } else if (!std::strcmp(argv[i], "--input-dir")) {
            if (!next_arg(argc, argv, i, "--input-dir", options.input_dir)) return false;
        } else if (!std::strcmp(argv[i], "--input-dirs")) {
            if (!next_arg(argc, argv, i, "--input-dirs", options.input_dirs)) return false;
        } else if (!std::strcmp(argv[i], "--arrival-trace")) {
            if (!next_arg(argc, argv, i, "--arrival-trace", options.arrival_trace)) return false;
        } else if (!std::strcmp(argv[i], "--ref-audio")) {
            if (!next_arg(argc, argv, i, "--ref-audio", options.ref_audio)) return false;
        } else if (!std::strcmp(argv[i], "--tts")) {
            options.tts = true;
        } else if (!std::strcmp(argv[i], "--t2w")) {
            options.t2w = true;
        } else if (!std::strcmp(argv[i], "--llm-policy")) {
            if (!next_arg(argc, argv, i, "--llm-policy", value)) return false;
            if (value == "per-user") {
                options.llm_policy = OmniLlmExecutionPolicy::PER_USER_CONTEXT;
            } else if (value == "continuous") {
                options.llm_policy = OmniLlmExecutionPolicy::CONTINUOUS_BATCH;
            } else {
                std::fprintf(stderr, "invalid --llm-policy: %s\n", value.c_str());
                return false;
            }
        } else if (!std::strcmp(argv[i], "--llm-max-batch-size")) {
            if (!next_arg(argc, argv, i, "--llm-max-batch-size", value)) return false;
            options.llm_max_batch_size = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--llm-max-batch-delay-us")) {
            if (!next_arg(argc, argv, i, "--llm-max-batch-delay-us", value)) return false;
            options.llm_max_batch_delay_us = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--llm-sliding-high-tokens")) {
            if (!next_arg(argc, argv, i, "--llm-sliding-high-tokens", value)) return false;
            options.llm_sliding_high_tokens = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--llm-sliding-low-tokens")) {
            if (!next_arg(argc, argv, i, "--llm-sliding-low-tokens", value)) return false;
            options.llm_sliding_low_tokens = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--embedding-tokens")) {
            if (!next_arg(argc, argv, i, "--embedding-tokens", value)) return false;
            options.embedding_tokens = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--embedding-token-step")) {
            if (!next_arg(argc, argv, i, "--embedding-token-step", value)) return false;
            options.embedding_token_step = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--expect-prefill-batch-min")) {
            if (!next_arg(argc, argv, i, "--expect-prefill-batch-min", value)) return false;
            options.expect_prefill_batch_min = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--expect-prefill-fallback-min")) {
            if (!next_arg(argc, argv, i, "--expect-prefill-fallback-min", value)) return false;
            options.expect_prefill_fallback_min = std::strtoull(value.c_str(), nullptr, 10);
        } else if (!std::strcmp(argv[i], "--expect-mixed-min")) {
            if (!next_arg(argc, argv, i, "--expect-mixed-min", value)) return false;
            options.expect_mixed_min = std::strtoull(value.c_str(), nullptr, 10);
        } else if (!std::strcmp(argv[i], "--expect-sliding-min")) {
            if (!next_arg(argc, argv, i, "--expect-sliding-min", value)) return false;
            options.expect_sliding_min = std::strtoull(value.c_str(), nullptr, 10);
        } else if (!std::strcmp(argv[i], "--token-replay")) {
            options.token_replay = true;
        } else if (!std::strcmp(argv[i], "--decode-token-replay")) {
            options.decode_token_replay = true;
        } else if (!std::strcmp(argv[i], "--verify-token-embeddings")) {
            options.verify_token_embeddings = true;
        } else if (!std::strcmp(argv[i], "--force-listen-count")) {
            if (!next_arg(argc, argv, i, "--force-listen-count", value)) return false;
            options.force_listen_count = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--cancel-user")) {
            if (!next_arg(argc, argv, i, "--cancel-user", value)) return false;
            options.cancel_user = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (!std::strcmp(argv[i], "--cancel-after-ms")) {
            if (!next_arg(argc, argv, i, "--cancel-after-ms", value)) return false;
            options.cancel_after_ms = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return false;
        }
    }
    return !options.model.empty() && options.users > 0 && options.frames > 0 &&
           options.max_new_tokens > 0 && options.n_ctx > 0 && options.n_batch > 0 &&
           options.n_ubatch >= 0;
}

bool verify_token_embedding_path(const std::shared_ptr<OmniSharedModelBundle> & bundle,
                                 const common_params & params) {
    const std::vector<llama_token> tokens = common_tokenize(
        llama_model_get_vocab(bundle->llm_model()),
        "<unit>embedding equivalence user 17 frame 3", false, true);
    if (tokens.empty()) {
        return false;
    }

    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.n_ctx = 128;
    cparams.n_batch = std::max<uint32_t>(64, static_cast<uint32_t>(tokens.size()));
    cparams.n_ubatch = cparams.n_batch;
    cparams.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(bundle->llm_model(), cparams);
    if (!ctx || !llama_set_embeddings_masked(ctx, true)) {
        if (ctx) llama_free(ctx);
        return false;
    }

    const int32_t n_embd = llama_model_n_embd(bundle->llm_model());
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(bundle->llm_model()));
    std::vector<float> logits_token(static_cast<size_t>(n_vocab));
    std::vector<float> hidden_token(static_cast<size_t>(n_embd));
    bool ok = true;
    {
        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            llama_set_embeddings_masked(ctx, false);
            llama_free(ctx);
            return false;
        }
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[static_cast<size_t>(i)];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i + 1 == batch.n_tokens;
        }
        ok = llama_decode(ctx, batch) == 0;
        if (ok) {
            const float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
            const float * hidden = llama_get_embeddings_ith(ctx, batch.n_tokens - 1);
            ok = logits && hidden;
            if (ok) {
                std::copy(logits, logits + n_vocab, logits_token.begin());
                std::copy(hidden, hidden + n_embd, hidden_token.begin());
            }
        }
        llama_batch_free(batch);
    }

    omni_context lookup;
    lookup.ctx_llama = ctx;
    lookup.model = bundle->llm_model();
    common_params params_copy = params;
    lookup.params = &params_copy;
    std::vector<float> embeddings;
    ok = ok && duplex_init_emb_cache_for_runtime(&lookup, &params_copy) &&
         duplex_tokens_to_embeddings(&lookup, tokens, embeddings);
    llama_memory_clear(llama_get_memory(ctx), false);

    std::vector<float> logits_embedding(static_cast<size_t>(n_vocab));
    std::vector<float> hidden_embedding(static_cast<size_t>(n_embd));
    if (ok) {
        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), n_embd, 1);
        if (!batch.embd || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            llama_set_embeddings_masked(ctx, false);
            llama_free(ctx);
            return false;
        }
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        std::memcpy(batch.embd, embeddings.data(), embeddings.size() * sizeof(float));
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i + 1 == batch.n_tokens;
        }
        ok = llama_decode(ctx, batch) == 0;
        if (ok) {
            const float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
            const float * hidden = llama_get_embeddings_ith(ctx, batch.n_tokens - 1);
            ok = logits && hidden;
            if (ok) {
                std::copy(logits, logits + n_vocab, logits_embedding.begin());
                std::copy(hidden, hidden + n_embd, hidden_embedding.begin());
            }
        }
        llama_batch_free(batch);
    }
    llama_set_embeddings_masked(ctx, false);
    llama_free(ctx);
    if (!ok) {
        return false;
    }

    float logits_max_abs = 0.0f;
    float hidden_max_abs = 0.0f;
    int32_t token_argmax = 0;
    int32_t embedding_argmax = 0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        logits_max_abs = std::max(logits_max_abs,
                                  std::fabs(logits_token[i] - logits_embedding[i]));
        if (logits_token[i] > logits_token[token_argmax]) token_argmax = i;
        if (logits_embedding[i] > logits_embedding[embedding_argmax]) embedding_argmax = i;
    }
    for (int32_t i = 0; i < n_embd; ++i) {
        hidden_max_abs = std::max(hidden_max_abs,
                                  std::fabs(hidden_token[i] - hidden_embedding[i]));
    }
    std::printf("[MULTICTX EMBEDDING CHECK] tokens=%zu logits_max_abs=%.9g "
                "hidden_max_abs=%.9g argmax_token=%d argmax_embedding=%d\n",
                tokens.size(), logits_max_abs, hidden_max_abs,
                token_argmax, embedding_argmax);
    return std::isfinite(logits_max_abs) && std::isfinite(hidden_max_abs) &&
           token_argmax == embedding_argmax && logits_max_abs <= 0.05f &&
           hidden_max_abs <= 0.01f;
}

struct FrameObservation {
    bool decision = false;
    bool drained = false;
    bool cancelled = false;
    bool is_speak = false;
    uint32_t audio_events = 0;
    std::string text;
    std::vector<llama_token> tokens;
    OmniMultiUserFrameTimings timings;
};

static void print_stage_stat(const char * name, const OmniStageTimingStats & stat) {
    std::printf("%s_count=%llu %s_total_ms=%.3f %s_avg_ms=%.3f %s_min_ms=%.3f %s_max_ms=%.3f ",
                name, static_cast<unsigned long long>(stat.count), name, stat.total_ms,
                name, stat.average_ms(), name, stat.min_ms, name, stat.max_ms);
}

static double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
        std::ceil(q * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

struct ArrivalTraceRow {
    uint64_t arrival_ms = 0;
    uint32_t user_id = 0;
    uint32_t frame_id = 0;
    std::string dataset_id;
    uint32_t frame_index = 0;
};

static std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static bool parse_u64(const std::string & text, uint64_t & value) {
    if (text.empty()) {
        return false;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    value = static_cast<uint64_t>(parsed);
    return true;
}

static std::vector<std::string> split_csv(const std::string & value) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            result.push_back(std::move(item));
        }
    }
    return result;
}

static bool read_arrival_trace(const std::string & path, std::vector<ArrivalTraceRow> & rows,
                               std::string & error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open arrival trace: " + path;
        return false;
    }
    std::string line;
    size_t line_number = 0;
    uint64_t previous_arrival_ms = 0;
    bool have_row = false;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line == "arrival_ms,user_id,frame_id,dataset_id,frame_index") {
            continue;
        }
        std::stringstream fields(line);
        std::vector<std::string> columns;
        std::string column;
        while (std::getline(fields, column, ',')) {
            columns.push_back(trim(column));
        }
        if (columns.size() != 5) {
            error = "arrival trace line " + std::to_string(line_number) +
                    " must have 5 comma-separated fields";
            return false;
        }
        uint64_t arrival_ms = 0;
        uint64_t user_id = 0;
        uint64_t frame_id = 0;
        uint64_t frame_index = 0;
        if (!parse_u64(columns[0], arrival_ms) || !parse_u64(columns[1], user_id) ||
                !parse_u64(columns[2], frame_id) || !parse_u64(columns[4], frame_index) ||
                user_id == 0 || user_id > std::numeric_limits<uint32_t>::max() ||
                frame_id == 0 || frame_id > std::numeric_limits<uint32_t>::max() ||
                frame_index > std::numeric_limits<uint32_t>::max() || columns[3].empty()) {
            error = "invalid numeric or dataset field at arrival trace line " +
                    std::to_string(line_number);
            return false;
        }
        if (have_row && arrival_ms < previous_arrival_ms) {
            error = "arrival trace must be sorted by nondecreasing arrival_ms (line " +
                    std::to_string(line_number) + ")";
            return false;
        }
        previous_arrival_ms = arrival_ms;
        have_row = true;
        rows.push_back({arrival_ms, static_cast<uint32_t>(user_id),
                        static_cast<uint32_t>(frame_id), columns[3],
                        static_cast<uint32_t>(frame_index)});
    }
    if (rows.empty()) {
        error = "arrival trace has no data rows";
        return false;
    }
    return true;
}

static std::string media_stem(const std::string & input_dir, uint32_t frame_index) {
    const std::filesystem::path directory(input_dir);
    const std::string prefix = directory.filename().string();
    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "%04u", frame_index);
    const std::filesystem::path candidate = directory / (prefix + "_" + suffix);
    if (std::filesystem::exists(candidate.string() + ".jpg") ||
            std::filesystem::exists(candidate.string() + ".wav")) {
        return candidate.string();
    }
    // Keep compatibility with the original fixture directory name if a caller
    // passes a directory whose basename is not the file prefix.
    return (directory / ("duplex_omni_test_case_" + std::string(suffix))).string();
}

} // namespace

int main(int argc, char ** argv) {
    Options options;
    if (!parse_args(argc, argv, options)) {
        return options.help ? 0 : 2;
    }

    common_params params;
    params.model.path = options.model;
    params.n_ctx = options.n_ctx;
    params.n_batch = options.n_batch;
    params.n_ubatch = options.n_ubatch > 0 ? options.n_ubatch : options.n_batch;
    params.n_gpu_layers = options.n_gpu_layers;
    params.n_parallel = 1;
    params.sampling.seed = 42;
    params.sampling.temp = 0.0f;
    params.vpm_model = options.vpm_model;
    params.apm_model = options.apm_model;
    params.tts_model = options.tts_model;

    common_init();
    std::string load_error;
    auto bundle = OmniSharedModelBundle::load(params, &load_error);
    if (!bundle) {
        std::fprintf(stderr, "[MULTICTX] %s\n", load_error.c_str());
        return 1;
    }
    if (options.verify_token_embeddings && !verify_token_embedding_path(bundle, params)) {
        std::fprintf(stderr, "[MULTICTX] token embedding equivalence check failed\n");
        return 1;
    }

    OmniMultiUserConfig config;
    config.max_users = options.users;
    config.max_pending_frames_per_user = options.frames + 1;
    config.max_total_pending_frames = options.users * (options.frames + 1);
    config.default_max_new_tokens = options.max_new_tokens;
    config.enable_vpm = !options.vpm_model.empty();
    config.enable_apm = !options.apm_model.empty();
    config.enable_tts = options.tts;
    config.enable_t2w = options.t2w;
    config.tts_bin_dir = options.t2w_dir;
    config.llm_execution_policy = options.llm_policy;
    config.llm_max_batch_size = options.llm_max_batch_size;
    config.llm_max_batch_delay_us = options.llm_max_batch_delay_us;
    config.llm_sliding_window_high_tokens = options.llm_sliding_high_tokens;
    config.llm_sliding_window_low_tokens = options.llm_sliding_low_tokens;
    config.duplex_force_listen_count = options.force_listen_count;
    OmniMultiContextRuntime runtime(bundle, params, config);
    if (!runtime.valid()) {
        std::fprintf(stderr, "[MULTICTX] runtime init failed: %s\n", runtime.error().c_str());
        return 1;
    }

    for (uint32_t user = 1; user <= options.users; ++user) {
        OmniUserSessionConfig session;
        session.voice_audio = options.ref_audio;
        if (!runtime.open_user(user, session)) {
            std::fprintf(stderr, "[MULTICTX] open_user failed user=%u\n", user);
            runtime.shutdown();
            return 1;
        }
    }

    if (!options.input_dir.empty() && !options.input_dirs.empty()) {
        std::fprintf(stderr, "[MULTICTX] use either --input-dir or --input-dirs, not both\n");
        runtime.shutdown();
        return 2;
    }
    std::vector<std::string> input_dirs = split_csv(options.input_dirs);
    if (input_dirs.empty() && !options.input_dir.empty()) {
        input_dirs.push_back(options.input_dir);
    }
    if (input_dirs.size() > 1 && input_dirs.size() != options.users) {
        std::fprintf(stderr, "[MULTICTX] --input-dirs has %zu entries, expected one or %u\n",
                     input_dirs.size(), options.users);
        runtime.shutdown();
        return 2;
    }
    if (input_dirs.size() == 1 && options.users > 1) {
        input_dirs.resize(options.users, input_dirs.front());
    }

    std::vector<llama_token> decode_replay;
    std::vector<llama_token> expected_replay_text;
    if (options.decode_token_replay) {
        const llama_vocab * vocab = llama_model_get_vocab(bundle->llm_model());
        const std::vector<llama_token> speak = common_tokenize(vocab, "<|speak|>", false, true);
        const std::vector<llama_token> text = common_tokenize(vocab, " hello", false, true);
        const std::vector<llama_token> end = common_tokenize(vocab, "<|chunk_eos|>", false, true);
        if (speak.size() != 1 || text.empty() || end.size() != 1) {
            std::fprintf(stderr, "[MULTICTX] failed to build fixed decode token trace\n");
            runtime.shutdown();
            return 1;
        }
        decode_replay.push_back(speak[0]);
        decode_replay.insert(decode_replay.end(), text.begin(), text.end());
        decode_replay.push_back(end[0]);
        expected_replay_text = text;
    }

    auto build_task = [&](uint32_t user, uint32_t frame, uint32_t frame_index) {
        OmniUserTask task;
        task.user_id = user;
        task.frame_id = frame;
        if (input_dirs.empty()) {
            const std::string prompt = "baseline user " + std::to_string(user) +
                                       " frame " + std::to_string(frame);
            if (options.token_replay) {
                task.tokens = common_tokenize(llama_model_get_vocab(bundle->llm_model()),
                                              prompt, false, true);
            } else {
                task.text = prompt;
            }
        }
        task.max_new_tokens = options.max_new_tokens;
        task.replay_decode_tokens = decode_replay;
        if (options.embedding_tokens > 0) {
            const uint32_t n_tokens = options.embedding_tokens +
                (user - 1) * options.embedding_token_step;
            task.llm_embedding_dim = llama_model_n_embd(bundle->llm_model());
            task.llm_audio_embeddings.assign(
                static_cast<size_t>(n_tokens) * task.llm_embedding_dim,
                0.001f * static_cast<float>(user + frame));
        }
        task.input.img_fname = options.image;
        task.input.aud_fname = options.audio;
        if (!input_dirs.empty()) {
            const std::string stem = media_stem(input_dirs[user - 1], frame_index);
            task.input.img_fname = stem + ".jpg";
            task.input.aud_fname = stem + ".wav";
        }
        task.run_tts = options.tts;
        task.run_t2w = options.t2w;
        if (options.tts) {
            task.tts_embedding_tokens = 1;
            task.tts_embedding_dim = bundle->tts_n_embd();
            task.tts_embeddings.assign(static_cast<size_t>(std::max(1, task.tts_embedding_dim)), 0.0f);
        }
        if (options.t2w && !options.tts) {
            task.audio_tokens = {4218, 4218, 4218};
        }
        return task;
    };

    std::vector<uint32_t> expected_frames(options.users + 1, options.frames);
    std::vector<ArrivalTraceRow> trace_rows;
    if (!options.arrival_trace.empty()) {
        std::string trace_error;
        if (!read_arrival_trace(options.arrival_trace, trace_rows, trace_error)) {
            std::fprintf(stderr, "[MULTICTX] %s\n", trace_error.c_str());
            runtime.shutdown();
            return 2;
        }
        std::vector<uint32_t> next_frame(options.users + 1, 1);
        std::vector<bool> seen_user(options.users + 1, false);
        for (const ArrivalTraceRow & row : trace_rows) {
            if (row.user_id > options.users || row.frame_id > options.frames ||
                    row.frame_id != next_frame[row.user_id]) {
                std::fprintf(stderr, "[MULTICTX] arrival trace frame order invalid for user=%u frame=%u\n",
                             row.user_id, row.frame_id);
                runtime.shutdown();
                return 2;
            }
            if (input_dirs.empty()) {
                std::fprintf(stderr, "[MULTICTX] --arrival-trace requires --input-dir or --input-dirs\n");
                runtime.shutdown();
                return 2;
            }
            const std::string expected_dataset =
                std::filesystem::path(input_dirs[row.user_id - 1]).filename().string();
            if (row.dataset_id != expected_dataset) {
                std::fprintf(stderr, "[MULTICTX] arrival trace dataset mismatch user=%u: got=%s expected=%s\n",
                             row.user_id, row.dataset_id.c_str(), expected_dataset.c_str());
                runtime.shutdown();
                return 2;
            }
            ++next_frame[row.user_id];
            seen_user[row.user_id] = true;
        }
        for (uint32_t user = 1; user <= options.users; ++user) {
            if (!seen_user[user]) {
                std::fprintf(stderr, "[MULTICTX] arrival trace has no rows for user=%u\n", user);
                runtime.shutdown();
                return 2;
            }
            expected_frames[user] = next_frame[user] - 1;
        }
        const auto run_begin = std::chrono::steady_clock::now();
        for (const ArrivalTraceRow & row : trace_rows) {
            const auto target = run_begin + std::chrono::milliseconds(row.arrival_ms);
            std::this_thread::sleep_until(target);
            OmniUserTask task = build_task(row.user_id, row.frame_id, row.frame_index);
            const OmniSubmitStatus status = runtime.submit(task);
            if (status != OmniSubmitStatus::ACCEPTED) {
                std::fprintf(stderr, "[MULTICTX] submit failed user=%u frame=%u status=%d\n",
                             row.user_id, row.frame_id, static_cast<int>(status));
                runtime.shutdown();
                return 1;
            }
        }
        std::printf("[MULTICTX TRACE] path=%s rows=%zu users=%u\n",
                    options.arrival_trace.c_str(), trace_rows.size(), options.users);
    } else {
        for (uint32_t frame = 1; frame <= options.frames; ++frame) {
            for (uint32_t user = 1; user <= options.users; ++user) {
                OmniUserTask task = build_task(user, frame, frame - 1);
                const OmniSubmitStatus status = runtime.submit(task);
                if (status != OmniSubmitStatus::ACCEPTED) {
                    std::fprintf(stderr, "[MULTICTX] submit failed user=%u frame=%u status=%d\n",
                                 user, frame, static_cast<int>(status));
                    runtime.shutdown();
                    return 1;
                }
                if (options.user_stagger_ms > 0 && user < options.users) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(options.user_stagger_ms));
                }
            }
            if (options.interval_ms > 0 && frame < options.frames) {
                std::this_thread::sleep_for(std::chrono::milliseconds(options.interval_ms));
            }
        }
    }

    if (options.cancel_user > 0) {
        if (options.cancel_after_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(options.cancel_after_ms));
        }
        if (!runtime.cancel_user(options.cancel_user)) {
            std::fprintf(stderr, "[MULTICTX] cancel_user failed user=%u\n", options.cancel_user);
            runtime.shutdown();
            return 1;
        }
    }

    std::map<std::pair<uint64_t, uint64_t>, FrameObservation> observations;
    for (uint32_t user = 1; user <= options.users; ++user) {
        uint32_t terminal = 0;
        uint64_t last_terminal_frame = 0;
        while (terminal < expected_frames[user]) {
            OmniMultiUserEvent event;
            if (!runtime.wait_event(user, event, 120000)) {
                std::fprintf(stderr, "[MULTICTX] event timeout user=%u\n", user);
                runtime.shutdown();
                return 1;
            }
            auto & observation = observations[{event.user_id, event.frame_id}];
            if (event.kind == OmniMultiUserEventKind::AUDIO_WINDOW) {
                ++observation.audio_events;
                std::printf("[MULTICTX] user=%llu frame=%llu stage=%s embedding=%zu audio_tokens=%zu pcm=%zu\n",
                            static_cast<unsigned long long>(event.user_id),
                            static_cast<unsigned long long>(event.frame_id), event.stage.c_str(),
                            event.embedding.size(), event.audio_token_ids.size(), event.pcm.size());
            } else if (event.kind == OmniMultiUserEventKind::DECISION) {
                std::printf("[MULTICTX] user=%llu frame=%llu stage=%s decision=%s "
                            "silent=%s finish=%d text_tokens=%zu text=%s\n",
                            static_cast<unsigned long long>(event.user_id),
                            static_cast<unsigned long long>(event.frame_id), event.stage.c_str(),
                            event.is_speak ? "SPEAK" : "LISTEN", event.silent ? "yes" : "no",
                            static_cast<int>(event.finish_reason), event.token_ids.size(), event.text.c_str());
                observation.decision = true;
                observation.is_speak = event.is_speak;
                observation.text = event.text;
                observation.tokens = event.token_ids;
                observation.timings = event.timings;
            } else if (event.kind == OmniMultiUserEventKind::FRAME_DRAINED) {
                if (event.frame_id <= last_terminal_frame) {
                    std::fprintf(stderr, "[MULTICTX] non-FIFO terminal event user=%u frame=%llu after=%llu\n",
                                 user, static_cast<unsigned long long>(event.frame_id),
                                 static_cast<unsigned long long>(last_terminal_frame));
                    runtime.shutdown();
                    return 1;
                }
                last_terminal_frame = event.frame_id;
                observation.drained = true;
                observation.timings = event.timings;
                std::printf("[MULTICTX TIMING] user=%llu frame=%llu vpm_ms=%.3f apm_ms=%.3f "
                            "llm_ms=%.3f frame_queue_ms=%.3f green_lease_wait_ms=%.3f llm_queue_ms=%.3f "
                            "llm_batch_wait_ms=%.3f llm_prefill_batch_wait_ms=%.3f "
                            "llm_prefill_ms=%.3f llm_decode_ms=%.3f first_token_ms=%.3f "
                            "first_audio_ms=%.3f tts_ms=%.3f t2w_ms=%.3f total_ms=%.3f\n",
                            static_cast<unsigned long long>(event.user_id),
                            static_cast<unsigned long long>(event.frame_id), event.timings.vpm_ms,
                            event.timings.apm_ms, event.timings.llm_ms,
                            event.timings.frame_queue_wait_ms, event.timings.green_lease_wait_ms,
                            event.timings.llm_queue_wait_ms,
                            event.timings.llm_batch_wait_ms,
                            event.timings.llm_prefill_batch_wait_ms,
                            event.timings.llm_prefill_ms, event.timings.llm_decode_ms,
                            event.timings.first_token_ms, event.timings.first_audio_ms,
                            event.timings.tts_ms,
                            event.timings.t2w_ms, event.timings.total_ms);
                ++terminal;
            } else if (event.kind == OmniMultiUserEventKind::CANCELLED &&
                       event.user_id == options.cancel_user) {
                if (event.frame_id <= last_terminal_frame) {
                    std::fprintf(stderr, "[MULTICTX] non-FIFO cancel event user=%u frame=%llu after=%llu\n",
                                 user, static_cast<unsigned long long>(event.frame_id),
                                 static_cast<unsigned long long>(last_terminal_frame));
                    runtime.shutdown();
                    return 1;
                }
                last_terminal_frame = event.frame_id;
                observation.cancelled = true;
                ++terminal;
                std::printf("[MULTICTX] user=%llu frame=%llu cancelled=%s\n",
                            static_cast<unsigned long long>(event.user_id),
                            static_cast<unsigned long long>(event.frame_id), event.error.c_str());
            } else if (event.kind == OmniMultiUserEventKind::FAILED ||
                       event.kind == OmniMultiUserEventKind::CANCELLED) {
                std::fprintf(stderr, "[MULTICTX] frame failed user=%llu frame=%llu: %s\n",
                             static_cast<unsigned long long>(event.user_id),
                             static_cast<unsigned long long>(event.frame_id), event.error.c_str());
                runtime.shutdown();
                return 1;
            }
        }
    }

    for (const auto & entry : observations) {
        const FrameObservation & observation = entry.second;
        if (observation.cancelled) {
            continue;
        }
        if (!observation.decision || !observation.drained) {
            std::fprintf(stderr, "[MULTICTX] incomplete frame user=%llu frame=%llu\n",
                         static_cast<unsigned long long>(entry.first.first),
                         static_cast<unsigned long long>(entry.first.second));
            runtime.shutdown();
            return 1;
        }
        const bool forced_listen = entry.first.second <= options.force_listen_count;
        if (forced_listen && (observation.is_speak || !observation.tokens.empty() ||
                              observation.audio_events != 0)) {
            std::fprintf(stderr, "[MULTICTX] forced LISTEN emitted text or audio user=%llu frame=%llu\n",
                         static_cast<unsigned long long>(entry.first.first),
                         static_cast<unsigned long long>(entry.first.second));
            runtime.shutdown();
            return 1;
        }
        if (options.decode_token_replay && !forced_listen &&
                observation.tokens != expected_replay_text) {
            std::fprintf(stderr, "[MULTICTX] decode replay mismatch user=%llu frame=%llu\n",
                         static_cast<unsigned long long>(entry.first.first),
                         static_cast<unsigned long long>(entry.first.second));
            runtime.shutdown();
            return 1;
        }
    }

    const void * first_context = nullptr;
    const void * first_t2w_shared = nullptr;
    std::vector<const void *> t2w_flow_backends;
    std::vector<const void *> t2w_vocoder_backends;
    std::vector<int32_t> sequence_ids;
    for (uint32_t user = 1; user <= options.users; ++user) {
        OmniMultiContextUserStats stats;
        runtime.get_user_stats(user, stats);
        OmniMultiUserStageStats stage_stats;
        runtime.get_user_stage_stats(user, stage_stats);
        if (user == 1) {
            first_context = stats.context_address;
            first_t2w_shared = stats.t2w_shared_weights_address;
        } else if (options.llm_policy == OmniLlmExecutionPolicy::CONTINUOUS_BATCH &&
                   stats.context_address != first_context) {
            std::fprintf(stderr, "[MULTICTX] continuous users do not share one LLM context\n");
            runtime.shutdown();
            return 1;
        } else if (options.llm_policy == OmniLlmExecutionPolicy::PER_USER_CONTEXT &&
                   stats.context_address == first_context) {
            std::fprintf(stderr, "[MULTICTX] per-user policy unexpectedly shares an LLM context\n");
            runtime.shutdown();
            return 1;
        }
        if (options.t2w && (stats.t2w_shared_weights_address == nullptr ||
                            stats.t2w_shared_weights_address != first_t2w_shared)) {
            std::fprintf(stderr, "[MULTICTX] T2W users do not share one weight owner\n");
            runtime.shutdown();
            return 1;
        }
        if (options.t2w) {
            if (stats.t2w_flow_backend_address == nullptr || stats.t2w_vocoder_backend_address == nullptr ||
                    std::find(t2w_flow_backends.begin(), t2w_flow_backends.end(),
                              stats.t2w_flow_backend_address) != t2w_flow_backends.end() ||
                    std::find(t2w_vocoder_backends.begin(), t2w_vocoder_backends.end(),
                              stats.t2w_vocoder_backend_address) != t2w_vocoder_backends.end()) {
                std::fprintf(stderr, "[MULTICTX] T2W users do not have independent backend instances\n");
                runtime.shutdown();
                return 1;
            }
            t2w_flow_backends.push_back(stats.t2w_flow_backend_address);
            t2w_vocoder_backends.push_back(stats.t2w_vocoder_backend_address);
        }
        if (std::find(sequence_ids.begin(), sequence_ids.end(), stats.llm_seq_id) != sequence_ids.end() &&
                options.llm_policy == OmniLlmExecutionPolicy::CONTINUOUS_BATCH) {
            std::fprintf(stderr, "[MULTICTX] duplicate shared LLM seq_id=%d\n", stats.llm_seq_id);
            runtime.shutdown();
            return 1;
        }
        sequence_ids.push_back(stats.llm_seq_id);
        std::printf("[MULTICTX] user=%u ctx=%p seq_id=%d vpm=%p apm=%p tts=%p t2w=%p t2w_shared=%p "
                    "t2w_flow_backend=%p t2w_vocoder_backend=%p t2w_max_active=%u "
                    "n_ctx=%u n_ctx_seq=%u n_seq_max=%u kv_bytes=%llu kv_used=%llu apm_kv=%llu "
                    "n_past=%d units=%zu turn=%d slides=%llu dropped=%llu owner=%llu\n",
                    user, stats.context_address, stats.llm_seq_id,
                    stats.vpm_context_address, stats.apm_context_address,
                    stats.tts_context_address, stats.t2w_executor_address,
                    stats.t2w_shared_weights_address,
                    stats.t2w_flow_backend_address, stats.t2w_vocoder_backend_address,
                    stats.t2w_max_active_executors,
                    stats.n_ctx, stats.n_ctx_seq, stats.n_seq_max,
                    static_cast<unsigned long long>(stats.kv_bytes_estimate),
                    static_cast<unsigned long long>(stats.kv_bytes_used_estimate),
                    static_cast<unsigned long long>(stats.apm_kv_bytes), stats.n_past,
                    stats.llm_unit_count, stats.llm_current_turn,
                    static_cast<unsigned long long>(stats.sliding_event_count),
                    static_cast<unsigned long long>(stats.sliding_dropped_tokens),
                    static_cast<unsigned long long>(stats.owner_thread_id));
        std::printf("[MULTICTX STAGES] user=%u ", user);
        print_stage_stat("VPM", stage_stats.vpm);
        print_stage_stat("APM", stage_stats.apm);
        print_stage_stat("LLM", stage_stats.llm);
        print_stage_stat("TTS", stage_stats.tts);
        print_stage_stat("T2W", stage_stats.t2w);
        std::printf("completed_frames=%llu speak_frames=%llu listen_frames=%llu\n",
                    static_cast<unsigned long long>(stage_stats.completed_frames),
                    static_cast<unsigned long long>(stage_stats.speak_frames),
                    static_cast<unsigned long long>(stage_stats.listen_frames));
    }
    const OmniLlmBatchStats batch_stats = runtime.get_llm_batch_stats();
    std::printf("[MULTICTX BATCH] prefill=%llu prefill_dispatch=%llu prefill_items=%llu "
                "prefill_avg_batch=%.3f prefill_max_batch=%u prefill_fallback=%llu "
                "prefill_heterogeneous_fallback=%llu padding_ratio=%.6f mixed=%llu ubatches=%llu dispatch=%llu items=%llu "
                "avg_batch=%.3f max_batch=%u fallback=%llu prefill_ms=%.3f decode_ms=%.3f "
                "prefill_wait_ms=%.3f prefill_wait_max_ms=%.3f batch_wait_ms=%.3f "
                "batch_wait_max_ms=%.3f graph_available=%d graph_hit=%llu graph_miss=%llu "
                "graph_capture=%llu graph_property_miss=%llu kv_tokens=%llu kv_peak=%llu "
                "kv_bytes=%llu kv_bytes_peak=%llu slides=%llu dropped=%llu\n",
                static_cast<unsigned long long>(batch_stats.prefill_count),
                static_cast<unsigned long long>(batch_stats.prefill_dispatch_count),
                static_cast<unsigned long long>(batch_stats.prefill_item_count),
                batch_stats.average_prefill_batch_size(), batch_stats.prefill_batch_size_max,
                static_cast<unsigned long long>(batch_stats.prefill_fallback_count),
                static_cast<unsigned long long>(batch_stats.prefill_heterogeneous_fallback_count),
                batch_stats.prefill_padding_ratio(),
                static_cast<unsigned long long>(batch_stats.mixed_dispatch_count),
                static_cast<unsigned long long>(batch_stats.ubatch_count),
                static_cast<unsigned long long>(batch_stats.decode_dispatch_count),
                static_cast<unsigned long long>(batch_stats.decode_item_count),
                batch_stats.average_batch_size(), batch_stats.decode_batch_size_max,
                static_cast<unsigned long long>(batch_stats.decode_fallback_count),
                batch_stats.prefill_service_ms, batch_stats.decode_service_ms,
                batch_stats.prefill_batch_wait_ms, batch_stats.prefill_batch_wait_max_ms,
                batch_stats.batch_wait_ms, batch_stats.batch_wait_max_ms,
                batch_stats.graph_stats_available ? 1 : 0,
                static_cast<unsigned long long>(batch_stats.graph_hits),
                static_cast<unsigned long long>(batch_stats.graph_misses),
                static_cast<unsigned long long>(batch_stats.graph_captures),
                static_cast<unsigned long long>(batch_stats.graph_property_misses),
                static_cast<unsigned long long>(batch_stats.kv_tokens_used),
                static_cast<unsigned long long>(batch_stats.kv_tokens_peak),
                static_cast<unsigned long long>(batch_stats.kv_bytes_used_estimate),
                static_cast<unsigned long long>(batch_stats.kv_bytes_peak_estimate),
                static_cast<unsigned long long>(batch_stats.sliding_event_count),
                static_cast<unsigned long long>(batch_stats.sliding_dropped_tokens));
    if (options.expect_prefill_batch_min > 0 &&
            batch_stats.prefill_batch_size_max < options.expect_prefill_batch_min) {
        std::fprintf(stderr, "[MULTICTX] expected prefill batch >= %u, got %u\n",
                     options.expect_prefill_batch_min, batch_stats.prefill_batch_size_max);
        runtime.shutdown();
        return 1;
    }
    if (batch_stats.prefill_heterogeneous_fallback_count < options.expect_prefill_fallback_min) {
        std::fprintf(stderr, "[MULTICTX] expected heterogeneous fallback >= %llu, got %llu\n",
                     static_cast<unsigned long long>(options.expect_prefill_fallback_min),
                     static_cast<unsigned long long>(batch_stats.prefill_heterogeneous_fallback_count));
        runtime.shutdown();
        return 1;
    }
    if (batch_stats.mixed_dispatch_count < options.expect_mixed_min) {
        std::fprintf(stderr, "[MULTICTX] expected mixed dispatches >= %llu, got %llu\n",
                     static_cast<unsigned long long>(options.expect_mixed_min),
                     static_cast<unsigned long long>(batch_stats.mixed_dispatch_count));
        runtime.shutdown();
        return 1;
    }
    if (batch_stats.sliding_event_count < options.expect_sliding_min) {
        std::fprintf(stderr, "[MULTICTX] expected sliding events >= %llu, got %llu\n",
                     static_cast<unsigned long long>(options.expect_sliding_min),
                     static_cast<unsigned long long>(batch_stats.sliding_event_count));
        runtime.shutdown();
        return 1;
    }
    std::vector<double> total_ms;
    std::vector<double> queue_ms;
    std::vector<double> first_token_ms;
    std::vector<double> first_audio_ms;
    for (const auto & entry : observations) {
        if (entry.second.cancelled || !entry.second.drained) {
            continue;
        }
        total_ms.push_back(entry.second.timings.total_ms);
        queue_ms.push_back(entry.second.timings.frame_queue_wait_ms);
        first_token_ms.push_back(entry.second.timings.first_token_ms);
        if (entry.second.timings.first_audio_ms > 0.0) {
            first_audio_ms.push_back(entry.second.timings.first_audio_ms);
        }
    }
    std::printf("[MULTICTX LATENCY] total_p50=%.3f total_p95=%.3f total_p99=%.3f "
                "queue_p50=%.3f queue_p95=%.3f queue_p99=%.3f "
                "first_token_p50=%.3f first_token_p95=%.3f first_token_p99=%.3f "
                "first_audio_p50=%.3f first_audio_p95=%.3f first_audio_p99=%.3f\n",
                percentile(total_ms, 0.50), percentile(total_ms, 0.95), percentile(total_ms, 0.99),
                percentile(queue_ms, 0.50), percentile(queue_ms, 0.95), percentile(queue_ms, 0.99),
                percentile(first_token_ms, 0.50), percentile(first_token_ms, 0.95), percentile(first_token_ms, 0.99),
                percentile(first_audio_ms, 0.50), percentile(first_audio_ms, 0.95), percentile(first_audio_ms, 0.99));
    std::printf("[MULTICTX] model_bytes=%llu users=%u frames=%u llm_policy=%s token_replay=%s decode_token_replay=%s\n",
                static_cast<unsigned long long>(runtime.model_bytes()), options.users, options.frames,
                options.llm_policy == OmniLlmExecutionPolicy::CONTINUOUS_BATCH ? "continuous" : "per-user",
                options.token_replay ? "yes" : "no",
                options.decode_token_replay ? "yes" : "no");

    for (uint32_t user = 1; user <= options.users; ++user) {
        if (!runtime.close_user(user, OmniUserCloseMode::DRAIN, 120000)) {
            std::fprintf(stderr, "[MULTICTX] close_user failed user=%u\n", user);
            runtime.shutdown();
            return 1;
        }
    }
    runtime.shutdown();
    return 0;
}
