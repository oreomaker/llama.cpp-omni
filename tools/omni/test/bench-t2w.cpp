/**
 * Token2Wav Memory-Bound & SM Scaling Benchmark
 *
 * Measures Token2Wav (Flow Matching + HiFiGAN vocoder) inference latency and
 * determines the minimum SM count needed via CUDA MPS scaling.
 *
 * Token2Wav pipeline per window:
 *   28 audio tokens -> Conformer encoder -> Flow Matching (DiT, 5-10 CFM steps)
 *       -> 80ch mel spectrogram -> HiFiGAN vocoder -> PCM audio
 *
 * Model sizes:
 *   Flow Matching: 155.8M params (~623 MB FP32)
 *   HiFiGAN:        20.8M params (~83 MB FP32)
 *   Total:         ~176.6M params (~706 MB)
 *
 * Build: added to tools/omni/CMakeLists.txt as llama-omni-bench-t2w
 *
 * Usage:
 *   llama-omni-bench-t2w --t2w-dir <token2wav-gguf-dir> [options]
 *   llama-omni-bench-t2w --t2w-dir ./token2wav-gguf --steps 50 --sm-fractions 0.1,0.2,0.5,1.0
 */

#include "llama.h"
#include "ggml.h"
#include "common.h"
#include "log.h"

#include "token2wav/token2wav-impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// GPU spec detection
// ============================================================================

struct GpuSpec {
    std::string name;
    int    sm_count       = 0;
    double bandwidth_gb_s = 0.0;
};

static GpuSpec detect_gpu() {
    GpuSpec gpu;
    // Try nvidia-smi
    std::string cmd =
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0 2>/dev/null";
    FILE * pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        char buf[512];
        if (fgets(buf, sizeof(buf), pipe)) {
            char name[256];
            if (sscanf(buf, "%[^,]", name) == 1) {
                gpu.name = name;
                size_t start = gpu.name.find_first_not_of(" ");
                if (start != std::string::npos) gpu.name = gpu.name.substr(start);
            }
        }
        pclose(pipe);
    }

    // Get SM count: use CUDA cores query if supported, else infer from GPU name
    cmd = "nvidia-smi --query-gpu=cuda.cores --format=csv,noheader -i 0 2>/dev/null";
    pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        char buf[256];
        if (fgets(buf, sizeof(buf), pipe)) {
            int cores = atoi(buf);
            if (cores > 0) gpu.sm_count = cores / 128;
        }
        pclose(pipe);
    }
    // Fallback: infer SM count from GPU name
    if (gpu.sm_count <= 0) {
        if (gpu.name.find("4090") != std::string::npos)      gpu.sm_count = 128;
        else if (gpu.name.find("4080") != std::string::npos) gpu.sm_count = 76;
        else if (gpu.name.find("4070") != std::string::npos) gpu.sm_count = 46;
        else if (gpu.name.find("3090") != std::string::npos) gpu.sm_count = 82;
        else if (gpu.name.find("H100") != std::string::npos) gpu.sm_count = 132;
        else if (gpu.name.find("A100") != std::string::npos) gpu.sm_count = 108;
        else if (gpu.name.find("L40") != std::string::npos)  gpu.sm_count = 142;
    }

    // Get bandwidth
    cmd = "nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader -i 0 2>/dev/null";
    pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        char buf[256];
        if (fgets(buf, sizeof(buf), pipe)) {
            double mem_clock_mhz = atof(buf);
            // Estimate bus width from known GPU models
            int bus_width = 5120;  // default H100
            if (gpu.name.find("H200") != std::string::npos)      bus_width = 6144;
            else if (gpu.name.find("H100") != std::string::npos) bus_width = 5120;
            else if (gpu.name.find("A100") != std::string::npos) bus_width = 5120;
            else if (gpu.name.find("4090") != std::string::npos) bus_width = 384;
            else if (gpu.name.find("4080") != std::string::npos) bus_width = 256;
            else if (gpu.name.find("3090") != std::string::npos) bus_width = 384;

            gpu.bandwidth_gb_s = mem_clock_mhz * 1e6 * (bus_width / 8) * 2.0 / 1e9;
        }
        pclose(pipe);
    }

    return gpu;
}

static GpuSpec g_gpu;

// ============================================================================
// Command-line arguments
// ============================================================================

static int    n_steps        = 50;
static int    n_warmup       = 5;
static int    n_timesteps    = 0;      // CFM steps (0 = auto from cache, typically 5)
static float  temperature    = 1.0f;
static std::string t2w_dir;
static std::string sm_fractions_str;
static std::string output_csv;
static int    user_sm_count  = 0;
static double user_bandwidth = 0.0;

// ============================================================================
// Helpers
// ============================================================================

static double vec_mean(const std::vector<double> & v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double vec_p95(std::vector<double> v) {
    if (v.empty()) return 0.0;
    size_t idx = (size_t)(v.size() * 0.95);
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

// ============================================================================
// Benchmark
// ============================================================================

struct T2WBenchResult {
    double time_ms_mean;
    double time_ms_p95;
    double time_ms_min;
    double windows_per_sec;
};

static T2WBenchResult bench_t2w_session(omni::flow::Token2WavSession & sess,
                                         int n_steps_to_run) {
    // Generate 28 random audio tokens (valid range: 0-6560)
    std::vector<int32_t> tokens(28);
    srand(42);
    for (int i = 0; i < 28; i++) {
        tokens[i] = rand() % 6561;
    }

    std::vector<double> times_ms;
    times_ms.reserve(n_steps_to_run);
    std::vector<float> wave_out;

    for (int i = 0; i < n_steps_to_run; i++) {
        wave_out.clear();
        bool is_last = (i == n_steps_to_run - 1);

        auto t0 = std::chrono::high_resolution_clock::now();
        if (!sess.feed_window(tokens.data(), 28, is_last, wave_out)) {
            fprintf(stderr, "[ERR] feed_window failed at step %d\n", i);
            break;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(dt);
    }

    if (times_ms.empty()) {
        return {0, 0, 0, 0};
    }

    T2WBenchResult r;
    r.time_ms_mean     = vec_mean(times_ms);
    r.time_ms_p95      = vec_p95(times_ms);
    r.time_ms_min      = *std::min_element(times_ms.begin(), times_ms.end());
    r.windows_per_sec  = 1000.0 / r.time_ms_mean;

    return r;
}

// ============================================================================
// Output
// ============================================================================

static void print_sep(const char * title = nullptr) {
    if (title) {
        std::cout << "\n" << std::string(78, '=') << "\n"
                  << title << "\n"
                  << std::string(78, '=') << "\n\n";
    }
}

static void print_header(const T2WBenchResult & r) {
    print_sep("Token2Wav SM Scaling Benchmark");

    std::cout << "GPU:              " << g_gpu.name << "\n";
    std::cout << "  SMs:            " << (user_sm_count > 0 ? user_sm_count : g_gpu.sm_count) << "\n";
    std::cout << "  Peak BW:        " << (user_bandwidth > 0 ? user_bandwidth : g_gpu.bandwidth_gb_s) << " GB/s\n";
    std::cout << "\n";
    std::cout << "T2W Model dir:    " << t2w_dir << "\n";
    std::cout << "CFM timesteps:    " << n_timesteps << "\n";
    std::cout << "Temperature:      " << temperature << "\n";
    std::cout << "Steps per trial:  " << n_steps << "\n";
    std::cout << "Warmup:           " << n_warmup << "\n\n";

    // Model size analysis
    std::cout << "--- Model Sizes (FP32) ---\n";
    std::cout << "  Flow Matching:   155.8M params, ~623 MB\n";
    std::cout << "  HiFiGAN:          20.8M params, ~ 83 MB\n";
    std::cout << "  Total:           176.6M params, ~706 MB\n\n";

    std::cout << "--- Per-Window Workload ---\n";
    std::cout << "  Input:           28 audio tokens\n";
    std::cout << "  Conformer:       6 blocks + 4 up-blocks (stride 2)\n";
    std::cout << "  CFM DiT:         16 layers, 8 heads, 512 hidden\n";
    std::cout << "  CFM steps:       " << n_timesteps << " (each step: DiT fwd + Euler)\n";
    std::cout << "  HiFiGAN:         120x upsampling, 3 resblock groups\n\n";
}

static void print_sm_scaling(const std::vector<double> & fractions,
                              const std::vector<T2WBenchResult> & results) {
    print_sep("SM Scaling Results");

    if (results.empty()) {
        std::cout << "  No results.\n\n";
        return;
    }

    std::cout << std::setw(10) << "SM Frac"
              << std::setw(10) << "~SMs"
              << std::setw(16) << "Mean(ms)"
              << std::setw(16) << "P95(ms)"
              << std::setw(16) << "Min(ms)"
              << std::setw(18) << "Windows/s"
              << "\n" << std::string(86, '-') << "\n";

    int total_sms = user_sm_count > 0 ? user_sm_count : g_gpu.sm_count;
    double best_time = 1e9;
    for (const auto & r : results) {
        if (r.time_ms_mean < best_time) best_time = r.time_ms_mean;
    }

    for (size_t i = 0; i < results.size() && i < fractions.size(); i++) {
        const auto & r = results[i];
        double frac = fractions[i];
        int approx_sms = std::max(1, (int)std::round(total_sms * frac));

        std::cout << std::fixed;
        std::cout << std::setw(10) << std::setprecision(2) << frac
                  << std::setw(10) << approx_sms
                  << std::setw(16) << std::setprecision(3) << r.time_ms_mean
                  << std::setw(16) << std::setprecision(3) << r.time_ms_p95
                  << std::setw(16) << std::setprecision(3) << r.time_ms_min
                  << std::setw(18) << std::setprecision(1) << r.windows_per_sec;

        if (r.time_ms_mean <= best_time * 1.05) {
            std::cout << "  <-- near-optimal";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Find saturation knee
    for (size_t i = 0; i < results.size(); i++) {
        double deg = (results[i].time_ms_mean - best_time) / best_time;
        if (deg < 0.05) {
            double sat_frac = fractions[i];
            int sat_sms = std::max(1, (int)std::round(total_sms * sat_frac));
            std::cout << "--- Saturation Point ---\n";
            std::cout << "  SM fraction: " << sat_frac * 100.0
                      << "% (~" << sat_sms << " SMs)\n";
            std::cout << "  Above this, adding SMs gives <5% improvement\n";
            std::cout << "  For mega-kernel: Token2Wav needs ~" << sat_sms
                      << " SMs to saturate compute\n\n";
            break;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if ((arg == "--t2w-dir" || arg == "-d") && i + 1 < argc) {
            t2w_dir = argv[++i];
        } else if (arg == "--steps" && i + 1 < argc) {
            n_steps = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            n_warmup = std::stoi(argv[++i]);
        } else if (arg == "--timesteps" && i + 1 < argc) {
            n_timesteps = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--sm-fractions" && i + 1 < argc) {
            sm_fractions_str = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (arg == "--sm-count" && i + 1 < argc) {
            user_sm_count = std::stoi(argv[++i]);
        } else if (arg == "--bandwidth" && i + 1 < argc) {
            user_bandwidth = std::stod(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            printf("Usage: llama-omni-bench-t2w --t2w-dir <token2wav-gguf-dir> [options]\n\n");
            printf("Token2Wav (Flow Matching + HiFiGAN) SM Scaling Benchmark\n\n");
            printf("Options:\n");
            printf("  -d, --t2w-dir DIR      Token2Wav GGUF directory (required)\n");
            printf("  --steps N               Windows per trial (default: 50)\n");
            printf("  --warmup N              Warmup windows (default: 5)\n");
            printf("  --timesteps N            CFM timesteps (default: 10)\n");
            printf("  --temperature F          Temperature (default: 1.0)\n");
            printf("  --sm-fractions LIST      SM fractions to test (e.g.: 0.1,0.2,0.3,0.5,1.0)\n");
            printf("  --output FILE            CSV output file\n");
            printf("  --sm-count N             Override GPU SM count\n");
            printf("  --bandwidth GB_S         Override GPU memory bandwidth\n");
            return 0;
        }
    }

    if (t2w_dir.empty()) {
        fprintf(stderr, "Error: Token2Wav GGUF directory required. Use --t2w-dir <dir>\n");
        return 1;
    }

    // Detect GPU
    g_gpu = detect_gpu();
    fprintf(stderr, "[INFO] GPU: %s, SMs=%d, BW=%.0f GB/s\n",
            g_gpu.name.c_str(), g_gpu.sm_count, g_gpu.bandwidth_gb_s);

    // Model file paths
    std::string encoder_gguf       = t2w_dir + "/encoder.gguf";
    std::string flow_matching_gguf = t2w_dir + "/flow_matching.gguf";
    std::string flow_extra_gguf    = t2w_dir + "/flow_extra.gguf";
    std::string prompt_cache_gguf  = t2w_dir + "/prompt_cache.gguf";
    std::string vocoder_gguf       = t2w_dir + "/hifigan2.gguf";

    // Verify all files exist
    for (const auto & f : {encoder_gguf, flow_matching_gguf, flow_extra_gguf,
                            prompt_cache_gguf, vocoder_gguf}) {
        if (FILE * fp = fopen(f.c_str(), "rb")) {
            fclose(fp);
        } else {
            fprintf(stderr, "Error: GGUF file not found: %s\n", f.c_str());
            return 1;
        }
    }

    // Parse SM fractions
    std::vector<double> fractions;
    if (!sm_fractions_str.empty()) {
        std::stringstream ss(sm_fractions_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            fractions.push_back(std::stod(item));
        }
    } else {
        // Default fractions for in-process testing
        fractions = {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0};
    }

    // Print header
    print_header({});

    // Check if running under MPS
    const char * mps_env = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
    bool mps_active = (mps_env != nullptr);

    if (!mps_active && sm_fractions_str.empty()) {
        // Just run a single benchmark at full SM count
        fprintf(stderr, "[INFO] No SM fractions specified. Running single benchmark at 100%% SMs.\n");

        omni::flow::Token2WavSession sess;
        fprintf(stderr, "[INFO] Step 1/2: Loading models...\n");
        fflush(stderr);
        if (!sess.t2w.load_models(encoder_gguf, flow_matching_gguf, flow_extra_gguf,
                                   vocoder_gguf, "gpu", "gpu")) {
            fprintf(stderr, "[ERR] Token2Wav::load_models failed\n");
            return 1;
        }
        fprintf(stderr, "[INFO] Step 2/2: Loading prompt cache & starting stream...\n");
        fflush(stderr);
        if (!sess.t2w.start_stream_with_prompt_cache_gguf(prompt_cache_gguf, n_timesteps, temperature)) {
            fprintf(stderr, "[ERR] Token2Wav::start_stream_with_prompt_cache_gguf failed\n");
            return 1;
        }
        fprintf(stderr, "[INFO] Token2Wav session ready.\n");

        fprintf(stderr, "[INFO] Warmup (%d windows)...\n", n_warmup);
        auto warmup_result = bench_t2w_session(sess, n_warmup);

        fprintf(stderr, "[INFO] Benchmark (%d windows)...\n", n_steps);
        auto result = bench_t2w_session(sess, n_steps);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Mean time/window:  " << result.time_ms_mean << " ms\n";
        std::cout << "  P95 time/window:   " << result.time_ms_p95 << " ms\n";
        std::cout << "  Windows/sec:       " << std::setprecision(1) << result.windows_per_sec << "\n";
        std::cout << "  (TTS generates ~25 audio tokens/window at ~1 window/sec)\n";
        std::cout << "  Real-time factor:  " << std::setprecision(1)
                  << result.windows_per_sec * 25.0 / 25.0 << "x (generates "
                  << result.windows_per_sec * 25.0 << " tokens/sec)\n\n";
    } else if (mps_active) {
        // Running under MPS - one fraction at a time
        double current_frac = std::stod(mps_env) / 100.0;
        fprintf(stderr, "[INFO] MPS active: %.0f%% SMs (fraction=%.2f)\n",
                std::stod(mps_env), current_frac);

        omni::flow::Token2WavSession sess;
        if (!sess.t2w.load_models(encoder_gguf, flow_matching_gguf, flow_extra_gguf,
                                   vocoder_gguf, "gpu", "gpu")) {
            fprintf(stderr, "[ERR] MPS mode: Token2Wav::load_models failed\n");
            return 1;
        }
        if (!sess.t2w.start_stream_with_prompt_cache_gguf(prompt_cache_gguf, n_timesteps, temperature)) {
            fprintf(stderr, "[ERR] MPS mode: Token2Wav::start_stream_with_prompt_cache_gguf failed\n");
            return 1;
        }

        auto result = bench_t2w_session(sess, n_steps);
        // Print data line for parent process to parse
        std::cout << std::fixed;
        std::cout << "T2W_DATA: " << current_frac << " "
                  << result.time_ms_mean << " " << result.time_ms_p95 << " "
                  << result.windows_per_sec << "\n";
    } else {
        // SM scaling via MPS re-exec
        fprintf(stderr, "[INFO] SM scaling via MPS re-exec...\n");

        // Check MPS daemon
        bool mps_daemon = (system("pgrep -x nvidia-cuda-mps > /dev/null 2>&1") == 0 ||
                           system("pgrep -x nvidia-cuda-mps-c > /dev/null 2>&1") == 0);

        if (!mps_daemon) {
            fprintf(stderr, "[ERR] MPS daemon not running.\n");
            fprintf(stderr, "[ERR] Start with: sudo nvidia-cuda-mps-control -d\n");
            fprintf(stderr, "[ERR] Then re-run with --sm-fractions.\n");
            return 1;
        }

        std::vector<T2WBenchResult> all_results;
        std::string self_path = argv[0];

        // Build base args (exclude --sm-fractions to avoid infinite recursion)
        std::string base_args;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--sm-fractions") == 0) { i++; continue; }
            base_args += " \"" + std::string(argv[i]) + "\"";
        }

        for (double frac : fractions) {
            int pct = std::max(1, (int)(frac * 100.0));
            fprintf(stderr, "[INFO] Testing SM fraction=%.2f (%d%%)...\n", frac, pct);

            const char * mps_pipe = getenv("CUDA_MPS_PIPE_DIRECTORY");
            std::string cmd =
                (mps_pipe ? "CUDA_MPS_PIPE_DIRECTORY=" + std::string(mps_pipe) + " " : "") +
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=" + std::to_string(pct) +
                " CUDA_VISIBLE_DEVICES=0" +
                " " + self_path + base_args + " 2>&1";  // keep stderr visible for diagnostics

            FILE * pipe = popen(cmd.c_str(), "r");
            if (!pipe) {
                fprintf(stderr, "[ERR] Failed to exec benchmark for fraction %.2f\n", frac);
                continue;
            }

            char buf[1024];
            T2WBenchResult r = {0, 0, 0, 0};
            while (fgets(buf, sizeof(buf), pipe)) {
                std::string line(buf);
                if (line.find("T2W_DATA:") != std::string::npos) {
                    double f;
                    if (sscanf(line.c_str(), "T2W_DATA: %lf %lf %lf %lf",
                               &f, &r.time_ms_mean, &r.time_ms_p95,
                               &r.windows_per_sec) >= 4) {
                        // success
                    }
                }
            }
            pclose(pipe);

            if (r.time_ms_mean > 0) {
                all_results.push_back(r);
            } else {
                fprintf(stderr, "[WARN] Failed to parse result for fraction %.2f\n", frac);
            }
        }

        print_sm_scaling(fractions, all_results);

        // CSV export
        if (!output_csv.empty()) {
            std::ofstream f(output_csv);
            if (f) {
                f << "sm_fraction,approx_sms,time_ms_mean,time_ms_p95,time_ms_min,windows_per_sec\n";
                int total_sms = user_sm_count > 0 ? user_sm_count : g_gpu.sm_count;
                for (size_t i = 0; i < all_results.size() && i < fractions.size(); i++) {
                    int approx_sms = std::max(1, (int)std::round(total_sms * fractions[i]));
                    f << fractions[i] << "," << approx_sms << ","
                      << all_results[i].time_ms_mean << ","
                      << all_results[i].time_ms_p95 << ","
                      << all_results[i].time_ms_min << ","
                      << all_results[i].windows_per_sec << "\n";
                }
                f.close();
                fprintf(stderr, "[INFO] Results written to %s\n", output_csv.c_str());
            }
        }
    }

    return 0;
}
