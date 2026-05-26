/**
 * Vision Encoder (SigLip/VPM) SM Scaling Benchmark
 *
 * Verifies vision encoder compute characteristics. Unlike TTS (memory-bound,
 * saturates at ~15% SM), vision encoding has very high arithmetic intensity
 * (~978 FLOP/Byte, compute-bound) and should show sustained scaling with SM count.
 *
 * Model: SigLip2 ViT + Resampler
 *   - 27 layers, hidden=1152, FFN=4304, 16 heads, ~460M params, ~920 MB BF16
 *   - Input: image pixels → Output: 64 tokens × 4096 dims
 *
 * Usage:
 *   llama-omni-bench-vision -m <vision_model.gguf> [options]
 *   llama-omni-bench-vision -m vision.gguf --steps 50 --sm-fractions 0.1,0.2,0.5,1.0
 */

#include "vision.h"
#include "llama.h"
#include "ggml.h"
#include "common.h"
#include "log.h"

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
    FILE * pipe = popen(
        "nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null", "r");
    if (pipe) {
        char buf[512];
        if (fgets(buf, sizeof(buf), pipe)) {
            gpu.name = buf;
            size_t start = gpu.name.find_first_not_of(" ");
            if (start != std::string::npos) gpu.name = gpu.name.substr(start);
            while (!gpu.name.empty() && gpu.name.back() == '\n')
                gpu.name.pop_back();
        }
        pclose(pipe);
    }

    if (gpu.name.find("4090") != std::string::npos)      { gpu.sm_count = 128; gpu.bandwidth_gb_s = 1008.0; }
    else if (gpu.name.find("4080") != std::string::npos) { gpu.sm_count = 76;  gpu.bandwidth_gb_s = 716.0; }
    else if (gpu.name.find("4070") != std::string::npos) { gpu.sm_count = 46;  gpu.bandwidth_gb_s = 504.0; }
    else if (gpu.name.find("3090") != std::string::npos) { gpu.sm_count = 82;  gpu.bandwidth_gb_s = 936.0; }
    else if (gpu.name.find("H100") != std::string::npos) { gpu.sm_count = 132; gpu.bandwidth_gb_s = 3352.0; }
    else if (gpu.name.find("H200") != std::string::npos) { gpu.sm_count = 132; gpu.bandwidth_gb_s = 4800.0; }
    else if (gpu.name.find("A100") != std::string::npos) { gpu.sm_count = 108; gpu.bandwidth_gb_s = 2039.0; }
    else if (gpu.name.find("L40") != std::string::npos)  { gpu.sm_count = 142; gpu.bandwidth_gb_s = 864.0; }
    else if (gpu.name.find("Orin") != std::string::npos) { gpu.sm_count = 16;  gpu.bandwidth_gb_s = 200.0; }

    return gpu;
}

static GpuSpec g_gpu;

// ============================================================================
// CLI arguments
// ============================================================================

static int    n_steps        = 50;
static int    n_warmup       = 5;
static int    n_threads      = 8;
static std::string model_path;
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
    if (idx >= v.size()) idx = v.size() - 1;
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

// ============================================================================
// Benchmark
// ============================================================================

struct VisionBenchResult {
    double time_ms_mean;
    double time_ms_p95;
    double time_ms_min;
    double encodes_per_sec;
};

static VisionBenchResult bench_vision_encode(
    vision_ctx * ctx,
    vision_image_f32 * img,
    int n_output,
    int n_embd,
    int n_steps_to_run)
{
    std::vector<float> output(n_output * n_embd);
    std::vector<double> times_ms;
    times_ms.reserve(n_steps_to_run);

    for (int i = 0; i < n_steps_to_run; i++) {
        std::fill(output.begin(), output.end(), 0.0f);

        auto t0 = std::chrono::high_resolution_clock::now();
        if (!vision_image_encode(ctx, n_threads, img, output.data())) {
            fprintf(stderr, "[ERR] vision_image_encode failed at step %d\n", i);
            break;
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(dt);
    }

    if (times_ms.empty()) return {0, 0, 0, 0};

    VisionBenchResult r;
    r.time_ms_mean    = vec_mean(times_ms);
    r.time_ms_p95     = vec_p95(times_ms);
    r.time_ms_min     = *std::min_element(times_ms.begin(), times_ms.end());
    r.encodes_per_sec = 1000.0 / r.time_ms_mean;
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

static void print_header() {
    print_sep("Vision Encoder (SigLip) SM Scaling Benchmark");

    int sm = user_sm_count > 0 ? user_sm_count : g_gpu.sm_count;
    double bw = user_bandwidth > 0 ? user_bandwidth : g_gpu.bandwidth_gb_s;

    std::cout << "GPU:              " << g_gpu.name << "\n";
    std::cout << "  SMs:            " << sm << "\n";
    std::cout << "  Peak BW:        " << bw << " GB/s\n";
    std::cout << "\n";
    std::cout << "Vision model:     " << model_path << "\n";
    std::cout << "Threads:          " << n_threads << "\n";
    std::cout << "Steps per trial:  " << n_steps << "\n";
    std::cout << "Warmup:           " << n_warmup << "\n\n";

    std::cout << "--- Model Specs (SigLip2 ViT + Resampler) ---\n";
    std::cout << "  Architecture:    27 layers, 1152 hidden, 4304 FFN, 16 heads\n";
    std::cout << "  Parameters:      ~460M (~920 MB BF16)\n";
    std::cout << "  Input:           image pixels (dynamic resolution)\n";
    std::cout << "  Output:          64 tokens × 4096 dims\n";
    std::cout << "  Compute:         ViT self-attn + cross-attn resampler\n";
    std::cout << "  Arithmetic Int:  ~978 FLOP/Byte (compute-bound)\n\n";
}

static void print_sm_scaling(const std::vector<double> & fractions,
                              const std::vector<VisionBenchResult> & results) {
    print_sep("SM Scaling Results");

    if (results.empty()) {
        std::cout << "  No results.\n\n";
        return;
    }

    int total_sms = user_sm_count > 0 ? user_sm_count : g_gpu.sm_count;

    std::cout << std::setw(10) << "SM Frac"
              << std::setw(10) << "~SMs"
              << std::setw(16) << "Mean(ms)"
              << std::setw(16) << "P95(ms)"
              << std::setw(16) << "Min(ms)"
              << std::setw(18) << "Encodes/s"
              << "\n" << std::string(86, '-') << "\n";

    double best_time = 1e9;
    for (const auto & r : results) {
        if (r.time_ms_mean > 0 && r.time_ms_mean < best_time)
            best_time = r.time_ms_mean;
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
                  << std::setw(18) << std::setprecision(1) << r.encodes_per_sec;

        if (r.time_ms_mean > 0 && r.time_ms_mean <= best_time * 1.05) {
            std::cout << "  <-- near-optimal";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Find saturation knee
    if (best_time < 1e8) {
        for (size_t i = 0; i < results.size(); i++) {
            if (results[i].time_ms_mean <= 0) continue;
            double deg = (results[i].time_ms_mean - best_time) / best_time;
            if (deg < 0.05) {
                double sat_frac = fractions[i];
                int sat_sms = std::max(1, (int)std::round(total_sms * sat_frac));
                std::cout << "--- Saturation Point ---\n";
                std::cout << "  SM fraction: " << sat_frac * 100.0
                          << "% (~" << sat_sms << " SMs) achieves near-optimal\n";
                break;
            }
        }

        // Compare with TTS (memory-bound) vs Vision (compute-bound)
        std::cout << "\n--- Expected Behavior ---\n";
        std::cout << "  TTS:    memory-bound (0.42 FLOP/Byte) → saturates at ~15-20% SM\n";
        std::cout << "  Vision: compute-bound (978 FLOP/Byte) → scales with SM count\n";
        if (results.size() >= 2 && results.front().time_ms_mean > 0
            && results.back().time_ms_mean > 0) {
            double speedup = results.front().time_ms_mean / results.back().time_ms_mean;
            std::cout << "  Measured speedup 10%→100% SM: " << std::setprecision(2)
                      << speedup << "x\n";
        }
    }
    std::cout << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--steps" && i + 1 < argc) {
            n_steps = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            n_warmup = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            n_threads = std::stoi(argv[++i]);
        } else if (arg == "--sm-fractions" && i + 1 < argc) {
            sm_fractions_str = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (arg == "--sm-count" && i + 1 < argc) {
            user_sm_count = std::stoi(argv[++i]);
        } else if (arg == "--bandwidth" && i + 1 < argc) {
            user_bandwidth = std::stod(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            printf("Usage: llama-omni-bench-vision -m <vision_model.gguf> [options]\n\n");
            printf("Vision Encoder (SigLip/VPM) SM Scaling Benchmark\n\n");
            printf("Options:\n");
            printf("  -m, --model PATH       Vision GGUF path (required)\n");
            printf("  --steps N               Encodes per trial (default: 50)\n");
            printf("  --warmup N              Warmup encodes (default: 5)\n");
            printf("  --threads N             CPU threads (default: 8)\n");
            printf("  --sm-fractions LIST     SM fractions (e.g.: 0.1,0.2,0.5,1.0)\n");
            printf("  --output FILE           CSV output file\n");
            printf("  --sm-count N            Override GPU SM count\n");
            printf("  --bandwidth GB_S        Override GPU memory bandwidth\n");
            return 0;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: Model path required. Use -m <vision_model.gguf>\n");
        return 1;
    }

    g_gpu = detect_gpu();
    fprintf(stderr, "[INFO] GPU: %s, SMs=%d, BW=%.0f GB/s\n",
            g_gpu.name.c_str(), g_gpu.sm_count, g_gpu.bandwidth_gb_s);

    // Parse SM fractions
    std::vector<double> fractions;
    if (!sm_fractions_str.empty()) {
        std::stringstream ss(sm_fractions_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            fractions.push_back(std::stod(item));
        }
    }

    print_header();

    const char * mps_env = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
    bool mps_active = (mps_env != nullptr);

    // --- Load model and create test image (shared by all paths) ---
    auto load_and_prep = [&](vision_ctx *& ctx, vision_image_f32_batch & batch,
                              int & n_output, int & n_embd) -> bool {
        vision_context_params vparams;
        vparams.use_gpu           = true;
        vparams.verbosity         = GGML_LOG_LEVEL_INFO;
        vparams.coreml_model_path = nullptr;

        ctx = vision_init(model_path.c_str(), vparams);
        if (!ctx) { fprintf(stderr, "[ERR] vision_init failed\n"); return false; }

        n_output = vision_n_output_tokens(ctx);
        n_embd   = vision_n_mmproj_embd(ctx);
        fprintf(stderr, "[INFO] Vision model: output=%d tokens x %d dims\n",
                n_output, n_embd);

        // Create synthetic RGB image (448x448 = typical MiniCPM-o overview size)
        const int img_size = 448;
        vision_image_u8 img_u8;
        img_u8.nx   = img_size;
        img_u8.ny   = img_size;
        img_u8.buf.resize(img_size * img_size * 3);
        srand(42);
        for (size_t i = 0; i < img_u8.buf.size(); i++)
            img_u8.buf[i] = (uint8_t)(rand() % 256);

        if (!vision_image_preprocess(ctx, &img_u8, &batch)) {
            fprintf(stderr, "[ERR] vision_image_preprocess failed\n");
            vision_free(ctx);
            return false;
        }
        fprintf(stderr, "[INFO] Preprocess: %zu image chunk(s), grid=%dx%d\n",
                batch.entries.size(), batch.grid_x, batch.grid_y);
        return true;
    };

    if (!mps_active && sm_fractions_str.empty()) {
        // === Single benchmark at full SM ===
        vision_ctx * ctx = nullptr;
        vision_image_f32_batch batch;
        int n_output, n_embd;
        if (!load_and_prep(ctx, batch, n_output, n_embd)) return 1;

        // Use the first (overview) image for repeated encode
        auto * img = batch.entries[0].get();

        fprintf(stderr, "[INFO] Warmup (%d encodes)...\n", n_warmup);
        bench_vision_encode(ctx, img, n_output, n_embd, n_warmup);

        fprintf(stderr, "[INFO] Benchmark (%d encodes)...\n", n_steps);
        auto result = bench_vision_encode(ctx, img, n_output, n_embd, n_steps);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Mean time/encode:  " << result.time_ms_mean << " ms\n";
        std::cout << "  P95 time/encode:   " << result.time_ms_p95 << " ms\n";
        std::cout << "  Encodes/sec:       " << std::setprecision(1)
                  << result.encodes_per_sec << "\n\n";

        vision_free(ctx);

    } else if (mps_active) {
        // === MPS active — single fraction, print data for parent ===
        double current_frac = std::stod(mps_env) / 100.0;
        fprintf(stderr, "[INFO] MPS active: %.0f%% SMs (fraction=%.2f)\n",
                std::stod(mps_env), current_frac);

        vision_ctx * ctx = nullptr;
        vision_image_f32_batch batch;
        int n_output, n_embd;
        if (!load_and_prep(ctx, batch, n_output, n_embd)) return 1;

        auto * img = batch.entries[0].get();

        bench_vision_encode(ctx, img, n_output, n_embd, n_warmup);
        auto result = bench_vision_encode(ctx, img, n_output, n_embd, n_steps);

        std::cout << "VISION_DATA: " << current_frac << " "
                  << result.time_ms_mean << " " << result.time_ms_p95 << " "
                  << result.encodes_per_sec << "\n";

        vision_free(ctx);

    } else {
        // === SM scaling via MPS re-exec ===
        fprintf(stderr, "[INFO] SM scaling via MPS re-exec...\n");

        bool mps_daemon = (system("pgrep -x nvidia-cuda-mps > /dev/null 2>&1") == 0 ||
                           system("pgrep -x nvidia-cuda-mps-c > /dev/null 2>&1") == 0);
        if (!mps_daemon) {
            fprintf(stderr, "[ERR] MPS daemon not running.\n");
            fprintf(stderr, "[ERR] Start: sudo nvidia-cuda-mps-control -d\n");
            return 1;
        }

        std::vector<VisionBenchResult> all_results;
        std::string self_path = argv[0];

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
                " " + self_path + base_args + " 2>&1";

            FILE * pipe = popen(cmd.c_str(), "r");
            if (!pipe) continue;

            char buf[1024];
            VisionBenchResult r = {0, 0, 0, 0};
            while (fgets(buf, sizeof(buf), pipe)) {
                std::string line(buf);
                if (line.find("VISION_DATA:") != std::string::npos) {
                    double f;
                    sscanf(line.c_str(), "VISION_DATA: %lf %lf %lf %lf",
                           &f, &r.time_ms_mean, &r.time_ms_p95,
                           &r.encodes_per_sec);
                }
            }
            pclose(pipe);

            if (r.time_ms_mean > 0) all_results.push_back(r);
            else fprintf(stderr, "[WARN] Failed to parse result for fraction %.2f\n", frac);
        }

        print_sm_scaling(fractions, all_results);

        if (!output_csv.empty()) {
            std::ofstream f(output_csv);
            if (f) {
                f << "sm_fraction,approx_sms,time_ms_mean,time_ms_p95,time_ms_min,encodes_per_sec\n";
                int total_sms = user_sm_count > 0 ? user_sm_count : g_gpu.sm_count;
                for (size_t i = 0; i < all_results.size() && i < fractions.size(); i++) {
                    int approx_sms = std::max(1, (int)std::round(total_sms * fractions[i]));
                    f << fractions[i] << "," << approx_sms << ","
                      << all_results[i].time_ms_mean << ","
                      << all_results[i].time_ms_p95 << ","
                      << all_results[i].time_ms_min << ","
                      << all_results[i].encodes_per_sec << "\n";
                }
                f.close();
                fprintf(stderr, "[INFO] Results written to %s\n", output_csv.c_str());
            }
        }
    }

    return 0;
}
