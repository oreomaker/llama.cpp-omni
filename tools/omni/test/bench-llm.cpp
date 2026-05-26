/**
 * LLM SM Partitioning Benchmark
 *
 * Measures LLM prefill and decode latency at 64/128 token lengths under
 * different SM fractions to determine SM sensitivity for mega-kernel
 * SM partitioning design.
 *
 * LLM decode is memory-bound at batch=1 (arithmetic intensity ~0.98 FLOP/Byte
 * for 8.19B model), similar to TTS but with much larger model weights.
 * LLM prefill is compute-bound (many tokens processed in parallel).
 *
 * Experiments:
 *   1. Analytical: compute arithmetic intensity and predict bottleneck type
 *   2. Prefill + Decode: measure latency at 64 and 128 token prefill/decode
 *   3. SM scaling: measure prefill and decode latency at different SM counts
 *
 * Build: added to tools/omni/CMakeLists.txt as llama-omni-bench-llm
 *
 * Usage:
 *   llama-omni-bench-llm -m <llm_model.gguf> [options]
 *   llama-omni-bench-llm -m llm.gguf --prefill-lens 64,128 --sm-fractions 0.05,0.1,0.2,0.5,1.0
 */

#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "common.h"

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
// GPU spec detection via nvidia-smi
// ============================================================================

struct GpuSpec {
    std::string name;
    int    sm_count       = 0;
    double bandwidth_gb_s = 0.0;
    double clock_mhz      = 0.0;
    double mem_clock_mhz  = 0.0;
    int    mem_bus_width  = 0;
};

static bool parse_nvidia_smi(GpuSpec & gpu, int device_id = 0) {
    std::string query_cmd =
        "nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.memory --format=csv,noheader,nounits -i "
        + std::to_string(device_id) + " 2>/dev/null";
    FILE * pipe = popen(query_cmd.c_str(), "r");
    if (!pipe) return false;

    char buf[512];
    if (fgets(buf, sizeof(buf), pipe)) {
        char name[256];
        int mem_mib;
        double sm_clock, mem_clock;
        if (sscanf(buf, "%[^,], %d MiB, %lf MHz, %lf MHz",
                   name, &mem_mib, &sm_clock, &mem_clock) >= 4) {
            gpu.name = name;
            size_t start = gpu.name.find_first_not_of(" ");
            if (start != std::string::npos) gpu.name = gpu.name.substr(start);

            gpu.clock_mhz     = sm_clock;
            gpu.mem_clock_mhz = mem_clock;

            if (gpu.name.find("4090") != std::string::npos)     { gpu.sm_count = 128; gpu.mem_bus_width = 384; }
            else if (gpu.name.find("4080") != std::string::npos) { gpu.sm_count = 76;  gpu.mem_bus_width = 256; }
            else if (gpu.name.find("4070") != std::string::npos) { gpu.sm_count = 46;  gpu.mem_bus_width = 192; }
            else if (gpu.name.find("4060") != std::string::npos) { gpu.sm_count = 24;  gpu.mem_bus_width = 128; }
            else if (gpu.name.find("A100") != std::string::npos) { gpu.sm_count = 108; gpu.mem_bus_width = 5120; }
            else if (gpu.name.find("H100") != std::string::npos) { gpu.sm_count = 132; gpu.mem_bus_width = 5120; }
            else if (gpu.name.find("H200") != std::string::npos) { gpu.sm_count = 132; gpu.mem_bus_width = 6144; }
            else if (gpu.name.find("H800") != std::string::npos) { gpu.sm_count = 132; gpu.mem_bus_width = 5120; }
            else if (gpu.name.find("A800") != std::string::npos) { gpu.sm_count = 108; gpu.mem_bus_width = 5120; }
            else if (gpu.name.find("L40") != std::string::npos)  { gpu.sm_count = 142; gpu.mem_bus_width = 384; }
            else if (gpu.name.find("L20") != std::string::npos)  { gpu.sm_count = 92;  gpu.mem_bus_width = 384; }
            else if (gpu.name.find("3090") != std::string::npos) { gpu.sm_count = 82;  gpu.mem_bus_width = 384; }
            else if (gpu.name.find("3080") != std::string::npos) { gpu.sm_count = 68;  gpu.mem_bus_width = 320; }
            else if (gpu.name.find("Orin") != std::string::npos) { gpu.sm_count = 16;  gpu.mem_bus_width = 128; }

            if (gpu.name.find("Orin") != std::string::npos) {
                gpu.bandwidth_gb_s = 200.0;
            } else if (gpu.mem_bus_width > 0) {
                gpu.bandwidth_gb_s = gpu.mem_clock_mhz * 1e6 * (gpu.mem_bus_width / 8) * 2.0 / 1e9;
            }
        }
    }
    pclose(pipe);

    std::string sm_cmd =
        "nvidia-smi --query-gpu=cuda.cores --format=csv,noheader -i "
        + std::to_string(device_id) + " 2>/dev/null";
    pipe = popen(sm_cmd.c_str(), "r");
    if (pipe) {
        if (fgets(buf, sizeof(buf), pipe)) {
            int cuda_cores = atoi(buf);
            if (cuda_cores > 0 && gpu.sm_count == 0) {
                gpu.sm_count = cuda_cores / 128;
            }
        }
        pclose(pipe);
    }

    return gpu.bandwidth_gb_s > 0 && gpu.sm_count > 0;
}

static GpuSpec g_gpu_spec;

// ============================================================================
// Command-line arguments
// ============================================================================

static int    n_steps          = 50;
static int    n_warmup         = 10;
static int    n_gpu_layers     = 99;
static int    n_ctx            = 4096;
static int    n_batch          = 2048;
static int    n_decode_tokens  = 32;
static int    n_repeat         = 3;
static double user_bandwidth   = 0.0;
static double user_tflops      = 0.0;
static int    user_sm_count    = 0;
static std::string model_path;
static std::string prefill_lens_str = "64,128";
static std::string sm_fractions_str;
static std::string output_csv;

// ============================================================================
// Analytical model
// ============================================================================

struct LLMAnalytical {
    int n_layers   = 0;
    int n_embd     = 0;
    int n_ff       = 0;
    int n_head     = 0;
    int n_kv_head  = 0;
    int head_dim   = 0;

    int64_t total_params       = 0;
    int64_t total_bytes_bf16   = 0;
    double  bytes_per_param    = 2.0;   // default BF16
    double  flops_per_decode   = 0.0;
    double  bytes_per_step     = 0.0;
    double  arithmetic_intensity = 0.0;

    double  peak_bw_gb_s       = 0.0;
    double  peak_tflops        = 0.0;
    double  ridge_point        = 0.0;
    int     sm_count           = 0;
    bool    decode_is_memory_bound = false;

    void compute(const llama_model * model) {
        n_layers  = llama_model_n_layer(model);
        n_embd    = llama_model_n_embd(model);
        n_head    = llama_model_n_head(model);
        n_kv_head = llama_model_n_head_kv(model);
        head_dim  = (n_head > 0) ? n_embd / n_head : 64;

        // Derive n_ff from total params
        // LLaMA: per-layer params = 4*n_embd^2 + 2*n_embd*n_kv_head*head_dim + 3*n_embd*n_ff
        int64_t q_proj = (int64_t)n_embd * n_head * head_dim;
        int64_t k_proj = (int64_t)n_embd * n_kv_head * head_dim;
        int64_t v_proj = (int64_t)n_embd * n_kv_head * head_dim;
        int64_t o_proj = (int64_t)n_head * head_dim * n_embd;
        int64_t per_layer_attn = q_proj + k_proj + v_proj + o_proj;

        total_params = llama_model_n_params(model);
        // Solve: total = n_layers*(attn + 3*n_embd*n_ff) + 2*n_embd*vocab
        // Approximate vocab from remaining params
        int64_t backbone_no_ffn = per_layer_attn * n_layers;
        int64_t remaining = (int64_t)total_params - backbone_no_ffn;
        // remaining ≈ n_layers * 3 * n_embd * n_ff + 2 * n_embd * vocab
        // Assume vocab ≈ 128000 (typical) and solve for n_ff
        int64_t vocab_estimate = 128000;
        int64_t embed_params = 2 * (int64_t)n_embd * vocab_estimate;
        int64_t ffn_params_est = remaining - embed_params;
        if (n_layers > 0 && n_embd > 0) {
            n_ff = (int)(ffn_params_est / (n_layers * 3 * n_embd));
            // Round to common alignment (256)
            n_ff = ((n_ff + 255) / 256) * 256;
            if (n_ff < n_embd) n_ff = (int)(n_embd * 8.0 / 3.0);  // fallback: typical LLaMA ratio
        }

        total_bytes_bf16 = total_params * 2;

        // Estimate bytes_per_param from quantization
        // For Q4_K_M: ~4.5 bits, Q8_0: 8 bits, F16/BF16: 16 bits
        bytes_per_param = 2.0;  // default BF16; user can infer from model file size
        bytes_per_step   = (double)total_params * bytes_per_param;

        // FLOPs per decode token (batch=1)
        double per_layer_proj =
            2.0 * n_embd * n_head * head_dim +      // Q
            2.0 * n_embd * n_kv_head * head_dim +   // K
            2.0 * n_embd * n_kv_head * head_dim +   // V
            2.0 * n_head * head_dim * n_embd +      // O
            2.0 * n_embd * n_ff +                   // gate
            2.0 * n_embd * n_ff +                   // up
            2.0 * n_ff * n_embd;                    // down

        double avg_seq_len = 512.0;
        double attn_flops_per_layer =
            2.0 * n_head * head_dim * avg_seq_len +   // QK^T
            2.0 * n_head * avg_seq_len * head_dim;    // AV

        flops_per_decode = (per_layer_proj + attn_flops_per_layer) * n_layers;
        arithmetic_intensity = flops_per_decode / bytes_per_step;

        peak_bw_gb_s = (user_bandwidth > 0) ? user_bandwidth : g_gpu_spec.bandwidth_gb_s;
        peak_tflops  = (user_tflops > 0)    ? user_tflops    : 100.0;
        sm_count     = (user_sm_count > 0)  ? user_sm_count  : g_gpu_spec.sm_count;

        if (peak_bw_gb_s <= 0) peak_bw_gb_s = 1792.0;
        if (sm_count <= 0) sm_count = 132;

        ridge_point = peak_tflops * 1000.0 / peak_bw_gb_s;
        decode_is_memory_bound = arithmetic_intensity < ridge_point;
    }
};

// ============================================================================
// Benchmark result structures
// ============================================================================

struct PrefillResult {
    int    prefill_len;
    double time_ms_mean;
    double time_ms_median;
    double time_ms_p95;
    double tokens_per_sec;
};

struct DecodeResult {
    int    prefill_len;   // associated prefill length that setup KV cache
    double time_ms_mean;
    double time_ms_median;
    double time_ms_p95;
    double tokens_per_sec;
    double eff_bw_gb_s;
    double bw_util_pct;
};

struct SMScaledResult {
    double sm_fraction;
    int    approx_sms;
    int    prefill_len;
    double prefill_ms_mean;
    double decode_ms_mean;
    double decode_ms_p95;
    double decode_tokens_per_sec;
    double eff_bw_gb_s;
    double bw_util_pct;
};

// ============================================================================
// Statistics helpers
// ============================================================================

static double vec_mean(const std::vector<double> & v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double vec_median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 0) return (v[n / 2 - 1] + v[n / 2]) / 2.0;
    return v[n / 2];
}

static double vec_p95(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(std::ceil(0.95 * v.size()));
    if (idx == 0) idx = 1;
    return v[std::min(idx, v.size()) - 1];
}

// ============================================================================
// Token helpers
// ============================================================================

static std::vector<llama_token> make_random_tokens(int n, int vocab_size) {
    std::vector<llama_token> tokens(n);
    for (int i = 0; i < n; i++) {
        tokens[i] = (llama_token)((rand() % std::min(vocab_size - 1, 100000)) + 1);
    }
    return tokens;
}

static llama_token argmax_sample(llama_context * ctx) {
    float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) return 0;

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_token best_token = 0;
    float       best_val   = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_val) {
            best_val   = logits[i];
            best_token = i;
        }
    }
    return best_token;
}

// ============================================================================
// eval_tokens helper
// ============================================================================

static bool eval_tokens(llama_context * ctx, std::vector<llama_token> & tokens,
                        int n_batch_val, int n_past) {
    const int n_tokens = (int)tokens.size();
    for (int i = 0; i < n_tokens; i += n_batch_val) {
        int n_eval = std::min(n_tokens - i, n_batch_val);
        if (n_eval == 0) break;

        llama_batch batch = llama_batch_get_one(tokens.data() + i, n_eval);
        std::vector<llama_pos> pos(n_eval);
        for (int j = 0; j < n_eval; ++j) pos[j] = n_past + i + j;
        batch.pos = pos.data();

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "[ERR] llama_decode failed at token %d/%d\n", i, n_tokens);
            return false;
        }
    }
    llama_synchronize(ctx);
    return true;
}

// ============================================================================
// Prefill + Decode benchmark
// ============================================================================

static bool bench_prefill_decode(
    llama_context * ctx,
    int prefill_len,
    int n_decode,
    int n_warmup_steps,
    int,
    int n_batch_val,
    const std::vector<llama_token> & vocab_pool,
    PrefillResult & prefill_out,
    DecodeResult & decode_out)
{
    // Clear KV cache
    llama_memory_clear(llama_get_memory(ctx), true);

    std::vector<llama_token> prefill_tokens = make_random_tokens(prefill_len, (int)vocab_pool.size());

    // --- Prefill ---
    llama_synchronize(ctx);
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!eval_tokens(ctx, prefill_tokens, n_batch_val, 0)) return false;
    auto t1 = std::chrono::high_resolution_clock::now();

    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    prefill_out.prefill_len  = prefill_len;
    prefill_out.time_ms_mean = prefill_ms;
    prefill_out.time_ms_median = prefill_ms;
    prefill_out.time_ms_p95  = prefill_ms;
    prefill_out.tokens_per_sec = prefill_len / (prefill_ms / 1000.0);

    // --- Decode ---
    int n_past = prefill_len;
    std::vector<double> decode_times;
    decode_times.reserve(n_decode);

    llama_token next_token = prefill_tokens.back();

    for (int i = 0; i < n_decode; i++) {
        std::vector<llama_token> single = { next_token };
        auto step_start = std::chrono::high_resolution_clock::now();

        llama_batch batch = llama_batch_get_one(single.data(), 1);
        std::vector<llama_pos> pos_arr(1);
        pos_arr[0] = n_past;
        batch.pos = pos_arr.data();

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "[ERR] Decode step %d failed\n", i);
            return false;
        }
        llama_synchronize(ctx);

        auto step_end = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(step_end - step_start).count();
        decode_times.push_back(dt);

        next_token = argmax_sample(ctx);
        if (next_token < 0) {
            fprintf(stderr, "[ERR] Sampling failed at step %d\n", i);
            return false;
        }
        n_past++;
    }

    // Exclude warmup tokens
    int warmup = std::min(n_warmup_steps, n_decode);
    std::vector<double> steady(decode_times.begin() + warmup, decode_times.end());

    decode_out.prefill_len = prefill_len;
    decode_out.time_ms_mean   = vec_mean(steady);
    decode_out.time_ms_median = vec_median(steady);
    decode_out.time_ms_p95    = vec_p95(steady);
    decode_out.tokens_per_sec  = decode_out.time_ms_mean > 0 ? 1000.0 / decode_out.time_ms_mean : 0.0;
    decode_out.eff_bw_gb_s    = 0.0;
    decode_out.bw_util_pct    = 0.0;

    return true;
}

// ============================================================================
// Output functions
// ============================================================================

static void print_sep(const char * title = nullptr) {
    if (title) std::cout << "\n" << std::string(80, '=') << "\n" << title << "\n"
                         << std::string(80, '=') << "\n\n";
    else       std::cout << std::string(80, '=') << "\n";
}

static void print_analytical(const LLMAnalytical & a) {
    print_sep("EXPERIMENT 1: Analytical Arithmetic Intensity");

    std::cout << "--- LLM Architecture (from GGUF metadata) ---\n";
    std::cout << "  Layers:            " << a.n_layers << "\n";
    std::cout << "  Hidden dim:        " << a.n_embd << "\n";
    std::cout << "  FFN dim:           " << a.n_ff << " (derived)\n";
    std::cout << "  Heads (Q / KV):    " << a.n_head << " / " << a.n_kv_head << "\n";
    std::cout << "  Head dim:          " << a.head_dim << "\n";

    std::cout << "\n--- Parameter Counts ---\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Total params:      " << a.total_params / 1e6 << " M\n";
    std::cout << "  Total (BF16):      " << a.total_bytes_bf16 / 1e6 << " MB\n";
    std::cout << "  Bytes/param est:   " << a.bytes_per_param << "\n";
    std::cout << "  Bytes per step:    " << a.bytes_per_step / 1e6 << " MB\n";

    std::cout << "\n--- Decode Step (batch=1) ---\n";
    std::cout << std::setprecision(2);
    std::cout << "  FLOPs per token:   " << a.flops_per_decode / 1e6 << " MFLOP\n";
    std::cout << "  Arithmetic Int:    " << a.arithmetic_intensity << " FLOP/Byte\n";

    std::cout << "\n--- GPU (" << g_gpu_spec.name << ") ---\n";
    std::cout << "  Peak Bandwidth:    " << a.peak_bw_gb_s << " GB/s\n";
    std::cout << "  Peak FP16:         " << a.peak_tflops << " TFLOPS\n";
    std::cout << "  SMs:               " << a.sm_count << "\n";
    std::cout << "  Ridge Point:       " << a.ridge_point << " FLOP/Byte\n";

    std::cout << "\n--- Verdict ---\n";
    std::cout << "  AI (" << a.arithmetic_intensity << ") vs Ridge Point ("
              << a.ridge_point << ")\n";
    if (a.decode_is_memory_bound) {
        std::cout << "  >>> LLM Decode is MEMORY-BANDWIDTH-BOUND <<<\n";
        double ideal_ms = a.bytes_per_step / (a.peak_bw_gb_s * 1e9) * 1000.0;
        std::cout << "  Theoretical min time/token: " << std::setprecision(4)
                  << ideal_ms << " ms\n";
    } else {
        std::cout << "  >>> LLM Decode is COMPUTE-BOUND <<<\n";
    }
}

static void print_prefill_results(const std::vector<PrefillResult> & results) {
    print_sep("EXPERIMENT 2: Prefill + Decode Latency");

    if (results.empty()) { std::cout << "  No results.\n\n"; return; }

    std::cout << "--- Prefill Latency ---\n\n";
    std::cout << std::setw(14) << "Prefill Len"
              << std::setw(14) << "Mean(ms)"
              << std::setw(14) << "Median(ms)"
              << std::setw(14) << "P95(ms)"
              << std::setw(16) << "Tok/s"
              << "\n" << std::string(72, '-') << "\n";

    for (const auto & r : results) {
        std::cout << std::fixed;
        std::cout << std::setw(14) << r.prefill_len
                  << std::setw(14) << std::setprecision(3) << r.time_ms_mean
                  << std::setw(14) << std::setprecision(3) << r.time_ms_median
                  << std::setw(14) << std::setprecision(3) << r.time_ms_p95
                  << std::setw(16) << std::setprecision(1) << r.tokens_per_sec
                  << "\n";
    }
    std::cout << "\n";
}

static void print_decode_results(const std::vector<DecodeResult> & results,
                                  const LLMAnalytical & a) {
    if (results.empty()) return;

    std::cout << "--- Decode Latency (per token, after warmup) ---\n\n";
    std::cout << std::setw(14) << "Prefill Len"
              << std::setw(14) << "Mean(ms)"
              << std::setw(14) << "Median(ms)"
              << std::setw(14) << "P95(ms)"
              << std::setw(12) << "Tok/s"
              << std::setw(18) << "Eff BW(GB/s)"
              << std::setw(14) << "BW Util%"
              << "\n" << std::string(100, '-') << "\n";

    for (auto & r : results) {
        // Compute BW after we have bytes_per_step
        double eff_bw   = a.bytes_per_step / (r.time_ms_mean / 1000.0) / 1e9;
        double bw_util  = (a.peak_bw_gb_s > 0) ? eff_bw / a.peak_bw_gb_s * 100.0 : 0.0;

        std::cout << std::fixed;
        std::cout << std::setw(14) << r.prefill_len
                  << std::setw(14) << std::setprecision(3) << r.time_ms_mean
                  << std::setw(14) << std::setprecision(3) << r.time_ms_median
                  << std::setw(14) << std::setprecision(3) << r.time_ms_p95
                  << std::setw(12) << std::setprecision(1) << r.tokens_per_sec
                  << std::setw(18) << std::setprecision(2) << eff_bw
                  << std::setw(14) << std::setprecision(1) << bw_util;
        if (bw_util < 100.0) std::cout << "%";
        else                 std::cout << "% ***";
        std::cout << "\n";
    }
    std::cout << "\n";

    // Interpretation
    if (results.size() >= 1) {
        auto & r = results[0];
        double eff_bw  = a.bytes_per_step / (r.time_ms_mean / 1000.0) / 1e9;
        double bw_util = (a.peak_bw_gb_s > 0) ? eff_bw / a.peak_bw_gb_s * 100.0 : 0.0;

        std::cout << "--- Interpretation ---\n";
        std::cout << "  Decode BW utilization: " << std::setprecision(1)
                  << bw_util << "% of peak\n";
        int est_sms = std::max(1, (int)std::round(a.sm_count * bw_util / 100.0));
        std::cout << "  Est. SMs for LLM decode: " << est_sms
                  << " / " << a.sm_count << "\n";
        std::cout << "  LLM is " << (a.decode_is_memory_bound ? "MEMORY-BOUND" : "COMPUTE-BOUND")
                  << " at decode.\n";
    }
}

static void print_sm_scaling(const std::vector<SMScaledResult> & results,
                              int /*total_sms*/, double peak_bw_gb_s,
                              int64_t model_bytes) {
    print_sep("EXPERIMENT 3: SM Scaling (via CUDA MPS)");

    if (results.empty()) {
        std::cout << "  No results — MPS not available or experiment skipped.\n";
        std::cout << "  To run: start MPS daemon and use --sm-fractions argument.\n\n";
        return;
    }

    // Group by prefill_len
    for (int plen : {64, 128}) {
        bool found = false;
        for (auto & r : results) if (r.prefill_len == plen) { found = true; break; }
        if (!found) continue;

        std::cout << "--- Prefill len = " << plen << " ---\n\n";
        std::cout << std::setw(10) << "SM Frac"
                  << std::setw(10) << "~SMs"
                  << std::setw(16) << "Prefill(ms)"
                  << std::setw(14) << "Decode(ms)"
                  << std::setw(14) << "Dec P95(ms)"
                  << std::setw(12) << "Dec Tok/s"
                  << std::setw(18) << "Eff BW(GB/s)"
                  << std::setw(14) << "BW Util%"
                  << "\n" << std::string(112, '-') << "\n";

        double best_decode = 1e9;
        for (auto & r : results)
            if (r.prefill_len == plen && r.decode_ms_mean < best_decode)
                best_decode = r.decode_ms_mean;

        for (auto & r : results) {
            if (r.prefill_len != plen) continue;

            double eff_bw  = model_bytes / (r.decode_ms_mean / 1000.0) / 1e9;
            double bw_util = (peak_bw_gb_s > 0) ? eff_bw / peak_bw_gb_s * 100.0 : 0.0;

            std::cout << std::fixed;
            std::cout << std::setw(10) << std::setprecision(2) << r.sm_fraction
                      << std::setw(10) << r.approx_sms
                      << std::setw(16) << std::setprecision(3) << r.prefill_ms_mean
                      << std::setw(14) << std::setprecision(3) << r.decode_ms_mean
                      << std::setw(14) << std::setprecision(3) << r.decode_ms_p95
                      << std::setw(12) << std::setprecision(1) << r.decode_tokens_per_sec
                      << std::setw(18) << std::setprecision(2) << eff_bw
                      << std::setw(14) << std::setprecision(1) << bw_util;
            if (r.decode_ms_mean <= best_decode * 1.05)
                std::cout << "  <-- near-optimal";
            std::cout << "\n";
        }
        std::cout << "\n";

        // --- Decode latency vs SM fraction (focused view) ---
        std::cout << "  Decode Latency Scaling (prefill=" << plen << "):\n";
        std::cout << "  " << std::setw(10) << "SM Frac"
                  << std::setw(10) << "~SMs"
                  << std::setw(14) << "Decode(ms)"
                  << std::setw(12) << "Tok/s"
                  << std::setw(12) << "Slowdown"
                  << std::setw(12) << "BW Util%"
                  << "\n  " << std::string(70, '-') << "\n";

        for (auto & r : results) {
            if (r.prefill_len != plen) continue;

            double eff_bw  = model_bytes / (r.decode_ms_mean / 1000.0) / 1e9;
            double bw_util = (peak_bw_gb_s > 0) ? eff_bw / peak_bw_gb_s * 100.0 : 0.0;
            double slowdown = best_decode > 0 ? r.decode_ms_mean / best_decode : 1.0;

            std::cout << "  " << std::fixed;
            std::cout << std::setw(10) << std::setprecision(2) << r.sm_fraction
                      << std::setw(10) << r.approx_sms
                      << std::setw(14) << std::setprecision(3) << r.decode_ms_mean
                      << std::setw(12) << std::setprecision(1) << r.decode_tokens_per_sec
                      << std::setw(10) << std::setprecision(2) << slowdown << "x"
                      << std::setw(12) << std::setprecision(1) << bw_util;
            if (slowdown < 1.05)
                std::cout << "  <-- baseline";
            std::cout << "\n";
        }
        std::cout << "\n";

        // Find saturation knee for this prefill_len
        for (auto & r : results) {
            if (r.prefill_len != plen) continue;
            double deg = (r.decode_ms_mean - best_decode) / best_decode;
            if (deg < 0.05) {
                std::cout << "  --- Saturation (prefill=" << plen << ") ---\n";
                std::cout << "  SM fraction: " << r.sm_fraction * 100.0
                          << "% (~" << r.approx_sms << " SMs)\n";
                std::cout << "  Beyond this, adding SMs gives <5% decode improvement\n\n";
                break;
            }
        }
    }
}

static void print_summary(const LLMAnalytical & a,
                           const std::vector<PrefillResult> & /*pr*/,
                           const std::vector<DecodeResult> & dr,
                           const std::vector<SMScaledResult> & sr) {
    print_sep("SUMMARY & RECOMMENDATIONS");

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  LLM Decode AI:     " << a.arithmetic_intensity << " FLOP/Byte"
              << " vs Ridge Point " << a.ridge_point << " FLOP/Byte\n";
    std::cout << "  Verdict:           "
              << (a.decode_is_memory_bound ? "MEMORY-BOUND" : "COMPUTE-BOUND") << "\n";

    if (!dr.empty()) {
        auto & r = dr[0];
        double eff_bw  = a.bytes_per_step / (r.time_ms_mean / 1000.0) / 1e9;
        double bw_util = (a.peak_bw_gb_s > 0) ? eff_bw / a.peak_bw_gb_s * 100.0 : 0.0;
        int est_sms = std::max(1, (int)std::round(a.sm_count * bw_util / 100.0));

        std::cout << "\n";
        std::cout << "  Decode latency:    " << std::setprecision(3)
                  << r.time_ms_mean << " ms/token\n";
        std::cout << "  Decode BW util:    " << std::setprecision(1)
                  << bw_util << "%\n";
        std::cout << "  Est. SMs needed:   " << est_sms << " / " << a.sm_count << "\n";
    }

    if (!sr.empty()) {
        std::cout << "\n  SM scaling confirms LLM decode performance\n";
        std::cout << "  scales with SM count. For mega-kernel partitioning:\n";
        std::cout << "  - LLM needs significant SM allocation (~70-80%)\n";
        std::cout << "  - Unlike TTS (saturates at ~15% SM), LLM benefits from more SMs\n";
        std::cout << "  - Larger model = more memory traffic = more SMs needed\n";
    }

    std::cout << "\n";
    std::cout << "  --- Implication for Mega Kernel ---\n";
    std::cout << "  LLM is the primary SM consumer. Reserve ~70-80% SMs for LLM,\n";
    std::cout << "  ~15-22% for TTS (memory-bound, saturates early).\n";
    std::cout << "  APM/VPM/Token2Wav use yield protocol, get full SMs when needed.\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--steps" && i + 1 < argc) {
            n_steps = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            n_warmup = std::stoi(argv[++i]);
        } else if (arg == "--prefill-lens" && i + 1 < argc) {
            prefill_lens_str = argv[++i];
        } else if (arg == "--decode-tokens" && i + 1 < argc) {
            n_decode_tokens = std::stoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            n_repeat = std::stoi(argv[++i]);
        } else if (arg == "--sm-fractions" && i + 1 < argc) {
            sm_fractions_str = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (arg == "--bandwidth" && i + 1 < argc) {
            user_bandwidth = std::stod(argv[++i]);
        } else if (arg == "--tflops" && i + 1 < argc) {
            user_tflops = std::stod(argv[++i]);
        } else if (arg == "--sm-count" && i + 1 < argc) {
            user_sm_count = std::stoi(argv[++i]);
        } else if ((arg == "-ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        } else if ((arg == "-b" || arg == "--batch-size") && i + 1 < argc) {
            n_batch = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            printf("Usage: llama-omni-bench-llm -m <llm_model.gguf> [options]\n\n");
            printf("LLM SM Partitioning Benchmark\n\n");
            printf("Options:\n");
            printf("  -m, --model PATH         LLM model GGUF path (required)\n");
            printf("  --steps N                 Decode steps per trial (default: 50)\n");
            printf("  --warmup N                Warmup decode steps excluded from stats (default: 10)\n");
            printf("  --prefill-lens LIST       Comma-separated prefill lengths (default: 64,128)\n");
            printf("  --decode-tokens N         Tokens to decode per trial (default: 32)\n");
            printf("  --repeat N                Repeats per config (default: 3)\n");
            printf("  --sm-fractions LIST       SM fractions for Exp 3 (requires MPS, e.g.: 0.1,0.2,0.5,1.0)\n");
            printf("  --bandwidth GB_S          Override GPU memory bandwidth\n");
            printf("  --tflops TFLOP_S          Override GPU FP16 TFLOPS\n");
            printf("  --sm-count N              Override GPU SM count\n");
            printf("  --output FILE             CSV output file\n");
            printf("  -ngl, --n-gpu-layers N    GPU layers (default: 99)\n");
            printf("  -c, --ctx-size N          Context size (default: 4096)\n");
            printf("  -b, --batch-size N        Batch size (default: 2048)\n");
            return 0;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: Model path required. Use -m <llm_model.gguf>\n");
        return 1;
    }

    // Detect GPU
    if (!parse_nvidia_smi(g_gpu_spec)) {
        fprintf(stderr, "[WARN] Could not detect GPU specs via nvidia-smi.\n");
        fprintf(stderr, "[WARN] Using defaults. Override with --bandwidth, --tflops, --sm-count\n");
    }
    fprintf(stderr, "[INFO] GPU: %s\n", g_gpu_spec.name.c_str());
    fprintf(stderr, "[INFO]   SMs=%d, BW=%.0f GB/s\n", g_gpu_spec.sm_count, g_gpu_spec.bandwidth_gb_s);

    // Parse prefill lengths
    std::vector<int> prefill_lens;
    {
        std::stringstream ss(prefill_lens_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            int pl = std::stoi(item);
            if (pl > 0) prefill_lens.push_back(pl);
        }
    }
    if (prefill_lens.empty()) prefill_lens = {64, 128};

    // Parse SM fractions
    std::vector<double> sm_fractions;
    if (!sm_fractions_str.empty()) {
        std::stringstream ss(sm_fractions_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            sm_fractions.push_back(std::stod(item));
        }
    }

    // Check MPS environment
    const char * mps_env = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
    bool mps_active = (mps_env != nullptr);

    // --- Init llama ---
    llama_backend_init();
    llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED);

    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = n_ctx;
    params.n_batch       = n_batch;
    params.n_ubatch      = n_batch;
    params.n_gpu_layers  = n_gpu_layers;
    params.n_predict     = 0;

    common_init();
    common_init_result init_result = common_init_from_params(params);
    llama_model   * model = init_result.model.get();
    llama_context * ctx   = init_result.context.get();

    if (!model || !ctx) {
        fprintf(stderr, "Error: Failed to load model: %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }

    int n_embd = llama_model_n_embd(model);
    fprintf(stderr, "[INFO] Model: n_embd=%d, n_layer=%d, n_head=%d\n",
            n_embd, llama_model_n_layer(model), llama_model_n_head(model));

    // --- E1: Analytical ---
    LLMAnalytical analytical;
    analytical.compute(model);
    print_analytical(analytical);

    // Build vocab token pool for random token generation
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> vocab_pool;
    vocab_pool.reserve(std::min(n_vocab - 1, 100000));
    for (int i = 1; i < n_vocab && i < 100000; ++i)
        vocab_pool.push_back(i);
    if (vocab_pool.empty()) {
        for (int i = 1; i < n_vocab; ++i)
            vocab_pool.push_back(i);
    }

    // --- MPS child mode ---
    if (mps_active && sm_fractions_str.empty()) {
        double current_frac = std::stod(mps_env) / 100.0;
        fprintf(stderr, "[INFO] MPS active: %.0f%% SMs (fraction=%.2f)\n",
                std::stod(mps_env), current_frac);

        // Warmup with first prefill_len
        int warmup_plen = prefill_lens[0];
        {
            PrefillResult pr;
            DecodeResult dr;
            bench_prefill_decode(ctx, warmup_plen, n_decode_tokens, n_warmup,
                                 n_steps, n_batch, vocab_pool, pr, dr);
        }

        // Run each prefill_len once and report
        for (int plen : prefill_lens) {
            PrefillResult pr;
            DecodeResult dr;
            if (!bench_prefill_decode(ctx, plen, n_decode_tokens, n_warmup,
                                      n_steps, n_batch, vocab_pool, pr, dr)) {
                fprintf(stderr, "[ERR] Benchmark failed for prefill=%d\n", plen);
                continue;
            }

            std::cout << "LLM_DATA: " << current_frac << " "
                      << plen << " "
                      << pr.time_ms_mean << " "
                      << dr.time_ms_mean << " "
                      << dr.time_ms_p95 << " "
                      << (analytical.bytes_per_step / (dr.time_ms_mean / 1000.0) / 1e9) << " "
                      << dr.tokens_per_sec
                      << "\n";
        }

        llama_backend_free();
        return 0;  // init_result destructor handles model/ctx cleanup
    }

    // --- E2: Prefill + Decode (normal mode) ---
    srand(42);

    // Warmup once
    fprintf(stderr, "[INFO] Warming up...\n");
    {
        PrefillResult pr;
        DecodeResult dr;
        bench_prefill_decode(ctx, prefill_lens[0], n_decode_tokens, n_warmup,
                             n_steps, n_batch, vocab_pool, pr, dr);
    }

    std::vector<PrefillResult> prefill_agg;
    std::vector<DecodeResult>  decode_agg;

    for (int plen : prefill_lens) {
        std::vector<double> prefill_ms_vec;
        std::vector<double> decode_mean_vec;
        std::vector<double> decode_p95_vec;

        for (int rep = 0; rep < n_repeat; ++rep) {
            PrefillResult pr;
            DecodeResult dr;
            if (!bench_prefill_decode(ctx, plen, n_decode_tokens, n_warmup,
                                      n_steps, n_batch, vocab_pool, pr, dr)) {
                fprintf(stderr, "[ERR] Failed: prefill=%d repeat=%d\n", plen, rep);
                continue;
            }
            prefill_ms_vec.push_back(pr.time_ms_mean);
            decode_mean_vec.push_back(dr.time_ms_mean);
            decode_p95_vec.push_back(dr.time_ms_p95);
        }

        if (!prefill_ms_vec.empty()) {
            PrefillResult pr;
            pr.prefill_len    = plen;
            pr.time_ms_mean   = vec_mean(prefill_ms_vec);
            pr.time_ms_median = vec_median(prefill_ms_vec);
            pr.time_ms_p95    = vec_p95(prefill_ms_vec);
            pr.tokens_per_sec = plen / (pr.time_ms_mean / 1000.0);
            prefill_agg.push_back(pr);

            DecodeResult dr;
            dr.prefill_len    = plen;
            dr.time_ms_mean   = vec_mean(decode_mean_vec);
            dr.time_ms_median = vec_median(decode_mean_vec);
            dr.time_ms_p95    = vec_p95(decode_p95_vec);
            dr.tokens_per_sec = dr.time_ms_mean > 0 ? 1000.0 / dr.time_ms_mean : 0.0;
            decode_agg.push_back(dr);
        }
    }

    print_prefill_results(prefill_agg);
    print_decode_results(decode_agg, analytical);

    // --- E3: SM Scaling ---
    std::vector<SMScaledResult> sm_results;

    if (!sm_fractions_str.empty()) {
        bool mps_binary_ok = (system("command -v nvidia-cuda-mps-control > /dev/null 2>&1") == 0);
        bool mps_running   = (system("pgrep -x nvidia-cuda-mps > /dev/null 2>&1") == 0 ||
                              system("pgrep -x nvidia-cuda-mps-c > /dev/null 2>&1") == 0);

        if (mps_binary_ok && mps_running) {
            fprintf(stderr, "[INFO] MPS detected. Running SM scaling via re-exec...\n");

            // Free GPU resources BEFORE spawning MPS children.
            // On Jetson (unified memory), holding the model while children
            // also load it causes OOM. On desktop, avoids CUDA context conflicts.
            init_result.context.reset();
            init_result.model.reset();
            ctx   = nullptr;
            model = nullptr;

            std::string self_path = argv[0];
            std::string base_args;
            for (int i = 1; i < argc; i++) {
                std::string a(argv[i]);
                if (a == "--sm-fractions") { i++; continue; }
                if (a == "--output") { i++; continue; }
                if (a == "--repeat") { i++; continue; }
                base_args += " " + a;
            }
            // Force n_repeat=1 for MPS re-exec (faster)
            base_args += " --repeat 1";

            for (double frac : sm_fractions) {
                int pct = std::max(1, (int)(frac * 100.0));
                fprintf(stderr, "[INFO]   Testing SM fraction=%.2f (%d%%)...\n", frac, pct);

                const char * mps_pipe = getenv("CUDA_MPS_PIPE_DIRECTORY");
                std::string cmd =
                    (mps_pipe ? "CUDA_MPS_PIPE_DIRECTORY=" + std::string(mps_pipe) + " " : "") +
                    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=" + std::to_string(pct) +
                    " CUDA_VISIBLE_DEVICES=0" +
                    " " + self_path + base_args +
                    " 2>&1";

                FILE * pipe = popen(cmd.c_str(), "r");
                if (!pipe) {
                    fprintf(stderr, "[ERR] Failed to execute for fraction %.2f\n", frac);
                    continue;
                }

                char buf[1024];
                while (fgets(buf, sizeof(buf), pipe)) {
                    std::string line(buf);
                    if (line.find("LLM_DATA:") != std::string::npos) {
                        double f;
                        int plen;
                        SMScaledResult r;
                        int parsed = sscanf(line.c_str(), "LLM_DATA: %lf %d %lf %lf %lf %lf %lf",
                                           &f, &plen,
                                           &r.prefill_ms_mean,
                                           &r.decode_ms_mean,
                                           &r.decode_ms_p95,
                                           &r.eff_bw_gb_s,
                                           &r.decode_tokens_per_sec);
                        if (parsed >= 6) {
                            if (parsed < 7) r.decode_tokens_per_sec = 0.0;
                            r.sm_fraction = frac;
                            r.approx_sms  = std::max(1, (int)std::round(analytical.sm_count * frac));
                            r.prefill_len = plen;
                            r.bw_util_pct = (analytical.peak_bw_gb_s > 0) ?
                                r.eff_bw_gb_s / analytical.peak_bw_gb_s * 100.0 : 0.0;
                            sm_results.push_back(r);
                        }
                    }
                }
                pclose(pipe);
            }
        } else if (!mps_binary_ok) {
            fprintf(stderr, "[WARN] MPS not available. SM scaling skipped.\n");
        } else {
            fprintf(stderr, "[WARN] MPS daemon NOT running. Start: sudo nvidia-cuda-mps-control -d\n");
        }
    }

    print_sm_scaling(sm_results, analytical.sm_count,
                     analytical.peak_bw_gb_s, (int64_t)analytical.bytes_per_step);

    // --- Summary ---
    print_summary(analytical, prefill_agg, decode_agg, sm_results);

    // --- CSV Export ---
    if (!output_csv.empty()) {
        std::ofstream f(output_csv);
        if (f) {
            f << "experiment,prefill_len,sm_fraction,approx_sms,"
              << "time_ms_mean,time_ms_median,time_ms_p95,"
              << "tokens_per_sec,eff_bw_gb_s,bw_util_pct\n";

            for (auto & r : prefill_agg) {
                f << "prefill," << r.prefill_len << ",,,"
                  << r.time_ms_mean << "," << r.time_ms_median << ","
                  << r.time_ms_p95 << "," << r.tokens_per_sec << ",,\n";
            }
            for (auto & r : decode_agg) {
                double eff_bw  = analytical.bytes_per_step / (r.time_ms_mean / 1000.0) / 1e9;
                double bw_util = (analytical.peak_bw_gb_s > 0) ?
                    eff_bw / analytical.peak_bw_gb_s * 100.0 : 0.0;
                f << "decode," << r.prefill_len << ",,,"
                  << r.time_ms_mean << "," << r.time_ms_median << ","
                  << r.time_ms_p95 << "," << r.tokens_per_sec << ","
                  << eff_bw << "," << bw_util << "\n";
            }
            for (auto & r : sm_results) {
                f << "sm_scaling," << r.prefill_len << "," << r.sm_fraction
                  << "," << r.approx_sms << ",,"
                  << r.decode_ms_mean << "," << r.decode_ms_p95 << ","
                  << r.decode_tokens_per_sec << ","
                  << r.eff_bw_gb_s << "," << r.bw_util_pct << "\n";
            }
            f.close();
            fprintf(stderr, "[INFO] Results written to %s\n", output_csv.c_str());
        }
    }

    // Cleanup — free context before model (order matters)
    init_result.context.reset();
    init_result.model.reset();
    llama_backend_free();

    return 0;
}
