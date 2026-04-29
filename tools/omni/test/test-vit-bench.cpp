/**
 * VIT (Vision Transformer) 推理性能基准测试
 *
 * 测试 VIT 模型在图片编码流程中的各阶段时延：
 *   - 图片加载 (load)
 *   - 图片预处理 (preprocess): u8 → f32 batch
 *   - 图片编码 (encode): f32 batch → embedding vector
 *   - 端到端 (e2e): load + preprocess + encode
 *
 * 用法:
 *   llama-omni-test-vit-bench -m <vision_model_path> -i <image_path> [options]
 *     --n-threads <n>   线程数 (默认 4)
 *     --no-gpu          禁用 GPU，仅使用 CPU
 *     --repeat <n>      重复次数 (默认 5)
 *     --warmup <n>      预热次数 (默认 2)
 *     --max-slice <n>   最大 slice 数量 (默认 -1，使用模型默认值)
 *     --coreml <path>   CoreML 模型路径 (macOS ANE 加速)
 *     -h, --help        显示帮助
 */

#include "vision.h"
#include "omni.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ==================== 计时工具 ====================

static double elapsed_ms(const std::chrono::high_resolution_clock::time_point & t0,
                         const std::chrono::high_resolution_clock::time_point & t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double median_of(std::vector<double> vals) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    const size_t n = vals.size();
    if (n % 2 == 0) {
        return (vals[n / 2 - 1] + vals[n / 2]) / 2.0;
    }
    return vals[n / 2];
}

static double p95_of(std::vector<double> vals) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    const size_t idx = static_cast<size_t>(std::ceil(0.95 * vals.size())) - 1;
    return vals[std::min(idx, vals.size() - 1)];
}

static double min_of(const std::vector<double> & vals) {
    if (vals.empty()) return 0.0;
    return *std::min_element(vals.begin(), vals.end());
}

static double max_of(const std::vector<double> & vals) {
    if (vals.empty()) return 0.0;
    return *std::max_element(vals.begin(), vals.end());
}

// ==================== 单次 VIT 推理 ====================

struct VitBenchResult {
    double load_ms;        // 图片加载耗时
    double preprocess_ms;  // 预处理耗时
    double encode_ms;      // 编码耗时
    double e2e_ms;         // 端到端耗时
    int    n_chunks;       // chunk 数量 (预处理后)
    int    n_tokens;       // 总 token 数量
    int    n_embd;         // embedding 维度
};

static bool run_single_vit_bench(vision_ctx *              ctx,
                                 int                       n_threads,
                                 const std::string &       image_path,
                                 VitBenchResult &          result) {
    auto t_e2e_start = std::chrono::high_resolution_clock::now();

    // 1. 加载图片
    auto t_load_start = std::chrono::high_resolution_clock::now();

    FILE * fp = fopen(image_path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open image file: %s\n", image_path.c_str());
        return false;
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    std::vector<unsigned char> image_bytes(file_size);
    size_t n_read = fread(image_bytes.data(), 1, file_size, fp);
    fclose(fp);
    if (n_read != static_cast<size_t>(file_size)) {
        fprintf(stderr, "Error: failed to read entire image file\n");
        return false;
    }

    int nx, ny, nc;
    unsigned char * img_data = stbi_load_from_memory(image_bytes.data(), (int) file_size, &nx, &ny, &nc, 3);
    if (!img_data) {
        fprintf(stderr, "Error: failed to decode image: %s\n", image_path.c_str());
        return false;
    }

    vision_image_u8 * img = vision_image_u8_init();
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    std::memcpy(img->buf.data(), img_data, 3 * nx * ny);
    stbi_image_free(img_data);

    auto t_load_end = std::chrono::high_resolution_clock::now();
    result.load_ms = elapsed_ms(t_load_start, t_load_end);

    // 2. 预处理: u8 → f32 batch
    auto t_preprocess_start = std::chrono::high_resolution_clock::now();

    vision_image_f32_batch batch;
    batch.entries.clear();
    if (!vision_image_preprocess(ctx, img, &batch)) {
        fprintf(stderr, "Error: vision_image_preprocess failed\n");
        vision_image_u8_free(img);
        return false;
    }
    vision_image_u8_free(img);

    auto t_preprocess_end = std::chrono::high_resolution_clock::now();
    result.preprocess_ms = elapsed_ms(t_preprocess_start, t_preprocess_end);
    result.n_chunks = static_cast<int>(batch.entries.size());

    // 3. 编码: f32 batch → embedding
    const int n_embd   = vision_n_mmproj_embd(ctx);
    const int n_tokens = vision_n_output_tokens(ctx);
    result.n_embd      = n_embd;
    result.n_tokens    = result.n_chunks * n_tokens;

    std::vector<float> embedding(n_embd * n_tokens * result.n_chunks);

    auto t_encode_start = std::chrono::high_resolution_clock::now();

    // 逐 chunk 编码 (与 omni.cpp 中 encode_image_with_vision_chunks 一致)
    for (size_t i = 0; i < batch.entries.size(); i++) {
        float * chunk_ptr = embedding.data() + i * n_embd * n_tokens;
        if (!vision_image_encode(ctx, n_threads, batch.entries[i].get(), chunk_ptr)) {
            fprintf(stderr, "Error: vision_image_encode failed for chunk %zu\n", i);
            return false;
        }
    }

    auto t_encode_end = std::chrono::high_resolution_clock::now();
    result.encode_ms = elapsed_ms(t_encode_start, t_encode_end);

    auto t_e2e_end = std::chrono::high_resolution_clock::now();
    result.e2e_ms = elapsed_ms(t_e2e_start, t_e2e_end);

    return true;
}

// ==================== 输出格式化 ====================

static void print_header() {
    printf("\n");
    printf("=============================================================================================================\n");
    printf("  VIT (Vision Transformer) Benchmark Results\n");
    printf("=============================================================================================================\n");
    printf("  %-10s | %-12s | %-12s | %-12s | %-12s | %-8s | %-8s\n",
           "load(ms)", "preproc(ms)", "encode(ms)", "e2e(ms)", "encode/chnk", "chunks", "tokens");
    printf("  -----------|--------------|--------------|--------------|--------------|----------|----------\n");
}

static void print_result(const VitBenchResult & r) {
    double encode_per_chunk = r.n_chunks > 0 ? r.encode_ms / r.n_chunks : 0.0;
    printf("  %10.2f | %10.2f   | %10.2f   | %10.2f   | %10.2f   | %6d   | %6d\n",
           r.load_ms, r.preprocess_ms, r.encode_ms, r.e2e_ms,
           encode_per_chunk, r.n_chunks, r.n_tokens);
}

static void print_footer() {
    printf("=============================================================================================================\n");
    printf("  encode/chnk = encode_ms / n_chunks (per-chunk encode time)\n");
    printf("=============================================================================================================\n\n");
}

// ==================== 帮助信息 ====================

static void show_usage(const char * prog_name) {
    printf(
        "VIT (Vision Transformer) Inference Benchmark\n\n"
        "Usage: %s -m <vision_model_path> -i <image_path> [options]\n\n"
        "Required:\n"
        "  -m <path>             VIT 模型路径 (GGUF 格式)\n"
        "  -i <path>             测试图片路径\n\n"
        "Options:\n"
        "  --n-threads <n>       线程数 (默认: 4)\n"
        "  --no-gpu              禁用 GPU，仅使用 CPU\n"
        "  --repeat <n>          重复次数 (默认: 5)\n"
        "  --warmup <n>          预热次数 (默认: 2)\n"
        "  --max-slice <n>       最大 slice 数量 (默认: -1，使用模型默认值)\n"
        "  --coreml <path>       CoreML 模型路径 (macOS ANE 加速)\n"
        "  --detail              输出每次迭代的详细时延\n"
        "  -h, --help            显示帮助\n\n"
        "Example:\n"
        "  %s -m ./models/MiniCPM-o-4_5-gguf/mmproj-model.gguf -i test.jpg\n"
        "  %s -m ./models/mmproj-model.gguf -i test.jpg --no-gpu --repeat 10\n",
        prog_name, prog_name, prog_name);
}

// ==================== Main ====================

int main(int argc, char ** argv) {
    ggml_time_init();

    std::string vision_model_path;
    std::string image_path;
    std::string coreml_path;
    int         n_threads   = 4;
    bool        use_gpu     = true;
    int         n_repeat    = 5;
    int         n_warmup    = 2;
    int         max_slice   = -1;
    bool        show_detail = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            show_usage(argv[0]);
            return 0;
        } else if (arg == "-m" && i + 1 < argc) {
            vision_model_path = argv[++i];
        } else if (arg == "-i" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--n-threads" && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        } else if (arg == "--no-gpu") {
            use_gpu = false;
        } else if (arg == "--repeat" && i + 1 < argc) {
            n_repeat = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            n_warmup = std::atoi(argv[++i]);
        } else if (arg == "--max-slice" && i + 1 < argc) {
            max_slice = std::atoi(argv[++i]);
        } else if (arg == "--coreml" && i + 1 < argc) {
            coreml_path = argv[++i];
        } else if (arg == "--detail") {
            show_detail = true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            show_usage(argv[0]);
            return 1;
        }
    }

    if (vision_model_path.empty()) {
        fprintf(stderr, "Error: -m <vision_model_path> is required\n\n");
        show_usage(argv[0]);
        return 1;
    }
    if (image_path.empty()) {
        fprintf(stderr, "Error: -i <image_path> is required\n\n");
        show_usage(argv[0]);
        return 1;
    }

    // ==================== 初始化 VIT 模型 ====================
    printf("=== Loading VIT Model ===\n");
    printf("  Model path:  %s\n", vision_model_path.c_str());
    printf("  Image path:  %s\n", image_path.c_str());
    printf("  n_threads:   %d\n", n_threads);
    printf("  use_gpu:     %s\n", use_gpu ? "true" : "false");
    printf("  max_slice:   %d\n", max_slice);
    if (!coreml_path.empty()) {
        printf("  CoreML path: %s\n", coreml_path.c_str());
    }

    vision_context_params ctx_params;
    ctx_params.use_gpu            = use_gpu;
    ctx_params.verbosity          = GGML_LOG_LEVEL_ERROR;
    ctx_params.coreml_model_path  = coreml_path.empty() ? nullptr : coreml_path.c_str();

    vision_ctx * ctx_vision = vision_init(vision_model_path.c_str(), ctx_params);
    if (!ctx_vision) {
        fprintf(stderr, "Error: failed to init vision model from %s\n", vision_model_path.c_str());
        return 1;
    }

    // 设置 max_slice_nums (如果指定)
    if (max_slice >= 1) {
        vision_set_max_slice_nums(ctx_vision, max_slice);
        printf("  max_slice override: %d\n", max_slice);
    }

    const int n_embd   = vision_n_mmproj_embd(ctx_vision);
    const int n_tokens = vision_n_output_tokens(ctx_vision);
    printf("  n_embd:      %d\n", n_embd);
    printf("  n_tokens:    %d (per chunk)\n", n_tokens);
    printf("  Model loaded successfully\n");
    printf("==========================\n");

    // ==================== Warmup ====================
    printf("\n=== Warmup (x%d) ===\n", n_warmup);
    for (int w = 0; w < n_warmup; w++) {
        VitBenchResult warmup_result;
        if (!run_single_vit_bench(ctx_vision, n_threads, image_path, warmup_result)) {
            fprintf(stderr, "Error: warmup iteration %d failed\n", w);
            vision_free(ctx_vision);
            return 1;
        }
        printf("  Warmup %d: e2e=%.2f ms (load=%.2f, preproc=%.2f, encode=%.2f)\n",
               w + 1, warmup_result.e2e_ms, warmup_result.load_ms,
               warmup_result.preprocess_ms, warmup_result.encode_ms);
    }

    // ==================== 运行基准测试 ====================
    printf("\n=== Running VIT Benchmark (repeat=%d) ===\n", n_repeat);

    std::vector<double>         load_vec, preprocess_vec, encode_vec, e2e_vec;
    std::vector<VitBenchResult> all_results;

    print_header();

    for (int rep = 0; rep < n_repeat; rep++) {
        VitBenchResult result;
        if (!run_single_vit_bench(ctx_vision, n_threads, image_path, result)) {
            fprintf(stderr, "Error: benchmark iteration %d failed\n", rep);
            continue;
        }

        load_vec.push_back(result.load_ms);
        preprocess_vec.push_back(result.preprocess_ms);
        encode_vec.push_back(result.encode_ms);
        e2e_vec.push_back(result.e2e_ms);
        all_results.push_back(result);

        if (show_detail) {
            printf("  [rep %2d] ", rep + 1);
            print_result(result);
        }
    }

    if (!show_detail) {
        // 输出中位数结果
        VitBenchResult median_result;
        median_result.load_ms       = median_of(load_vec);
        median_result.preprocess_ms = median_of(preprocess_vec);
        median_result.encode_ms     = median_of(encode_vec);
        median_result.e2e_ms        = median_of(e2e_vec);
        median_result.n_chunks      = all_results.empty() ? 0 : all_results[0].n_chunks;
        median_result.n_tokens      = all_results.empty() ? 0 : all_results[0].n_tokens;
        median_result.n_embd        = all_results.empty() ? 0 : all_results[0].n_embd;

        printf("  [median] ");
        print_result(median_result);
    }

    print_footer();

    // ==================== 汇总统计 ====================
    printf("=== Summary Statistics ===\n");
    printf("  %-12s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n",
           "stage", "min(ms)", "median(ms)", "p95(ms)", "max(ms)", "mean(ms)", "stdev(ms)");
    printf("  -------------|------------|------------|------------|------------|------------|------------\n");

    auto compute_stats = [](const std::vector<double> & vals, const char * name) {
        if (vals.empty()) {
            printf("  %-12s | %10s | %10s | %10s | %10s | %10s | %10s\n",
                   name, "-", "-", "-", "-", "-", "-");
            return;
        }
        double sum = 0.0;
        for (double v : vals) { sum += v; }
        double mean = sum / vals.size();

        double sq_sum = 0.0;
        for (double v : vals) { sq_sum += (v - mean) * (v - mean); }
        double stdev = std::sqrt(sq_sum / vals.size());

        printf("  %-12s | %8.2f   | %8.2f   | %8.2f   | %8.2f   | %8.2f   | %8.2f\n",
               name, min_of(vals), median_of(vals), p95_of(vals), max_of(vals), mean, stdev);
    };

    compute_stats(load_vec, "load");
    compute_stats(preprocess_vec, "preprocess");
    compute_stats(encode_vec, "encode");
    compute_stats(e2e_vec, "e2e (total)");

    printf("\n");

    // ==================== 详细 decode step 时延 ====================
    if (show_detail && !all_results.empty()) {
        printf("=== Per-Iteration Details ===\n");
        for (size_t i = 0; i < all_results.size(); ++i) {
            printf("  [%2zu] load=%.2f  preproc=%.2f  encode=%.2f  e2e=%.2f  chunks=%d  tokens=%d\n",
                   i, all_results[i].load_ms, all_results[i].preprocess_ms,
                   all_results[i].encode_ms, all_results[i].e2e_ms,
                   all_results[i].n_chunks, all_results[i].n_tokens);
        }
        printf("\n");
    }

    // ==================== 清理 ====================
    vision_free(ctx_vision);

    printf("=== VIT Benchmark finished ===\n");
    return 0;
}
