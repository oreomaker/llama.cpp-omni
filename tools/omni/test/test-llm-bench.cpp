/**
 * LLM 推理性能基准测试
 *
 * 测试不同上下文长度和 prefill 长度下的 LLM 推理时延：
 *   - 上下文长度: 128, 256, 512, 1024, 2048
 *   - Prefill 长度: 16, 32, 64
 *   - Decode 时延: 每个配置下生成固定数量 token 的平均时延
 *
 * 用法:
 *   llama-omni-test-llm-bench -m <llm_model_path> [options]
 *     -c <n>           上下文大小 (默认 4096)
 *     -ngl <n>         GPU 层数 (默认 99)
 *     -b <n>           batch size (默认 2048)
 *     --decode-tok <n> decode token 数量 (默认 32)
 *     --repeat <n>     每个配置重复次数 (默认 3)
 */

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ==================== 配置 ====================

static const std::vector<int> CONTEXT_LENGTHS  = { 128, 256, 512, 1024, 2048 };
static const std::vector<int> PREFILL_LENGTHS  = { 16, 32, 64, 128, 256 };
static const int              DECODE_TOKENS    = 32;
static const int              WARMUP_DECODES   = 4;

// ==================== 计时工具 ====================

struct BenchResult {
    int    ctx_len;
    int    prefill_len;
    double prefill_ms;
    double prefill_tps;   // tokens per second
    double decode_avg_ms; // 平均每 token decode 时延
    double decode_tps;    // tokens per second
    double decode_first_ms; // 第一个 decode token 时延 (含 prefill 后首次前向)
    std::vector<double> decode_step_ms;
};

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

// ==================== 核心评估函数 ====================

static bool eval_tokens(llama_context * ctx, std::vector<llama_token> & tokens, int n_batch, int n_past) {
    const int n_tokens = static_cast<int>(tokens.size());
    for (int i = 0; i < n_tokens; i += n_batch) {
        const int n_eval = std::min(n_tokens - i, n_batch);
        if (n_eval == 0) break;

        llama_batch batch = llama_batch_get_one(tokens.data() + i, n_eval);

        // 设置 position
        std::vector<llama_pos> pos(n_eval);
        for (int j = 0; j < n_eval; ++j) {
            pos[j] = n_past + i + j;
        }
        batch.pos = pos.data();

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "llama_decode failed at token %d/%d\n", i, n_tokens);
            return false;
        }
    }

    // 等待 GPU 执行完成，确保后续计时和 logits 读取的准确性
    llama_synchronize(ctx);
    return true;
}

// Argmax 采样: 从 logits 中取概率最大的 token
static llama_token argmax_sample(llama_context * ctx) {
    float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) return -1;

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_token best_token = 0;
    float       best_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best_token = i;
        }
    }
    return best_token;
}

// ==================== 单次基准测试 ====================

static bool run_single_bench(llama_context *          ctx,
                             int                      n_batch,
                             int                      ctx_len,
                             int                      prefill_len,
                             int                      n_decode_tokens,
                             const std::vector<llama_token> & vocab_token_pool,
                             BenchResult &            result) {
    result.ctx_len    = ctx_len;
    result.prefill_len = prefill_len;

    // 1. 清空 KV cache
    llama_memory_clear(llama_get_memory(ctx), true);

    // 2. 用随机 token 填充上下文 (模拟已有的 KV cache)
    //    使用词汇表中的有效 token，避免 special token
    std::mt19937 rng(42); // 固定种子保证可重复性
    std::vector<llama_token> fill_tokens(ctx_len);
    for (int i = 0; i < ctx_len; ++i) {
        fill_tokens[i] = vocab_token_pool[rng() % vocab_token_pool.size()];
    }

    if (!eval_tokens(ctx, fill_tokens, n_batch, 0)) {
        fprintf(stderr, "Failed to fill context\n");
        return false;
    }
    int n_past = ctx_len;

    // 3. 生成 prefill tokens
    std::vector<llama_token> prefill_tokens(prefill_len);
    for (int i = 0; i < prefill_len; ++i) {
        prefill_tokens[i] = vocab_token_pool[rng() % vocab_token_pool.size()];
    }

    // 4. 测量 prefill 时延
    //    先 sync 确保前序 context fill 的 GPU 工作全部完成
    llama_synchronize(ctx);
    auto prefill_start = std::chrono::high_resolution_clock::now();
    if (!eval_tokens(ctx, prefill_tokens, n_batch, n_past)) {
        fprintf(stderr, "Prefill failed\n");
        return false;
    }
    // eval_tokens 内部已调用 llama_synchronize，GPU 工作已完成
    auto prefill_end = std::chrono::high_resolution_clock::now();
    n_past += prefill_len;

    result.prefill_ms  = elapsed_ms(prefill_start, prefill_end);
    result.prefill_tps = prefill_len / (result.prefill_ms / 1000.0);

    // 5. Decode: 逐 token 生成并测量时延
    result.decode_step_ms.clear();
    result.decode_step_ms.reserve(n_decode_tokens);

    llama_token last_token = prefill_tokens.back();

    for (int i = 0; i < n_decode_tokens; ++i) {
        // eval 单个 token
        std::vector<llama_token> single_tok = { last_token };
        auto step_start = std::chrono::high_resolution_clock::now();
        if (!eval_tokens(ctx, single_tok, n_batch, n_past)) {
            fprintf(stderr, "Decode step %d failed\n", i);
            return false;
        }

        // 采样 (argmax)
        last_token = argmax_sample(ctx);
        if (last_token < 0) {
            fprintf(stderr, "Sampling failed at step %d\n", i);
            return false;
        }

        auto step_end = std::chrono::high_resolution_clock::now();
        double step_time = elapsed_ms(step_start, step_end);
        result.decode_step_ms.push_back(step_time);
        n_past++;
    }

    // 计算 decode 统计 (跳过前 WARMUP_DECODES 个 token)
    const int warmup = std::min(WARMUP_DECODES, n_decode_tokens);
    if (n_decode_tokens > warmup) {
        double sum = 0.0;
        for (int i = warmup; i < n_decode_tokens; ++i) {
            sum += result.decode_step_ms[i];
        }
        result.decode_avg_ms   = sum / (n_decode_tokens - warmup);
        result.decode_tps      = 1000.0 / result.decode_avg_ms;
    } else {
        result.decode_avg_ms = 0.0;
        result.decode_tps    = 0.0;
    }
    result.decode_first_ms = result.decode_step_ms.empty() ? 0.0 : result.decode_step_ms[0];

    return true;
}

// ==================== 输出格式化 ====================

static void print_header() {
    printf("\n");
    printf("================================================================================================\n");
    printf("  LLM Inference Benchmark Results\n");
    printf("================================================================================================\n");
    printf("  %-8s | %-10s | %-14s | %-12s | %-14s | %-12s | %-10s\n",
           "ctx_len", "prefill_len", "prefill(ms)", "prefill(tps)", "decode_avg(ms)", "decode(tps)", "1st_tok(ms)");
    printf("  ---------|------------|----------------|--------------|----------------|--------------|------------\n");
}

static void print_result(const BenchResult & r) {
    printf("  %-8d | %-10d | %12.2f   | %10.1f   | %12.2f   | %10.1f   | %10.2f\n",
           r.ctx_len, r.prefill_len,
           r.prefill_ms, r.prefill_tps,
           r.decode_avg_ms, r.decode_tps,
           r.decode_first_ms);
}

static void print_footer() {
    printf("================================================================================================\n");
    printf("  Note: decode_avg excludes first %d warmup tokens\n", WARMUP_DECODES);
    printf("  tps = tokens per second\n");
    printf("================================================================================================\n\n");
}

static void print_decode_detail(const BenchResult & r) {
    printf("    decode steps (ctx=%d, prefill=%d): ", r.ctx_len, r.prefill_len);
    for (size_t i = 0; i < r.decode_step_ms.size(); ++i) {
        if (i > 0) printf(", ");
        printf("%.2f", r.decode_step_ms[i]);
    }
    printf(" ms\n");
}

// ==================== 帮助信息 ====================

static void show_usage(const char * prog_name) {
    printf(
        "LLM Inference Benchmark\n\n"
        "Usage: %s -m <llm_model_path> [options]\n\n"
        "Required:\n"
        "  -m <path>             LLM 模型路径\n\n"
        "Options:\n"
        "  -c, --ctx-size <n>    上下文大小 (默认: 4096)\n"
        "  -ngl <n>              GPU 层数 (默认: 99)\n"
        "  -b, --batch-size <n>  batch size (默认: 2048)\n"
        "  --decode-tok <n>      每次 decode token 数量 (默认: 32)\n"
        "  --repeat <n>          每个配置重复次数 (默认: 3)\n"
        "  --detail              输出每个 decode step 的详细时延\n"
        "  -h, --help            显示帮助\n\n"
        "Example:\n"
        "  %s -m ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf -ngl 99\n",
        prog_name, prog_name);
}

// ==================== Main ====================

int main(int argc, char ** argv) {
    ggml_time_init();

    std::string llm_path;
    int         n_ctx        = 4096;
    int         n_gpu_layers = 99;
    int         n_batch      = 2048;
    int         n_decode_tok = DECODE_TOKENS;
    int         n_repeat     = 3;
    bool        show_detail  = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            show_usage(argv[0]);
            return 0;
        } else if (arg == "-m" && i + 1 < argc) {
            llm_path = argv[++i];
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            n_ctx = std::atoi(argv[++i]);
        } else if (arg == "-ngl" && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if ((arg == "-b" || arg == "--batch-size") && i + 1 < argc) {
            n_batch = std::atoi(argv[++i]);
        } else if (arg == "--decode-tok" && i + 1 < argc) {
            n_decode_tok = std::atoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            n_repeat = std::atoi(argv[++i]);
        } else if (arg == "--detail") {
            show_detail = true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            show_usage(argv[0]);
            return 1;
        }
    }

    if (llm_path.empty()) {
        fprintf(stderr, "Error: -m <llm_model_path> is required\n\n");
        show_usage(argv[0]);
        return 1;
    }

    // 验证 context 长度配置不超过 n_ctx
    for (int cl : CONTEXT_LENGTHS) {
        for (int pl : PREFILL_LENGTHS) {
            if (cl + pl > n_ctx) {
                fprintf(stderr, "Warning: context_length(%d) + prefill_length(%d) > n_ctx(%d), will skip\n", cl, pl, n_ctx);
            }
        }
    }

    common_init();

    // ==================== 加载模型 ====================
    printf("=== Loading LLM Model ===\n");
    printf("  Path: %s\n", llm_path.c_str());
    printf("  n_ctx: %d\n", n_ctx);
    printf("  n_gpu_layers: %d\n", n_gpu_layers);
    printf("  n_batch: %d\n", n_batch);

    common_params params;
    params.model.path   = llm_path;
    params.n_ctx        = n_ctx;
    params.n_batch      = n_batch;
    params.n_gpu_layers = n_gpu_layers;
    params.n_predict    = 0; // 不需要预测, 仅 benchmark

    common_init_result init_result = common_init_from_params(params);
    llama_model *   model = init_result.model.get();
    llama_context * ctx   = init_result.context.get();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "Error: Failed to load model\n");
        return 1;
    }

    // 设置线程数
    llama_set_n_threads(ctx, params.cpuparams.n_threads, params.cpuparams.n_threads);

    const int n_embd = llama_model_n_embd(model);
    printf("  n_embd: %d\n", n_embd);
    printf("  Model loaded successfully\n");
    printf("==========================\n");

    // ==================== 构建词汇表 token 池 ====================
    // 使用普通文本 token (避免 special tokens)
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> vocab_token_pool;
    vocab_token_pool.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        // 简单过滤: 只使用 id < 100000 的 token, 通常覆盖大部分常用 token
        // 且跳过 id=0 (通常是 bos/unk)
        if (i > 0 && i < 100000) {
            vocab_token_pool.push_back(i);
        }
    }
    if (vocab_token_pool.empty()) {
        // fallback: 使用所有 token
        for (int i = 1; i < n_vocab; ++i) {
            vocab_token_pool.push_back(i);
        }
    }
    printf("  Vocab token pool: %zu tokens\n", vocab_token_pool.size());

    // ==================== Warmup ====================
    printf("\n=== Warmup ===\n");
    {
        BenchResult warmup_result;
        run_single_bench(ctx, n_batch, 128, 16, 4, vocab_token_pool, warmup_result);
        llama_memory_clear(llama_get_memory(ctx), true);
        printf("  Warmup done\n");
    }

    // ==================== 运行基准测试 ====================
    printf("\n=== Running Benchmark (repeat=%d, decode_tok=%d) ===\n", n_repeat, n_decode_tok);

    // 收集所有结果用于汇总
    struct AggregatedResult {
        int ctx_len;
        int prefill_len;
        std::vector<double> prefill_ms_vec;
        std::vector<double> decode_avg_ms_vec;
        std::vector<double> decode_first_ms_vec;
    };

    std::vector<AggregatedResult> aggregated;
    std::vector<BenchResult>      all_results;

    print_header();

    for (int ctx_len : CONTEXT_LENGTHS) {
        for (int prefill_len : PREFILL_LENGTHS) {
            if (ctx_len + prefill_len > n_ctx) {
                printf("  %-8d | %-10d | SKIPPED (ctx_len + prefill_len > n_ctx=%d)\n", ctx_len, prefill_len, n_ctx);
                continue;
            }

            AggregatedResult agg;
            agg.ctx_len     = ctx_len;
            agg.prefill_len = prefill_len;

            for (int rep = 0; rep < n_repeat; ++rep) {
                BenchResult result;
                if (!run_single_bench(ctx, n_batch, ctx_len, prefill_len, n_decode_tok,
                                      vocab_token_pool, result)) {
                    fprintf(stderr, "  Benchmark failed: ctx=%d, prefill=%d, repeat=%d\n",
                            ctx_len, prefill_len, rep);
                    continue;
                }

                agg.prefill_ms_vec.push_back(result.prefill_ms);
                agg.decode_avg_ms_vec.push_back(result.decode_avg_ms);
                agg.decode_first_ms_vec.push_back(result.decode_first_ms);
                all_results.push_back(result);
            }

            // 输出中位数结果
            if (!agg.prefill_ms_vec.empty()) {
                BenchResult median_result;
                median_result.ctx_len     = ctx_len;
                median_result.prefill_len = prefill_len;
                median_result.prefill_ms  = median_of(agg.prefill_ms_vec);
                median_result.prefill_tps = prefill_len / (median_result.prefill_ms / 1000.0);
                median_result.decode_avg_ms   = median_of(agg.decode_avg_ms_vec);
                median_result.decode_tps      = median_result.decode_avg_ms > 0 ?
                                                    1000.0 / median_result.decode_avg_ms : 0.0;
                median_result.decode_first_ms = median_of(agg.decode_first_ms_vec);
                print_result(median_result);
                aggregated.push_back(agg);
            }
        }
    }

    print_footer();

    // ==================== 汇总统计 ====================
    printf("=== Summary Statistics ===\n");
    printf("  %-8s | %-10s | %-14s | %-14s | %-14s | %-14s\n",
           "ctx_len", "prefill_len", "prefill_p50", "prefill_p95", "decode_p50", "decode_p95");
    printf("  ---------|------------|----------------|----------------|----------------|----------------\n");

    for (const auto & agg : aggregated) {
        double prefill_p50 = median_of(agg.prefill_ms_vec);
        double prefill_p95 = p95_of(agg.prefill_ms_vec);
        double decode_p50  = median_of(agg.decode_avg_ms_vec);
        double decode_p95  = p95_of(agg.decode_avg_ms_vec);
        printf("  %-8d | %-10d | %12.2f   | %12.2f   | %12.2f   | %12.2f\n",
               agg.ctx_len, agg.prefill_len,
               prefill_p50, prefill_p95,
               decode_p50, decode_p95);
    }
    printf("\n");

    // ==================== 详细 decode step 时延 ====================
    if (show_detail) {
        printf("=== Decode Step Details (last repeat per config) ===\n");
        // 找到每个配置的最后一次重复结果
        for (size_t i = 0; i < all_results.size(); ++i) {
            bool is_last = (i + 1 >= all_results.size()) ||
                           (all_results[i + 1].ctx_len != all_results[i].ctx_len ||
                            all_results[i + 1].prefill_len != all_results[i].prefill_len);
            if (is_last) {
                print_decode_detail(all_results[i]);
            }
        }
        printf("\n");
    }

    // ==================== LLM 内部性能统计 ====================
    llama_perf_context_print(ctx);

    printf("\n=== Benchmark finished ===\n");
    return 0;
}
