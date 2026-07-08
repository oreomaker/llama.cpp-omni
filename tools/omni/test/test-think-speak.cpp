/**
 * Think-speak (边想边说) test — M1 pure-text (default) and M2 spoken (--tts).
 *
 * Milestone M1 (THINK_SPEAK_PLAN.md §7/§8 P4): drive the think-speak state
 * machine on a text math problem, greedily and deterministically, entirely
 * inside the runtime — no TTS, no audio input. Prints each think/answer block
 * and asserts the golden sample reaches endofthink and answers "four".
 *
 * Milestone M2 (§8 P5, --tts): same state machine with use_tts=true + async=true.
 * answer/final blocks route hidden states to the TTS thread → WAV under
 * round_000/tts_wav; think blocks stay silent. Asserts audio is produced.
 *
 * Flow:
 *   1. resolve model paths (LLM = think GGUF; audio/vision/tts via sibling symlinks)
 *   2. omni_init(media_type=1, use_tts, enable_thinking=true)  -> think prompts + greedy
 *   3. omni_think_speak_generate(question) -> ThinkSpeakResult
 *   4. (M2) wait for round_000/tts_wav/generation_done.flag; stop + join TTS/T2W threads
 *   5. print blocks + assert stopped_reason / answer (+ WAV produced on M2)
 */

#include "omni-impl.h"
#include "omni.h"
#include "think_speak.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <dirent.h>
#endif

// ==================== helpers (mirror test-duplex.cpp) ====================

struct TestModelPaths {
    std::string llm;
    std::string vision;
    std::string audio;
    std::string tts;
    std::string projector;
    std::string base_dir;
};

static std::string get_parent_dir(const std::string & path) {
    size_t last_slash = path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        return path.substr(0, last_slash);
    }
    return ".";
}

static bool file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f) { fclose(f); return true; }
    return false;
}

// Count *.wav files in a directory (M2 audio assertion). 0 on missing dir.
static int count_wav_files(const std::string & dir) {
    int n = 0;
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    DIR * d = opendir(dir.c_str());
    if (!d) return 0;
    struct dirent * ent;
    while ((ent = readdir(d)) != nullptr) {
        const std::string name = ent->d_name;
        if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".wav") == 0) n++;
    }
    closedir(d);
#endif
    return n;
}

// ---- minimal WAV merge (no ffmpeg dependency) ----
// The T2W thread writes one canonical PCM WAV per chunk (wav_0.wav..wav_N.wav,
// s16le/24kHz/mono). Concatenating the raw data payloads in NUMERIC order yields a
// single playable clip. Parses fmt+data by scanning RIFF chunks so it tolerates
// non-44-byte headers; format is taken from the first file (all chunks share it).

// Read one canonical PCM WAV: fill fmt fields, append its data payload to out_pcm.
static bool wav_read_pcm(const std::string & path, uint16_t & channels, uint32_t & sample_rate,
                         uint16_t & bits, std::vector<uint8_t> & out_pcm) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsz < 44) { fclose(f); return false; }
    std::vector<uint8_t> buf((size_t) fsz);
    const bool read_ok = fread(buf.data(), 1, (size_t) fsz, f) == (size_t) fsz;
    fclose(f);
    if (!read_ok) return false;
    if (memcmp(buf.data(), "RIFF", 4) != 0 || memcmp(buf.data() + 8, "WAVE", 4) != 0) return false;

    bool got_fmt = false, got_data = false;
    size_t p = 12;
    while (p + 8 <= (size_t) fsz) {
        const uint8_t * id = buf.data() + p;
        uint32_t csz;
        memcpy(&csz, buf.data() + p + 4, 4);
        const size_t body = p + 8;
        if (body + csz > (size_t) fsz) csz = (uint32_t)((size_t) fsz - body);  // clamp
        if (memcmp(id, "fmt ", 4) == 0 && csz >= 16) {
            memcpy(&channels,    buf.data() + body + 2,  2);
            memcpy(&sample_rate, buf.data() + body + 4,  4);
            memcpy(&bits,        buf.data() + body + 14, 2);
            got_fmt = true;
        } else if (memcmp(id, "data", 4) == 0) {
            out_pcm.insert(out_pcm.end(), buf.data() + body, buf.data() + body + csz);
            got_data = true;
        }
        p = body + csz + (csz & 1);  // chunks are word-aligned
    }
    return got_fmt && got_data;
}

static bool wav_write_pcm(const std::string & path, uint16_t channels, uint32_t sample_rate,
                          uint16_t bits, const std::vector<uint8_t> & pcm) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t data_sz     = (uint32_t) pcm.size();
    const uint32_t byte_rate   = sample_rate * channels * (bits / 8);
    const uint16_t block_align = (uint16_t)(channels * (bits / 8));
    const uint32_t riff_sz     = 36 + data_sz;
    const uint16_t pcm_fmt = 1, fmt_sz16 = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff_sz, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); { uint32_t v = fmt_sz16; fwrite(&v, 4, 1, f); }
    fwrite(&pcm_fmt, 2, 1, f); fwrite(&channels, 2, 1, f); fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f); fwrite(&block_align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data_sz, 4, 1, f);
    if (!pcm.empty()) fwrite(pcm.data(), 1, pcm.size(), f);
    fclose(f);
    return true;
}

// Merge wav_<n>.wav in `dir` (NUMERIC order — wav_2 before wav_10) into out_path.
// Returns #chunks merged, or -1 on failure.
static int merge_round_wavs(const std::string & dir, const std::string & out_path) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    std::vector<std::pair<long, std::string>> files;
    DIR * d = opendir(dir.c_str());
    if (!d) return -1;
    struct dirent * ent;
    while ((ent = readdir(d)) != nullptr) {
        const std::string n = ent->d_name;
        if (n.size() > 8 && n.compare(0, 4, "wav_") == 0 && n.compare(n.size() - 4, 4, ".wav") == 0) {
            const std::string num = n.substr(4, n.size() - 8);
            char * end = nullptr;
            const long idx = strtol(num.c_str(), &end, 10);
            if (end && *end == '\0') files.emplace_back(idx, n);
        }
    }
    closedir(d);
    if (files.empty()) return -1;
    std::sort(files.begin(), files.end(),
              [](const std::pair<long, std::string> & a, const std::pair<long, std::string> & b) {
                  return a.first < b.first;
              });

    uint16_t channels = 1, bits = 16;
    uint32_t sample_rate = 24000;
    std::vector<uint8_t> pcm;
    int merged = 0;
    for (const auto & fp : files) {
        if (wav_read_pcm(dir + "/" + fp.second, channels, sample_rate, bits, pcm)) merged++;
    }
    if (merged == 0 || !wav_write_pcm(out_path, channels, sample_rate, bits, pcm)) return -1;
    return merged;
#else
    (void) dir; (void) out_path;
    return -1;
#endif
}

static TestModelPaths resolve_model_paths(const std::string & llm_path) {
    TestModelPaths paths;
    paths.llm       = llm_path;
    paths.base_dir  = get_parent_dir(llm_path);
    paths.vision    = paths.base_dir + "/vision/MiniCPM-o-4_5-vision-F16.gguf";
    paths.audio     = paths.base_dir + "/audio/MiniCPM-o-4_5-audio-F16.gguf";
    paths.tts       = paths.base_dir + "/tts/MiniCPM-o-4_5-tts-F16.gguf";
    paths.projector = paths.base_dir + "/tts/MiniCPM-o-4_5-projector-F16.gguf";
    return paths;
}

static std::string to_lower(std::string s) {
    for (char & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

static void show_usage(const char * prog) {
    printf("Think-speak M1 pure-text test\n\n");
    printf("用法: %s [选项]\n\n", prog);
    printf("  -m <path>                 think LLM GGUF (默认 /cache/zhanghao/model/o45-think-gguf/MiniCPM-o-4_5-think-F16.gguf)\n");
    printf("  --text-question <str>     题面 (默认: 黄金样例 multi_step_reasoning-000000)\n");
    printf("  --audio-question <path>   音频问题 WAV 文件路径 (与 --text-question 互斥)\n");
    printf("  --tts                     M2 出声模式 (use_tts=true, async=true, 落 WAV)；默认关=M1 纯文本\n");
    printf("  -c, --ctx-size <n>        上下文大小 (默认 4096)\n");
    printf("  -ngl <n>                  n_gpu_layers (默认 99)\n");
    printf("  -o <dir>                  输出目录 (默认 ./tools/omni/output)\n");
    printf("  --think-budget <n>        每 think 块 token 预算 (默认 25)\n");
    printf("  --answer-budget <n>       每 (非最终) answer 块 token 预算 (默认 4)\n");
    printf("  --endofthink-lookahead <n> 预算用尽后确认 <endofthink> 的额外步 (默认 12)\n");
    printf("  --final-answer-max-new-tokens <n> (默认 3000)\n");
    printf("  --think-hard-limit <n>    (默认 100)\n");
    printf("  --answer-hard-limit <n>   (默认 100)\n");
    printf("  --final-answer-hard-limit <n> (默认 500)\n");
    printf("  --max-auto-blocks <n>     (默认 100)\n");
    printf("  -h, --help                显示帮助\n");
}

int main(int argc, char ** argv) {
    std::string llm_path = "/cache/zhanghao/model/o45-think-gguf/MiniCPM-o-4_5-think-F16.gguf";
    // 黄金样例 multi_step_reasoning-000000 的口播转写 (context_transcript)。7+12-15 = 4。
    std::string question =
        "Paige raised 7 goldfish and 12 catfish in the pond but stray cats loved eating them. "
        "Now she has 15 left. How many fishes disappeared?";
    std::string audio_question;  // --audio-question: WAV path for audio input mode
    std::string output_dir = "./tools/omni/output";
    int n_ctx        = 4096;
    int n_gpu_layers = 99;
    bool use_tts     = false;  // --tts: M2 spoken mode (async + WAV)

    ThinkSpeakConfig cfg;  // §3 缺省值

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { show_usage(argv[0]); return 0; }
        else if (arg == "-m" && i + 1 < argc) { llm_path = argv[++i]; }
        else if (arg == "--text-question" && i + 1 < argc) { question = argv[++i]; }
        else if (arg == "--audio-question" && i + 1 < argc) { audio_question = argv[++i]; }
        else if (arg == "--tts") { use_tts = true; }
        else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) { n_ctx = std::atoi(argv[++i]); }
        else if (arg == "-ngl" && i + 1 < argc) { n_gpu_layers = std::atoi(argv[++i]); }
        else if (arg == "-o" && i + 1 < argc) { output_dir = argv[++i]; }
        else if (arg == "--think-budget" && i + 1 < argc) { cfg.think_budget = std::atoi(argv[++i]); }
        else if (arg == "--answer-budget" && i + 1 < argc) { cfg.answer_budget = std::atoi(argv[++i]); }
        else if (arg == "--endofthink-lookahead" && i + 1 < argc) { cfg.endofthink_lookahead = std::atoi(argv[++i]); }
        else if (arg == "--final-answer-max-new-tokens" && i + 1 < argc) { cfg.final_answer_max_new_tokens = std::atoi(argv[++i]); }
        else if (arg == "--think-hard-limit" && i + 1 < argc) { cfg.think_hard_limit = std::atoi(argv[++i]); }
        else if (arg == "--answer-hard-limit" && i + 1 < argc) { cfg.answer_hard_limit = std::atoi(argv[++i]); }
        else if (arg == "--final-answer-hard-limit" && i + 1 < argc) { cfg.final_answer_hard_limit = std::atoi(argv[++i]); }
        else if (arg == "--max-auto-blocks" && i + 1 < argc) { cfg.max_auto_blocks = std::atoi(argv[++i]); }
        else { fprintf(stderr, "未知参数: %s\n", arg.c_str()); show_usage(argv[0]); return 1; }
    }

    const bool audio_input = !audio_question.empty();

    TestModelPaths paths = resolve_model_paths(llm_path);
    printf("=== Think-speak %s test ===\n", use_tts ? "M2 spoken (--tts)" : "M1 pure-text");
    printf("  LLM:    %s\n", paths.llm.c_str());
    printf("  Audio encoder: %s\n", paths.audio.c_str());
    printf("  TTS:    %s (%s)\n", paths.tts.c_str(), use_tts ? "enabled" : "disabled");
    if (audio_input) {
        printf("  Audio question: %s\n", audio_question.c_str());
    } else {
        printf("  Text question: %s\n", question.c_str());
    }
    printf("  cfg: think_budget=%d answer_budget=%d lookahead=%d final_max=%d "
           "think_hard=%d answer_hard=%d final_hard=%d max_blocks=%d\n",
           cfg.think_budget, cfg.answer_budget, cfg.endofthink_lookahead, cfg.final_answer_max_new_tokens,
           cfg.think_hard_limit, cfg.answer_hard_limit, cfg.final_answer_hard_limit, cfg.max_auto_blocks);

    if (!file_exists(paths.llm))   { fprintf(stderr, "Error: LLM not found: %s\n", paths.llm.c_str());   return 1; }
    if (!file_exists(paths.audio)) { fprintf(stderr, "Error: Audio encoder not found: %s\n", paths.audio.c_str()); return 1; }
    if (use_tts && !file_exists(paths.tts)) { fprintf(stderr, "Error: TTS not found: %s\n", paths.tts.c_str()); return 1; }
    if (audio_input && !file_exists(audio_question)) { fprintf(stderr, "Error: Audio question not found: %s\n", audio_question.c_str()); return 1; }

    common_params params;
    params.model.path   = paths.llm;
    params.vpm_model    = paths.vision;
    params.apm_model    = paths.audio;
    params.tts_model    = paths.tts;
    params.n_ctx        = n_ctx;
    params.n_gpu_layers = n_gpu_layers;

    // 贪心、确定性 —— 对齐 Python demo 的 --no-do-sample (argmax)。
    // think-speak 复用 ctx_sampler，temp<=0 即贪心 (无独立 greedy 开关)。M2 下 TTS
    // 用其自身采样器 (tts_temperature)，与 LLM 贪心无关。
    params.sampling.temp = 0.0f;
    params.sampling.seed = 42;

    std::string tts_bin_dir = get_parent_dir(paths.tts);

    common_init();

    auto ctx_omni = omni_init(&params, /*media_type=*/1, /*use_tts=*/use_tts, tts_bin_dir,
                              /*tts_gpu_layers=*/-1, /*token2wav_device=*/"gpu:0",
                              /*duplex_mode=*/false,
                              /*existing_model=*/nullptr, /*existing_ctx=*/nullptr,
                              /*base_output_dir=*/output_dir,
                              /*enable_thinking=*/true);
    if (ctx_omni == nullptr) {
        fprintf(stderr, "Error: omni_init failed\n");
        return 1;
    }
    // M2 需要 async=true 才会起 TTS/T2W 线程并走隐藏态交接；M1 保持同步。
    ctx_omni->async = use_tts;

    ThinkSpeakResult result;
    const bool ok = omni_think_speak_generate(ctx_omni, cfg, question, &result, audio_question);
    if (!ok) {
        fprintf(stderr, "Error: omni_think_speak_generate failed\n");
        omni_free(ctx_omni);
        return 1;
    }

    // ---- M2: 等待音频生成落盘，并停/收线程 (镜像 single_test_audio) ----
    int wav_count = 0;
    if (use_tts && ctx_omni->async) {
        const std::string wav_dir   = output_dir + "/round_000/tts_wav";
        const std::string done_flag = wav_dir + "/generation_done.flag";
        fprintf(stderr, "Waiting for audio generation to complete (%s)...\n", done_flag.c_str());
        for (int i = 0; i < 1200; ++i) {  // 最多等 120 秒
            if (file_exists(done_flag)) { fprintf(stderr, "Audio generation completed.\n"); break; }
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
            usleep(100000);  // 100ms
#endif
        }
        omni_stop_threads(ctx_omni);
        if (ctx_omni->llm_thread.joinable()) { ctx_omni->llm_thread.join(); }
        if (ctx_omni->tts_thread.joinable()) { ctx_omni->tts_thread.join(); printf("tts thread end\n"); }
        if (ctx_omni->t2w_thread.joinable()) { ctx_omni->t2w_thread.join(); printf("t2w thread end\n"); }
        wav_count = count_wav_files(wav_dir);
        printf("  tts_wav dir: %s (%d wav files)\n", wav_dir.c_str(), wav_count);

        // 把 82 个 chunk 按数字序合并成一条 merged.wav（无 ffmpeg 依赖）。
        if (wav_count > 0) {
            const std::string merged = output_dir + "/round_000/merged.wav";
            const int n = merge_round_wavs(wav_dir, merged);
            if (n > 0) {
                printf("  merged %d chunks -> %s\n", n, merged.c_str());
            } else {
                fprintf(stderr, "  WARN: merge failed (dir=%s)\n", wav_dir.c_str());
            }
        }
    }

    // ---- 逐块打印 ----
    printf("\n==================== think-speak result ====================\n");
    for (const auto & blk : result.blocks) {
        printf("---- block %d%s ----\n", blk.block_index, blk.endofthink ? " [endofthink]" : "");
        printf("  🧠 think : %s\n", blk.think.c_str());
        printf("  🔊 answer: %s\n", blk.answer.c_str());
    }
    printf("------------------------------------------------------------\n");
    printf("  final_answer  : %s\n", result.final_answer.c_str());
    printf("  stopped_reason: %s\n", result.stopped_reason.c_str());
    printf("  blocks        : %zu\n", result.blocks.size());
    printf("============================================================\n");

    // ---- 断言 (§7 黄金样例) ----
    const std::string ans_lc = to_lower(result.final_answer);
    const bool stop_ok   = (result.stopped_reason == "endofthink_answer_finished");
    const bool answer_ok = (ans_lc.find("four") != std::string::npos) ||
                           (result.final_answer.find("4") != std::string::npos);

    printf("\n[ASSERT] stopped_reason == endofthink_answer_finished : %s (got \"%s\")\n",
           stop_ok ? "PASS" : "FAIL", result.stopped_reason.c_str());
    printf("[ASSERT] final answer contains four/4                 : %s\n",
           answer_ok ? "PASS" : "FAIL");

    // M2: 断言至少落了一个 WAV（answer 块出声）。
    bool audio_ok = true;
    if (use_tts) {
        audio_ok = (wav_count > 0);
        printf("[ASSERT] M2 audio produced (wav files > 0)           : %s (%d)\n",
               audio_ok ? "PASS" : "FAIL", wav_count);
    }

    const bool pass = stop_ok && answer_ok && audio_ok;
    printf("\n=== Think-speak %s test: %s ===\n",
           use_tts ? "M2 spoken" : "M1 pure-text", pass ? "PASS ✅" : "FAIL ❌");

    llama_perf_context_print(ctx_omni->ctx_llama);
    omni_free(ctx_omni);
    return pass ? 0 : 1;
}
