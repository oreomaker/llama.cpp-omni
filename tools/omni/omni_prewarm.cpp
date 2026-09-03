// omni_prewarm.cpp — init 预热（移植自主工作区 omni_sm.inc 的 omni_sm_prewarm，
// 去 SM / 去实验旋钮后的简化版）。
//
// 在 omni_init 里、duplex_prepare 之后调用一次，做两件事：
//   1. 代表输入 graph 预热：用代表输入走与真实帧同一条 fused prefill + decode
//      代码路径，把首帧的一次性 GPU 代价（CUDA graph 捕获、内核/模块加载）
//      从关键路径挪到 init。
//   2. vocoder 预热：token2wav_session->t2w.warmup_vocoder() 空跑一次，
//      把 stream 创建/首图捕获提前。
//
// 代表输入不靠用户指定：默认用仓库测试帧 0000.jpg/0000.wav（与真实帧形状一致），
// 文件不存在则跳过 VPM/APM 与 fused 路径，优雅降级到 BOS 兜底 + vocoder。
//
// 环境变量（全部可选，正常启动设 0 个）：
//   OMNI_PREWARM      总开关，0 = 关（graph 预热 + vocoder 一起关）
//   OMNI_PREWARM_IMAGE 代表图像路径覆盖
//   OMNI_PREWARM_AUDIO 代表音频路径覆盖

#include "omni-impl.h"
#include "omni.h"
#include "omni_internal.h"
#include "audition.h"
#include "token2wav/token2wav-impl.h"
#include "common/common.h"
#include "common/sampling.h"

#include <sys/stat.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

void omni_prewarm(struct omni_context * ctx_omni) {
#ifdef GGML_USE_CUDA
    if (ctx_omni == nullptr) {
        return;
    }

    const char * env = getenv("OMNI_PREWARM");
    const bool prewarm_on = env == nullptr || atoi(env) != 0;
    if (!prewarm_on) {
        return;   // 总开关关闭：graph 预热 + vocoder 一起关（与无预热 baseline 对比口径）
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    // 代表输入路径：env 覆盖优先，否则仓库自带的测试帧 0000。
    // 与 --test 模式同假设：从仓库根运行，路径是相对仓库根的。
    const char * image_env = getenv("OMNI_PREWARM_IMAGE");
    const char * audio_env = getenv("OMNI_PREWARM_AUDIO");
    const std::string image_path = (image_env && *image_env)
        ? std::string(image_env)
        : std::string("tools/omni/assets/test_case/duplex_omni_test_case/duplex_omni_test_case_0000.jpg");
    const std::string audio_path = (audio_env && *audio_env)
        ? std::string(audio_env)
        : std::string("tools/omni/assets/test_case/duplex_omni_test_case/duplex_omni_test_case_0000.wav");

    auto file_exists = [](const std::string & p) {
        struct stat st;
        return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
    };
    const bool have_image = file_exists(image_path);
    const bool have_audio = file_exists(audio_path);

    bool vpm_ok = false;
    bool apm_ok = false;
    bool llm_ok = false;
    bool voc_ok = false;
    int  fused_ntok = 0;

    std::vector<std::vector<float>> vision_embed;
    std::vector<float> audio_embed;

    // ---- VPM 代表输入 ----
    if (have_image && ctx_omni->ctx_vision) {
        vpm_ok = omni_image_embed_make_chunks_with_filename(
            ctx_omni->ctx_vision, ctx_omni->params->cpuparams.n_threads, image_path, vision_embed);
    }

    // ---- APM 代表输入 ----
    if (have_audio && ctx_omni->ctx_audio) {
        omni_embed * emb = omni_audio_embed_make_with_filename(
            ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, audio_path);
        apm_ok = emb != nullptr && emb->n_pos > 0;
        if (apm_ok && emb != nullptr && ctx_omni->ctx_llama != nullptr) {
            const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
            audio_embed.assign(emb->embed, emb->embed + (size_t)emb->n_pos * hidden_size);
        }
        if (emb) omni_embed_free(emb);
        // 流式 Whisper KV cache 不能被代表输入推进（否则真实帧的缓存位置会被破坏）。
        audition_whisper_clear_kv_cache(ctx_omni->ctx_audio);
    }

    auto clear_llm_state = [&]() {
        llama_synchronize(ctx_omni->ctx_llama);
        llama_memory_clear(llama_get_memory(ctx_omni->ctx_llama), true);
        ctx_omni->n_past = 0;
        ctx_omni->position_offset = 0;
        sliding_window_reset(ctx_omni);
    };
    auto prepare_bucket = [&](int bucket) {
        clear_llm_state();
        if (bucket <= 0) return true;
        const llama_token bos = llama_vocab_bos(llama_model_get_vocab(ctx_omni->model));
        int n_past = 0;
        while (n_past < bucket) {
            const int n_tokens = std::min(bucket - n_past, std::max(1, ctx_omni->params->n_batch));
            llama_batch batch = llama_batch_init(n_tokens, 0, 1);
            batch.n_tokens = n_tokens;
            for (int i = 0; i < n_tokens; ++i) {
                batch.token[i] = bos;
                batch.pos[i] = n_past + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = i + 1 == n_tokens;
            }
            const bool ok = llama_decode(ctx_omni->ctx_llama, batch) == 0;
            llama_batch_free(batch);
            if (!ok) {
                clear_llm_state();
                return false;
            }
            n_past += n_tokens;
        }
        llama_synchronize(ctx_omni->ctx_llama);
        ctx_omni->n_past = bucket;
        return true;
    };

    // ---- LLM 预热：走与真实帧同一条 fused prefill + decode 路径 ----
    if (ctx_omni->ctx_llama && ctx_omni->model && ctx_omni->duplex != nullptr &&
            (!vision_embed.empty() || !audio_embed.empty())) {
        const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
        prepare_bucket(0);
        llm_ok = llm_fused_prefill(ctx_omni, ctx_omni->params, vision_embed, audio_embed,
                                   hidden_size, &fused_ntok);   // P77 图
        llama_synchronize(ctx_omni->ctx_llama);
        prepare_bucket(0);
        int p77_ntok = 0;
        llm_ok &= llm_fused_prefill(ctx_omni, ctx_omni->params, vision_embed, audio_embed,
                                    hidden_size, &p77_ntok);    // D1 前置（再热一次 P77）
        llama_synchronize(ctx_omni->ctx_llama);
        const int decode_pos = ctx_omni->n_past;
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.token[0] = llama_vocab_bos(llama_model_get_vocab(ctx_omni->model));
        batch.pos[0] = decode_pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        llm_ok &= (llama_decode(ctx_omni->ctx_llama, batch) == 0);   // D1 图
        llama_synchronize(ctx_omni->ctx_llama);
        llama_batch_free(batch);
        clear_llm_state();
    } else if (ctx_omni->ctx_llama && ctx_omni->model) {
        // ---- BOS 兜底：无代表输入可用时，单 BOS token 热 decode ----
        llama_token tok = llama_vocab_bos(llama_model_get_vocab(ctx_omni->model));
        llama_batch batch = llama_batch_get_one(&tok, 1);
        llm_ok = llama_decode(ctx_omni->ctx_llama, batch) == 0;
        llama_synchronize(ctx_omni->ctx_llama);
        llama_memory_clear(llama_get_memory(ctx_omni->ctx_llama), true);
        ctx_omni->n_past = 0;
        if (ctx_omni->sliding_window_config.mode != "off") {
            sliding_window_reset(ctx_omni);
        }
    }

    // ---- SM 各 plan 的 EC 预建 ----
    // 遍历 4 个 preset 对 5 组 backend 各 set_sm_group 一次，把每个 plan 的 green context
    // 建好并留在 green_ctx_cache。运行期切到任一 plan 零重建 EC。STEADY 放最后，末态 STEADY。
    if (ctx_omni->sm_sched.enabled && ctx_omni->ctx_llama && ctx_omni->model &&
            ctx_omni->duplex != nullptr) {
        SmSched & s = ctx_omni->sm_sched;
        const int order[4] = {SM_PLAN_ENC_T2W, SM_PLAN_ENC_LLM, SM_PLAN_ENC_TTS, SM_PLAN_STEADY};
        auto apply_plan_all = [&](int plan) {
            auto set_one = [&](ggml_backend_t b, int group) {
                if (b == nullptr) return;
                ggml_backend_cuda_sm_plan * p = s.presets[plan];
                if (p != nullptr) ggml_backend_cuda_set_sm_group(b, p, group, plan);
            };
            set_one(s.backend[SM_G_LLM], SM_G_LLM);
            set_one(s.backend[SM_G_TTS], SM_G_TTS);
            set_one(s.backend[SM_G_APM], SM_G_APM);
            set_one(s.backend[SM_G_VPM], SM_G_VPM);
            for (auto b : s.t2w_backends) set_one(b, SM_G_T2W);
            s.applied_plan = plan;
        };
        for (int p : order) {
            apply_plan_all(p);
        }
        print_with_timestamp("[WARM] SM EC prebuilt for 4 plans, applied=STEADY\n");
    }

    // ---- vocoder 预热（session 存在时无条件） ----
    if (ctx_omni->token2wav_session) {
        voc_ok = ctx_omni->token2wav_session->t2w.warmup_vocoder();
    }

    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    print_with_timestamp("[WARM] image=%s audio=%s fused_ntok=%d VPM=%s APM=%s LLM=%s VOC=%s elapsed=%.1fms\n",
                         image_path.c_str(), audio_path.c_str(), fused_ntok,
                         vpm_ok ? "ok" : "skip", apm_ok ? "ok" : "skip",
                         llm_ok ? "ok" : "skip", voc_ok ? "ok" : "skip", ms);
    print_with_timestamp("=== omni_prewarm done ===\n");
#else
    (void) ctx_omni;
#endif
}
