// Duplex scheduling layer (converted from the former omni_duplex.inc include).
// Owns frame orchestration and worker threads; model/graph execution is in the
// compute layer (omni.cpp) via the omni_internal.h interface.
#include "omni-impl.h"
#include "omni.h"
#include "omni_internal.h"
#include "duplex_core.h"
#include "token2wav/token2wav-impl.h"
#include "common/common.h"
#include "common/sampling.h"

#include <cstring>
#include <future>

// ==========================================================================
// Duplex pipeline and high-level session implementation
// ==========================================================================

// This region serves duplex_mode only. In async mode frames flow through
// encoder_thread -> llm_thread -> TTS/T2W; the public stream/session APIs at
// the end of this file are the only entry points used by callers.

static void duplex_encoder_thread_func(omni_context * ctx_omni, common_params * params);
static void duplex_llm_thread_func(omni_context * ctx_omni, common_params * params);
static bool duplex_init_emb_cache(omni_context * ctx_omni, common_params * params);

bool duplex_prepare(omni_context * ctx_omni, common_params * params) {
    if (!ctx_omni || !ctx_omni->duplex_mode || !params) return false;
    if (ctx_omni->duplex == nullptr) ctx_omni->duplex = new DuplexPipeline();
    if (!ctx_omni->duplex->emb_cache.initialized) {
        duplex_init_emb_cache(ctx_omni, params);
    }
    return ctx_omni->duplex->emb_cache.initialized;
}

// 启动 duplex 双线程；幂等。
void duplex_start_threads(omni_context * ctx_omni, common_params * params) {
    if (!ctx_omni || !ctx_omni->duplex_mode) return;

    duplex_prepare(ctx_omni, params);
    DuplexPipeline * dup = ctx_omni->duplex;

    if (dup->running.load()) return;  // 已在运行

    dup->running.store(true);

    // Stage 3: 懒加载 special-token embedding。失败不是致命错误，fused 路径会自动回退。
    if (!dup->emb_cache.initialized) {
        duplex_init_emb_cache(ctx_omni, params);
    }

    if (!dup->encoder_thread.joinable()) {
        dup->encoder_thread = std::thread(duplex_encoder_thread_func, ctx_omni, params);
        print_with_timestamp("Duplex: encoder_thread created\n");
    }
    if (!dup->llm_thread.joinable()) {
        dup->llm_thread = std::thread(duplex_llm_thread_func, ctx_omni, params);
        print_with_timestamp("Duplex: llm_thread created\n");
    }
}

// 停止 duplex 双线程、清理队列。omni_free 中调用。
void duplex_stop_threads(omni_context * ctx_omni) {
    if (!ctx_omni || ctx_omni->duplex == nullptr) return;
    DuplexPipeline * dup = ctx_omni->duplex;

    dup->running.store(false);
    dup->encoder_cv.notify_all();
    dup->llm_cv.notify_all();
    dup->decode_done_cv.notify_all();

    if (dup->encoder_thread.joinable()) dup->encoder_thread.join();
    if (dup->llm_thread.joinable())     dup->llm_thread.join();

    // 清理残留
    while (!dup->encoder_queue.empty()) {
        delete dup->encoder_queue.front();
        dup->encoder_queue.pop();
    }
    while (!dup->prefill_queue.empty()) {
        delete dup->prefill_queue.front();
        dup->prefill_queue.pop();
    }
    if (dup->pending_decode) {
        dup->pending_decode->done.store(true);
        dup->pending_decode->ok.store(false);
        dup->decode_done_cv.notify_all();
        dup->pending_decode = nullptr;
    }

    delete dup;
    ctx_omni->duplex = nullptr;
    print_with_timestamp("Duplex: threads stopped & pipeline destroyed\n");
}

// ---------------------------------------------------------------------------
// encoder thread：从 encoder_queue 取请求，做 VPM/APM 编码，打包入 prefill_queue
// 阶段 2：VPM 和 APM 用 std::async 真并行。
//   ctx_vision 与 ctx_audio 各自独立 backend / buffer，不共享状态，
//   所以 CPU 侧启动可以完全并行；GPU 侧则由 Metal scheduler 决定实际并行度
//   （两个独立 backend 实例会各自维护 command queue）。
// 探针：打印 VPM / APM / pack 的分项耗时，便于对比。
// ---------------------------------------------------------------------------
static void duplex_encoder_thread_func(omni_context * ctx_omni, common_params * params) {
    print_with_timestamp("Duplex: encoder_thread started (Stage 2: VPM || APM)\n");
    DuplexPipeline * dup = ctx_omni->duplex;
    const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));

    while (dup->running.load()) {
        DuplexEncodeReq * req = nullptr;
        {
            std::unique_lock<std::mutex> lk(dup->encoder_mtx);
            dup->encoder_cv.wait(lk, [&]{
                return !dup->encoder_queue.empty() || !dup->running.load();
            });
            if (!dup->running.load()) break;
            req = dup->encoder_queue.front();
            dup->encoder_queue.pop();
        }
        dup->encoder_cv.notify_all();  // 唤醒可能在等队列空位的 producer
        if (!req) continue;

        DuplexPrefillPacket * packet = new DuplexPrefillPacket();
        packet->index = req->index;

        const bool has_img = !req->img_fname.empty() && ctx_omni->ctx_vision != nullptr;
        const bool has_aud = !req->aud_fname.empty();

        auto t_enc_begin = std::chrono::high_resolution_clock::now();

        // vision_set_max_slice_nums 只改 ctx_vision 的一个配置字段，不影响 ctx_audio，
        // 所以放在 VPM 任务启动之前（主线程）设置是安全的；若放 VPM 任务内部设置，
        // 也不会被 APM 看到。
        if (has_img && req->max_slice_nums >= 1) {
            vision_set_max_slice_nums(ctx_omni->ctx_vision, req->max_slice_nums);
        }

        // ---- VPM 任务 ----
        auto vpm_task = [&]() -> double {
            if (!has_img) return 0.0;
            auto t0 = std::chrono::high_resolution_clock::now();
            if (!omni_image_embed_make_chunks_with_filename(
                    ctx_omni->ctx_vision,
                    params->cpuparams.n_threads,
                    req->img_fname,
                    packet->vision_embed)) {
                LOG_ERR("Duplex encoder: vision encode failed for %s\n",
                        req->img_fname.c_str());
                packet->vision_embed.clear();
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        // ---- APM 任务 ----
        auto apm_task = [&]() -> double {
            if (!has_aud) return 0.0;
            auto t0 = std::chrono::high_resolution_clock::now();
            auto * audio_embeds = omni_audio_embed_make_with_filename(
                ctx_omni->ctx_audio, params->cpuparams.n_threads, req->aud_fname);
            if (audio_embeds != nullptr && audio_embeds->n_pos > 0) {
                packet->audio_embed.resize(audio_embeds->n_pos * hidden_size);
                std::memcpy(packet->audio_embed.data(), audio_embeds->embed,
                            packet->audio_embed.size() * sizeof(float));
                omni_embed_free(audio_embeds);
            } else {
                LOG_WRN("Duplex encoder: audio encode returned empty for %s\n",
                        req->aud_fname.c_str());
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        double vpm_ms = 0.0, apm_ms = 0.0;
        if (has_img && has_aud) {
            // 真并行：APM 放到独立线程，VPM 在当前线程跑
            auto apm_future = std::async(std::launch::async, apm_task);
            vpm_ms = vpm_task();
            apm_ms = apm_future.get();
        } else if (has_img) {
            vpm_ms = vpm_task();
        } else if (has_aud) {
            apm_ms = apm_task();
        }

        auto t_enc_end = std::chrono::high_resolution_clock::now();
        double enc_wall_ms = std::chrono::duration<double, std::milli>(t_enc_end - t_enc_begin).count();

        print_with_timestamp(
            "[prof] encoder index=%d VPM=%.1fms APM=%.1fms wall=%.1fms parallel_savings=%.1fms\n",
            req->index, vpm_ms, apm_ms, enc_wall_ms,
            (vpm_ms + apm_ms) - enc_wall_ms);

        delete req;

        // ---- push to llm prefill_queue ----
        {
            std::unique_lock<std::mutex> lk(dup->llm_mtx);
            dup->llm_cv.wait(lk, [&]{
                return dup->prefill_queue.size() < DuplexPipeline::PREFILL_QUEUE_CAP
                    || !dup->running.load();
            });
            if (!dup->running.load()) {
                delete packet;
                break;
            }
            dup->prefill_queue.push(packet);
        }
        dup->llm_cv.notify_all();
    }
    print_with_timestamp("Duplex: encoder_thread stopped\n");
}

// ---------------------------------------------------------------------------
// llm thread 的辅助：把 prefill packet 按多段 eval 序列写入 KV cache。
// 这是 fused prefill（duplex_do_prefill_one_fused）失败时的回退路径。
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Stage 3: 从 LLM gguf 读 token_embd.weight，把 duplex prefill 里用到的
// 固定 special-token 字符串 tokenize 后 dequantize 成 fp32 embedding，
// 供 fused prefill 路径一次性拼入 embed buffer。
//
// 内存占用：最多 10 来个 token × 4096 float × 4B ≈ 200KB。
// 失败（gguf 打不开 / tensor 类型不支持 dequantize）时返回 false，
// fused prefill 自动回退到多段 llama_decode 的老路径。
// ---------------------------------------------------------------------------
static bool duplex_dequantize_row(const struct ggml_tensor * t, int64_t row_idx,
                                  int64_t hidden_size, std::vector<float> & out) {
    const auto * traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) {
        LOG_WRN("duplex_emb_cache: type %d has no to_float traits\n", (int)t->type);
        return false;
    }
    // ne[0]=hidden_size, ne[1]=vocab_size, memory layout row-major [vocab, hidden]
    // 每行字节数 = (hidden_size / blck_size) * type_size
    if (traits->blck_size <= 0 || hidden_size % traits->blck_size != 0) {
        LOG_WRN("duplex_emb_cache: hidden %lld not divisible by blck_size %lld\n",
                (long long)hidden_size, (long long)traits->blck_size);
        return false;
    }
    const size_t bytes_per_row = (hidden_size / traits->blck_size) * traits->type_size;
    const char * row_data = (const char *)t->data + (size_t)row_idx * bytes_per_row;
    out.resize(hidden_size);
    traits->to_float(row_data, out.data(), hidden_size);
    return true;
}

static bool duplex_init_emb_cache(omni_context * ctx_omni, common_params * params) {
    if (!ctx_omni || !ctx_omni->duplex) return false;
    DuplexPipeline * dup = ctx_omni->duplex;
    if (dup->emb_cache.initialized) return true;

    const std::string gguf_path = params ? params->model.path : std::string();
    if (gguf_path.empty()) {
        LOG_WRN("duplex_emb_cache: params->model.path is empty\n");
        return false;
    }

    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &ctx_meta };
    struct gguf_context * ctx_gguf = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!ctx_gguf) {
        LOG_WRN("duplex_emb_cache: failed to open llm gguf '%s'\n", gguf_path.c_str());
        return false;
    }

    int64_t emb_idx = gguf_find_tensor(ctx_gguf, "token_embd.weight");
    if (emb_idx < 0) {
        LOG_WRN("duplex_emb_cache: token_embd.weight not found\n");
        gguf_free(ctx_gguf); if (ctx_meta) ggml_free(ctx_meta); return false;
    }
    struct ggml_tensor * emb_t = ggml_get_tensor(ctx_meta, "token_embd.weight");
    if (!emb_t) {
        LOG_WRN("duplex_emb_cache: ggml_get_tensor failed\n");
        gguf_free(ctx_gguf); if (ctx_meta) ggml_free(ctx_meta); return false;
    }

    int64_t hidden = emb_t->ne[0];
    int64_t vocab  = emb_t->ne[1];
    const int llm_hidden = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    if ((int64_t)llm_hidden != hidden) {
        LOG_WRN("duplex_emb_cache: hidden mismatch (llm=%d, gguf=%lld)\n",
                llm_hidden, (long long)hidden);
        gguf_free(ctx_gguf);
        if (ctx_meta) ggml_free(ctx_meta);
        return false;
    }
    dup->emb_cache.hidden_size = (int)hidden;

    // 需要缓存的固定字符串。为了和老 duplex_do_prefill_one 严格对齐，
    // 每个字符串都用 llama_tokenize 拿到 token id 序列，逐 row dequantize 拼接。
    const std::vector<std::string> fixed_texts = {
        "<unit><image>", "</image>",
        "<slice>",       "</slice>",
        "\n",
        "<unit>",        // 纯音频 unit 的前缀
    };

    int total_rows_loaded = 0;
    for (const auto & text : fixed_texts) {
        // 和 eval_string 里的 tokenize 参数保持一致（add_special=false, parse_special=true）。
        std::vector<llama_token> toks =
            common_tokenize(ctx_omni->ctx_llama, text, false, true);
        if (toks.empty()) {
            LOG_WRN("duplex_emb_cache: tokenize '%s' got 0 tokens\n", text.c_str());
            continue;
        }
        std::vector<float> buf;
        buf.reserve(toks.size() * (size_t)hidden);
        bool ok = true;
        for (auto id : toks) {
            if (id < 0 || (int64_t)id >= vocab) {
                LOG_WRN("duplex_emb_cache: token id %d out of vocab %lld\n",
                        (int)id, (long long)vocab);
                ok = false; break;
            }
            std::vector<float> row;
            if (!duplex_dequantize_row(emb_t, id, hidden, row)) { ok = false; break; }
            buf.insert(buf.end(), row.begin(), row.end());
            total_rows_loaded++;
        }
        if (!ok) continue;
        dup->emb_cache.text_ntok[text] = (int)toks.size();
        dup->emb_cache.text_emb[text]  = std::move(buf);
    }

    // 注意：先在释放 ctx_meta 之前拷贝所有需要的 metadata，避免 use-after-free
    const char * type_name_cached = ggml_type_name(emb_t->type);
    std::string type_name_str(type_name_cached ? type_name_cached : "?");

    gguf_free(ctx_gguf);
    if (ctx_meta) ggml_free(ctx_meta);

    if (dup->emb_cache.text_emb.empty()) {
        LOG_WRN("duplex_emb_cache: no text embed cached, fused prefill disabled\n");
        return false;
    }

    dup->emb_cache.initialized = true;
    print_with_timestamp(
        "[Duplex Stage 3] emb_cache initialized: %zu texts, %d rows, hidden=%d, src=%s\n",
        dup->emb_cache.text_emb.size(), total_rows_loaded, (int)hidden,
        type_name_str.c_str());
    return true;
}

// 把 text 对应的 embedding buffer 追加到 fused 末尾
static inline bool append_text_emb(DuplexPipeline * dup,
                                   const std::string & text,
                                   std::vector<float> & fused,
                                   int & fused_ntok) {
    auto it = dup->emb_cache.text_emb.find(text);
    if (it == dup->emb_cache.text_emb.end()) return false;
    fused.insert(fused.end(), it->second.begin(), it->second.end());
    fused_ntok += dup->emb_cache.text_ntok[text];
    return true;
}

static inline void append_raw_emb(std::vector<float> & fused,
                                  int & fused_ntok,
                                  const float * src, int n_tokens,
                                  int hidden_size) {
    fused.insert(fused.end(), src, src + (size_t)n_tokens * hidden_size);
    fused_ntok += n_tokens;
}

// ---------------------------------------------------------------------------
// Stage 3: fused prefill 路径。
// 把一个 chunk 的 "<unit><image>" + vision_embed + "</image>" + (slices)
// + audio_embed 拼成单块 buffer，一次 prefill_with_emb 写入 KV，
// 消除原来 5~7 次小 llama_decode 的 kernel launch 开销。
//
// 如果 emb_cache 未初始化，回退到原始多段路径（duplex_do_prefill_one）。
// ---------------------------------------------------------------------------
static bool duplex_do_prefill_one_fused(omni_context * ctx_omni, common_params * params,
                                        DuplexPrefillPacket * packet,
                                        int hidden_size,
                                        int * out_fused_ntok = nullptr) {
    DuplexPipeline * dup = ctx_omni->duplex;
    if (!dup->emb_cache.initialized) return false;
    if (dup->emb_cache.hidden_size != hidden_size) return false;

    auto t_begin  = std::chrono::high_resolution_clock::now();
    int  n_past_0 = ctx_omni->n_past;

    int n_audio_tokens = (hidden_size > 0)
        ? (int)(packet->audio_embed.size() / hidden_size) : 0;
    bool has_audio  = (n_audio_tokens > 0);
    bool has_vision = !packet->vision_embed.empty();

    if (ctx_omni->sliding_window_config.mode != "off") {
        sliding_window_register_unit_start(ctx_omni);
    }

    std::vector<float> fused;
    fused.reserve(128 * (size_t)hidden_size);
    int fused_ntok = 0;

    bool build_ok = true;
    if (has_vision) {
        int n_chunks         = (int)packet->vision_embed.size();
        int tokens_per_chunk = (int)packet->vision_embed[0].size() / hidden_size;
        bool has_slices      = (n_chunks > 1);

        build_ok &= append_text_emb(dup, "<unit><image>", fused, fused_ntok);
        if (build_ok) {
            append_raw_emb(fused, fused_ntok,
                           packet->vision_embed[0].data(), tokens_per_chunk, hidden_size);
        }
        build_ok &= append_text_emb(dup, "</image>", fused, fused_ntok);
        if (build_ok && has_slices) {
            for (int i = 1; i < n_chunks && build_ok; i++) {
                build_ok &= append_text_emb(dup, "<slice>", fused, fused_ntok);
                if (build_ok) {
                    append_raw_emb(fused, fused_ntok,
                                   packet->vision_embed[i].data(), tokens_per_chunk, hidden_size);
                }
                build_ok &= append_text_emb(dup, "</slice>", fused, fused_ntok);
            }
            build_ok &= append_text_emb(dup, "\n", fused, fused_ntok);
        }
        if (has_audio && build_ok) {
            append_raw_emb(fused, fused_ntok,
                           packet->audio_embed.data(), n_audio_tokens, hidden_size);
        }
    } else {
        build_ok &= append_text_emb(dup, "<unit>", fused, fused_ntok);
        if (has_audio && build_ok) {
            append_raw_emb(fused, fused_ntok,
                           packet->audio_embed.data(), n_audio_tokens, hidden_size);
        }
    }

    if (!build_ok) {
        // 某个特殊串缓存缺失，放弃 fused 路径、让调用方走 fallback。
        LOG_WRN("duplex_fused: cache miss, fallback to multi-step prefill\n");
        return false;
    }
    if (out_fused_ntok != nullptr) *out_fused_ntok = fused_ntok;
    print_with_timestamp("[LLM PREFILL] n_past_before=%d n_tokens=%d batch=%d kv_offset=%d topology=fused\n",
                         n_past_0, fused_ntok, params->n_batch, n_past_0);
    auto t_after_embed = std::chrono::high_resolution_clock::now();

    // One shot prefill: the fused token count is input-dependent.  The current
    // validation case uses 77; do not turn that case into a global invariant.
    if (!prefill_with_emb(ctx_omni, params, fused.data(),
                          fused_ntok, params->n_batch, &ctx_omni->n_past)) {
        LOG_ERR("duplex_fused: prefill_with_emb failed\n");
        return false;
    }
    auto t_after_llm = std::chrono::high_resolution_clock::now();

    if (ctx_omni->sliding_window_config.mode != "off") {
        std::string input_type = has_vision ? "omni" : "audio";
        sliding_window_register_unit_end(ctx_omni, input_type, {}, false);
        sliding_window_enforce(ctx_omni);
    }
    auto t_after_sliding = std::chrono::high_resolution_clock::now();

    // [计时修正] llama_decode 内部走 ggml_backend_sched_graph_compute_async，
    // 异步提交、不同步。若不在此等待，prefill 的 GPU 完成时间会被紧随其后的
    // decode 第一次 llama_get_logits_ith（内部 llama_synchronize）吞进 decode 的
    // 计时窗口，造成 prefill/decode 的 ms 归属漂移。显式同步：让 prefill ms 含
    // GPU 完成，decode ms 只计自身；性能中性。
    llama_synchronize(ctx_omni->ctx_llama);

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
    double ms_embed   = std::chrono::duration<double, std::milli>(t_after_embed  - t_begin).count();
    double ms_llm     = std::chrono::duration<double, std::milli>(t_after_llm    - t_after_embed).count();
    double ms_sliding = std::chrono::duration<double, std::milli>(t_after_sliding- t_after_llm).count();
    double ms_sync    = std::chrono::duration<double, std::milli>(t_end - t_after_sliding).count();
    print_with_timestamp(
        "[prof] llm prefill (fused) n_past=%d->%d tokens=%d ms=%.1f [embed=%.1f llm=%.1f sliding=%.1f sync=%.1f]\n",
        n_past_0, ctx_omni->n_past,
        ctx_omni->n_past - n_past_0, ms, ms_embed, ms_llm, ms_sliding, ms_sync);
    return true;
}

static void duplex_do_prefill_one(omni_context * ctx_omni, common_params * params,
                                  DuplexPrefillPacket * packet,
                                  int hidden_size) {
    auto t_begin   = std::chrono::high_resolution_clock::now();
    int  n_past_0  = ctx_omni->n_past;

    // 单 packet 版本：语义上等价于原 batch.size()==1 的情况。
    // duplex 下 prefill/decode 必须严格 1-1 配对，否则 decode 的 KV 上下文会被
    // 后续 chunk 的 prefill 污染；因此 llm_thread 每轮只消费 1 个 packet。
    {
        // 注册 unit 开始（滑窗记账）
        if (ctx_omni->sliding_window_config.mode != "off") {
            sliding_window_register_unit_start(ctx_omni);
        }

        int n_audio_tokens = (hidden_size > 0)
            ? (int)(packet->audio_embed.size() / hidden_size) : 0;
        bool has_audio  = (n_audio_tokens > 0);
        bool has_vision = !packet->vision_embed.empty();

        if (has_vision) {
            int n_chunks         = (int)packet->vision_embed.size();
            int tokens_per_chunk = (int)packet->vision_embed[0].size() / hidden_size;
            bool has_slices      = (n_chunks > 1);

            // duplex 格式：<unit><image>[overview]</image>(<slice>[slice]</slice>)*\n[audio]
            eval_string(ctx_omni, params, "<unit><image>",
                        params->n_batch, &ctx_omni->n_past, false);
            prefill_with_emb(ctx_omni, params, packet->vision_embed[0].data(),
                             tokens_per_chunk, params->n_batch, &ctx_omni->n_past);
            eval_string(ctx_omni, params, "</image>",
                        params->n_batch, &ctx_omni->n_past, false);
            if (has_slices) {
                for (int i = 1; i < n_chunks; i++) {
                    eval_string(ctx_omni, params, "<slice>",
                                params->n_batch, &ctx_omni->n_past, false);
                    prefill_with_emb(ctx_omni, params, packet->vision_embed[i].data(),
                                     tokens_per_chunk, params->n_batch, &ctx_omni->n_past);
                    eval_string(ctx_omni, params, "</slice>",
                                params->n_batch, &ctx_omni->n_past, false);
                }
                eval_string(ctx_omni, params, "\n",
                            params->n_batch, &ctx_omni->n_past, false);
            }
            if (has_audio) {
                // duplex 下 audio 直接跟在 vision 后面（无 audio_start/end 标签）
                prefill_with_emb(ctx_omni, params, packet->audio_embed.data(),
                                 n_audio_tokens, params->n_batch, &ctx_omni->n_past);
            }
        } else {
            // 纯音频 unit
            eval_string(ctx_omni, params, "<unit>",
                        params->n_batch, &ctx_omni->n_past, false);
            if (has_audio) {
                prefill_with_emb(ctx_omni, params, packet->audio_embed.data(),
                                 n_audio_tokens, params->n_batch, &ctx_omni->n_past);
            }
        }

        if (ctx_omni->sliding_window_config.mode != "off") {
            std::string input_type = has_vision ? "omni" : "audio";
            sliding_window_register_unit_end(ctx_omni, input_type, {}, false);
        }
    }

    if (ctx_omni->sliding_window_config.mode != "off") {
        sliding_window_enforce(ctx_omni);
    }

    // [计时修正] 同 duplex_do_prefill_one_fused：显式同步，让 prefill ms 含 GPU 完成，
    // 避免 prefill 的 GPU 时间被 decode 的首次同步吞走（性能中性）。
    llama_synchronize(ctx_omni->ctx_llama);

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
    print_with_timestamp(
        "[prof] llm prefill n_past=%d->%d tokens=%d ms=%.1f\n",
        n_past_0, ctx_omni->n_past,
        ctx_omni->n_past - n_past_0, ms);
}

// ---------------------------------------------------------------------------
// llm thread 的辅助：执行一轮 duplex decode。
// 语义与原 stream_decode 的 duplex 分支一致（含尾部 response unit 注册与 round boundary），
// 但抽掉了 simplex 相关分支与冗余的 llama_mtx 加锁——新路径下 ctx_llama 已由 llm 线程独占。
// 返回 false 表示内部异常或被 break 打断。
// ---------------------------------------------------------------------------
static bool duplex_do_decode(omni_context * ctx_omni, common_params * params,
                             const std::string & debug_dir, int round_idx, DuplexDecodeReq * req) {
    // ---- 轮次同步（与老 stream_decode 对齐） ----
    if (round_idx >= 0) {
        if (ctx_omni->simplex_round_idx != round_idx) {
            print_with_timestamp("Duplex decode: sync round_idx %d -> %d\n",
                                 ctx_omni->simplex_round_idx, round_idx);
            ctx_omni->simplex_round_idx = round_idx;
            ctx_omni->wav_turn_base     = round_idx * 1000;
        }
    }

    ctx_omni->stream_decode_start_time = std::chrono::high_resolution_clock::now();
    auto t_dec_begin = ctx_omni->stream_decode_start_time;
    int  n_past_dec_0 = ctx_omni->n_past;

    print_with_timestamp("[DUPLEX CONTRACT] frame=%d stage=D ctx_owner=llm_thread n_past_before=%d n_keep=%d n_ctx=%d prefill_complete=1\n",
                         round_idx, ctx_omni->n_past, ctx_omni->n_keep, params->n_ctx);

    // duplex 下 llm_generation_done 不重置（与老路径对齐）
    ctx_omni->ended_with_listen = false;

    if (ctx_omni->break_event.load()) {
        ctx_omni->break_event.store(false);
        print_with_timestamp("Duplex decode: reset break_event at start\n");
    }

    // 清空上一轮 text 残留
    {
        std::lock_guard<std::mutex> lock(ctx_omni->text_mtx);
        ctx_omni->text_queue.clear();
        ctx_omni->text_done_flag  = false;
        ctx_omni->text_streaming  = true;
    }

    if (ctx_omni->use_tts) {
        ctx_omni->speek_done = false;
    }

    // decode 开始时的 cache 长度（prefill 已完成写入，这里取值才准确）
    int decode_start_cache_len = ctx_omni->n_past;

    // ---- force_listen：会话开局强制 LISTEN N 次 ----
    if (ctx_omni->force_listen_used < ctx_omni->force_listen_count) {
        ctx_omni->force_listen_used++;
        ctx_omni->slide_last_was_listen = true;
        ctx_omni->ended_with_listen     = true;
        ctx_omni->current_turn_ended    = false;
        if (ctx_omni->use_tts) {
            ctx_omni->speek_done = true;
        }
        print_with_timestamp("Duplex decode: force_listen %d/%d, emit __IS_LISTEN__\n",
                             ctx_omni->force_listen_used, ctx_omni->force_listen_count);
        {
            std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
            ctx_omni->text_queue.push_back("__IS_LISTEN__");
            ctx_omni->text_done_flag = true;
            ctx_omni->text_streaming = false;
            ctx_omni->text_cv.notify_all();
        }
        return true;
    }

    // ---- 采样主循环（搬自老 stream_decode duplex 分支） ----
    const int max_tgt_len = params->n_predict < 0 ? params->n_ctx : params->n_predict;
    const int step_size   = 10;   // chunk 推送给 TTS 的 LLM token 数量
    bool llm_finish                  = false;
    bool local_is_end_of_turn        = false;
    int  current_chunk_tokens        = 0;
    const int max_chunk_tokens       = ctx_omni->max_new_speak_tokens_per_chunk;
    bool chunk_limit_reached         = false;
    const int llm_n_embd             = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    int llm_chunk_idx                = 0;

    std::string response;
    bool discard_turn = false;
    for (int il = 0; il < max_tgt_len; ) {
        if (ctx_omni->break_event.load()) {
            llm_finish = true;
            discard_turn = true;
            break;
        }

        response.clear();
        int jl = 0;
        int total_tokens_generated = 0;
        std::vector<llama_token> chunk_token_ids;
        std::vector<float>       chunk_hidden_states;
        local_is_end_of_turn     = false;
        bool req_first_token_logged = false;

        while (jl < step_size && !llm_finish
               && !ctx_omni->break_event.load()
               && !chunk_limit_reached)
        {
            const char * tmp = nullptr;
            float *      hidden_states = nullptr;
            llama_token  sampled_token = 0;

            tmp = llama_loop_with_hidden_and_token(
                ctx_omni, params, ctx_omni->ctx_sampler,
                ctx_omni->n_past, hidden_states, sampled_token);

            total_tokens_generated++;

            // 本帧首个输出 token 的绝对 wall 时间（epoch ms）
            if (!req_first_token_logged) {
                req_first_token_logged = true;
                if (req != nullptr) {
                    req->t_first_token_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                }
            }

            if (tmp != nullptr && hidden_states != nullptr
                && is_valid_tts_token(sampled_token)) {
                chunk_token_ids.push_back(sampled_token);
                chunk_hidden_states.insert(chunk_hidden_states.end(),
                                           hidden_states, hidden_states + llm_n_embd);
                jl++;
                current_chunk_tokens++;
                if (max_chunk_tokens > 0 && current_chunk_tokens >= max_chunk_tokens) {
                    chunk_limit_reached = true;
                }
            }

            if (tmp == nullptr) {
                LOG_ERR("Duplex decode: llama_loop returned nullptr\n");
                discard_turn = true;
                llm_finish = true;
                break;
            }

            OmniTokenType token_type = get_token_type(ctx_omni, sampled_token);

            if (token_type == OmniTokenType::TURN_EOS
                || token_type == OmniTokenType::TTS_EOS
                || token_type == OmniTokenType::EOS) {
                local_is_end_of_turn           = true;
                ctx_omni->current_turn_ended   = true;
                print_with_timestamp("Duplex decode: turn_eos (type=%d), wait for chunk_eos\n",
                                     (int)token_type);
            } else if (token_type == OmniTokenType::LISTEN) {
                if (!ctx_omni->slide_last_was_listen.load()) {
                    local_is_end_of_turn = true;
                    print_with_timestamp("Duplex decode: LISTEN after SPEAK, flush TTS\n");
                }
            }

            if (is_end_token(ctx_omni, sampled_token)) {
                llm_finish = true;

                if (token_type == OmniTokenType::TURN_EOS
                    || token_type == OmniTokenType::TTS_EOS
                    || token_type == OmniTokenType::EOS) {
                    ctx_omni->current_turn_ended = true;
                }

                if (token_type == OmniTokenType::LISTEN) {
                    ctx_omni->ended_with_listen     = true;
                    ctx_omni->slide_last_was_listen = true;
                    {
                        std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
                        ctx_omni->text_queue.push_back("__IS_LISTEN__");
                        ctx_omni->text_cv.notify_all();
                    }
                } else {
                    ctx_omni->slide_last_was_listen = false;
                }
                break;  // 不把结束 token 加入 response
            }

            response += std::string(tmp);
        }

        if (ctx_omni->break_event.load()) {
            // Do not publish a partially generated chunk after an asynchronous
            // break.  The KV state is owned by this worker and is left at the
            // observed position; the next request starts from the normal
            // break/reset path.
            discard_turn = true;
            llm_finish = true;
            response.clear();
            chunk_token_ids.clear();
            chunk_hidden_states.clear();
        }

        // chunk 限制：强制补 <|chunk_eos|> token 结束本轮
        if (chunk_limit_reached && !discard_turn) {
            if (ctx_omni->special_token_chunk_eos >= 0) {
                std::vector<llama_token> t = {ctx_omni->special_token_chunk_eos};
                eval_tokens(ctx_omni, params, t, params->n_batch, &ctx_omni->n_past);
            }
            llm_finish            = true;
            current_chunk_tokens  = 0;
        }

        // duplex chunk 末尾写入 </unit>
        if (ctx_omni->special_token_unit_end >= 0 && !discard_turn) {
            std::vector<llama_token> t = {ctx_omni->special_token_unit_end};
            eval_tokens(ctx_omni, params, t, params->n_batch, &ctx_omni->n_past);
        }

        il += total_tokens_generated;

        // 清理 response 中的特殊结束 token
        {
            static const std::vector<std::string> end_tokens = {
                "<|tts_eos|>", "</s>", "<|listen|>", "<|turn_eos|>",
                "<|chunk_eos|>", "<|chunk_tts_eos|>"
            };
            for (const auto & t : end_tokens) {
                size_t p = response.find(t);
                if (p != std::string::npos) response = response.substr(0, p);
            }
            size_t speak_pos = response.find("<|speak|>");
            while (speak_pos != std::string::npos) {
                response.erase(speak_pos, std::string("<|speak|>").length());
                speak_pos = response.find("<|speak|>");
            }
        }

        // 推送到 text_queue（SSE）
        if (!discard_turn && !response.empty()) {
            std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
            ctx_omni->text_queue.push_back(response);
            ctx_omni->text_cv.notify_all();
        }

        // 推送到 TTS queue
        if (!discard_turn && ctx_omni->use_tts && ctx_omni->tts_thread_info
            && (!response.empty() || llm_finish))
        {
            LLMOut * llm_out = new LLMOut();
            llm_out->text            = response;
            llm_out->n_past          = ctx_omni->n_past;
            llm_out->llm_finish      = llm_finish;
            llm_out->debug_dir       = debug_dir;
            llm_out->token_ids       = chunk_token_ids;
            llm_out->hidden_states   = chunk_hidden_states;
            llm_out->n_embd          = llm_n_embd;
            llm_out->is_end_of_turn  = local_is_end_of_turn;
            llm_out->frame_id        = round_idx >= 0 ? (uint64_t) round_idx : 0;
            llm_out->chunk_idx       = llm_chunk_idx;

            {
                std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
                ctx_omni->tts_thread_info->cv.wait(lock, [&]{
                    return ctx_omni->tts_thread_info->queue.size()
                           < (size_t)ctx_omni->tts_thread_info->MAX_QUEUE_SIZE;
                });
                // duplex 模式下无论 speek_done 都推送
                llm_out->enqueue_time = std::chrono::steady_clock::now();
                ctx_omni->tts_thread_info->queue.push(llm_out);
            }
            ctx_omni->tts_thread_info->cv.notify_all();
            print_with_timestamp("[DUPLEX CONTRACT] frame=%d edge=LLMOut producer=D consumer=TTS chunk=%d tokens=%zu\n",
                                 round_idx, llm_chunk_idx, chunk_token_ids.size());
        }

        ++llm_chunk_idx;

        if (llm_finish) break;
    }

    // ---- 推送轮次结束标记 ----
    {
        std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
        if (!ctx_omni->ended_with_listen) {
            ctx_omni->text_queue.push_back("__END_OF_TURN__");
        }
        ctx_omni->text_done_flag = true;
        ctx_omni->text_cv.notify_all();
        ctx_omni->text_streaming = false;
    }

    // ---- response unit 注册 ----
    if (ctx_omni->sliding_window_config.mode != "off") {
        int response_len = ctx_omni->n_past - decode_start_cache_len;
        if (response_len > 0) {
            UnitEntry entry;
            entry.unit_id   = ctx_omni->next_unit_id++;
            entry.length    = response_len;
            entry.type      = "response";
            entry.is_listen = ctx_omni->ended_with_listen.load();
            entry.turn_id   = ctx_omni->current_turn_id;
            ctx_omni->unit_history.push_back(entry);
            print_with_timestamp(
                "[SW] duplex register response unit: unit_id=%d len=%d is_listen=%d turn_id=%d\n",
                entry.unit_id, entry.length, (int)entry.is_listen, entry.turn_id);
        }
    }

    // ---- round boundary（duplex 下仅在 LISTEN 结束时记录） ----
    if (ctx_omni->ended_with_listen) {
        ctx_omni->round_start_positions.push_back(ctx_omni->n_past);
        print_with_timestamp("Duplex decode: round boundary at n_past=%d, total=%zu\n",
                             ctx_omni->n_past, ctx_omni->round_start_positions.size());
        ctx_omni->current_turn_id++;
    }

    auto t_dec_end = std::chrono::high_resolution_clock::now();
    double dec_ms = std::chrono::duration<double, std::milli>(t_dec_end - t_dec_begin).count();
    print_with_timestamp(
        "[prof] llm decode n_past=%d->%d tokens=%d ms=%.1f listen=%d\n",
        n_past_dec_0, ctx_omni->n_past, ctx_omni->n_past - n_past_dec_0,
        dec_ms, (int)ctx_omni->ended_with_listen.load());
    return true;
}

// ---------------------------------------------------------------------------
// llm thread 主循环
//
// 严格 1:1 配对语义：每次 pending_decode 出现时，最多消费 *一个* prefill packet
// 然后做 decode。这是 duplex 正确性的关键：
//   - 如果 drain 多个 packet 再 decode，KV 会被后续 chunk 的 prefill 污染；
//   - 如果 decode 先于对应 prefill 执行，KV 缺少本轮 chunk 的上下文。
//
// chunk 0（index=0，system prompt 初始化轮）由 stream_prefill 的老路径直接
// 在 ctx_llama 上 eval，不走 encoder/llm_thread，此时 in_flight=0，
// duplex_decode 进来时 llm_thread 直接 decode 不消费 packet。
//
// chunk N (N≥1): duplex_prefill 把 encode req 推给 encoder_thread，
// encoder_thread 编完 push 到 prefill_queue（in_flight 计数从 +1 到
// llm_thread 消费后 -1）。duplex_decode push pending_decode，
// llm_thread 看到 pending_decode && in_flight>0 时，wait prefill_queue 非空
// 并消费 1 个 packet 做 LLM prefill，然后 decode。
// ---------------------------------------------------------------------------
static void duplex_llm_thread_func(omni_context * ctx_omni, common_params * params) {
    print_with_timestamp("Duplex: llm_thread started (strict 1:1 prefill/decode)\n");
    DuplexPipeline * dup = ctx_omni->duplex;
    const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));

    while (dup->running.load()) {
        DuplexDecodeReq * decode_req = nullptr;

        // ---- Phase 0: 等待 decode 请求 ----
        {
            std::unique_lock<std::mutex> lk(dup->llm_mtx);
            dup->llm_cv.wait(lk, [&]{
                return dup->pending_decode != nullptr
                    || !dup->running.load()
                    || ctx_omni->break_event.load();
            });
            if (!dup->running.load()) break;

            // break_event：丢弃当前 prefill 和 decode 请求
            if (ctx_omni->break_event.load()) {
                int n_dropped = 0;
                while (!dup->prefill_queue.empty()) {
                    delete dup->prefill_queue.front();
                    dup->prefill_queue.pop();
                    n_dropped++;
                }
                if (n_dropped > 0) {
                    dup->in_flight_prefill.fetch_sub(n_dropped);
                    dup->in_flight_cv.notify_all();
                }
                if (dup->pending_decode) {
                    dup->pending_decode->done.store(true);
                    dup->pending_decode->ok.store(false);
                    dup->pending_decode = nullptr;
                    dup->decode_done_cv.notify_all();
                }
                lk.unlock();
                dup->llm_cv.notify_all();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            decode_req = dup->pending_decode;
            dup->pending_decode = nullptr;
        }
        dup->llm_cv.notify_all();  // 解锁 duplex_decode 里新来请求的 wait
        if (!decode_req) continue;

        // ---- Phase 1: 消费最多 1 个 prefill packet（如果 encoder 还在途） ----
        //
        // 判断"是否有对应的 prefill 在途"的方式：in_flight_prefill 计数。
        //   - chunk 0（老路径 prefill）：in_flight=0，直接 decode
        //   - chunk N (N≥1)：in_flight ≥ 1，必有 packet 在途；
        //     wait prefill_queue 非空，消费队头（encoder 是单线程 FIFO 顺序）
        if (dup->in_flight_prefill.load() > 0) {
            DuplexPrefillPacket * packet = nullptr;
            {
                std::unique_lock<std::mutex> lk(dup->llm_mtx);
                dup->llm_cv.wait(lk, [&]{
                    return !dup->prefill_queue.empty()
                        || !dup->running.load()
                        || ctx_omni->break_event.load();
                });
                if (!dup->running.load()) break;
                if (!ctx_omni->break_event.load() && !dup->prefill_queue.empty()) {
                    packet = dup->prefill_queue.front();
                    dup->prefill_queue.pop();
                }
            }
            dup->llm_cv.notify_all();  // encoder 在等 prefill_queue 腾位

            if (packet) {
                // Stage 3: 先试 fused（1 次 llama_decode），失败回退到老 5-7 段路径。
                if (!duplex_do_prefill_one_fused(ctx_omni, params, packet, hidden_size)) {
                    duplex_do_prefill_one(ctx_omni, params, packet, hidden_size);
                }
                print_with_timestamp("[DUPLEX CONTRACT] frame=%d stage=P ctx_owner=llm_thread kv_written=1 n_past_after=%d\n",
                                     packet->index, ctx_omni->n_past);
                delete packet;
                dup->in_flight_prefill.fetch_sub(1);
                dup->in_flight_cv.notify_all();
            }
        }

        // ---- Phase 2: decode ----
        if (!ctx_omni->break_event.load()) {
            bool ok = duplex_do_decode(ctx_omni, params,
                                       decode_req->debug_dir, decode_req->round_idx, decode_req);
            print_with_timestamp("[DUPLEX CONTRACT] frame=%d pair=P->D sequence=1:1 decode_ok=%d\n",
                                 decode_req->round_idx, ok ? 1 : 0);
            decode_req->ok.store(ok);
        } else {
            decode_req->ok.store(false);
        }
        decode_req->done.store(true);
        dup->decode_done_cv.notify_all();
    }
    print_with_timestamp("Duplex: llm_thread stopped\n");
}

// ---------------------------------------------------------------------------
// 对外入口：duplex_prefill / duplex_decode
// 由 stream_prefill / stream_decode 入口按 duplex_mode 路由调用
// ---------------------------------------------------------------------------
bool duplex_prefill(omni_context * ctx_omni,
                           const std::string & aud_fname,
                           const std::string & img_fname,
                           int index, int max_slice_nums) {
    if (!ctx_omni || !ctx_omni->duplex) {
        LOG_ERR("duplex_prefill: pipeline not initialized\n");
        return false;
    }
    DuplexPipeline * dup = ctx_omni->duplex;

    DuplexEncodeReq * req = new DuplexEncodeReq();
    req->aud_fname      = aud_fname;
    req->img_fname      = img_fname;
    req->index          = index;
    req->max_slice_nums = max_slice_nums;

    {
        std::unique_lock<std::mutex> lk(dup->encoder_mtx);
        dup->encoder_cv.wait(lk, [&]{
            return dup->encoder_queue.size() < DuplexPipeline::ENCODER_QUEUE_CAP
                || !dup->running.load();
        });
        if (!dup->running.load()) {
            delete req;
            return false;
        }
        dup->encoder_queue.push(req);
    }
    // 计数提交的 prefill；llm_thread 处理完会递减。
    // 必须在 notify encoder 之前 ++，否则 llm_thread 可能在 notify 前就把 prefill
    // 处理完并 decrement 到 -1。
    dup->in_flight_prefill.fetch_add(1);
    dup->encoder_cv.notify_all();
    return true;
}

bool duplex_decode(omni_context * ctx_omni,
                          const std::string & debug_dir,
                          int round_idx,
                          double * out_t_first_token_ms) {
    if (!ctx_omni || !ctx_omni->duplex) {
        LOG_ERR("duplex_decode: pipeline not initialized\n");
        return false;
    }
    DuplexPipeline * dup = ctx_omni->duplex;

    // 直接 push pending_decode，"等对应 prefill 到达"的责任由 llm_thread 承担。
    // 这样 duplex_decode 返回速度只受 LLM prefill+decode 影响，不再等 encoder。
    // encoder 可以继续提前编码后续 chunk 的 embed，把 VPM 时间隐藏在 LLM 工作流里。
    DuplexDecodeReq req;
    req.debug_dir = debug_dir;
    req.round_idx = round_idx;

    {
        std::unique_lock<std::mutex> lk(dup->llm_mtx);
        dup->decode_done_cv.wait(lk, [&]{
            return dup->pending_decode == nullptr || !dup->running.load();
        });
        if (!dup->running.load()) return false;
        dup->pending_decode = &req;
    }
    dup->llm_cv.notify_all();

    // 等 llm_thread 完成本轮 decode（包含消费 1 个 prefill packet + decode 采样）
    {
        std::unique_lock<std::mutex> lk(dup->llm_mtx);
        dup->decode_done_cv.wait(lk, [&]{
            return req.done.load() || !dup->running.load();
        });
    }
    if (out_t_first_token_ms != nullptr) {
        *out_t_first_token_ms = req.t_first_token_ms;
    }
    return req.ok.load();
}

// ============================================================================
// ===== END DUPLEX PIPELINE ==================================================
// ============================================================================

bool stream_prefill(struct omni_context * ctx_omni, std::string aud_fname, std::string img_fname, int index, int max_slice_nums, std::string text) {

    // 路由：duplex_mode && async && index > 0 && system prompt 已初始化时，走 duplex 路径。
    // index == 0（system prompt 初始化、voice cloning ref_audio prefill、线程启动）
    // 仍走下方主体；线程启动点在其末尾调用 duplex_start_threads。
    // 下方主体（本次 if 之后的部分）是非 duplex 的旧同步/异步实现。
    if (ctx_omni->duplex_mode && ctx_omni->async && index > 0
        && ctx_omni->system_prompt_initialized && ctx_omni->duplex != nullptr) {
        return duplex_prefill(ctx_omni, aud_fname, img_fname, index, max_slice_nums);
    }

    // 只有在新一轮开始时 (index == 0) 才需要等待上一轮 TTS 完成
    // 同一轮内的后续 prefill (index >= 1) 不需要等待
    if (ctx_omni->use_tts && index == 0 && ctx_omni->warmup_done.load() && !ctx_omni->duplex_mode) {
        // 如果 break_event 已触发，跳过等待（上一轮已被打断）
        if (ctx_omni->break_event.load()) {
            print_with_timestamp("TTS: break_event active, skipping wait for previous round\n");
            ctx_omni->speek_done = true;
            ctx_omni->break_event.store(false);
            speek_cv.notify_all();
        }
        print_with_timestamp("TTS: 等待上一轮语音生成完成\n");
        std::unique_lock<std::mutex> lock(speek_mtx);
        // 添加超时等待，避免永久卡住
        auto wait_result = speek_cv.wait_for(lock, std::chrono::seconds(5), [&]{return ctx_omni->speek_done || ctx_omni->break_event.load(); });
        if (!wait_result) {
            // 强制设置为 true 以继续
            ctx_omni->speek_done = true;
        }
        // 等待完成后重置 speek_done，为下一轮做准备
        ctx_omni->speek_done = false;
        
        // 清理 TTS 队列残留数据，避免与新轮次混淆
        if (ctx_omni->tts_thread_info && !ctx_omni->duplex_mode) {
            std::lock_guard<std::mutex> tts_lock(ctx_omni->tts_thread_info->mtx);
            auto& tts_queue = ctx_omni->tts_thread_info->queue;
            while (!tts_queue.empty()) {
                LLMOut* old_out = tts_queue.front();
                tts_queue.pop();
                delete old_out;
            }
            print_with_timestamp("stream_prefill: cleared TTS queue for new turn\n");
        }
    } else if (ctx_omni->use_tts && index == 0 && !ctx_omni->duplex_mode) {
        // 否则 LLM 输出会被丢弃（因为 speek_done 初始值为 true）
        ctx_omni->speek_done = false;
        
        // 首次初始化时也要清理队列
        if (ctx_omni->tts_thread_info) {
            std::lock_guard<std::mutex> tts_lock(ctx_omni->tts_thread_info->mtx);
            auto& tts_queue = ctx_omni->tts_thread_info->queue;
            while (!tts_queue.empty()) {
                LLMOut* old_out = tts_queue.front();
                tts_queue.pop();
                delete old_out;
            }
        }
    } else if (ctx_omni->use_tts) {
    }
    
    // ctx_omni->need_speek = false;
    const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    
    std::string voice_clone_prompt = "";
    std::string assistant_prompt = "";
    
    if (ctx_omni->media_type == 1){ // audio
        // 如果 audio_voice_clone_prompt 以 "<|" 开头（特殊 token），不添加前缀
        // 这允许完全控制 prompt 格式（例如使用 system 而不是 user）
        if (ctx_omni->audio_voice_clone_prompt.substr(0, 2) == "<|") {
            voice_clone_prompt = ctx_omni->audio_voice_clone_prompt;
        } else {
            voice_clone_prompt = "<|im_start|>user\n" + ctx_omni->audio_voice_clone_prompt;
        }
        // 如果 audio_assistant_prompt 以 "<|" 开头（特殊 token），不添加前缀
        if (ctx_omni->audio_assistant_prompt.substr(0, 2) == "<|") {
            assistant_prompt = ctx_omni->audio_assistant_prompt;
        } else {
            assistant_prompt = "<|im_start|>user\n" + ctx_omni->audio_assistant_prompt;
        }
    }
    else if (ctx_omni->media_type == 2){ // omni
        if (ctx_omni->omni_voice_clone_prompt.substr(0, 2) == "<|") {
            voice_clone_prompt = ctx_omni->omni_voice_clone_prompt;
        } else {
            voice_clone_prompt = "<|im_start|>user\n" + ctx_omni->omni_voice_clone_prompt;
        }
        if (ctx_omni->omni_assistant_prompt.substr(0, 2) == "<|") {
            assistant_prompt = ctx_omni->omni_assistant_prompt;
        } else {
            assistant_prompt = "<|im_start|>user\n" + ctx_omni->omni_assistant_prompt;
        }
    }
    // 这是因为 omni_init 中可能会调用 stream_prefill(voice_audio, "", 0)，
    // 然后测试脚本又会调用 stream_prefill(audio_0, "", 0)
    // 如果不检查这个标志，系统 prompt 会被评估两次，导致格式混乱
    if (index == 0 && !ctx_omni->system_prompt_initialized) {
        print_with_timestamp("stream_prefill: n_past = %d\n voice_clone_prompt = %s\n assistant_prompt = %s\n", ctx_omni->n_past, voice_clone_prompt.c_str(), assistant_prompt.c_str());
        // tc-todo
        // llama_kv_cache_clear(ctx_omni->ctx_llama);
        
        // 初始化格式（对齐 Python 双工模型的 _init_duplex_session）：
        //   1. feed prefix_system_prompt（含 <|audio_start|>）
        //   2. feed ref_audio 的 APM embedding
        //   3. feed suffix_system_prompt（含 <|audio_end|><|im_end|>）
        //
        // 完整格式：
        //   <|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>
        //   [ref_audio APM embedding]
        //   <|audio_end|><|im_end|>
        
        if (ctx_omni->duplex_mode && aud_fname.length() > 0) {
            // 双工模式：参考音频需要送入 LLM
            
            // Step 1: 评估 prefix (voice_clone_prompt，包含 <|audio_start|>)
            eval_string(ctx_omni, ctx_omni->params, voice_clone_prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
            
            // Step 2: 获取并 prefill 参考音频的 APM embedding
            auto * audio_embeds = omni_audio_embed_make_with_filename(ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, aud_fname);
            if (audio_embeds != nullptr && audio_embeds->n_pos > 0) {
                prefill_with_emb(ctx_omni, ctx_omni->params, audio_embeds->embed, audio_embeds->n_pos, 
                                ctx_omni->params->n_batch, &ctx_omni->n_past);
                omni_embed_free(audio_embeds);
            } else {
            }
            
            // Step 3: 评估 suffix (assistant_prompt，包含 <|audio_end|><|im_end|>)
            eval_string(ctx_omni, ctx_omni->params, assistant_prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
        } else {
            const bool has_ref_audio_slot =
                voice_clone_prompt.find("<|audio_start|>") != std::string::npos &&
                assistant_prompt.find("<|audio_end|>") != std::string::npos;

            if (!has_ref_audio_slot) {
                // Python no-TTS chat template has a plain system message, with
                // no ref-audio embedding in the prompt.
                eval_string(ctx_omni, ctx_omni->params, voice_clone_prompt.c_str(),
                            ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                eval_string(ctx_omni, ctx_omni->params, assistant_prompt.c_str(),
                            ctx_omni->params->n_batch, &ctx_omni->n_past, false);
            } else {
                // 非双工 TTS/voice-clone 的 system prompt:
                // <|im_start|>system\nprefix\n<|audio_start|>[ref_audio]<|audio_end|>suffix<|im_end|>\n
                std::string system_ref_audio = !aud_fname.empty()
                    ? aud_fname
                    : (!ctx_omni->ref_audio_path.empty()
                        ? ctx_omni->ref_audio_path
                        : "tools/omni/assets/default_ref_audio/default_ref_audio.wav");
                print_with_timestamp("system prompt ref_audio: %s\n", system_ref_audio.c_str());

                eval_string(ctx_omni, ctx_omni->params, voice_clone_prompt.c_str(),
                            ctx_omni->params->n_batch, &ctx_omni->n_past, false);

                auto * ref_audio_embeds = omni_audio_embed_make_with_filename(
                    ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, system_ref_audio);
                if (ref_audio_embeds != nullptr && ref_audio_embeds->n_pos > 0) {
                    print_with_timestamp("system prompt ref_audio embedding: n_pos=%d\n", ref_audio_embeds->n_pos);
                    prefill_with_emb(ctx_omni, ctx_omni->params, ref_audio_embeds->embed, ref_audio_embeds->n_pos,
                                    ctx_omni->params->n_batch, &ctx_omni->n_past);
                    omni_embed_free(ref_audio_embeds);
                } else {
                    print_with_timestamp("WARNING: failed to load system prompt ref_audio: %s\n", system_ref_audio.c_str());
                }

                eval_string(ctx_omni, ctx_omni->params, assistant_prompt.c_str(),
                            ctx_omni->params->n_batch, &ctx_omni->n_past, false);
            }
        }
        
        // 标记系统 prompt 已初始化
        ctx_omni->system_prompt_initialized = true;

        //把这步完成再开llm线程以防冲突
        ctx_omni->n_keep = ctx_omni->n_past;
        print_with_timestamp("🔒 n_keep 设置为 %d (system prompt tokens)，这部分永远不会被滑动窗口删除\n", ctx_omni->n_keep);
        // 双工模式：assistant_prompt 不含 <|im_start|>user\n，需要 eval_prefix 补充
        // 非双工模式：assistant_prompt 末尾已含 <|im_start|>user\n，无需重复添加
        if (ctx_omni->duplex_mode) {
            eval_prefix(ctx_omni, ctx_omni->params);
        }
        
        // index=0 时 aud_fname 通常是 ref_audio（voice cloning），已在上面作为
        // system prompt 的一部分 prefill；用户音频应从 index>=1 传入，这里不再处理。
        print_with_timestamp("stream_prefill(index=0): system prompt 初始化完成，ref_audio 已在其中 prefill\n");

        // 注册 system prompt 保护长度（滑窗不删除这段）
        sliding_window_register_system_prompt(ctx_omni);

        print_with_timestamp("n_past = %d\n", ctx_omni->n_past);
        
        if (ctx_omni->async){
            print_with_timestamp("create llm & tts thread (duplex=%d)\n", (int)ctx_omni->duplex_mode);

            if (ctx_omni->duplex_mode) {
                // duplex 路径不启动单工 llm_thread_func，改启动 encoder + llm 两条常驻线程。
                duplex_start_threads(ctx_omni, ctx_omni->params);
            } else {
                // simplex 路径保持原状
                if (!ctx_omni->llm_thread.joinable()) {
                    llm_thread_running = true;
                    ctx_omni->llm_thread = std::thread(llm_thread_func, ctx_omni, ctx_omni->params);
                    print_with_timestamp("create llm thread success\n");
                }
            }

            if (ctx_omni->use_tts && !ctx_omni->tts_thread.joinable()) {
                tts_thread_running = true;
                // 按 duplex_mode 选择 TTS 线程函数
                if (ctx_omni->duplex_mode) {
                    ctx_omni->tts_thread = std::thread(tts_thread_func_duplex, ctx_omni, ctx_omni->params);
                    print_with_timestamp("create tts thread (duplex mode) success\n");
                } else {
                    ctx_omni->tts_thread = std::thread(tts_thread_func, ctx_omni, ctx_omni->params);
                    print_with_timestamp("create tts thread (simplex mode) success\n");
                }
            }

            // Start T2W thread if TTS is enabled and thread is not already running
            if (ctx_omni->use_tts && ctx_omni->t2w_thread_info && !ctx_omni->t2w_thread.joinable()) {
                t2w_thread_running = true;
                ctx_omni->t2w_thread = std::thread(t2w_thread_func, ctx_omni, ctx_omni->params);
                print_with_timestamp("create t2w thread success\n");
            }
        }

    }
    else {
        if (!ctx_omni->async) {
            if (img_fname.length() > 0 && ctx_omni->ctx_vision == nullptr) {
                LOG_WRN("%s: image provided but ctx_vision is NULL (media_type=%d), skipping: %s\n",
                        __func__, ctx_omni->media_type, img_fname.c_str());
            }
            if (img_fname.length() > 0 && ctx_omni->ctx_vision != nullptr) {
                if (max_slice_nums >= 1) {
                    vision_set_max_slice_nums(ctx_omni->ctx_vision, max_slice_nums);
                    LOG_INF("%s: [临时] max_slice_nums=%d for this prefill\n", __func__, max_slice_nums);
                }
                std::vector<std::vector<float>> vision_chunks;
                if (!omni_image_embed_make_chunks_with_filename(ctx_omni->ctx_vision, 
                        ctx_omni->params->cpuparams.n_threads, img_fname, vision_chunks)) {
                    LOG_ERR("%s: failed to create vision embeddings for %s\n", __func__, img_fname.c_str());
                    return false;
                }
                
                int n_chunks = (int)vision_chunks.size();
                int tokens_per_chunk = (int)vision_chunks[0].size() / hidden_size;
                bool has_slices = (n_chunks > 1);
                
                std::string prefix = "<unit>";
                eval_string(ctx_omni, ctx_omni->params, prefix.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                
                // Overview
                eval_string(ctx_omni, ctx_omni->params, "<image>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                prefill_with_emb(ctx_omni, ctx_omni->params, vision_chunks[0].data(), tokens_per_chunk, ctx_omni->params->n_batch, &ctx_omni->n_past);
                eval_string(ctx_omni, ctx_omni->params, "</image>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                
                // Slices (V2.6 schema)
                if (has_slices) {
                    for (int i = 1; i < n_chunks; i++) {
                        eval_string(ctx_omni, ctx_omni->params, "<slice>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                        prefill_with_emb(ctx_omni, ctx_omni->params, vision_chunks[i].data(), tokens_per_chunk, ctx_omni->params->n_batch, &ctx_omni->n_past);
                        eval_string(ctx_omni, ctx_omni->params, "</slice>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                    }
                    eval_string(ctx_omni, ctx_omni->params, "\n", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                }
                LOG_INF("%s: prefilled %d vision chunks (%d tokens each)\n", __func__, n_chunks, tokens_per_chunk);
            }
            if (aud_fname.length() > 0) {
                print_with_timestamp("stream_prefill(index=%d): processing user audio: %s\n", index, aud_fname.c_str());
                auto * embeds = omni_audio_embed_make_with_filename(ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, aud_fname);
                // 音频过短时 audition_audio_preprocess 会 pad 静音到 100ms；这里仍做安全检查，失败则跳过该帧音频
                if (embeds != nullptr && embeds->n_pos > 0) {
                    print_with_timestamp("stream_prefill(index=%d): user audio embedding: n_pos=%d\n", index, embeds->n_pos);
                    // 添加音频标记，与 index=0 保持一致
                    eval_string(ctx_omni, ctx_omni->params, "<|audio_start|>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                    prefill_with_emb(ctx_omni, ctx_omni->params, embeds->embed, embeds->n_pos, ctx_omni->params->n_batch, &ctx_omni->n_past);
                    eval_string(ctx_omni, ctx_omni->params, "<|audio_end|>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                    omni_embed_free(embeds);
                } else {
                    LOG_WRN("%s: audio encoding failed, skipping audio for this frame\n", __func__);
                }
            }
        }
        else {
            // async 模式：将 embeds 加入队列，由 LLM 线程处理
            
            const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
            omni_embeds * omni_embeds = new struct omni_embeds();
            //video
            if (img_fname.length() > 0) {
                if (ctx_omni->ctx_vision == nullptr) {
                    LOG_WRN("%s: image provided but ctx_vision is NULL (media_type=%d), skipping: %s\n",
                            __func__, ctx_omni->media_type, img_fname.c_str());
                } else {
                    LOG_INF("%s: img_fname:%s\n", __func__, img_fname.c_str());
                    if (max_slice_nums >= 1) {
                        vision_set_max_slice_nums(ctx_omni->ctx_vision, max_slice_nums);
                        LOG_INF("%s: [临时] max_slice_nums=%d for this prefill\n", __func__, max_slice_nums);
                    }
                    if (!omni_image_embed_make_chunks_with_filename(ctx_omni->ctx_vision, 
                            ctx_omni->params->cpuparams.n_threads, img_fname, omni_embeds->vision_embed)) {
                        LOG_ERR("%s: failed to create vision embeddings for %s\n", __func__, img_fname.c_str());
                        delete omni_embeds;
                        return false;
                    }
                    LOG_INF("%s: vision_embed has %d chunks\n", __func__, (int)omni_embeds->vision_embed.size());
                }
            }
            //audio
            // 只有在音频路径非空时才处理音频
            if (aud_fname.length() > 0) {
                LOG_INF("%s: aud_fname:%s\n", __func__, aud_fname.c_str());
                auto * audio_embeds = omni_audio_embed_make_with_filename(ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, aud_fname);
                // 音频过短时 audition_audio_preprocess 会 pad 静音到 100ms；这里仍做安全检查，失败则跳过（audio_embed 保持为空）
                if (audio_embeds != nullptr && audio_embeds->n_pos > 0) {
                    //save to buffer
                    LOG_INF("%s: audio_embeds->n_pos: %d ,hidden_size: %d\n", __func__, audio_embeds->n_pos, hidden_size);
                    omni_embeds->audio_embed.resize(audio_embeds->n_pos * hidden_size);
                    std::memcpy(omni_embeds->audio_embed.data(), audio_embeds->embed, omni_embeds->audio_embed.size() * sizeof(float));
                    omni_embed_free(audio_embeds);
                } else {
                    LOG_WRN("%s: audio encoding failed, skipping audio for this frame: %s\n", __func__, aud_fname.c_str());
                }
            }
            omni_embeds->index = index;
            // text 仅作为字面量带过去，由 LLM 线程在锁保护下做 eval_string。
            if (!text.empty()) {
                omni_embeds->user_text = text;
            }
            // <|im_start|>user\n 已在 sys prompt 末尾添加，后续轮次在 stream_decode 结束时添加，这里不再设置
            
            std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
            ctx_omni->llm_thread_info->cv.wait(lock, [&] { return ctx_omni->llm_thread_info->queue.size() < ctx_omni->llm_thread_info->MAX_QUEUE_SIZE; });
            ctx_omni->llm_thread_info->queue.push(omni_embeds);

            //notify the llm
            lock.unlock();
            ctx_omni->llm_thread_info->cv.notify_all();
        }
    }
    // 打印 stream_prefill 结束时的状态
    print_with_timestamp("\n\nc++ finish stream_prefill(index=%d). n_past=%d, n_keep=%d, n_ctx=%d\n\n",
                         index, ctx_omni->n_past, ctx_omni->n_keep, ctx_omni->params->n_ctx);
    return true;
}

bool stream_decode(struct omni_context * ctx_omni, std::string debug_dir, int round_idx,
                   double * out_t_first_token_ms) {
    // 路由：duplex_mode && async && system prompt 已初始化时，走 duplex 路径。
    // duplex_decode 会把请求投递到 duplex_llm_thread 并同步等待完成，
    // 返回前 text_queue / tts_thread_info->queue 都已就绪，对调用方（HTTP SSE /
    // test-duplex.cpp）行为与原 stream_decode 一致。
    // 下方主体（本次 if 之后的部分）是非 duplex 的旧同步/异步实现。
    if (ctx_omni->duplex_mode && ctx_omni->async
        && ctx_omni->system_prompt_initialized && ctx_omni->duplex != nullptr) {
        return duplex_decode(ctx_omni, debug_dir, round_idx, out_t_first_token_ms);
    }

    // NOTE: 不再自动归档旧输出目录，因为这会导致同一 session 中每轮对话的输出被移走
    // 如果需要归档，可以在新 session 开始时（omni_init）手动调用
    // move_old_output_to_archive();
    
    // 轮次同步：调用方指定了 round_idx 就立即同步 simplex_round_idx，
    // 解决 TTS 线程异步递增 round_idx 导致的竞态（Python 端递增 current_round_number
    // 时 C++ TTS 线程可能还没处理完上一轮）。
    //
    // 注意：新 session 的 KV cache 清理在 update_session_config 中处理，这里只做同 session 内的轮次同步。
    if (round_idx >= 0 && !ctx_omni->duplex_mode) {
        if (ctx_omni->simplex_round_idx != round_idx) {
            print_with_timestamp("📍 [轮次同步] 调用方指定 round_idx=%d，当前 simplex_round_idx=%d，强制同步\n",
                                round_idx, ctx_omni->simplex_round_idx);
            ctx_omni->simplex_round_idx = round_idx;
            // 同时更新 wav_turn_base 以保持一致性
            ctx_omni->wav_turn_base = round_idx * 1000;
        }
    }
    
    // 不再清空 llm_debug/chunk_* 目录：该清空与 TTS 线程写入存在竞态。
    // 场景：第一轮 stream_decode 完成但 TTS 还在处理 chunk_0-9 时，第二轮开始
    // 清空目录会删掉 TTS 已写入的 chunk。现在每个 round 有独立目录（round_000…），无需清空。
    
    // Record start time (t=0) for WAV file naming
    ctx_omni->stream_decode_start_time = std::chrono::high_resolution_clock::now();
    
    // 打印 stream_decode 开始时的关键状态
    print_with_timestamp("📍 stream_decode 开始: n_past=%d, n_keep=%d, n_ctx=%d, duplex_mode=%d\n",
                         ctx_omni->n_past, ctx_omni->n_keep, ctx_omni->params->n_ctx, ctx_omni->duplex_mode);

    // decode_start_cache_len 不在这里取值：此刻 n_past 还是 prefill 之前的值。
    // stream_decode 内部是 prefill 写入 KV → decode 采样 的顺序，prefill 那段的 KV 已由
    // sliding_window_register_unit_start/_end 记成 omni/audio unit；若在此锁定 n_past，
    // 结束时 end-start 会把 prefill 段重复记进 response unit（一段 KV 被两条 unit 记账，
    // 表现为 "CONSISTENCY ERROR! sum(units) >> cache" 以及滑窗多删 KV）。
    // 所以 decode_start_cache_len 必须等下面的 "wait prefill done" 之后再取。
    int decode_start_cache_len = -1;

    // 重置 ended_with_listen：每次 decode 开始假设以非 listen 结束（需清理 KV cache），
    // 若 LLM 检测到 <|listen|> 再置 true。
    // 重置 llm_generation_done（TTS 线程据此决定是否添加 text_eos_embed）。
    if (!ctx_omni->duplex_mode) ctx_omni->llm_generation_done.store(false);
    ctx_omni->ended_with_listen = false;

    // decode 开始时重置 break_event：否则若 T2W 还在等待数据没来得及重置，
    // 新一轮 decode 会检测到 break_event=true 立即退出、不生成任何 token。
    if (ctx_omni->duplex_mode && ctx_omni->break_event.load()) {
        ctx_omni->break_event.store(false);
        print_with_timestamp("📍 stream_decode: reset break_event from true to false\n");
    }
    
    // 清空上一轮的文本队列，重置状态标志
    {
        std::lock_guard<std::mutex> lock(ctx_omni->text_mtx);
        ctx_omni->text_queue.clear();
        ctx_omni->text_done_flag = false;
        ctx_omni->text_streaming = true;
    }
    
    if (ctx_omni->async){
        // 确保线程已启动（若 prefill 是同步模式执行的，线程可能尚未启动）
        if (!ctx_omni->tts_thread.joinable() && ctx_omni->use_tts) {
            tts_thread_running = true;
            if (ctx_omni->duplex_mode) {
                ctx_omni->tts_thread = std::thread(tts_thread_func_duplex, ctx_omni, ctx_omni->params);
                print_with_timestamp("stream_decode: create tts thread (duplex mode)\n");
            } else {
                ctx_omni->tts_thread = std::thread(tts_thread_func, ctx_omni, ctx_omni->params);
                print_with_timestamp("stream_decode: create tts thread (simplex mode)\n");
            }
        }
        if (!ctx_omni->t2w_thread.joinable() && ctx_omni->use_tts && ctx_omni->t2w_thread_info) {
            t2w_thread_running = true;
            ctx_omni->t2w_thread = std::thread(t2w_thread_func, ctx_omni, ctx_omni->params);
            print_with_timestamp("stream_decode: create t2w thread\n");
        }
        
        ctx_omni->need_speek = true;
        //ctx_omni->llm_thread.join();
        ctx_omni->llm_thread_info->cv.notify_all();
        print_with_timestamp("wait prefill done\n");
        std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
        g_decode_cv.wait(lock, []{ return prefill_done; });
        prefill_done = false;
    }

    // 现在 prefill 已写完 KV、unit_history 里已有对应的 omni/audio unit；
    // 此刻的 n_past 才是 decode 真正要写入的起始位置。后面 register response unit 用
    // (n_past_now - decode_start_cache_len)，正好等于 decode 自身写入的长度
    // （含单工模式下面 eval 的 <|im_end|>...<|tts_bos|> assistant prompt，
    //  那段也算进 response unit 一起删，不会重复记账——它没经过 prefill 的
    //  register_unit_start/_end）。
    decode_start_cache_len = ctx_omni->n_past;

    // 只有启用 TTS 时才设置 speek_done 为 false
    if (ctx_omni->use_tts) {
        ctx_omni->speek_done = false;
    }
    
    // 按模式设置不同的 assistant generation prompt（对齐 Python MiniCPM-o-4_5-latest）：
    //
    // === 非双工模式 (Simplex) ===
    // Python default_tts_chat_template:
    //   {% if add_generation_prompt %}{{ '<|im_start|>assistant\n' + think_str + '<|tts_bos|>' }}{% endif %}
    // think_str = "<think>\n\n</think>\n\n"
    //
    // stream_prefill 已添加 <|audio_start|>[audio]<|audio_end|>，这里只需
    // 关闭用户消息并加 assistant generation prompt：
    //   <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>
    //
    // === 双工模式 (Duplex) ===
    // 双工模式用 <unit> 标记，不用标准 chat template；stream_prefill 添加
    // <unit>[audio_embed]（无 audio_start/end），模型自动输出 <|speak|> / <|listen|> 控制对话。

    if (ctx_omni->duplex_mode) {
        // 双工模式不加 assistant prompt：模型根据上下文自动决定说话/继续监听，
        // 输出 <|speak|>xxx<|chunk_eos|> 或 <|listen|><|chunk_eos|>。
        print_with_timestamp("stream_decode: 双工模式，跳过 assistant prompt\n");

        // [Case 2 抢答] 会话开局前 N 次 stream_decode 强制 LISTEN，
        // 与 Python duplex_config.force_listen_count=3 对齐。
        // 防止 browser MediaStreamTrack 开启时的瞬态噪声 + 强 system prompt
        // 组合导致模型第一个 chunk 直接 SPEAK。
        // 这里直接发 __IS_LISTEN__ 到 text_queue 走 LISTEN 分支，不启动 LLM 采样。
        // 用户音频的 KV cache 已经在 stream_prefill 阶段写入，所以不会丢失上下文。
        if (ctx_omni->force_listen_used < ctx_omni->force_listen_count) {
            ctx_omni->force_listen_used++;
            ctx_omni->slide_last_was_listen = true;
            ctx_omni->ended_with_listen = true;
            ctx_omni->current_turn_ended = false;
            if (ctx_omni->use_tts) {
                ctx_omni->speek_done = true;
            }
            print_with_timestamp("LLM Duplex: force_listen %d/%d, skip generation and emit __IS_LISTEN__\n",
                                 ctx_omni->force_listen_used, ctx_omni->force_listen_count);
            if (ctx_omni->async) {
                std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
                ctx_omni->text_queue.push_back("__IS_LISTEN__");
                ctx_omni->text_done_flag = true;
                ctx_omni->text_streaming = false;
                ctx_omni->text_cv.notify_all();
            }
            return true;
        }
    } else if (ctx_omni->use_tts) {
        // 非双工 TTS 模式：需包含 <|tts_bos|> 让模型开始生成 TTS 文本。
        // stream_prefill 已添加 <|audio_start|>[audio]<|audio_end|>，这里关闭用户消息并加 assistant prompt。
        std::string prompt = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>";
        print_with_timestamp("📍 [单工TTS] 添加 assistant prompt: \"%s\", n_past=%d\n", 
                            prompt.c_str(), ctx_omni->n_past);
        {
            eval_string(ctx_omni, ctx_omni->params, prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
        }
        print_with_timestamp("📍 [单工TTS] assistant prompt 完成, n_past=%d\n", ctx_omni->n_past);
    } else {
        // 非双工纯 LLM 模式：用空 thinking block，避免模型在面向用户的回答里直接生成 <think>...</think>。
        std::string prompt = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        {
            eval_string(ctx_omni, ctx_omni->params, prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
        }
    }
    LOG_INF("<user>%s\n", ctx_omni->params->prompt.c_str());
    LOG_INF("<assistant>");
    const int max_tgt_len = ctx_omni->params->n_predict < 0 ? ctx_omni->params->n_ctx : ctx_omni->params->n_predict;
    print_with_timestamp("LLM decode: max_tgt_len = %d, n_predict = %d, n_ctx = %d\n", 
                         max_tgt_len, ctx_omni->params->n_predict, ctx_omni->params->n_ctx);
    // LLM chunk size: 每chunk推送给TTS的LLM tokens数量
    // 原始Python: generate_chunk_size=10
    // 注意：step_size影响TTS条件长度，可能影响音质
    // step_size=5: 首响更快(612ms)但可能影响音质
    // step_size=10: 首响稍慢(791ms)但音质更稳定
    int step_size = 10;  // 恢复原始值
    std::string response = "";
    
    // tts streaming memory
    std::string tts_txt = "";
    int chunk_idx = 0;
    std::vector<llama_token> audio_input_ids;
    // TODO write to specific buffers
    std::vector<float> tts_output;
    tts_output.resize(1/* batch_size */ * (ctx_omni->params->n_ctx /* seq_len */ * 2) * 256);
    bool llm_finish = false;
    bool llm_first_token_logged = false;
    
    // 记录当前 chunk 是否为 turn 的结束（随 LLMOut 传给 TTS 线程，避免全局状态的时序问题）
    bool local_is_end_of_turn = false;
    // 双工模式当前 chunk 已生成的 token 数（用于 chunk 上限、及时响应打断）
    int current_chunk_tokens = 0;
    for (int il = 0; il < max_tgt_len; ) {
        // 外层循环也检测 break_event
        if (ctx_omni->break_event.load()) {
            llm_finish = true;
            break;
        }
        fflush(stdout);
        response = "";
        fflush(stdout);
        
        // 注意: speek_done=true 现在只表示"TTS 完成，可接受新 prefill"
        // 不再用于控制 LLM 退出。LLM 应该正常完成直到 EOS 或达到最大长度。
        // 打断逻辑通过 need_speek 或其他机制处理。
        fflush(stdout);
        
        int jl = 0;  // 计数有效的 TTS token 数量
        int total_tokens_generated = 0;  // 计数总共生成的 token 数量（包括被过滤的）
        // 收集当前 chunk 的 token IDs 和 hidden states 用于 TTS 条件生成
        // 只收集有效的 TTS token，确保每次给 TTS 的都是 step_size 个有效 token
        std::vector<llama_token> chunk_token_ids;
        std::vector<float> chunk_hidden_states;
        int llm_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));

        // 每个 chunk 开始时重置 is_end_of_turn，仅检测到 TURN_EOS/TTS_EOS/EOS 时置 true
        local_is_end_of_turn = false;

        // chunk 限制只在双工模式生效：每个 chunk 最多 max_new_speak_tokens_per_chunk 个 token
        // 便于及时响应打断；单工无限制，LLM 生成直到 EOS。
        int max_chunk_tokens = ctx_omni->duplex_mode ? ctx_omni->max_new_speak_tokens_per_chunk : 0;
        bool chunk_limit_reached = (max_chunk_tokens > 0 && current_chunk_tokens >= max_chunk_tokens);
        {
            fflush(stdout);
            // 循环直到收集满 step_size 个有效 token（非生成 step_size 个 token）；
            // 同时检测 break_event 与 chunk 上限，支持打断。
            while (jl < step_size && !llm_finish && !ctx_omni->break_event.load() && !chunk_limit_reached) {
                // streaming llm
                const char * tmp = nullptr;
                float * hidden_states = nullptr;

                llama_token sampled_token = 0;
                {
                    std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
                    // 使用新函数获取token文本、hidden state和token ID
                    tmp = llama_loop_with_hidden_and_token(ctx_omni, ctx_omni->params, ctx_omni->ctx_sampler, ctx_omni->n_past, hidden_states, sampled_token);
                }
                
                total_tokens_generated++;

                // 只收集有效的 TTS token；特殊 token（如 <think>、</think>、换行）不计入 step_size
                if (tmp != nullptr && hidden_states != nullptr) {
                    if (is_valid_tts_token(sampled_token)) {
                        chunk_token_ids.push_back(sampled_token);
                        chunk_hidden_states.insert(chunk_hidden_states.end(), hidden_states, hidden_states + llm_n_embd);
                        jl++;  // 只有有效 token 才增加计数

                        // 更新当前 chunk 的 token 计数，达到上限即截断
                        current_chunk_tokens++;

                        if (max_chunk_tokens > 0 && current_chunk_tokens >= max_chunk_tokens) {
                            chunk_limit_reached = true;
                        }
                    }
                }
                
                // if (hidden_states != nullptr) {
                //     int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
                //     // 打印第一个 embedding 的前5个数字
                //     printf("First embedding (first 5): ");
                //     for (int i = 0; i < 5 && i < n_embd; i++) {
                //         printf("%.6f ", hidden_states[i]);
                //     }
                //     printf("\n");
                //     // 打印最后一个 embedding 的后5个数字 (这里只有1个token，所以第一个和最后一个是同一个)
                //     printf("Last embedding (last 5): ");
                //     for (int i = n_embd - 5; i < n_embd; i++) {
                //         if (i >= 0) {
                //             printf("%.6f ", hidden_states[i]);
                //         }
                //     }
                //     printf("\n");
                //     free(hidden_states);
                // }
                if (!llm_first_token_logged) {
                    llm_first_token_logged = true;
                }
                if (tmp == nullptr) {
                    LOG_ERR("llama_loop returned nullptr!");
                    break;
                }
                
                // 用缓存的 token ID 做类型检测，比字符串比较更高效
                OmniTokenType token_type = get_token_type(ctx_omni, sampled_token);
                if (token_type != OmniTokenType::NORMAL) {
                }

                if (ctx_omni->duplex_mode) {
                    // turn_eos 处理（对齐 Python）：turn_eos 不触发 LLM 跳出，只标记 is_end_of_turn；
                    // LLM 继续生成直到 chunk_eos/listen 经 is_end_token() 正常跳出。
                    // turn_eos 本身作为 special token 被过滤（不进文本 response）；
                    // is_end_of_turn 传给 TTS 线程，告知这是最后一个 chunk。
                    if (token_type == OmniTokenType::TURN_EOS ||
                        token_type == OmniTokenType::TTS_EOS ||
                        token_type == OmniTokenType::EOS) {
                        local_is_end_of_turn = true;
                        ctx_omni->current_turn_ended = true;
                        print_with_timestamp("LLM Duplex: turn_eos detected (type=%d), "
                                            "set is_end_of_turn=true (not breaking, wait for chunk_eos)\n",
                                            (int)token_type);
                        // 不 break，不设 llm_finish，继续生成直到 chunk_eos/listen
                    } else if (token_type == OmniTokenType::LISTEN) {
                        // LISTEN 表示切回听状态（修复尾音串接）：
                        // - 之前在 SPEAK（slide_last_was_listen=false）：本轮发言刚结束，
                        //   TTS buffer 可能有不足 chunk_size(25) 的残留 token，必须 flush，
                        //   否则会被带进下一轮，尾音串到下一轮开头。
                        // - 之前在 listen（slide_last_was_listen=true，如会话开局用户未说话、
                        //   或模型连续生成 listen）：没有 speak 内容要 flush，不触发 is_end_of_turn
                        //   （否则 TTS 会误加 text_eos_embed 生成杂音）。
                        // slide_last_was_listen 由 LISTEN 分支末尾更新，此处读到的是
                        // "上一个 chunk 结束时的状态"，正好用于判断。
                        if (!ctx_omni->slide_last_was_listen.load()) {
                            local_is_end_of_turn = true;
                            print_with_timestamp("LLM Duplex: LISTEN detected after SPEAK, set is_end_of_turn=true to flush TTS buffer\n");
                        } else {
                            print_with_timestamp("LLM Duplex: LISTEN during listen state, skip flush\n");
                        }
                    }
                }
                
                if (is_end_token(ctx_omni, sampled_token)){
                    llm_finish = true;
                    
                    // 设置 llm_generation_done（TTS 线程据此决定是否添加 text_eos_embed）
                    if (!ctx_omni->duplex_mode) ctx_omni->llm_generation_done.store(true);
                    print_with_timestamp("LLM: detected end token, set llm_generation_done=true\n");

                    // 设置 current_turn_ended：只有 TURN_EOS / TTS_EOS 才算轮次真正结束；
                    // CHUNK_EOS/CHUNK_TTS_EOS 只是 chunk 结束，LISTEN 只是暂时切到听状态，
                    // 用户说完后模型还要继续回复——这样 TTS KV cache 才能在多轮
                    // speak-listen-speak 中保持连续。
                    if (token_type == OmniTokenType::TURN_EOS ||
                        token_type == OmniTokenType::TTS_EOS ||
                        token_type == OmniTokenType::EOS) {
                        ctx_omni->current_turn_ended = true;
                    } else if (token_type == OmniTokenType::LISTEN) {
                        // LISTEN: 不设置 current_turn_ended，保持 TTS 状态连续
                    }

                    // <|listen|> 特殊处理：双工下表示模型主动切到听状态，
                    // 经 text_queue 通知 SSE 客户端，并置 ended_with_listen，
                    // 让 stream_decode 末尾不清理 KV cache。
                    if (token_type == OmniTokenType::LISTEN && ctx_omni->duplex_mode) {
                        ctx_omni->ended_with_listen = true;
                        ctx_omni->slide_last_was_listen = true;
                        
                        if (ctx_omni->async) {
                            std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
                            ctx_omni->text_queue.push_back("__IS_LISTEN__");
                            ctx_omni->text_cv.notify_all();
                        }
                    } else if (ctx_omni->duplex_mode) {
                        ctx_omni->slide_last_was_listen = false;
                    }
                    
                    // Don't add end tokens to response
                    break;
                }

                // Copy tmp to a local string immediately to avoid issues with static string
                std::string tmp_str(tmp);
                response += tmp_str;
                fflush(stdout);
            }
            fflush(stdout);
            fflush(stdout);
        }
        fflush(stdout);
        
        // 达到 chunk 限制即结束当前 decode（与 Python 双工 server 一致：每次 generate
        // 只返回一个 chunk，客户端再次调用 stream_decode 拿下一个）。
        if (chunk_limit_reached) {

            // 达到 chunk 限制时强制补 <|chunk_eos|> token（对齐 Python
            // self.decoder.feed(self.decoder.embed_token(self.chunk_eos_token_id))）
            if (ctx_omni->special_token_chunk_eos >= 0) {
                std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
                // Feed chunk_eos token to model (update KV cache)
                std::vector<llama_token> chunk_eos_tokens = {ctx_omni->special_token_chunk_eos};
                eval_tokens(ctx_omni, ctx_omni->params, chunk_eos_tokens, 
                           ctx_omni->params->n_batch, &ctx_omni->n_past);
            }
            // 这样 SSE 流会结束，客户端可以再次调用 decode
            llm_finish = true;
            // 注意：不重置 current_chunk_tokens，下次 decode 会从 0 开始
            current_chunk_tokens = 0;
        }
        
        // add </unit> token after each chunk
        if (ctx_omni->duplex_mode && ctx_omni->special_token_unit_end >= 0) {
            std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
            // Feed </unit> token to model (update KV cache)
            std::vector<llama_token> unit_end_tokens = {ctx_omni->special_token_unit_end};
            eval_tokens(ctx_omni, ctx_omni->params, unit_end_tokens, 
                       ctx_omni->params->n_batch, &ctx_omni->n_past);
        }
        fflush(stdout);
        if (!response.empty()) {
            fflush(stdout);
        } else {
            fflush(stdout);
        }
        fflush(stdout);
        if (il == 0){
        }
        // 用总生成 token 数推进 il（与 max_tgt_len 比较）
        il += total_tokens_generated;

        // 移除响应中的所有特殊结束 token。
        // 注意：<|speak|> 是开始说话的标记而非结束 token，移除但不截断后面内容。
        {
            static const std::vector<std::string> end_token_strings = {
                "<|tts_eos|>",
                "</s>",
                "<|listen|>",
                "<|turn_eos|>",
                "<|chunk_eos|>",
                "<|chunk_tts_eos|>"
                // 注意：移除了 <|speak|>，它不是结束 token
            };
            
            // 对于结束 token，截断其后的内容
            for (const auto& delimiter : end_token_strings) {
                size_t end = response.find(delimiter);
                if (end != std::string::npos) {
                    response = response.substr(0, end);
                }
            }
            
            // 开始/控制标记直接移除（不截断后面内容）
            static const std::vector<std::string> inline_token_strings = {
                "<|speak|>",
                "<think>",
                "</think>",
            };
            for (const auto & token : inline_token_strings) {
                size_t pos = response.find(token);
                while (pos != std::string::npos) {
                    response.erase(pos, token.length());
                    pos = response.find(token);
                }
            }
        }
        fflush(stdout);
        // push text fragment to text_queue (both sync and async modes)
        if (!response.empty()) {
            std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
            ctx_omni->text_queue.push_back(response);
            ctx_omni->text_cv.notify_all();
        }
        if (ctx_omni->async){
            fflush(stdout);
            
            if (ctx_omni->use_tts && ctx_omni->tts_thread_info && (!response.empty() || llm_finish)) {
                // LLM chunk timing
                auto llm_chunk_time = std::chrono::high_resolution_clock::now();
                auto llm_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    llm_chunk_time - ctx_omni->stream_decode_start_time).count();
                fflush(stdout);
                LLMOut * llm_out = new LLMOut();
                llm_out->text = response;
                llm_out->n_past = ctx_omni->n_past;
                llm_out->llm_finish = llm_finish;
                llm_out->debug_dir = debug_dir;
                // 填充token IDs和hidden states用于TTS条件生成
                llm_out->token_ids = chunk_token_ids;
                llm_out->hidden_states = chunk_hidden_states;
                llm_out->n_embd = llm_n_embd;
                // 传递 is_end_of_turn（随数据一起传递，确保 TTS 处理的是当前 chunk 对应的状态）
                llm_out->is_end_of_turn = local_is_end_of_turn;

                // 打印 LLM 推送给 TTS 的数据
                {
                    std::string token_ids_str = "";
                    for (size_t i = 0; i < chunk_token_ids.size() && i < 20; i++) {
                        token_ids_str += std::to_string(chunk_token_ids[i]);
                        if (i < chunk_token_ids.size() - 1 && i < 19) token_ids_str += " ";
                    }
                    if (chunk_token_ids.size() > 20) token_ids_str += "...";
                    
                    print_with_timestamp("LLM->TTS: text='%s', n_tokens=%zu, hidden_size=%zu, n_embd=%d, token_ids=[%s]\n",
                                        response.c_str(),
                                        chunk_token_ids.size(),
                                        chunk_hidden_states.size(),
                                        llm_n_embd,
                                        token_ids_str.c_str());
                }
                
                fflush(stdout);
                fflush(stdout);
                std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
                fflush(stdout);
                ctx_omni->tts_thread_info->cv.wait(lock, [&] { return ctx_omni->tts_thread_info->queue.size() < ctx_omni->tts_thread_info->MAX_QUEUE_SIZE; });
                fflush(stdout);
                fflush(stdout);
                // 双工模式下 LLM 始终推送数据到 TTS 队列：每个 chunk 需独立处理，
                // 不能因上一 chunk 完成（speek_done=true）就丢弃新数据；
                // 仅非双工模式才用 speek_done 避免重复处理。
                if (!ctx_omni->speek_done || ctx_omni->duplex_mode){
                    llm_out->enqueue_time = std::chrono::steady_clock::now();
                    fflush(stdout);
                    ctx_omni->tts_thread_info->queue.push(llm_out);
                    fflush(stdout);
                    //notify the tts thread
                    ctx_omni->tts_thread_info->cv.notify_all();
                    fflush(stdout);
                } else {
                    // speek_done is true, delete llm_out to prevent memory leak
                    fflush(stdout);
                    delete llm_out;
                    llm_out = nullptr;
                    fflush(stdout);
                }
                fflush(stdout);
            }
            fflush(stdout);
        }else{
            fflush(stdout);
        }
        fflush(stdout);
        if (llm_finish) break;
    }
    fflush(stdout);
    // 推送轮次结束标记
    // mark text done
    {
        std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
        // 推送 end_of_turn 标记，让客户端知道当前轮次结束
        if (!ctx_omni->duplex_mode || !ctx_omni->ended_with_listen) {
            ctx_omni->text_queue.push_back("__END_OF_TURN__");
        }

        ctx_omni->text_done_flag = true;
        ctx_omni->text_cv.notify_all();
        ctx_omni->text_streaming = false;
    }
    // Safety checks before cleanup
    if (ctx_omni == nullptr) {
        LOG_ERR("stream_decode: ctx_omni is nullptr in cleanup!");
        return false;
    }
    if (ctx_omni->ctx_llama == nullptr) {
        LOG_ERR("stream_decode: ctx_omni->ctx_llama is nullptr in cleanup!");
        return false;
    }
    if (ctx_omni->params == nullptr) {
        LOG_ERR("stream_decode: ctx_omni->params is nullptr in cleanup!");
        return false;
    }
    
    // ==================== 轮次边界记录与滑窗检查 ====================
    // 单工多轮对话，在 decode 结束时：
    // 1. 先检查并执行滑窗（基于之前的轮次边界）
    // 2. 再记录当前轮次的结束边界（作为下一轮的开始位置）
    if (!ctx_omni->duplex_mode) {
        // 检查是否需要滑窗，确保下一轮有足够空间（n_past > n_ctx - reserved_space 时触发）
        const int reserved_space = 1024;  // 预留空间
        const int n_ctx = ctx_omni->params->n_ctx;
        
        if (ctx_omni->n_past > n_ctx - reserved_space) {
            print_with_timestamp("⚠️ Decode 结束滑窗检查: n_past=%d > n_ctx-reserved=%d，需要滑窗\n",
                                 ctx_omni->n_past, n_ctx - reserved_space);
            
            // 调用滑窗函数，传入 reserved_space 作为需要腾出的空间
            kv_cache_slide_window(ctx_omni, ctx_omni->params, reserved_space);
        } else {
            print_with_timestamp("📍 Decode 结束: n_past=%d, 剩余空间=%d, 无需滑窗\n",
                                 ctx_omni->n_past, n_ctx - ctx_omni->n_past);
        }
        
        // 记录当前轮次的结束位置（即下一轮的开始位置）。一个完整轮次 =
        // 用户提问（可能多个 audio prefill）+ 模型回答，这样按轮次删除时可整删"问答对"。
        ctx_omni->round_start_positions.push_back(ctx_omni->n_past);
        print_with_timestamp("📍 轮次 %zu 结束，记录边界于 n_past=%d\n",
                             ctx_omni->round_start_positions.size(), ctx_omni->n_past);
        // 本 turn 已完结，之后注册的 UnitEntry 归入下一个 turn
        ctx_omni->current_turn_id++;

        // 为下一轮准备 <|im_end|>\n<|im_start|>user\n：
        // 第一轮的 <|im_start|>user\n 在 sys prompt 末尾，后续轮次需在 decode 结束时
        // 添加，结束当前 assistant 回复并开始新一轮 user 输入。
        eval_string(ctx_omni, ctx_omni->params, "<|im_end|>\n<|im_start|>user\n", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
        print_with_timestamp("📍 为下一轮准备: eval <|im_end|>\\n<|im_start|>user\\n, n_past=%d\n", ctx_omni->n_past);
    }
    
    // ==================== response unit 注册 ====================
    // 把这一轮 decode 期间通过采样写入 KV 的所有 token 打包成一个
    // type = "response" 的 UnitEntry，挂到 unit_history 里。
    //
    // 为什么必须做这件事：
    //   1. unit-based 滑窗（sliding_window_enforce + sliding_window_drop_next_unit）
    //      只会按 unit_history 里的条目去 drop。如果 response 段从来没被注册，
    //      它就永远停留在 KV 里，谁也删不掉，长会话下迟早把上下文撑爆。
    //   2. 一致性检查 expected = preserve + sum(units.length) 也要求每一段
    //      占用 KV 的 token 都有对应记账，否则会反复打印
    //      "❌ CONSISTENCY ERROR! preserve=X + sum(units)=Y != cache=Z"。
    //
    // 注意点：
    //   - type 必须不是 "system"，这样 sliding_window_drop_next_unit 才愿意删它
    //     （那里是 `if (entry.type == "system") continue;`）。
    //   - 真正释放 KV 的链路是
    //       sliding_window_drop_next_unit
    //         -> sliding_window_drop_unit
    //         -> sliding_window_drop_tokens_from_cache
    //     而 sliding_window_drop_tokens_from_cache 已经在前面修过：
    //     llama_memory_seq_rm 之后会跟一次 llama_memory_seq_add(..., -length)，
    //     把后段 position 向前平移，所以删 response unit 也是安全的、
    //     不会再触发 "inconsistent sequence positions"。
    //   - length 用 (n_past - decode_start_cache_len)，这是 decode 期间
    //     KV 实际增长的字节数，和 prefill 路径上 register_unit_end 的算法
    //     （current_cache_len - pending_unit_start_cache_len）保持一致。
    //   - 必须只在 sliding_window_config.mode != "off" 时执行，跟 prefill 路径上
    //     的 sliding_window_register_unit_start/_end 保持一致。
    //     原因：mode == "off" 时谁都不会消费 unit_history，如果在这里继续 push，
    //     会得到一个只增不减的 vector（轻度内存泄漏），而且一旦中途切回 basic，
    //     里面残留的 response 条目会让一致性检查直接对不上。
    if (ctx_omni->sliding_window_config.mode != "off") {
        int decode_end_cache_len = ctx_omni->n_past;

        // decode_start_cache_len 默认值是 -1，正常路径上会在 "wait prefill done"
        // 之后被赋成"prefill 完成时的 n_past"。如果中途异常退出导致没赋值，
        // 就跳过本次记账，避免把整段 cache 当 response 误记。
        if (decode_start_cache_len < 0) {
            print_with_timestamp(
                "[SW] register response unit: skipped, decode_start_cache_len 未初始化"
                " (可能 prefill 没正常完成)，跳过记账\n");
        } else {
            int response_len = decode_end_cache_len - decode_start_cache_len;
            if (response_len > 0) {
                UnitEntry entry;
                entry.unit_id = ctx_omni->next_unit_id++;
                entry.length  = response_len;
                entry.type    = "response";
                entry.is_listen = ctx_omni->ended_with_listen.load();
                // response 段和同轮的 prefill unit 共享 turn_id，
                // 这样按 turn 丢时会把“用户输入 + 模型回复”作为一个整体一次性丢掉。
                entry.turn_id = ctx_omni->current_turn_id;
                // generated_tokens 暂不在这里收集：decode 的 token 流已经通过
                // text_queue / TTS 通道处理掉了，这里只关心 KV 长度的记账。
                ctx_omni->unit_history.push_back(entry);

                print_with_timestamp(
                    "[SW] register response unit: unit_id=%d type=response len=%d "
                    "is_listen=%d turn_id=%d | cache=%d preserve=%d total_units=%zu\n",
                    entry.unit_id, entry.length, (int)entry.is_listen, entry.turn_id,
                    decode_end_cache_len, ctx_omni->system_preserve_length,
                    ctx_omni->unit_history.size());
            } else {
                print_with_timestamp(
                    "[SW] register response unit: skipped, response_len=%d "
                    "(start=%d, end=%d) — decode 没有写入新 token，无需记账\n",
                    response_len, decode_start_cache_len, decode_end_cache_len);
            }
        }
    }

    // 双工模式：只在 LISTEN 结束时记录 round 边界（一个完整的"说+听"轮次结束）
    // 这样 round 覆盖完整的 [unit+SPEAK...unit+LISTEN] 序列，
    // 滑窗按 round 截断时不会把连续 SPEAK 或 unit 从中间切开
    if (ctx_omni->duplex_mode) {
        if (ctx_omni->ended_with_listen) {
            ctx_omni->round_start_positions.push_back(ctx_omni->n_past);
            print_with_timestamp("📍 round boundary at n_past=%d, total rounds=%zu\n",
                                 ctx_omni->n_past, ctx_omni->round_start_positions.size());
            // 双工下以 ended_with_listen 作为一个完整 turn 的结束
            //（一个 turn 覆盖 [unit+SPEAK...unit+LISTEN] 的完整闭环）
            ctx_omni->current_turn_id++;
        }
    } else {
        // clean_kvcache(ctx_omni);
        // eval_prefix(ctx_omni, ctx_omni->params);
    }
    
    return true;
}

// ============================================================================
// ===== DUPLEX SESSION (high-level) ==========================================
// ----------------------------------------------------------------------------
// 把 stream_prefill / stream_decode 的"两步式 + producer 节奏不同步"调度封装成
// 一个简单接口：调用方只管 push_frame / wait_next_frame。
//
// 内部双 worker 模型：
//   - prefill_worker: 从 pending 队列拿 frame，调 stream_prefill(idx>=1)；
//                     stream_prefill 在 duplex 路径下立即返回（push 到 encoder_queue）
//   - decode_worker:  拿到 prefill 已提交的 frame_id，调 stream_decode；
//                     stream_decode 阻塞等 duplex_llm_thread 完成本帧 prefill+decode
// 两 worker 之间通过 decode_pending 队列保持顺序，与 duplex_llm_thread_func 内的
// 1:1 配对配合，确保 KV cache 严格按 push 顺序写入。
// ============================================================================


static void duplex_session_prefill_worker_func(omni_context * ctx_omni) {
    DuplexSession * sess = ctx_omni->duplex_session;
    while (true) {
        OmniDuplexPendingFrame pf;
        {
            std::unique_lock<std::mutex> lk(sess->pending_mtx);
            sess->pending_cv.wait(lk, [&]{
                return !sess->pending_frames.empty() || !sess->running.load();
            });
            if (!sess->running.load() && sess->pending_frames.empty()) break;
            pf = std::move(sess->pending_frames.front());
            sess->pending_frames.pop();
        }
        // pending 出队 → 通知 push_frame 阻塞侧（如有）队列有空位
        sess->pending_cv.notify_all();

        // 调底层 stream_prefill。duplex 路径下这是非阻塞的（推到 encoder_queue 立即返回）。
        bool ok = stream_prefill(ctx_omni, pf.frame.aud_fname, pf.frame.img_fname,
                                 (int)pf.frame_id, pf.frame.max_slice_nums);

        OmniDuplexInflightFrame inf;
        inf.frame_id       = pf.frame_id;
        inf.user_seq       = pf.frame.user_seq;
        inf.t_push         = pf.t_push;
        inf.t_push_steady  = pf.t_push_steady;
        inf.t_prefilled    = std::chrono::high_resolution_clock::now();
        inf.prefill_failed = !ok;
        {
            std::unique_lock<std::mutex> lk(sess->decode_mtx);
            sess->decode_pending.push(std::move(inf));
        }
        sess->decode_cv.notify_all();
    }
}

static void duplex_session_decode_worker_func(omni_context * ctx_omni) {
    DuplexSession * sess = ctx_omni->duplex_session;
    while (true) {
        OmniDuplexInflightFrame inf;
        {
            std::unique_lock<std::mutex> lk(sess->decode_mtx);
            sess->decode_cv.wait(lk, [&]{
                return !sess->decode_pending.empty() || !sess->running.load();
            });
            if (!sess->running.load() && sess->decode_pending.empty()) break;
            inf = std::move(sess->decode_pending.front());
            sess->decode_pending.pop();
        }

        OmniDuplexFrameResult r;
        r.user_seq = inf.user_seq;
        r.frame_id = inf.frame_id;

        if (inf.prefill_failed) {
            r.ok = false;
        } else {
            // 等本帧 LLM 完成
            double t_first_token_ms = 0.0;
            auto t_dec_start = std::chrono::high_resolution_clock::now();
            bool ok = stream_decode(ctx_omni, sess->debug_dir, (int)inf.frame_id, &t_first_token_ms);
            auto t_dec_end   = std::chrono::high_resolution_clock::now();

            r.ok       = ok;
            // TTFT：push → 本帧首个输出 token（绝对时间差）
            if (ok && t_first_token_ms > 0.0) {
                const double t_push_ms = std::chrono::duration<double, std::milli>(
                    inf.t_push.time_since_epoch()).count();
                r.ms_ttft = t_first_token_ms - t_push_ms;
            }
            r.is_speak = !ctx_omni->ended_with_listen.load();

            // 收集本帧文本（剔除控制 token）
            {
                std::lock_guard<std::mutex> lock(ctx_omni->text_mtx);
                while (!ctx_omni->text_queue.empty()) {
                    std::string piece = ctx_omni->text_queue.front();
                    ctx_omni->text_queue.pop_front();
                    if (piece == "__IS_LISTEN__" || piece == "__END_OF_TURN__") continue;
                    r.text += piece;
                }
            }

            r.n_past_after = ctx_omni->n_past;
            r.ms_decode = std::chrono::duration<double, std::milli>(t_dec_end - t_dec_start).count();
            r.ms_total  = std::chrono::duration<double, std::milli>(t_dec_end - inf.t_push).count();
        }
        r.ms_prefill_submit = std::chrono::duration<double, std::milli>(inf.t_prefilled - inf.t_push).count();

        {
            std::unique_lock<std::mutex> lk(sess->done_mtx);
            sess->done_results.push(std::move(r));
        }
        sess->done_cv.notify_all();
        sess->in_flight.fetch_sub(1);
    }
}

bool omni_duplex_session_begin(struct omni_context * ctx_omni,
                               const std::string & voice_audio,
                               const std::string & debug_dir) {
    if (!ctx_omni) {
        LOG_ERR("omni_duplex_session_begin: ctx_omni is null\n");
        return false;
    }
    if (ctx_omni->duplex_session != nullptr) {
        LOG_WRN("omni_duplex_session_begin: session already exists, refuse to begin twice\n");
        return false;
    }
    if (!ctx_omni->duplex_mode || !ctx_omni->async) {
        LOG_ERR("omni_duplex_session_begin: requires duplex_mode=true && async=true\n");
        return false;
    }

    // 1. 走 index=0 的 stream_prefill：初始化 system prompt + voice clone +
    //    启动 duplex pipeline (encoder_thread / llm_thread) + tts/t2w 线程。
    if (!stream_prefill(ctx_omni, voice_audio, /*img*/"", /*index*/0)) {
        LOG_ERR("omni_duplex_session_begin: stream_prefill(voice_audio, idx=0) failed\n");
        return false;
    }

    // 2. 创建 session 并启动 worker 线程
    auto * sess = new DuplexSession();
    sess->running.store(true);
    sess->debug_dir = debug_dir.empty() ? std::string("./") : debug_dir;
    ctx_omni->duplex_session = sess;

    sess->prefill_worker = std::thread(duplex_session_prefill_worker_func, ctx_omni);
    sess->decode_worker  = std::thread(duplex_session_decode_worker_func,  ctx_omni);

    print_with_timestamp("omni_duplex_session_begin: session ready (workers started)\n");
    return true;
}

int64_t omni_duplex_push_frame(struct omni_context * ctx_omni,
                               const OmniDuplexFrame & frame) {
    if (!ctx_omni || !ctx_omni->duplex_session) return -1;
    DuplexSession * sess = ctx_omni->duplex_session;
    if (!sess->running.load()) return -1;

    OmniDuplexPendingFrame pf;
    pf.frame    = frame;
    pf.frame_id = sess->frame_id_counter.fetch_add(1) + 1;  // chunk 0 已被 session_begin 占用
    pf.t_push   = std::chrono::high_resolution_clock::now();
    pf.t_push_steady = std::chrono::steady_clock::now();

    {
        std::unique_lock<std::mutex> lk(sess->pending_mtx);
        sess->pending_cv.wait(lk, [&]{
            return sess->pending_frames.size() < DuplexSession::PENDING_MAX || !sess->running.load();
        });
        if (!sess->running.load()) {
            return -1;
        }
        sess->pending_frames.push(std::move(pf));
    }
    sess->in_flight.fetch_add(1);
    sess->pending_cv.notify_all();
    return pf.frame_id;
}

bool omni_duplex_wait_next_frame(struct omni_context * ctx_omni,
                                 OmniDuplexFrameResult * out,
                                 int timeout_ms) {
    if (!ctx_omni || !ctx_omni->duplex_session || !out) return false;
    DuplexSession * sess = ctx_omni->duplex_session;

    std::unique_lock<std::mutex> lk(sess->done_mtx);
    if (timeout_ms < 0) {
        sess->done_cv.wait(lk, [&]{
            return !sess->done_results.empty() || !sess->running.load();
        });
    } else if (timeout_ms == 0) {
        if (sess->done_results.empty()) return false;
    } else {
        bool got = sess->done_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&]{
            return !sess->done_results.empty() || !sess->running.load();
        });
        if (!got) return false;
    }
    if (sess->done_results.empty()) return false;
    *out = std::move(sess->done_results.front());
    sess->done_results.pop();
    return true;
}

void omni_duplex_get_wav_walls(struct omni_context * ctx_omni,
                               std::vector<double> & out) {
    if (ctx_omni == nullptr) return;
    std::lock_guard<std::mutex> lk(ctx_omni->t2w_wav_mtx);
    out = ctx_omni->t2w_wav_walls;
}

bool omni_duplex_drain_tts_audio(struct omni_context * ctx_omni,
                                 int max_wait_ms,
                                 int idle_ms) {
    if (!ctx_omni) return false;
    if (!ctx_omni->async || !ctx_omni->use_tts) return true;

    const int tick_ms = 100;
    const int idle_ticks_required = std::max(1, idle_ms / tick_ms);
    const int max_ticks            = std::max(idle_ticks_required, max_wait_ms / tick_ms);

    int idle_ticks = 0;
    for (int i = 0; i < max_ticks; ++i) {
        if (omni_tts_queues_empty(ctx_omni)) {
            if (++idle_ticks >= idle_ticks_required) return true;
        } else {
            idle_ticks = 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(tick_ms));
    }
    return false;
}

void omni_duplex_session_end(struct omni_context * ctx_omni) {
    if (!ctx_omni || !ctx_omni->duplex_session) return;
    DuplexSession * sess = ctx_omni->duplex_session;

    // 1. 等所有已 push 的帧 LLM 完成（drain）
    //    注意：调用方负责 wait_next_frame 把 done_results 取空；这里只等 in_flight=0。
    while (sess->in_flight.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // 2. 通知 worker 退出
    sess->running.store(false);
    sess->pending_cv.notify_all();
    sess->decode_cv.notify_all();
    sess->done_cv.notify_all();

    if (sess->prefill_worker.joinable()) sess->prefill_worker.join();
    if (sess->decode_worker.joinable())  sess->decode_worker.join();

    delete sess;
    ctx_omni->duplex_session = nullptr;
    print_with_timestamp("omni_duplex_session_end: session destroyed\n");
}

bool stop_speek(struct omni_context * ctx_omni){
    ctx_omni->speek_done = true;
    if (ctx_omni->use_tts && ctx_omni->tts_thread_info) {
        std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
        while(!ctx_omni->tts_thread_info->queue.empty()){
            LLMOut *llm_out = ctx_omni->tts_thread_info->queue.front();
            if (llm_out) {
                delete llm_out;
                llm_out = nullptr;
            }
            ctx_omni->tts_thread_info->queue.pop();
        }
        ctx_omni->tts_thread_info->cv.notify_all();
    }
    return true;
}

bool clean_kvcache(struct omni_context * ctx_omni) {
    
    if (ctx_omni->clean_kvcache) {
        print_with_timestamp("🧹 clean_kvcache: 清理 KV cache, 删除范围=[%d, %d), n_keep=%d\n",
                             ctx_omni->n_keep, ctx_omni->n_past, ctx_omni->n_keep);
        
        // 获取 memory 对象并清理 KV cache
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
        if (mem) {
            // 删除 [n_keep, n_past) 范围的所有 token，保留 system prompt 等
            bool rm_ok = llama_memory_seq_rm(mem, 0, ctx_omni->n_keep, ctx_omni->n_past);
            if (!rm_ok) {
                print_with_timestamp("🧹 clean_kvcache: llama_memory_seq_rm 失败\n");
            } else {
                print_with_timestamp("🧹 clean_kvcache: llama_memory_seq_rm 成功\n");
            }
        } else {
            print_with_timestamp("🧹 clean_kvcache: 无法获取 memory 对象\n");
        }
        
        // 重置 n_past 到 n_keep
        int old_n_past = ctx_omni->n_past;
        ctx_omni->n_past = ctx_omni->n_keep;
        print_with_timestamp("🧹 clean_kvcache: n_past 从 %d 重置到 %d\n", old_n_past, ctx_omni->n_past);
        
        // 重置滑窗状态
        sliding_window_reset(ctx_omni);
        // 但保留 system_preserve_length，因为 n_keep 部分仍然保留
        ctx_omni->system_preserve_length = ctx_omni->n_keep;
        print_with_timestamp("🧹 clean_kvcache: 滑窗状态已重置, system_preserve_length=%d\n", ctx_omni->system_preserve_length);
    } else {
        print_with_timestamp("🧹 clean_kvcache: clean_kvcache=false, 跳过清理\n");
    }
    
    return true;
}
