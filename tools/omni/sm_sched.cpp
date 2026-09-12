// sm_sched.cpp — 段租约调度器实现。
//
// 单锁 + 单一 snapshot（active|reserved 位图）+ 纯函数决策（sm_active_set.h）
// + 设备级整卡切换（sm_apply_plan，只在 active==0 的 quiescent 点发生）。
// 非 CUDA 平台整体 no-op。

#include "sm_sched.h"

#ifdef GGML_USE_CUDA

#    include "audition.h"                  // audition_get_backend
#    include "omni.h"                      // omni_context
#    include "omni_internal.h"             // print_with_timestamp
#    include "token2wav/token2wav-impl.h"  // Token2WavSession::t2w.gpu_backends()
#    include "vision.h"                    // vision_get_backend

#    include <cctype>
#    include <chrono>
#    include <cstdarg>
#    include <cstdio>
#    include <cstdlib>
#    include <cstring>
#    include <ctime>
#    include <string>

// SM 观测走 stderr：stdout 重定向到文件时是全缓冲，进程若中途挂起，缓冲内容会丢。
// 调度诊断不能丢。整行一次性 fwrite 到 stderr（单次写，glibc 按调用加锁），与
// print_with_timestamp 一致，避免行被并发打印从中间切断。
static void sm_log(const char * fmt, ...) {
    const auto now = std::chrono::system_clock::now();
    const auto in_time_t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    std::tm buf;
    localtime_r(&in_time_t, &buf);
    char ts[64];
    strftime(ts, sizeof(ts), "%H:%M:%S", &buf);

    char line[2048];
    const int n = snprintf(line, sizeof(line), "%s.%03d ", ts, (int) ms.count());
    if (n < 0) return;
    va_list args;
    va_start(args, fmt);
    const int m = vsnprintf(line + n, sizeof(line) - (size_t) n, fmt, args);
    va_end(args);
    if (m < 0) return;

    const size_t avail   = sizeof(line) - (size_t) n;
    const size_t written = (size_t) m < avail ? (size_t) m : avail - 1;
    fwrite(line, 1, (size_t) n + written, stderr);
}

static const char * sm_plan_name(int plan) {
    switch (plan) {
        case SM_PLAN_GLOBAL:  return "GLOBAL";
        case SM_PLAN_STEADY:  return "STEADY";
        case SM_PLAN_ENC_T2W: return "ENC_T2W";
        case SM_PLAN_ENC_LLM: return "ENC_LLM";
        case SM_PLAN_ENC_TTS: return "ENC_TTS";
        default:              return "?";
    }
}

static double sm_now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int sm_component_from_name(const std::string & name) {
    if (name == "LLM") {
        return SM_G_LLM;
    }
    if (name == "TTS") {
        return SM_G_TTS;
    }
    if (name == "APM") {
        return SM_G_APM;
    }
    if (name == "VPM") {
        return SM_G_VPM;
    }
    if (name == "T2W") {
        return SM_G_T2W;
    }
    return -1;
}

// Parse one optional physical-group sharing set, for example APM+TTS+T2W.
// The logical component bits remain separate; this only changes group layout.
static bool sm_parse_share_set(const std::string & text, bool shared[SM_COMPONENT_COUNT]) {
    for (int c = 0; c < SM_COMPONENT_COUNT; ++c) {
        shared[c] = false;
    }
    if (text.empty()) {
        return true;
    }

    size_t pos = 0;
    int    n   = 0;
    while (pos < text.size()) {
        const size_t      end       = text.find('+', pos);
        const std::string name      = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        const int         component = sm_component_from_name(name);
        if (component < 0 || shared[component]) {
            return false;
        }
        shared[component] = true;
        ++n;
        if (end == std::string::npos) {
            break;
        }
        pos = end + 1;
        if (pos == text.size()) {
            return false;
        }
    }
    return n >= 2;
}

static bool sm_parse_sms_list(const std::string & text, int out[SM_COMPONENT_COUNT]) {
    int    n   = 0;
    size_t pos = 0;
    while (pos < text.size() && n < SM_COMPONENT_COUNT) {
        while (pos < text.size() && (text[pos] == ',' || std::isspace((unsigned char) text[pos]))) {
            ++pos;
        }
        if (pos >= text.size() || text[pos] < '0' || text[pos] > '9') {
            return false;
        }
        int value = 0;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
            value = value * 10 + (text[pos] - '0');
            ++pos;
        }
        out[n++] = value;
    }
    while (pos < text.size() && std::isspace((unsigned char) text[pos])) {
        ++pos;
    }
    return n == SM_COMPONENT_COUNT && pos == text.size();
}

static bool sm_build_layout(const int      logical_sms[SM_COMPONENT_COUNT],
                            const bool     shared[SM_COMPONENT_COUNT],
                            SmPlanLayout & out) {
    out              = SmPlanLayout{};
    int shared_group = -1;
    out.n_groups     = 0;
    for (int c = 0; c < SM_COMPONENT_COUNT; ++c) {
        if (shared[c]) {
            if (shared_group < 0) {
                shared_group = out.n_groups++;
            }
            out.group_of[c] = shared_group;
        } else {
            out.group_of[c] = out.n_groups++;
        }
    }
    for (int c = 0; c < SM_COMPONENT_COUNT; ++c) {
        out.sms[out.group_of[c]] += logical_sms[c];
    }
    return true;
}

static void sm_log_layout(const char * prefix, const SmPlanLayout & layout) {
    char line[256];
    int  n = std::snprintf(line, sizeof(line), "[SM] %s groups=%d LLM=%d TTS=%d APM=%d VPM=%d T2W=%d sms=", prefix,
                           layout.n_groups, layout.group_of[SM_G_LLM], layout.group_of[SM_G_TTS],
                           layout.group_of[SM_G_APM], layout.group_of[SM_G_VPM], layout.group_of[SM_G_T2W]);
    if (n < 0 || (size_t) n >= sizeof(line)) {
        return;
    }
    for (int i = 0; i < layout.n_groups; ++i) {
        n += std::snprintf(line + n, sizeof(line) - (size_t) n, "%s%d", i == 0 ? "" : ",", layout.sms[i]);
        if (n < 0 || (size_t) n >= sizeof(line)) {
            return;
        }
    }
    sm_log("%s\n", line);
}

// 设备级整卡切换：5 个 backend + t2w_backends 一次 set_sm_group。
// plan == SM_PLAN_GLOBAL：group_idx=-1 回全局（primary 全 SM），cache_key=-1 不缓存。
// 否则按 plan 的物理 group 映射绑定，cache_key=plan（槽制缓存，切回零重建/零重捕获）。
// 任一 backend 切换失败 → 回滚已切的 backend 到旧 plan，返回 false（不推进 epoch）。
static bool sm_apply_plan(SmSched & s, int plan) {
    const int64_t t0 = ggml_time_us();
    const int old_plan = s.applied_plan;

    struct One {
        ggml_backend_t b;
        int            component;
    };
    std::vector<One> ones;
    ones.push_back({s.backend[SM_G_LLM], SM_G_LLM});
    ones.push_back({s.backend[SM_G_TTS], SM_G_TTS});
    ones.push_back({s.backend[SM_G_APM], SM_G_APM});
    ones.push_back({s.backend[SM_G_VPM], SM_G_VPM});
    for (auto b : s.t2w_backends) ones.push_back({b, SM_G_T2W});

    auto switch_one = [&](ggml_backend_t b, int component, int cache_key) -> bool {
        if (b == nullptr) {
            return true;
        }
        if (plan == SM_PLAN_GLOBAL) {
            return ggml_backend_cuda_set_sm_group(b, nullptr, -1, cache_key);
        }
        ggml_backend_cuda_sm_plan * p = s.presets[plan];
        if (p == nullptr) {
            return false;
        }
        const SmPlanLayout & layout = s.layouts[plan];
        if (layout.n_groups <= 0 || layout.n_groups > SM_COMPONENT_COUNT || component < 0 ||
            component >= SM_COMPONENT_COUNT) {
            return false;
        }
        return ggml_backend_cuda_set_sm_group(b, p, layout.group_of[component], cache_key);
    };

    std::vector<One> switched;
    bool ok = true;
    for (auto & o : ones) {
        if (o.b == nullptr) continue;
        if (!switch_one(o.b, o.component, plan)) {
            ok = false;
            break;
        }
        switched.push_back(o);
    }
    if (!ok && !switched.empty()) {
        for (auto & o : switched) {
            if (old_plan == SM_PLAN_GLOBAL) {
                ggml_backend_cuda_set_sm_group(o.b, nullptr, -1, old_plan);
            } else {
                ggml_backend_cuda_sm_plan * p = s.presets[old_plan];
                if (p) {
                    const SmPlanLayout & layout = s.layouts[old_plan];
                    ggml_backend_cuda_set_sm_group(o.b, p, layout.group_of[o.component], old_plan);
                }
            }
        }
        fprintf(stderr, "[SM SWITCH] plan=%s apply failed, rolled back to %s\n",
                sm_plan_name(plan), sm_plan_name(old_plan));
        return false;
    }

    s.switch_count++;
    s.switch_total_ms += (ggml_time_us() - t0) / 1000.0;
    s.epoch++;
    sm_log("[SM SWITCH] plan=%s epoch=%llu total=%.3fms\n",
           sm_plan_name(plan), (unsigned long long) s.epoch,
           (ggml_time_us() - t0) / 1000.0);
    return ok;
}

void sm_enter(SmSched & s, uint32_t bit) {
    if (!s.enabled) return;
    std::unique_lock<std::mutex> lk(s.mtx);

    const uint64_t ticket = s.next_ticket++;
    while (ticket != s.serving_ticket) s.cv.wait(lk);   // FIFO 准入
    s.reserved |= bit;
    for (;;) {
        const SmDecision d = sm_active_set_decide(s.active, s.reserved);
        if (d.plan == s.applied_plan) break;
        if (s.active != 0) { s.cv.wait(lk); continue; } // 等全卡 quiescent（释放锁）
        if (!sm_apply_plan(s, d.plan)) {
            // Apply failure leaves every backend on old_plan.  Continue in that
            // plan rather than publishing a false scheduler snapshot.
            sm_log("[SM] plan=%s unavailable, keep applied=%s\n",
                   sm_plan_name(d.plan), sm_plan_name(s.applied_plan));
            break;
        }
        s.applied_plan = d.plan;
    }
    s.active |= bit;
    s.reserved &= ~bit;
    sm_log("[SM SEG] enter bit=0x%x plan=%s epoch=%llu t=%.3fms\n",
           bit, sm_plan_name(s.applied_plan),
           (unsigned long long) s.epoch, sm_now_ms());
    s.serving_ticket++;
    s.cv.notify_all();
}

void sm_segment_boundary(SmSched & s, uint32_t bit) {
    if (!s.enabled) return;
    std::unique_lock<std::mutex> lk(s.mtx);

    s.active &= ~bit;
    s.reserved |= bit;
    sm_log("[SM SEG] exit bit=0x%x next=1 plan=%s epoch=%llu t=%.3fms\n",
           bit, sm_plan_name(s.applied_plan),
           (unsigned long long) s.epoch, sm_now_ms());

    // A yielded segment remains reserved while other active segments reach
    // their own synchronized boundaries.  The last one performs the single
    // device-wide apply, then all yielded continuations can resume.
    for (;;) {
        const SmDecision d = sm_active_set_decide(s.active, s.reserved);
        if (d.plan != s.applied_plan) {
            if (s.active != 0) {
                s.cv.wait(lk);
                continue;
            }
            if (!sm_apply_plan(s, d.plan)) {
                sm_log("[SM] boundary plan=%s unavailable, keep applied=%s\n",
                       sm_plan_name(d.plan), sm_plan_name(s.applied_plan));
            } else {
                s.applied_plan = d.plan;
            }
        }

        s.active |= bit;
        s.reserved &= ~bit;
        sm_log("[SM SEG] enter bit=0x%x plan=%s epoch=%llu t=%.3fms (cont)\n",
               bit, sm_plan_name(s.applied_plan),
               (unsigned long long) s.epoch, sm_now_ms());
        s.cv.notify_all();
        return;
    }
}

void sm_exit(SmSched & s, uint32_t bit) {
    if (!s.enabled) return;
    std::unique_lock<std::mutex> lk(s.mtx);
    s.active &= ~bit;
    s.reserved &= ~bit;
    if (s.active == 0) {                                  // 最后一个 active 段退出 → deferred apply
        const SmDecision d = sm_active_set_decide(s.active, s.reserved);
        if (d.plan != s.applied_plan) {
            if (sm_apply_plan(s, d.plan)) {
                s.applied_plan = d.plan;
            } else {
                sm_log("[SM] exit plan=%s unavailable, keep applied=%s\n",
                       sm_plan_name(d.plan), sm_plan_name(s.applied_plan));
            }
        }
    }
    sm_log("[SM SEG] exit bit=0x%x next=0 plan=%s epoch=%llu t=%.3fms\n",
           bit, sm_plan_name(s.applied_plan),
           (unsigned long long) s.epoch, sm_now_ms());
    s.cv.notify_all();
}

void sm_task_finish(SmSched & s, int selected_plan, uint32_t bit,
                    const char * kind, uint64_t frame_id) {
    if (!s.enabled) return;
    std::lock_guard<std::mutex> lk(s.mtx);
    sm_log("[SM TL] finish kind=%s bit=0x%x frame=%llu selected_plan=%s applied_plan=%s epoch=%llu t=%.3fms\n",
           kind, bit, (unsigned long long) frame_id,
           sm_plan_name(selected_plan), sm_plan_name(s.applied_plan),
           (unsigned long long) s.epoch, sm_now_ms());
}

void sm_sched_free(SmSched & s) {
    std::lock_guard<std::mutex> lk(s.mtx);
    s.enabled = false;
    s.active = 0;
    s.reserved = 0;
    for (auto & plan : s.presets) {
        if (plan != nullptr) {
            ggml_backend_cuda_sm_plan_free(plan);
            plan = nullptr;
        }
    }
    s.t2w_backends.clear();
}

// ---------------------------------------------------------------------------
// init 装配：OMNI_SM_SMS 控制。DYN = 4 preset + 动态切换；显式列表 = 单静态 plan
// （配额实验用，可选）。不设 / 0 = 关闭调度。
// ---------------------------------------------------------------------------
void sm_sched_init(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) return;
    const char * env_c = getenv("OMNI_SM_SMS");
    if (env_c == nullptr || *env_c == '\0' || strcmp(env_c, "0") == 0) {
        return;  // 关闭调度
    }

    const std::string spec(env_c);
    const size_t      at = spec.find('@');
    if (at != std::string::npos && spec.find('@', at + 1) != std::string::npos) {
        fprintf(stderr, "[SM] OMNI_SM_SMS='%s' has more than one '@', scheduler disabled\n", env_c);
        return;
    }
    const std::string base                       = at == std::string::npos ? spec : spec.substr(0, at);
    const std::string share                      = at == std::string::npos ? std::string() : spec.substr(at + 1);
    bool              shared[SM_COMPONENT_COUNT] = {};
    if (!sm_parse_share_set(share, shared)) {
        fprintf(stderr, "[SM] OMNI_SM_SMS='%s' invalid share set '%s', scheduler disabled\n", env_c, share.c_str());
        return;
    }

    SmSched & s = ctx_omni->sm_sched;
    // [LLM,TTS,APM,VPM,T2W]
    static const int DYN_SMS[4][SM_COMPONENT_COUNT] = {
        /* STEADY  */ { 40, 8,  8, 56, 16 },
        /* ENC_T2W    */
        { 2,  2,  2, 78, 44 },
        /* ENC_LLM    */
        { 40, 2,  4, 80, 2  },
        /* ENC_TTS    */
        { 2,  28, 2, 94, 2  },
    };
    const int  plan_ids[4]                    = { SM_PLAN_STEADY, SM_PLAN_ENC_T2W, SM_PLAN_ENC_LLM, SM_PLAN_ENC_TTS };
    int        static_sms[SM_COMPONENT_COUNT] = {};
    const bool is_dyn                         = base == "DYN";
    if (!is_dyn && (!sm_parse_sms_list(base, static_sms) || static_sms[SM_G_LLM] <= 0)) {
        fprintf(stderr, "[SM] OMNI_SM_SMS='%s' invalid base strategy, scheduler disabled\n", env_c);
        return;
    }

    auto install_plan = [&](int plan_id, const int logical_sms[SM_COMPONENT_COUNT]) -> bool {
        SmPlanLayout layout;
        if (!sm_build_layout(logical_sms, shared, layout)) {
            return false;
        }
        s.layouts[plan_id] = layout;
        s.presets[plan_id] = ggml_backend_cuda_sm_plan_create(0, layout.sms, layout.n_groups, false);
        return s.presets[plan_id] != nullptr;
    };

    bool plans_ok = true;
    if (is_dyn) {
        for (int i = 0; i < 4; ++i) {
            plans_ok = install_plan(plan_ids[i], DYN_SMS[i]) && plans_ok;
        }
    } else {
        plans_ok = install_plan(SM_PLAN_STEADY, static_sms);
    }
    if (!plans_ok) {
        for (auto & plan : s.presets) {
            if (plan != nullptr) {
                ggml_backend_cuda_sm_plan_free(plan);
                plan = nullptr;
            }
        }
        fprintf(stderr, "[SM] OMNI_SM_SMS='%s' plan creation failed, scheduler disabled\n", env_c);
        return;
    }

    // wire backends（duplex_prepare 之后、prewarm 之前）
    s.backend[SM_G_LLM] = llama_get_gpu_backend(ctx_omni->ctx_llama);
    s.backend[SM_G_TTS] = llama_get_gpu_backend(ctx_omni->ctx_tts_llama);
    s.backend[SM_G_APM] = audition_get_backend(ctx_omni->ctx_audio);
    s.backend[SM_G_VPM] = vision_get_backend(ctx_omni->ctx_vision);
    if (ctx_omni->token2wav_session != nullptr) {
        s.t2w_backends = ctx_omni->token2wav_session->gpu_backends();
    }

    s.enabled = true;
    s.applied_plan = SM_PLAN_STEADY;
    sm_log("[SM] OMNI_SM_SMS=%s enabled presets=%s/%s/%s/%s\n", env_c, sm_plan_name(SM_PLAN_STEADY),
           sm_plan_name(SM_PLAN_ENC_T2W), sm_plan_name(SM_PLAN_ENC_LLM), sm_plan_name(SM_PLAN_ENC_TTS));
    for (int i = 0; i < 4; ++i) {
        if (s.presets[plan_ids[i]] != nullptr) {
            char label[64];
            std::snprintf(label, sizeof(label), "layout plan=%s", sm_plan_name(plan_ids[i]));
            sm_log_layout(label, s.layouts[plan_ids[i]]);
        }
    }
}

#else
void sm_enter(SmSched &, uint32_t) {}
void sm_segment_boundary(SmSched &, uint32_t) {}
void sm_exit(SmSched &, uint32_t) {}
void sm_task_finish(SmSched &, int, uint32_t, const char *, uint64_t) {}
void sm_sched_free(SmSched &) {}
void sm_sched_init(struct omni_context *) {}
#endif
