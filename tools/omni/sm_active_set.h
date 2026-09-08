#pragma once
#include <cstdint>

// 逻辑组件索引（与 preset 表顺序一致：LLM,TTS,APM,VPM,T2W）。
// 该索引用于 active/reserved 位图，不等同于物理 SM group 索引。
enum : int {
    SM_G_LLM = 0, SM_G_TTS = 1, SM_G_APM = 2, SM_G_VPM = 3, SM_G_T2W = 4,
    SM_COMPONENT_COUNT = 5,
};
// 物理分片位
enum : uint32_t {
    SM_BIT_LLM = 1u << SM_G_LLM, SM_BIT_TTS = 1u << SM_G_TTS,
    SM_BIT_APM = 1u << SM_G_APM, SM_BIT_VPM = 1u << SM_G_VPM,
    SM_BIT_T2W = 1u << SM_G_T2W,
};
// 整卡 plan id（GLOBAL 是虚拟 plan，无句柄）
enum : int { SM_PLAN_GLOBAL = -1, SM_PLAN_STEADY = 0, SM_PLAN_ENC_T2W = 4, SM_PLAN_ENC_LLM = 5, SM_PLAN_ENC_TTS = 6 };

struct SmDecision {
    int plan = SM_PLAN_GLOBAL;
    uint32_t logical_mask = 0;   // bit0=ENC bit1=LLM bit2=TALKER
    bool fallback = false;
};

// 纯函数：live set (active|reserved) → 整卡 plan。逻辑 = 主工作区 omni_sm_active_set_decide。
inline SmDecision sm_active_set_decide(uint32_t active, uint32_t reserved) {
    SmDecision out;
    const uint32_t mask = active | reserved;
    const bool enc  = (mask & (SM_BIT_APM | SM_BIT_VPM)) != 0;
    const bool llm  = (mask & SM_BIT_LLM) != 0;
    const bool tts  = (mask & SM_BIT_TTS) != 0;
    const bool t2w  = (mask & SM_BIT_T2W) != 0;
    const bool talk = tts || t2w;
    out.logical_mask = (enc ? 1u : 0u) | (llm ? 2u : 0u) | (talk ? 4u : 0u);
    const int groups = (enc ? 1 : 0) + (llm ? 1 : 0) + (talk ? 1 : 0);
    if (groups <= 1) return out;                          // 空或单逻辑组 → GLOBAL
    if (enc && llm && !talk) { out.plan = SM_PLAN_ENC_LLM; return out; }
    if (enc && talk && !llm) {
        if (t2w && !tts) { out.plan = SM_PLAN_ENC_T2W; return out; }
        if (tts && !t2w) { out.plan = SM_PLAN_ENC_TTS; return out; }
        out.fallback = true;                              // enc+tts+t2w
    } else { out.fallback = true; }                       // llm+talk / 三组 / 其它
    out.plan = SM_PLAN_STEADY;
    return out;
}
