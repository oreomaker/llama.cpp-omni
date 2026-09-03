#pragma once
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>
#include <ggml.h>
#include "sm_active_set.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"   // ggml_backend_cuda_sm_plan
#endif

// 段租约调度器：单锁 + 单一 snapshot。enabled=false 时全部 no-op（= 无调度开销）。
//
// 一个 GPU 计算 = 一串带同步边界的段；调度只在段边界发生。逻辑任务首段通过
// sm_enter 申请租约，后续段通过 sm_segment_boundary 在安全点协作恢复，任务结束时
// sm_exit 归还租约。租约绑定"当前整卡 plan"；跨一次整卡重划即失效。
struct SmSched {
    bool enabled = false;
    std::mutex mtx;
    std::condition_variable cv;
    uint32_t active = 0;          // 正在 GPU 计算的段位
    uint32_t reserved = 0;        // 已准入未激活 / 段间让出中的 continuation 位
    int applied_plan = SM_PLAN_GLOBAL;
    uint64_t epoch = 0;           // 成功整卡切换次数
    uint64_t next_ticket = 1, serving_ticket = 1;   // 准入 FIFO
#ifdef GGML_USE_CUDA
    ggml_backend_t backend[5] = {};                 // 索引 = SM_G_*；T2W 用 t2w_backends
    std::vector<ggml_backend_t> t2w_backends;
    ggml_backend_cuda_sm_plan * presets[7] = {};    // 索引 = plan id（0..6）；STEADY=0
#endif
    uint64_t switch_count = 0;
    double switch_total_ms = 0.0;
};

void sm_enter(SmSched & s, uint32_t bit);                 // 逻辑任务/首段开始
void sm_segment_boundary(SmSched & s, uint32_t bit);      // 当前段完成，等待边界汇合后恢复下一段
void sm_exit(SmSched & s, uint32_t bit);                  // 逻辑任务结束
// 逻辑任务收尾观测：调用方在最后一次 sm_exit 之后调用，fold 一行精简观测。
// selected_plan = 任务首次 sm_enter 完成时的 applied（调用方在首次 enter 后自行抓取传入）。
void sm_task_finish(SmSched & s, int selected_plan, uint32_t bit,
                    const char * kind, uint64_t frame_id);
void sm_sched_free(SmSched & s);

// init 装配：OMNI_SM_SMS=DYN 时建 4 个 preset + wire backends + enabled=true；
// 不设 / 0 时 no-op（关闭调度）。duplex_prepare 之后、prewarm 之前调用。
struct omni_context;
void sm_sched_init(struct omni_context * ctx_omni);
