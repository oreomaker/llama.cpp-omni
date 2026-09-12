#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <stdint.h>

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

// ---------------------------------------------------------------------------
// SM 计划（green context 协调式 split）：把设备的 SM 一次切成 N 个 disjoint 分组，
// 各 backend 显式使用其中一组，避免多 backend 并发时 per-backend 各自 split 导致的
// SM 重叠/均等时间片（见 tools/omni/SM_MULTI_GC_HARNESS.md §4.1）。
// 仅 CUDA >= 13.1 生效；否则 sm_plan_create 返回 NULL，init_sm 退化为普通 init（全 SM）。
// ---------------------------------------------------------------------------
typedef struct ggml_backend_cuda_sm_plan ggml_backend_cuda_sm_plan;

// 一次协调 split 出 n_groups 个 disjoint 分组。balanced_wq=1 时各分组额外使用
// cudaDevWorkqueueConfigScopeGreenCtxBalanced（在 disjoint 之上约 +7%，非必需）。
// 返回 NULL 表示不支持/失败。
GGML_BACKEND_API ggml_backend_cuda_sm_plan * ggml_backend_cuda_sm_plan_create(
    int device, const int * sms, int n_groups, bool balanced_wq);
GGML_BACKEND_API void ggml_backend_cuda_sm_plan_free(ggml_backend_cuda_sm_plan * plan);

// 取该 device 当前注册的 SM 计划（sm_plan_create 时注册，free 时注销）；无则 NULL。
// 供直接创建 backend 的调用方（如 T2W loader）按组号取 plan。
GGML_BACKEND_API ggml_backend_cuda_sm_plan * ggml_backend_cuda_sm_plan_get(int device);

// 用计划中的第 group_idx 组 SM 创建 CUDA backend（green context 用该 disjoint 分组）。
// plan 为 NULL 时按 device 查注册表；都找不到 / 索引越界 / CUDA < 13.1 时回退为普通
// ggml_backend_cuda_init（全 SM，primary）。
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init_sm(
    int device, const ggml_backend_cuda_sm_plan * plan, int group_idx);

// 返回 plan 第 group_idx 组实际分到的 SM 数（cudaDevSmResourceSplit 回写值）；
// plan 为 NULL / 索引越界 / CUDA < 13.1 时返回 -1。用于确认"同一整卡多次 split 各得完整分区"。
GGML_BACKEND_API int ggml_backend_cuda_sm_plan_group_sms(
    const ggml_backend_cuda_sm_plan * plan, int group_idx);

// 运行时切换 backend 使用的 SM 分组（green context）。调用方**必须**保证此时 backend 空闲
// （已同步、无在跑的 chunk/帧）。cache_key >= 0：EC + stream + graph 都按 key 缓存，切回同一分组
// 时零重建/零重捕获（Phase 2，SM_GC_60 §7）；cache_key < 0：每次重建 EC、销毁 stream、清 graph。
// group_idx == -1：回全局（解绑 green context，本 backend 拿满设备全部 SM，等价于不隔离；plan 忽略）。
// 返回 true 表示切换成功（green ctx 就绪 / 已回全局）；CUDA < 13.1 / plan 或 group 无效时返回 false。
GGML_BACKEND_API bool ggml_backend_cuda_set_sm_group(
    ggml_backend_t backend, const ggml_backend_cuda_sm_plan * plan, int group_idx, int cache_key);

// Cumulative graph and SM-switch observations for a CUDA backend.  These are
// intentionally lightweight counters so host schedulers can attribute cold
// starts and graph misses to a component without parsing stderr.
typedef struct ggml_backend_cuda_graph_stats {
    uint64_t hits;
    uint64_t misses;
    uint64_t captures;
    uint64_t instances;
    // Instances in the currently selected SM slot only.  The legacy
    // instances field includes retained graph slots for diagnostics.
    uint64_t current_instances;
    uint64_t slot_hits;
    uint64_t slot_misses;
    uint64_t property_misses;
    uint64_t warmup_resets;
    uint64_t event_serial;
    uint64_t last_graph_key;
    uint64_t last_graph_uid;
    int32_t  last_graph_result;
    int32_t  current_slot_key;
} ggml_backend_cuda_graph_stats;

enum ggml_backend_cuda_graph_result {
    GGML_CUDA_GRAPH_RESULT_NONE = 0,
    GGML_CUDA_GRAPH_RESULT_MISS,
    GGML_CUDA_GRAPH_RESULT_CAPTURE,
    GGML_CUDA_GRAPH_RESULT_HIT,
    GGML_CUDA_GRAPH_RESULT_PROPERTY_MISS,
};

GGML_BACKEND_API bool ggml_backend_cuda_get_graph_stats(
    ggml_backend_t backend, ggml_backend_cuda_graph_stats * stats);

typedef struct ggml_backend_cuda_sm_switch_stats {
    double total_ms;
    double stream_ec_ms;
} ggml_backend_cuda_sm_switch_stats;

GGML_BACKEND_API bool ggml_backend_cuda_get_last_sm_switch_stats(
    ggml_backend_t backend, ggml_backend_cuda_sm_switch_stats * stats);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
