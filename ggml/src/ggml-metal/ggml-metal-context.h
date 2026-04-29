#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
void ggml_metal_free(ggml_metal_t ctx);

void ggml_metal_synchronize(ggml_metal_t ctx);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);
bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

//
// timed events for profiling (uses MTLCommandBuffer GPUStartTime/GPUEndTime)
//

typedef struct ggml_metal_timed_event * ggml_metal_timed_event_t;

ggml_metal_timed_event_t ggml_metal_timed_event_init(void);
void ggml_metal_timed_event_free(ggml_metal_timed_event_t event);
void ggml_metal_timed_event_synchronize(ggml_metal_timed_event_t event);
bool ggml_metal_timed_event_is_completed(ggml_metal_timed_event_t event);
double ggml_metal_timed_event_get_gpu_start_time(ggml_metal_timed_event_t event);
double ggml_metal_timed_event_get_gpu_end_time(ggml_metal_timed_event_t event);

// record a timed event on the backend's current command buffer (cmd_buf_last)
void ggml_metal_record_timed_event(ggml_metal_t ctx, ggml_metal_timed_event_t event);

#ifdef __cplusplus
}
#endif
