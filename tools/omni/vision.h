#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <memory>

struct vision_ctx;

struct vision_image_size {
    int width;
    int height;
};

// RGB uint8 image
struct vision_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// For images, buf.size() == nx*ny*3
//     Memory layout: RGBRGBRGB...
// For audio, only one channel is used, buf.size() == nx*ny
//     nx will be n_frames and ny will be n_mel
struct vision_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

// forward declarations for deleters
void vision_image_u8_free(struct vision_image_u8  * img);
void vision_image_f32_free(struct vision_image_f32 * img);

struct vision_image_u8_deleter {
    void operator()(vision_image_u8 * val) { vision_image_u8_free(val); }
};
typedef std::unique_ptr<vision_image_u8, vision_image_u8_deleter> vision_image_u8_ptr;
struct vision_image_u8_batch {
    std::vector<vision_image_u8_ptr> entries;
};

struct vision_image_f32_deleter {
    void operator()(vision_image_f32 * val) { vision_image_f32_free(val); }
};
typedef std::unique_ptr<vision_image_f32, vision_image_f32_deleter> vision_image_f32_ptr;
struct vision_image_f32_batch {
    std::vector<vision_image_f32_ptr> entries;

    // for llava-uhd style models, we need to know the grid size
    // note: entries.size() == grid_x * grid_y + 1 (one overview image)
    int grid_x = 0;
    int grid_y = 0;

    vision_image_f32_batch clone() const {
        vision_image_f32_batch new_batch{
            /* entries  */ {},
            /* grid_x   */ grid_x,
            /* grid_y   */ grid_y,
        };
        new_batch.entries.reserve(entries.size());
        for (const auto & entry : entries) {
            new_batch.entries.emplace_back(new vision_image_f32(*entry));
        }
        return new_batch;
    }
};

struct vision_context_params {
    bool use_gpu;
    enum ggml_log_level verbosity;
    const char * coreml_model_path; // path to CoreML model (.mlmodelc) for ANE acceleration
};

struct vision_block_group_timing {
    int first_layer = 0;
    int last_layer  = 0;
    double elapsed_ms = 0.0;
};

struct vision_block_group_profile {
    int n_layers        = 0;
    int block_group_size = 0;
    int graph_nodes     = 0;
    std::vector<vision_block_group_timing> segments;
};

// Called after a grouped graph range has completed and the scheduler has
// synchronized the range's backend.  The callback is optional and must not
// mutate the graph or its tensors.
using vision_segment_callback_t = void (*)(int segment_id, int segment_count,
                                           void * user_data);


//
// vision functions
//

// vision layers
struct vision_ctx * vision_init(const char * fname, struct vision_context_params ctx_params);
void vision_free(struct vision_ctx * ctx);

// 返回 VPM 的 CUDA/GPU 计算 backend（无则 NULL）。当前无调用方，预留给调度层。
ggml_backend_t vision_get_backend(struct vision_ctx * ctx);

// vision processor
void vision_image_u8_free(struct vision_image_u8  * img);
void vision_image_f32_free(struct vision_image_f32 * img);
struct vision_image_u8        * vision_image_u8_init (void);
struct vision_image_f32       * vision_image_f32_init(void);
struct vision_image_f32_batch * vision_image_f32_batch_init(void);

// decode an encoded image (PNG/JPEG/...) from memory into a vision_image_u8
bool vision_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct vision_image_u8 * img);

// vision query
int vision_n_output_tokens(const struct vision_ctx * ctx);
int vision_n_mmproj_embd(const struct vision_ctx * ctx);
int vision_n_layers(const struct vision_ctx * ctx);

// vision preprocess
bool vision_image_preprocess(struct vision_ctx * ctx, const struct vision_image_u8 * img, struct vision_image_f32_batch * res_imgs );

// vision forward
bool vision_image_encode      (struct vision_ctx * ctx, int n_threads, struct vision_image_f32 * img, float * vec);
bool vision_image_batch_encode(struct vision_ctx * ctx, int n_threads, const struct vision_image_f32_batch * imgs, float * vec);

// Benchmark-only execution mode: dispatches the existing ViT graph in views
// ending after every block_group_size transformer blocks.  The logical graph,
// tensor allocation and final output are unchanged; block_group_size <= 0 is
// equivalent to vision_image_batch_encode().
bool vision_image_batch_encode_block_groups(
    struct vision_ctx * ctx,
    int n_threads,
    const struct vision_image_f32_batch * imgs,
    int block_group_size,
    float * vec,
    struct vision_block_group_profile * profile);

// Runtime variant with an optional cooperative-yield callback.  The original
// benchmark-only entry point remains unchanged for callers that only need
// timing/profile data.
bool vision_image_batch_encode_block_groups_with_callback(
    struct vision_ctx * ctx,
    int n_threads,
    const struct vision_image_f32_batch * imgs,
    int block_group_size,
    float * vec,
    struct vision_block_group_profile * profile,
    vision_segment_callback_t callback,
    void * callback_user_data);

bool vision_image_encode_with_callback(
    struct vision_ctx * ctx,
    int n_threads,
    struct vision_image_f32 * img,
    float * vec,
    int block_group_size,
    vision_segment_callback_t callback,
    void * callback_user_data);

// batch encode 开关：是否对多个尺寸相同的 slice 启用批量编码优化
// 默认关闭；只有大图/高清高刷等 slice 数较多的场景才建议开启
void vision_set_batch_encode(struct vision_ctx * ctx, bool enable);
bool vision_get_batch_encode(const struct vision_ctx * ctx);

// CoreML / ANE support
void vision_set_coreml_model_path(struct vision_ctx * ctx, const char * coreml_model_path);
bool vision_coreml_warmup(struct vision_ctx * ctx);
bool vision_image_batch_encode_coreml(struct vision_ctx * ctx, const struct vision_image_f32_batch * imgs, float * vec);
