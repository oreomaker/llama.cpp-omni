// Encoder compute: VPM image embed / APM audio embed — compute layer: model/graph execution. No frame/queue/worker knowledge.
// ==========================================================================

#include "omni-impl.h"
#include "omni.h"
#include "omni_internal.h"
#include "common/common.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool vpm_profile_enabled() {
    const char * value = std::getenv("OMNI_VPM_PROFILE");
    return value != nullptr && std::atoi(value) != 0;
}

static bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long *sizeOut) {
    auto file = fopen(path, "rb");
    if (file == NULL) {
        LOG_ERR("%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char *)malloc(fileSize); // Allocate memory to hold the file data
    if (buffer == NULL) {
        LOG_ERR("%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        perror("Memory allocation error");
        fclose(file);
        return false;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file); // Read the file into the buffer
    if (ferror(file)) {
        die_fmt("read error: %s", strerror(errno));
    }
    if (ret != (size_t) fileSize) {
        die("unexpectedly reached end of file");
    }
    fclose(file); // Close the file

    *bytesOut = buffer;
    *sizeOut = fileSize;
    return true;
}

void omni_embed_free(struct omni_embed * embed) {
    free(embed->embed);
    free(embed);
}

// 高清模式：返回分离的 chunk embeds（二维 vector）
// 返回值: vision_chunks[0] = overview, vision_chunks[1..n] = slices
// 当 slice 数 > 1 时，使用 batched encoding 加速
static bool encode_image_with_vision_chunks(vision_ctx * ctx_vision, int n_threads, const vision_image_u8 * img, 
                                            std::vector<std::vector<float>> & vision_chunks,
                                            vision_segment_callback_t segment_callback = nullptr,
                                            void * segment_callback_user_data = nullptr) {
    const int64_t t_img_enc_start_us = ggml_time_us();
    vision_image_f32_batch img_res_v;
    img_res_v.entries.resize(0);
    img_res_v.entries.clear();
    const int64_t t_preprocess_start_us = ggml_time_us();
    if (!vision_image_preprocess(ctx_vision, img, &img_res_v)) {
        LOG_ERR("%s: unable to preprocess image\n", __func__);
        img_res_v.entries.clear();
        return false;
    }
    if (vpm_profile_enabled()) {
        std::fprintf(stderr, "[VPM PROF] preprocess=%.3fms images=%zu t=%.3fms\n",
                     (ggml_time_us() - t_preprocess_start_us) / 1000.0,
                     img_res_v.entries.size(), ggml_time_us() / 1000.0);
    }
    
    int n_embd = vision_n_mmproj_embd(ctx_vision);
    int n_tokens = vision_n_output_tokens(ctx_vision);
    const int n_total = (int)img_res_v.entries.size();
    
    vision_chunks.clear();
    vision_chunks.resize(n_total);

    // entry[0] is the overview image; entry[1..n] are slices (all same size)
    // strategy: encode overview separately, batch all slices together
    // 批量编码优化由 vision_set_batch_encode 控制（默认关闭）。该优化只在
    // 大图/高清高刷等 slice 数较多时收益明显，并非通用，因此需显式开启。
    const int n_slices = n_total - 1;
    const bool use_batched = vision_get_batch_encode(ctx_vision) && (n_slices > 1);

    // 1) encode the overview image (entry 0)
    {
        const int64_t t_step_us = ggml_time_us();
        vision_chunks[0].resize(n_embd * n_tokens);
        const int group_size = segment_callback != nullptr ? 4 : 0;
        bool encoded = vision_image_encode_with_callback(
            ctx_vision, n_threads, img_res_v.entries[0].get(), vision_chunks[0].data(),
            group_size, segment_callback, segment_callback_user_data);
        if (!encoded) {
            LOG_ERR("Unable to encode overview image\n");
            return false;
        }
        LOG_INF("%s: overview encoded in %8.2f ms\n", __func__, (ggml_time_us() - t_step_us) / 1000.0);
    }

    // 2) encode slices
    if (n_slices > 0) {
        if (use_batched) {
            // batched encoding: group all slices into a single batch
            const int64_t t_batch_us = ggml_time_us();
            vision_image_f32_batch slice_batch;
            for (int i = 1; i < n_total; i++) {
                slice_batch.entries.push_back(std::move(img_res_v.entries[i]));
            }

            std::vector<float> batched_output(n_embd * n_tokens * n_slices);
            const int group_size = segment_callback != nullptr ? 4 : 0;
            bool encoded = segment_callback != nullptr
                ? vision_image_batch_encode_block_groups_with_callback(
                    ctx_vision, n_threads, &slice_batch, group_size, batched_output.data(),
                    nullptr, segment_callback, segment_callback_user_data)
                : vision_image_batch_encode(ctx_vision, n_threads, &slice_batch, batched_output.data());
            if (!encoded) {
                LOG_ERR("Unable to batch-encode %d slices\n", n_slices);
                return false;
            }

            // split batched output into individual chunks
            const int per_image_floats = n_embd * n_tokens;
            for (int i = 0; i < n_slices; i++) {
                vision_chunks[1 + i].resize(per_image_floats);
                std::memcpy(vision_chunks[1 + i].data(),
                            batched_output.data() + i * per_image_floats,
                            per_image_floats * sizeof(float));
            }

            LOG_INF("%s: %d slices batch-encoded in %8.2f ms\n", __func__, n_slices, (ggml_time_us() - t_batch_us) / 1000.0);
        } else {
            // serial path: encode every slice individually (entries[1..n_total-1])
            const int64_t t_step_us = ggml_time_us();
            for (int i = 1; i < n_total; i++) {
                vision_chunks[i].resize(n_embd * n_tokens);
                const int group_size = segment_callback != nullptr ? 4 : 0;
                bool encoded = vision_image_encode_with_callback(
                    ctx_vision, n_threads, img_res_v.entries[i].get(), vision_chunks[i].data(),
                    group_size, segment_callback, segment_callback_user_data);
                if (!encoded) {
                    // entries[i] is the i-th slice (entry 0 is the overview), so the
                    // 1-based slice number equals i, out of n_slices total.
                    LOG_ERR("Unable to encode slice %d of %d\n", i, n_slices);
                    return false;
                }
            }
            LOG_INF("%s: %d slice(s) encoded serially in %8.2f ms\n", __func__, n_slices, (ggml_time_us() - t_step_us) / 1000.0);
        }
    }

    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;
    int total_tokens = n_total * n_tokens;
    LOG_INF("\n%s: image encoded in %8.2f ms by vision (%d chunks, %d total tokens, grid: %dx%d)\n", 
            __func__, t_img_enc_ms, n_total, total_tokens,
            img_res_v.grid_x, img_res_v.grid_y);

    return true;
}

static int query_gpu_memory_used_mb(int device_id = 0) {
#ifdef GGML_USE_CUDA
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_cuda_get_device_memory(device_id, &free_bytes, &total_bytes);
    if (total_bytes == 0) return -1;
    return (int)((total_bytes - free_bytes) / (1024 * 1024));
#else
    (void) device_id;
    return -1;
#endif
}

void omni_bench_vision(struct vision_ctx * ctx_vision, int n_threads, const char * image_path) {
    unsigned char* image_bytes;
    long image_bytes_length;
    if (!load_file_to_bytes(image_path, &image_bytes, &image_bytes_length)) {
        LOG_ERR("bench: failed to load %s\n", image_path);
        return;
    }
    vision_image_u8 * img = vision_image_u8_init();
    if (!vision_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        vision_image_u8_free(img);
        free(image_bytes);
        LOG_ERR("bench: can't load image\n");
        return;
    }
    free(image_bytes);

    int n_embd = vision_n_mmproj_embd(ctx_vision);
    int n_tokens = vision_n_output_tokens(ctx_vision);
    const int per_image_floats = n_embd * n_tokens;
    const int bench_runs = 3;

    const int baseline_vram = query_gpu_memory_used_mb();
    LOG_INF("\n================================================================\n");
    LOG_INF("  Vision Encoding Benchmark: Serial vs Batched (GPU %d)\n", 0);
    LOG_INF("================================================================\n");
    LOG_INF("Image: %s\n", image_path);
    LOG_INF("Per-slice: %d embd x %d tokens = %d floats (%.1f MB)\n",
            n_embd, n_tokens, per_image_floats, per_image_floats * 4.0f / 1048576.0f);
    LOG_INF("Baseline VRAM: %d MB\n", baseline_vram);
    LOG_INF("NOTE: VRAM figures are device-wide (cudaMemGetInfo), single-point samples, "
            "NOT per-process peak; they include other processes on the same GPU.\n");
    LOG_INF("Bench runs per config: %d\n\n", bench_runs);

    struct bench_result {
        int max_slice_cfg;
        int actual_slices;
        int grid_x, grid_y;
        double serial_ms;
        double batch_ms;
        double speedup;
        int vram_serial_mb;
        int vram_batch_mb;
        double max_diff;
        double avg_diff;
    };
    std::vector<bench_result> results;

    const int slice_configs[] = {1, 2, 3, 4, 6, 8, 9, 12, 16, 20, 25, 30, 36};
    const int n_configs = sizeof(slice_configs) / sizeof(slice_configs[0]);

    // warmup with 1 slice
    vision_set_max_slice_nums(ctx_vision, 1);
    {
        vision_image_f32_batch warmup_batch;
        vision_image_preprocess(ctx_vision, img, &warmup_batch);
        if (!warmup_batch.entries.empty()) {
            std::vector<float> tmp(per_image_floats);
            vision_image_encode(ctx_vision, n_threads, warmup_batch.entries[0].get(), tmp.data());
        }
    }

    for (int ci = 0; ci < n_configs; ci++) {
        int max_s = slice_configs[ci];
        vision_set_max_slice_nums(ctx_vision, max_s);

        vision_image_f32_batch img_res_v;
        if (!vision_image_preprocess(ctx_vision, img, &img_res_v)) {
            LOG_ERR("bench: preprocess failed for max_slice=%d\n", max_s);
            continue;
        }

        const int n_total = (int)img_res_v.entries.size();
        const int n_slices = n_total - 1;

        if (n_slices < 2) {
            LOG_INF("[max_slice=%2d] grid=%dx%d => %d slice(s), skipping (need >=2)\n",
                    max_s, img_res_v.grid_x, img_res_v.grid_y, n_slices);
            continue;
        }

        LOG_INF("\n--- max_slice_nums=%2d => grid=%dx%d, %d slices ---\n",
                max_s, img_res_v.grid_x, img_res_v.grid_y, n_slices);

        bench_result br;
        br.max_slice_cfg = max_s;
        br.actual_slices = n_slices;
        br.grid_x = img_res_v.grid_x;
        br.grid_y = img_res_v.grid_y;

        // --- serial encoding ---
        std::vector<std::vector<float>> serial_out(n_slices);
        double serial_total = 0;
        for (int r = 0; r < bench_runs; r++) {
            const int64_t t0 = ggml_time_us();
            for (int i = 0; i < n_slices; i++) {
                serial_out[i].resize(per_image_floats);
                vision_image_encode(ctx_vision, n_threads, img_res_v.entries[1 + i].get(), serial_out[i].data());
            }
            double ms = (ggml_time_us() - t0) / 1000.0;
            serial_total += ms;
            LOG_INF("  serial  run %d: %8.2f ms\n", r, ms);
        }
        br.serial_ms = serial_total / bench_runs;
        br.vram_serial_mb = query_gpu_memory_used_mb();

        // --- batched encoding ---
        std::vector<float> batch_out(per_image_floats * n_slices);
        double batch_total = 0;
        for (int r = 0; r < bench_runs; r++) {
            vision_image_f32_batch slice_batch;
            for (int i = 0; i < n_slices; i++) {
                vision_image_f32_ptr cp(new vision_image_f32(*img_res_v.entries[1 + i]));
                slice_batch.entries.push_back(std::move(cp));
            }
            const int64_t t0 = ggml_time_us();
            vision_image_batch_encode(ctx_vision, n_threads, &slice_batch, batch_out.data());
            double ms = (ggml_time_us() - t0) / 1000.0;
            batch_total += ms;
            LOG_INF("  batched run %d: %8.2f ms\n", r, ms);
        }
        br.batch_ms = batch_total / bench_runs;
        br.vram_batch_mb = query_gpu_memory_used_mb();
        br.speedup = br.serial_ms / br.batch_ms;

        // --- correctness ---
        double max_d = 0, sum_d = 0;
        int n_vals = per_image_floats * n_slices;
        for (int i = 0; i < n_slices; i++) {
            for (int j = 0; j < per_image_floats; j++) {
                double d = std::abs((double)serial_out[i][j] - (double)batch_out[i * per_image_floats + j]);
                if (d > max_d) max_d = d;
                sum_d += d;
            }
        }
        br.max_diff = max_d;
        br.avg_diff = sum_d / n_vals;

        LOG_INF("  => serial=%7.1f ms  batch=%7.1f ms  speedup=%.2fx  vram_s=%dMB vram_b=%dMB  max_diff=%.3e avg_diff=%.3e\n",
                br.serial_ms, br.batch_ms, br.speedup,
                br.vram_serial_mb, br.vram_batch_mb, br.max_diff, br.avg_diff);

        results.push_back(br);
    }
    vision_image_u8_free(img);

    // === summary table ===
    LOG_INF("\n\n================================================================\n");
    LOG_INF("                      SUMMARY TABLE\n");
    LOG_INF("================================================================\n");
    LOG_INF("%-6s %-6s %-8s %-10s %-10s %-8s %-10s %-10s %-10s %-10s\n",
            "MaxS", "Grid", "Slices", "Serial(ms)", "Batch(ms)", "Speedup",
            "VRAM_S(MB)", "VRAM_B(MB)", "MaxDiff", "AvgDiff");
    LOG_INF("%-6s %-6s %-8s %-10s %-10s %-8s %-10s %-10s %-10s %-10s\n",
            "------", "------", "--------", "----------", "----------", "--------",
            "----------", "----------", "----------", "----------");
    for (auto & r : results) {
        char grid_str[16];
        snprintf(grid_str, sizeof(grid_str), "%dx%d", r.grid_x, r.grid_y);
        LOG_INF("%-6d %-6s %-8d %-10.1f %-10.1f %-8.2f %-10d %-10d %-10.3e %-10.3e\n",
                r.max_slice_cfg, grid_str, r.actual_slices,
                r.serial_ms, r.batch_ms, r.speedup,
                r.vram_serial_mb, r.vram_batch_mb,
                r.max_diff, r.avg_diff);
    }
    LOG_INF("================================================================\n");
    LOG_INF("Baseline VRAM (model loaded, no encoding): %d MB\n", baseline_vram);
    LOG_INF("VRAM_S/VRAM_B are device-wide single-point samples (not per-process peak).\n");
    LOG_INF("================================================================\n\n");
}

// 保留原有函数用于兼容（将所有 chunk 拼成一个 flat buffer）
static bool encode_image_with_vision(vision_ctx * ctx_vision, int n_threads, const vision_image_u8 * img, float * image_embd, int * n_img_pos) {
    std::vector<std::vector<float>> vision_chunks;
    if (!encode_image_with_vision_chunks(ctx_vision, n_threads, img, vision_chunks)) {
        return false;
    }
    
    int n_embd = vision_n_mmproj_embd(ctx_vision);
    int n_tokens = vision_n_output_tokens(ctx_vision);
    int n_img_pos_out = 0;
    
    for (size_t i = 0; i < vision_chunks.size(); i++) {
        std::memcpy(image_embd + n_img_pos_out * n_embd, vision_chunks[i].data(), n_embd * n_tokens * sizeof(float));
        n_img_pos_out += n_tokens;
    }
    *n_img_pos = n_img_pos_out;
    LOG_INF("%s: image embedding created: %d tokens from %d chunks\n", __func__, *n_img_pos, (int)vision_chunks.size());
    
    return true;
}

static void build_vision_image_from_data(const stbi_uc * data, int nx, int ny, vision_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    std::memcpy(img->buf.data(), data, img->buf.size());
}

bool vision_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct vision_image_u8 * img) {
    int nx, ny, nc;
    auto * data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_ERR("%s: failed to decode image bytes\n", __func__);
        return false;
    }
    build_vision_image_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

struct omni_embed * omni_image_embed_make_with_bytes(struct vision_ctx * ctx_vision, int n_threads, const unsigned char * image_bytes, int image_bytes_length) {
    vision_image_u8 * img = vision_image_u8_init();
    if (!vision_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        vision_image_u8_free(img);
        LOG_ERR("%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }
    int num_max_patches = 10;
    float * image_embed = (float *)malloc(vision_n_mmproj_embd(ctx_vision) * vision_n_output_tokens(ctx_vision) * num_max_patches * sizeof(float));
    if (!image_embed) {
        vision_image_u8_free(img);
        LOG_ERR("Unable to allocate memory for image embeddings\n");
        return NULL;
    }

    LOG_INF("%s: omni_image_embed_make_with_filename s1\n", __func__);
    int n_img_pos = 0;
    if (!encode_image_with_vision(ctx_vision, n_threads, img, image_embed, &n_img_pos)) {
        vision_image_u8_free(img);
        free(image_embed);
        LOG_ERR("%s: cannot encode image, aborting\n", __func__);
        return NULL;
    }
    LOG_INF("%s: omni_image_embed_make_with_filename s2\n", __func__);

    vision_image_u8_free(img);
    auto result = (omni_embed*)malloc(sizeof(omni_embed));
    result->embed = image_embed;
    result->n_pos = n_img_pos;
    return result;
}

struct omni_embed * omni_image_embed_make_with_filename(struct vision_ctx * ctx_vision, int n_threads, std::string image_path) {
    unsigned char* image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(image_path.c_str(), &image_bytes, &image_bytes_length);
    if (!loaded) {
        LOG_ERR("%s: failed to load %s\n", __func__, image_path.c_str());
        return NULL;
    }
    LOG_INF("%s: omni_image_embed_make_with_filename: %s\n", __func__, image_path.c_str());
    omni_embed *embed = omni_image_embed_make_with_bytes(ctx_vision, n_threads, image_bytes, image_bytes_length);
    free(image_bytes);

    return embed;
}

// 高清模式：创建带 chunks 的 vision embed（用于 V2.6 slice schema）
// 返回的 vector: [0] = overview, [1..n] = slices
bool omni_image_embed_make_chunks_with_filename(struct vision_ctx * ctx_vision, int n_threads,
                                                 std::string image_path,
                                                 std::vector<std::vector<float>> & vision_chunks,
                                                 vision_segment_callback_t segment_callback,
                                                 void * segment_callback_user_data) {
    unsigned char* image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(image_path.c_str(), &image_bytes, &image_bytes_length);
    if (!loaded) {
        LOG_ERR("%s: failed to load %s\n", __func__, image_path.c_str());
        return false;
    }
    
    vision_image_u8 * img = vision_image_u8_init();
    if (!vision_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        vision_image_u8_free(img);
        free(image_bytes);
        LOG_ERR("%s: can't load image from bytes, is it a valid image?", __func__);
        return false;
    }
    free(image_bytes);
    
    bool success = encode_image_with_vision_chunks(ctx_vision, n_threads, img, vision_chunks,
                                                   segment_callback, segment_callback_user_data);
    vision_image_u8_free(img);
    
    if (success) {
        LOG_INF("%s: created %d vision chunks from %s\n", __func__, (int)vision_chunks.size(), image_path.c_str());
    }
    return success;
}

bool audition_read_binary_file(const char * fname, std::vector<uint8_t> * buf_res) {
    FILE * f = fopen(fname, "rb");
    if (!f) {
        LOG_ERR("Unable to open file %s: %s\n", fname, strerror(errno));
        return false;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf_res->resize(file_size);

    size_t n_read = fread(buf_res->data(), 1, file_size, f);
    fclose(f);
    if (n_read != (size_t)file_size) {
        LOG_ERR("Failed to read entire file %s\n", fname);
        return false;
    }

    return true;
}

struct omni_embed * omni_audio_embed_make_with_bytes(audition_ctx * ctx_audio, int n_threads, audition_audio_u8 * audio) {
    audition_audio_f32 * res_auds = audition_audio_f32_init();
    // printf("omni_audio_embed_make_with_bytes 1 :\n");
    if (!audition_audio_preprocess(ctx_audio, audio, &res_auds)) {
        LOG_ERR("%s: failed to preprocess audio file\n", __func__);
        audition_audio_f32_free(res_auds);
        return NULL;
    }
    // printf("omni_audio_embed_make_with_bytes 2 :\n");
    // 分配空间
    int n_embd = audition_n_mmproj_embd(ctx_audio);
    int n_tokens = audition_n_output_tokens(ctx_audio, res_auds);
    std::vector<float> output_buffer(n_embd * n_tokens);
    
    if (!audition_audio_encode(ctx_audio, n_threads, res_auds, output_buffer.data())) {
        LOG_ERR("%s: cannot encode audio, aborting\n", __func__);
        audition_audio_f32_free(res_auds);
        return NULL;
    }
    // printf("omni_audio_embed_make_with_bytes 4 :\n");
    auto result = (omni_embed*)malloc(sizeof(omni_embed));
    result->embed = (float *)malloc(output_buffer.size() * sizeof(float));
    if (!result->embed) {
        free(result);
        audition_audio_f32_free(res_auds);
        LOG_ERR("%s: failed to allocate memory for audio embeddings\n", __func__);
        return NULL;
    }
    std::memcpy(result->embed, output_buffer.data(), output_buffer.size() * sizeof(float));
    result->n_pos = n_tokens;
    
    audition_audio_f32_free(res_auds);
    // printf("===audio embed tokens: %d %d\n", result->n_pos, ret.buf.size() / ret.n_len);
    return result;
}

struct omni_embed * omni_audio_embed_make_with_filename(struct audition_ctx * ctx_audio, int n_threads, std::string audio_path) {
    audition_audio_u8 * audio = audition_audio_u8_init();
    // printf("omni_audio_embed_make_with_filename 1 :%s\n", audio_path.c_str());
    if (!audition_read_binary_file(audio_path.c_str(), &audio->buf)) {
        LOG_ERR("%s: failed to read audio file %s\n", __func__,  audio_path.c_str());
        return NULL;
    }
    // printf("omni_audio_embed_make_with_filename 2 :%s\n", audio_path.c_str());
    omni_embed *embed = omni_audio_embed_make_with_bytes(ctx_audio, n_threads, audio);
    if (embed == NULL) {
        LOG_ERR("%s: failed to preprocess audio file, %s\n", __func__, audio_path.c_str());
    }

    audition_audio_u8_free(audio);
    // printf("omni_audio_embed_make_with_filename 3 :%s\n", audio_path.c_str());
    return embed;
}

//
// omni llm eval
//
