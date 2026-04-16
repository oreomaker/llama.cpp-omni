#include "omni.h"

#include "audition.h"
#include "common/common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "omni-impl.h"
#include "omni-llm-stage.h"
#include "omni-log.h"
#include "omni-output.h"
#include "omni-python-t2w.h"
#include "omni-runtime-init.h"
#include "omni-sliding-window.h"
#include "omni-t2w-stage.h"
#include "omni-token-protocol.h"
#include "omni-tts-stage.h"
#include "omni-turn-coordinator.h"
#include "omni-worker-coordinator.h"
#include "token2wav/token2wav-impl.h"
#include "vision.h"

#ifdef GGML_USE_CUDA
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#ifdef _WIN32
#    include <direct.h>
#    include <io.h>
#    include <process.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <windows.h>
// Windows compatibility macros
#    define popen   _popen
#    define pclose  _pclose
#    define unlink  _unlink
#    define stat    _stat
#    define S_IFDIR _S_IFDIR
#else
#    include <sys/stat.h>
#    include <sys/time.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

//
// omni structure
//
namespace {
struct unit_buffer {
    std::vector<float> buffer;
    std::string        text;
    bool               completed   = false;
    int                unit_n_past = 0;
    float              duration;
};
}  // namespace

struct omni_output {
    std::vector<unit_buffer *> output;
    int                        idx;
};

omni_context::omni_context() = default;

omni_context::~omni_context() = default;

static double timing_elapsed_ms(const std::chrono::high_resolution_clock::time_point & start,
                                const std::chrono::high_resolution_clock::time_point & end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void duplex_timing_set_active_chunk(struct omni_context * ctx_omni, int chunk_idx) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    ctx_omni->active_duplex_chunk_idx = chunk_idx;
    ctx_omni->duplex_chunk_timings[chunk_idx];
}

static int duplex_timing_get_active_chunk(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    return ctx_omni->active_duplex_chunk_idx;
}

static void duplex_timing_note_vit(struct omni_context * ctx_omni, int chunk_idx, double ms) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.vit_embedding_ms            = timing.vit_embedding_ms < 0.0 ? ms : timing.vit_embedding_ms + ms;
}

static void duplex_timing_note_audio(struct omni_context * ctx_omni, int chunk_idx, double ms) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.audio_embedding_ms          = timing.audio_embedding_ms < 0.0 ? ms : timing.audio_embedding_ms + ms;
}

static void duplex_timing_note_tts(struct omni_context * ctx_omni, int chunk_idx, double ms, int audio_token_count) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.tts_audio_token_ms          = timing.tts_audio_token_ms < 0.0 ? ms : timing.tts_audio_token_ms + ms;
    timing.tts_audio_token_count += audio_token_count;
    timing.tts_done = true;
}

static void duplex_timing_note_t2w(struct omni_context * ctx_omni,
                                   int                   chunk_idx,
                                   double                ms,
                                   int                   window_count,
                                   bool                  done) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto &                      timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.token2wav_ms                = timing.token2wav_ms < 0.0 ? ms : timing.token2wav_ms + ms;
    timing.token2wav_window_count += window_count;
    if (done) {
        timing.token2wav_done = true;
    }
}

//
// omni mtmd embed
//
bool omni_eval_embed(llama_context * ctx_llama, const struct omni_embed * omni_embed, int n_batch, int * n_past) {
    int n_embd = llama_n_embd(llama_get_model(ctx_llama));

    for (int i = 0; i < omni_embed->n_pos; i += n_batch) {
        int         n_eval = std::min(omni_embed->n_pos - i, n_batch);
        llama_batch batch  = {};
        batch.n_tokens     = int32_t(n_eval);
        batch.embd         = (omni_embed->embed + i * n_embd);
        std::vector<llama_pos> pos_vec(n_eval);
        for (int j = 0; j < n_eval; j++) {
            pos_vec[j] = *n_past + j;
        }
        batch.pos = pos_vec.data();

        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

bool prefill_with_emb(struct omni_context * ctx_omni,
                      common_params *       params,
                      float *               embed,
                      int                   n_pos,
                      int                   n_batch,
                      int *                 n_past) {
    kv_cache_slide_window(ctx_omni, params, n_pos);

    int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    for (int i = 0; i < n_pos; i += n_batch) {
        int         n_eval = std::min(n_pos - i, n_batch);
        llama_batch batch  = {};
        batch.n_tokens     = int32_t(n_eval);
        batch.embd         = (embed + i * n_embd);
        std::vector<llama_pos> pos_vec(n_eval);
        for (int j = 0; j < n_eval; j++) {
            pos_vec[j] = *n_past + j;
        }
        batch.pos = pos_vec.data();

        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

// 与 prefill_with_emb 类似，但会将每次 decode 的 hidden_state 保存并拼接到 hidden_states 中
// hidden_states 由函数内部分配空间，大小为 n_pos * n_embd * sizeof(float)，调用者负责释放
bool prefill_emb_with_hidden(struct omni_context * ctx_omni,
                             common_params *       params,
                             float *               embed,
                             int                   n_pos,
                             int                   n_batch,
                             int *                 n_past,
                             float *&              hidden_states) {
    if (n_pos == 0) {
        hidden_states = nullptr;
        return true;
    }

    kv_cache_slide_window(ctx_omni, params, n_pos);

    int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));

    // 在函数内部分配空间
    hidden_states = (float *) malloc(n_pos * n_embd * sizeof(float));
    if (hidden_states == nullptr) {
        LOG_ERR("%s : failed to allocate memory for hidden_states\n", __func__);
        return false;
    }

    int tokens_processed = 0;

    for (int i = 0; i < n_pos; i += n_batch) {
        int n_eval = std::min(n_pos - i, n_batch);

        llama_batch batch = {};
        batch.n_tokens    = int32_t(n_eval);
        batch.embd        = (embed + i * n_embd);
        std::vector<llama_pos> pos_vec(n_eval);
        for (int j = 0; j < n_eval; j++) {
            pos_vec[j] = *n_past + j;
        }
        batch.pos = pos_vec.data();

        // 启用 embeddings 输出
        llama_set_embeddings(ctx_omni->ctx_llama, true);

        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            llama_set_embeddings(ctx_omni->ctx_llama, false);
            free(hidden_states);
            hidden_states = nullptr;
            return false;
        }

        // 获取当前 batch 的 embeddings 并复制到 hidden_states
        float * emb = llama_get_embeddings(ctx_omni->ctx_llama);
        if (emb != nullptr) {
            memcpy(hidden_states + tokens_processed * n_embd, emb, n_eval * n_embd * sizeof(float));
        }

        llama_set_embeddings(ctx_omni->ctx_llama, false);

        *n_past += n_eval;
        tokens_processed += n_eval;
    }
    return true;
}

static bool load_file_to_bytes(const char * path, unsigned char ** bytesOut, long * sizeOut) {
    auto * file = fopen(path, "rb");
    if (file == NULL) {
        LOG_ERR("%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto * buffer = (unsigned char *) malloc(fileSize);  // Allocate memory to hold the file data
    if (buffer == NULL) {
        LOG_ERR("%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        perror("Memory allocation error");
        fclose(file);
        return false;
    }
    errno      = 0;
    size_t ret = fread(buffer, 1, fileSize, file);  // Read the file into the buffer
    if (ferror(file)) {
        die_fmt("read error: %s", strerror(errno));
    }
    if (ret != (size_t) fileSize) {
        die("unexpectedly reached end of file");
    }
    fclose(file);  // Close the file

    *bytesOut = buffer;
    *sizeOut  = fileSize;
    return true;
}

void omni_embed_free(struct omni_embed * embed) {
    free(embed->embed);
    free(embed);
}

// 🔧 [高清模式] 返回分离的 chunk embeds（二维 vector）
// 返回值: vision_chunks[0] = overview, vision_chunks[1..n] = slices
static bool encode_image_with_vision_chunks(vision_ctx *                      ctx_vision,
                                            int                               n_threads,
                                            const vision_image_u8 *           img,
                                            std::vector<std::vector<float>> & vision_chunks) {
    const int64_t          t_img_enc_start_us = ggml_time_us();
    vision_image_f32_batch img_res_v;
    img_res_v.entries.resize(0);
    img_res_v.entries.clear();
    if (!vision_image_preprocess(ctx_vision, img, &img_res_v)) {
        LOG_ERR("%s: unable to preprocess image\n", __func__);
        img_res_v.entries.clear();
        return false;
    }

    int n_embd   = vision_n_mmproj_embd(ctx_vision);
    int n_tokens = vision_n_output_tokens(ctx_vision);

    vision_chunks.clear();
    vision_chunks.resize(img_res_v.entries.size());

    for (size_t i = 0; i < img_res_v.entries.size(); i++) {
        const int64_t t_img_enc_step_start_us = ggml_time_us();

        // 为每个 chunk 分配空间
        vision_chunks[i].resize(n_embd * n_tokens);

        bool encoded = vision_image_encode(ctx_vision, n_threads, img_res_v.entries[i].get(), vision_chunks[i].data());
        if (!encoded) {
            LOG_ERR("Unable to encode image - spatial_unpad - subimage %d of %d\n", (int) i + 1,
                    (int) img_res_v.entries.size());
            return false;
        }
        const int64_t t_img_enc_steop_batch_us = ggml_time_us();
        LOG_INF("%s: step %d of %d encoded in %8.2f ms\n", __func__, (int) i + 1, (int) img_res_v.entries.size(),
                (t_img_enc_steop_batch_us - t_img_enc_step_start_us) / 1000.0);
    }
    const int64_t t_img_enc_batch_us = ggml_time_us();
    LOG_INF("%s: all %d chunks encoded in %8.2f ms (grid: %dx%d)\n", __func__, (int) img_res_v.entries.size(),
            (t_img_enc_batch_us - t_img_enc_start_us) / 1000.0, img_res_v.grid_x, img_res_v.grid_y);

    const int64_t t_img_enc_end_us = ggml_time_us();
    float         t_img_enc_ms     = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;
    int           total_tokens     = (int) vision_chunks.size() * n_tokens;
    LOG_INF("\n%s: image encoded in %8.2f ms by vision (%8.2f ms per chunk, %d total tokens)\n", __func__, t_img_enc_ms,
            t_img_enc_ms / vision_chunks.size(), total_tokens);

    return true;
}

// 保留原有函数用于兼容（将所有 chunk 拼成一个 flat buffer）
static bool encode_image_with_vision(vision_ctx *            ctx_vision,
                                     int                     n_threads,
                                     const vision_image_u8 * img,
                                     float *                 image_embd,
                                     int *                   n_img_pos) {
    std::vector<std::vector<float>> vision_chunks;
    if (!encode_image_with_vision_chunks(ctx_vision, n_threads, img, vision_chunks)) {
        return false;
    }

    int n_embd        = vision_n_mmproj_embd(ctx_vision);
    int n_tokens      = vision_n_output_tokens(ctx_vision);
    int n_img_pos_out = 0;

    for (size_t i = 0; i < vision_chunks.size(); i++) {
        std::memcpy(image_embd + n_img_pos_out * n_embd, vision_chunks[i].data(), n_embd * n_tokens * sizeof(float));
        n_img_pos_out += n_tokens;
    }
    *n_img_pos = n_img_pos_out;
    LOG_INF("%s: image embedding created: %d tokens from %d chunks\n", __func__, *n_img_pos,
            (int) vision_chunks.size());

    return true;
}

static void build_vision_image_from_data(const stbi_uc * data, int nx, int ny, vision_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    std::memcpy(img->buf.data(), data, img->buf.size());
}

static bool vision_image_load_from_bytes(const unsigned char *    bytes,
                                         size_t                   bytes_length,
                                         struct vision_image_u8 * img) {
    int    nx;
    int    ny;
    int    nc;
    auto * data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_ERR("%s: failed to decode image bytes\n", __func__);
        return false;
    }
    build_vision_image_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

struct omni_embed * omni_image_embed_make_with_bytes(struct vision_ctx *   ctx_vision,
                                                     int                   n_threads,
                                                     const unsigned char * image_bytes,
                                                     int                   image_bytes_length) {
    vision_image_u8 * img = vision_image_u8_init();
    if (!vision_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        vision_image_u8_free(img);
        LOG_ERR("%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }
    int     num_max_patches = 10;
    float * image_embed     = (float *) malloc(vision_n_mmproj_embd(ctx_vision) * vision_n_output_tokens(ctx_vision) *
                                               num_max_patches * sizeof(float));
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
    auto * result = (omni_embed *) malloc(sizeof(omni_embed));
    result->embed = image_embed;
    result->n_pos = n_img_pos;
    return result;
}

struct omni_embed * omni_image_embed_make_with_filename(struct vision_ctx * ctx_vision,
                                                        int                 n_threads,
                                                        const std::string & image_path) {
    unsigned char * image_bytes;
    long            image_bytes_length;
    auto            loaded = load_file_to_bytes(image_path.c_str(), &image_bytes, &image_bytes_length);
    if (!loaded) {
        LOG_ERR("%s: failed to load %s\n", __func__, image_path.c_str());
        return NULL;
    }
    LOG_INF("%s: omni_image_embed_make_with_filename: %s\n", __func__, image_path.c_str());
    omni_embed * embed = omni_image_embed_make_with_bytes(ctx_vision, n_threads, image_bytes, image_bytes_length);
    free(image_bytes);

    return embed;
}

// 🔧 [高清模式] 创建带 chunks 的 vision embed（用于 V2.6 slice schema）
// 返回的 vector: [0] = overview, [1..n] = slices
static bool omni_image_embed_make_chunks_with_filename(struct vision_ctx *               ctx_vision,
                                                       int                               n_threads,
                                                       const std::string &               image_path,
                                                       std::vector<std::vector<float>> & vision_chunks) {
    unsigned char * image_bytes;
    long            image_bytes_length;
    auto            loaded = load_file_to_bytes(image_path.c_str(), &image_bytes, &image_bytes_length);
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

    bool success = encode_image_with_vision_chunks(ctx_vision, n_threads, img, vision_chunks);
    vision_image_u8_free(img);

    if (success) {
        LOG_INF("%s: created %d vision chunks from %s\n", __func__, (int) vision_chunks.size(), image_path.c_str());
    }
    return success;
}

static bool audition_read_binary_file(const char * fname, std::vector<uint8_t> * buf_res) {
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
    if (n_read != (size_t) file_size) {
        LOG_ERR("Failed to read entire file %s\n", fname);
        return false;
    }

    return true;
}

static struct omni_embed * omni_audio_embed_make_with_bytes(audition_ctx *      ctx_audio,
                                                            int                 n_threads,
                                                            audition_audio_u8 * audio) {
    audition_audio_f32 * res_auds = audition_audio_f32_init();
    // printf("omni_audio_embed_make_with_bytes 1 :\n");
    if (!audition_audio_preprocess(ctx_audio, audio, &res_auds)) {
        LOG_ERR("%s: failed to preprocess audio file\n", __func__);
        audition_audio_f32_free(res_auds);
        return NULL;
    }
    // printf("omni_audio_embed_make_with_bytes 2 :\n");
    // 分配空间
    int                n_embd   = audition_n_mmproj_embd(ctx_audio);
    int                n_tokens = audition_n_output_tokens(ctx_audio, res_auds);
    std::vector<float> output_buffer(n_embd * n_tokens);

    if (!audition_audio_encode(ctx_audio, n_threads, res_auds, output_buffer.data())) {
        LOG_ERR("%s: cannot encode audio, aborting\n", __func__);
        audition_audio_f32_free(res_auds);
        return NULL;
    }
    // printf("omni_audio_embed_make_with_bytes 4 :\n");
    auto * result = (omni_embed *) malloc(sizeof(omni_embed));
    result->embed = (float *) malloc(output_buffer.size() * sizeof(float));
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

struct omni_embed * omni_audio_embed_make_with_filename(struct audition_ctx * ctx_audition,
                                                        int                   n_threads,
                                                        const std::string &   audio_path) {
    audition_audio_u8 * audio = audition_audio_u8_init();
    // printf("omni_audio_embed_make_with_filename 1 :%s\n", audio_path.c_str());
    if (!audition_read_binary_file(audio_path.c_str(), &audio->buf)) {
        LOG_ERR("%s: failed to read audio file %s\n", __func__, audio_path.c_str());
        return NULL;
    }
    // printf("omni_audio_embed_make_with_filename 2 :%s\n", audio_path.c_str());
    omni_embed * embed = omni_audio_embed_make_with_bytes(ctx_audition, n_threads, audio);
    if (embed == NULL) {
        LOG_ERR("%s: failed to preprocess audio file, %s\n", __func__, audio_path.c_str());
    }

    audition_audio_u8_free(audio);
    // printf("omni_audio_embed_make_with_filename 3 :%s\n", audio_path.c_str());
    return embed;
}

//
// TTS specific helper functions
//

// ==============================================================================
// Projector Semantic 实现 (精度验证版本)
// 使用 ggml 后端进行计算，支持 CUDA 加速
// forward(x): relu(linear1(x)) -> linear2
// ==============================================================================
bool projector_init(projector_model & model, const std::string & fname, bool use_cuda) {
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), params);
    if (!ctx_gguf) {
        LOG_ERR("Projector: failed to open '%s'\n", fname.c_str());
        return false;
    }

#ifdef GGML_USE_CUDA
    if (use_cuda) {
        model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
        if (!model.backend) {
            model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        }
    } else {
        model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
#else
    (void) use_cuda;
    model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
#endif

    if (!model.backend) {
        LOG_ERR("Projector: failed to init backend\n");
        gguf_free(ctx_gguf);
        return false;
    }

    model.buf_type = ggml_backend_get_default_buffer_type(model.backend);

    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);

    size_t                  ctx_size   = ggml_tensor_overhead() * n_tensors;
    struct ggml_init_params ctx_params = {
        /*.mem_size   = */ ctx_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    model.ctx_w = ggml_init(ctx_params);

    for (int64_t i = 0; i < n_tensors; i++) {
        const char *         name   = gguf_get_tensor_name(ctx_gguf, i);
        enum ggml_type       type   = gguf_get_tensor_type(ctx_gguf, i);
        struct ggml_tensor * tensor = nullptr;

        if (strcmp(name, "linear1.weight") == 0) {
            tensor                     = ggml_new_tensor_2d(model.ctx_w, type, 4096, 768);
            model.layer.linear1_weight = tensor;
            model.hparams.in_dim       = 4096;
            model.hparams.out_dim      = 768;
        } else if (strcmp(name, "linear1.bias") == 0) {
            tensor                   = ggml_new_tensor_1d(model.ctx_w, type, 768);
            model.layer.linear1_bias = tensor;
        } else if (strcmp(name, "linear2.weight") == 0) {
            tensor                     = ggml_new_tensor_2d(model.ctx_w, type, 768, 768);
            model.layer.linear2_weight = tensor;
        } else if (strcmp(name, "linear2.bias") == 0) {
            tensor                   = ggml_new_tensor_1d(model.ctx_w, type, 768);
            model.layer.linear2_bias = tensor;
        } else {
            continue;
        }

        if (tensor) {
            ggml_set_name(tensor, name);
        }
    }

    model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);

    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        LOG_ERR("Projector: failed to open file for reading\n");
        return false;
    }

    for (int64_t i = 0; i < n_tensors; i++) {
        const char *         name   = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model.ctx_w, name);
        if (!tensor) {
            continue;
        }

        size_t offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
        fseek(f, offset, SEEK_SET);

        size_t tensor_size = ggml_nbytes(tensor);
        void * data        = malloc(tensor_size);
        if (fread(data, 1, tensor_size, f) != tensor_size) {
            LOG_ERR("Projector: failed to read tensor %s\n", name);
            free(data);
            fclose(f);
            return false;
        }

        ggml_backend_tensor_set(tensor, data, 0, tensor_size);
        free(data);
    }

    fclose(f);
    gguf_free(ctx_gguf);

    model.initialized = true;
    return true;
}

void projector_free(projector_model & model) {
    if (model.ctx_w) {
        ggml_free(model.ctx_w);
    }
    if (model.buf_w) {
        ggml_backend_buffer_free(model.buf_w);
    }
    if (model.backend) {
        ggml_backend_free(model.backend);
    }
    model.ctx_w       = nullptr;
    model.buf_w       = nullptr;
    model.backend     = nullptr;
    model.initialized = false;
}

static struct ggml_cgraph * projector_build_graph(projector_model &     model,
                                                  struct ggml_context * ctx,
                                                  struct ggml_tensor *  input) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // linear1 + relu
    struct ggml_tensor * hidden = ggml_mul_mat(ctx, model.layer.linear1_weight, input);
    hidden                      = ggml_add(ctx, hidden, model.layer.linear1_bias);
    hidden                      = ggml_relu(ctx, hidden);

    // linear2
    struct ggml_tensor * output = ggml_mul_mat(ctx, model.layer.linear2_weight, hidden);
    output                      = ggml_add(ctx, output, model.layer.linear2_bias);

    ggml_build_forward_expand(gf, output);
    return gf;
}

std::vector<float> projector_forward(projector_model & model, const float * input_data, int n_tokens) {
    const int in_dim  = model.hparams.in_dim;
    const int out_dim = model.hparams.out_dim;

    // 🔧 [安全检查] 验证参数
    if (n_tokens <= 0 || n_tokens > 10000) {
        LOG_ERR("projector_forward: invalid n_tokens=%d\n", n_tokens);
        return {};
    }
    if (in_dim <= 0 || in_dim > 10000 || out_dim <= 0 || out_dim > 10000) {
        LOG_ERR("projector_forward: invalid dimensions in_dim=%d, out_dim=%d\n", in_dim, out_dim);
        return {};
    }

    size_t                  ctx_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
    struct ggml_init_params params   = {
        /*.mem_size   = */ ctx_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, n_tokens);
    ggml_set_name(input, "input");
    ggml_set_input(input);

    struct ggml_cgraph * gf = projector_build_graph(model, ctx, input);

    ggml_backend_buffer_t buf_compute = ggml_backend_alloc_ctx_tensors(ctx, model.backend);
    if (!buf_compute) {
        LOG_ERR("Projector: failed to allocate compute buffer\n");
        ggml_free(ctx);
        return {};
    }

    ggml_backend_tensor_set(input, input_data, 0, n_tokens * in_dim * sizeof(float));

    enum ggml_status status = ggml_backend_graph_compute(model.backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("Projector: graph compute failed with status %d\n", (int) status);
        ggml_backend_buffer_free(buf_compute);
        ggml_free(ctx);
        return {};
    }

    struct ggml_tensor * output = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
    std::vector<float>   result(n_tokens * out_dim);
    ggml_backend_tensor_get(output, result.data(), 0, n_tokens * out_dim * sizeof(float));

    ggml_backend_buffer_free(buf_compute);
    ggml_free(ctx);

    return result;
}

// ==============================================================================

// Load TTS weights from GGUF file
bool load_tts_weights_from_gguf(struct omni_context * ctx_omni, const char * tts_model_path) {
    OmniTTSAuxWeights & tts_aux = ctx_omni->tts_aux;

    // Initialize GGUF context
    struct ggml_context *   ctx_meta = NULL;
    struct gguf_init_params params   = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_meta,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(tts_model_path, params);
    if (!ctx_gguf) {
        LOG_ERR("TTS: Failed to load GGUF file: %s\n", tts_model_path);
        return false;
    }

    // Load emb_code.0.weight: (num_audio_tokens=6562, hidden_size=768)
    // This is used to convert audio token IDs to embeddings during decode phase
    const char * emb_code_name = "emb_code.0.weight";
    int64_t      emb_code_idx  = gguf_find_tensor(ctx_gguf, emb_code_name);
    if (emb_code_idx >= 0) {
        struct ggml_tensor * emb_code_tensor = ggml_get_tensor(ctx_meta, emb_code_name);
        if (emb_code_tensor) {
            // emb_code is Embedding(num_audio_tokens, hidden_size)
            // In PyTorch: weight shape is (num_audio_tokens, hidden_size) = [6562, 768]
            // In GGUF: stored as (hidden_size, num_audio_tokens) = [768, 6562] (transposed)
            int64_t dim0 = emb_code_tensor->ne[0];
            int64_t dim1 = emb_code_tensor->ne[1];

            // Determine which dimension is which based on expected values
            int64_t num_audio_tokens = 6562;
            int64_t hidden_size      = 768;

            // GGUF stores as (hidden_size, num_audio_tokens) = [768, 6562]
            if (dim0 == hidden_size && dim1 == num_audio_tokens) {
                // Correct: stored as (hidden_size, num_audio_tokens)
            } else if (dim0 == num_audio_tokens && dim1 == hidden_size) {
                // Stored as (num_audio_tokens, hidden_size) - need to transpose
                num_audio_tokens = dim0;
                hidden_size      = dim1;
            } else {
                LOG_ERR("TTS: emb_code.0.weight has unexpected shape [%ld, %ld], expected [768, 6562] or [6562, 768]\n",
                        dim0, dim1);
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }

            // Allocate memory and copy data
            size_t emb_code_size    = dim0 * dim1 * sizeof(float);
            tts_aux.emb_code_weight = (float *) malloc(emb_code_size);
            if (!tts_aux.emb_code_weight) {
                LOG_ERR("TTS: Failed to allocate memory for emb_code.0.weight\n");
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }

            // Copy/convert tensor data based on type
            enum ggml_type emb_code_type     = emb_code_tensor->type;
            int64_t        emb_code_elements = dim0 * dim1;

            if (emb_code_type == GGML_TYPE_F32) {
                // F32: direct copy
                memcpy(tts_aux.emb_code_weight, emb_code_tensor->data, emb_code_size);
            } else if (emb_code_type == GGML_TYPE_F16) {
                // F16: convert to F32
                const ggml_fp16_t * src_f16 = (const ggml_fp16_t *) emb_code_tensor->data;
                for (int64_t i = 0; i < emb_code_elements; ++i) {
                    tts_aux.emb_code_weight[i] = ggml_fp16_to_fp32(src_f16[i]);
                }
            } else {
                LOG_ERR("TTS: emb_code.0.weight has unsupported type: %d\n", emb_code_type);
                free(tts_aux.emb_code_weight);
                tts_aux.emb_code_weight = nullptr;
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }

            tts_aux.emb_code_vocab_size           = num_audio_tokens;  // 6562
            tts_aux.emb_code_hidden_size          = hidden_size;       // 768
            // NOTE: GGUF data is stored in row-major order, matching NumPy/PyTorch
            // Even when metadata shape is [768, 6562], the actual data layout is (6562, 768) row-major
            // So we should access as: weight[token_idx * hidden_size + j]
            // This means: emb_code_stored_as_transposed should be FALSE
            tts_aux.emb_code_stored_as_transposed = false;  // Data is always (vocab_size, hidden_size) in memory
        } else {
            LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", emb_code_name);
            ggml_free(ctx_meta);
            gguf_free(ctx_gguf);
            return false;
        }
    } else {
        LOG_ERR("TTS: Tensor %s not found in GGUF file (this is OK if using token IDs)\n", emb_code_name);
        // Note: emb_code is optional if we use token IDs, but we prefer embeddings
    }

    // Load emb_text.weight: (vocab_size=152064, hidden_size=768)
    // PyTorch: nn.Embedding(vocab_size, hidden_size) -> weight shape is [vocab_size, hidden_size] = [152064, 768]
    // GGUF: may be stored as [hidden_size, vocab_size] = [768, 152064] (transposed)
    const char * emb_text_name = "emb_text.weight";
    int64_t      emb_text_idx  = gguf_find_tensor(ctx_gguf, emb_text_name);
    if (emb_text_idx >= 0) {
        struct ggml_tensor * emb_text_tensor = ggml_get_tensor(ctx_meta, emb_text_name);
        if (emb_text_tensor) {
            int64_t dim0 = emb_text_tensor->ne[0];
            int64_t dim1 = emb_text_tensor->ne[1];

            // Expected values
            int64_t expected_vocab_size  = 152064;
            int64_t expected_hidden_size = 768;

            // GGML tensor 维度理解：
            // ne[0] = 最内层维度 (stride=1) = hidden_size = 768
            // ne[1] = 外层维度 = vocab_size = 152064
            // 内存布局是 row-major，即 [vocab_size][hidden_size]
            // 所以 ne[0]=768, ne[1]=152064 意味着数据已经是 [vocab_size, hidden_size] 格式
            // 不需要转置！

            int64_t vocab_size;
            int64_t hidden_size;

            if (dim0 == expected_hidden_size && dim1 == expected_vocab_size) {
                // GGML shape: ne[0]=768, ne[1]=152064
                // 这意味着内存布局是 [vocab_size=152064][hidden_size=768]
                // 不需要转置
                vocab_size  = dim1;  // 152064
                hidden_size = dim0;  // 768
            } else if (dim0 == expected_vocab_size && dim1 == expected_hidden_size) {
                // GGML shape: ne[0]=152064, ne[1]=768 (unusual)
                // 这意味着内存布局是 [hidden_size=768][vocab_size=152064]
                // 这种情况需要转置
                vocab_size  = dim0;  // 152064
                hidden_size = dim1;  // 768
                LOG_ERR("TTS: emb_text.weight has unusual GGML shape, not handled\n");
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            } else {
                LOG_ERR("TTS: emb_text.weight has unexpected shape [%ld, %ld], expected ne=[%ld, %ld]\n", dim0, dim1,
                        expected_hidden_size, expected_vocab_size);
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }

            // Allocate memory for the weight
            size_t emb_text_size    = vocab_size * hidden_size * sizeof(float);
            tts_aux.emb_text_weight = (float *) malloc(emb_text_size);
            if (!tts_aux.emb_text_weight) {
                LOG_ERR("TTS: Failed to allocate memory for emb_text.weight\n");
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }

            // Copy/convert tensor data based on type
            enum ggml_type emb_text_type     = emb_text_tensor->type;
            int64_t        emb_text_elements = vocab_size * hidden_size;

            if (emb_text_type == GGML_TYPE_F32) {
                // F32: direct copy
                memcpy(tts_aux.emb_text_weight, emb_text_tensor->data, emb_text_size);
            } else if (emb_text_type == GGML_TYPE_F16) {
                // F16: convert to F32
                const ggml_fp16_t * src_f16 = (const ggml_fp16_t *) emb_text_tensor->data;
                for (int64_t i = 0; i < emb_text_elements; ++i) {
                    tts_aux.emb_text_weight[i] = ggml_fp16_to_fp32(src_f16[i]);
                }
            } else {
                LOG_ERR("TTS: emb_text.weight has unsupported type: %d\n", emb_text_type);
                free(tts_aux.emb_text_weight);
                tts_aux.emb_text_weight = nullptr;
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }

            tts_aux.emb_text_vocab_size  = vocab_size;   // 152064
            tts_aux.emb_text_hidden_size = hidden_size;  // 768
        } else {
            LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", emb_text_name);
            ggml_free(ctx_meta);
            gguf_free(ctx_gguf);
            return false;
        }
    } else {
        LOG_ERR("TTS: Tensor %s not found in GGUF file\n", emb_text_name);
        ggml_free(ctx_meta);
        gguf_free(ctx_gguf);
        return false;
    }

    // Load projector_semantic weights
    const char * projector_names[] = { "projector_semantic.linear1.weight", "projector_semantic.linear1.bias",
                                       "projector_semantic.linear2.weight", "projector_semantic.linear2.bias" };

    float ** projector_ptrs[] = { &tts_aux.projector_semantic_linear1_weight, &tts_aux.projector_semantic_linear1_bias,
                                  &tts_aux.projector_semantic_linear2_weight,
                                  &tts_aux.projector_semantic_linear2_bias };

    // PyTorch nn.Linear(in_features, out_features) weight shape is (out_features, in_features)
    // GGUF may store as (out_features, in_features) or (in_features, out_features)
    // We need to detect and handle both cases
    int64_t expected_shapes_pytorch[][2] = {
        { 768, 4096 }, // linear1.weight: PyTorch shape (out_features, in_features)
        { 768, 0    }, // linear1.bias (1D)
        { 768, 768  }, // linear2.weight: PyTorch shape (out_features, in_features)
        { 768, 0    }  // linear2.bias (1D)
    };

    int64_t expected_shapes_transposed[][2] = {
        { 4096, 768 }, // linear1.weight: transposed shape (in_features, out_features)
        { 768,  0   }, // linear1.bias (1D)
        { 768,  768 }, // linear2.weight: same for square matrix
        { 768,  0   }  // linear2.bias (1D)
    };

    // Track whether weights need transposition
    bool need_transpose[2] = { false, false };  // [linear1, linear2]

    for (int i = 0; i < 4; i++) {
        int64_t tensor_idx = gguf_find_tensor(ctx_gguf, projector_names[i]);
        if (tensor_idx >= 0) {
            struct ggml_tensor * tensor = ggml_get_tensor(ctx_meta, projector_names[i]);
            if (tensor) {
                int64_t dim0 = tensor->ne[0];
                int64_t dim1 = (ggml_n_dims(tensor) > 1) ? tensor->ne[1] : 0;

                if (i % 2 == 0) {  // weight (2D)
                    // Check if stored as PyTorch shape (out_features, in_features) or transposed
                    bool is_pytorch_shape =
                        (dim0 == expected_shapes_pytorch[i][0] && dim1 == expected_shapes_pytorch[i][1]);
                    bool is_transposed_shape =
                        (dim0 == expected_shapes_transposed[i][0] && dim1 == expected_shapes_transposed[i][1]);

                    if (is_pytorch_shape) {
                        // Stored as PyTorch shape (out_features, in_features), need to transpose
                        need_transpose[i / 2] = true;
                    } else if (is_transposed_shape) {
                        // Already transposed, use directly
                        need_transpose[i / 2] = false;
                    } else {
                        LOG_ERR("TTS: %s has unexpected shape: [%ld, %ld], expected [%ld, %ld] or [%ld, %ld]\n",
                                projector_names[i], dim0, dim1, expected_shapes_pytorch[i][0],
                                expected_shapes_pytorch[i][1], expected_shapes_transposed[i][0],
                                expected_shapes_transposed[i][1]);
                        // Try to continue, assume PyTorch shape
                        need_transpose[i / 2] = true;
                    }
                } else {  // bias (1D)
                    if (dim0 != expected_shapes_pytorch[i][0] || dim1 != 0) {
                        LOG_ERR("TTS: %s has wrong shape: [%ld, %ld], expected [%ld, 0]\n", projector_names[i], dim0,
                                dim1, expected_shapes_pytorch[i][0]);
                    }
                }

                // Check tensor type for F16 conversion
                enum ggml_type proj_tensor_type = tensor->type;

                if (i % 2 == 0 && need_transpose[i / 2]) {
                    // Weight needs transposition: allocate transposed size (always F32 output)
                    int64_t in_dim          = expected_shapes_pytorch[i][1];  // PyTorch in_features
                    int64_t out_dim         = expected_shapes_pytorch[i][0];  // PyTorch out_features
                    size_t  transposed_size = in_dim * out_dim * sizeof(float);
                    *projector_ptrs[i]      = (float *) malloc(transposed_size);
                    if (!*projector_ptrs[i]) {
                        LOG_ERR("TTS: Failed to allocate memory for transposed %s\n", projector_names[i]);
                        // Clean up
                        for (int j = 0; j < i; j++) {
                            if (*projector_ptrs[j]) {
                                free(*projector_ptrs[j]);
                                *projector_ptrs[j] = nullptr;
                            }
                        }
                        ggml_free(ctx_meta);
                        gguf_free(ctx_gguf);
                        return false;
                    }

                    // Transpose: src[out_dim][in_dim] -> dst[in_dim][out_dim], handling F16 if needed
                    float * dst_data = *projector_ptrs[i];
                    if (proj_tensor_type == GGML_TYPE_F32) {
                        const float * src_data = (const float *) tensor->data;
                        for (int64_t out = 0; out < out_dim; out++) {
                            for (int64_t in = 0; in < in_dim; in++) {
                                dst_data[in * out_dim + out] = src_data[out * in_dim + in];
                            }
                        }
                    } else if (proj_tensor_type == GGML_TYPE_F16) {
                        const ggml_fp16_t * src_data = (const ggml_fp16_t *) tensor->data;
                        for (int64_t out = 0; out < out_dim; out++) {
                            for (int64_t in = 0; in < in_dim; in++) {
                                dst_data[in * out_dim + out] = ggml_fp16_to_fp32(src_data[out * in_dim + in]);
                            }
                        }
                    } else {
                        LOG_ERR("TTS: %s has unsupported type: %d\n", projector_names[i], proj_tensor_type);
                        free(*projector_ptrs[i]);
                        *projector_ptrs[i] = nullptr;
                        ggml_free(ctx_meta);
                        gguf_free(ctx_gguf);
                        return false;
                    }
                } else {
                    // Direct copy (bias or already transposed weight), handle F16
                    int64_t num_elements = (dim1 > 0) ? dim0 * dim1 : dim0;
                    size_t  output_size  = num_elements * sizeof(float);
                    *projector_ptrs[i]   = (float *) malloc(output_size);
                    if (!*projector_ptrs[i]) {
                        LOG_ERR("TTS: Failed to allocate memory for %s\n", projector_names[i]);
                        // Clean up
                        for (int j = 0; j < i; j++) {
                            if (*projector_ptrs[j]) {
                                free(*projector_ptrs[j]);
                                *projector_ptrs[j] = nullptr;
                            }
                        }
                        ggml_free(ctx_meta);
                        gguf_free(ctx_gguf);
                        return false;
                    }

                    if (proj_tensor_type == GGML_TYPE_F32) {
                        memcpy(*projector_ptrs[i], tensor->data, output_size);
                    } else if (proj_tensor_type == GGML_TYPE_F16) {
                        const ggml_fp16_t * src_f16 = (const ggml_fp16_t *) tensor->data;
                        for (int64_t k = 0; k < num_elements; ++k) {
                            (*projector_ptrs[i])[k] = ggml_fp16_to_fp32(src_f16[k]);
                        }
                    } else {
                        LOG_ERR("TTS: %s has unsupported type: %d\n", projector_names[i], proj_tensor_type);
                        free(*projector_ptrs[i]);
                        *projector_ptrs[i] = nullptr;
                        ggml_free(ctx_meta);
                        gguf_free(ctx_gguf);
                        return false;
                    }
                    if (dim1 > 0) {
                    }
                }
            } else {
                LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", projector_names[i]);
                // Clean up
                for (int j = 0; j < i; j++) {
                    if (*projector_ptrs[j]) {
                        free(*projector_ptrs[j]);
                        *projector_ptrs[j] = nullptr;
                    }
                }
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
        } else {
            LOG_ERR("TTS: Tensor %s not found in GGUF file\n", projector_names[i]);
            // Clean up
            for (int j = 0; j < i; j++) {
                if (*projector_ptrs[j]) {
                    free(*projector_ptrs[j]);
                    *projector_ptrs[j] = nullptr;
                }
            }
            ggml_free(ctx_meta);
            gguf_free(ctx_gguf);
            return false;
        }
    }

    // Set projector dimensions
    tts_aux.projector_semantic_input_dim  = 4096;
    tts_aux.projector_semantic_output_dim = 768;

    // Load head_code.weight: (hidden_size=768, num_audio_tokens=6562)
    // Note: num_vq=1, so we only load head_code.0.weight
    const char * head_code_name = "head_code.0.weight";
    int64_t      head_code_idx  = gguf_find_tensor(ctx_gguf, head_code_name);
    if (head_code_idx >= 0) {
        struct ggml_tensor * head_code_tensor = ggml_get_tensor(ctx_meta, head_code_name);
        if (head_code_tensor) {
            // head_code is Linear(hidden_size, num_audio_tokens, bias=False)
            // In PyTorch: weight shape is (num_audio_tokens, hidden_size) = [6562, 768]
            // In GGUF: stored as (hidden_size, num_audio_tokens) = [768, 6562] (already transposed)
            int64_t dim0 = head_code_tensor->ne[0];
            int64_t dim1 = (ggml_n_dims(head_code_tensor) > 1) ? head_code_tensor->ne[1] : 0;

            // Expected shape in GGUF: (hidden_size=768, num_audio_tokens=6562)
            int64_t expected_hidden_size      = 768;
            int64_t expected_num_audio_tokens = 6562;

            // Allocate memory for weight: (hidden_size, num_audio_tokens) = [768, 6562]
            size_t head_code_size    = expected_hidden_size * expected_num_audio_tokens * sizeof(float);
            tts_aux.head_code_weight = (float *) malloc(head_code_size);
            if (!tts_aux.head_code_weight) {
                LOG_ERR("TTS: Failed to allocate memory for head_code.0.weight\n");
                // Clean up already loaded weights
                if (tts_aux.emb_text_weight) {
                    free(tts_aux.emb_text_weight);
                    tts_aux.emb_text_weight = nullptr;
                }
                for (int j = 0; j < 4; j++) {
                    if (*projector_ptrs[j]) {
                        free(*projector_ptrs[j]);
                        *projector_ptrs[j] = nullptr;
                    }
                }
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }

            // CRITICAL FIX: The conversion script transposes head_code weight to [768, 6562] before saving,
            // but the GGUF metadata shape may still be [6562, 768] due to how add_tensor works.
            // We need to detect the actual data format by testing both layouts.
            //
            // Strategy: Try both formats and see which one produces correct logits.
            // But a simpler approach: Since the conversion script always transposes to [768, 6562],
            // and the actual data is stored in that format, we should always use the data as-is
            // (treating it as [768, 6562]) regardless of metadata shape.
            //
            // However, to be safe, we'll check: if metadata says [6562, 768], the data might be
            // in that format (old conversion) or already transposed (new conversion).
            // We'll use a heuristic: check if the first few values match what we expect.

            const float * src_data = (const float *) head_code_tensor->data;

            // CRITICAL FIX: Based on conversion script analysis, the script always transposes
            // head_code to [768, 6562] before saving. So the actual data is always [768, 6562],
            // regardless of metadata shape. We should NOT transpose based on metadata.
            //
            // However, if metadata says [6562, 768], it might be an old conversion that didn't transpose.
            // We'll use a simple heuristic: if metadata says [6562, 768], assume data needs transpose.
            // If metadata says [768, 6562], assume data is already correct.

            if (!((dim0 == expected_hidden_size && dim1 == expected_num_audio_tokens) ||
                  (dim0 == expected_num_audio_tokens && dim1 == expected_hidden_size))) {
                LOG_ERR("TTS: head_code.0.weight has unexpected shape [%ld, %ld], expected [%d, %d] or [%d, %d]\n",
                        dim0, dim1, expected_hidden_size, expected_num_audio_tokens, expected_num_audio_tokens,
                        expected_hidden_size);
                // Clean up
                free(tts_aux.head_code_weight);
                tts_aux.head_code_weight = nullptr;
                if (tts_aux.emb_text_weight) {
                    free(tts_aux.emb_text_weight);
                    tts_aux.emb_text_weight = nullptr;
                }
                for (int j = 0; j < 4; j++) {
                    if (*projector_ptrs[j]) {
                        free(*projector_ptrs[j]);
                        *projector_ptrs[j] = nullptr;
                    }
                }
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }

            // Check tensor type and copy/convert accordingly
            // ⚡ 优化：转置存储为 [6562, 768]，使每个output token的权重连续存储
            // 这样在计算logits时可以用高效的向量点积
            // 原始: weight[j * 6562 + i] = W[j, i]  (j=hidden, i=output)
            // 转置后: weight[i * 768 + j] = W[i, j] (i=output, j=hidden)
            enum ggml_type tensor_type = head_code_tensor->type;
            print_with_timestamp("TTS: head_code shape: dim0=%ld, dim1=%ld, will transpose to [%ld, %ld]\n", dim0, dim1,
                                 expected_num_audio_tokens, expected_hidden_size);

            if (tensor_type == GGML_TYPE_F32) {
                // F32: copy with transpose from [768, 6562] to [6562, 768]
                for (int64_t i = 0; i < expected_num_audio_tokens; ++i) {
                    for (int64_t j = 0; j < expected_hidden_size; ++j) {
                        // src: [j * 6562 + i], dst: [i * 768 + j]
                        tts_aux.head_code_weight[i * expected_hidden_size + j] =
                            src_data[j * expected_num_audio_tokens + i];
                    }
                }
            } else if (tensor_type == GGML_TYPE_F16) {
                // F16: convert to F32 with transpose
                const ggml_fp16_t * src_f16 = (const ggml_fp16_t *) src_data;
                for (int64_t i = 0; i < expected_num_audio_tokens; ++i) {
                    for (int64_t j = 0; j < expected_hidden_size; ++j) {
                        tts_aux.head_code_weight[i * expected_hidden_size + j] =
                            ggml_fp16_to_fp32(src_f16[j * expected_num_audio_tokens + i]);
                    }
                }
            } else {
                LOG_ERR("TTS: head_code.0.weight has unsupported type: %d\n", tensor_type);
                free(tts_aux.head_code_weight);
                tts_aux.head_code_weight = nullptr;
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }

            tts_aux.head_code_hidden_size      = expected_hidden_size;
            tts_aux.head_code_num_audio_tokens = expected_num_audio_tokens;

            // 🔍 调试：验证加载的数据
            // Python weight[0, 0:5] = [0.01385498, -0.01647949, 0.0111084, -0.01367188, -0.01141357]
            // C++ head_code_weight[0..4] 应该和 Python 一致
            print_with_timestamp("TTS: head_code loaded, verifying first few values:\n");
            print_with_timestamp("  head_code_weight[0] = %.8f (expect ~0.01385498)\n", tts_aux.head_code_weight[0]);
            print_with_timestamp("  head_code_weight[1] = %.8f (expect ~-0.01647949)\n", tts_aux.head_code_weight[1]);
            print_with_timestamp("  head_code_weight[768] = %.8f (this is weight[1, 0], expect ~0.04150391)\n",
                                 tts_aux.head_code_weight[768]);
        } else {
            LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", head_code_name);
            // Clean up
            if (tts_aux.emb_text_weight) {
                free(tts_aux.emb_text_weight);
                tts_aux.emb_text_weight = nullptr;
            }
            for (int j = 0; j < 4; j++) {
                if (*projector_ptrs[j]) {
                    free(*projector_ptrs[j]);
                    *projector_ptrs[j] = nullptr;
                }
            }
            ggml_free(ctx_meta);
            gguf_free(ctx_gguf);
            return false;
        }
    } else {
        LOG_ERR("TTS: Tensor %s not found in GGUF file\n", head_code_name);
        // Clean up
        if (tts_aux.emb_text_weight) {
            free(tts_aux.emb_text_weight);
            tts_aux.emb_text_weight = nullptr;
        }
        for (int j = 0; j < 4; j++) {
            if (*projector_ptrs[j]) {
                free(*projector_ptrs[j]);
                *projector_ptrs[j] = nullptr;
            }
        }
        ggml_free(ctx_meta);
        gguf_free(ctx_gguf);
        return false;
    }

    ggml_free(ctx_meta);
    gguf_free(ctx_gguf);
    return true;
}

static bool eval_tokens_tts(struct omni_context *    ctx_omni,
                            common_params *          params,
                            std::vector<llama_token> tokens,
                            int                      n_batch,
                            int *                    n_past_tts) {
    (void) params;
    fflush(stdout);
    int N = (int) tokens.size();
    fflush(stdout);
    // Note: TTS model might need different KV cache management
    // For now, we'll use a simple approach similar to LLM
    fflush(stdout);
    fflush(stdout);
    for (int i = 0; i < N; i += n_batch) {
        fflush(stdout);
        int n_eval = std::min((int) tokens.size() - i, n_batch);
        fflush(stdout);
        if (n_eval == 0) {
            fflush(stdout);
            break;
        }
        fflush(stdout);

        // Use llama_batch_get_one and manually set pos
        // Note: llama_batch_get_one may return batch with nullptr pos, so we need to handle it
        llama_batch batch = llama_batch_get_one(&tokens[i], n_eval);
        fflush(stdout);

        // If batch.pos is nullptr, we need to allocate it
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            fflush(stdout);
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }

        // Set pos values to ensure correct KV cache position
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = *n_past_tts + j;
        }
        fflush(stdout);

        // Enable embeddings output for TTS model (needed for head_code logits calculation)
        llama_set_embeddings(ctx_omni->ctx_tts_llama, true);
        fflush(stdout);
        int decode_ret = llama_decode(ctx_omni->ctx_tts_llama, batch);

        // Keep embeddings enabled for sample_tts_token to use

        if (decode_ret != 0) {
            LOG_ERR("%s : failed to eval TTS tokens. token %d/%d (batch size %d, n_past %d), decode_ret=%d\n", __func__,
                    i, N, n_batch, *n_past_tts, decode_ret);
            return false;
        }
        *n_past_tts += n_eval;
    }
    return true;
}

static bool eval_string_tts(struct omni_context * ctx_omni,
                            common_params *       params,
                            const char *          str,
                            int                   n_batch,
                            int *                 n_past_tts,
                            bool                  add_bos) {
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_omni->ctx_tts_llama, str2, add_bos, true);
    return eval_tokens_tts(ctx_omni, params, embd_inp, n_batch, n_past_tts);
}

// 使用embedding作为TTS输入的prefill函数（类似prefill_with_emb，但针对TTS模型）
bool prefill_with_emb_tts(struct omni_context * ctx_omni,
                          common_params *       params,
                          float *               embed,
                          int                   n_pos,
                          int                   n_batch,
                          int *                 n_past_tts) {
    (void) params;
    // 🔧 [安全检查] 验证输入参数
    if (n_pos <= 0) {
        LOG_ERR("%s: invalid n_pos=%d, skipping\n", __func__, n_pos);
        return false;
    }
    if (n_pos > 10000) {
        LOG_ERR("%s: n_pos=%d seems too large, likely data corruption\n", __func__, n_pos);
        return false;
    }
    if (!ctx_omni->ctx_tts_llama || !ctx_omni->model_tts) {
        LOG_ERR("%s: TTS model not loaded\n", __func__);
        return false;
    }

    int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));

    // 🔧 [安全检查] 验证 n_embd 是合理值
    if (n_embd <= 0 || n_embd > 10000) {
        LOG_ERR("%s: invalid n_embd=%d from TTS model, likely model corruption\n", __func__, n_embd);
        return false;
    }

    // 🔧 [安全检查] 检查乘法溢出
    if (n_pos > (INT_MAX / n_embd)) {
        LOG_ERR("%s: n_pos=%d * n_embd=%d would overflow\n", __func__, n_pos, n_embd);
        return false;
    }

    // Save condition embeddings for first audio token re-forward (if not already saved)
    // This is needed to match Python's behavior: first audio token re-forwards the condition
    if (!ctx_omni->tts_condition_saved && n_pos > 0) {
        ctx_omni->tts_condition_embeddings.resize(n_pos * n_embd);
        std::memcpy(ctx_omni->tts_condition_embeddings.data(), embed, n_pos * n_embd * sizeof(float));
        ctx_omni->tts_condition_length = n_pos;
        ctx_omni->tts_condition_n_embd = n_embd;
        ctx_omni->tts_condition_saved  = true;
    }

    // Save the starting position before the loop
    int text_start_pos = *n_past_tts;

    // Check if we need to save all hidden states (for alignment testing)
    const char * save_hidden_states_dir = getenv("TTS_SAVE_HIDDEN_STATES_DIR");
    bool         save_all_hidden_states = (save_hidden_states_dir != nullptr);

    for (int i = 0; i < n_pos; i += n_batch) {
        int n_eval = std::min(n_pos - i, n_batch);

        llama_batch batch = {};
        batch.n_tokens    = int32_t(n_eval);
        batch.embd        = (embed + i * n_embd);  // 使用embedding作为输入

        // 设置pos值以确保正确的KV cache位置
        // Python: pos_ids = torch.arange(text_start_pos, text_start_pos + condition_length)
        // C++: batch.pos[j] = text_start_pos + i + j (where i is the offset within the current batch)
        std::vector<llama_pos> pos_vec(n_eval);
        batch.pos = pos_vec.data();
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = text_start_pos + i + j;  // Fix: use text_start_pos + i + j instead of *n_past_tts + j
        }

        // Enable embeddings output for TTS model (needed for head_code logits calculation)
        llama_set_embeddings(ctx_omni->ctx_tts_llama, true);

        if (llama_decode(ctx_omni->ctx_tts_llama, batch)) {
            LOG_ERR("%s : failed to eval TTS embeddings. pos %d/%d (batch size %d, n_past %d)\n", __func__, i, n_pos,
                    n_batch, *n_past_tts);
            llama_set_embeddings(ctx_omni->ctx_tts_llama, false);
            return false;
        }

        // Save hidden states for each token in the batch (for alignment testing)
        // Note: llama_get_embeddings_ith uses negative indices relative to the end of the batch
        // For a batch of n_eval tokens: -1 is last, -2 is second-to-last, ..., -n_eval is first
        if (save_all_hidden_states) {
            for (int j = 0; j < n_eval; j++) {
                int           token_idx    = text_start_pos + i + j;
                // Get j-th token in current batch: j=0 -> -n_eval (first), j=n_eval-1 -> -1 (last)
                int           llama_idx    = j - n_eval;
                const float * hidden_state = llama_get_embeddings_ith(ctx_omni->ctx_tts_llama, llama_idx);
                if (hidden_state) {
                    char filepath[512];
                    snprintf(filepath, sizeof(filepath), "%s/hidden_states_%03d.bin", save_hidden_states_dir,
                             token_idx);
                    FILE * f = fopen(filepath, "wb");
                    if (f) {
                        fwrite(&token_idx, sizeof(int32_t), 1, f);
                        fwrite(&n_embd, sizeof(int32_t), 1, f);
                        fwrite(hidden_state, sizeof(float), n_embd, f);
                        fclose(f);
                    }
                } else {
                    LOG_WRN("TTS: Failed to get hidden state for token %d (llama_idx=%d)\n", token_idx, llama_idx);
                }
            }
        }

        // Keep embeddings enabled for sample_tts_token to use
    }

    // Update n_past_tts after all tokens are processed
    *n_past_tts = text_start_pos + n_pos;

    return true;
}

// Check if a token is an audio token (based on config: num_audio_tokens = 6562)
// Audio tokens are typically in a specific range
static bool is_audio_token(llama_token token, int audio_bos_token_id = 151687, int num_audio_tokens = 6562) {
    // Audio tokens are typically in range [audio_bos_token_id, audio_bos_token_id + num_audio_tokens)
    // Check if token is in the audio token range
    return (token >= audio_bos_token_id && token < audio_bos_token_id + num_audio_tokens);
}

struct omni_context * omni_init(struct common_params * params,
                                int                    media_type,
                                bool                   use_tts,
                                const std::string &    tts_bin_dir,
                                int                    tts_gpu_layers,
                                const std::string &    token2wav_device,
                                bool                   duplex_mode,
                                llama_model *          existing_model,
                                llama_context *        existing_ctx,
                                const std::string &    base_output_dir) {
    print_with_timestamp("=== omni_init start\n");
    auto * ctx_omni = new omni_context();

    // ── 1. Basic context config ───────────────────────────────────────────
    ctx_omni->params      = params;
    ctx_omni->media_type  = media_type;
    ctx_omni->use_tts     = use_tts;
    ctx_omni->duplex_mode = duplex_mode;
    omni_session_sync_round_meta(ctx_omni);
    ctx_omni->base_output_dir = base_output_dir;
    print_with_timestamp("media_type = %d, duplex_mode = %d, base_output_dir = %s\n", media_type, duplex_mode,
                         base_output_dir.c_str());

    if (!base_output_dir.empty()) {
        omni_archive_output_dir(base_output_dir, base_output_dir + ".old");
    }

    omni_init_prompt_templates(ctx_omni, duplex_mode);

    // ── 2. LLM runtime ───────────────────────────────────────────────────
    if (!omni_init_llm_runtime(ctx_omni, params, existing_model, existing_ctx)) {
        delete ctx_omni;
        return NULL;
    }

    // ── 3. TTS runtime ───────────────────────────────────────────────────
    if (use_tts && !params->tts_model.empty()) {
        if (!omni_init_tts_runtime(ctx_omni, params, tts_bin_dir, tts_gpu_layers)) {
            omni_release_llm_runtime(ctx_omni);
            delete ctx_omni;
            return NULL;
        }
    }

    // ── 4. Audio/Vision runtime ───────────────────────────────────────────
    ctx_omni->omni_emb.resize((64 + 10 + 1) * 4096);  // temp fix for omni embed
    ctx_omni->audio_emb.resize((10 + 1) * 4096);      // temp fix for audio embed
    if (!omni_init_audio_vision_runtime(ctx_omni, params)) {
        if (ctx_omni->use_tts) {
            omni_release_tts_runtime(ctx_omni);
        }
        omni_release_llm_runtime(ctx_omni);
        delete ctx_omni;
        return NULL;
    }

    // ── 5. Worker thread infos + Token2Wav ───────────────────────────────
    ctx_omni->llm_thread_info = new LLMThreadInfo(1000);
    if (ctx_omni->use_tts) {
        LOG_INF("init tts....");
        ctx_omni->tts_thread_info = new TTSThreadInfo(1);
        ctx_omni->omni_output     = new omni_output();
        ctx_omni->tts_bin_dir     = tts_bin_dir;
        LOG_INF("init t2w....");
        ctx_omni->t2w_thread_info = new T2WThreadInfo(25);

        omni_init_token2wav_runtime(ctx_omni, tts_bin_dir, token2wav_device);
    }

    // ── 6. Finalize ───────────────────────────────────────────────────────
    ctx_omni->async = true;
    omni_init_token_protocol(ctx_omni);
    omni_warmup_ane(ctx_omni);

    print_with_timestamp("=== omni_init success: ctx_llama = %p\n", (void *) ctx_omni->ctx_llama);
    return ctx_omni;
}

//
// ANE/CoreML warmup — pre-load models into NPU to avoid first-inference latency
//
void omni_warmup_ane(struct omni_context * ctx_omni) {
#if defined(__APPLE__)
    if (!ctx_omni) {
        return;
    }

    LOG_INF("%s: starting ANE/CoreML warmup...\n", __func__);

    // 1. Vision ANE warmup
    if (ctx_omni->ctx_vision) {
        vision_coreml_warmup(ctx_omni->ctx_vision);
    }

    // 2. Future: audio ANE warmup
    // if (ctx_omni->ctx_audio) {
    //     audition_coreml_warmup(ctx_omni->ctx_audio);
    // }

    // 3. Future: other module ANE warmup
    // ...

    LOG_INF("%s: ANE/CoreML warmup finished\n", __func__);
#else
    (void) ctx_omni;
#endif
}

bool omni_tts_queues_empty(struct omni_context * ctx_omni) {
    bool tts_empty = true;
    bool t2w_empty = true;
    if (ctx_omni->tts_thread_info) {
        std::lock_guard<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
        tts_empty = ctx_omni->tts_thread_info->queue.empty();
    }
    if (ctx_omni->t2w_thread_info) {
        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
        t2w_empty = ctx_omni->t2w_thread_info->queue.empty();
    }
    return tts_empty && t2w_empty;
}

bool omni_get_duplex_chunk_timing(struct omni_context *             ctx_omni,
                                  int                               chunk_idx,
                                  struct omni_duplex_chunk_timing * out_timing) {
    if (ctx_omni == nullptr || out_timing == nullptr || chunk_idx < 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto                        it = ctx_omni->duplex_chunk_timings.find(chunk_idx);
    if (it == ctx_omni->duplex_chunk_timings.end()) {
        return false;
    }
    *out_timing = it->second;
    return true;
}

// 停止所有线程（发送信号，不等待）
void omni_stop_threads(struct omni_context * ctx_omni) {
    omni_request_worker_shutdown(ctx_omni);
    print_with_timestamp("omni_stop_threads: stop signals sent\n");
}

void omni_free(struct omni_context * ctx_omni) {
    omni_shutdown_worker_threads(ctx_omni);
    omni_release_audio_vision_runtime(ctx_omni);
    if (ctx_omni->use_tts) {
        omni_release_tts_runtime(ctx_omni);
    }
    omni_release_llm_runtime(ctx_omni);
    omni_release_thread_info(ctx_omni);

    // omni_output cleanup: struct definition is local to omni.cpp
    if (ctx_omni->use_tts && ctx_omni->omni_output) {
        for (auto & buffer : ctx_omni->omni_output->output) {
            delete buffer;
        }
        ctx_omni->omni_output->output.clear();
        delete ctx_omni->omni_output;
    }

    llama_backend_free();
    delete ctx_omni;
}

// ==================== 语言设置函数 ====================
// 设置语言并更新 system prompt（zh=中文，en=英文）
// 基于 Python MiniCPM-o-4_5 modeling_minicpmo.py 中的 audio_assistant 模式 prompt
static void omni_set_language(struct omni_context * ctx_omni, const std::string & lang) {
    if (ctx_omni == nullptr) {
        print_with_timestamp("omni_set_language: ctx_omni is null\n");
        return;
    }

    ctx_omni->language = lang;
    print_with_timestamp("omni_set_language: setting language to '%s'\n", lang.c_str());

    if (ctx_omni->duplex_mode) {
        // 双工模式：prompt 固定使用英文（与 Python 对齐）
        ctx_omni->audio_voice_clone_prompt =
            "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|><|im_end|>\n";
        ctx_omni->omni_voice_clone_prompt =
            "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|><|im_end|>\n";
    } else {
        // 非双工模式（audio_assistant 模式）：根据语言设置 prompt
        if (lang == "en") {
            // 英文 prompt（来自 Python modeling_minicpmo.py）
            ctx_omni->audio_voice_clone_prompt =
                "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>";
            ctx_omni->audio_assistant_prompt =
                "<|audio_end|>Please assist users while maintaining this voice style. Please answer the user's "
                "questions seriously and in a high quality. Please chat with the user in a highly human-like and oral "
                "style. You are a helpful assistant developed by ModelBest: "
                "MiniCPM-Omni.<|im_end|>\n<|im_start|>user\n";

            ctx_omni->omni_voice_clone_prompt =
                "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>";
            ctx_omni->omni_assistant_prompt =
                "<|audio_end|>Please assist users while maintaining this voice style. Please answer the user's "
                "questions seriously and in a high quality. Please chat with the user in a highly human-like and oral "
                "style.<|im_end|>\n<|im_start|>user\n";
        } else {
            // 中文 prompt（默认，来自 Python modeling_minicpmo.py）
            ctx_omni->audio_voice_clone_prompt =
                "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
            ctx_omni->audio_assistant_prompt =
                "<|audio_end|>"
                "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你"
                "是由面壁智能开发的人工智能助手：面壁小钢炮。<|im_end|>\n<|im_start|>user\n";

            ctx_omni->omni_voice_clone_prompt =
                "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
            ctx_omni->omni_assistant_prompt =
                "<|audio_end|>"
                "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。<|"
                "im_end|>\n<|im_start|>user\n";
        }
    }

    // 🔧 [关键] 重置 system_prompt_initialized，让下次 stream_prefill(index=0) 重新 prefill system prompt
    ctx_omni->session.prompt.system_prompt_initialized = false;

    print_with_timestamp(
        "omni_set_language: prompts updated for language '%s', system_prompt_initialized reset to false\n",
        lang.c_str());
}

static void process_audio(struct omni_context * ctx_omni,
                          struct omni_embed *   embeds,
                          common_params *       params,
                          bool                  save_spk_emb = false) {
    (void) save_spk_emb;
    LOG_INF("%s: audio token past: %d\n", __func__, ctx_omni->session.n_past);
    omni_eval_embed(ctx_omni->ctx_llama, embeds, params->n_batch, &ctx_omni->session.n_past);
    LOG_INF("%s: audio token past after eval: %d\n", __func__, ctx_omni->session.n_past);
}

static void eval_prefix(struct omni_context * ctx_omni, common_params * params) {
    std::string prefix = "<|im_start|>user\n";
    std::cout << "prefix : " << prefix << '\n';
    omni_llm_stage_eval_string(ctx_omni, params, prefix.c_str(), params->n_batch, &ctx_omni->session.n_past, false);
}

static void eval_prefix_with_hidden(struct omni_context * ctx_omni, common_params * params, float *& hidden_states) {
    std::string prefix = "<|im_start|>user\n";
    std::cout << "prefix : " << prefix << '\n';
    omni_llm_stage_eval_string_with_hidden(ctx_omni, params, prefix.c_str(), params->n_batch, &ctx_omni->session.n_past,
                                           false, hidden_states);
}

namespace {
struct OmniBootstrapPrompts {
    std::string voice_clone_prompt;
    std::string assistant_prompt;
};
}  // namespace

static std::string omni_normalize_prefill_prompt(const std::string & prompt) {
    if (prompt.rfind("<|", 0) == 0) {
        return prompt;
    }
    return "<|im_start|>user\n" + prompt;
}

static OmniBootstrapPrompts omni_select_bootstrap_prompts(const struct omni_context * ctx_omni) {
    const bool          use_omni_prompt = ctx_omni->media_type == 2;
    const std::string & raw_voice_clone_prompt =
        use_omni_prompt ? ctx_omni->omni_voice_clone_prompt : ctx_omni->audio_voice_clone_prompt;
    const std::string & raw_assistant_prompt =
        use_omni_prompt ? ctx_omni->omni_assistant_prompt : ctx_omni->audio_assistant_prompt;

    return {
        omni_normalize_prefill_prompt(raw_voice_clone_prompt),
        omni_normalize_prefill_prompt(raw_assistant_prompt),
    };
}

static std::string omni_get_bootstrap_ref_audio_path(const struct omni_context * ctx_omni,
                                                     const std::string &         aud_fname) {
    if (ctx_omni->duplex_mode && !aud_fname.empty()) {
        return aud_fname;
    }
    return ctx_omni->ref_audio_path.empty() ? "tools/omni/assets/default_ref_audio/default_ref_audio.wav" :
                                              ctx_omni->ref_audio_path;
}

// Step B: bootstrap the session once, including system prompt and ref-audio prefill.
static bool omni_run_session_bootstrap_if_needed(struct omni_context *        ctx_omni,
                                                 const OmniPrefillSetup &     setup,
                                                 const OmniBootstrapPrompts & prompts,
                                                 const std::string &          aud_fname,
                                                 int                          index) {
    if (!setup.need_bootstrap) {
        return true;
    }

    print_with_timestamp("stream_prefill: n_past = %d\n voice_clone_prompt = %s\n assistant_prompt = %s\n",
                         ctx_omni->session.n_past, prompts.voice_clone_prompt.c_str(),
                         prompts.assistant_prompt.c_str());

    const std::string bootstrap_ref_audio = omni_get_bootstrap_ref_audio_path(ctx_omni, aud_fname);
    if (!ctx_omni->duplex_mode || bootstrap_ref_audio != aud_fname) {
        print_with_timestamp("system prompt ref_audio: %s\n", bootstrap_ref_audio.c_str());
    }

    omni_llm_stage_eval_string(ctx_omni, ctx_omni->params, prompts.voice_clone_prompt.c_str(),
                               ctx_omni->params->n_batch, &ctx_omni->session.n_past, false);

    auto   ref_audio_embed_start = std::chrono::high_resolution_clock::now();
    auto * ref_audio_embeds      = omni_audio_embed_make_with_filename(
        ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, bootstrap_ref_audio);
    duplex_timing_note_audio(ctx_omni, index,
                             timing_elapsed_ms(ref_audio_embed_start, std::chrono::high_resolution_clock::now()));
    if (ref_audio_embeds != nullptr && ref_audio_embeds->n_pos > 0) {
        print_with_timestamp("system prompt ref_audio embedding: n_pos=%d\n", ref_audio_embeds->n_pos);
        prefill_with_emb(ctx_omni, ctx_omni->params, ref_audio_embeds->embed, ref_audio_embeds->n_pos,
                         ctx_omni->params->n_batch, &ctx_omni->session.n_past);
        omni_embed_free(ref_audio_embeds);
    } else {
        print_with_timestamp("WARNING: failed to load system prompt ref_audio: %s\n", bootstrap_ref_audio.c_str());
    }

    omni_llm_stage_eval_string(ctx_omni, ctx_omni->params, prompts.assistant_prompt.c_str(), ctx_omni->params->n_batch,
                               &ctx_omni->session.n_past, false);

    ctx_omni->session.prompt.system_prompt_initialized = true;
    ctx_omni->session.prompt.n_keep                    = ctx_omni->session.n_past;
    print_with_timestamp("🔒 n_keep 设置为 %d (system prompt tokens)，这部分永远不会被滑动窗口删除\n",
                         ctx_omni->session.prompt.n_keep);
    eval_prefix(ctx_omni, ctx_omni->params);

    print_with_timestamp("stream_prefill(index=0): system prompt 初始化完成，ref_audio 已在其中 prefill\n");
    sliding_window_register_system_prompt(ctx_omni);
    print_with_timestamp("n_past = %d\n", ctx_omni->session.n_past);

    if (setup.should_start_workers) {
        const OmniWorkerThreadFns worker_fns = {
            omni_llm_stage_worker_loop,
            omni_tts_worker_loop_simplex,
            omni_tts_worker_loop_duplex,
            t2w_thread_func,
        };
        omni_ensure_prefill_workers_started(ctx_omni, worker_fns);

        if (ctx_omni->duplex_mode && ctx_omni->llm_thread_info != nullptr) {
            ctx_omni->llm_thread_info->decode_requested.store(true);
            ctx_omni->llm_thread_info->cv.notify_one();
        }
    }

    return true;
}

// Step C: encode image/audio input into a single prefill payload.
static bool omni_encode_prefill_input(struct omni_context * ctx_omni,
                                      const std::string &   aud_fname,
                                      const std::string &   img_fname,
                                      int                   index,
                                      int                   max_slice_nums,
                                      struct omni_embeds &  encoded) {
    const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    encoded.index         = index;

    if (!img_fname.empty()) {
        if (max_slice_nums >= 1 && ctx_omni->ctx_vision != nullptr) {
            vision_set_max_slice_nums(ctx_omni->ctx_vision, max_slice_nums);
            LOG_INF("%s: [临时] max_slice_nums=%d for this prefill\n", __func__, max_slice_nums);
        }

        auto       vit_embed_start = std::chrono::high_resolution_clock::now();
        const bool image_embed_ok  = omni_image_embed_make_chunks_with_filename(
            ctx_omni->ctx_vision, ctx_omni->params->cpuparams.n_threads, img_fname, encoded.vision_embed);
        duplex_timing_note_vit(ctx_omni, index,
                               timing_elapsed_ms(vit_embed_start, std::chrono::high_resolution_clock::now()));
        if (!image_embed_ok) {
            LOG_ERR("%s: failed to create vision embeddings for %s\n", __func__, img_fname.c_str());
            return false;
        }
        LOG_INF("%s: vision_embed has %d chunks\n", __func__, (int) encoded.vision_embed.size());
    }

    if (!aud_fname.empty()) {
        print_with_timestamp("stream_prefill(index=%d): processing user audio: %s\n", index, aud_fname.c_str());
        auto   audio_embed_start = std::chrono::high_resolution_clock::now();
        auto * audio_embeds =
            omni_audio_embed_make_with_filename(ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, aud_fname);
        duplex_timing_note_audio(ctx_omni, index,
                                 timing_elapsed_ms(audio_embed_start, std::chrono::high_resolution_clock::now()));
        if (audio_embeds != nullptr && audio_embeds->n_pos > 0) {
            print_with_timestamp("stream_prefill(index=%d): user audio embedding: n_pos=%d\n", index,
                                 audio_embeds->n_pos);
            encoded.audio_embed.resize(audio_embeds->n_pos * hidden_size);
            std::memcpy(encoded.audio_embed.data(), audio_embeds->embed, encoded.audio_embed.size() * sizeof(float));
            omni_embed_free(audio_embeds);
        } else {
            LOG_WRN("%s: audio encoding failed, skipping audio for this frame: %s\n", __func__, aud_fname.c_str());
        }
    }

    return true;
}

// Step D: submit the encoded payload to the shared LLM stage.
static bool omni_submit_llm_prefill(struct omni_context * ctx_omni, std::unique_ptr<struct omni_embeds> encoded) {
    if (!ctx_omni->async) {
        omni_llm_stage_prefill_apply(ctx_omni, ctx_omni->params, *encoded);
        omni_llm_stage_finalize_prefill(ctx_omni);
        return true;
    }

    std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
    ctx_omni->llm_thread_info->cv.wait(lock, [&] {
        return ctx_omni->llm_thread_info->queue.size() < static_cast<size_t>(ctx_omni->llm_thread_info->MAX_QUEUE_SIZE);
    });
    ctx_omni->llm_thread_info->queue.push(encoded.release());
    lock.unlock();
    ctx_omni->llm_thread_info->cv.notify_all();
    return true;
}

// Helper function to play WAV file
static void play_wav_file(const std::string & wav_file_path) {
#ifndef _WIN32
    // Play audio asynchronously using fork() to avoid blocking TTS thread
    pid_t pid = fork();
    if (pid == 0) {
#    ifdef __APPLE__
        execl("/usr/bin/afplay", "afplay", wav_file_path.c_str(), (char *) NULL);
#    else
        execl("/usr/bin/aplay", "aplay", wav_file_path.c_str(), (char *) NULL);
#    endif
        _exit(1);
    } else if (pid > 0) {
        // Parent process: continue without waiting
    } else {
        std::string play_cmd;
#    ifdef __APPLE__
        play_cmd = "afplay \"" + wav_file_path + "\" &";
#    else
        play_cmd = "aplay \"" + wav_file_path + "\" &";
#    endif
        LOG_WRN("TTS: fork() failed, using system() fallback for audio playback\n");
        system(play_cmd.c_str());
    }
#endif
    // Windows: no-op (audio playback handled by frontend)
}

// Helper function to move old output directory to old_output/<id>/
static void move_old_output_to_archive() {
    const std::string base_output_dir     = "./tools/omni/output";
    const std::string old_output_base_dir = "./old_output";

    // Helper function to check if directory exists and has content
    auto dir_has_content = [](const std::string & dir_path) -> bool {
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0) {
            return false;  // Directory doesn't exist
        }
        if (!(info.st_mode & S_IFDIR)) {
            return false;  // Not a directory
        }

        // Check if directory has any files/subdirectories
#ifdef _WIN32
        std::string cmd = "dir /b \"" + dir_path + "\" 2>NUL | findstr /r \".\" >NUL 2>&1";
#else
        std::string cmd = "test -n \"$(ls -A " + dir_path + " 2>/dev/null)\"";
#endif
        int ret = system(cmd.c_str());
        return (ret == 0);  // Returns 0 if directory has content
    };

    // Helper function to create directory
    auto create_dir = [](const std::string & dir_path) -> bool {
        struct stat info;
        if (stat(dir_path.c_str(), &info) == 0) {
            if (!(info.st_mode & S_IFDIR)) {
                LOG_ERR("Output path exists but is not a directory: %s\n", dir_path.c_str());
                return false;
            }
            return true;
        }

        // Directory doesn't exist, try to create it
        if (!cross_platform_mkdir_p(dir_path)) {
            LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    };

    // Helper function to find next available ID in old_output directory
    auto get_next_output_id = [](const std::string & old_output_base) -> int {
        // Ensure old_output base directory exists
        cross_platform_mkdir_p(old_output_base);

        // Find maximum ID in old_output directory
        int max_id = -1;
#ifdef _WIN32
        std::string find_cmd = "dir /b \"" + old_output_base + "\" 2>NUL";
#else
        std::string find_cmd = "ls -1 " + old_output_base + " 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1";
#endif
        FILE * pipe = popen(find_cmd.c_str(), "r");
        if (pipe) {
            char buffer[128];
#ifdef _WIN32
            // On Windows, read all entries and find the max numeric ID
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                std::string result(buffer);
                while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                    result.pop_back();
                }
                if (!result.empty()) {
                    try {
                        int id = std::stoi(result);
                        if (id > max_id)
                            max_id = id;
                    } catch (...) {
                    }
                }
            }
#else
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                std::string result(buffer);
                // Remove trailing newline
                while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                    result.pop_back();
                }
                if (!result.empty()) {
                    try {
                        max_id = std::stoi(result);
                    } catch (...) {
                        max_id = -1;
                    }
                }
            }
#endif
            pclose(pipe);
        }

        return max_id + 1;  // Next ID is max_id + 1 (or 0 if no existing IDs)
    };

    // Check if output directory has content
    bool output_has_content = false;

    // Check if base output directory exists and has content
    if (dir_has_content(base_output_dir)) {
        output_has_content = true;
    } else {
        // Check subdirectories
        if (dir_has_content(base_output_dir + "/llm_debug") || dir_has_content(base_output_dir + "/tts_txt") ||
            dir_has_content(base_output_dir + "/tts_wav")) {
            output_has_content = true;
        }
    }

    if (output_has_content) {
        int         next_id        = get_next_output_id(old_output_base_dir);
        std::string old_output_dir = old_output_base_dir + "/" + std::to_string(next_id);

        // Create old_output/<id> directory
        if (!create_dir(old_output_dir)) {
            LOG_ERR("Failed to create old_output directory: %s\n", old_output_dir.c_str());
        } else {
            // Move entire output directory contents to old_output/<id>/
            // Use find + xargs for more reliable moving
#ifdef _WIN32
            std::string move_cmd = "robocopy \"" + base_output_dir + "\" \"" + old_output_dir + "\" /E /MOVE >NUL 2>&1";
            system(move_cmd.c_str());
            // robocopy returns non-zero on success (exit codes < 8 are success)
            // Re-create the base output dir since robocopy /MOVE removes it
            cross_platform_mkdir_p(base_output_dir);
#else
            std::string move_cmd = "find " + base_output_dir + " -mindepth 1 -maxdepth 1 -exec mv {} " +
                                   old_output_dir + "/ \\; 2>/dev/null";
            int         ret      = system(move_cmd.c_str());
            if (ret == 0) {
            } else {
                // Fallback: try simple mv command
                std::string fallback_cmd =
                    "sh -c 'cd " + base_output_dir + " && mv * " + old_output_dir + "/ 2>/dev/null || true'";
                ret = system(fallback_cmd.c_str());
                if (ret == 0) {
                } else {
                    LOG_WRN("Failed to move old output directory (may be empty or already moved)\n");
                }
            }
#endif
        }
    } else {
    }
}

// ==============================================================================
// TTS Thread Function - Duplex Mode
// 双工模式专用的 TTS 线程函数
// 与单工版本的主要差异：
// 1. 不需要 simplex_round_idx 管理和 round_XXX 输出目录
// 2. TTS KV cache 跨 chunk 保持（由 is_end_of_turn 控制是否重置）
// 3. 使用 generate_audio_tokens_local（双工版本，max_audio_tokens=26）
bool stream_prefill(struct omni_context * ctx_omni,
                    const std::string &   aud_fname,
                    const std::string &   img_fname,
                    int                   index,
                    int                   max_slice_nums) {
    if (ctx_omni->duplex_mode) {
        duplex_timing_set_active_chunk(ctx_omni, index);
    }

    const OmniPrefillSetup     setup   = omni_turn_coordinator_prepare_prefill(ctx_omni, index);
    const OmniBootstrapPrompts prompts = omni_select_bootstrap_prompts(ctx_omni);

    if (!omni_run_session_bootstrap_if_needed(ctx_omni, setup, prompts, aud_fname, index)) {
        return false;
    }

    if (!setup.need_bootstrap) {
        auto encoded = std::make_unique<struct omni_embeds>();
        if (!omni_encode_prefill_input(ctx_omni, aud_fname, img_fname, index, max_slice_nums, *encoded)) {
            return false;
        }
        if (!omni_submit_llm_prefill(ctx_omni, std::move(encoded))) {
            return false;
        }
    }

    // 🔧 [诊断] 打印 stream_prefill 结束时的状态
    print_with_timestamp("\n\nc++ finish stream_prefill(index=%d). n_past=%d, n_keep=%d, n_ctx=%d\n\n", index,
                         ctx_omni->session.n_past, ctx_omni->session.prompt.n_keep, ctx_omni->params->n_ctx);
    return true;
}

static bool omni_can_decode(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        LOG_ERR("stream_decode: ctx_omni is nullptr!");
        return false;
    }
    if (ctx_omni->ctx_llama == nullptr) {
        LOG_ERR("stream_decode: ctx_omni->ctx_llama is nullptr!");
        return false;
    }
    if (ctx_omni->params == nullptr) {
        LOG_ERR("stream_decode: ctx_omni->params is nullptr!");
        return false;
    }
    return true;
}

static bool omni_pipeline_wait_decode_result(struct omni_context * ctx_omni) {
    std::unique_lock<std::mutex> lock(ctx_omni->pipeline_result_mtx);
    ctx_omni->pipeline_result_cv.wait(lock, [&] {
        return !ctx_omni->pipeline_result_queue.empty() || !ctx_omni->workers.llm_thread_running;
    });

    if (ctx_omni->pipeline_result_queue.empty()) {
        return false;
    }

    const PipelineDecodeResult result = ctx_omni->pipeline_result_queue.front();
    ctx_omni->pipeline_result_queue.pop();
    ctx_omni->turn.ended_with_listen.store(result.ended_with_listen);
    return result.decode_ok;
}

bool stream_decode(struct omni_context * ctx_omni, std::string debug_dir, int round_idx) {
    if (!omni_can_decode(ctx_omni)) {
        return false;
    }

    const OmniWorkerThreadFns       worker_fns = {
        omni_llm_stage_worker_loop,
        omni_tts_worker_loop_simplex,
        omni_tts_worker_loop_duplex,
        t2w_thread_func,
    };

    if (ctx_omni->async) {
        {
            // The worker reads request metadata right before it starts decode.
            std::lock_guard<std::mutex> lock(ctx_omni->pipeline_request_mtx);
            ctx_omni->pipeline_debug_dir = std::move(debug_dir);
            ctx_omni->pipeline_round_idx = round_idx;
        }

        omni_ensure_prefill_workers_started(ctx_omni, worker_fns);

        if (!ctx_omni->duplex_mode && ctx_omni->llm_thread_info != nullptr) {
            // Simplex async needs an explicit "prefill is complete, decode now" signal.
            ctx_omni->llm_thread_info->decode_requested.store(true);
            ctx_omni->llm_thread_info->cv.notify_one();
        }

        return omni_pipeline_wait_decode_result(ctx_omni);
    }

    const OmniLlmStageDecodeRequest request = { std::move(debug_dir), round_idx };

    omni_turn_coordinator_prepare_decode(ctx_omni, request.round_idx);

    LOG_INF("<user>%s\n", ctx_omni->params->prompt.c_str());
    LOG_INF("<assistant>");

    if (!omni_llm_stage_decode_run(ctx_omni, request)) {
        return false;
    }

    omni_turn_coordinator_close(ctx_omni, OmniTurnCloseKind::finish);
    return true;
}

bool stop_speek(struct omni_context * ctx_omni) {
    omni_turn_coordinator_close(ctx_omni, OmniTurnCloseKind::abort, "stop_speek");
    return true;
}

bool clean_kvcache(struct omni_context * ctx_omni) {
    if (ctx_omni->clean_kvcache) {
        print_with_timestamp("🧹 clean_kvcache: 清理 KV cache, 删除范围=[%d, %d), n_keep=%d\n",
                             ctx_omni->session.prompt.n_keep, ctx_omni->session.n_past,
                             ctx_omni->session.prompt.n_keep);

        // 获取 memory 对象并清理 KV cache
        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
        if (mem) {
            // 删除 [n_keep, n_past) 范围的所有 token，保留 system prompt 等
            bool rm_ok = llama_memory_seq_rm(mem, 0, ctx_omni->session.prompt.n_keep, ctx_omni->session.n_past);
            if (!rm_ok) {
                print_with_timestamp("🧹 clean_kvcache: llama_memory_seq_rm 失败\n");
            } else {
                print_with_timestamp("🧹 clean_kvcache: llama_memory_seq_rm 成功\n");
            }
        } else {
            print_with_timestamp("🧹 clean_kvcache: 无法获取 memory 对象\n");
        }

        // 重置 n_past 到 n_keep
        int old_n_past           = ctx_omni->session.n_past;
        ctx_omni->session.n_past = ctx_omni->session.prompt.n_keep;
        print_with_timestamp("🧹 clean_kvcache: n_past 从 %d 重置到 %d\n", old_n_past, ctx_omni->session.n_past);

        // 🔧 [#39 滑动窗口] 重置滑窗状态，但保留 n_keep 对应的 system prompt 区间
        sliding_window_reset_after_kvcache_clean(ctx_omni);
        print_with_timestamp("🧹 clean_kvcache: 滑窗状态已重置, system_preserve_length=%d\n",
                             ctx_omni->session.prompt.system_preserve_length);
    } else {
        print_with_timestamp("🧹 clean_kvcache: clean_kvcache=false, 跳过清理\n");
    }

    return true;
}
