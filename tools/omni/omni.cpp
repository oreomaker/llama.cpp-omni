#include "omni-impl.h"
#include "omni-output.h"
#include "omni-python-t2w.h"
#include "omni-sliding-window.h"
#include "omni-t2w-stage.h"
#include "omni-tts-stage.h"
#include "omni-turn-coordinator.h"
#include "omni-token-protocol.h"
#include "omni-worker-coordinator.h"
#include "vision.h"
#include "audition.h"
#include "omni.h"
#include "token2wav/token2wav-impl.h"

#include "llama.h"
#include "common/common.h"
#include "common/sampling.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <iostream>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include <set>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <sstream>
#include <random>
#include <cstdarg>
#include <signal.h>

#ifdef _WIN32
    #include <windows.h>
    #include <direct.h>
    #include <io.h>
    #include <process.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    // Windows compatibility macros
    #define popen  _popen
    #define pclose _pclose
    #define unlink _unlink
    #define stat   _stat
    #define S_IFDIR _S_IFDIR
#else
    #include <sys/time.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <unistd.h>
#endif

//
// Forward declarations
//
void print_with_timestamp(const char* format, ...);
void llm_thread_func(struct omni_context * ctx_omni, common_params * params);
void tts_thread_func(struct omni_context * ctx_omni, common_params * params);
void tts_thread_func_duplex(struct omni_context * ctx_omni, common_params * params);

//
// omni structure
//
struct unit_buffer {
    std::vector<float> buffer;
    std::string text;
    bool completed = false;
    int unit_n_past = 0;
    float duration;
};

struct omni_output {
    std::vector<unit_buffer *> output;
    int idx;
};

omni_context::omni_context()
    : n_past(session.n_past)
    , n_keep(session.prompt.n_keep)
    , round_start_positions(session.round_start_positions)
    , max_preserved_context(session.max_preserved_context)
    , sliding_window_config(session.sliding_window_config)
    , unit_history(session.unit_history)
    , next_unit_id(session.next_unit_id)
    , pending_unit_id(session.pending_unit_id)
    , pending_unit_start_cache_len(session.pending_unit_start_cache_len)
    , system_preserve_length(session.prompt.system_preserve_length)
    , position_offset(session.position_offset)
    , sliding_event_count(session.sliding_event_count)
    , total_dropped_tokens(session.total_dropped_tokens)
    , total_dropped_units(session.total_dropped_units)
    , need_speek(gate.prefill_requested)
    , speek_done(gate.speech_ready)
    , current_turn_ended(turn.current_turn_ended)
    , break_event(gate.break_event)
    , session_stop_event(gate.session_stop_event)
    , ended_with_listen(turn.ended_with_listen)
    , llm_generation_done(gate.llm_generation_done)
    , system_prompt_initialized(session.prompt.system_prompt_initialized)
    , text_streaming(gate.text_streaming)
    , text_done_flag(gate.text_done)
    , wav_turn_base(session.current_round.wav_turn_base)
    , simplex_round_idx(session.current_round.round_idx) {
    session.current_round.duplex_mode = false;
}

omni_context::~omni_context() = default;

static double timing_elapsed_ms(
    const std::chrono::high_resolution_clock::time_point & start,
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
    auto & timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.vit_embedding_ms = timing.vit_embedding_ms < 0.0 ? ms : timing.vit_embedding_ms + ms;
}

static void duplex_timing_note_audio(struct omni_context * ctx_omni, int chunk_idx, double ms) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto & timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.audio_embedding_ms = timing.audio_embedding_ms < 0.0 ? ms : timing.audio_embedding_ms + ms;
}

static void duplex_timing_note_tts(struct omni_context * ctx_omni, int chunk_idx, double ms, int audio_token_count) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto & timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.tts_audio_token_ms = timing.tts_audio_token_ms < 0.0 ? ms : timing.tts_audio_token_ms + ms;
    timing.tts_audio_token_count += audio_token_count;
    timing.tts_done = true;
}

static void duplex_timing_mark_tts_done(struct omni_context * ctx_omni, int chunk_idx) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto & timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    if (timing.tts_audio_token_ms < 0.0) {
        timing.tts_audio_token_ms = 0.0;
    }
    timing.tts_done = true;
}

static void duplex_timing_note_t2w(struct omni_context * ctx_omni, int chunk_idx, double ms, int window_count, bool done) {
    if (ctx_omni == nullptr || chunk_idx < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto & timing = ctx_omni->duplex_chunk_timings[chunk_idx];
    timing.token2wav_ms = timing.token2wav_ms < 0.0 ? ms : timing.token2wav_ms + ms;
    timing.token2wav_window_count += window_count;
    if (done) {
        timing.token2wav_done = true;
    }
}

//
// omni mtmd embed
//
bool omni_eval_embed(llama_context * ctx_llama, const struct omni_embed * omni_embed, int n_batch, int * n_past) {
    int n_embd  = llama_n_embd(llama_get_model(ctx_llama));

    for (int i = 0; i < omni_embed->n_pos; i += n_batch) {
        int n_eval = omni_embed->n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd     = (omni_embed->embed + i*n_embd);
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

bool prefill_with_emb(struct omni_context * ctx_omni, common_params * params, float* embed, int n_pos, int n_batch, int*n_past) {
    kv_cache_slide_window(ctx_omni, params, n_pos);
    
    int n_embd  = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    for (int i = 0; i < n_pos; i += n_batch) {
        int n_eval = n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd     = (embed + i*n_embd);
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
bool prefill_emb_with_hidden(struct omni_context * ctx_omni, common_params * params, float* embed, int n_pos, int n_batch, int* n_past, float *& hidden_states) {
    if (n_pos == 0) {
        hidden_states = nullptr;
        return true;
    }

    kv_cache_slide_window(ctx_omni, params, n_pos);

    int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));

    // 在函数内部分配空间
    hidden_states = (float *)malloc(n_pos * n_embd * sizeof(float));
    if (hidden_states == nullptr) {
        LOG_ERR("%s : failed to allocate memory for hidden_states\n", __func__);
        return false;
    }

    int tokens_processed = 0;

    for (int i = 0; i < n_pos; i += n_batch) {
        int n_eval = n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }

        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd     = (embed + i * n_embd);
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

// 🔧 [高清模式] 返回分离的 chunk embeds（二维 vector）
// 返回值: vision_chunks[0] = overview, vision_chunks[1..n] = slices
static bool encode_image_with_vision_chunks(vision_ctx * ctx_vision, int n_threads, const vision_image_u8 * img, 
                                            std::vector<std::vector<float>> & vision_chunks) {
    const int64_t t_img_enc_start_us = ggml_time_us();
    vision_image_f32_batch img_res_v;
    img_res_v.entries.resize(0);
    img_res_v.entries.clear();
    if (!vision_image_preprocess(ctx_vision, img, &img_res_v)) {
        LOG_ERR("%s: unable to preprocess image\n", __func__);
        img_res_v.entries.clear();
        return false;
    }
    
    int n_embd = vision_n_mmproj_embd(ctx_vision);
    int n_tokens = vision_n_output_tokens(ctx_vision);
    
    vision_chunks.clear();
    vision_chunks.resize(img_res_v.entries.size());
    
    for (size_t i = 0; i < img_res_v.entries.size(); i++) {
        const int64_t t_img_enc_step_start_us = ggml_time_us();
        
        // 为每个 chunk 分配空间
        vision_chunks[i].resize(n_embd * n_tokens);
        
        bool encoded = vision_image_encode(ctx_vision, n_threads, img_res_v.entries[i].get(), vision_chunks[i].data());
        if (!encoded) {
            LOG_ERR("Unable to encode image - spatial_unpad - subimage %d of %d\n", (int) i+1, (int) img_res_v.entries.size());
            return false;
        }
        const int64_t t_img_enc_steop_batch_us = ggml_time_us();
        LOG_INF("%s: step %d of %d encoded in %8.2f ms\n", __func__, (int)i+1, (int)img_res_v.entries.size(), (t_img_enc_steop_batch_us - t_img_enc_step_start_us) / 1000.0);
    }
    const int64_t t_img_enc_batch_us = ggml_time_us();
    LOG_INF("%s: all %d chunks encoded in %8.2f ms (grid: %dx%d)\n", __func__, 
            (int)img_res_v.entries.size(), (t_img_enc_batch_us - t_img_enc_start_us) / 1000.0,
            img_res_v.grid_x, img_res_v.grid_y);

    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;
    int total_tokens = (int)vision_chunks.size() * n_tokens;
    LOG_INF("\n%s: image encoded in %8.2f ms by vision (%8.2f ms per chunk, %d total tokens)\n", 
            __func__, t_img_enc_ms, t_img_enc_ms / vision_chunks.size(), total_tokens);

    return true;
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

// 🔧 [高清模式] 创建带 chunks 的 vision embed（用于 V2.6 slice schema）
// 返回的 vector: [0] = overview, [1..n] = slices
bool omni_image_embed_make_chunks_with_filename(struct vision_ctx * ctx_vision, int n_threads, 
                                                 std::string image_path, 
                                                 std::vector<std::vector<float>> & vision_chunks) {
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
    
    bool success = encode_image_with_vision_chunks(ctx_vision, n_threads, img, vision_chunks);
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

static bool eval_tokens(struct omni_context* ctx_omni, common_params* params, std::vector<llama_token> tokens, int n_batch, int * n_past, bool get_emb = false) {
    int N = (int) tokens.size();
    kv_cache_slide_window(ctx_omni, params, N);

    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (n_eval == 0)
            break;
        if (get_emb) {
            llama_set_embeddings(ctx_omni->ctx_llama, true);
        }
        // llama_batch_get_one 返回的 batch.pos 可能是 nullptr，需要手动设置
        llama_batch batch = llama_batch_get_one(&tokens[i], n_eval);
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = *n_past + j;  // 从当前 n_past 位置开始
        }
        
        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        if (get_emb) {
            llama_set_embeddings(ctx_omni->ctx_llama, false);
        }
        *n_past += n_eval;
    }
    return true;
}

// 与 eval_tokens 类似，但会将每次 decode 的 hidden_state 保存并拼接到 hidden_states 中
// hidden_states 由函数内部分配空间，大小为 N * n_embd * sizeof(float)，调用者负责释放
static bool eval_tokens_with_hidden(struct omni_context* ctx_omni, common_params* params, std::vector<llama_token> tokens, int n_batch, int * n_past, float *& hidden_states) {
    int N = (int) tokens.size();
    if (N == 0) {
        hidden_states = nullptr;
        return true;
    }

    kv_cache_slide_window(ctx_omni, params, N);

    const int n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));

    // 在函数内部分配空间
    hidden_states = (float *)malloc(N * n_embd * sizeof(float));
    if (hidden_states == nullptr) {
        LOG_ERR("%s : failed to allocate memory for hidden_states\n", __func__);
        return false;
    }

    int tokens_processed = 0;

    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (n_eval == 0)
            break;

        // 启用 embeddings 输出
        llama_set_embeddings(ctx_omni->ctx_llama, true);
        // llama_batch_get_one 返回的 batch.pos 可能是 nullptr，需要手动设置
        llama_batch batch = llama_batch_get_one(&tokens[i], n_eval);
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        for (int j = 0; j < n_eval; j++) {
            batch.pos[j] = *n_past + j;  // 从当前 n_past 位置开始
        }

        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            llama_set_embeddings(ctx_omni->ctx_llama, false);
            free(hidden_states);
            hidden_states = nullptr;
            return false;
        }

        // 获取当前 batch 的 embeddings 并复制到 hidden_states
        float * emb = llama_get_embeddings(ctx_omni->ctx_llama);
        if (emb != nullptr) {
            // 将当前 batch 的 embeddings 复制到 hidden_states 的对应位置
            memcpy(hidden_states + tokens_processed * n_embd, emb, n_eval * n_embd * sizeof(float));
        }

        llama_set_embeddings(ctx_omni->ctx_llama, false);

        *n_past += n_eval;
        tokens_processed += n_eval;
    }
    return true;
}

static bool eval_id(struct omni_context * ctx_omni, common_params* params, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_omni, params, tokens, 1, n_past);
}

static bool eval_id_with_hidden(struct omni_context * ctx_omni, common_params* params, int id, int * n_past, float *& hidden_states) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens_with_hidden(ctx_omni, params, tokens, 1, n_past, hidden_states);
}

static bool eval_string(struct omni_context * ctx_omni, common_params* params, const char* str, int n_batch, int * n_past, bool add_bos, bool get_emb = false) {
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_omni->ctx_llama, str2, add_bos, true);
    return eval_tokens(ctx_omni, params, embd_inp, n_batch, n_past, get_emb);
}

static bool eval_string_with_hidden(struct omni_context * ctx_omni, common_params* params, const char* str, int n_batch, int * n_past, bool add_bos, float *& hidden_states) {
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_omni->ctx_llama, str2, add_bos, true);
    return eval_tokens_with_hidden(ctx_omni, params, embd_inp, n_batch, n_past, hidden_states);
}

static const char * sample(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past) {
    const llama_token id = common_sampler_sample(smpl, ctx_omni->ctx_llama, -1);
    common_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx_omni->ctx_llama)), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_omni->ctx_llama, id);
    }
    eval_id(ctx_omni, params, id, n_past);
    return ret.c_str();
}

static const char * sample_with_hidden(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past, float *& hidden_states) {
    const llama_token id = common_sampler_sample(smpl, ctx_omni->ctx_llama, -1);
    common_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx_omni->ctx_llama)), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_omni->ctx_llama, id);
    }
    eval_id_with_hidden(ctx_omni, params, id, n_past, hidden_states);
    return ret.c_str();
}

static const char * llama_loop(struct omni_context * ctx_omni, common_params *params, struct common_sampler * smpl, int &n_past) {
    const char * tmp = sample(smpl, ctx_omni, params, &n_past);
    return tmp;
}

// 修改sample_with_hidden来返回token ID（通过引用参数）
// 🔧 [双工模式] 支持 listen_prob_scale 参数，增加 <|listen|> 的采样概率
// 🔧 [双工模式] 支持 forbidden_token_ids，禁止采样 <|tts_pad|> 等 token
static const char * sample_with_hidden_and_token(struct common_sampler * smpl, struct omni_context * ctx_omni, common_params* params, int * n_past, float *& hidden_states, llama_token & token_id) {
    float * logits = llama_get_logits_ith(ctx_omni->ctx_llama, -1);
    
    // 🔧 [双工模式] 在采样前调整 logits
    if (ctx_omni->duplex_mode) {
        if (logits != nullptr) {
            // 1. 调整 <|listen|> 的 logit（listen_prob_scale）
            // listen_prob_scale > 1.0 会增加 <|listen|> 的概率，让模型更倾向于先听
            if (ctx_omni->special_token_listen >= 0) {
                // 使用 listen_prob_scale 调整 <|listen|> 的 logit
                // 默认值 1.0 不改变，> 1.0 增加 listen 概率
                // 这里我们使用加法而不是乘法，因为 logit 可能是负数
                // 添加一个偏置值来增加 listen 的概率
                // listen_prob_bias = log(listen_prob_scale) ≈ (listen_prob_scale - 1.0) for small values
                float listen_bias = (ctx_omni->listen_prob_scale - 1.0f) * 2.0f;  // 放大效果
                logits[ctx_omni->special_token_listen] += listen_bias;
            }
            
            // 2. 🔧 [与 Python 对齐] 禁止采样 <|tts_pad|> token
            // Python: self.forbidden_token_ids = [self.tts_pad_id] + list(bad_token_ids)
            //         logits[:, self.forbidden_token_ids] = float("-inf")
            // <|tts_pad|> 是填充 token，模型不应该主动生成它
            // 如果不禁止，模型可能生成 <|speak|> → <|tts_pad|> → <|chunk_eos|>，导致无有效输出
            if (ctx_omni->special_token_tts_pad >= 0) {
                logits[ctx_omni->special_token_tts_pad] = -INFINITY;
            }
        }
    }
    
    // 🔧 [Length Penalty] 调整 EOS token 的 logit 值（单工模式）
    // length_penalty > 1.0 会降低 EOS 概率，让模型生成更长的输出
    if (!ctx_omni->duplex_mode && ctx_omni->length_penalty != 1.0f && ctx_omni->special_token_tts_eos >= 0) {
        if (logits != nullptr) {
            float eos_logit = logits[ctx_omni->special_token_tts_eos];
            if (eos_logit > 0) {
                // logit > 0 时，除以 length_penalty 来降低概率
                logits[ctx_omni->special_token_tts_eos] = eos_logit / ctx_omni->length_penalty;
            } else {
                // logit <= 0 时，乘以 length_penalty 来降低概率
                logits[ctx_omni->special_token_tts_eos] = eos_logit * ctx_omni->length_penalty;
            }
        }
    }
    
    const llama_token id = common_sampler_sample(smpl, ctx_omni->ctx_llama, -1);
    token_id = id;  // 保存token ID
    common_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx_omni->ctx_llama)), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_omni->ctx_llama, id);
    }
    eval_id_with_hidden(ctx_omni, params, id, n_past, hidden_states);
    return ret.c_str();
}

static const char * llama_loop_with_hidden(struct omni_context * ctx_omni, common_params *params, struct common_sampler * smpl, int &n_past, float *& hidden_states) {
    llama_token dummy_token;
    const char * tmp = sample_with_hidden_and_token(smpl, ctx_omni, params, &n_past, hidden_states, dummy_token);
    return tmp;
}

// 新增：返回token ID的版本
static const char * llama_loop_with_hidden_and_token(struct omni_context * ctx_omni, common_params *params, struct common_sampler * smpl, int &n_past, float *& hidden_states, llama_token & token_id) {
    const char * tmp = sample_with_hidden_and_token(smpl, ctx_omni, params, &n_past, hidden_states, token_id);
    return tmp;
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
    (void)use_cuda;
    model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
#endif

    if (!model.backend) {
        LOG_ERR("Projector: failed to init backend\n");
        gguf_free(ctx_gguf);
        return false;
    }
    
    model.buf_type = ggml_backend_get_default_buffer_type(model.backend);
    
    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    
    size_t ctx_size = ggml_tensor_overhead() * n_tensors;
    struct ggml_init_params ctx_params = {
        /*.mem_size   = */ ctx_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    model.ctx_w = ggml_init(ctx_params);
    
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        enum ggml_type type = gguf_get_tensor_type(ctx_gguf, i);
        struct ggml_tensor * tensor = nullptr;
        
        if (strcmp(name, "linear1.weight") == 0) {
            tensor = ggml_new_tensor_2d(model.ctx_w, type, 4096, 768);
            model.layer.linear1_weight = tensor;
            model.hparams.in_dim = 4096;
            model.hparams.out_dim = 768;
        } else if (strcmp(name, "linear1.bias") == 0) {
            tensor = ggml_new_tensor_1d(model.ctx_w, type, 768);
            model.layer.linear1_bias = tensor;
        } else if (strcmp(name, "linear2.weight") == 0) {
            tensor = ggml_new_tensor_2d(model.ctx_w, type, 768, 768);
            model.layer.linear2_weight = tensor;
        } else if (strcmp(name, "linear2.bias") == 0) {
            tensor = ggml_new_tensor_1d(model.ctx_w, type, 768);
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
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model.ctx_w, name);
        if (!tensor) continue;
        
        size_t offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
        fseek(f, offset, SEEK_SET);
        
        size_t tensor_size = ggml_nbytes(tensor);
        void * data = malloc(tensor_size);
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
    if (model.ctx_w) ggml_free(model.ctx_w);
    if (model.buf_w) ggml_backend_buffer_free(model.buf_w);
    if (model.backend) ggml_backend_free(model.backend);
    model.ctx_w = nullptr;
    model.buf_w = nullptr;
    model.backend = nullptr;
    model.initialized = false;
}

static struct ggml_cgraph * projector_build_graph(projector_model & model, struct ggml_context * ctx, struct ggml_tensor * input) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    
    // linear1 + relu
    struct ggml_tensor * hidden = ggml_mul_mat(ctx, model.layer.linear1_weight, input);
    hidden = ggml_add(ctx, hidden, model.layer.linear1_bias);
    hidden = ggml_relu(ctx, hidden);
    
    // linear2
    struct ggml_tensor * output = ggml_mul_mat(ctx, model.layer.linear2_weight, hidden);
    output = ggml_add(ctx, output, model.layer.linear2_bias);
    
    ggml_build_forward_expand(gf, output);
    return gf;
}

std::vector<float> projector_forward(projector_model & model, const float * input_data, int n_tokens) {
    const int in_dim = model.hparams.in_dim;
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
    
    size_t ctx_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
    struct ggml_init_params params = {
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
        LOG_ERR("Projector: graph compute failed with status %d\n", (int)status);
        ggml_backend_buffer_free(buf_compute);
        ggml_free(ctx);
        return {};
    }
    
    struct ggml_tensor * output = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
    std::vector<float> result(n_tokens * out_dim);
    ggml_backend_tensor_get(output, result.data(), 0, n_tokens * out_dim * sizeof(float));
    
    ggml_backend_buffer_free(buf_compute);
    ggml_free(ctx);
    
    return result;
}
// ==============================================================================

// Load TTS weights from GGUF file
bool load_tts_weights_from_gguf(struct omni_context * ctx_omni, const char * tts_model_path) {
    
    // Initialize GGUF context
    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params params = {
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
    int64_t emb_code_idx = gguf_find_tensor(ctx_gguf, emb_code_name);
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
            int64_t hidden_size = 768;
            
            // GGUF stores as (hidden_size, num_audio_tokens) = [768, 6562]
            if (dim0 == hidden_size && dim1 == num_audio_tokens) {
                // Correct: stored as (hidden_size, num_audio_tokens)
            } else if (dim0 == num_audio_tokens && dim1 == hidden_size) {
                // Stored as (num_audio_tokens, hidden_size) - need to transpose
                num_audio_tokens = dim0;
                hidden_size = dim1;
            } else {
                LOG_ERR("TTS: emb_code.0.weight has unexpected shape [%ld, %ld], expected [768, 6562] or [6562, 768]\n", dim0, dim1);
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Allocate memory and copy data
            size_t emb_code_size = dim0 * dim1 * sizeof(float);
            ctx_omni->emb_code_weight = (float *)malloc(emb_code_size);
            if (!ctx_omni->emb_code_weight) {
                LOG_ERR("TTS: Failed to allocate memory for emb_code.0.weight\n");
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Copy/convert tensor data based on type
            enum ggml_type emb_code_type = emb_code_tensor->type;
            int64_t emb_code_elements = dim0 * dim1;
            
            if (emb_code_type == GGML_TYPE_F32) {
                // F32: direct copy
                memcpy(ctx_omni->emb_code_weight, emb_code_tensor->data, emb_code_size);
            } else if (emb_code_type == GGML_TYPE_F16) {
                // F16: convert to F32
                const ggml_fp16_t * src_f16 = (const ggml_fp16_t *)emb_code_tensor->data;
                for (int64_t i = 0; i < emb_code_elements; ++i) {
                    ctx_omni->emb_code_weight[i] = ggml_fp16_to_fp32(src_f16[i]);
                }
            } else {
                LOG_ERR("TTS: emb_code.0.weight has unsupported type: %d\n", emb_code_type);
                free(ctx_omni->emb_code_weight);
                ctx_omni->emb_code_weight = nullptr;
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            ctx_omni->emb_code_vocab_size = num_audio_tokens;  // 6562
            ctx_omni->emb_code_hidden_size = hidden_size;     // 768
            // NOTE: GGUF data is stored in row-major order, matching NumPy/PyTorch
            // Even when metadata shape is [768, 6562], the actual data layout is (6562, 768) row-major
            // So we should access as: weight[token_idx * hidden_size + j]
            // This means: emb_code_stored_as_transposed should be FALSE
            ctx_omni->emb_code_stored_as_transposed = false; // Data is always (vocab_size, hidden_size) in memory
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
    int64_t emb_text_idx = gguf_find_tensor(ctx_gguf, emb_text_name);
    if (emb_text_idx >= 0) {
        struct ggml_tensor * emb_text_tensor = ggml_get_tensor(ctx_meta, emb_text_name);
        if (emb_text_tensor) {
            int64_t dim0 = emb_text_tensor->ne[0];
            int64_t dim1 = emb_text_tensor->ne[1];
            
            // Expected values
            int64_t expected_vocab_size = 152064;
            int64_t expected_hidden_size = 768;
            
            // GGML tensor 维度理解：
            // ne[0] = 最内层维度 (stride=1) = hidden_size = 768
            // ne[1] = 外层维度 = vocab_size = 152064
            // 内存布局是 row-major，即 [vocab_size][hidden_size]
            // 所以 ne[0]=768, ne[1]=152064 意味着数据已经是 [vocab_size, hidden_size] 格式
            // 不需要转置！
            
            int64_t vocab_size, hidden_size;
            
            if (dim0 == expected_hidden_size && dim1 == expected_vocab_size) {
                // GGML shape: ne[0]=768, ne[1]=152064
                // 这意味着内存布局是 [vocab_size=152064][hidden_size=768]
                // 不需要转置
                vocab_size = dim1;   // 152064
                hidden_size = dim0;  // 768
            } else if (dim0 == expected_vocab_size && dim1 == expected_hidden_size) {
                // GGML shape: ne[0]=152064, ne[1]=768 (unusual)
                // 这意味着内存布局是 [hidden_size=768][vocab_size=152064]
                // 这种情况需要转置
                vocab_size = dim0;   // 152064
                hidden_size = dim1;  // 768
                LOG_ERR("TTS: emb_text.weight has unusual GGML shape, not handled\n");
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            } else {
                LOG_ERR("TTS: emb_text.weight has unexpected shape [%ld, %ld], expected ne=[%ld, %ld]\n", 
                       dim0, dim1, expected_hidden_size, expected_vocab_size);
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Allocate memory for the weight
            size_t emb_text_size = vocab_size * hidden_size * sizeof(float);
            ctx_omni->emb_text_weight = (float *)malloc(emb_text_size);
            if (!ctx_omni->emb_text_weight) {
                LOG_ERR("TTS: Failed to allocate memory for emb_text.weight\n");
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            // Copy/convert tensor data based on type
            enum ggml_type emb_text_type = emb_text_tensor->type;
            int64_t emb_text_elements = vocab_size * hidden_size;
            
            if (emb_text_type == GGML_TYPE_F32) {
                // F32: direct copy
                memcpy(ctx_omni->emb_text_weight, emb_text_tensor->data, emb_text_size);
            } else if (emb_text_type == GGML_TYPE_F16) {
                // F16: convert to F32
                const ggml_fp16_t * src_f16 = (const ggml_fp16_t *)emb_text_tensor->data;
                for (int64_t i = 0; i < emb_text_elements; ++i) {
                    ctx_omni->emb_text_weight[i] = ggml_fp16_to_fp32(src_f16[i]);
                }
            } else {
                LOG_ERR("TTS: emb_text.weight has unsupported type: %d\n", emb_text_type);
                free(ctx_omni->emb_text_weight);
                ctx_omni->emb_text_weight = nullptr;
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            ctx_omni->emb_text_vocab_size = vocab_size;   // 152064
            ctx_omni->emb_text_hidden_size = hidden_size;  // 768
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
    const char * projector_names[] = {
        "projector_semantic.linear1.weight",
        "projector_semantic.linear1.bias",
        "projector_semantic.linear2.weight",
        "projector_semantic.linear2.bias"
    };
    
    float ** projector_ptrs[] = {
        &ctx_omni->projector_semantic_linear1_weight,
        &ctx_omni->projector_semantic_linear1_bias,
        &ctx_omni->projector_semantic_linear2_weight,
        &ctx_omni->projector_semantic_linear2_bias
    };
    
    // PyTorch nn.Linear(in_features, out_features) weight shape is (out_features, in_features)
    // GGUF may store as (out_features, in_features) or (in_features, out_features)
    // We need to detect and handle both cases
    int64_t expected_shapes_pytorch[][2] = {
        {768, 4096},  // linear1.weight: PyTorch shape (out_features, in_features)
        {768, 0},     // linear1.bias (1D)
        {768, 768},   // linear2.weight: PyTorch shape (out_features, in_features)
        {768, 0}      // linear2.bias (1D)
    };
    
    int64_t expected_shapes_transposed[][2] = {
        {4096, 768},  // linear1.weight: transposed shape (in_features, out_features)
        {768, 0},     // linear1.bias (1D)
        {768, 768},   // linear2.weight: same for square matrix
        {768, 0}      // linear2.bias (1D)
    };
    
    // Track whether weights need transposition
    bool need_transpose[2] = {false, false};  // [linear1, linear2]
    
    for (int i = 0; i < 4; i++) {
        int64_t tensor_idx = gguf_find_tensor(ctx_gguf, projector_names[i]);
        if (tensor_idx >= 0) {
            struct ggml_tensor * tensor = ggml_get_tensor(ctx_meta, projector_names[i]);
            if (tensor) {
                int64_t dim0 = tensor->ne[0];
                int64_t dim1 = (ggml_n_dims(tensor) > 1) ? tensor->ne[1] : 0;
                
                if (i % 2 == 0) {  // weight (2D)
                    // Check if stored as PyTorch shape (out_features, in_features) or transposed
                    bool is_pytorch_shape = (dim0 == expected_shapes_pytorch[i][0] && 
                                           dim1 == expected_shapes_pytorch[i][1]);
                    bool is_transposed_shape = (dim0 == expected_shapes_transposed[i][0] && 
                                               dim1 == expected_shapes_transposed[i][1]);
                    
                    if (is_pytorch_shape) {
                        // Stored as PyTorch shape (out_features, in_features), need to transpose
                        need_transpose[i / 2] = true;
                    } else if (is_transposed_shape) {
                        // Already transposed, use directly
                        need_transpose[i / 2] = false;
                    } else {
                        LOG_ERR("TTS: %s has unexpected shape: [%ld, %ld], expected [%ld, %ld] or [%ld, %ld]\n",
                               projector_names[i], dim0, dim1, 
                               expected_shapes_pytorch[i][0], expected_shapes_pytorch[i][1],
                               expected_shapes_transposed[i][0], expected_shapes_transposed[i][1]);
                        // Try to continue, assume PyTorch shape
                        need_transpose[i / 2] = true;
                    }
                } else {  // bias (1D)
                    if (dim0 != expected_shapes_pytorch[i][0] || dim1 != 0) {
                        LOG_ERR("TTS: %s has wrong shape: [%ld, %ld], expected [%ld, 0]\n",
                               projector_names[i], dim0, dim1, expected_shapes_pytorch[i][0]);
                    }
                }
                
                // Check tensor type for F16 conversion
                enum ggml_type proj_tensor_type = tensor->type;
                
                if (i % 2 == 0 && need_transpose[i / 2]) {
                    // Weight needs transposition: allocate transposed size (always F32 output)
                    int64_t in_dim = expected_shapes_pytorch[i][1];  // PyTorch in_features
                    int64_t out_dim = expected_shapes_pytorch[i][0];  // PyTorch out_features
                    size_t transposed_size = in_dim * out_dim * sizeof(float);
                    *projector_ptrs[i] = (float *)malloc(transposed_size);
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
                        const float * src_data = (const float *)tensor->data;
                        for (int64_t out = 0; out < out_dim; out++) {
                            for (int64_t in = 0; in < in_dim; in++) {
                                dst_data[in * out_dim + out] = src_data[out * in_dim + in];
                            }
                        }
                    } else if (proj_tensor_type == GGML_TYPE_F16) {
                        const ggml_fp16_t * src_data = (const ggml_fp16_t *)tensor->data;
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
                    size_t output_size = num_elements * sizeof(float);
                    *projector_ptrs[i] = (float *)malloc(output_size);
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
                        const ggml_fp16_t * src_f16 = (const ggml_fp16_t *)tensor->data;
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
    ctx_omni->projector_semantic_input_dim = 4096;
    ctx_omni->projector_semantic_output_dim = 768;
    
    // Load head_code.weight: (hidden_size=768, num_audio_tokens=6562)
    // Note: num_vq=1, so we only load head_code.0.weight
    const char * head_code_name = "head_code.0.weight";
    int64_t head_code_idx = gguf_find_tensor(ctx_gguf, head_code_name);
    if (head_code_idx >= 0) {
        struct ggml_tensor * head_code_tensor = ggml_get_tensor(ctx_meta, head_code_name);
        if (head_code_tensor) {
            // head_code is Linear(hidden_size, num_audio_tokens, bias=False)
            // In PyTorch: weight shape is (num_audio_tokens, hidden_size) = [6562, 768]
            // In GGUF: stored as (hidden_size, num_audio_tokens) = [768, 6562] (already transposed)
            int64_t dim0 = head_code_tensor->ne[0];
            int64_t dim1 = (ggml_n_dims(head_code_tensor) > 1) ? head_code_tensor->ne[1] : 0;
            
            // Expected shape in GGUF: (hidden_size=768, num_audio_tokens=6562)
            int64_t expected_hidden_size = 768;
            int64_t expected_num_audio_tokens = 6562;
            
            // Allocate memory for weight: (hidden_size, num_audio_tokens) = [768, 6562]
            size_t head_code_size = expected_hidden_size * expected_num_audio_tokens * sizeof(float);
            ctx_omni->head_code_weight = (float *)malloc(head_code_size);
            if (!ctx_omni->head_code_weight) {
                LOG_ERR("TTS: Failed to allocate memory for head_code.0.weight\n");
                // Clean up already loaded weights
                if (ctx_omni->emb_text_weight) {
                    free(ctx_omni->emb_text_weight);
                    ctx_omni->emb_text_weight = nullptr;
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
            
            const float * src_data = (const float *)head_code_tensor->data;
            bool need_transpose = false;
            
            // CRITICAL FIX: Based on conversion script analysis, the script always transposes
            // head_code to [768, 6562] before saving. So the actual data is always [768, 6562],
            // regardless of metadata shape. We should NOT transpose based on metadata.
            //
            // However, if metadata says [6562, 768], it might be an old conversion that didn't transpose.
            // We'll use a simple heuristic: if metadata says [6562, 768], assume data needs transpose.
            // If metadata says [768, 6562], assume data is already correct.
            
            if (dim0 == expected_hidden_size && dim1 == expected_num_audio_tokens) {
                // Metadata says (768, 6562) - data should already be in correct format
                need_transpose = false;
            } else if (dim0 == expected_num_audio_tokens && dim1 == expected_hidden_size) {
                // Metadata says (6562, 768) - but conversion script may have already transposed the data
                // We need to check: if conversion script transposed, data is actually [768, 6562] and we should NOT transpose
                // If conversion script didn't transpose, data is [6562, 768] and we SHOULD transpose
                //
                // Since the conversion script ALWAYS transposes (line 351: W_transposed = W.T),
                // the data is always [768, 6562] regardless of metadata.
                // So we should NOT transpose.
                need_transpose = false;  // CRITICAL FIX: Don't transpose, data is already [768, 6562]
            } else {
                LOG_ERR("TTS: head_code.0.weight has unexpected shape [%ld, %ld], expected [%d, %d] or [%d, %d]\n",
                       dim0, dim1, expected_hidden_size, expected_num_audio_tokens, expected_num_audio_tokens, expected_hidden_size);
                // Clean up
                free(ctx_omni->head_code_weight);
                ctx_omni->head_code_weight = nullptr;
                if (ctx_omni->emb_text_weight) {
                    free(ctx_omni->emb_text_weight);
                    ctx_omni->emb_text_weight = nullptr;
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
            int64_t total_elements = expected_hidden_size * expected_num_audio_tokens;
            
            print_with_timestamp("TTS: head_code shape: dim0=%ld, dim1=%ld, will transpose to [%ld, %ld]\n", 
                                dim0, dim1, expected_num_audio_tokens, expected_hidden_size);
            
            if (tensor_type == GGML_TYPE_F32) {
                // F32: copy with transpose from [768, 6562] to [6562, 768]
                for (int64_t i = 0; i < expected_num_audio_tokens; ++i) {
                    for (int64_t j = 0; j < expected_hidden_size; ++j) {
                        // src: [j * 6562 + i], dst: [i * 768 + j]
                        ctx_omni->head_code_weight[i * expected_hidden_size + j] = src_data[j * expected_num_audio_tokens + i];
                    }
                }
            } else if (tensor_type == GGML_TYPE_F16) {
                // F16: convert to F32 with transpose
                const ggml_fp16_t * src_f16 = (const ggml_fp16_t *)src_data;
                for (int64_t i = 0; i < expected_num_audio_tokens; ++i) {
                    for (int64_t j = 0; j < expected_hidden_size; ++j) {
                        ctx_omni->head_code_weight[i * expected_hidden_size + j] = 
                            ggml_fp16_to_fp32(src_f16[j * expected_num_audio_tokens + i]);
                    }
                }
            } else {
                LOG_ERR("TTS: head_code.0.weight has unsupported type: %d\n", tensor_type);
                free(ctx_omni->head_code_weight);
                ctx_omni->head_code_weight = nullptr;
                ggml_free(ctx_meta);
                gguf_free(ctx_gguf);
                return false;
            }
            
            ctx_omni->head_code_hidden_size = expected_hidden_size;
            ctx_omni->head_code_num_audio_tokens = expected_num_audio_tokens;
            
            // 🔍 调试：验证加载的数据
            // Python weight[0, 0:5] = [0.01385498, -0.01647949, 0.0111084, -0.01367188, -0.01141357]
            // C++ head_code_weight[0..4] 应该和 Python 一致
            print_with_timestamp("TTS: head_code loaded, verifying first few values:\n");
            print_with_timestamp("  head_code_weight[0] = %.8f (expect ~0.01385498)\n", ctx_omni->head_code_weight[0]);
            print_with_timestamp("  head_code_weight[1] = %.8f (expect ~-0.01647949)\n", ctx_omni->head_code_weight[1]);
            print_with_timestamp("  head_code_weight[768] = %.8f (this is weight[1, 0], expect ~0.04150391)\n", ctx_omni->head_code_weight[768]);
        } else {
            LOG_ERR("TTS: Failed to get tensor %s from GGUF context\n", head_code_name);
            // Clean up
            if (ctx_omni->emb_text_weight) {
                free(ctx_omni->emb_text_weight);
                ctx_omni->emb_text_weight = nullptr;
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
        if (ctx_omni->emb_text_weight) {
            free(ctx_omni->emb_text_weight);
            ctx_omni->emb_text_weight = nullptr;
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

static bool eval_tokens_tts(struct omni_context* ctx_omni, common_params* params, std::vector<llama_token> tokens, int n_batch, int * n_past_tts) {
    fflush(stdout);
    int N = (int) tokens.size();
    fflush(stdout);
    // Note: TTS model might need different KV cache management
    // For now, we'll use a simple approach similar to LLM
    fflush(stdout);
    fflush(stdout);
    for (int i = 0; i < N; i += n_batch) {
        fflush(stdout);
        int n_eval = (int) tokens.size() - i;
        fflush(stdout);
        if (n_eval > n_batch) {
            n_eval = n_batch;
            fflush(stdout);
        }
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
            LOG_ERR("%s : failed to eval TTS tokens. token %d/%d (batch size %d, n_past %d), decode_ret=%d\n", 
                    __func__, i, N, n_batch, *n_past_tts, decode_ret);
            return false;
        }
        *n_past_tts += n_eval;
    }
    return true;
}

static bool eval_string_tts(struct omni_context * ctx_omni, common_params* params, const char* str, int n_batch, int * n_past_tts, bool add_bos) {
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_omni->ctx_tts_llama, str2, add_bos, true);
    return eval_tokens_tts(ctx_omni, params, embd_inp, n_batch, n_past_tts);
}

// 使用embedding作为TTS输入的prefill函数（类似prefill_with_emb，但针对TTS模型）
bool prefill_with_emb_tts(struct omni_context* ctx_omni, common_params* params, float* embed, int n_pos, int n_batch, int* n_past_tts) {
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
        ctx_omni->tts_condition_saved = true;
    }
    
    // Save the starting position before the loop
    int text_start_pos = *n_past_tts;
    
    // Check if we need to save all hidden states (for alignment testing)
    const char* save_hidden_states_dir = getenv("TTS_SAVE_HIDDEN_STATES_DIR");
    bool save_all_hidden_states = (save_hidden_states_dir != nullptr);
    
    for (int i = 0; i < n_pos; i += n_batch) {
        int n_eval = n_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        
        llama_batch batch = {};
        batch.n_tokens = int32_t(n_eval);
        batch.embd = (embed + i * n_embd);  // 使用embedding作为输入
        
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
            LOG_ERR("%s : failed to eval TTS embeddings. pos %d/%d (batch size %d, n_past %d)\n", 
                    __func__, i, n_pos, n_batch, *n_past_tts);
            llama_set_embeddings(ctx_omni->ctx_tts_llama, false);
            return false;
        }
        
        // Save hidden states for each token in the batch (for alignment testing)
        // Note: llama_get_embeddings_ith uses negative indices relative to the end of the batch
        // For a batch of n_eval tokens: -1 is last, -2 is second-to-last, ..., -n_eval is first
        if (save_all_hidden_states) {
            for (int j = 0; j < n_eval; j++) {
                int token_idx = text_start_pos + i + j;
                // Get j-th token in current batch: j=0 -> -n_eval (first), j=n_eval-1 -> -1 (last)
                int llama_idx = j - n_eval;
                const float* hidden_state = llama_get_embeddings_ith(ctx_omni->ctx_tts_llama, llama_idx);
                if (hidden_state) {
                    char filepath[512];
                    snprintf(filepath, sizeof(filepath), "%s/hidden_states_%03d.bin", save_hidden_states_dir, token_idx);
                    FILE* f = fopen(filepath, "wb");
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

// Simple UTF-8 incomplete byte checker
// Returns the number of incomplete bytes at the end of the string
// Returns 0 if the string ends with a complete UTF-8 sequence
static size_t findIncompleteUtf8(const std::string& str) {
    if (str.empty()) return 0;
    
    size_t len = str.length();
    
    // Check from the end backwards to find incomplete UTF-8 sequences
    size_t pos = len;
    int expected_continuation_bytes = 0;
    
    while (pos > 0) {
        unsigned char c = (unsigned char)str[pos - 1];
        
        if ((c & 0x80) == 0) {
            // ASCII character (0xxxxxxx), complete sequence
            break;
        } else if ((c & 0xC0) == 0x80) {
            // Continuation byte (10xxxxxx), part of a multi-byte sequence
            expected_continuation_bytes++;
            pos--;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence start (110xxxxx)
            // Should have exactly 1 continuation byte after it
            if (expected_continuation_bytes == 1) {
                // Complete 2-byte sequence
                break;
            } else {
                // Incomplete: missing continuation byte(s)
                return len - pos + 1;  // Return incomplete bytes including the start byte
            }
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence start (1110xxxx)
            // Should have exactly 2 continuation bytes after it
            if (expected_continuation_bytes == 2) {
                // Complete 3-byte sequence
                break;
            } else {
                // Incomplete: missing continuation byte(s)
                return len - pos + (3 - expected_continuation_bytes);
            }
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence start (11110xxx)
            // Should have exactly 3 continuation bytes after it
            if (expected_continuation_bytes == 3) {
                // Complete 4-byte sequence
                break;
            } else {
                // Incomplete: missing continuation byte(s)
                return len - pos + (4 - expected_continuation_bytes);
            }
        } else {
            // Invalid UTF-8 byte pattern (should not happen in valid UTF-8)
            // Treat as complete to avoid breaking valid text
            break;
        }
    }
    
    // If we've checked all bytes and still have expected continuation bytes,
    // the sequence is incomplete
    if (pos == 0 && expected_continuation_bytes > 0) {
        return len;
    }
    
    return 0;  // String ends with complete UTF-8 sequence
}

//
// omni main
//
void print_with_timestamp(const char* format, ...)
{
    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    // 格式化时间戳
    std::tm buf;
#ifdef _WIN32
    localtime_s(&buf, &in_time_t);
#else
    localtime_r(&in_time_t, &buf);
#endif
    std::cout << std::put_time(&buf, "%H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count() << " ";
    
    // 打印格式化字符串
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

static struct llama_model * llama_init(common_params * params, std::string model_path) {
    llama_backend_init();
    llama_numa_init(params->numa);
    
    llama_model_params model_params = common_model_params_to_llama(*params);
    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n" , __func__);
        return NULL;
    }
    return model;
}

// TTS专用模型加载 - 支持独立的GPU层数设置
// 通过环境变量 TTS_GPU_LAYERS 控制，-1 表示使用与LLM相同的设置
static struct llama_model * llama_init_tts(common_params * params, std::string model_path, int n_gpu_layers_override = -1) {
    llama_backend_init();
    llama_numa_init(params->numa);
    
    llama_model_params model_params = common_model_params_to_llama(*params);
    
    // 如果指定了override值(>=0)，使用它；否则保持与LLM相同的设置
    if (n_gpu_layers_override >= 0) {
        model_params.n_gpu_layers = n_gpu_layers_override;
    }
    
    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load TTS model\n" , __func__);
        return NULL;
    }
    return model;
}

struct omni_context * omni_init(struct common_params * params, int media_type, bool use_tts, std::string tts_bin_dir,
                                int tts_gpu_layers, const std::string & token2wav_device, bool duplex_mode,
                                llama_model * existing_model, llama_context * existing_ctx,
                                const std::string & base_output_dir) {
    // process the prompt
    print_with_timestamp("=== omni_init start\n");
    // if (params->prompt.empty() && params->interactive == false) {
    //     LOG_INF("prompt should be given or interactive mode should be on");
    //     return NULL;
    // }
    // auto ctx_omni = (struct omni_context *)malloc(sizeof(omni_context));
    auto ctx_omni = new omni_context();

    ctx_omni->params = params;
    ctx_omni->media_type = media_type;
    ctx_omni->use_tts = use_tts;
    ctx_omni->duplex_mode = duplex_mode;
    omni_session_sync_round_meta(ctx_omni);
    ctx_omni->base_output_dir = base_output_dir;  // 🔧 [多实例支持] 设置可配置的输出目录
    print_with_timestamp("media_type = %d, duplex_mode = %d, base_output_dir = %s\n", media_type, duplex_mode, base_output_dir.c_str());
    // 🔧 [对齐 Python MiniCPM-o-4_5-latest] prompt 格式
    // Python default_tts_chat_template:
    //   {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}
    //   {% if add_generation_prompt %}{{ '<|im_start|>assistant\n' + think_str + '<|tts_bos|>' }}{% endif %}
    // 
    // Python audio_assistant 模式的 system prompt:
    //   vc_prompt_prefix = "模仿音频样本的音色并生成新的内容。"
    //   vc_prompt_suffix = "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
    //   sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
    // 
    // 完整格式 (audio_assistant 模式):
    //   <|im_start|>system
    //   模仿音频样本的音色并生成新的内容。
    //   <|audio_start|>[ref_audio_embed]<|audio_end|>你的任务是用这种声音模式来当一个助手。...
    //   <|im_end|>
    //   <|im_start|>user
    //   <|audio_start|>[user_audio_embed]<|audio_end|>
    //   <|im_end|>
    //   <|im_start|>assistant
    //   <think>
    //   
    //   </think>
    //   
    //   <|tts_bos|>
    // 
    // 注意: voice_clone_prompt 是 system prompt 的 prefix，assistant_prompt 是 system prompt 的 suffix
    //       stream_decode 会添加实际的 assistant generation prompt
    if (duplex_mode) {
        // 🔧 [与 Python 对齐] Audio 双工模式：嵌入参考音频
        // 双工模式不需要 <|im_start|>user\n，用 <unit> 标记用户输入
        ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|><|im_end|>\n";
        
        // 🔧 [修复] Omni 双工模式：也需要嵌入参考音频，格式与 Audio 双工相同
        ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|><|im_end|>\n";
    } else {
        // 🔧 [与 Python 对齐] 非双工模式 Audio 格式 (audio_assistant 模式)
        // 格式: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n
        // 🔧 [整合] 在 sys prompt 末尾直接添加 <|im_start|>user\n，不再在 stream_prefill 里动态添加
        // 这样更稳妥，不依赖 Python 端的 counter 重置
        ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。<|im_end|>\n<|im_start|>user\n";
        
        // Omni 模式（非双工）：与 Audio 模式类似，末尾也添加 <|im_start|>user\n
        ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。<|im_end|>\n<|im_start|>user\n";
    }

    llama_model * model = nullptr;
    llama_context * ctx_llama = nullptr;
    
    // 支持模型复用（单工模式常用）
    if (existing_model != nullptr && existing_ctx != nullptr) {
        print_with_timestamp("=== omni_init: reusing existing LLM model and context\n");
        model = existing_model;
        ctx_llama = existing_ctx;
        ctx_omni->owns_model = false;  // 不拥有模型，omni_free 时不释放
        
        // 🔧 [模式切换修复] 清理 LLM 的 KV cache，避免位置冲突
        // 当从一个模式切换到另一个模式时，需要清理旧的 KV cache
        llama_memory_t mem = llama_get_memory(ctx_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);  // 清除 sequence 0 的所有 KV cache
            print_with_timestamp("=== omni_init: cleared LLM KV cache for mode switch\n");
        }
    } else {
        // 加载新模型
        print_with_timestamp("=== omni_init: loading new LLM model\n");
        model = llama_init(params, params->model.path);
        if (model == NULL) {
            return NULL;
        }
        llama_context_params ctx_params = common_context_params_to_llama(*params);
        ctx_params.n_ctx                = params->n_ctx;
        
        ctx_llama = llama_new_context_with_model(model, ctx_params);
        if (ctx_llama == NULL) {
            LOG_ERR("%s: error: failed to create the llama_context\n" , __func__);
            return NULL;
        }
        ctx_omni->owns_model = true;  // 拥有模型，omni_free 时需要释放
    }
    
    struct common_sampler * sampler = common_sampler_init(model, params->sampling);
    ctx_omni->ctx_llama = ctx_llama;
    ctx_omni->model = model;
    ctx_omni->ctx_sampler = sampler;

    if (use_tts && !params->tts_model.empty()) {
        print_with_timestamp("=== omni_init: loading TTS model\n");
        // 使用TTS专用的模型加载函数，支持独立的GPU层数设置
        // tts_gpu_layers 从 omni_init 参数传入，-1 表示使用与LLM相同的设置
        print_with_timestamp("TTS model: loading with n_gpu_layers=%d\n", tts_gpu_layers);
        llama_model * tts_model = llama_init_tts(params, params->tts_model, tts_gpu_layers);
        if (tts_model == NULL) {
            LOG_ERR("%s: error: failed to init TTS model from %s\n", __func__, params->tts_model.c_str());
            llama_free(ctx_llama);
            llama_free_model(model);
            common_sampler_free(sampler);
            delete ctx_omni;
            return NULL;
        }
        
        // TTS 模型使用独立的上下文参数
        // 注意：TTS 模型可能需要不同的上下文大小和批处理大小
        llama_context_params tts_ctx_params = common_context_params_to_llama(*params);
        // 如果 TTS 模型需要更小的上下文窗口，可以在这里调整
        // 例如：tts_ctx_params.n_ctx = std::min(params->n_ctx, 2048); // 限制 TTS 上下文大小
        tts_ctx_params.n_ctx = params->n_ctx;  // 暂时使用相同的 n_ctx，后续可以根据需要调整
        
        llama_context * ctx_tts_llama = llama_new_context_with_model(tts_model, tts_ctx_params);
        if (ctx_tts_llama == NULL) {
            LOG_ERR("%s: error: failed to create the TTS llama_context\n", __func__);
            llama_free_model(tts_model);
            // 清理已分配的资源
            llama_free(ctx_llama);
            llama_free_model(model);
            common_sampler_free(sampler);
            delete ctx_omni;
            return NULL;
        }
        
        // 创建 TTS 的采样器
        // 🔧 TTS流式采样参数 - 与 Python ras_sampling 对齐：
        // Python TTSSamplingParams 默认 temperature=0.8 (modeling_minicpmo.py line 75)
        common_params_sampling tts_sampling = params->sampling;
        tts_sampling.temp = 0.8f;              // 🔧 [与 Python 对齐] TTSSamplingParams.temperature=0.8
        tts_sampling.top_p = 0.85f;  // 🔧 [与 Python 对齐] TTSSamplingParams.top_p=0.85             // 🔧 [与 Python streaming 对齐] top_p=0.8
        tts_sampling.top_k = 25;               // top_k = 25 (ras_sampling 参数)
        tts_sampling.penalty_repeat = 1.05f;   // repetition_penalty = 1.05
        tts_sampling.min_p = 0.01f;            // min_p = 0.01
        // Python: CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, num_code, 16)
        tts_sampling.penalty_last_n = 16;      // past_window = 16 (与Python对齐)
        struct common_sampler * tts_sampler = common_sampler_init(tts_model, tts_sampling);
        print_with_timestamp("TTS sampler: temp=%.2f, top_p=%.2f, top_k=%d, rep_penalty=%.2f\n",
                            tts_sampling.temp, tts_sampling.top_p, tts_sampling.top_k, tts_sampling.penalty_repeat);
        
        ctx_omni->model_tts = tts_model;
        ctx_omni->ctx_tts_llama = ctx_tts_llama;
        ctx_omni->ctx_tts_sampler = tts_sampler;
        
        // Load TTS weights from GGUF file
        print_with_timestamp("TTS: loading weights from GGUF (emb_code, emb_text, projector_semantic, head_code)...\n");
        if (!load_tts_weights_from_gguf(ctx_omni, params->tts_model.c_str())) {
            LOG_ERR("%s: error: failed to load TTS weights from %s\n", __func__, params->tts_model.c_str());
            llama_free(ctx_tts_llama);
            llama_free_model(tts_model);
            common_sampler_free(tts_sampler);
            llama_free(ctx_llama);
            llama_free_model(model);
            common_sampler_free(sampler);
            delete ctx_omni;
            return NULL;
        }
        print_with_timestamp("TTS: weights loaded successfully\n");
        
        // Load Projector Semantic from GGUF file
        // 路径: {tts_bin_dir}/MiniCPM-o-4_5-projector-F16.gguf
        std::string projector_path = tts_bin_dir + "/MiniCPM-o-4_5-projector-F16.gguf";
        print_with_timestamp("Projector: loading from %s\n", projector_path.c_str());
        if (projector_init(ctx_omni->projector, projector_path, true)) {
            print_with_timestamp("Projector: loaded successfully\n");
        } else {
            print_with_timestamp("Projector: failed to load, will use fallback implementation\n");
        }
    }

    ctx_omni->omni_emb.resize((64 + 10 + 1) * 4096); // temp fix for omni embed
    ctx_omni->audio_emb.resize((10 + 1) * 4096); // temp fix for audio embed
    print_with_timestamp("=== omni_init: loading APM model\n");
    if (params->apm_model.empty()) {
        LOG_ERR("%s: error: apm_model path is empty\n", __func__);
        if (ctx_omni->use_tts) {
            llama_free(ctx_omni->ctx_tts_llama);
            llama_free_model(ctx_omni->model_tts);
            common_sampler_free(ctx_omni->ctx_tts_sampler);
        }
        llama_free(ctx_llama);
        llama_free_model(model);
        common_sampler_free(sampler);
        delete ctx_omni;
        return NULL;
    }
    ctx_omni->ctx_audio = audition_init(params->apm_model.c_str(), audition_context_params{true, GGML_LOG_LEVEL_INFO});
    print_with_timestamp("APM: init from %s\n", params->apm_model.c_str());
    if (ctx_omni->ctx_audio == nullptr) {
        LOG_ERR("%s: error: failed to init audition model from %s\n", __func__, params->apm_model.c_str());
        // 清理 TTS 模型资源（如果已加载）
        if (ctx_omni->use_tts) {
            llama_free(ctx_omni->ctx_tts_llama);
            llama_free_model(ctx_omni->model_tts);
            common_sampler_free(ctx_omni->ctx_tts_sampler);
        }
        llama_free(ctx_llama);
        llama_free_model(model);
        common_sampler_free(sampler);
        delete ctx_omni;
        return NULL;
    }

    ctx_omni->n_past = 0;
    
    if (media_type == 2) {
        LOG_INF("init vision....");
        const char * vision_path = ctx_omni->params->vpm_model.c_str();
        auto * ctx_vision = vision_init(vision_path, vision_context_params{true, GGML_LOG_LEVEL_INFO, nullptr});
        ctx_omni->ctx_vision = ctx_vision;

        // Set CoreML model path if available (for vision ANE acceleration)
        // Note: .mlmodelc is a directory, not a file, so use stat instead of ifstream
        if (ctx_vision && !ctx_omni->params->vision_coreml_model_path.empty()) {
            struct stat coreml_stat;
            if (stat(ctx_omni->params->vision_coreml_model_path.c_str(), &coreml_stat) == 0) {
                vision_set_coreml_model_path(ctx_vision, ctx_omni->params->vision_coreml_model_path.c_str());
                LOG_INF("Vision CoreML model path set to: %s\n", ctx_omni->params->vision_coreml_model_path.c_str());
            } else {
                LOG_WRN("Vision CoreML model path does not exist: %s, skipping ANE\n", ctx_omni->params->vision_coreml_model_path.c_str());
            }
        }
    }
    
    ctx_omni->llm_thread_info = new LLMThreadInfo(1000);
    if (ctx_omni->use_tts) {
        LOG_INF("init tts....");
        ctx_omni->tts_thread_info = new TTSThreadInfo(1);
        ctx_omni->omni_output = new omni_output();
        ctx_omni->tts_bin_dir = tts_bin_dir;
        
        // Initialize T2W thread info
        LOG_INF("init t2w....");
        ctx_omni->t2w_thread_info = new T2WThreadInfo(25);  // Queue size of 10 chunks
        
        // Initialize C++ Token2Wav session
        // Try to load token2wav GGUF models from {model_dir}/token2wav-gguf/
        // Fallback to tools/omni/token2wav-gguf if not found
        ctx_omni->token2wav_initialized = false;
        
        // 🔧 如果使用 Python Token2Wav，跳过 C++ 的初始化以节省显存
        bool skip_cpp_token2wav = ctx_omni->use_python_token2wav;
        
        // Check if token2wav model files exist
        // 优先检查 HF 模型目录下的 token2wav-gguf (tts_bin_dir 的父目录)
        // 目录结构: {model_dir}/token2wav-gguf/
        std::string gguf_root_dir = tts_bin_dir;
        size_t last_slash = gguf_root_dir.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            gguf_root_dir = gguf_root_dir.substr(0, last_slash);  // 获取 tts 的父目录
        }
        ctx_omni->token2wav_model_dir = gguf_root_dir + "/token2wav-gguf";
        
        std::string encoder_test = ctx_omni->token2wav_model_dir + "/encoder.gguf";
        {
            std::ifstream f(encoder_test);
            if (!f.good()) {
                // 尝试备用路径 (本地开发用)
                ctx_omni->token2wav_model_dir = "tools/omni/token2wav-gguf";
                print_with_timestamp("Token2Wav: trying fallback path %s\n", ctx_omni->token2wav_model_dir.c_str());
            } else {
                print_with_timestamp("Token2Wav: found models in %s\n", ctx_omni->token2wav_model_dir.c_str());
            }
        }
        std::string encoder_gguf = ctx_omni->token2wav_model_dir + "/encoder.gguf";
        std::string flow_matching_gguf = ctx_omni->token2wav_model_dir + "/flow_matching.gguf";
        std::string flow_extra_gguf = ctx_omni->token2wav_model_dir + "/flow_extra.gguf";
        std::string vocoder_gguf = ctx_omni->token2wav_model_dir + "/hifigan2.gguf";
        std::string prompt_cache_gguf = ctx_omni->token2wav_model_dir + "/prompt_cache.gguf";
        
        // Check if all files exist
        bool all_files_exist = true;
        std::vector<std::string> gguf_files = {encoder_gguf, flow_matching_gguf, flow_extra_gguf, vocoder_gguf};
        for (const auto& file : gguf_files) {
            std::ifstream f(file);
            if (!f.good()) {
                all_files_exist = false;
                break;
            } else {
            }
        }
        
        if (all_files_exist && !skip_cpp_token2wav) {
            print_with_timestamp("Token2Wav: all model files found, initializing session...\n");
            ctx_omni->token2wav_session = std::make_unique<omni::flow::Token2WavSession>();
            
            // Device configuration - 使用 omni_init 传入的 token2wav_device 参数
            // 格式: "gpu", "gpu:0", "gpu:1", "cpu"
            std::string device_token2mel = token2wav_device;

            // Vocoder 设备策略：
            //   CUDA: vocoder 跟随 token2wav_device（GPU），因为 CUDA kernel launch 开销低
            //   Metal (macOS): vocoder 强制用 CPU，因为 vocoder 有大量小操作，
            //     Metal kernel dispatch 开销累积后远慢于 CPU 直接计算
            //   可通过 OMNI_VOC_DEVICE 环境变量覆盖
            const char * voc_dev_env = getenv("OMNI_VOC_DEVICE");
            std::string device_vocoder;
            if (voc_dev_env) {
                device_vocoder = voc_dev_env;
                print_with_timestamp("Token2Wav: vocoder device overridden by OMNI_VOC_DEVICE=%s\n", voc_dev_env);
            } else {
#ifdef GGML_USE_CUDA
                device_vocoder = token2wav_device;
                print_with_timestamp("Token2Wav: CUDA detected, vocoder using GPU (%s)\n", device_vocoder.c_str());
#else
                device_vocoder = "cpu";
                print_with_timestamp("Token2Wav: non-CUDA backend, vocoder using CPU for better performance\n");
#endif
            }
            
            // 🔧 优先使用 prompt_bundle (setup_cache 路径)，否则 fallback 到 prompt_cache.gguf
            std::string prompt_bundle_dir = "tools/omni/assets/default_ref_audio";
            std::string spk_file = prompt_bundle_dir + "/spk_f32.bin";
            std::string tokens_file = prompt_bundle_dir + "/prompt_tokens_i32.bin";
            std::string mel_file = prompt_bundle_dir + "/prompt_mel_btc_f32.bin";
            
            bool use_prompt_bundle = false;
            {
                std::ifstream f1(spk_file), f2(tokens_file), f3(mel_file);
                use_prompt_bundle = f1.good() && f2.good() && f3.good();
            }
            
            bool init_ok = false;
            // 优先级: prompt_cache.gguf > prompt_bundle (实时计算 fallback)
            print_with_timestamp("Token2Wav: using prompt_cache from %s\n", prompt_cache_gguf.c_str());
            init_ok = ctx_omni->token2wav_session->init_from_prompt_cache_gguf(
                    encoder_gguf, flow_matching_gguf, flow_extra_gguf, prompt_cache_gguf,
                    vocoder_gguf, device_token2mel, device_vocoder, 5, 1.0f);
            if (!init_ok && use_prompt_bundle) {
                print_with_timestamp("Token2Wav: prompt_cache failed, fallback to prompt_bundle from %s\n", prompt_bundle_dir.c_str());
                init_ok = ctx_omni->token2wav_session->init_from_prompt_bundle(
                        encoder_gguf, flow_matching_gguf, flow_extra_gguf, prompt_bundle_dir,
                        vocoder_gguf, device_token2mel, device_vocoder, 5, 1.0f);
            }
            // Fallback to CPU
            if (!init_ok) {
                print_with_timestamp("Token2Wav: GPU init failed, trying CPU mode...\n");
                ctx_omni->token2wav_session.reset();
                ctx_omni->token2wav_session = std::make_unique<omni::flow::Token2WavSession>();
                init_ok = ctx_omni->token2wav_session->init_from_prompt_cache_gguf(
                        encoder_gguf, flow_matching_gguf, flow_extra_gguf, prompt_cache_gguf,
                        vocoder_gguf, "cpu", "cpu", 5, 1.0f);
            }
            
            if (init_ok) {
                ctx_omni->token2wav_initialized = true;
                // Initialize token2wav buffer with 3 silence tokens (4218) as Python does
                // Python: buffer = [4218] * 3  # 预先放入3个前缀静音token
                ctx_omni->token2wav_buffer.clear();
                ctx_omni->token2wav_buffer = {4218, 4218, 4218};
                ctx_omni->token2wav_wav_idx = 0;
                print_with_timestamp("Token2Wav: initialized successfully\n");
            } else {
                ctx_omni->token2wav_session.reset();
                print_with_timestamp("Token2Wav: initialization failed\n");
            }
        } else {
            print_with_timestamp("Token2Wav: model files not found in %s\n", ctx_omni->token2wav_model_dir.c_str());
        }
        
        // ==================== 初始化 Python Token2Wav ====================
        // 🔧 默认使用 Python Token2Wav（精度更高）
        // 设置 Python T2W 脚本目录和模型目录
        // Python T2W 脚本目录：tools/omni/pyt2w/
        // Python T2W 模型目录：dependencies/token2wav/
        
        // 计算 Python T2W 脚本目录（相对于 tts_bin_dir）
        // tts_bin_dir 通常是 /xxx/tools/omni/convert/gguf/token2wav-gguf
        // 我们需要 /xxx/tools/omni/pyt2w
        std::string t2w_script_dir = tts_bin_dir;  // /xxx/tools/omni/convert/gguf/token2wav-gguf
        // 回退到 tools/omni/
        size_t convert_pos = t2w_script_dir.find("/convert/gguf/tts");
        if (convert_pos != std::string::npos) {
            t2w_script_dir = t2w_script_dir.substr(0, convert_pos) + "/pyt2w";
        } else if ((convert_pos = t2w_script_dir.find("/convert/gguf")) != std::string::npos) {
            t2w_script_dir = t2w_script_dir.substr(0, convert_pos) + "/pyt2w";
        } else {
            // 尝试从当前工作目录构建
            t2w_script_dir = "./tools/omni/pyt2w";
        }
        ctx_omni->python_t2w_script_dir = t2w_script_dir;
        
        // Python T2W 模型目录（stepaudio2 模型）
        // 默认路径：相对于 script_dir 的 token2wav 子目录
        ctx_omni->python_t2w_model_dir = t2w_script_dir + "/token2wav";
        
        // 参考音频路径
        std::string ref_audio_path = "tools/omni/assets/default_ref_audio/default_ref_audio.wav";
        
        print_with_timestamp("Python T2W: script_dir=%s, model_dir=%s\n", 
                             ctx_omni->python_t2w_script_dir.c_str(),
                             ctx_omni->python_t2w_model_dir.c_str());
        
        if (ctx_omni->use_python_token2wav) {
            print_with_timestamp("Python T2W: 使用 Python Token2Wav 实现\n");
            
            // 🔧 Python T2W GPU 配置
            // C++ LLM+TTS 占用约 22GB，Python T2W 占用约 3.3GB
            // 单卡 24GB 放不下，需要配置独立 GPU
            // 
            // 通过环境变量 PYTHON_T2W_GPU 配置独立 GPU
            // 例如: export PYTHON_T2W_GPU=1  # Python T2W 使用 GPU 1
            // 
            // 优先级：PYTHON_T2W_GPU 环境变量 > 外部 CUDA_VISIBLE_DEVICES > token2wav_device
            const char* env_python_t2w_gpu = getenv("PYTHON_T2W_GPU");
            if (env_python_t2w_gpu && strlen(env_python_t2w_gpu) > 0) {
                ctx_omni->python_t2w_dedicated_gpu = env_python_t2w_gpu;
            }
            
            ctx_omni->python_t2w_gpu_id = "";
            
            if (!ctx_omni->python_t2w_dedicated_gpu.empty()) {
                // 使用配置的独立 GPU
                ctx_omni->python_t2w_gpu_id = ctx_omni->python_t2w_dedicated_gpu;
                print_with_timestamp("Python T2W: 使用独立 GPU %s (C++ 和 Python 分开)\n", ctx_omni->python_t2w_gpu_id.c_str());
            } else {
                const char* env_cuda_visible = getenv("CUDA_VISIBLE_DEVICES");
                if (env_cuda_visible && strlen(env_cuda_visible) > 0) {
                    // 外部已设置，Python 子进程会继承，不需要额外设置
                    print_with_timestamp("Python T2W: 继承外部 CUDA_VISIBLE_DEVICES=%s (与 C++ 共用)\n", env_cuda_visible);
                } else if (token2wav_device.find("gpu") != std::string::npos) {
                    // 外部未设置，从 token2wav_device 提取
                    size_t colon_pos = token2wav_device.find(':');
                    if (colon_pos != std::string::npos) {
                        ctx_omni->python_t2w_gpu_id = token2wav_device.substr(colon_pos + 1);
                    } else {
                        ctx_omni->python_t2w_gpu_id = "0";
                    }
                    print_with_timestamp("Python T2W: 设置 CUDA_VISIBLE_DEVICES=%s (与 C++ 共用)\n", ctx_omni->python_t2w_gpu_id.c_str());
                } else {
                    print_with_timestamp("Python T2W: CPU 模式\n");
                }
            }
            
            // 启动 Python 服务
            if (omni_start_python_t2w_service(ctx_omni)) {
                // 初始化模型
                if (omni_init_python_t2w_model(ctx_omni, token2wav_device)) {
                    // 设置参考音频
                    if (omni_set_python_t2w_ref_audio(ctx_omni, ref_audio_path)) {
                        print_with_timestamp("Python T2W: 初始化成功\n");
                    } else {
                        print_with_timestamp("Python T2W: 设置参考音频失败\n");
                        ctx_omni->use_python_token2wav = false;
                    }
                } else {
                    print_with_timestamp("Python T2W: 初始化模型失败\n");
                    ctx_omni->use_python_token2wav = false;
                }
            } else {
                print_with_timestamp("Python T2W: 启动服务失败\n");
                ctx_omni->use_python_token2wav = false;
            }
            
            // 如果 Python 初始化失败，回退到 C++ 实现
            if (!ctx_omni->use_python_token2wav) {
                print_with_timestamp("Python T2W: 回退到 C++ 实现\n");
            }
        } else {
            print_with_timestamp("Token2Wav: 使用 C++ 实现\n");
        }
    }
    ctx_omni->async = true;
    
    omni_init_token_protocol(ctx_omni);
        
    // ANE/CoreML warmup: pre-load models into NPU to avoid first-inference latency
    omni_warmup_ane(ctx_omni);

    print_with_timestamp("=== omni_init success: ctx_llama = %p\n", (void*)ctx_omni->ctx_llama);
    return ctx_omni;
}

//
// ANE/CoreML warmup — pre-load models into NPU to avoid first-inference latency
//
void omni_warmup_ane(struct omni_context * ctx_omni) {
#if defined(__APPLE__)
    if (!ctx_omni) return;

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
    (void)ctx_omni;
#endif
}

bool omni_tts_queues_empty(struct omni_context * ctx_omni) {
    bool tts_empty = true, t2w_empty = true;
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

bool omni_get_duplex_chunk_timing(struct omni_context * ctx_omni, int chunk_idx, struct omni_duplex_chunk_timing * out_timing) {
    if (ctx_omni == nullptr || out_timing == nullptr || chunk_idx < 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(ctx_omni->duplex_timing_mtx);
    auto it = ctx_omni->duplex_chunk_timings.find(chunk_idx);
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
    omni_request_worker_shutdown(ctx_omni);
    
    // 等待 llm 和 tts thread 停止
    ctx_omni->workers.llm_thread_running = false; // Signal the thread to stop
    if (ctx_omni->llm_thread.joinable()) {
        ctx_omni->llm_thread_info->cv.notify_all(); // Wake up the thread if it's waiting
        ctx_omni->llm_thread.join(); // Wait for the thread to finish
    }
 
    if (ctx_omni->use_tts) {
        ctx_omni->workers.tts_thread_running = false; // Signal the thread to stop
        if (ctx_omni->tts_thread.joinable()) {
            ctx_omni->tts_thread_info->cv.notify_all(); // Wake up the thread if it's waiting
            ctx_omni->tts_thread.join(); // Wait for the thread to finish
        }
        
        // Stop T2W thread
        ctx_omni->workers.t2w_thread_running = false; // Signal the thread to stop
        if (ctx_omni->t2w_thread.joinable()) {
            ctx_omni->t2w_thread_info->cv.notify_all(); // Wake up the thread if it's waiting
            ctx_omni->t2w_thread.join(); // Wait for the thread to finish
        }
    }
    
    delete ctx_omni->ctx_vision;
    audition_free(ctx_omni->ctx_audio);
    
    if (ctx_omni->use_tts) {
        llama_free(ctx_omni->ctx_tts_llama);
        llama_free_model(ctx_omni->model_tts);
        common_sampler_free(ctx_omni->ctx_tts_sampler);
        
        // Free TTS weights
        if (ctx_omni->emb_code_weight) {
            free(ctx_omni->emb_code_weight);
            ctx_omni->emb_code_weight = nullptr;
        }
        if (ctx_omni->emb_text_weight) {
            free(ctx_omni->emb_text_weight);
            ctx_omni->emb_text_weight = nullptr;
        }
        if (ctx_omni->projector_semantic_linear1_weight) {
            free(ctx_omni->projector_semantic_linear1_weight);
            ctx_omni->projector_semantic_linear1_weight = nullptr;
        }
        if (ctx_omni->projector_semantic_linear1_bias) {
            free(ctx_omni->projector_semantic_linear1_bias);
            ctx_omni->projector_semantic_linear1_bias = nullptr;
        }
        if (ctx_omni->projector_semantic_linear2_weight) {
            free(ctx_omni->projector_semantic_linear2_weight);
            ctx_omni->projector_semantic_linear2_weight = nullptr;
        }
        if (ctx_omni->projector_semantic_linear2_bias) {
            free(ctx_omni->projector_semantic_linear2_bias);
            ctx_omni->projector_semantic_linear2_bias = nullptr;
        }
        if (ctx_omni->head_code_weight) {
            free(ctx_omni->head_code_weight);
            ctx_omni->head_code_weight = nullptr;
        }
        
        // Free C++ Token2Wav session
        if (ctx_omni->token2wav_session) {
            ctx_omni->token2wav_session.reset();
            ctx_omni->token2wav_initialized = false;
            LOG_INF("Token2Wav (C++): session released\n");
        }
        
        // 🔧 停止 Python Token2Wav 服务
        if (ctx_omni->python_t2w_initialized) {
            omni_stop_python_t2w_service(ctx_omni);
        }
        
        // Free ggml-based projector model
        if (ctx_omni->projector.initialized) {
            projector_free(ctx_omni->projector);
        }
    }
    
    // 🔧 [单双工适配] 只有在拥有模型时才释放 LLM model 和 context
    // 如果是外部传入的模型（模型复用），则不释放
    if (ctx_omni->owns_model) {
        llama_free(ctx_omni->ctx_llama);
        llama_free_model(ctx_omni->model);
    }
    common_sampler_free(ctx_omni->ctx_sampler);
    // delete ctx_omni->ctx_tts;
    delete ctx_omni->llm_thread_info;
    delete ctx_omni->audio_input_manager;
    
    // omni_output 还要把里面 output 的每个元素也 delete 下
    if (ctx_omni->use_tts) {
        delete ctx_omni->tts_thread_info;
        delete ctx_omni->t2w_thread_info;
        
        // omni_output 还要把里面 output 的每个元素也 delete 下
        if (ctx_omni->omni_output) {
            for (auto &buffer : ctx_omni->omni_output->output) {
                delete buffer;
            }
            ctx_omni->omni_output->output.clear(); // Clear the vector
            delete ctx_omni->omni_output;
        }
    }

    llama_backend_free();

    delete ctx_omni;
}

// ==================== 语言设置函数 ====================
// 设置语言并更新 system prompt（zh=中文，en=英文）
// 基于 Python MiniCPM-o-4_5 modeling_minicpmo.py 中的 audio_assistant 模式 prompt
void omni_set_language(struct omni_context * ctx_omni, const std::string & lang) {
    if (ctx_omni == nullptr) {
        print_with_timestamp("omni_set_language: ctx_omni is null\n");
        return;
    }
    
    ctx_omni->language = lang;
    print_with_timestamp("omni_set_language: setting language to '%s'\n", lang.c_str());
    
    if (ctx_omni->duplex_mode) {
        // 双工模式：prompt 固定使用英文（与 Python 对齐）
        ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|><|im_end|>\n";
        ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|><|im_end|>\n";
    } else {
        // 非双工模式（audio_assistant 模式）：根据语言设置 prompt
        if (lang == "en") {
            // 英文 prompt（来自 Python modeling_minicpmo.py）
            ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>";
            ctx_omni->audio_assistant_prompt = "<|audio_end|>Please assist users while maintaining this voice style. Please answer the user's questions seriously and in a high quality. Please chat with the user in a highly human-like and oral style. You are a helpful assistant developed by ModelBest: MiniCPM-Omni.<|im_end|>\n<|im_start|>user\n";
            
            ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>";
            ctx_omni->omni_assistant_prompt = "<|audio_end|>Please assist users while maintaining this voice style. Please answer the user's questions seriously and in a high quality. Please chat with the user in a highly human-like and oral style.<|im_end|>\n<|im_start|>user\n";
        } else {
            // 中文 prompt（默认，来自 Python modeling_minicpmo.py）
            ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
            ctx_omni->audio_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。<|im_end|>\n<|im_start|>user\n";
            
            ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
            ctx_omni->omni_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。<|im_end|>\n<|im_start|>user\n";
        }
    }
    
    // 🔧 [关键] 重置 system_prompt_initialized，让下次 stream_prefill(index=0) 重新 prefill system prompt
    ctx_omni->system_prompt_initialized = false;
    
    print_with_timestamp("omni_set_language: prompts updated for language '%s', system_prompt_initialized reset to false\n", lang.c_str());
}

static void process_audio(struct omni_context * ctx_omni, struct omni_embed * embeds, common_params * params, bool save_spk_emb=false) {
    LOG_INF("%s: audio token past: %d\n", __func__, ctx_omni->n_past);
    omni_eval_embed(ctx_omni->ctx_llama, embeds, params->n_batch, &ctx_omni->n_past);
    LOG_INF("%s: audio token past after eval: %d\n", __func__, ctx_omni->n_past);
}

void eval_prefix(struct omni_context* ctx_omni, common_params* params){
    std::string prefix = "<|im_start|>user\n";
    std::cout << "prefix : " << prefix << std::endl;
    eval_string(ctx_omni, params, prefix.c_str(), params->n_batch, &ctx_omni->n_past, false);
}

void eval_prefix_with_hidden(struct omni_context* ctx_omni, common_params* params, float *& hidden_states){
    std::string prefix = "<|im_start|>user\n";
    std::cout << "prefix : " << prefix << std::endl;
    eval_string_with_hidden(ctx_omni, params, prefix.c_str(), params->n_batch, &ctx_omni->n_past, false, hidden_states);
}

struct OmniBootstrapPrompts {
    std::string voice_clone_prompt;
    std::string assistant_prompt;
};

static std::string omni_normalize_prefill_prompt(const std::string & prompt) {
    if (prompt.rfind("<|", 0) == 0) {
        return prompt;
    }
    return "<|im_start|>user\n" + prompt;
}

static OmniBootstrapPrompts omni_select_bootstrap_prompts(const struct omni_context * ctx_omni) {
    const bool use_omni_prompt = ctx_omni->media_type == 2;
    const std::string & raw_voice_clone_prompt = use_omni_prompt
        ? ctx_omni->omni_voice_clone_prompt
        : ctx_omni->audio_voice_clone_prompt;
    const std::string & raw_assistant_prompt = use_omni_prompt
        ? ctx_omni->omni_assistant_prompt
        : ctx_omni->audio_assistant_prompt;

    return {
        omni_normalize_prefill_prompt(raw_voice_clone_prompt),
        omni_normalize_prefill_prompt(raw_assistant_prompt),
    };
}

static std::string omni_get_bootstrap_ref_audio_path(const struct omni_context * ctx_omni, const std::string & aud_fname) {
    if (ctx_omni->duplex_mode && !aud_fname.empty()) {
        return aud_fname;
    }
    return ctx_omni->ref_audio_path.empty()
        ? "tools/omni/assets/default_ref_audio/default_ref_audio.wav"
        : ctx_omni->ref_audio_path;
}

// Step B: bootstrap the session once, including system prompt and ref-audio prefill.
static bool omni_run_session_bootstrap_if_needed(
        struct omni_context * ctx_omni,
        const OmniPrefillSetup & setup,
        const OmniBootstrapPrompts & prompts,
        const std::string & aud_fname,
        int index) {
    if (!setup.need_bootstrap) {
        return true;
    }

    print_with_timestamp(
        "stream_prefill: n_past = %d\n voice_clone_prompt = %s\n assistant_prompt = %s\n",
        ctx_omni->n_past,
        prompts.voice_clone_prompt.c_str(),
        prompts.assistant_prompt.c_str());

    const std::string bootstrap_ref_audio = omni_get_bootstrap_ref_audio_path(ctx_omni, aud_fname);
    if (!ctx_omni->duplex_mode || bootstrap_ref_audio != aud_fname) {
        print_with_timestamp("system prompt ref_audio: %s\n", bootstrap_ref_audio.c_str());
    }

    eval_string(ctx_omni, ctx_omni->params, prompts.voice_clone_prompt.c_str(),
                ctx_omni->params->n_batch, &ctx_omni->n_past, false);

    auto ref_audio_embed_start = std::chrono::high_resolution_clock::now();
    auto * ref_audio_embeds = omni_audio_embed_make_with_filename(
        ctx_omni->ctx_audio,
        ctx_omni->params->cpuparams.n_threads,
        bootstrap_ref_audio);
    duplex_timing_note_audio(
        ctx_omni,
        index,
        timing_elapsed_ms(ref_audio_embed_start, std::chrono::high_resolution_clock::now()));
    if (ref_audio_embeds != nullptr && ref_audio_embeds->n_pos > 0) {
        print_with_timestamp("system prompt ref_audio embedding: n_pos=%d\n", ref_audio_embeds->n_pos);
        prefill_with_emb(
            ctx_omni,
            ctx_omni->params,
            ref_audio_embeds->embed,
            ref_audio_embeds->n_pos,
            ctx_omni->params->n_batch,
            &ctx_omni->n_past);
        omni_embed_free(ref_audio_embeds);
    } else {
        print_with_timestamp("WARNING: failed to load system prompt ref_audio: %s\n", bootstrap_ref_audio.c_str());
    }

    eval_string(ctx_omni, ctx_omni->params, prompts.assistant_prompt.c_str(),
                ctx_omni->params->n_batch, &ctx_omni->n_past, false);

    ctx_omni->system_prompt_initialized = true;
    ctx_omni->n_keep = ctx_omni->n_past;
    print_with_timestamp("🔒 n_keep 设置为 %d (system prompt tokens)，这部分永远不会被滑动窗口删除\n", ctx_omni->n_keep);
    eval_prefix(ctx_omni, ctx_omni->params);

    print_with_timestamp("stream_prefill(index=0): system prompt 初始化完成，ref_audio 已在其中 prefill\n");
    sliding_window_register_system_prompt(ctx_omni);
    print_with_timestamp("n_past = %d\n", ctx_omni->n_past);

    if (setup.should_start_workers) {
        const OmniWorkerThreadFns worker_fns = {
            llm_thread_func,
            tts_thread_func,
            tts_thread_func_duplex,
            t2w_thread_func,
        };
        omni_ensure_prefill_workers_started(ctx_omni, worker_fns);
    }

    return true;
}

// Step C: encode image/audio input into a single prefill payload.
static bool omni_encode_prefill_input(
        struct omni_context * ctx_omni,
        const std::string & aud_fname,
        const std::string & img_fname,
        int index,
        int max_slice_nums,
        struct omni_embeds & encoded) {
    const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    encoded.index = index;

    if (!img_fname.empty()) {
        if (max_slice_nums >= 1 && ctx_omni->ctx_vision != nullptr) {
            vision_set_max_slice_nums(ctx_omni->ctx_vision, max_slice_nums);
            LOG_INF("%s: [临时] max_slice_nums=%d for this prefill\n", __func__, max_slice_nums);
        }

        auto vit_embed_start = std::chrono::high_resolution_clock::now();
        const bool image_embed_ok = omni_image_embed_make_chunks_with_filename(
            ctx_omni->ctx_vision,
            ctx_omni->params->cpuparams.n_threads,
            img_fname,
            encoded.vision_embed);
        duplex_timing_note_vit(
            ctx_omni,
            index,
            timing_elapsed_ms(vit_embed_start, std::chrono::high_resolution_clock::now()));
        if (!image_embed_ok) {
            LOG_ERR("%s: failed to create vision embeddings for %s\n", __func__, img_fname.c_str());
            return false;
        }
        LOG_INF("%s: vision_embed has %d chunks\n", __func__, (int) encoded.vision_embed.size());
    }

    if (!aud_fname.empty()) {
        print_with_timestamp("stream_prefill(index=%d): processing user audio: %s\n", index, aud_fname.c_str());
        auto audio_embed_start = std::chrono::high_resolution_clock::now();
        auto * audio_embeds = omni_audio_embed_make_with_filename(
            ctx_omni->ctx_audio,
            ctx_omni->params->cpuparams.n_threads,
            aud_fname);
        duplex_timing_note_audio(
            ctx_omni,
            index,
            timing_elapsed_ms(audio_embed_start, std::chrono::high_resolution_clock::now()));
        if (audio_embeds != nullptr && audio_embeds->n_pos > 0) {
            print_with_timestamp("stream_prefill(index=%d): user audio embedding: n_pos=%d\n", index, audio_embeds->n_pos);
            encoded.audio_embed.resize(audio_embeds->n_pos * hidden_size);
            std::memcpy(encoded.audio_embed.data(), audio_embeds->embed, encoded.audio_embed.size() * sizeof(float));
            omni_embed_free(audio_embeds);
        } else {
            LOG_WRN("%s: audio encoding failed, skipping audio for this frame: %s\n", __func__, aud_fname.c_str());
        }
    }

    return true;
}

// Shared LLM stage apply path used by both sync stream_prefill and llm_thread_func.
static void omni_llm_stage_prefill_apply(
        struct omni_context * ctx_omni,
        struct common_params * params,
        const struct omni_embeds & embeds) {
    const int hidden_size = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));

    if (ctx_omni->sliding_window_config.mode != "off") {
        sliding_window_register_unit_start(ctx_omni);
    }

    if (!embeds.vision_embed.empty()) {
        const int n_chunks = (int) embeds.vision_embed.size();
        const int tokens_per_chunk = (int) embeds.vision_embed[0].size() / hidden_size;
        const int n_audio_tokens = embeds.audio_embed.size() / hidden_size;
        const bool has_audio = n_audio_tokens > 0;
        const bool has_slices = n_chunks > 1;

        if (ctx_omni->duplex_mode) {
            eval_string(ctx_omni, params, "<unit><image>", params->n_batch, &ctx_omni->n_past, false);
        } else {
            eval_string(ctx_omni, params, "<image>", params->n_batch, &ctx_omni->n_past, false);
        }

        prefill_with_emb(ctx_omni, params, const_cast<float *>(embeds.vision_embed[0].data()), tokens_per_chunk,
                        params->n_batch, &ctx_omni->n_past);
        eval_string(ctx_omni, params, "</image>", params->n_batch, &ctx_omni->n_past, false);

        if (has_slices) {
            for (int i = 1; i < n_chunks; ++i) {
                eval_string(ctx_omni, params, "<slice>", params->n_batch, &ctx_omni->n_past, false);
                prefill_with_emb(ctx_omni, params, const_cast<float *>(embeds.vision_embed[i].data()), tokens_per_chunk,
                                params->n_batch, &ctx_omni->n_past);
                eval_string(ctx_omni, params, "</slice>", params->n_batch, &ctx_omni->n_past, false);
            }
            eval_string(ctx_omni, params, "\n", params->n_batch, &ctx_omni->n_past, false);
        }

        print_with_timestamp("Omni模式: %d vision chunks (%d tokens each), %d audio tokens, has_slices=%d\n",
                            n_chunks, tokens_per_chunk, n_audio_tokens, has_slices);

        if (has_audio) {
            if (!ctx_omni->duplex_mode) {
                eval_string(ctx_omni, params, "<|audio_start|>", params->n_batch, &ctx_omni->n_past, false);
            }
            prefill_with_emb(ctx_omni, params, const_cast<float *>(embeds.audio_embed.data()), n_audio_tokens,
                            params->n_batch, &ctx_omni->n_past);
            if (!ctx_omni->duplex_mode) {
                eval_string(ctx_omni, params, "<|audio_end|>", params->n_batch, &ctx_omni->n_past, false);
            }
        }
    } else {
        const int n_audio_tokens = embeds.audio_embed.size() / hidden_size;
        print_with_timestamp("用户语音: %d audio tokens\n", n_audio_tokens);

        if (ctx_omni->duplex_mode) {
            eval_string(ctx_omni, params, "<unit>", params->n_batch, &ctx_omni->n_past, false);
        } else {
            eval_string(ctx_omni, params, "<|audio_start|>", params->n_batch, &ctx_omni->n_past, false);
        }

        if (n_audio_tokens > 0) {
            prefill_with_emb(ctx_omni, params, const_cast<float *>(embeds.audio_embed.data()), n_audio_tokens,
                            params->n_batch, &ctx_omni->n_past);
        }

        if (!ctx_omni->duplex_mode) {
            eval_string(ctx_omni, params, "<|audio_end|>", params->n_batch, &ctx_omni->n_past, false);
        }
    }

    if (ctx_omni->sliding_window_config.mode != "off") {
        const std::string input_type = embeds.vision_embed.empty() ? "audio" : "omni";
        sliding_window_register_unit_end(ctx_omni, input_type, {}, false);
    }
}

static void omni_finalize_llm_prefill(struct omni_context * ctx_omni) {
    if (ctx_omni->sliding_window_config.mode != "off") {
        sliding_window_enforce(ctx_omni);
    }
}

// Step D: submit the encoded payload to the shared LLM stage.
static bool omni_submit_llm_prefill(struct omni_context * ctx_omni, std::unique_ptr<struct omni_embeds> encoded) {
    if (!ctx_omni->async) {
        omni_llm_stage_prefill_apply(ctx_omni, ctx_omni->params, *encoded);
        omni_finalize_llm_prefill(ctx_omni);
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

/**
 * LLM线程函数：负责处理多模态（视觉+音频）嵌入的前缀填充（prefill）
 * 
 * 这个函数在一个独立线程中运行，主要职责是：
 * 1. 从队列中获取视觉和音频嵌入数据
 * 2. 将嵌入数据组合成LLM可以处理的格式
 * 3. 执行前缀填充，为后续的文本生成做准备
 * 4. 协调与解码线程的同步
 * 
 * 运行逻辑：
 * - 主循环持续运行，直到 llm_thread_running 为 false
 * - 等待条件：队列不为空 OR need_speek 为 true OR 线程需要停止
 * - 两个主要分支：
 *   分支1：队列不为空 -> 处理嵌入数据的前缀填充
 *   分支2：队列为空且 need_speek 为 true -> 通知解码线程可以开始生成
 */
void llm_thread_func(omni_context* ctx_omni, common_params* params){
    print_with_timestamp("LLM thread started\n");

    // ========== 主循环：持续处理嵌入数据 ==========
    while(ctx_omni->workers.llm_thread_running){
        // 获取队列的互斥锁，保护共享资源
        std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
        auto& queue = ctx_omni->llm_thread_info->queue;

        // 打印当前状态（调试用）
        
        // ========== 等待条件满足 ==========
        // 等待以下任一条件满足：
        // 1. 队列不为空（有新嵌入数据需要处理）
        // 2. need_speek 为 true（需要开始生成文本）
        // 3. llm_thread_running 为 false（线程需要停止）
        ctx_omni->llm_thread_info->cv.wait(lock, [&] { 
            return !queue.empty() || ctx_omni->need_speek || !ctx_omni->workers.llm_thread_running; 
        });
        
        // 检查是否需要停止线程
        if (!ctx_omni->workers.llm_thread_running) {
            break;
        }

        // ========== 分支1：处理队列中的嵌入数据（前缀填充） ==========
        if (!queue.empty()){
            // 🔧 [诊断] 打印 prefill 开始时的 n_past
            print_with_timestamp("LLM thread: start prefill, n_past=%d, n_keep=%d, n_ctx=%d\n",
                                 ctx_omni->n_past, ctx_omni->n_keep, params->n_ctx);
            
            // 🔧 [修复] prefill 阶段不清除 KV cache，保持累积
            // 无论单工还是双工模式，prefill 都是累积用户输入
            // KV cache 只在以下情况清除：
            //   1. 新会话开始（通过 reset API）
            //   2. 滑动窗口触发（context 满了）
            print_with_timestamp("LLM thread: prefill continuing, n_past=%d (no KV cache clear)\n", ctx_omni->n_past);
            
            // 标记前缀填充未完成，防止解码线程过早开始
            omni_mark_prefill_started(ctx_omni);
            
            // 步骤1：批量取出队列中的所有嵌入数据
            // 这样可以一次性处理多个嵌入，提高效率
            std::vector<omni_embeds*> llm_embeds;
            llm_embeds.clear();
            while (!queue.empty()) {
                llm_embeds.push_back(queue.front());
                queue.pop();
            }
            
            // 释放锁，允许其他线程继续向队列添加数据
            lock.unlock();
            
            // 如果批量处理多个嵌入，打印日志
            print_with_timestamp("Batch processing %zu llm prefill\n", llm_embeds.size());

            // 通知等待的生产者线程，队列有空间了
            ctx_omni->llm_thread_info->cv.notify_all();

            // 🔧 [与 Python 对齐] 只有非双工模式才添加 <|im_start|>user\n
            // 双工模式: 直接用 <unit> 标记用户输入开始，不需要 <|im_start|>user\n
            // 非双工模式: <|im_start|>system....<|im_end|>\n<|im_start|>user\n<|audio_start|>audio<|audio_end|><|im_end|>\n<|im_start|>assistant...
            // 🔧 [整合] <|im_start|>user\n 已在 sys prompt 末尾添加（第一轮），
            // 后续轮次在 stream_decode 结束时添加
            // 不再需要在这里动态添加

            // 🔧 [重构] 逐个处理嵌入数据，正确添加特殊标记
            // 遍历所有嵌入数据
            for (int il = 0; il < (int)llm_embeds.size(); ++il) {
                auto embeds = llm_embeds[il];
                omni_llm_stage_prefill_apply(ctx_omni, params, *embeds);
                // 释放嵌入数据的内存（由生产者线程分配）
                delete embeds;
            }
            
            // 🔧 [诊断] 打印 prefill 结束后的 n_past
            print_with_timestamp("LLM thread: prefill done, n_past=%d, n_keep=%d, 本次消耗 %d tokens, duplex_mode=%d\n",
                                 ctx_omni->n_past, ctx_omni->n_keep, 
                                 ctx_omni->n_past - ctx_omni->n_keep,
                                 ctx_omni->duplex_mode);
            
            omni_finalize_llm_prefill(ctx_omni);
        }
        
        // ========== 分支2：队列为空且需要开始生成文本 ==========
        // 这个分支在以下情况触发：
        // 1. 所有嵌入数据都已处理完成（队列为空）
        // 2. 解码线程设置了 need_speek = true，表示需要开始生成文本

        if (queue.empty() && ctx_omni->need_speek){
            // 标记前缀填充完成
            omni_mark_prefill_completed(ctx_omni);
            
            // 如果使用TTS，重置speek_done标志，允许TTS线程开始工作
            if (ctx_omni->use_tts && !ctx_omni->duplex_mode) {
                ctx_omni->speek_done = false;
            }
            
            // 重置need_speek标志
            ctx_omni->need_speek = false;
            
            // 通知等待的解码线程：前缀填充已完成，可以开始生成文本了
        }
    }
}

// Helper function to play WAV file
static void play_wav_file(const std::string& wav_file_path) {
#ifndef _WIN32
    // Play audio asynchronously using fork() to avoid blocking TTS thread
    pid_t pid = fork();
    if (pid == 0) {
        #ifdef __APPLE__
            execl("/usr/bin/afplay", "afplay", wav_file_path.c_str(), (char*)NULL);
        #else
            execl("/usr/bin/aplay", "aplay", wav_file_path.c_str(), (char*)NULL);
        #endif
        _exit(1);
    } else if (pid > 0) {
        // Parent process: continue without waiting
    } else {
        std::string play_cmd;
        #ifdef __APPLE__
            play_cmd = "afplay \"" + wav_file_path + "\" &";
        #else
            play_cmd = "aplay \"" + wav_file_path + "\" &";
        #endif
        LOG_WRN("TTS: fork() failed, using system() fallback for audio playback\n");
        system(play_cmd.c_str());
    }
#endif
    // Windows: no-op (audio playback handled by frontend)
}


// Helper function to move old output directory to old_output/<id>/
static void move_old_output_to_archive() {
    const std::string base_output_dir = "./tools/omni/output";
    const std::string old_output_base_dir = "./old_output";
    
    // Helper function to check if directory exists and has content
    auto dir_has_content = [](const std::string& dir_path) -> bool {
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
    auto create_dir = [](const std::string& dir_path) -> bool {
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0) {
            // Directory doesn't exist, try to create it
            if (!cross_platform_mkdir_p(dir_path)) {
                LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
                return false;
            }
            return true;
        } else if (!(info.st_mode & S_IFDIR)) {
            LOG_ERR("Output path exists but is not a directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    };
    
    // Helper function to find next available ID in old_output directory
    auto get_next_output_id = [](const std::string& old_output_base) -> int {
        // Ensure old_output base directory exists
        cross_platform_mkdir_p(old_output_base);
        
        // Find maximum ID in old_output directory
        int max_id = -1;
#ifdef _WIN32
        std::string find_cmd = "dir /b \"" + old_output_base + "\" 2>NUL";
#else
        std::string find_cmd = "ls -1 " + old_output_base + " 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1";
#endif
        FILE* pipe = popen(find_cmd.c_str(), "r");
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
                        if (id > max_id) max_id = id;
                    } catch (...) {}
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
        if (dir_has_content(base_output_dir + "/llm_debug") || 
            dir_has_content(base_output_dir + "/tts_txt") ||
            dir_has_content(base_output_dir + "/tts_wav")) {
            output_has_content = true;
        }
    }
    
    if (output_has_content) {
        int next_id = get_next_output_id(old_output_base_dir);
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
            std::string move_cmd = "find " + base_output_dir + " -mindepth 1 -maxdepth 1 -exec mv {} " + old_output_dir + "/ \\; 2>/dev/null";
            int ret = system(move_cmd.c_str());
            if (ret == 0) {
            } else {
                // Fallback: try simple mv command
                std::string fallback_cmd = "sh -c 'cd " + base_output_dir + " && mv * " + old_output_dir + "/ 2>/dev/null || true'";
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
// ==============================================================================
void tts_thread_func_duplex(struct omni_context * ctx_omni, common_params *params) {
    // TTS model state
    int tts_n_past = 0;
    std::vector<llama_token> audio_tokens;
    std::vector<llama_token> all_audio_tokens;
    std::string debug_dir = "";
    bool tts_finish = false;
    bool llm_finish = false;
    int chunk_idx = 0;
    std::string incomplete_bytes;
    
    // 双工模式：固定输出目录（不使用 round_XXX 子目录）
    // 🔧 [多实例支持] 使用可配置的 base_output_dir
    const std::string& base_output_dir = ctx_omni->base_output_dir;
    const std::string tts_output_dir = base_output_dir + "/tts_txt";
    const std::string llm_debug_output_dir = base_output_dir + "/llm_debug";
    const std::string tts_wav_output_dir = base_output_dir + "/tts_wav";
    OmniRoundMeta active_round_meta = omni_session_round_meta(ctx_omni);
    
    // Helper function to create directory
    auto create_dir = [](const std::string& dir_path) {
        if (!cross_platform_mkdir_p(dir_path)) {
            LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    };
    
    // 创建输出目录
    create_dir(tts_output_dir);
    create_dir(llm_debug_output_dir);
    create_dir(tts_wav_output_dir);
    
    // 🔧 [修复重复生成问题] 标志位：当前 turn 是否已经执行过 turn_eos flush
    bool turn_eos_flushed = false;

    print_with_timestamp("TTS thread (duplex mode) started\n");

    // Multi Round Persistent Loop
    while(ctx_omni->workers.tts_thread_running) {
        if (!ctx_omni->workers.tts_thread_running) {
            break;
        }
        
        // 🔧 [双工模式] 打断检测
        if (ctx_omni->break_event.load()) {
            // 清空 TTS 队列
            omni_clear_tts_queue(ctx_omni);
            // 重置状态
            llm_finish = false;
            tts_finish = false;
            chunk_idx = 0;
            tts_n_past = 0;
            audio_tokens.clear();
            all_audio_tokens.clear();
            incomplete_bytes.clear();
            // 清除 TTS KV cache
            llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
            if (mem) {
                llama_memory_seq_rm(mem, 0, 0, -1);
                print_with_timestamp("TTS Duplex: break_event - cleared TTS KV cache\n");
            }
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            continue;
        }

        std::string llm_text = "";
        std::vector<llama_token> current_chunk_token_ids;
        std::vector<float> current_chunk_hidden_states;
        int current_chunk_n_embd = 0;
        int current_chunk_perf_idx = -1;
        
        // 🔧 [修复双工缺字问题] 从 LLMOut 获取 is_end_of_turn 状态
        bool accumulated_is_end_of_turn = false;
            
        // Wait for queue
        if (!llm_finish || (llm_finish && llm_text.empty())) {
            std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
            auto& queue = ctx_omni->tts_thread_info->queue;
            ctx_omni->tts_thread_info->cv.wait(lock, [&] { 
                return !queue.empty() || !ctx_omni->workers.tts_thread_running || ctx_omni->break_event.load(); 
            });
            
            if (ctx_omni->break_event.load()) {
                lock.unlock();
                continue;
            }
                
            if (!ctx_omni->workers.tts_thread_running) {
                break;
            }
            
            // 清空 current_chunk 数据
            current_chunk_token_ids.clear();
            current_chunk_hidden_states.clear();
            current_chunk_n_embd = 0;
            accumulated_is_end_of_turn = false;
            current_chunk_perf_idx = -1;
            
            // 累积所有队列中的数据
            while (!queue.empty()) {
                LLMOut *llm_out = queue.front();
                if (ctx_omni->duplex_mode &&
                    current_chunk_perf_idx >= 0 &&
                    llm_out->duplex_chunk_idx >= 0 &&
                    llm_out->duplex_chunk_idx != current_chunk_perf_idx) {
                    break;
                }
                if (current_chunk_perf_idx < 0 && llm_out->duplex_chunk_idx >= 0) {
                    current_chunk_perf_idx = llm_out->duplex_chunk_idx;
                }
                queue.pop();
                llm_finish |= llm_out->llm_finish;
                accumulated_is_end_of_turn |= llm_out->is_end_of_turn;
                active_round_meta = llm_out->round_meta;
                
                if (!ctx_omni->speek_done || ctx_omni->duplex_mode) {
                    llm_text += llm_out->text;
                    debug_dir = llm_out->debug_dir;
                }
                
                // 累积数据
                if (!llm_out->token_ids.empty() && !llm_out->hidden_states.empty()) {
                    current_chunk_token_ids.insert(current_chunk_token_ids.end(), 
                                                   llm_out->token_ids.begin(), 
                                                   llm_out->token_ids.end());
                    current_chunk_hidden_states.insert(current_chunk_hidden_states.end(), 
                                                       llm_out->hidden_states.begin(), 
                                                       llm_out->hidden_states.end());
                    current_chunk_n_embd = llm_out->n_embd;
                }
                delete llm_out;
            }
            lock.unlock();
            ctx_omni->tts_thread_info->cv.notify_all();
            
            // 双工模式：如果有新数据，继续处理
            // 🔧 [关键诊断] 每次取出 LLMOut 后都打印状态
            // 🔧 [修复双工缺字问题] 使用 accumulated_is_end_of_turn 而非全局状态
            print_with_timestamp("TTS Duplex: after queue - speek_done=%d, llm_finish=%d, token_ids.size=%zu, is_end_of_turn=%d, llm_text.len=%zu\n",
                                ctx_omni->speek_done ? 1 : 0,
                                llm_finish ? 1 : 0,
                                current_chunk_token_ids.size(),
                                accumulated_is_end_of_turn ? 1 : 0,
                                llm_text.size());
            
            if (ctx_omni->speek_done && llm_finish) {
                if (ctx_omni->duplex_mode && !current_chunk_token_ids.empty()) {
                    ctx_omni->speek_done = false;
                } else if (ctx_omni->duplex_mode && accumulated_is_end_of_turn) {
                    ctx_omni->speek_done = false;
                    print_with_timestamp("TTS Duplex: is_end_of_turn=true, will call TTS to flush buffer\n");
                } else {
                    // LISTEN/CHUNK_EOS 且没有实际文本
                    llm_finish = false;
                    llm_text.clear();
                    
                    if (ctx_omni->t2w_thread_info) {
                        T2WOut *t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final = false;
                        t2w_out->is_chunk_end = true;
                        t2w_out->round_meta = active_round_meta;
                        t2w_out->duplex_chunk_idx = current_chunk_perf_idx;
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    duplex_timing_mark_tts_done(ctx_omni, current_chunk_perf_idx);
                    continue;
                }
            }
        }
            
        std::string& response = llm_text;
        // 处理不完整的 UTF-8 字节
        if (!incomplete_bytes.empty()) {
            response = incomplete_bytes + response;
            incomplete_bytes.clear();
        }
        size_t incomplete_len = findIncompleteUtf8(response);
        if (incomplete_len > 0) {
            incomplete_bytes = response.substr(response.size() - incomplete_len, incomplete_len);
            response = response.substr(0, response.size() - incomplete_len);
        }
        
        // Skip empty responses
        if (response.empty() && !llm_finish) {
            if (ctx_omni->speek_done) {
                llm_finish = false;
                if (ctx_omni->duplex_mode && !accumulated_is_end_of_turn) {
                    // 保持状态
                } else {
                    // 轮次结束，重置 TTS 状态
                    chunk_idx = 0;
                    tts_n_past = 0;
                    audio_tokens.clear();
                    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                    if (mem) {
                        llama_memory_seq_rm(mem, 0, 0, -1);
                    }
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                }
            }
            continue;
        }
            
        // Tokenize text input
        std::vector<llama_token> text_tokens = common_tokenize(ctx_omni->ctx_tts_llama, response, false, true);
        
        if (text_tokens.empty() && !llm_finish) {
            continue;
        }
        
        // Handle empty final chunk
        if (text_tokens.empty() && response.empty() && llm_finish) {
            ctx_omni->speek_done = true;
            ctx_omni->warmup_done = true;
            ctx_omni->workers.speek_cv.notify_all();
            
            if (ctx_omni->duplex_mode && !accumulated_is_end_of_turn) {
                // LISTEN/CHUNK_EOS
                if (ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens.clear();
                    t2w_out->is_final = false;
                    t2w_out->is_chunk_end = true;
                    t2w_out->round_meta = active_round_meta;
                    t2w_out->duplex_chunk_idx = current_chunk_perf_idx;
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                }
                duplex_timing_mark_tts_done(ctx_omni, current_chunk_perf_idx);
                tts_finish = false;
                llm_finish = false;
                continue;
            } else if (ctx_omni->duplex_mode && accumulated_is_end_of_turn) {
                // turn 结束，继续调用 TTS flush buffer
                print_with_timestamp("TTS Duplex: empty final chunk but is_end_of_turn=true, will call TTS to flush buffer\n");
            } else {
                // 非双工模式
                if (ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens.clear();
                    t2w_out->is_final = true;
                    t2w_out->is_chunk_end = false;
                    t2w_out->round_meta = active_round_meta;
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                }
                tts_finish = false;
                llm_finish = false;
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                if (mem && !ctx_omni->duplex_mode) {
                    llama_memory_seq_rm(mem, 0, 0, -1);
                }
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                continue;
            }
        }
        
        // Check for LLM data
        bool has_llm_data = (!current_chunk_token_ids.empty() && !current_chunk_hidden_states.empty() && current_chunk_n_embd > 0);
        
        if (has_llm_data) {
            int current_chunk_idx = chunk_idx;
            
            // 收到有效 LLM 数据，重置 turn_eos_flushed 标志位
            if (turn_eos_flushed) {
                turn_eos_flushed = false;
            }
            
            print_with_timestamp("TTS Duplex: processing chunk_idx=%d, n_tokens=%zu, is_end_of_turn=%d\n",
                                chunk_idx, current_chunk_token_ids.size(), accumulated_is_end_of_turn ? 1 : 0);
            
            OmniTtsPreparedChunk prepared_chunk;
            if (!omni_tts_prepare_duplex_chunk(
                    ctx_omni,
                    llm_debug_output_dir,
                    current_chunk_idx,
                    llm_text,
                    current_chunk_token_ids,
                    current_chunk_hidden_states,
                    current_chunk_n_embd,
                    prepared_chunk)) {
                duplex_timing_mark_tts_done(ctx_omni, current_chunk_perf_idx);
                continue;
            }

            int n_tokens_filtered = prepared_chunk.n_tokens_filtered;
            const int tts_n_embd = prepared_chunk.tts_n_embd;
            std::vector<float> & merged_embeddings = prepared_chunk.merged_embeddings;
            const bool merged_success = prepared_chunk.merged_success;
            
            // Generate audio tokens using duplex version
            // 🔧 [修复尾音问题] 当 is_end_of_turn=true 时，即使 merged_embeddings 为空，
            // 也要调用 TTS 生成（只添加 text_eos_embed），让 TTS flush 它的 buffer
            bool should_call_tts = (merged_success && !merged_embeddings.empty()) || 
                                   (accumulated_is_end_of_turn && ctx_omni->duplex_mode);
            
            if (should_call_tts) {
                std::vector<int32_t> audio_tokens_out;
                bool is_end_of_turn = accumulated_is_end_of_turn;
                
                if (merged_embeddings.empty() && is_end_of_turn) {
                    print_with_timestamp("TTS Duplex: is_end_of_turn=true with empty embeddings, calling TTS to flush\n");
                    n_tokens_filtered = 0;
                }
                
                bool tts_gen_success = omni_tts_generate_audio_tokens_local(ctx_omni, params, merged_embeddings,
                                                n_tokens_filtered, tts_n_embd, current_chunk_idx, current_chunk_perf_idx,
                                                audio_tokens_out, is_end_of_turn, active_round_meta, tts_wav_output_dir);
                
                if (tts_gen_success) {
                    all_audio_tokens.insert(all_audio_tokens.end(), audio_tokens_out.begin(), audio_tokens_out.end());
                    
                    if (is_end_of_turn && ctx_omni->duplex_mode) {
                        turn_eos_flushed = true;
                    }
                }
            } else {
                duplex_timing_mark_tts_done(ctx_omni, current_chunk_perf_idx);
            }
            
            ++chunk_idx;
            llm_text.clear();
            response.clear();
            
            // Handle final chunk
            if (llm_finish) {
                tts_finish = true;
                ctx_omni->speek_done = true;
                ctx_omni->warmup_done = true;
                ctx_omni->workers.speek_cv.notify_all();
                
                omni_merge_wav_files(tts_wav_output_dir, chunk_idx + 1);
                
                if (ctx_omni->duplex_mode && !accumulated_is_end_of_turn) {
                    // LISTEN/CHUNK_EOS: 保持 TTS 状态
                    if (ctx_omni->t2w_thread_info) {
                        T2WOut *t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final = false;
                        t2w_out->is_chunk_end = true;
                        t2w_out->round_meta = active_round_meta;
                        t2w_out->duplex_chunk_idx = current_chunk_perf_idx;
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    llm_finish = false;
                    tts_finish = false;
                } else {
                    // 真正的轮次结束
                    if (ctx_omni->t2w_thread_info && !turn_eos_flushed) {
                        T2WOut *t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final = true;
                        t2w_out->round_meta = active_round_meta;
                        t2w_out->duplex_chunk_idx = current_chunk_perf_idx;
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    // 清除 TTS KV cache
                    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                    if (mem) {
                        llama_memory_seq_rm(mem, 0, 0, -1);
                    }
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                    tts_n_past = 0;
                    audio_tokens.clear();
                    all_audio_tokens.clear();
                    llm_finish = false;
                    tts_finish = false;
                }
            }
            continue;
        } else if (ctx_omni->duplex_mode && accumulated_is_end_of_turn && llm_finish) {
            // turn 结束但没有新数据，调用 TTS flush buffer
            if (turn_eos_flushed) {
                print_with_timestamp("TTS Duplex: turn_eos already flushed, skipping TTS generation\n");
            } else {
                print_with_timestamp("TTS Duplex: no LLM data but is_end_of_turn=true, calling TTS to flush buffer\n");
                
                const int tts_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
                std::vector<float> empty_embeddings;
                std::vector<int32_t> audio_tokens_out;
                int n_tokens_for_tts = 0;
                int current_chunk_idx = chunk_idx;
                
                bool tts_gen_success = omni_tts_generate_audio_tokens_local(ctx_omni, params, empty_embeddings,
                                                n_tokens_for_tts, tts_n_embd, current_chunk_idx, current_chunk_perf_idx,
                                                audio_tokens_out, true, active_round_meta, tts_wav_output_dir);
                
                if (tts_gen_success) {
                    all_audio_tokens.insert(all_audio_tokens.end(), audio_tokens_out.begin(), audio_tokens_out.end());
                } else {
                    if (ctx_omni->t2w_thread_info) {
                        T2WOut *t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final = true;
                        t2w_out->is_chunk_end = false;
                        t2w_out->round_meta = active_round_meta;
                        t2w_out->duplex_chunk_idx = current_chunk_perf_idx;
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                }
                
                turn_eos_flushed = true;
            }
            
            // 重置 TTS 状态
            omni_merge_wav_files(tts_wav_output_dir, chunk_idx + 1);
            llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
            if (mem) {
                llama_memory_seq_rm(mem, 0, 0, -1);
            }
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            tts_n_past = 0;
            audio_tokens.clear();
            all_audio_tokens.clear();
            llm_finish = false;
            tts_finish = false;
            ctx_omni->speek_done = true;
            ctx_omni->warmup_done = true;
            ctx_omni->workers.speek_cv.notify_all();
            
            llm_text.clear();
            response.clear();
            continue;
        } else {
            llm_text.clear();
            response.clear();
            continue;
        }
    }
    
    print_with_timestamp("TTS thread (duplex mode) stopped\n");
}

void tts_thread_func(struct omni_context * ctx_omni, common_params *params) {
    // TTS model state
    int tts_n_past = 0;  // TTS model's n_past counter
    std::vector<llama_token> audio_tokens;  // Collected audio tokens
    std::vector<llama_token> all_audio_tokens;  // All audio tokens collected across all chunks
    std::string debug_dir = "";
    bool tts_finish = false;
    bool llm_finish = false;
    int chunk_idx = 0;
    std::string incomplete_bytes;
    
    // 🔧 [多实例支持] 使用可配置的 base_output_dir
    const std::string& base_output_dir = ctx_omni->base_output_dir;
    OmniRoundMeta active_round_meta = omni_session_round_meta(ctx_omni);
    
    // Helper function to create directory
    auto create_dir = [](const std::string& dir_path) {
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0) {
            // Directory doesn't exist, try to create it
            if (!cross_platform_mkdir_p(dir_path)) {
                LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
                return false;
            }
            return true;
        } else if (!(info.st_mode & S_IFDIR)) {
            LOG_ERR("Output path exists but is not a directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    };
    
    // 🔧 [单工模式] Helper function to get round-specific output directory
    // 单工模式下返回 output/round_XXX，双工模式下返回 output
    auto get_round_output_dir = [&](const OmniRoundMeta & round_meta) -> std::string {
        return omni_round_output_dir(base_output_dir, round_meta);
    };
    
    // 🔧 [单工模式] 动态目录路径（每轮开始时更新）
    std::string current_round_dir = get_round_output_dir(active_round_meta);
    std::string tts_output_dir = current_round_dir + "/tts_txt";
    std::string llm_debug_output_dir = current_round_dir + "/llm_debug";
    std::string tts_wav_output_dir = current_round_dir + "/tts_wav";
    
    // 记录上一次创建目录的 round_idx，避免重复创建
    int last_created_round_idx = -1;
    
    // 🔧 [单工模式] Helper function to update output directories for current round
    auto update_output_dirs = [&]() {
        current_round_dir = get_round_output_dir(active_round_meta);
        tts_output_dir = current_round_dir + "/tts_txt";
        llm_debug_output_dir = current_round_dir + "/llm_debug";
        tts_wav_output_dir = current_round_dir + "/tts_wav";
        
        // 只在新的 round 时创建目录
        if (active_round_meta.round_idx != last_created_round_idx || active_round_meta.duplex_mode) {
            create_dir(tts_output_dir);
            create_dir(llm_debug_output_dir);
            create_dir(tts_wav_output_dir);
            last_created_round_idx = active_round_meta.round_idx;
            
            if (!active_round_meta.duplex_mode) {
                print_with_timestamp("TTS: 创建单工模式输出目录: %s\n", current_round_dir.c_str());
            }
        }
    };
    
    // 初始创建目录
    update_output_dirs();
    
    // WAV timing log file (will be created in current round directory)
    auto create_wav_timing_file = [&]() {
        std::string wav_timing_file = tts_wav_output_dir + "/wav_timing.txt";
        FILE* f_timing = fopen(wav_timing_file.c_str(), "w");
        if (f_timing) {
            fprintf(f_timing, "# WAV file generation timing log\n");
            fprintf(f_timing, "# Format: chunk_index, elapsed_time_ms (since stream_decode start), file_size_bytes, request_duration_ms\n");
            fprintf(f_timing, "# Time 0 is when stream_decode() function starts\n");
            fclose(f_timing);
        }
    };
    create_wav_timing_file();

    print_with_timestamp("TTS thread started\n");

    // Multi Round Persistent Loop
    while(ctx_omni->workers.tts_thread_running) {
        
        if (!ctx_omni->workers.tts_thread_running) {
            break;
        }
        
        // 🔧 [P0-打断检测] 检测 break_event 并清空队列、重置状态
        if (ctx_omni->break_event.load()) {
            // 清空 TTS 队列（队列元素类型是 LLMOut*）
            omni_clear_tts_queue(ctx_omni);
            
            // 重置状态
            llm_finish = false;
            tts_finish = false;
            chunk_idx = 0;
            tts_n_past = 0;
            audio_tokens.clear();
            all_audio_tokens.clear();
            incomplete_bytes.clear();  // 🔧 [多轮对话修复] 清理不完整的 UTF-8 字节
            // 🔧 [多轮对话修复] 清理 TTS 累积状态，避免混淆
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            // 不清除 break_event，让 T2W 线程也能检测到
            continue;
        }

        std::string llm_text = "";
        // 保存当前chunk的token IDs和hidden states用于TTS条件生成
        std::vector<llama_token> current_chunk_token_ids;
        std::vector<float> current_chunk_hidden_states;
        int current_chunk_n_embd = 0;
            
        // Always wait for queue if not finished, or if finished but need to reset state
        if (!llm_finish || (llm_finish && llm_text.empty())) {
            std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
                
            auto& queue = ctx_omni->tts_thread_info->queue;
            // 🔧 [P0-打断检测] 在等待时也监听 break_event
            ctx_omni->tts_thread_info->cv.wait(lock, [&] { 
                return !queue.empty() || !ctx_omni->workers.tts_thread_running || ctx_omni->break_event.load(); 
            });
            
            // 检测到 break_event 时跳过当前处理
            if (ctx_omni->break_event.load()) {
                lock.unlock();
                continue;
            }
                
            if (!ctx_omni->workers.tts_thread_running) {
                break;
            }
            // 🔧 [关键修复] 每次只处理一个 chunk，不要一次性取出所有
            // 之前的问题：while (!queue.empty()) 会取出所有 chunk，导致：
            // - llm_text 累积了多个 chunk 的文本
            // - 但 token_ids/hidden_states 被覆盖，只保留最后一个 chunk
            // - TTS 用 10 token 的 condition 去生成 20 token 文本的语音 → 内容丢失！
            // 
            // Python 逻辑：每次只处理一个 chunk，生成对应的 audio tokens
            if (!queue.empty()) {
                LLMOut *llm_out = queue.front();
                llm_finish |= llm_out->llm_finish;
                active_round_meta = llm_out->round_meta;
                // 只取一个 chunk 的数据
                if (!ctx_omni->speek_done || ctx_omni->duplex_mode) {
                    llm_text = llm_out->text;  // 注意：= 而不是 +=
                    debug_dir = llm_out->debug_dir;
                }
                // 保存这一个 chunk 的 token IDs 和 hidden states
                if (!llm_out->token_ids.empty() && !llm_out->hidden_states.empty()) {
                    current_chunk_token_ids = llm_out->token_ids;
                    current_chunk_hidden_states = llm_out->hidden_states;
                    current_chunk_n_embd = llm_out->n_embd;
                    
                    // 🔧 [诊断日志] 打印 TTS 接收到的数据
                    std::string token_ids_str = "";
                    for (size_t i = 0; i < current_chunk_token_ids.size() && i < 20; i++) {
                        token_ids_str += std::to_string(current_chunk_token_ids[i]);
                        if (i < current_chunk_token_ids.size() - 1 && i < 19) token_ids_str += " ";
                    }
                    if (current_chunk_token_ids.size() > 20) token_ids_str += "...";
                    
                    print_with_timestamp("TTS<-LLM: chunk_idx=%d, text='%s', n_tokens=%zu, hidden_size=%zu, token_ids=[%s]\n",
                                        chunk_idx,
                                        llm_text.c_str(),
                                        current_chunk_token_ids.size(),
                                        current_chunk_hidden_states.size(),
                                        token_ids_str.c_str());
                }
                delete llm_out;
                queue.pop();
            }
            lock.unlock();
            ctx_omni->tts_thread_info->cv.notify_all();
            update_output_dirs();
            
            // 🔧 [诊断] 打印取出数据后的关键状态
            print_with_timestamp("TTS: after queue pop - speek_done=%d, llm_finish=%d, llm_text.empty=%d, token_ids.size=%zu\n",
                                ctx_omni->speek_done, llm_finish, llm_text.empty(), current_chunk_token_ids.size());
            
            // If speek_done is true but we received llm_finish=true, handle state transition
            if (ctx_omni->speek_done && llm_finish) {
                print_with_timestamp("TTS: speek_done=true and llm_finish=true, resetting state for next round\n");
                // 🔧 [关键修复 - 与 Python 对齐] 在双工模式下，如果有新数据，应该继续处理
                // Python: 每次 streaming_generate 调用都会独立处理，不会因为之前的状态跳过
                if (ctx_omni->duplex_mode && !current_chunk_token_ids.empty()) {
                    // 双工模式下有新数据，重置 speek_done 并继续处理
                    ctx_omni->speek_done = false;
                    // 不执行 continue，继续后续的 TTS 处理
                } else {
                    // 没有新数据，或者非双工模式：重置状态并跳过
                    llm_finish = false;
                    llm_text.clear();
                    chunk_idx = 0;
                    tts_n_past = 0;
                    audio_tokens.clear();
                    // 🔧 [多轮对话修复] 清理 TTS 累积状态
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                    continue;  // Skip processing and wait for next input
                }
            }
        }
            
        std::string& response = llm_text;
        // 拼接上一个循环保存的不完整字节
        if (!incomplete_bytes.empty()) {
            print_with_timestamp("TTS: prepending incomplete_bytes (len=%zu) to response (len=%zu)\n", 
                                incomplete_bytes.length(), response.length());
            response = incomplete_bytes + response;
            incomplete_bytes.clear();
        }
        // 检查当前响应是否有不完整的 UTF-8 序列
        size_t incomplete_len = findIncompleteUtf8(response);
        if (incomplete_len > 0) {
            print_with_timestamp("TTS: detected incomplete UTF-8 sequence at end: incomplete_len=%zu, response_len=%zu\n", 
                                incomplete_len, response.length());
            // 保存不完整的字节
            incomplete_bytes = response.substr(response.size() - incomplete_len, incomplete_len);
            // 仅处理完整的部分
            response = response.substr(0, response.size() - incomplete_len);
            print_with_timestamp("TTS: after truncation: response_len=%zu, incomplete_bytes_len=%zu\n", 
                                response.length(), incomplete_bytes.length());
        } else {
            // 确保没有残留的不完整字节
            incomplete_bytes.clear();
        }
        
        // 🔧 [诊断] 打印 response 状态
        print_with_timestamp("TTS: before empty check - response.empty=%d, response='%s', llm_finish=%d\n",
                            response.empty(), response.substr(0, 50).c_str(), llm_finish);
        
        // Skip empty responses (but allow processing if llm_finish is true to handle end of generation)
        if (response.empty() && !llm_finish) {
            // If speek_done is true but we're still getting empty responses, just continue waiting
            // Don't reset speek_done here - let stream_prefill reset it after it wakes up
            if (ctx_omni->speek_done) {
                print_with_timestamp("TTS: speek_done=true with empty response, keeping state (waiting for stream_prefill)\n");
                llm_finish = false;
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                // 🔧 [多轮对话修复] 清理 TTS 累积状态
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                // NOTE: 不重置 speek_done，让 stream_prefill 完成等待后重置
            }
            continue;
        }
        fflush(stdout);
            
        // 🔧 [修复] 空 response + llm_finish 的提前检查
        // 必须在 tokenize 之前检查，因为空字符串 tokenize 可能返回非空结果（BOS token）
        if (response.empty() && llm_finish) {
            // 🔧 [修复] 当收到 llm_finish=true 但没有新数据时，
            // 需要 flush tts_token_buffer 中剩余的 tokens，并发送 is_final=true 到 T2W
            print_with_timestamp("TTS: received llm_finish=true with no data, finalizing (tts_token_buffer=%zu)\n",
                                ctx_omni->tts_token_buffer.size());
            
            // 🔧 [修复] Flush tts_token_buffer 中剩余的 tokens 到 T2W
            if (ctx_omni->t2w_thread_info && !ctx_omni->tts_token_buffer.empty()) {
                T2WOut *t2w_out = new T2WOut();
                t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(), 
                                             ctx_omni->tts_token_buffer.end());
                t2w_out->is_final = false;  // 先发送剩余 tokens
                t2w_out->round_meta = active_round_meta;
                {
                    std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    ctx_omni->t2w_thread_info->queue.push(t2w_out);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
                print_with_timestamp("TTS: flushed %zu remaining tokens from tts_token_buffer\n", 
                                    ctx_omni->tts_token_buffer.size());
                ctx_omni->tts_token_buffer.clear();
            }
            
            // 🔧 [修复] 发送 is_final=true 到 T2W，让 T2W 写入 generation_done.flag
            if (ctx_omni->t2w_thread_info) {
                const OmniRoundMeta completed_round_meta = active_round_meta;
                
                T2WOut *t2w_final = new T2WOut();
                t2w_final->audio_tokens.clear();
                t2w_final->is_final = true;
                t2w_final->round_meta = completed_round_meta;
                {
                    std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    ctx_omni->t2w_thread_info->queue.push(t2w_final);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
                print_with_timestamp("TTS: sent is_final=true to T2W (llm_finish with no data)\n");
            }
            
            ctx_omni->speek_done = true;
            ctx_omni->warmup_done = true;  // 第一轮对话结束，后续 prefill 需要等待
            ctx_omni->workers.speek_cv.notify_all();
            print_with_timestamp("TTS: finished processing all chunks (llm_finish with no data path)\n");
            // 非双工模式或真正的轮次结束：完全重置
            tts_finish = false;
            llm_finish = false;
            chunk_idx = 0;
            tts_n_past = 0;
            audio_tokens.clear();
            // 🔧 [多轮对话修复] 清理 TTS 累积状态
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            continue;
        }
        
        // 2. 根据Python实现，将LLM输出转换为TTS条件（hidden_text_merge模式）
        // Python流程：
        //   1. 使用TTS的emb_text层（152064词表）处理LLM token: llm_embeds = self.tts.emb_text(yield_chunk_token_ids)
        //   2. 使用projector_semantic将LLM hidden states（4096维）投影到TTS hidden_dim（768维）:
        //      hidden_embeds = self.tts.projector_semantic(output.last_hidden_states)
        //   3. 归一化: hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        //   4. 合并: tts_embeds = llm_embeds + hidden_embeds
        //   5. 添加audio_bos_token的embedding
        //   6. 将tts_embeds作为condition传递给TTS（使用batch.embd而不是token IDs）
        fflush(stdout);
        
        // 🔧 [诊断] 打印 current_chunk_n_embd 的值，确认数据是否正确传递
        print_with_timestamp("TTS: DEBUG before has_llm_data - token_ids.size=%zu, hidden_states.size=%zu, n_embd=%d\n",
                            current_chunk_token_ids.size(), current_chunk_hidden_states.size(), current_chunk_n_embd);
        
        // 使用从队列中获取的token IDs和hidden states
        bool has_llm_data = (!current_chunk_token_ids.empty() && !current_chunk_hidden_states.empty() && current_chunk_n_embd > 0);
        
        // 🔧 [移除] 之前这里有一段"等待 50ms 并累积更多消息"的逻辑
        // 这会导致多个 chunk 被合并，破坏了"每次只处理一个 chunk"的逻辑
        // Python 不会累积多个 chunk，而是每个 chunk 独立处理
        // llm_finish 标志会在下一次从队列取数据时正确捕获
        
        // 🔧 [诊断日志] 打印关键数据状态
        print_with_timestamp("TTS: has_llm_data=%d, token_ids=%zu, hidden_states=%zu, n_embd=%d, llm_finish=%d\n",
                             has_llm_data, current_chunk_token_ids.size(), 
                             current_chunk_hidden_states.size(), current_chunk_n_embd, llm_finish);
        
        // // 🔧 [调试] 打印传给 TTS 的 token IDs
        // if (has_llm_data) {
        //     for (size_t i = 0; i < current_chunk_token_ids.size(); i++) {
        //     }
            
        //     // 打印每个 token 对应的 hidden state 前3个值
        //     for (size_t i = 0; i < current_chunk_token_ids.size(); i++) {
        //         size_t offset = i * current_chunk_n_embd;
        //     }
        // }
        
        if (has_llm_data) {
            // 🔧 [单工模式] 在每轮第一个 chunk 时更新输出目录
            // 确保每轮数据保存在独立的 round_XXX 子目录下
            if (chunk_idx == 0 && !ctx_omni->duplex_mode) {
                update_output_dirs();
                create_wav_timing_file();
            }
            
            // Save current chunk_idx to ensure consistent directory naming
            // This is important because chunk_idx will be incremented after processing
            int current_chunk_idx = chunk_idx;
            
            OmniTtsPreparedChunk prepared_chunk;
            if (!omni_tts_prepare_simplex_chunk(
                    ctx_omni,
                    llm_debug_output_dir,
                    current_chunk_idx,
                    response,
                    current_chunk_token_ids,
                    current_chunk_hidden_states,
                    current_chunk_n_embd,
                    prepared_chunk)) {
                continue;
            }

            int n_tokens_filtered = prepared_chunk.n_tokens_filtered;
            const int tts_n_embd = prepared_chunk.tts_n_embd;
            std::vector<float> & merged_embeddings = prepared_chunk.merged_embeddings;
            const bool merged_success = prepared_chunk.merged_success;
            
            // 🔧 [与 Python 对齐] is_final_text_chunk 需要在外层声明，供后面调用使用
            // 🔧 [修复] 当 llm_finish=true 时，这是最后一个 text chunk，需要设置 is_final_text_chunk=true
            // 这样 TTS 会 flush 剩余的 tts_token_buffer，并正确处理 text_eos_embed
            bool is_final_text_chunk = llm_finish;
            
            // Generate audio tokens using local TTS model (if merged_embeddings available)
            // or fall back to TTS server
            // Use local TTS model if merged_embeddings were successfully computed
            if (merged_success && !merged_embeddings.empty()) {
                std::vector<int32_t> audio_tokens;
                
                // 🔧 [单双工适配] 根据 duplex_mode 选择不同的函数
                bool tts_gen_success = false;
                // 🔧 [与 Python 对齐] 传递 is_final_text_chunk，用于 flush buffer
                tts_gen_success = omni_tts_generate_audio_tokens_local_simplex(ctx_omni, params, merged_embeddings,
                                                n_tokens_filtered, tts_n_embd, current_chunk_idx,
                                                audio_tokens, active_round_meta, tts_wav_output_dir, is_final_text_chunk);
                if (tts_gen_success) {
                    // Save audio tokens for external token2wav processing (backup)
                    // token2wav expects relative token IDs (0-6561)
                    std::string tokens_txt_file = tts_wav_output_dir + "/audio_tokens_chunk_" + 
                                                  std::to_string(current_chunk_idx) + ".txt";
                    FILE* f_tokens = fopen(tokens_txt_file.c_str(), "w");
                    if (f_tokens) {
                        for (size_t i = 0; i < audio_tokens.size(); ++i) {
                            fprintf(f_tokens, "%d", audio_tokens[i]);
                            if (i < audio_tokens.size() - 1) fprintf(f_tokens, ",");
                        }
                        fprintf(f_tokens, "\n");
                        fclose(f_tokens);
                    }
                    
                    // Accumulate all audio tokens for final WAV generation
                    all_audio_tokens.insert(all_audio_tokens.end(), 
                                           audio_tokens.begin(), audio_tokens.end());
                    
                    // 🔧 [与 Python 流式双工对齐] tokens 已在 generate_audio_tokens_local 内部流式推送
                    // T2W 端的 buffer 会持续累积，并按滑动窗口处理，保留 lookahead
                    // Python 逻辑：buffer 滑动 CHUNK_SIZE (25)，保留 pre_lookahead (3)
                    
                    // Also save audio tokens to file for the chunk (for debugging)
                    // This is separate from the token2wav sliding window
                } else {
                    LOG_ERR("TTS Local: failed for chunk %d\n", current_chunk_idx);
                }
            }
            
            // Always increment chunk_idx after processing
            // This ensures each chunk gets its own directory and data is not overwritten
            ++chunk_idx;
            
            // Clear processed text
            llm_text.clear();
            response.clear();
            
            // If this is the final chunk, mark as finished and merge all WAV files
            if (llm_finish) {
                // 🔧 [与 Python 对齐] LLM 完成时，flush 剩余的 tts_token_buffer
                // 因为 is_final_text_chunk 的判断可能不准确（队列时序问题）
                if (!ctx_omni->tts_token_buffer.empty() && ctx_omni->t2w_thread_info) {
                    print_with_timestamp("TTS: llm_finish=true, flushing remaining %zu tokens from tts_token_buffer\n",
                                        ctx_omni->tts_token_buffer.size());
                    
                    T2WOut *t2w_out = new T2WOut();
                    if (t2w_out) {
                        t2w_out->audio_tokens.assign(
                            ctx_omni->tts_token_buffer.begin(),
                            ctx_omni->tts_token_buffer.end()
                        );
                        t2w_out->is_final = false;  // 还不是最后一个，后面还有 turn_end 的 is_final
                        t2w_out->round_meta = active_round_meta;
                        
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    ctx_omni->tts_token_buffer.clear();
                }
                
                tts_finish = true;
                ctx_omni->speek_done = true;
                ctx_omni->warmup_done = true;  // 第一轮对话结束，后续 prefill 需要等待
                ctx_omni->workers.speek_cv.notify_all();
                print_with_timestamp("TTS: finished processing all chunks\n");
                
                // Merge all WAV files into a single file
                omni_merge_wav_files(tts_wav_output_dir, chunk_idx + 1);
                // Python: end_of_turn = last_id in turn_terminator_token_ids
                const OmniRoundMeta completed_round_meta = active_round_meta;
                
                // 🔧 发送 is_final=true 到 T2W 队列，通知 T2W 重置 buffer
                // 注意：T2W 端只在双工模式下调用 Token2WavSession::reset()
                // 单工模式下只重置 token_buffer 为静音 tokens，不调用 reset()
                // 🔧 [修复最后一个字没说完] 移除 !all_audio_tokens.empty() 条件
                // 原因：is_final=true 必须发送，否则 T2W 不会 flush 最后的 buffer
                if (ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens.clear();  // 空tokens，只是通知final
                    t2w_out->is_final = true;
                    t2w_out->round_meta = completed_round_meta;
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                    print_with_timestamp("TTS: sent is_final=true to T2W queue (turn end)\n");
                }
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                // 🔧 [修复竞态条件] 不在这里重置 tts_condition_saved
                // 原因：新一轮的 generate_audio_tokens_local 会在 chunk_idx == 0 时重新设置
                // 如果在这里重置，可能与新一轮的 TTS 处理产生竞态条件
                // 旧版单工代码也没有在这里重置
                if (ctx_omni->duplex_mode) {
                    ctx_omni->tts_condition_saved = false;
                }
                
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                all_audio_tokens.clear();
                llm_finish = false;
                tts_finish = false;
                // WAV 编号基数已改为从 round metadata 派生，这里不再做线程内补偿。
            }
            
            continue;  // Skip the rest of the TTS processing
        } else {
            // Clear processed text and continue
            llm_text.clear();
            response.clear();
            continue;
        }
        
        // OLD TTS PROCESSING CODE BELOW (DISABLED)
        // All code below is disabled and replaced by TTS service calls above
        #if 0
        // 准备TTS输入：使用合并后的embedding（hidden_text_merge模式）
        const int tts_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
        std::vector<float> tts_condition_embeddings;  // 合并后的embedding
        std::vector<llama_token> input_tokens;  // Fallback: 如果无法生成embedding，使用token IDs
        
        if (has_llm_data) {
            // 步骤1: 使用TTS的emb_text层处理LLM token IDs
            int n_tokens = (int)current_chunk_token_ids.size();
            std::vector<float> llm_embeds(n_tokens * tts_n_embd, 0.0f);
            bool emb_text_success = true;
            
            for (int i = 0; i < n_tokens; i++) {
                llama_token token_id = current_chunk_token_ids[i];
                float * emb = llm_embeds.data() + i * tts_n_embd;
                if (!tts_emb_text(ctx_omni, token_id, emb, tts_n_embd)) {
                    emb_text_success = false;
                    break;
                }
            }
            
            if (emb_text_success) {
                // 步骤2: 使用projector_semantic投影LLM hidden states
                // 验证hidden states的shape
                
                // 打印前几个hidden state的值用于调试
                if (current_chunk_hidden_states.size() >= 5) {
                }
                
                std::vector<float> projected_hidden(n_tokens * tts_n_embd, 0.0f);
                bool projector_success = tts_projector_semantic(ctx_omni,
                                                                 current_chunk_hidden_states.data(),
                                                                 n_tokens,
                                                                 current_chunk_n_embd,
                                                                 projected_hidden.data(),
                                                                 tts_n_embd);
                
                if (projector_success) {
                    // 打印投影后的前几个值
                    if (projected_hidden.size() >= 5) {
                    }
                    
                    // 步骤3: 归一化projected hidden states
                    normalize_l2_per_token(projected_hidden.data(), n_tokens, tts_n_embd);
                    
                    // 打印归一化后的前几个值
                    if (projected_hidden.size() >= 5) {
                    }
                    
                    // 步骤4: 合并: tts_embeds = llm_embeds + hidden_embeds
                    // 🔧 [安全检查] 防止创建过大的 vector 导致崩溃
                    size_t cond_size = (size_t)n_tokens * tts_n_embd;
                    if (n_tokens <= 0 || n_tokens > 10000 || 
                        tts_n_embd <= 0 || tts_n_embd > 10000 ||
                        cond_size > 100000000) {  // 100M elements max
                        LOG_ERR("TTS: invalid condition size: n_tokens=%d, tts_n_embd=%d, cond_size=%zu\n",
                                n_tokens, tts_n_embd, cond_size);
                        break;  // 跳过这个 chunk
                    }
                    tts_condition_embeddings.resize(cond_size);
                    for (size_t i = 0; i < cond_size; i++) {
                        tts_condition_embeddings[i] = llm_embeds[i] + projected_hidden[i];
                    }
                    
                    // 打印合并后的前几个值
                    if (tts_condition_embeddings.size() >= 5) {
                    }
                    
                    // Save LLM debug data: text, hidden states, and merged embeddings
                    {
                        // Create chunk-specific directory
                        std::string chunk_dir = llm_debug_output_dir + "/chunk_" + std::to_string(chunk_idx);
                        create_dir(chunk_dir);
                        
                        // 1. Save LLM text output
                        std::string text_file = chunk_dir + "/llm_text.txt";
                        FILE *f_text = fopen(text_file.c_str(), "w");
                        if (f_text) {
                            fprintf(f_text, "%s", response.c_str());
                            fclose(f_text);
                        }
                        
                        // 2. Save LLM token IDs
                        std::string token_ids_file = chunk_dir + "/llm_token_ids.txt";
                        FILE *f_tokens = fopen(token_ids_file.c_str(), "w");
                        if (f_tokens) {
                            fprintf(f_tokens, "Token IDs (%zu):\n", current_chunk_token_ids.size());
                            for (size_t i = 0; i < current_chunk_token_ids.size(); ++i) {
                                fprintf(f_tokens, "%d", current_chunk_token_ids[i]);
                                if (i < current_chunk_token_ids.size() - 1) fprintf(f_tokens, " ");
                            }
                            fprintf(f_tokens, "\n");
                            fclose(f_tokens);
                        }
                        
                        // 3. Save LLM hidden states (binary format for precision)
                        std::string hidden_file = chunk_dir + "/llm_hidden_states.bin";
                        FILE *f_hidden = fopen(hidden_file.c_str(), "wb");
                        if (f_hidden) {
                            // Write header: n_tokens, n_embd
                            int32_t header[2] = {n_tokens, current_chunk_n_embd};
                            fwrite(header, sizeof(int32_t), 2, f_hidden);
                            // Write hidden states data
                            fwrite(current_chunk_hidden_states.data(), sizeof(float), current_chunk_hidden_states.size(), f_hidden);
                            fclose(f_hidden);
                        }
                        
                        // 4. Save LLM hidden states as text (for easy inspection)
                        std::string hidden_txt_file = chunk_dir + "/llm_hidden_states.txt";
                        FILE *f_hidden_txt = fopen(hidden_txt_file.c_str(), "w");
                        if (f_hidden_txt) {
                            fprintf(f_hidden_txt, "Hidden States (shape: [%d, %d]):\n", n_tokens, current_chunk_n_embd);
                            for (int i = 0; i < n_tokens; ++i) {
                                fprintf(f_hidden_txt, "Token %d: ", i);
                                for (int j = 0; j < current_chunk_n_embd; ++j) {
                                    fprintf(f_hidden_txt, "%.6f", current_chunk_hidden_states[i * current_chunk_n_embd + j]);
                                    if (j < current_chunk_n_embd - 1) fprintf(f_hidden_txt, " ");
                                }
                                fprintf(f_hidden_txt, "\n");
                            }
                            fclose(f_hidden_txt);
                        }
                        
                        // 5. Save merged embeddings (binary format)
                        std::string merged_file = chunk_dir + "/merged_embeddings.bin";
                        FILE *f_merged = fopen(merged_file.c_str(), "wb");
                        if (f_merged) {
                            // Write header: n_tokens, tts_n_embd
                            int32_t header[2] = {n_tokens, tts_n_embd};
                            fwrite(header, sizeof(int32_t), 2, f_merged);
                            // Write merged embeddings data
                            fwrite(tts_condition_embeddings.data(), sizeof(float), tts_condition_embeddings.size(), f_merged);
                            fclose(f_merged);
                        }
                        
                        // 6. Save merged embeddings as text (for easy inspection)
                        std::string merged_txt_file = chunk_dir + "/merged_embeddings.txt";
                        FILE *f_merged_txt = fopen(merged_txt_file.c_str(), "w");
                        if (f_merged_txt) {
                            fprintf(f_merged_txt, "Merged Embeddings (shape: [%d, %d]):\n", n_tokens, tts_n_embd);
                            for (int i = 0; i < n_tokens; ++i) {
                                fprintf(f_merged_txt, "Token %d: ", i);
                                for (int j = 0; j < tts_n_embd; ++j) {
                                    fprintf(f_merged_txt, "%.6f", tts_condition_embeddings[i * tts_n_embd + j]);
                                    if (j < tts_n_embd - 1) fprintf(f_merged_txt, " ");
                                }
                                fprintf(f_merged_txt, "\n");
                            }
                            fclose(f_merged_txt);
                        }
                        
                        // 7. Save intermediate results: llm_embeds and projected_hidden
                        std::string llm_embeds_file = chunk_dir + "/llm_embeds.txt";
                        FILE *f_llm_embeds = fopen(llm_embeds_file.c_str(), "w");
                        if (f_llm_embeds) {
                            fprintf(f_llm_embeds, "LLM Embeddings from emb_text (shape: [%d, %d]):\n", n_tokens, tts_n_embd);
                            for (int i = 0; i < n_tokens; ++i) {
                                fprintf(f_llm_embeds, "Token %d: ", i);
                                for (int j = 0; j < tts_n_embd; ++j) {
                                    fprintf(f_llm_embeds, "%.6f", llm_embeds[i * tts_n_embd + j]);
                                    if (j < tts_n_embd - 1) fprintf(f_llm_embeds, " ");
                                }
                                fprintf(f_llm_embeds, "\n");
                            }
                            fclose(f_llm_embeds);
                        }
                        
                        std::string projected_file = chunk_dir + "/projected_hidden.txt";
                        FILE *f_projected = fopen(projected_file.c_str(), "w");
                        if (f_projected) {
                            fprintf(f_projected, "Projected Hidden States (shape: [%d, %d], after normalization):\n", n_tokens, tts_n_embd);
                            for (int i = 0; i < n_tokens; ++i) {
                                fprintf(f_projected, "Token %d: ", i);
                                for (int j = 0; j < tts_n_embd; ++j) {
                                    fprintf(f_projected, "%.6f", projected_hidden[i * tts_n_embd + j]);
                                    if (j < tts_n_embd - 1) fprintf(f_projected, " ");
                                }
                                fprintf(f_projected, "\n");
                            }
                            fclose(f_projected);
                        }
                    }
                    
                    // 🔧 [修复] 不在 merge embed 阶段添加 audio_bos
                    // audio_bos 将在 prefill 之前动态添加
                } else {
                    emb_text_success = false;
                }
            }
            
            if (!emb_text_success) {
                // Fallback: 使用token IDs
                input_tokens.insert(input_tokens.end(), current_chunk_token_ids.begin(), current_chunk_token_ids.end());
                input_tokens.push_back(audio_bos_token_id);
            }
        } else {
            // 没有LLM数据，使用text_tokens作为fallback
            input_tokens.insert(input_tokens.end(), text_tokens.begin(), text_tokens.end());
            input_tokens.push_back(audio_bos_token_id);
        }
        
        // 3. Reset TTS KV cache ONLY when starting a completely new round (speek_done was true and we're starting fresh)
        // IMPORTANT: Python's TTSStreamingGenerator maintains past_key_values across chunks:
        //   - First call: past_key_values = None (reset)
        //   - Subsequent chunks: past_key_values = self.past_key_values (continue)
        //   - Only reset on end_of_turn or new round
        // We should NOT clear KV cache for each new chunk from LLM if LLM is still generating
        // Only clear when speek_done was true (indicating a new round)
        fflush(stdout);
        
        // Only reset if we're starting a new round (speek_done was true, meaning previous round finished)
        // AND we have previous state (tts_n_past > 0)
        // This matches Python behavior: reset only on end_of_turn or new round
        static bool last_speek_done = false;
        bool should_reset = (last_speek_done && ctx_omni->speek_done == false && tts_n_past > 0);
        last_speek_done = ctx_omni->speek_done;
        
        if (should_reset) {
            fflush(stdout);
            // Clear KV cache to start fresh for new round
            llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
            if (mem) {
                llama_memory_seq_rm(mem, 0, 0, -1);
            } else {
            }
            fflush(stdout);
            tts_n_past = 0;
        } else {
            fflush(stdout);
        }
        
        // 4. Evaluate input (使用embedding或token IDs)
        // Python: condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
        // Python: condition_length = current_condition.shape[1]
        // Python: pos_ids = torch.arange(self.text_start_pos, self.text_start_pos + condition_length)
        // Python: self.text_start_pos += condition_length + len(chunk_generated_tokens)
        if (!tts_condition_embeddings.empty()) {
            // 🔧 [修复] 在 prefill 之前动态添加 audio_bos embedding
            // Python 中 audio_bos 是在 TTS 类内部（TTSStreamingGenerator.generate_with_buffer）添加的
            // 每个 chunk 都会加 audio_bos: condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
            std::vector<float> audio_bos_embed(tts_n_embd, 0.0f);
            if (tts_emb_text(ctx_omni, audio_bos_token_id, audio_bos_embed.data(), tts_n_embd)) {
                // Append audio_bos to tts_condition_embeddings
                tts_condition_embeddings.insert(tts_condition_embeddings.end(), 
                                                audio_bos_embed.begin(), audio_bos_embed.end());
                print_with_timestamp("TTS: 在 prefill 前添加 audio_bos (chunk_idx=%d, new_size=%zu)\n", 
                                    chunk_idx, tts_condition_embeddings.size() / tts_n_embd);
            } else {
                LOG_ERR("TTS: failed to get audio_bos embedding for prefill\n");
            }
            
            // 使用合并后的embedding作为输入（正确的实现方式）
            int condition_length = (int)(tts_condition_embeddings.size() / tts_n_embd);
            fflush(stdout);
            
            // 使用prefill_emb_with_hidden类似的逻辑，但针对TTS模型
            // Python: prefill时，position_ids从text_start_pos开始，到text_start_pos + condition_length
            // C++: batch.pos[j] = tts_n_past + j，其中tts_n_past对应text_start_pos
            int n_pos = condition_length;
            int text_start_pos_before = tts_n_past;  // 保存prefill前的text_start_pos
            if (!prefill_with_emb_tts(ctx_omni, params, tts_condition_embeddings.data(), n_pos, params->n_batch, &tts_n_past)) {
                LOG_ERR("Failed to evaluate TTS input embeddings\n");
                // Clear the processed text to prevent infinite loop
                llm_text.clear();
                response.clear();
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                // 🔧 [多轮对话修复] 清理 TTS 累积状态
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                fflush(stdout);
                continue;
            }
                int condition_length_processed = tts_n_past - text_start_pos_before;
        } else {
            // Fallback: 使用token IDs作为输入
            fflush(stdout);
            fflush(stdout);
            if (!eval_tokens_tts(ctx_omni, params, input_tokens, params->n_batch, &tts_n_past)) {
                LOG_ERR("Failed to evaluate TTS input tokens\n");
                // Clear the processed text to prevent infinite loop
                llm_text.clear();
                response.clear();
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                // 🔧 [多轮对话修复] 清理 TTS 累积状态
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                fflush(stdout);
                continue;
            }
        }
        
        // 4. Generate audio tokens using TTS model
        // Following Python flow: decode 1 token at a time, accumulate to buffer, yield when buffer reaches 25 tokens
        audio_tokens.clear();
        bool audio_gen_finish = false;
        int audio_token_count = 0;
        
        // Limit audio tokens per chunk: 25 tokens per chunk (each chunk corresponds to ~10 LLM tokens)
        // 每一轮的限制：LLM生成10个token → TTS生成25个token
        // 接下来LLM再生成10个token → TTS再生成25个token，以此类推
        int text_token_count = (int)text_tokens.size();
        // Fixed at 25 tokens per chunk (matching Python's audio_token_chunk_size=25)
        int max_audio_tokens_for_text = 25;
        
        // Buffer for accumulating audio tokens (yield when reaching 25 tokens, matching Python chunk_size=25)
        const int audio_token_chunk_size = 25;  // Yield size matching Python
        std::vector<llama_token> audio_token_buffer;
        
        // 🔧 [差异3修复] 当前 chunk 的 tokens，用于 repetition penalty
        std::vector<llama_token> chunk_generated_tokens_tts;
        
        // Generate audio tokens one by one (matching Python decode flow)
        while (audio_token_count < max_audio_tokens_for_text && !audio_gen_finish && !ctx_omni->speek_done) {
            fflush(stdout);
            
            // Decode 1 audio token at a time (matching Python: each decode generates 1 token)
            // 🔧 [差异2&3修复] 传入：
            // - all_audio_tokens: 用于判断 is_first_token_overall
            // - chunk_generated_tokens_tts: 用于 repetition penalty（只用当前 chunk 的 tokens）
            // - audio_token_count: token_index_in_chunk，用于判断 is_chunk_first_token
            // - force_no_eos: false（这个路径已有 25 tokens 的限制，不需要额外阻止 EOS）
            llama_token audio_token = sample_tts_token_simplex(ctx_omni->ctx_tts_sampler, ctx_omni, params, &tts_n_past, &all_audio_tokens, &chunk_generated_tokens_tts, audio_token_count, false);
            fflush(stdout);
            
            audio_token_buffer.push_back(audio_token);
            audio_tokens.push_back(audio_token);
            all_audio_tokens.push_back(audio_token);
            chunk_generated_tokens_tts.push_back(audio_token);  // 🔧 [差异3修复]
            audio_token_count++;
            
            // Check for end of audio generation
            bool is_audio = is_audio_token(audio_token, audio_bos_token_id, num_audio_tokens);
            int relative_idx = audio_token - audio_bos_token_id;
            fflush(stdout);
            
            // Check for audio EOS token (relative index 6561, absolute 158248)
            // Note: EOS token should only end audio generation for current chunk, not set speek_done
            // speek_done should only be set when llm_finish is true
            if (audio_token == audio_eos_token_id || relative_idx == 6561) {
                fflush(stdout);
                audio_gen_finish = true;
                // Only set tts_finish if LLM has also finished, otherwise continue to next chunk
                if (llm_finish) {
                    tts_finish = true;
                }
                break;
            }
            
            // Yield when buffer reaches chunk_size (25 tokens), matching Python yield logic
            if (audio_token_buffer.size() >= audio_token_chunk_size) {
                fflush(stdout);
                
                // Push audio tokens to T2W queue for token2wav processing
                if (ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens = audio_token_buffer;  // Copy the 25 tokens
                    t2w_out->is_final = false;  // Not final, more chunks may come
                    t2w_out->round_meta = omni_session_round_meta(ctx_omni);
                    
                    {
                        std::unique_lock<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        // Check if queue has space
                        while (ctx_omni->t2w_thread_info->queue.size() >= ctx_omni->t2w_thread_info->MAX_QUEUE_SIZE) {
                            // Wait for space in queue (with timeout to avoid deadlock)
                            lock.unlock();
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            lock.lock();
                        }
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                }
                
                // Yield this chunk (create unit_buffer and save)
                // Note: In Python, each yield creates a chunk, but here we create one buffer per 25-token chunk
                struct unit_buffer *res = new unit_buffer();
                res->text = response;
                res->unit_n_past = ctx_omni->n_past;
                res->completed = false;  // Not completed yet, more tokens may come
                
                // TODO: Convert audio tokens to audio waveform
                // For now, create a placeholder buffer
                res->buffer.resize(audio_token_chunk_size * 100);  // Placeholder size
                
                if (!res->buffer.empty()) {
                    // 读写使用 mutex 保护
                    std::unique_lock<std::mutex> lock(ctx_omni->workers.buffer_mutex);
                    ctx_omni->omni_output->output.push_back(res);
                    ctx_omni->omni_output->idx += 1;
                } else {
                    delete res;
                }
                
                // Clear buffer after yield, matching Python behavior
                audio_token_buffer.clear();
            }
        }
        
        // If we exited the loop because we reached max_audio_tokens_for_text (25 tokens per chunk),
        // mark audio_gen_finish as true to indicate this chunk is complete
        // This allows the next iteration to process the next chunk from LLM (if LLM is still generating)
        if (!audio_gen_finish && audio_token_count >= max_audio_tokens_for_text && !ctx_omni->speek_done) {
            fflush(stdout);
            audio_gen_finish = true;
        }
        fflush(stdout);
        
        // Save audio tokens to file for debugging/analysis
        // Note: Save relative indices (0-6561) for token2wav compatibility
        if (!audio_tokens.empty()) {
            // Always save to fixed output directory
            const int audio_bos_token_id = 151687;
            std::string token_file = tts_output_dir + "/tts_audio_tokens_chunk_" + std::to_string(chunk_idx) + ".txt";
            FILE *f = fopen(token_file.c_str(), "w");
            if (f) {
                fprintf(f, "Text: %s\n", response.c_str());
                fprintf(f, "Audio tokens (%zu) [relative_index]:\n", audio_tokens.size());
                for (size_t i = 0; i < audio_tokens.size(); ++i) {
                    int absolute_id = audio_tokens[i];
                    int relative_idx = absolute_id - audio_bos_token_id;
                    // Verify token is in valid range
                    if (absolute_id < audio_bos_token_id || relative_idx >= 6562) {
                        LOG_ERR("TTS: WARNING - token %d (relative_idx=%d) is outside valid range [%d, %d)\n", 
                                absolute_id, relative_idx, audio_bos_token_id, audio_bos_token_id + 6562);
                        // Still save relative index, but mark as invalid
                        relative_idx = -1;  // Mark as invalid
                    }
                    // Save relative index only (for token2wav compatibility)
                    fprintf(f, "%d", relative_idx);
                    if (i < audio_tokens.size() - 1) fprintf(f, ",");
                }
                fprintf(f, "\n");
                fclose(f);
            } else {
                LOG_ERR("Failed to open file for writing: %s\n", token_file.c_str());
            }
        }
        
        // Create final unit_buffer for remaining tokens in buffer (if any)
        if (!audio_token_buffer.empty()) {
            // Push remaining tokens to T2W queue (final chunk)
            if (ctx_omni->t2w_thread_info) {
                T2WOut *t2w_out = new T2WOut();
                t2w_out->audio_tokens = audio_token_buffer;  // Copy remaining tokens
                t2w_out->is_final = (audio_gen_finish || llm_finish);  // Mark as final if generation finished
                t2w_out->round_meta = omni_session_round_meta(ctx_omni);
                
                {
                    std::unique_lock<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    // Check if queue has space
                    while (ctx_omni->t2w_thread_info->queue.size() >= ctx_omni->t2w_thread_info->MAX_QUEUE_SIZE) {
                        lock.unlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        lock.lock();
                    }
                    ctx_omni->t2w_thread_info->queue.push(t2w_out);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
            }
            
            struct unit_buffer *res = new unit_buffer();
            res->text = response;
            res->unit_n_past = ctx_omni->n_past;
            res->completed = (audio_gen_finish || llm_finish);
            
            // TODO: Convert audio tokens to audio waveform
            res->buffer.resize(audio_token_buffer.size() * 100);  // Placeholder size
            
            if (!res->buffer.empty()) {
                // 读写使用 mutex 保护
                std::unique_lock<std::mutex> lock(ctx_omni->workers.buffer_mutex);
                ctx_omni->omni_output->output.push_back(res);
                ctx_omni->omni_output->idx += 1;
            } else {
                delete res;
            }
        }
        
        // Clear processed text after generating audio tokens for this chunk
        // This ensures we wait for new text in the next iteration instead of reprocessing the same text
        // Python: self.text_start_pos += condition_length + len(chunk_generated_tokens)
        // C++: tts_n_past is already updated during prefill (condition_length) and decode (len(chunk_generated_tokens))
        // So tts_n_past already equals text_start_pos after this chunk
        int condition_length_final = (int)(tts_condition_embeddings.empty() ? input_tokens.size() : (tts_condition_embeddings.size() / tts_n_embd));
        int chunk_generated_tokens_count = (int)audio_tokens.size();
        fflush(stdout);
        
        llm_text.clear();
        response.clear();
        ++chunk_idx;
        
        if (ctx_omni->speek_done) {
            // 可能是外部强行打断了，清除内部所有状态
            printf("speek done setted by outside, clear all internal states\n");
            tts_finish = true;
            // Reset TTS model state by clearing KV cache
            // Note: This might need llama_kv_cache_clear or similar function
            tts_n_past = 0;
        }
        
        // Only finish and set speek_done when BOTH LLM and TTS have finished
        // If only audio_gen_finish is true but llm_finish is false, continue to next chunk
        if (llm_finish && (tts_finish || audio_gen_finish)) {
            // Save all collected audio tokens to a summary file
            if (!all_audio_tokens.empty()) {
                // Always save to fixed output directory
                // Note: Save relative indices (0-6561) for token2wav compatibility
                const int audio_bos_token_id = 151687;
                std::string summary_file = tts_output_dir + "/tts_all_audio_tokens_summary.txt";
                FILE *f = fopen(summary_file.c_str(), "w");
                if (f) {
                    fprintf(f, "Total audio tokens: %zu\n", all_audio_tokens.size());
                    fprintf(f, "Audio token range: relative [0, %d) (absolute [%d, %d))\n", 
                            6562, audio_bos_token_id, audio_bos_token_id + 6562);
                    fprintf(f, "Audio tokens [relative_index]:\n");
                    int invalid_count = 0;
                    for (size_t i = 0; i < all_audio_tokens.size(); ++i) {
                        int absolute_id = all_audio_tokens[i];
                        int relative_idx = absolute_id - audio_bos_token_id;
                        // Verify token is in valid range
                        if (absolute_id < audio_bos_token_id || relative_idx >= 6562 || relative_idx < 0) {
                            LOG_ERR("TTS: WARNING - token %d (relative_idx=%d) is outside valid range [%d, %d)\n", 
                                    absolute_id, relative_idx, audio_bos_token_id, audio_bos_token_id + 6562);
                            relative_idx = -1;  // Mark as invalid
                            invalid_count++;
                        }
                        // Save relative index only (for token2wav compatibility)
                        fprintf(f, "%d", relative_idx);
                        if (i < all_audio_tokens.size() - 1) {
                            if ((i + 1) % 20 == 0) fprintf(f, "\n");
                            else fprintf(f, " ");
                        }
                    }
                    fprintf(f, "\n");
                    if (invalid_count > 0) {
                        fprintf(f, "WARNING: Found %d tokens outside valid range [%d, %d)\n", 
                                invalid_count, audio_bos_token_id, audio_bos_token_id + 6562);
                    }
                    fclose(f);
                    fflush(stdout);
                } else {
                    LOG_ERR("Failed to open summary file for writing: %s\n", summary_file.c_str());
                }
            }
            
            // 🔧 [与 Python 流式双工对齐] 发送 is_final=true 到 T2W 队列，flush 剩余 buffer
            // Python: is_last_chunk=True 时会 flush 所有剩余 tokens
            // T2W 端 buffer 已经累积了所有 tokens，这里只是通知它 flush
            if (ctx_omni->t2w_thread_info && !all_audio_tokens.empty()) {
                T2WOut *t2w_out = new T2WOut();
                t2w_out->audio_tokens.clear();  // 空tokens，只是通知final
                t2w_out->is_final = true;
                t2w_out->round_meta = omni_session_round_meta(ctx_omni);
                
                {
                    std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    ctx_omni->t2w_thread_info->queue.push(t2w_out);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
            }
            
            // 旧代码（已禁用，改用T2W线程处理）
            /*
            if (ctx_omni->token2wav_initialized && ctx_omni->token2wav_session && 
                !ctx_omni->token2wav_buffer.empty()) {
                // ... 旧的同步处理代码 ...
            }
            */
            
            // 兼容旧代码的变量定义（避免编译错误）
            bool _unused_final_wav_placeholder = false;
            if (_unused_final_wav_placeholder && ctx_omni->token2wav_initialized && ctx_omni->token2wav_session && 
                !ctx_omni->token2wav_buffer.empty()) {
                std::vector<float> final_wav;
                if (ctx_omni->token2wav_session->feed_window(ctx_omni->token2wav_buffer, true, final_wav)) {
                    if (!final_wav.empty()) {
                        const int sample_rate = omni::flow::Token2Wav::kSampleRate;
                        std::string wav_path = tts_wav_output_dir + "/wav_" + 
                                               std::to_string(ctx_omni->token2wav_wav_idx) + ".wav";
                        
                        // Convert float to int16 and write WAV
                        const int16_t num_channels = 1;
                        const int16_t bits_per_sample = 16;
                        const int16_t block_align = num_channels * (bits_per_sample / 8);
                        const int32_t byte_rate = sample_rate * block_align;
                        
                        std::vector<int16_t> pcm(final_wav.size());
                        for (size_t i = 0; i < final_wav.size(); ++i) {
                            float x = final_wav[i];
                            if (!std::isfinite(x)) x = 0.0f;
                            x = std::max(-1.0f, std::min(1.0f, x));
                            float y = x * 32767.0f;
                            if (y >= 32767.0f) pcm[i] = 32767;
                            else if (y <= -32768.0f) pcm[i] = -32768;
                            else pcm[i] = (int16_t)y;
                        }
                        
                        uint32_t data_bytes = (uint32_t)(pcm.size() * sizeof(int16_t));
                        uint32_t riff_size = 36u + data_bytes;
                        
                        FILE* f_wav = fopen(wav_path.c_str(), "wb");
                        if (f_wav) {
                            fwrite("RIFF", 1, 4, f_wav);
                            fwrite(&riff_size, 4, 1, f_wav);
                            fwrite("WAVE", 1, 4, f_wav);
                            fwrite("fmt ", 1, 4, f_wav);
                            uint32_t fmt_size = 16;
                            uint16_t audio_format = 1;
                            fwrite(&fmt_size, 4, 1, f_wav);
                            fwrite(&audio_format, 2, 1, f_wav);
                            fwrite(&num_channels, 2, 1, f_wav);
                            fwrite(&sample_rate, 4, 1, f_wav);
                            fwrite(&byte_rate, 4, 1, f_wav);
                            fwrite(&block_align, 2, 1, f_wav);
                            fwrite(&bits_per_sample, 2, 1, f_wav);
                            fwrite("data", 1, 4, f_wav);
                            fwrite(&data_bytes, 4, 1, f_wav);
                            fwrite(pcm.data(), 1, data_bytes, f_wav);
                            fclose(f_wav);
                        }
                        ctx_omni->token2wav_wav_idx++;
                    }
                }
                
                // Reset buffer for next round (re-initialize with 3 silence tokens)
                ctx_omni->token2wav_buffer.clear();
                ctx_omni->token2wav_buffer = {4218, 4218, 4218};
                ctx_omni->token2wav_wav_idx = 0;
            }
            
            tts_finish = false;
            chunk_idx = 0;
            llm_finish = false;
            tts_n_past = 0;  // Reset TTS model state
            audio_tokens.clear();
            all_audio_tokens.clear();  // Clear for next round
            printf("\ntts finished\n");

            ctx_omni->speek_done = true;
            ctx_omni->warmup_done = true;  // 第一轮对话结束，后续 prefill 需要等待
            ctx_omni->workers.speek_cv.notify_all();
        } else if (audio_gen_finish && !llm_finish) {
            // Current chunk finished (either reached 25 tokens or EOS token) but LLM is still generating
            // Reset audio_gen_finish to allow processing next chunk from LLM
            // This ensures: LLM生成10个token → TTS生成25个token → LLM再生成10个token → TTS再生成25个token
            fflush(stdout);
            audio_gen_finish = false;  // Reset for next chunk
            // Note: llm_text and response are already cleared above, so we'll wait for next chunk in the next iteration
        }
        #endif  // End of disabled old TTS processing code
    }
}

bool stream_prefill(struct omni_context * ctx_omni, std::string aud_fname, std::string img_fname, int index, int max_slice_nums) {
    if (ctx_omni->duplex_mode) {
        duplex_timing_set_active_chunk(ctx_omni, index);
    }

    const OmniPrefillSetup setup = omni_turn_coordinator_prepare_prefill(ctx_omni, index);
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
    print_with_timestamp("\n\nc++ finish stream_prefill(index=%d). n_past=%d, n_keep=%d, n_ctx=%d\n\n",
                         index, ctx_omni->n_past, ctx_omni->n_keep, ctx_omni->params->n_ctx);
    return true;
}

struct LlmDecodeRequest {
    std::string debug_dir;
    int round_idx = -1;
};

struct LlmDecodeRuntime {
    int max_tgt_len = 0;
    int step_size = 10;
    int llm_n_embd = 0;
    int generated_decode_tokens = 0;
    int current_chunk_tokens = 0;
    bool llm_finish = false;
    bool llm_first_token_logged = false;
};

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

// Step F: build the decode prefix inside the LLM stage protocol builder.
static std::string omni_build_decode_prefix(const struct omni_context * ctx_omni) {
    if (ctx_omni->duplex_mode) {
        return "";
    }
    if (ctx_omni->use_tts) {
        return "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>";
    }
    return "<|im_end|>\n<|im_start|>assistant\n";
}

static void omni_apply_decode_prefix(struct omni_context * ctx_omni, const std::string & prompt) {
    if (prompt.empty()) {
        print_with_timestamp("stream_decode: 双工模式，跳过 assistant prompt\n");
        return;
    }

    if (ctx_omni->use_tts) {
        print_with_timestamp("📍 [单工TTS] 添加 assistant prompt: \"%s\", n_past=%d\n",
                            prompt.c_str(), ctx_omni->n_past);
    }

    eval_string(ctx_omni, ctx_omni->params, prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);

    if (ctx_omni->use_tts) {
        print_with_timestamp("📍 [单工TTS] assistant prompt 完成, n_past=%d\n", ctx_omni->n_past);
    }
}

static LlmDecodeRuntime omni_init_decode_runtime(struct omni_context * ctx_omni) {
    LlmDecodeRuntime runtime;
    runtime.max_tgt_len = ctx_omni->params->n_predict < 0 ? ctx_omni->params->n_ctx : ctx_omni->params->n_predict;
    runtime.llm_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_llama));
    print_with_timestamp("LLM decode: max_tgt_len = %d, n_predict = %d, n_ctx = %d\n",
                         runtime.max_tgt_len, ctx_omni->params->n_predict, ctx_omni->params->n_ctx);
    return runtime;
}

static void omni_mark_decode_turn_end(
        struct omni_context * ctx_omni,
        OmniTokenType token_type,
        bool & is_end_of_turn) {
    if (!ctx_omni->duplex_mode) {
        return;
    }

    if (token_type == OmniTokenType::TURN_EOS ||
        token_type == OmniTokenType::TTS_EOS ||
        token_type == OmniTokenType::EOS) {
        is_end_of_turn = true;
        ctx_omni->current_turn_ended = true;
        print_with_timestamp("LLM Duplex: turn_eos detected (type=%d), "
                            "set is_end_of_turn=true (not breaking, wait for chunk_eos)\n",
                            (int) token_type);
    }
}

static void omni_handle_decode_end_token(
        struct omni_context * ctx_omni,
        OmniTokenType token_type) {
    if (!ctx_omni->duplex_mode) {
        ctx_omni->llm_generation_done.store(true);
        print_with_timestamp("LLM: detected end token, set llm_generation_done=true\n");
    }

    if (token_type == OmniTokenType::TURN_EOS ||
        token_type == OmniTokenType::TTS_EOS ||
        token_type == OmniTokenType::EOS) {
        ctx_omni->current_turn_ended = true;
    }

    if (token_type == OmniTokenType::LISTEN && ctx_omni->duplex_mode) {
        ctx_omni->ended_with_listen = true;

        if (ctx_omni->async) {
            std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
            ctx_omni->text_queue.push_back("__IS_LISTEN__");
            ctx_omni->text_cv.notify_all();
        }
    }
}

static void omni_strip_decode_special_tokens(std::string & response) {
    static const std::vector<std::string> end_token_strings = {
        "<|tts_eos|>",
        "</s>",
        "<|listen|>",
        "<|turn_eos|>",
        "<|chunk_eos|>",
        "<|chunk_tts_eos|>",
    };

    for (const auto & delimiter : end_token_strings) {
        const size_t end = response.find(delimiter);
        if (end != std::string::npos) {
            response = response.substr(0, end);
        }
    }

    size_t speak_pos = response.find("<|speak|>");
    while (speak_pos != std::string::npos) {
        response.erase(speak_pos, std::string("<|speak|>").length());
        speak_pos = response.find("<|speak|>");
    }
}

static void omni_publish_decode_response(
        struct omni_context * ctx_omni,
        const std::string & response) {
    if (response.empty()) {
        return;
    }

    std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
    ctx_omni->text_queue.push_back(response);
    ctx_omni->text_cv.notify_all();
}

static void omni_dispatch_decode_chunk_to_tts(
        struct omni_context * ctx_omni,
        const LlmDecodeRequest & request,
        const std::string & response,
        const std::vector<llama_token> & chunk_token_ids,
        const std::vector<float> & chunk_hidden_states,
        bool llm_finish,
        bool is_end_of_turn,
        int llm_n_embd) {
    if (!ctx_omni->async ||
        !ctx_omni->use_tts ||
        ctx_omni->tts_thread_info == nullptr ||
        (response.empty() && !llm_finish)) {
        return;
    }

    LLMOut * llm_out = new LLMOut();
    llm_out->text = response;
    llm_out->n_past = ctx_omni->n_past;
    llm_out->llm_finish = llm_finish;
    llm_out->debug_dir = request.debug_dir;
    llm_out->round_meta = omni_session_round_meta(ctx_omni);
    llm_out->token_ids = chunk_token_ids;
    llm_out->hidden_states = chunk_hidden_states;
    llm_out->n_embd = llm_n_embd;
    llm_out->is_end_of_turn = is_end_of_turn;
    llm_out->duplex_chunk_idx = duplex_timing_get_active_chunk(ctx_omni);

    {
        std::string token_ids_str;
        for (size_t i = 0; i < chunk_token_ids.size() && i < 20; ++i) {
            token_ids_str += std::to_string(chunk_token_ids[i]);
            if (i + 1 < chunk_token_ids.size() && i < 19) {
                token_ids_str += " ";
            }
        }
        if (chunk_token_ids.size() > 20) {
            token_ids_str += "...";
        }

        print_with_timestamp("LLM->TTS: text='%s', n_tokens=%zu, hidden_size=%zu, n_embd=%d, token_ids=[%s]\n",
                            response.c_str(),
                            chunk_token_ids.size(),
                            chunk_hidden_states.size(),
                            llm_n_embd,
                            token_ids_str.c_str());
    }

    std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
    ctx_omni->tts_thread_info->cv.wait(lock, [&] {
        return ctx_omni->tts_thread_info->queue.size() <
            static_cast<size_t>(ctx_omni->tts_thread_info->MAX_QUEUE_SIZE);
    });

    if (!ctx_omni->speek_done || ctx_omni->duplex_mode) {
        ctx_omni->tts_thread_info->queue.push(llm_out);
        ctx_omni->tts_thread_info->cv.notify_all();
    } else {
        delete llm_out;
    }
}

static void omni_finish_decode_text_stream(struct omni_context * ctx_omni) {
    std::lock_guard<std::mutex> tl(ctx_omni->text_mtx);
    if (!ctx_omni->duplex_mode || !ctx_omni->ended_with_listen) {
        ctx_omni->text_queue.push_back("__END_OF_TURN__");
    }

    ctx_omni->text_done_flag = true;
    ctx_omni->text_cv.notify_all();
    ctx_omni->text_streaming = false;
}

bool stream_decode(struct omni_context * ctx_omni, std::string debug_dir, int round_idx) {
    if (!omni_can_decode(ctx_omni)) {
        return false;
    }

    const LlmDecodeRequest request = { std::move(debug_dir), round_idx };
    const OmniWorkerThreadFns worker_fns = {
        llm_thread_func,
        tts_thread_func,
        tts_thread_func_duplex,
        t2w_thread_func,
    };
    omni_turn_coordinator_prepare_decode(ctx_omni, request.round_idx, worker_fns);
    omni_apply_decode_prefix(ctx_omni, omni_build_decode_prefix(ctx_omni));

    LOG_INF("<user>%s\n", ctx_omni->params->prompt.c_str());
    LOG_INF("<assistant>");
    LlmDecodeRuntime runtime = omni_init_decode_runtime(ctx_omni);
    std::string response = "";

    for (; runtime.generated_decode_tokens < runtime.max_tgt_len; ) {
        if (ctx_omni->break_event.load()) {
            runtime.llm_finish = true;
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
        // 收集当前chunk的token IDs和hidden states用于TTS条件生成
        // 🔧 [优化] 只收集有效的 TTS token，确保每次给 TTS 的都是 step_size 个有效 token
        std::vector<llama_token> chunk_token_ids;
        std::vector<float> chunk_hidden_states;
        const int llm_n_embd = runtime.llm_n_embd;
        bool local_is_end_of_turn = false;
        
        // 🔧 [单双工适配] chunk 限制只在双工模式下生效
        // - 双工模式: 每个 chunk 最多 max_new_speak_tokens_per_chunk 个 tokens，便于及时响应打断
        // - 单工模式: 无限制，LLM 生成直到 EOS
        int max_chunk_tokens = ctx_omni->duplex_mode ? ctx_omni->max_new_speak_tokens_per_chunk : 0;
        bool chunk_limit_reached = (max_chunk_tokens > 0 && runtime.current_chunk_tokens >= max_chunk_tokens);
        {
            fflush(stdout);
            // 🔧 [重要] 循环直到收集到 step_size 个有效 token，而不是生成 step_size 个 token
            // 🔧 [P0-打断检测] 检测 break_event，支持双工模式下的打断
            // 🔧 [P2-chunk限制] 检测 max_new_speak_tokens_per_chunk，便于及时响应打断
            while (jl < runtime.step_size && !runtime.llm_finish && !ctx_omni->break_event.load() && !chunk_limit_reached) {
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
                
                // 🔧 [过滤逻辑] 只收集有效的 TTS token
                // 特殊 token（如 <think>, </think>, 换行等）不计入 step_size
                if (tmp != nullptr && hidden_states != nullptr) {
                    if (omni_tts_is_valid_token(sampled_token)) {
                        // 有效 token：收集并计入计数
                        chunk_token_ids.push_back(sampled_token);
                        chunk_hidden_states.insert(chunk_hidden_states.end(), hidden_states, hidden_states + llm_n_embd);
                        jl++;  // 只有有效 token 才增加计数
                        
                        // 🔧 [调试] 打印收集的 token 和 hidden states 摘要
                        
                        // 🔧 [P2-chunk限制] 更新当前 chunk 的 token 计数
                        runtime.current_chunk_tokens++;
                        
                        // 检查是否达到 chunk 限制
                        if (max_chunk_tokens > 0 && runtime.current_chunk_tokens >= max_chunk_tokens) {
                            chunk_limit_reached = true;
                        }
                    } else {
                        // 🔧 [调试] 打印被过滤的 token
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
                if (!runtime.llm_first_token_logged) {
                    runtime.llm_first_token_logged = true;
                }
                if (tmp == nullptr) {
                    LOG_ERR("llama_loop returned nullptr!");
                    break;
                }
                
                // 🔧 [调试日志] 记录每个生成的 token 到文件
                
                // 🔧 [使用 token ID 检测] 使用缓存的 token ID 进行检测，比字符串比较更高效
                OmniTokenType token_type = omni_get_token_type(ctx_omni, sampled_token);
                if (token_type != OmniTokenType::NORMAL) {
                }

                omni_mark_decode_turn_end(ctx_omni, token_type, local_is_end_of_turn);
                
                if (omni_is_end_token(ctx_omni, sampled_token)){
                    runtime.llm_finish = true;
                    omni_handle_decode_end_token(ctx_omni, token_type);
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
        
        // 🔧 [P2-chunk限制] 如果达到 chunk 限制，结束当前 decode
        // 这与 Python 双工 server 行为一致：每次 generate 只返回一个 chunk
        // 客户端需要再次调用 stream_decode 获取下一个 chunk
        if (chunk_limit_reached) {
            
            // 🔧 [P0-修复] 与 Python 对齐：达到 chunk 限制时，强制添加 <|chunk_eos|> token
            // Python: self.decoder.feed(self.decoder.embed_token(self.chunk_eos_token_id))
            if (ctx_omni->special_token_chunk_eos >= 0) {
                std::lock_guard<std::mutex> llama_lock(ctx_omni->llama_mtx);
                // Feed chunk_eos token to model (update KV cache)
                std::vector<llama_token> chunk_eos_tokens = {ctx_omni->special_token_chunk_eos};
                eval_tokens(ctx_omni, ctx_omni->params, chunk_eos_tokens, 
                           ctx_omni->params->n_batch, &ctx_omni->n_past);
            }
            // 这样 SSE 流会结束，客户端可以再次调用 decode
            runtime.llm_finish = true;
            // 注意：不重置 current_chunk_tokens，下次 decode 会从 0 开始
            runtime.current_chunk_tokens = 0;
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
        runtime.generated_decode_tokens += total_tokens_generated;
        omni_strip_decode_special_tokens(response);
        omni_publish_decode_response(ctx_omni, response);
        omni_dispatch_decode_chunk_to_tts(
            ctx_omni,
            request,
            response,
            chunk_token_ids,
            chunk_hidden_states,
            runtime.llm_finish,
            local_is_end_of_turn,
            llm_n_embd);
        fflush(stdout);
        if (runtime.llm_finish) break;
    }
    fflush(stdout);
    omni_finish_decode_text_stream(ctx_omni);
    omni_turn_coordinator_close(ctx_omni, OmniTurnCloseKind::finish);
    return true;
}

bool stop_speek(struct omni_context * ctx_omni){
    omni_turn_coordinator_close(ctx_omni, OmniTurnCloseKind::abort, "stop_speek");
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
        
        // 🔧 [#39 滑动窗口] 重置滑窗状态，但保留 n_keep 对应的 system prompt 区间
        sliding_window_reset_after_kvcache_clean(ctx_omni);
        print_with_timestamp("🧹 clean_kvcache: 滑窗状态已重置, system_preserve_length=%d\n", ctx_omni->system_preserve_length);
    } else {
        print_with_timestamp("🧹 clean_kvcache: clean_kvcache=false, 跳过清理\n");
    }
    
    return true;
}
