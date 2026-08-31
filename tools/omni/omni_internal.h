#pragma once

#include "omni.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <vector>

// Shared implementation helpers. These are intentionally outside omni.h so
// they do not become part of the public omni API.
void print_with_timestamp(const char * format, ...);
bool cross_platform_mkdir_p(const std::string & path);
bool read_wav_pcm16_data(const std::string & wav_path, std::vector<int16_t> & pcm);

// Worker stop flags are process-wide in the current implementation. Keep the
// declarations here while worker ownership is being disentangled.
extern std::atomic<bool> t2w_thread_running;
extern std::atomic<bool> tts_thread_running;
extern std::atomic<bool> llm_thread_running;

// Token2Wav service/worker entry points used by omni_init/free and duplex.
bool start_python_t2w_service(omni_context * ctx);
void stop_python_t2w_service(omni_context * ctx);
bool send_python_t2w_command(omni_context * ctx, const std::string & cmd_json,
                             std::string & response);
bool init_python_t2w_model(omni_context * ctx, const std::string & device);
bool set_python_t2w_ref_audio(omni_context * ctx, const std::string & ref_audio_path);
bool process_python_t2w_tokens(omni_context * ctx, const std::vector<int32_t> & tokens,
                                bool last_chunk, const std::string & output_path,
                                double & inference_time_ms, double & audio_duration);
bool reset_python_t2w_cache(omni_context * ctx);
void t2w_thread_func(omni_context * ctx, common_params * params);

// TTS worker entry points (implemented in tts_worker.cpp).  The duplex/single
// TTS worker loops are part of the scheduling layer; they call the compute
// layer functions below.
void tts_thread_func_duplex(struct omni_context * ctx_omni, common_params * params);
void tts_thread_func(struct omni_context * ctx_omni, common_params * params);

// Prompt-eval 与单工 LLM 线程。
//   eval_prefix / eval_prefix_with_hidden：把 "<|im_start|>user\n" 前缀写入 KV。
//     eval_prefix 在 duplex 的 stream_prefill(index=0) 里被调用；
//     eval_prefix_with_hidden 目前无活跃调用方。
//   llm_thread_func：单工（非 duplex）模式的 LLM 线程。
void eval_prefix(struct omni_context * ctx_omni, common_params * params);
void eval_prefix_with_hidden(struct omni_context * ctx_omni, common_params * params,
                             float *& hidden_states);
void llm_thread_func(struct omni_context * ctx_omni, common_params * params);

// ============================================================================
// Compute layer (implemented in compute_llm.cpp / compute_tts.cpp / compute_enc.cpp).
//
// These are the model/graph execution operations.  They have no knowledge of
// frames, queues, or worker scheduling — the duplex/tts scheduling modules
// call them, but never call llama_decode directly.
// ============================================================================

enum class OmniTokenType {
    NORMAL,           // 普通 token (文本、audio code 等)
    SPEAK,            // <|speak|> - 开始说话
    LISTEN,           // <|listen|> - 开始听 (双工)
    CHUNK_EOS,        // <|chunk_eos|> - 语义 chunk 结束
    CHUNK_TTS_EOS,    // <|chunk_tts_eos|> - TTS chunk 结束
    TURN_EOS,         // <|turn_eos|> - 轮次结束
    TTS_EOS,          // <|tts_eos|> - 旧版 TTS 结束 (单工)
    EOS               // </s> - 序列结束
};

struct LLMOut {
    std::string text;
    int n_past;
    bool llm_finish = false;
    std::string debug_dir;
    // 添加token IDs和hidden states用于TTS条件生成
    std::vector<llama_token> token_ids;  // LLM生成的token IDs
    std::vector<float> hidden_states;   // LLM 的 hidden states (n_tokens * n_embd)
    int n_embd = 0;  // hidden states 维度

    // 该 chunk 是否为当前轮次的最后一个 chunk（随 LLMOut 传递，避免全局时序问题）
    bool is_end_of_turn = false;
    uint64_t frame_id = 0;
    int chunk_idx = -1;
    // Set immediately before queue insertion so TALKER diagnostics can
    // distinguish producer queue wait from TTS compute time.
    std::chrono::steady_clock::time_point enqueue_time{};
};

// Process-wide TTS coordination used by both the compute layer and the TTS
// worker.  Defined in omni.cpp.
struct TTSThreadInfo {
    const int MAX_QUEUE_SIZE;
    std::queue<LLMOut *> queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    int n_past = 0;

    TTSThreadInfo(int maxQueueSize) : MAX_QUEUE_SIZE(maxQueueSize) {}
};
extern std::mutex speek_mtx;
extern std::condition_variable speek_cv;
extern std::mutex buffer_mutex;
extern std::condition_variable g_decode_cv;
extern bool prefill_done;

// Special-token lookup tables used by TTS token classification (defined in compute_tts.cpp).
extern const std::vector<llama_token> g_special_token_ids;
extern const std::set<llama_token> g_known_empty_token_ids;

// Token classification.
OmniTokenType get_token_type(struct omni_context * ctx, llama_token token);
bool is_end_token(struct omni_context * ctx, llama_token token);
bool is_chunk_end_token(struct omni_context * ctx, llama_token token);
const char * get_token_type_name(OmniTokenType type);
bool is_valid_tts_token(llama_token tid);
bool is_audio_token(llama_token token, int audio_bos_token_id = 151687,
                    int num_audio_tokens = 6562);

// LLM graph ops.
bool eval_tokens(struct omni_context * ctx_omni, common_params * params,
                 std::vector<llama_token> tokens, int n_batch, int * n_past,
                 bool get_emb = false);
bool eval_tokens_with_hidden(struct omni_context * ctx_omni, common_params * params,
                             std::vector<llama_token> tokens, int n_batch, int * n_past,
                             float *& hidden_states);
bool eval_string(struct omni_context * ctx_omni, common_params * params, const char * str,
                 int n_batch, int * n_past, bool add_bos, bool get_emb = false);
bool eval_string_with_hidden(struct omni_context * ctx_omni, common_params * params,
                             const char * str, int n_batch, int * n_past, bool add_bos,
                             float *& hidden_states);
bool prefill_with_emb(struct omni_context * ctx_omni, common_params * params,
                      float * embed, int n_pos, int n_batch, int * n_past);
bool prefill_emb_with_hidden(struct omni_context * ctx_omni, common_params * params,
                             float * embed, int n_pos, int n_batch, int * n_past,
                             float *& hidden_states);
// 把 vision/audio embedding 拼成 fused buffer 一次性 prefill（fused 路径）。
// 需要 ctx_omni->emb_cache 已初始化（duplex_prepare 填充）且 hidden 匹配；失败返回 false。
// 调度层（duplex_do_prefill_one_fused）与 init 预热（omni_prewarm）共用同一入口。
bool llm_fused_prefill(struct omni_context * ctx_omni, common_params * params,
                       const std::vector<std::vector<float>> & vision_embed,
                       const std::vector<float> & audio_embed,
                       int hidden_size, int * out_fused_ntok);
const char * sample(struct common_sampler * smpl, struct omni_context * ctx_omni,
                    common_params * params, int * n_past);
const char * sample_with_hidden_and_token(struct common_sampler * smpl,
                                          struct omni_context * ctx_omni,
                                          common_params * params, int * n_past,
                                          float *& hidden_states, llama_token & token_id);
const char * llama_loop(struct omni_context * ctx_omni, common_params * params,
                        struct common_sampler * smpl, int & n_past);
const char * llama_loop_with_hidden_and_token(struct omni_context * ctx_omni,
                                              common_params * params,
                                              struct common_sampler * smpl,
                                              int & n_past, float *& hidden_states,
                                              llama_token & token_id);
std::mt19937 * get_sampler_rng(struct common_sampler * smpl);

// KV sliding window.
void kv_cache_slide_window(struct omni_context * ctx_omni, common_params * params,
                           int chunk_size);
void sliding_window_reset(struct omni_context * ctx_omni);
int sliding_window_register_unit_start(struct omni_context * ctx_omni);
void sliding_window_register_unit_end(struct omni_context * ctx_omni,
                                      const std::string & input_type,
                                      const std::vector<llama_token> & generated_tokens,
                                      bool is_listen);
void sliding_window_register_system_prompt(struct omni_context * ctx_omni);
bool sliding_window_drop_tokens_from_cache(struct omni_context * ctx_omni, int length);
bool sliding_window_enforce(struct omni_context * ctx_omni);

// Encoder ops.
struct omni_embed * omni_image_embed_make_with_filename(struct vision_ctx * ctx_vision,
                                                        int n_threads, std::string image_path);
bool omni_image_embed_make_chunks_with_filename(struct vision_ctx * ctx_vision,
                                                int n_threads, std::string image_path,
                                                std::vector<std::vector<float>> & vision_chunks);
struct omni_embed * omni_audio_embed_make_with_filename(struct audition_ctx * ctx_audio,
                                                        int n_threads, std::string audio_path);
void omni_embed_free(struct omni_embed * emb);

// TTS compute ops.
bool prefill_with_emb_tts(struct omni_context * ctx_omni, common_params * params,
                          float * embed, int n_pos, int n_batch, int * n_past_tts);
bool tts_emb_text(struct omni_context * ctx_omni, llama_token token_id,
                  float * embedding_out, int tts_n_embd);
bool tts_projector_semantic(struct omni_context * ctx_omni,
                            const float * llm_hidden_states, int n_tokens, int llm_n_embd,
                            float * projected_hidden_states, int tts_n_embd);
llama_token sample_tts_token(struct common_sampler * smpl, struct omni_context * ctx_omni,
                             common_params * params, int * n_past_tts,
                             const std::vector<llama_token> * all_generated_tokens,
                             const std::vector<llama_token> * chunk_generated_tokens,
                             int token_index_in_chunk, bool force_no_eos,
                             bool is_final_text_chunk = false);
llama_token sample_tts_token_simplex(struct common_sampler * smpl,
                                     struct omni_context * ctx_omni, common_params * params,
                                     int * n_past_tts,
                                     const std::vector<llama_token> * all_generated_tokens,
                                     int token_index_in_chunk, bool force_no_eos = false,
                                     bool is_final_text_chunk = false);
void normalize_l2_per_token(float * embeddings, int n_tokens, int n_embd, float eps = 1e-8f);
size_t findIncompleteUtf8(const std::string & str);
bool generate_audio_tokens_local(struct omni_context * ctx_omni, common_params * params,
                                 const std::vector<float> & merged_embeddings,
                                 int n_tokens, int tts_n_embd, int chunk_idx,
                                 std::vector<int32_t> & output_audio_tokens,
                                 bool is_end_of_turn, const std::string & output_dir,
                                 std::chrono::steady_clock::time_point queue_enter,
                                 uint64_t frame_id);
bool generate_audio_tokens_local_simplex(struct omni_context * ctx_omni, common_params * params,
                                         const std::vector<float> & merged_embeddings,
                                         int n_tokens, int tts_n_embd, int chunk_idx,
                                         std::vector<int32_t> & output_audio_tokens,
                                         const std::string & output_dir,
                                         bool is_final_text_chunk);
bool load_tts_weights_from_gguf(struct omni_context * ctx_omni, const char * tts_model_path);
