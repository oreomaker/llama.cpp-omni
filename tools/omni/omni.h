#include "ggml.h"
#include "llama.h"
#include "omni-runtime-messages.h"
#include "omni-session-state.h"
#include "omni-worker-state.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Windows compatibility: pid_t is not defined on MSVC
#ifdef _WIN32
typedef int pid_t;
#endif

struct vision_ctx;
struct audition_ctx;
struct audition_audio_f32;

// Forward declaration for C++ Token2Wav
namespace omni {
namespace flow {
class Token2WavSession;
}
}  // namespace omni

//
// omni ctx
//
struct omni_embed {
    float * embed;
    int     n_pos;
};

struct omni_embeds {
    // 🔧 [高清模式] vision_embed 改为二维 vector
    // vision_embed[0] = overview embed (64 tokens * hidden_size)
    // vision_embed[1..n] = slice embeds (各 64 tokens * hidden_size)
    std::vector<std::vector<float>> vision_embed;
    std::vector<float>              audio_embed;
    int                             index    = 0;
    int                             end_flag = false;
};

struct PipelineDecodeResult {
    // Worker-to-caller completion payload for one async decode cycle.
    bool ended_with_listen = false;
    bool decode_ok         = false;
};

struct LLMThreadInfo {
    int                                   MAX_QUEUE_SIZE;
    std::queue<omni_embeds *>             queue;
    std::mutex                            mtx;
    std::condition_variable               cv;
    // Simplex sets this explicitly; duplex auto-promotes queued prefill to decode.
    std::atomic<bool>                     decode_requested{ false };
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    LLMThreadInfo(int maxQueueSize) : MAX_QUEUE_SIZE(maxQueueSize) {}
};

struct T2WThreadInfo {
    int                                   MAX_QUEUE_SIZE;
    std::queue<T2WOut *>                  queue;
    std::mutex                            mtx;
    std::condition_variable               cv;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    T2WThreadInfo(int maxQueueSize) : MAX_QUEUE_SIZE(maxQueueSize) {}
};

struct projector_hparams {
    int32_t in_dim  = 4096;  // 输入维度 (LLM hidden size)
    int32_t out_dim = 768;   // 输出维度 (TTS embedding size)
};

struct projector_layer {
    struct ggml_tensor * linear1_weight = nullptr;  // [in_dim, out_dim]
    struct ggml_tensor * linear1_bias   = nullptr;  // [out_dim]
    struct ggml_tensor * linear2_weight = nullptr;  // [out_dim, out_dim]
    struct ggml_tensor * linear2_bias   = nullptr;  // [out_dim]
};

struct projector_model {
    projector_hparams hparams;
    projector_layer   layer;

    struct ggml_context *      ctx_w       = nullptr;
    ggml_backend_buffer_t      buf_w       = nullptr;
    ggml_backend_t             backend     = nullptr;
    ggml_backend_buffer_type_t buf_type    = nullptr;
    bool                       initialized = false;
};

struct OmniTTSAuxWeights {
    // Auxiliary tensors extracted from the TTS GGUF for omni's custom TTS path.
    float * emb_code_weight               = nullptr;
    int     emb_code_vocab_size           = 0;
    int     emb_code_hidden_size          = 0;
    bool    emb_code_stored_as_transposed = false;

    float * emb_text_weight      = nullptr;
    int     emb_text_vocab_size  = 0;
    int     emb_text_hidden_size = 0;

    float * projector_semantic_linear1_weight = nullptr;
    float * projector_semantic_linear1_bias   = nullptr;
    float * projector_semantic_linear2_weight = nullptr;
    float * projector_semantic_linear2_bias   = nullptr;
    int     projector_semantic_input_dim      = 0;
    int     projector_semantic_output_dim     = 0;

    float * head_code_weight           = nullptr;
    int     head_code_hidden_size      = 0;
    int     head_code_num_audio_tokens = 0;
};

struct OmniTTSProjectorRuntime {
    // Preferred ggml projector runtime; aux weights keep the legacy fallback tensors.
    struct projector_model projector;
};

struct omni_duplex_chunk_timing {
    double vit_embedding_ms       = -1.0;
    double audio_embedding_ms     = -1.0;
    double tts_audio_token_ms     = -1.0;
    double token2wav_ms           = -1.0;
    int    tts_audio_token_count  = 0;
    int    token2wav_window_count = 0;
    bool   tts_done               = false;
    bool   token2wav_done         = false;
};

struct omni_context {
    struct vision_ctx *   ctx_vision = NULL;
    struct audition_ctx * ctx_audio  = NULL;

    struct llama_context *  ctx_llama   = NULL;
    struct llama_model *    model       = NULL;
    struct common_sampler * ctx_sampler = NULL;

    // 🔧 [单双工适配] 是否拥有模型（用于 omni_free 时决定是否释放模型）
    // true: omni_init 内部加载的模型，omni_free 时需要释放
    // false: 外部传入的已有模型（模型复用），omni_free 时不释放
    bool owns_model = true;

    // 🔧 [Length Penalty] 用于调整 EOS token 的采样概率
    // length_penalty > 1.0 会降低 EOS 概率，让模型生成更长的输出
    // length_penalty < 1.0 会增加 EOS 概率，让模型更早结束
    float length_penalty = 1.0f;

    struct llama_context *  ctx_tts_llama   = NULL;
    struct llama_model *    model_tts       = NULL;
    struct common_sampler * ctx_tts_sampler = NULL;

    // struct TTSContext * ctx_tts = NULL;
    struct vocal_ctx *                  vocal = NULL;
    std::shared_ptr<std::vector<float>> spk_embeds;
    std::vector<float>                  audio_emb;
    std::vector<float>                  omni_emb;
    int                                 output_audio_round_per_text[5] = { 16, 8, 4, 2, 2 };
    int                                 output_audio_chunk_size[5]     = { 5, 10, 20, 40, 40 };

    struct omni_output * omni_output = NULL;
    OmniSessionState     session;
    OmniTurnState        turn;
    OmniSessionGate      gate;

    bool                   async = false;
    std::thread            llm_thread;
    std::thread            tts_thread;
    std::thread            t2w_thread;
    struct LLMThreadInfo * llm_thread_info = NULL;
    struct TTSThreadInfo * tts_thread_info = NULL;
    struct T2WThreadInfo * t2w_thread_info = NULL;
    OmniWorkerState        workers;

    // 预热标志：第一轮对话视为预热（例如音色克隆参考音频），完成后设为 true
    std::atomic<bool> warmup_done{ false };

    // ==================== 双工模式参数 ====================
    // 每个 chunk 最大生成 token 数（用于限制单次 speak 长度，便于及时响应打断）
    // 设置为 0 表示无限制
    int max_new_speak_tokens_per_chunk = 26;

    // listen_prob_scale: 调整 <|listen|> token 的采样概率
    // 1.0: Python 默认
    float listen_prob_scale = 1.0f;

    // 是否启用双工模式
    // simplex: 单工模式，用户说完后模型回复，回复完用户再说
    // duplex: 双工模式，模型可以在任意时刻决定听/说切换
    bool duplex_mode = false;

    class AudioInputManager * audio_input_manager = NULL;

    // models path and other configs
    struct common_params * params = NULL;

    // 当前是以「语音通话」还是「视频通话」模式进入的，0 = 语音，1 = 视频；
    int         media_type     = 0;
    int         use_tts        = false;
    std::string tts_bin_dir    = "";
    std::string ref_audio_path = "";  // 参考音频路径（用于音色克隆）

    // 🔧 [高清/高刷模式]
    // high_image: 高清模式，max_slice_nums 设置为 2，vision 可以看到更多细节
    // high_refresh: 高刷模式，1秒5帧，第1帧作为主图，后4帧stack合并成一张图
    //               注意：stack 处理在 Python server 层实现，C++ 只是标记
    bool high_image   = false;
    bool high_refresh = false;

    // 🔧 [多实例支持] 可配置的输出目录，避免多个服务实例冲突
    std::string base_output_dir = "./tools/omni/output";

    // 每次会话，是否清除 kv cache（默认开启自动清理 kv cache）
    bool clean_kvcache = true;

    std::string omni_voice_clone_prompt  = "";
    std::string omni_assistant_prompt    = "";
    std::string audio_voice_clone_prompt = "";
    std::string audio_assistant_prompt   = "";

    // 语言设置 (用于 prompt 生成)
    std::string language = "zh";

    // text streaming queue for server
    std::mutex              text_mtx;
    std::condition_variable text_cv;
    std::deque<std::string> text_queue;

    // Async caller stores the next decode request here before waking the LLM worker.
    std::mutex                       pipeline_request_mtx;
    std::string                      pipeline_debug_dir = "";
    int                              pipeline_round_idx = -1;
    // Async worker posts one result per completed decode cycle.
    std::mutex                       pipeline_result_mtx;
    std::condition_variable          pipeline_result_cv;
    std::queue<PipelineDecodeResult> pipeline_result_queue;

    // llama inference mutex - 保护 ctx_llama 的推理操作
    std::mutex llama_mtx;

    OmniTTSAuxWeights       tts_aux;
    OmniTTSProjectorRuntime tts_projector;

    // TTS condition embeddings (for first audio token re-forward)
    // Used to store the condition embeddings so we can re-forward them for the first audio token
    // This ensures KV cache state matches Python's behavior (past_key_values=None on first forward)
    std::vector<float> tts_condition_embeddings;      // Condition embeddings (n_tokens * n_embd)
    int                tts_condition_length = 0;      // Number of tokens in condition
    int                tts_condition_n_embd = 0;      // Embedding dimension (768)
    bool               tts_condition_saved  = false;  // Whether condition has been saved

    // 🔧 TTS KV cache 累计位置（用于保持跨 chunk 的上下文连续性）
    // Python TTSStreamingGenerator 使用 text_start_pos 来跟踪位置
    int tts_n_past_accumulated = 0;

    // 🔧 [关键修复] TTS 已生成的所有 audio tokens（跨 chunk 累积）
    // Python: self.all_generated_tokens 是类成员变量，跨 chunk 持续累积
    // 用于：1. RAS 重复检测（需要完整历史）2. 正确判断 audio_bos（只有第一个 token 才是）
    std::vector<llama_token> tts_all_generated_tokens;

    // 🔧 [与 Python 对齐] TTS audio token buffer（跨 text chunk 累积）
    // Python: self._token_buffer 是类成员变量，用于累积 audio token
    // 只有满足 chunk_size (25) 才会 yield，不足的保留到下一个 text chunk
    std::vector<int32_t> tts_token_buffer;

    // Timestamp for stream_decode start (used for WAV file naming)
    std::chrono::high_resolution_clock::time_point stream_decode_start_time;

    // Duplex per-chunk timing, used by test-duplex.cpp to print async stage costs.
    std::mutex                                        duplex_timing_mtx;
    std::unordered_map<int, omni_duplex_chunk_timing> duplex_chunk_timings;
    int                                               active_duplex_chunk_idx = -1;

    // C++ Token2Wav session for audio synthesis
    std::unique_ptr<omni::flow::Token2WavSession> token2wav_session;
    bool                                          token2wav_initialized = false;
    std::string                                   token2wav_model_dir;  // Directory containing token2wav GGUF models

    // 🔧 [Python Token2Wav] 使用 Python stepaudio2 库实现的 Token2Wav
    // 设置为 true 时使用 Python 实现（精度更高），false 时使用 C++ 实现
    // macOS 上默认使用 C++ 实现（无 CUDA）
    bool        use_python_token2wav = false;
    std::string python_t2w_script_dir;  // Python Token2Wav 脚本目录
    std::string python_t2w_model_dir;   // Python Token2Wav 模型目录

    // Python Token2Wav 服务进程 (通过 popen 启动)
    FILE *      python_t2w_stdin       = nullptr;  // 写入命令
    FILE *      python_t2w_stdout      = nullptr;  // 读取响应
    pid_t       python_t2w_pid         = -1;       // 进程 ID
    bool        python_t2w_initialized = false;
    std::string python_t2w_gpu_id;                 // GPU ID (如 "0", "1")

    // 🔧 Python T2W 独立 GPU 配置
    // C++ LLM+TTS 占用约 22GB，Python T2W 占用约 3.3GB
    // 单卡 24GB 放不下，需要使用独立 GPU
    // 设置为空字符串表示使用与 C++ 相同的 GPU
    std::string python_t2w_dedicated_gpu = "";  // 独立 GPU ID，如 "1"

    // Token2Wav sliding window buffer (跨 chunk 保持状态)
    // Python 逻辑: buffer 初始填充 3 个静音 token (4218)
    // 每次取 28 个 tokens (25 main + 3 lookahead)，处理后移动 25 个，保留 3 个重叠
    std::vector<int32_t> token2wav_buffer;
    int                  token2wav_wav_idx = 0;  // 输出 WAV 文件计数器

    // ==================== 特殊 Token ID ====================
    // 在 omni_init 时从词表查找并缓存
    llama_token special_token_speak         = -1;  // <|speak|>: 模型开始说话
    llama_token special_token_listen        = -1;  // <|listen|>: 模型开始听（双工）
    llama_token special_token_chunk_eos     = -1;  // <|chunk_eos|>: 语义 chunk 结束
    llama_token special_token_chunk_tts_eos = -1;  // <|chunk_tts_eos|>: TTS chunk 结束
    llama_token special_token_turn_eos      = -1;  // <|turn_eos|>: 轮次结束
    llama_token special_token_tts_eos       = -1;  // <|tts_eos|>: 旧版 TTS 结束
    llama_token special_token_eos           = -1;  // </s>: 序列结束
    llama_token tts_bos_token_id            = -1;  // <|tts_bos|>: TTS 开始（用于双工强制继续说话）
    llama_token special_token_unit_end      = -1;  // </unit>: unit 结束标记（双工 chunk 边界）
    llama_token special_token_tts_pad       = -1;  // <|tts_pad|>: TTS 填充（双工模式下禁止采样）

    omni_context();
    ~omni_context();

    omni_context(const omni_context &)             = delete;
    omni_context & operator=(const omni_context &) = delete;
};

//
// omni embed
//
bool prefill_with_emb(struct omni_context *  ctx_omni,
                      struct common_params * params,
                      float *                embed,
                      int                    n_pos,
                      int                    n_batch,
                      int *                  n_past);
bool prefill_emb_with_hidden(struct omni_context *  ctx_omni,
                             struct common_params * params,
                             float *                embed,
                             int                    n_pos,
                             int                    n_batch,
                             int *                  n_past,
                             float *&               hidden_states);
bool omni_eval_embed(struct llama_context * ctx_llama, const struct omni_embed * embed, int n_batch, int * n_past);
void omni_embed_free(struct omni_embed * embed);
struct omni_embed * omni_image_embed_make_with_bytes(struct vision_ctx *   ctx_vision,
                                                     int                   n_threads,
                                                     const unsigned char * image_bytes,
                                                     int                   image_bytes_length);
struct omni_embed * omni_image_embed_make_with_filename(struct vision_ctx * ctx_vision,
                                                        int                 n_threads,
                                                        const std::string & image_path);
struct omni_embed * omni_audio_embed_make_with_bytes(struct audition_ctx * ctx_audition,
                                                     int                   n_threads,
                                                     audition_audio_f32 *  audio);
struct omni_embed * omni_audio_embed_make_with_filename(struct audition_ctx * ctx_audition,
                                                        int                   n_threads,
                                                        const std::string &   audio_path);

//
// omni main
//
struct omni_context * omni_init(struct common_params * params,
                                int                    media_type,
                                bool                   use_tts,
                                const std::string &    tts_bin_dir,
                                int                    tts_gpu_layers   = -1,
                                const std::string &    token2wav_device = "gpu:0",
                                bool                   duplex_mode      = false,
                                llama_model *          existing_model   = nullptr,
                                llama_context *        existing_ctx     = nullptr,
                                const std::string &    base_output_dir  = "./tools/omni/output");

void omni_free(struct omni_context * ctx_omni);

// ANE/CoreML warmup — call once after omni_init to pre-load models into NPU
void omni_warmup_ane(struct omni_context * ctx_omni);

// 检查 TTS 和 T2W 队列是否都为空
bool omni_tts_queues_empty(struct omni_context * ctx_omni);
bool omni_get_duplex_chunk_timing(struct omni_context *             ctx_omni,
                                  int                               chunk_idx,
                                  struct omni_duplex_chunk_timing * out_timing);

// 停止所有线程（在 join 之前调用）
void omni_stop_threads(struct omni_context * ctx_omni);

bool stream_prefill(struct omni_context * ctx_omni,
                    const std::string &   aud_fname,
                    const std::string &   img_fname = "",
                    int                   index     = 0,
                    int max_slice_nums              = -1);  // -1 表示使用全局设置，>=1 表示本次 prefill 的 slice 数量

bool stream_decode(struct omni_context * ctx_omni,
                   std::string           debug_dir,
                   int                   round_idx = -1);  // round_idx: 由调用方指定的轮次索引，-1 表示使用内部计数

bool stop_speek(struct omni_context * ctx_omni);

bool clean_kvcache(struct omni_context * ctx_omni);

// TTS 推理函数声明（用于 test_tts_inference.cpp）
bool load_tts_weights_from_gguf(struct omni_context * ctx_omni, const char * tts_model_path);
bool prefill_with_emb_tts(struct omni_context * ctx_omni,
                          common_params *       params,
                          float *               embed,
                          int                   n_pos,
                          int                   n_batch,
                          int *                 n_past_tts);

// Projector 函数声明（精度验证版本）
bool               projector_init(projector_model & model, const std::string & fname, bool use_cuda);
void               projector_free(projector_model & model);
std::vector<float> projector_forward(projector_model & model, const float * input_data, int n_tokens);

// ==================== 高清模式函数声明 ====================
// 设置 vision max_slice_nums 覆盖值，用于高清模式
void vision_set_max_slice_nums(struct vision_ctx * ctx_vision, int max_slice_nums);
