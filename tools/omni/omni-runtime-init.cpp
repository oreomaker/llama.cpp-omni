#include "omni-runtime-init.h"

#include "audition.h"
#include "common/common.h"
#include "common/sampling.h"
#include "llama.h"
#include "omni-impl.h"
#include "omni-log.h"
#include "omni-python-t2w.h"
#include "omni-worker-coordinator.h"
#include "omni.h"
#include "token2wav/token2wav-impl.h"
#include "vision.h"

#ifdef _WIN32
#    include <sys/stat.h>
#    include <sys/types.h>
#    define stat    _stat
#    define S_IFDIR _S_IFDIR
#else
#    include <sys/stat.h>
#    include <sys/types.h>
#endif

#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

// ── Static model-loading helpers (moved from omni.cpp) ───────────────────

static struct llama_model * llama_init(common_params * params, const std::string & model_path) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = common_model_params_to_llama(*params);
    llama_model *      model        = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return NULL;
    }
    return model;
}

// TTS专用模型加载 - 支持独立的GPU层数设置
static struct llama_model * llama_init_tts(common_params *     params,
                                           const std::string & model_path,
                                           int                 n_gpu_layers_override = -1) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = common_model_params_to_llama(*params);
    if (n_gpu_layers_override >= 0) {
        model_params.n_gpu_layers = n_gpu_layers_override;
    }

    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load TTS model\n", __func__);
        return NULL;
    }
    return model;
}

// ── Prompt templates ──────────────────────────────────────────────────────

void omni_init_prompt_templates(struct omni_context * ctx_omni, bool duplex_mode) {
    if (duplex_mode) {
        ctx_omni->audio_voice_clone_prompt =
            "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|><|im_end|>\n";

        ctx_omni->omni_voice_clone_prompt =
            "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|><|im_end|>\n";
    } else {
        ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt =
            "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。"
            "请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
            "<|im_end|>\n<|im_start|>user\n";

        ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt =
            "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。"
            "请用高自然度的方式和用户聊天。<|im_end|>\n<|im_start|>user\n";
    }
}

// ── LLM runtime ───────────────────────────────────────────────────────────

bool omni_init_llm_runtime(struct omni_context *  ctx_omni,
                           struct common_params * params,
                           struct llama_model *   existing_model,
                           struct llama_context * existing_ctx) {
    llama_model *   model     = nullptr;
    llama_context * ctx_llama = nullptr;

    if (existing_model != nullptr && existing_ctx != nullptr) {
        print_with_timestamp("=== omni_init: reusing existing LLM model and context\n");
        model                = existing_model;
        ctx_llama            = existing_ctx;
        ctx_omni->owns_model = false;

        llama_memory_t mem = llama_get_memory(ctx_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
            print_with_timestamp("=== omni_init: cleared LLM KV cache for mode switch\n");
        }
    } else {
        print_with_timestamp("=== omni_init: loading new LLM model\n");
        model = llama_init(params, params->model.path);
        if (model == NULL) {
            return false;
        }

        llama_context_params ctx_params = common_context_params_to_llama(*params);
        ctx_params.n_ctx                = params->n_ctx;

        ctx_llama = llama_new_context_with_model(model, ctx_params);
        if (ctx_llama == NULL) {
            LOG_ERR("%s: error: failed to create the llama_context\n", __func__);
            return false;
        }
        ctx_omni->owns_model = true;
    }

    ctx_omni->ctx_llama   = ctx_llama;
    ctx_omni->model       = model;
    ctx_omni->ctx_sampler = common_sampler_init(model, params->sampling);
    return true;
}

// ── TTS runtime ───────────────────────────────────────────────────────────

bool omni_init_tts_runtime(struct omni_context *  ctx_omni,
                           struct common_params * params,
                           const std::string &    tts_bin_dir,
                           int                    tts_gpu_layers) {
    print_with_timestamp("=== omni_init: loading TTS model\n");
    print_with_timestamp("TTS model: loading with n_gpu_layers=%d\n", tts_gpu_layers);

    llama_model * tts_model = llama_init_tts(params, params->tts_model, tts_gpu_layers);
    if (tts_model == NULL) {
        LOG_ERR("%s: error: failed to init TTS model from %s\n", __func__, params->tts_model.c_str());
        return false;
    }

    llama_context_params tts_ctx_params = common_context_params_to_llama(*params);
    tts_ctx_params.n_ctx                = params->n_ctx;

    llama_context * ctx_tts_llama = llama_new_context_with_model(tts_model, tts_ctx_params);
    if (ctx_tts_llama == NULL) {
        LOG_ERR("%s: error: failed to create the TTS llama_context\n", __func__);
        llama_free_model(tts_model);
        return false;
    }

    // 🔧 TTS流式采样参数 - 与 Python ras_sampling 对齐
    common_params_sampling tts_sampling = params->sampling;
    tts_sampling.temp                   = 0.8f;
    tts_sampling.top_p                  = 0.85f;
    tts_sampling.top_k                  = 25;
    tts_sampling.penalty_repeat         = 1.05f;
    tts_sampling.min_p                  = 0.01f;
    tts_sampling.penalty_last_n         = 16;
    struct common_sampler * tts_sampler = common_sampler_init(tts_model, tts_sampling);
    print_with_timestamp("TTS sampler: temp=%.2f, top_p=%.2f, top_k=%d, rep_penalty=%.2f\n", tts_sampling.temp,
                         tts_sampling.top_p, tts_sampling.top_k, tts_sampling.penalty_repeat);

    ctx_omni->model_tts       = tts_model;
    ctx_omni->ctx_tts_llama   = ctx_tts_llama;
    ctx_omni->ctx_tts_sampler = tts_sampler;

    print_with_timestamp("TTS: loading weights from GGUF (emb_code, emb_text, projector_semantic, head_code)...\n");
    if (!load_tts_weights_from_gguf(ctx_omni, params->tts_model.c_str())) {
        LOG_ERR("%s: error: failed to load TTS weights from %s\n", __func__, params->tts_model.c_str());
        common_sampler_free(ctx_omni->ctx_tts_sampler);
        ctx_omni->ctx_tts_sampler = nullptr;
        llama_free(ctx_omni->ctx_tts_llama);
        ctx_omni->ctx_tts_llama = nullptr;
        llama_free_model(ctx_omni->model_tts);
        ctx_omni->model_tts = nullptr;
        return false;
    }
    print_with_timestamp("TTS: weights loaded successfully\n");

    std::string projector_path = tts_bin_dir + "/MiniCPM-o-4_5-projector-F16.gguf";
    print_with_timestamp("Projector: loading from %s\n", projector_path.c_str());
    if (projector_init(ctx_omni->projector, projector_path, true)) {
        print_with_timestamp("Projector: loaded successfully\n");
    } else {
        print_with_timestamp("Projector: failed to load, will use fallback implementation\n");
    }

    return true;
}

// ── Audio/Vision runtime ──────────────────────────────────────────────────

bool omni_init_audio_vision_runtime(struct omni_context * ctx_omni, struct common_params * params) {
    if (params->apm_model.empty()) {
        LOG_ERR("%s: error: apm_model path is empty\n", __func__);
        return false;
    }

    print_with_timestamp("=== omni_init: loading APM model\n");
    ctx_omni->ctx_audio =
        audition_init(params->apm_model.c_str(), audition_context_params{ true, GGML_LOG_LEVEL_INFO });
    print_with_timestamp("APM: init from %s\n", params->apm_model.c_str());
    if (ctx_omni->ctx_audio == nullptr) {
        LOG_ERR("%s: error: failed to init audition model from %s\n", __func__, params->apm_model.c_str());
        return false;
    }

    ctx_omni->n_past = 0;

    if (ctx_omni->media_type == 2) {
        LOG_INF("init vision....");
        const char * vision_path = ctx_omni->params->vpm_model.c_str();
        auto *       ctx_vision = vision_init(vision_path, vision_context_params{ true, GGML_LOG_LEVEL_INFO, nullptr });
        ctx_omni->ctx_vision    = ctx_vision;

        if (ctx_vision && !ctx_omni->params->vision_coreml_model_path.empty()) {
            struct stat coreml_stat;
            if (stat(ctx_omni->params->vision_coreml_model_path.c_str(), &coreml_stat) == 0) {
                vision_set_coreml_model_path(ctx_vision, ctx_omni->params->vision_coreml_model_path.c_str());
                LOG_INF("Vision CoreML model path set to: %s\n", ctx_omni->params->vision_coreml_model_path.c_str());
            } else {
                LOG_WRN("Vision CoreML model path does not exist: %s, skipping ANE\n",
                        ctx_omni->params->vision_coreml_model_path.c_str());
            }
        }
    }

    return true;
}

// ── Token2Wav runtime ─────────────────────────────────────────────────────

void omni_init_token2wav_runtime(struct omni_context * ctx_omni,
                                 const std::string &   tts_bin_dir,
                                 const std::string &   token2wav_device) {
    ctx_omni->token2wav_initialized = false;

    // 🔧 如果使用 Python Token2Wav，跳过 C++ 的初始化以节省显存
    bool skip_cpp_token2wav = ctx_omni->use_python_token2wav;

    // 优先检查 HF 模型目录下的 token2wav-gguf (tts_bin_dir 的父目录)
    std::string gguf_root_dir = tts_bin_dir;
    size_t      last_slash    = gguf_root_dir.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        gguf_root_dir = gguf_root_dir.substr(0, last_slash);
    }
    ctx_omni->token2wav_model_dir = gguf_root_dir + "/token2wav-gguf";

    {
        std::string   encoder_test = ctx_omni->token2wav_model_dir + "/encoder.gguf";
        std::ifstream f(encoder_test);
        if (!f.good()) {
            ctx_omni->token2wav_model_dir = "tools/omni/token2wav-gguf";
            print_with_timestamp("Token2Wav: trying fallback path %s\n", ctx_omni->token2wav_model_dir.c_str());
        } else {
            print_with_timestamp("Token2Wav: found models in %s\n", ctx_omni->token2wav_model_dir.c_str());
        }
    }

    std::string encoder_gguf       = ctx_omni->token2wav_model_dir + "/encoder.gguf";
    std::string flow_matching_gguf = ctx_omni->token2wav_model_dir + "/flow_matching.gguf";
    std::string flow_extra_gguf    = ctx_omni->token2wav_model_dir + "/flow_extra.gguf";
    std::string vocoder_gguf       = ctx_omni->token2wav_model_dir + "/hifigan2.gguf";
    std::string prompt_cache_gguf  = ctx_omni->token2wav_model_dir + "/prompt_cache.gguf";

    bool                     all_files_exist = true;
    std::vector<std::string> gguf_files      = { encoder_gguf, flow_matching_gguf, flow_extra_gguf, vocoder_gguf };
    for (const auto & file : gguf_files) {
        std::ifstream f(file);
        if (!f.good()) {
            all_files_exist = false;
            break;
        }
    }

    if (all_files_exist && !skip_cpp_token2wav) {
        print_with_timestamp("Token2Wav: all model files found, initializing session...\n");
        ctx_omni->token2wav_session = std::make_unique<omni::flow::Token2WavSession>();

        const std::string & device_token2mel = token2wav_device;

        // Vocoder 设备策略：CUDA 跟随 token2wav_device，否则强制 CPU
        const char * voc_dev_env = getenv("OMNI_VOC_DEVICE");
        std::string  device_vocoder;
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

        std::string prompt_bundle_dir = "tools/omni/assets/default_ref_audio";
        std::string spk_file          = prompt_bundle_dir + "/spk_f32.bin";
        std::string tokens_file       = prompt_bundle_dir + "/prompt_tokens_i32.bin";
        std::string mel_file          = prompt_bundle_dir + "/prompt_mel_btc_f32.bin";

        bool use_prompt_bundle = false;
        {
            std::ifstream f1(spk_file);
            std::ifstream f2(tokens_file);
            std::ifstream f3(mel_file);
            use_prompt_bundle = f1.good() && f2.good() && f3.good();
        }

        bool init_ok = false;
        print_with_timestamp("Token2Wav: using prompt_cache from %s\n", prompt_cache_gguf.c_str());
        init_ok = ctx_omni->token2wav_session->init_from_prompt_cache_gguf(
            encoder_gguf, flow_matching_gguf, flow_extra_gguf, prompt_cache_gguf, vocoder_gguf, device_token2mel,
            device_vocoder, 5, 1.0f);
        if (!init_ok && use_prompt_bundle) {
            print_with_timestamp("Token2Wav: prompt_cache failed, fallback to prompt_bundle from %s\n",
                                 prompt_bundle_dir.c_str());
            init_ok = ctx_omni->token2wav_session->init_from_prompt_bundle(
                encoder_gguf, flow_matching_gguf, flow_extra_gguf, prompt_bundle_dir, vocoder_gguf, device_token2mel,
                device_vocoder, 5, 1.0f);
        }
        if (!init_ok) {
            print_with_timestamp("Token2Wav: GPU init failed, trying CPU mode...\n");
            ctx_omni->token2wav_session.reset();
            ctx_omni->token2wav_session = std::make_unique<omni::flow::Token2WavSession>();
            init_ok = ctx_omni->token2wav_session->init_from_prompt_cache_gguf(encoder_gguf, flow_matching_gguf,
                                                                               flow_extra_gguf, prompt_cache_gguf,
                                                                               vocoder_gguf, "cpu", "cpu", 5, 1.0f);
        }

        if (init_ok) {
            ctx_omni->token2wav_initialized = true;
            ctx_omni->token2wav_buffer.clear();
            ctx_omni->token2wav_buffer  = { 4218, 4218, 4218 };
            ctx_omni->token2wav_wav_idx = 0;
            print_with_timestamp("Token2Wav: initialized successfully\n");
        } else {
            ctx_omni->token2wav_session.reset();
            print_with_timestamp("Token2Wav: initialization failed\n");
        }
    } else {
        print_with_timestamp("Token2Wav: model files not found in %s\n", ctx_omni->token2wav_model_dir.c_str());
    }

    // ── Python Token2Wav ──────────────────────────────────────────────────

    // Python T2W 脚本目录（相对于 tts_bin_dir 推断）
    std::string t2w_script_dir = tts_bin_dir;
    size_t      convert_pos    = t2w_script_dir.find("/convert/gguf/tts");
    if (convert_pos != std::string::npos) {
        t2w_script_dir = t2w_script_dir.substr(0, convert_pos) + "/pyt2w";
    } else if ((convert_pos = t2w_script_dir.find("/convert/gguf")) != std::string::npos) {
        t2w_script_dir = t2w_script_dir.substr(0, convert_pos) + "/pyt2w";
    } else {
        t2w_script_dir = "./tools/omni/pyt2w";
    }
    ctx_omni->python_t2w_script_dir = t2w_script_dir;
    ctx_omni->python_t2w_model_dir  = t2w_script_dir + "/token2wav";

    std::string ref_audio_path = "tools/omni/assets/default_ref_audio/default_ref_audio.wav";

    print_with_timestamp("Python T2W: script_dir=%s, model_dir=%s\n", ctx_omni->python_t2w_script_dir.c_str(),
                         ctx_omni->python_t2w_model_dir.c_str());

    if (ctx_omni->use_python_token2wav) {
        print_with_timestamp("Python T2W: 使用 Python Token2Wav 实现\n");

        const char * env_python_t2w_gpu = getenv("PYTHON_T2W_GPU");
        if (env_python_t2w_gpu && strlen(env_python_t2w_gpu) > 0) {
            ctx_omni->python_t2w_dedicated_gpu = env_python_t2w_gpu;
        }

        ctx_omni->python_t2w_gpu_id = "";

        if (!ctx_omni->python_t2w_dedicated_gpu.empty()) {
            ctx_omni->python_t2w_gpu_id = ctx_omni->python_t2w_dedicated_gpu;
            print_with_timestamp("Python T2W: 使用独立 GPU %s (C++ 和 Python 分开)\n",
                                 ctx_omni->python_t2w_gpu_id.c_str());
        } else {
            const char * env_cuda_visible = getenv("CUDA_VISIBLE_DEVICES");
            if (env_cuda_visible && strlen(env_cuda_visible) > 0) {
                print_with_timestamp("Python T2W: 继承外部 CUDA_VISIBLE_DEVICES=%s (与 C++ 共用)\n", env_cuda_visible);
            } else if (token2wav_device.find("gpu") != std::string::npos) {
                size_t colon_pos = token2wav_device.find(':');
                if (colon_pos != std::string::npos) {
                    ctx_omni->python_t2w_gpu_id = token2wav_device.substr(colon_pos + 1);
                } else {
                    ctx_omni->python_t2w_gpu_id = "0";
                }
                print_with_timestamp("Python T2W: 设置 CUDA_VISIBLE_DEVICES=%s (与 C++ 共用)\n",
                                     ctx_omni->python_t2w_gpu_id.c_str());
            } else {
                print_with_timestamp("Python T2W: CPU 模式\n");
            }
        }

        if (omni_start_python_t2w_service(ctx_omni)) {
            if (omni_init_python_t2w_model(ctx_omni, token2wav_device)) {
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

        if (!ctx_omni->use_python_token2wav) {
            print_with_timestamp("Python T2W: 回退到 C++ 实现\n");
        }
    } else {
        print_with_timestamp("Token2Wav: 使用 C++ 实现\n");
    }
}

// ── Shutdown worker threads ───────────────────────────────────────────────

void omni_shutdown_worker_threads(struct omni_context * ctx_omni) {
    omni_request_worker_shutdown(ctx_omni);

    ctx_omni->workers.llm_thread_running = false;
    if (ctx_omni->llm_thread.joinable()) {
        if (ctx_omni->llm_thread_info) {
            ctx_omni->llm_thread_info->cv.notify_all();
        }
        ctx_omni->llm_thread.join();
    }

    if (ctx_omni->use_tts) {
        ctx_omni->workers.tts_thread_running = false;
        if (ctx_omni->tts_thread.joinable()) {
            if (ctx_omni->tts_thread_info) {
                ctx_omni->tts_thread_info->cv.notify_all();
            }
            ctx_omni->tts_thread.join();
        }

        ctx_omni->workers.t2w_thread_running = false;
        if (ctx_omni->t2w_thread.joinable()) {
            if (ctx_omni->t2w_thread_info) {
                ctx_omni->t2w_thread_info->cv.notify_all();
            }
            ctx_omni->t2w_thread.join();
        }
    }
}

// ── Release helpers ───────────────────────────────────────────────────────

void omni_release_audio_vision_runtime(struct omni_context * ctx_omni) {
    delete ctx_omni->ctx_vision;
    ctx_omni->ctx_vision = nullptr;
    audition_free(ctx_omni->ctx_audio);
    ctx_omni->ctx_audio = nullptr;
}

void omni_release_tts_runtime(struct omni_context * ctx_omni) {
    llama_free(ctx_omni->ctx_tts_llama);
    ctx_omni->ctx_tts_llama = nullptr;
    llama_free_model(ctx_omni->model_tts);
    ctx_omni->model_tts = nullptr;
    common_sampler_free(ctx_omni->ctx_tts_sampler);
    ctx_omni->ctx_tts_sampler = nullptr;

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

    if (ctx_omni->token2wav_session) {
        ctx_omni->token2wav_session.reset();
        ctx_omni->token2wav_initialized = false;
        LOG_INF("Token2Wav (C++): session released\n");
    }

    if (ctx_omni->python_t2w_initialized) {
        omni_stop_python_t2w_service(ctx_omni);
    }

    if (ctx_omni->projector.initialized) {
        projector_free(ctx_omni->projector);
    }
}

void omni_release_llm_runtime(struct omni_context * ctx_omni) {
    if (ctx_omni->owns_model) {
        llama_free(ctx_omni->ctx_llama);
        ctx_omni->ctx_llama = nullptr;
        llama_free_model(ctx_omni->model);
        ctx_omni->model = nullptr;
    }
    common_sampler_free(ctx_omni->ctx_sampler);
    ctx_omni->ctx_sampler = nullptr;
}

void omni_release_thread_info(struct omni_context * ctx_omni) {
    delete ctx_omni->audio_input_manager;
    ctx_omni->audio_input_manager = nullptr;

    delete ctx_omni->llm_thread_info;
    ctx_omni->llm_thread_info = nullptr;

    if (ctx_omni->use_tts) {
        delete ctx_omni->tts_thread_info;
        ctx_omni->tts_thread_info = nullptr;
        delete ctx_omni->t2w_thread_info;
        ctx_omni->t2w_thread_info = nullptr;
    }
}
