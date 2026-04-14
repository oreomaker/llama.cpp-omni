#pragma once

#include <string>

struct omni_context;
struct common_params;
struct llama_model;
struct llama_context;

// ── Prompt templates ──────────────────────────────────────────────────────
// Fill ctx_omni->audio/omni voice_clone_prompt and assistant_prompt.
void omni_init_prompt_templates(struct omni_context * ctx_omni, bool duplex_mode);

// ── Init helpers (return false on fatal error) ────────────────────────────

// Load or reuse LLM model + context; init sampler.
bool omni_init_llm_runtime(struct omni_context * ctx_omni,
                            struct common_params * params,
                            struct llama_model *   existing_model,
                            struct llama_context * existing_ctx);

// Load TTS model + context + sampler + aux weights + projector runtime.
// Only call when use_tts is true and params->tts_model is non-empty.
bool omni_init_tts_runtime(struct omni_context *  ctx_omni,
                            struct common_params * params,
                            const std::string &   tts_bin_dir,
                            int                   tts_gpu_layers);

// Init APM (always) and VPM (when media_type == 2).
// Sets ctx_omni->session.n_past = 0.
bool omni_init_audio_vision_runtime(struct omni_context *  ctx_omni,
                                     struct common_params * params);

// Init C++ and (optionally) Python Token2Wav session.
// Non-fatal: failure is logged and gracefully degraded.
// Only call when use_tts is true.
void omni_init_token2wav_runtime(struct omni_context * ctx_omni,
                                  const std::string &  tts_bin_dir,
                                  const std::string &  token2wav_device);

// ── Shutdown + release helpers ────────────────────────────────────────────

// Signal all workers to stop, then join all worker threads.
void omni_shutdown_worker_threads(struct omni_context * ctx_omni);

// Free vision context (delete) and audition context.
void omni_release_audio_vision_runtime(struct omni_context * ctx_omni);

// Free TTS model/context/sampler/aux weights/projector runtime/token2wav session.
// Only call when use_tts is true.
void omni_release_tts_runtime(struct omni_context * ctx_omni);

// Free LLM context + model (when owns_model) and sampler.
void omni_release_llm_runtime(struct omni_context * ctx_omni);

// Free LLMThreadInfo, TTSThreadInfo, T2WThreadInfo and audio_input_manager.
void omni_release_thread_info(struct omni_context * ctx_omni);
