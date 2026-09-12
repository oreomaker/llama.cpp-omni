#pragma once

#include "multi_user_multictx_runtime.h"

// Concrete LLM engines are private to multi_user_llm_engine.cpp. The runtime
// creates them through these factories and keeps media/downstream scheduling
// outside the shared llama_context owner thread.

using OmniLlmPrefillFn = std::function<bool(const OmniLlmEngineRequest &)>;
using OmniLlmDecodeFn = std::function<bool(const OmniLlmRequestKey &)>;
using OmniLlmPollFn = std::function<bool(const OmniLlmRequestKey &, OmniLlmEngineResult &, int)>;
using OmniLlmDrainFn = std::function<bool(int)>;

std::shared_ptr<OmniLlmEngine> omni_create_continuous_batch_llm_engine(
    std::shared_ptr<OmniSharedModelBundle> bundle,
    const common_params & params,
    const OmniMultiUserConfig & config);

std::unique_ptr<OmniLlmEngine> omni_create_per_user_llm_engine_adapter(
    uint64_t user_id,
    llama_context * ctx,
    std::atomic<int32_t> * n_past,
    std::atomic<uint64_t> * owner_thread_id,
    OmniLlmPrefillFn prefill_fn,
    OmniLlmDecodeFn decode_fn,
    OmniLlmPollFn poll_fn,
    OmniLlmDrainFn drain_fn);
