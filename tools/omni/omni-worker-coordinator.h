#pragma once

struct common_params;
struct omni_context;

using OmniWorkerThreadFn = void (*)(struct omni_context *, struct common_params *);

struct OmniWorkerThreadFns {
    OmniWorkerThreadFn llm         = nullptr;
    OmniWorkerThreadFn tts_simplex = nullptr;
    OmniWorkerThreadFn tts_duplex  = nullptr;
    OmniWorkerThreadFn t2w         = nullptr;
};

void omni_clear_tts_queue(struct omni_context * ctx_omni, const char * log_reason = nullptr);
void omni_ensure_prefill_workers_started(struct omni_context * ctx_omni, const OmniWorkerThreadFns & worker_fns);
void omni_request_worker_shutdown(struct omni_context * ctx_omni);
