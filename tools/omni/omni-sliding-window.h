#pragma once

#include "llama.h"

#include <string>
#include <vector>

struct omni_context;
struct common_params;

void kv_cache_slide_window(struct omni_context * ctx_omni, struct common_params * params, int chunk_size);

int sliding_window_register_unit_start(struct omni_context * ctx_omni);
void sliding_window_register_unit_end(
        struct omni_context * ctx_omni,
        const std::string & input_type,
        const std::vector<llama_token> & generated_tokens = {},
        bool is_listen = false);
void sliding_window_register_system_prompt(struct omni_context * ctx_omni);

bool sliding_window_enforce(struct omni_context * ctx_omni);
bool sliding_window_drop_tokens_from_cache(struct omni_context * ctx_omni, int length);
void sliding_window_reset(struct omni_context * ctx_omni);
void sliding_window_reset_after_kvcache_clean(struct omni_context * ctx_omni);

void omni_finalize_decode_round(struct omni_context * ctx_omni);
