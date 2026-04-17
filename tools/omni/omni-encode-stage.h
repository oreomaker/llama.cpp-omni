#pragma once

#include <memory>
#include <string>

struct common_params;
struct omni_context;
struct omni_embeds;

bool omni_encode_prefill_input(struct omni_context * ctx_omni,
                               const std::string &   aud_fname,
                               const std::string &   img_fname,
                               int                   index,
                               int                   max_slice_nums,
                               struct omni_embeds &  encoded);
bool omni_submit_llm_prefill(struct omni_context * ctx_omni, std::unique_ptr<struct omni_embeds> encoded);
void omni_encode_worker_loop(struct omni_context * ctx_omni, struct common_params * params);
