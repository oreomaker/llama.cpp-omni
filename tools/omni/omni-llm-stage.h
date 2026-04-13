#pragma once

#include "llama.h"

#include <string>
#include <vector>

struct common_params;
struct omni_context;

struct OmniLlmStageDecodeRequest {
    std::string debug_dir;
    int         round_idx = -1;
};

struct OmniLlmStageDecodeChunk {
    std::string              text;
    std::vector<llama_token> token_ids;
    std::vector<float>       hidden_states;
    bool                     llm_finish     = false;
    bool                     is_end_of_turn = false;
};

struct OmniLlmStageDecodeResult {
    bool llm_finish              = false;
    bool interrupted             = false;
    bool ended_with_listen       = false;
    int  generated_decode_tokens = 0;
};

bool omni_llm_stage_eval_tokens(struct omni_context *    ctx_omni,
                                struct common_params *   params,
                                std::vector<llama_token> tokens,
                                int                      n_batch,
                                int *                    n_past,
                                bool                     get_emb = false);
bool omni_llm_stage_eval_string(struct omni_context * ctx_omni,
                                struct common_params * params,
                                const char *           str,
                                int                    n_batch,
                                int *                  n_past,
                                bool                   add_bos,
                                bool                   get_emb = false);
bool omni_llm_stage_eval_string_with_hidden(struct omni_context * ctx_omni,
                                            struct common_params * params,
                                            const char *           str,
                                            int                    n_batch,
                                            int *                  n_past,
                                            bool                   add_bos,
                                            float *&               hidden_states);

void omni_llm_stage_finalize_decode_round(struct omni_context * ctx_omni);
bool omni_llm_stage_decode_run(struct omni_context *                ctx_omni,
                               const OmniLlmStageDecodeRequest &    request,
                               OmniLlmStageDecodeResult *           out_result = nullptr);
