#pragma once

#include "llama.h"

#include <string>
#include <vector>

struct common_params;
struct omni_embeds;
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

struct OmniLlmStageDecodeState {
    int  max_tgt_len             = 0;
    int  step_size               = 10;
    int  llm_n_embd              = 0;
    int  generated_decode_tokens = 0;
    int  current_chunk_tokens    = 0;
    bool llm_finish              = false;
    bool llm_first_token_logged  = false;
    bool interrupted             = false;
};

struct OmniLlmStageDecodeSlice {
    OmniLlmStageDecodeChunk chunk;
    int                     total_tokens_generated = 0;
    bool                    chunk_limit_reached    = false;
    bool                    interrupted            = false;
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

void omni_llm_stage_prefill_apply(struct omni_context *      ctx_omni,
                                  struct common_params *     params,
                                  const struct omni_embeds & embeds);
void omni_llm_stage_finalize_prefill(struct omni_context * ctx_omni);
void omni_llm_stage_worker_loop(struct omni_context * ctx_omni, struct common_params * params);
void omni_llm_stage_finalize_decode_round(struct omni_context * ctx_omni);
void omni_llm_stage_decode_begin(struct omni_context * ctx_omni, OmniLlmStageDecodeState * state);
bool omni_llm_stage_decode_slice(struct omni_context *      ctx_omni,
                                 OmniLlmStageDecodeState *  state,
                                 OmniLlmStageDecodeSlice *  out_slice);
void omni_llm_stage_complete_decode_slice(struct omni_context *     ctx_omni,
                                          OmniLlmStageDecodeState * state,
                                          OmniLlmStageDecodeSlice * slice);
void omni_llm_stage_publish_decode_slice(struct omni_context *                 ctx_omni,
                                         const OmniLlmStageDecodeRequest &     request,
                                         const OmniLlmStageDecodeState &       state,
                                         const OmniLlmStageDecodeSlice &       slice);
void omni_llm_stage_finish_decode(struct omni_context *           ctx_omni,
                                  const OmniLlmStageDecodeState & state,
                                  OmniLlmStageDecodeResult *      out_result);
bool omni_llm_stage_decode_run(struct omni_context *                ctx_omni,
                               const OmniLlmStageDecodeRequest &    request,
                               OmniLlmStageDecodeResult *           out_result = nullptr);
