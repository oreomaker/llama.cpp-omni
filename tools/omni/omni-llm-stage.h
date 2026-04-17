#pragma once

#include "llama.h"

#include <string>
#include <vector>

struct common_params;
struct common_sampler;
struct omni_embeds;
struct omni_context;

struct OmniLlmStageDecodeRequest {
    std::string debug_dir;
    int         round_idx = -1;
};

enum class OmniLlmSchedulePolicy {
    DECODE_FIRST,
    PREFILL_PREEMPTIVE,
    MICRO_BATCH,
};

struct OmniLlmStageDecodeChunk {
    std::string              text;
    std::vector<llama_token> token_ids;
    std::vector<float>       hidden_states;
    bool                     llm_finish     = false;
    bool                     is_end_of_turn = false;
};

struct OmniLlmDecodeSliceResult {
    bool                     finished         = false;
    bool                     llm_finish       = false;
    bool                     interrupted      = false;
    bool                     preempted        = false;
    bool                     ended_with_listen = false;
    int                      generated_tokens = 0;
    std::string              text;
    std::vector<llama_token> token_ids;
    std::vector<float>       hidden_states;
    bool                     is_end_of_turn   = false;
};

struct OmniLlmStageDecodeResult {
    bool llm_finish              = false;
    bool interrupted             = false;
    bool preempted               = false;
    bool ended_with_listen       = false;
    int  generated_decode_tokens = 0;
};

enum class OmniLlmFrontierReplayKind {
    NONE,
    TOKENS,
    EMBEDS,
};

struct OmniLlmFrontierReplay {
    OmniLlmFrontierReplayKind kind      = OmniLlmFrontierReplayKind::NONE;
    int                       begin_pos = 0;
    int                       n_tokens  = 0;
    int                       n_embd    = 0;
    std::vector<llama_token>  tokens;
    std::vector<float>        embeds;
};

struct OmniLlmSchedulerState {
    llama_seq_id active_seq      = 0;
    llama_seq_id staging_seq     = 1;

    bool decode_active           = false;
    bool staged_ready            = false;

    int  active_chunk_idx        = -1;
    int  staged_chunk_idx        = -1;

    int  branch_n_past           = 0;
    int  staged_begin_pos        = 0;
    int  staged_n_past           = 0;
    int  spec_decode_tail        = 0;
    struct common_sampler * staged_sampler = nullptr;
    std::vector<float> pending_frontier_logits;
    std::vector<float> staged_frontier_logits;
    OmniLlmFrontierReplay staged_frontier_replay;
};

bool omni_llm_stage_eval_tokens(struct omni_context *    ctx_omni,
                                struct common_params *   params,
                                std::vector<llama_token> tokens,
                                int                      n_batch,
                                int *                    n_past,
                                bool                     get_emb = false);
bool omni_llm_stage_eval_tokens_seq(struct omni_context *    ctx_omni,
                                    struct common_params *   params,
                                    std::vector<llama_token> tokens,
                                    int                      n_batch,
                                    int *                    n_past,
                                    llama_seq_id             seq_id,
                                    bool                     get_emb = false,
                                    OmniLlmFrontierReplay *  replay = nullptr);
bool omni_llm_stage_eval_string(struct omni_context * ctx_omni,
                                struct common_params * params,
                                const char *           str,
                                int                    n_batch,
                                int *                  n_past,
                                bool                   add_bos,
                                bool                   get_emb = false);
bool omni_llm_stage_eval_string_seq(struct omni_context * ctx_omni,
                                    struct common_params * params,
                                    const char *           str,
                                    int                    n_batch,
                                    int *                  n_past,
                                    llama_seq_id           seq_id,
                                    bool                   add_bos,
                                    bool                   get_emb = false,
                                    OmniLlmFrontierReplay * replay = nullptr);
bool omni_llm_stage_eval_string_with_hidden(struct omni_context * ctx_omni,
                                            struct common_params * params,
                                            const char *           str,
                                            int                    n_batch,
                                            int *                  n_past,
                                            bool                   add_bos,
                                            float *&               hidden_states);

bool omni_llm_stage_prefill_apply(struct omni_context *      ctx_omni,
                                  struct common_params *     params,
                                  const struct omni_embeds & embeds);
bool omni_llm_stage_prefill_apply_seq(struct omni_context *      ctx_omni,
                                      struct common_params *     params,
                                      const struct omni_embeds & embeds,
                                      llama_seq_id               seq_id,
                                      int *                      n_past,
                                      OmniLlmFrontierReplay *    replay = nullptr);
void omni_llm_stage_finalize_prefill(struct omni_context * ctx_omni);
void omni_llm_stage_worker_loop(struct omni_context * ctx_omni, struct common_params * params);
void omni_llm_stage_finalize_decode_round(struct omni_context * ctx_omni);
bool omni_llm_stage_decode_slice(struct omni_context *             ctx_omni,
                                 const OmniLlmStageDecodeRequest & request,
                                 int                               max_tokens,
                                 OmniLlmDecodeSliceResult *        out_result,
                                 OmniLlmSchedulerState *           scheduler = nullptr);
bool omni_llm_stage_decode_run(struct omni_context *                ctx_omni,
                               const OmniLlmStageDecodeRequest &    request,
                               OmniLlmStageDecodeResult *           out_result = nullptr);
