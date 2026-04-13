#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct OmniRoundMeta;
struct common_params;
struct omni_context;

struct OmniTtsPreparedChunk {
    int                n_tokens_orig     = 0;
    int                n_tokens_filtered = 0;
    int                tts_n_embd        = 0;
    bool               merged_success    = false;
    std::vector<float> merged_embeddings;
};

bool omni_tts_is_valid_token(llama_token tid);

bool omni_tts_prepare_duplex_chunk(struct omni_context *            ctx_omni,
                                   const std::string &              llm_debug_output_dir,
                                   int                              current_chunk_idx,
                                   const std::string &              llm_text,
                                   const std::vector<llama_token> & current_chunk_token_ids,
                                   const std::vector<float> &       current_chunk_hidden_states,
                                   int                              current_chunk_n_embd,
                                   OmniTtsPreparedChunk &           prepared_chunk);

bool omni_tts_prepare_simplex_chunk(struct omni_context *            ctx_omni,
                                    const std::string &              llm_debug_output_dir,
                                    int                              current_chunk_idx,
                                    const std::string &              response,
                                    const std::vector<llama_token> & current_chunk_token_ids,
                                    const std::vector<float> &       current_chunk_hidden_states,
                                    int                              current_chunk_n_embd,
                                    OmniTtsPreparedChunk &           prepared_chunk);

bool omni_tts_generate_audio_tokens_local(struct omni_context *      ctx_omni,
                                          struct common_params *     params,
                                          const std::vector<float> & merged_embeddings,
                                          int                        n_tokens,
                                          int                        tts_n_embd,
                                          int                        chunk_idx,
                                          int                        duplex_chunk_idx,
                                          std::vector<int32_t> &     output_audio_tokens,
                                          bool                       is_end_of_turn,
                                          const OmniRoundMeta &      round_meta,
                                          const std::string &        output_dir = "");

bool omni_tts_generate_audio_tokens_local_simplex(struct omni_context *      ctx_omni,
                                                  struct common_params *     params,
                                                  const std::vector<float> & merged_embeddings,
                                                  int                        n_tokens,
                                                  int                        tts_n_embd,
                                                  int                        chunk_idx,
                                                  std::vector<int32_t> &     output_audio_tokens,
                                                  const OmniRoundMeta &      round_meta,
                                                  const std::string &        output_dir          = "",
                                                  bool                       is_final_text_chunk = false);
