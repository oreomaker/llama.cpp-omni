// think_speak.cpp — LLM state machine for think-speak decode.
// Alternates silent <think> blocks with spoken answer blocks using token budgets
// and <endofthink>/<pad> substring matching. TTS routing is done via the caller-
// provided ThinkSpeakTTSHook (answer phases only); think phases are always silent.

#include "omni.h"
#include "think_speak.h"

#include "common/common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

// True if text ends with a partial (incomplete) prefix of "<endofthink>".
// Grants lookahead past budget so a forming marker is not truncated.
bool has_pending_endofthink_prefix(const std::string & text) {
    const std::string eot = think_speak::END_OF_THINK;
    if (text.find(eot) != std::string::npos) {
        return false;  // already complete
    }
    const size_t start = text.rfind('<');
    if (start == std::string::npos) {
        return false;
    }
    const std::string suffix = text.substr(start);
    // eot.startswith(suffix), suffix non-empty (and necessarily shorter than eot)
    return !suffix.empty() && suffix.size() <= eot.size() &&
           eot.compare(0, suffix.size(), suffix) == 0;
}

// Split text at "<endofthink>": before = reasoning, after = inline answer prefix (already in KV).
void split_endofthink_tail(const std::string & text, std::string & before, std::string & after) {
    const std::string eot = think_speak::END_OF_THINK;
    const size_t pos = text.find(eot);
    if (pos == std::string::npos) {
        before = text;
        after.clear();
        return;
    }
    before = text.substr(0, pos);
    after  = text.substr(pos + eot.size());
}

// Strip trailing "</think>" if the model emitted it (controller injects it separately).
std::string strip_think_close(const std::string & text) {
    const std::string tc = think_speak::THINK_CLOSE;
    if (text.size() >= tc.size() && text.compare(text.size() - tc.size(), tc.size(), tc) == 0) {
        return text.substr(0, text.size() - tc.size());
    }
    return text;
}


struct CollectOpts {
    int  budget;
    int  hard_limit;        // 0 = disabled
    int  lookahead;
    bool match_endofthink;
};

// Sample tokens until a stop condition. When tts_hook != nullptr, accumulates
// valid token ids + hidden states and flushes every chunk_size tokens.
ThinkSpeakStop think_speak_collect(omni_context * ctx, const CollectOpts & o, std::string & out_text,
                                   ThinkSpeakTTSHook * tts_hook,
                                   std::vector<llama_token> & tts_ids,
                                   std::vector<float> & tts_hidden,
                                   std::string & tts_text) {
    int         toks = 0;
    int         look = o.lookahead;
    std::string joined;
    const int   n_embd = llama_n_embd(llama_get_model(ctx->ctx_llama));

    while (true) {
        if (o.hard_limit > 0 && toks >= o.hard_limit) {
            return ThinkSpeakStop::HardLimit;
        }
        if (toks >= o.budget &&
            !(o.match_endofthink && has_pending_endofthink_prefix(joined) && look-- > 0)) {
            return ThinkSpeakStop::Budget;
        }

        llama_token tok = -1;
        float *     hid = nullptr;
        const char * piece = llama_loop_with_hidden_and_token(
            ctx, ctx->params, ctx->ctx_sampler, ctx->n_past, hid, tok);
        if (is_end_token(ctx, tok)) {
            if (hid != nullptr) free(hid);
            return ThinkSpeakStop::StopToken;
        }

        const std::string s = (piece != nullptr) ? piece : "";
        joined   += s;
        out_text += s;
        toks++;

        if (tts_hook && tts_hook->flush && hid != nullptr && is_valid_tts_token(tok)) {
            tts_ids.push_back(tok);
            tts_hidden.insert(tts_hidden.end(), hid, hid + n_embd);
            tts_text += s;
            if ((int) tts_ids.size() >= tts_hook->chunk_size) {
                tts_hook->flush(ctx, tts_text, tts_ids, tts_hidden, /*is_final=*/false);
            }
        }

        if (hid != nullptr) {
            free(hid);
            hid = nullptr;
        }

        if (o.match_endofthink) {
            if (joined.find(think_speak::END_OF_THINK) != std::string::npos) {
                return ThinkSpeakStop::EndOfThink;
            }
            if (joined.find(think_speak::PAD) != std::string::npos) {
                return ThinkSpeakStop::Pad;
            }
        }
    }
}

}  // namespace

bool think_speak_decode(omni_context * ctx, const ThinkSpeakConfig & cfg, int /*round_idx*/,
                        ThinkSpeakTTSHook * tts_hook) {
    if (ctx == nullptr || ctx->ctx_llama == nullptr || ctx->params == nullptr) {
        return false;
    }
    const int n_batch = ctx->params->n_batch;

    ThinkSpeakResult * out = ctx->think_speak_out;
    if (out != nullptr) {
        out->blocks.clear();
        out->final_answer.clear();
        out->stopped_reason.clear();
    }

    const bool tts_tpl = cfg.use_tts_template && ctx->use_tts;

    // TTS buffers persist across blocks (think blocks don't feed, answer blocks do).
    std::vector<llama_token> tts_ids;
    std::vector<float>       tts_hidden;
    std::string              tts_text;

    std::string stopped_reason;
    std::string all_answers;

    for (int b = 0; b < cfg.max_auto_blocks; ++b) {
        // ---- THINK (silent) --------------------------------------------------
        push_think_start(ctx);

        const char * think_bos = (b == 0)
            ? (tts_tpl ? think_speak::ASSISTANT_THINK_BOS : think_speak::ASSISTANT_THINK_BOS_NOTTS)
            : think_speak::THINK_OPEN;
        eval_string(ctx, ctx->params, think_bos, n_batch, &ctx->n_past, /*add_bos=*/false);

        std::string think_raw;
        const CollectOpts topt{cfg.think_budget, cfg.think_hard_limit, cfg.endofthink_lookahead,
                               /*match_endofthink=*/true};
        const ThinkSpeakStop think_stop = think_speak_collect(ctx, topt, think_raw,
                                                        /*tts_hook=*/nullptr, tts_ids, tts_hidden, tts_text);

        std::string think_text = strip_think_close(think_raw);
        const bool  endofthink = (think_stop == ThinkSpeakStop::EndOfThink);

        ThinkSpeakBlock blk;
        blk.block_index = b;
        blk.endofthink  = endofthink;

        std::string inline_prefix;
        if (endofthink) {
            std::string before, after;
            split_endofthink_tail(think_text, before, after);
            think_text    = before;
            inline_prefix = after;
        } else {
            eval_string(ctx, ctx->params, think_speak::THINK_CLOSE, n_batch, &ctx->n_past, /*add_bos=*/false);
        }
        blk.think = think_text;

        push_think_text_fragment(ctx, think_text);
        push_think_end(ctx);

        fprintf(stderr, "[think-speak] block %d think(%zu chars, stop=%d)\n", b, think_text.size(), (int) think_stop);

        if (think_stop == ThinkSpeakStop::HardLimit) {
            if (out) out->blocks.push_back(blk);
            stopped_reason = "think_hard_limit";
            break;
        }
        if (think_stop == ThinkSpeakStop::StopToken) {
            if (out) out->blocks.push_back(blk);
            stopped_reason = "think_finished";
            break;
        }

        // ---- ANSWER (spoken) -------------------------------------------------
        if (endofthink) {
            int requested = cfg.final_answer_max_new_tokens;
            if (cfg.final_answer_hard_limit > 0) {
                requested = std::min(requested, cfg.final_answer_hard_limit);
            }
            std::string ans;
            const CollectOpts fopt{requested, /*hard_limit=*/0, /*lookahead=*/0,
                                   /*match_endofthink=*/false};
            const ThinkSpeakStop fs = think_speak_collect(ctx, fopt, ans,
                                                          tts_hook, tts_ids, tts_hidden, tts_text);

            blk.answer = inline_prefix + ans;
            if (!all_answers.empty() && !blk.answer.empty()) all_answers += ' ';
            all_answers += blk.answer;
            if (out) {
                out->blocks.push_back(blk);
            }
            push_think_speak_user_text(ctx, blk.answer);

            // Flush any remaining TTS tokens from the final answer block.
            if (tts_hook && tts_hook->flush && !tts_ids.empty()) {
                tts_hook->flush(ctx, tts_text, tts_ids, tts_hidden, /*is_final=*/false);
            }

            if (fs == ThinkSpeakStop::StopToken) {
                stopped_reason = "endofthink_answer_finished";
            } else if (cfg.final_answer_hard_limit > 0 && requested >= cfg.final_answer_hard_limit) {
                stopped_reason = "final_answer_hard_limit";
            } else {
                stopped_reason = "endofthink_answer_max_tokens";
            }
            fprintf(stderr, "[think-speak] block %d final(%zu chars) -> %s\n",
                    b, blk.answer.size(), stopped_reason.c_str());
            break;
        }

        // Non-final answer.
        int requested = cfg.answer_budget;
        if (cfg.answer_hard_limit > 0) {
            requested = std::min(requested, cfg.answer_hard_limit);
        }
        std::string ans;
        const CollectOpts aopt{requested, /*hard_limit=*/0, /*lookahead=*/0,
                               /*match_endofthink=*/false};
        const ThinkSpeakStop answer_stop = think_speak_collect(ctx, aopt, ans,
                                                      tts_hook, tts_ids, tts_hidden, tts_text);

        blk.answer = ans;
        if (!all_answers.empty() && !ans.empty()) all_answers += ' ';
        all_answers += ans;
        if (out) out->blocks.push_back(blk);
        push_think_speak_user_text(ctx, ans);

        // Flush accumulated TTS tokens at answer block boundary (aligned with
        // Python: each answer block independently streams its audio).
        if (tts_hook && tts_hook->flush && !tts_ids.empty()) {
            tts_hook->flush(ctx, tts_text, tts_ids, tts_hidden, /*is_final=*/false);
        }

        fprintf(stderr, "[think-speak] block %d answer(%zu chars, stop=%d)\n", b, ans.size(), (int) answer_stop);

        if (answer_stop == ThinkSpeakStop::StopToken) {
            stopped_reason = "answer_finished";
            break;
        }
        if (cfg.answer_hard_limit > 0 && requested >= cfg.answer_hard_limit) {
            stopped_reason = "answer_hard_limit";
            break;
        }
    }

    if (stopped_reason.empty()) {
        stopped_reason = "max_auto_blocks";
    }
    if (out != nullptr) {
        out->final_answer   = all_answers;
        out->stopped_reason = stopped_reason;
    }

    if (tts_hook && tts_hook->flush) {
        tts_hook->flush(ctx, tts_text, tts_ids, tts_hidden, /*is_final=*/true);
    }

    fprintf(stderr, "[think-speak] stopped_reason=%s, blocks=%zu\n",
            stopped_reason.c_str(), out ? out->blocks.size() : (size_t) 0);
    return true;
}

bool omni_think_speak_generate(omni_context * ctx, const ThinkSpeakConfig & cfg,
                               const std::string & question, ThinkSpeakResult * out,
                               const std::string & audio_path) {
    if (ctx == nullptr || ctx->ctx_llama == nullptr || ctx->params == nullptr) {
        return false;
    }

    if (!ctx->system_prompt_initialized) {
        if (!stream_prefill(ctx, /*aud_fname=*/"", /*img_fname=*/"", /*index=*/0)) {
            return false;
        }
    }

    if (!audio_path.empty()) {
        auto * embeds = omni_audio_embed_make_with_filename(
            ctx->ctx_audio, ctx->params->cpuparams.n_threads, audio_path);
        if (embeds == nullptr || embeds->n_pos <= 0) {
            fprintf(stderr, "[think-speak] failed to encode audio: %s\n", audio_path.c_str());
            return false;
        }
        eval_string(ctx, ctx->params, "<|audio_start|>",
                    ctx->params->n_batch, &ctx->n_past, /*add_bos=*/false);
        prefill_with_emb(ctx, ctx->params, embeds->embed, embeds->n_pos,
                         ctx->params->n_batch, &ctx->n_past);
        eval_string(ctx, ctx->params, "<|audio_end|>",
                    ctx->params->n_batch, &ctx->n_past, /*add_bos=*/false);
        omni_embed_free(embeds);
        eval_string(ctx, ctx->params, think_speak::TRAIN_USER_TEXT,
                    ctx->params->n_batch, &ctx->n_past, /*add_bos=*/false);
        fprintf(stderr, "[think-speak] audio user turn injected: %s\n", audio_path.c_str());
    } else {
        const std::string user_text = think_speak::make_text_question(question);
        eval_string(ctx, ctx->params, user_text.c_str(), ctx->params->n_batch, &ctx->n_past, /*add_bos=*/false);
    }

    think_speak_decode_begin(ctx, /*round_idx=*/0);

    ThinkSpeakTTSHook hook;
    ThinkSpeakTTSHook * hook_ptr = nullptr;
    if (ctx->async && ctx->use_tts) {
        hook.chunk_size = 10;
        hook.flush      = think_speak_tts_chunk_flush;
        hook_ptr        = &hook;
    }

    ctx->think_speak_out = out;
    const bool ok = think_speak_decode(ctx, cfg, /*round_idx=*/0, hook_ptr);
    ctx->think_speak_out = nullptr;

    think_speak_decode_finalize(ctx);
    return ok;
}
