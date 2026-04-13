#include "omni-sliding-window.h"

#include "common/common.h"
#include "omni-log.h"
#include "omni.h"

#include <algorithm>

static bool omni_sliding_eval_tokens(struct omni_context *    ctx_omni,
                                     struct common_params *   params,
                                     std::vector<llama_token> tokens,
                                     int                      n_batch,
                                     int *                    n_past) {
    const int n_tokens = (int) tokens.size();
    kv_cache_slide_window(ctx_omni, params, n_tokens);

    for (int i = 0; i < n_tokens; i += n_batch) {
        int n_eval = n_tokens - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (n_eval == 0) {
            break;
        }

        llama_batch            batch = llama_batch_get_one(tokens.data() + i, n_eval);
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        for (int j = 0; j < n_eval; ++j) {
            batch.pos[j] = *n_past + j;
        }

        if (llama_decode(ctx_omni->ctx_llama, batch)) {
            return false;
        }
        *n_past += n_eval;
    }

    return true;
}

static bool omni_sliding_eval_string(struct omni_context *  ctx_omni,
                                     struct common_params * params,
                                     const char *           str,
                                     int                    n_batch,
                                     int *                  n_past,
                                     bool                   add_bos) {
    std::string              str_buf = str;
    std::vector<llama_token> tokens  = common_tokenize(ctx_omni->ctx_llama, str_buf, add_bos, true);
    return omni_sliding_eval_tokens(ctx_omni, params, std::move(tokens), n_batch, n_past);
}

static int get_cache_length(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr) {
        return 0;
    }
    return ctx_omni->session.n_past;
}

void kv_cache_slide_window(struct omni_context * ctx_omni, struct common_params * params, int chunk_size) {
    const int n_ctx = params->n_ctx;

    if (ctx_omni->session.n_past + chunk_size < n_ctx) {
        return;
    }

    print_with_timestamp("⚠️ KV Cache 滑动窗口触发: n_past=%d, chunk_size=%d, n_ctx=%d, n_keep=%d, 轮次数=%zu\n",
                         ctx_omni->session.n_past, chunk_size, n_ctx, ctx_omni->session.prompt.n_keep, ctx_omni->session.round_start_positions.size());

    int n_discard      = 0;
    int delete_end_pos = ctx_omni->session.prompt.n_keep;

    if (ctx_omni->session.max_preserved_context > 0 && ctx_omni->session.round_start_positions.size() >= 1) {
        const auto & rounds            = ctx_omni->session.round_start_positions;
        int          cumulative_length = 0;
        int          keep_from_round   = (int) rounds.size();
        const int    total_rounds      = (int) rounds.size();

        for (int i = total_rounds - 1; i >= 0; --i) {
            const int round_start  = (i == 0) ? ctx_omni->session.prompt.n_keep : rounds[i - 1];
            const int round_end    = rounds[i];
            const int round_length = round_end - round_start;

            if (cumulative_length + round_length > ctx_omni->session.max_preserved_context) {
                break;
            }

            cumulative_length += round_length;
            keep_from_round = i;
        }

        if (keep_from_round >= total_rounds) {
            keep_from_round = total_rounds - 1;
        }

        const int delete_start = ctx_omni->session.prompt.n_keep;
        delete_end_pos         = (keep_from_round == 0) ? ctx_omni->session.prompt.n_keep : rounds[keep_from_round - 1];

        if (delete_end_pos > delete_start) {
            n_discard = delete_end_pos - delete_start;

            print_with_timestamp("⚠️ 按轮次删除: 删除轮次 0-%d，保留轮次 %d-%d，保留长度=%d\n", keep_from_round - 1,
                                 keep_from_round, total_rounds - 1, cumulative_length);

            std::vector<int> new_rounds;
            for (int i = keep_from_round; i < total_rounds; ++i) {
                new_rounds.push_back(rounds[i] - n_discard);
            }
            ctx_omni->session.round_start_positions = new_rounds;

            print_with_timestamp("⚠️ 更新轮次边界: 新边界数=%zu，首轮结束位置=%d\n", new_rounds.size(),
                                 new_rounds.empty() ? -1 : new_rounds[0]);
        } else {
            print_with_timestamp("⚠️ 没有可删除的完整轮次（keep_from_round=%d），回退到按比例删除\n", keep_from_round);
            n_discard = 0;
        }
    }

    if (n_discard == 0) {
        const int n_left = ctx_omni->session.n_past - ctx_omni->session.prompt.n_keep;
        n_discard        = n_left / 2;
        delete_end_pos   = ctx_omni->session.prompt.n_keep + n_discard;

        if (n_left <= 0 || n_discard <= 0) {
            print_with_timestamp("⚠️ KV Cache 滑动窗口: 边界检查失败 n_left=%d, n_discard=%d，跳过滑动\n", n_left,
                                 n_discard);
            return;
        }

        std::vector<int> new_rounds;
        for (int pos : ctx_omni->session.round_start_positions) {
            if (pos > delete_end_pos) {
                new_rounds.push_back(pos - n_discard);
            }
        }
        ctx_omni->session.round_start_positions = new_rounds;

        print_with_timestamp("⚠️ 按比例删除后轮次边界: 剩余 %zu 个轮次\n", new_rounds.size());
        print_with_timestamp("⚠️ 按比例删除: n_left=%d, n_discard=%d, 删除范围=[%d, %d)\n", n_left, n_discard,
                             ctx_omni->session.prompt.n_keep, delete_end_pos);
    }

    print_with_timestamp("⚠️ KV Cache 滑动窗口执行: 删除范围=[%d, %d), n_discard=%d\n", ctx_omni->session.prompt.n_keep, delete_end_pos,
                         n_discard);

    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
    if (mem) {
        const bool rm_ok = llama_memory_seq_rm(mem, 0, ctx_omni->session.prompt.n_keep, delete_end_pos);
        (void) rm_ok;
        llama_memory_seq_add(mem, 0, delete_end_pos, ctx_omni->session.n_past, -n_discard);
    }

    const int old_n_past = ctx_omni->session.n_past;
    ctx_omni->session.n_past -= n_discard;
    print_with_timestamp("⚠️ KV Cache 滑动窗口完成: n_past 从 %d 减少到 %d\n", old_n_past, ctx_omni->session.n_past);
}

void sliding_window_reset(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return;
    }

    const int old_unit_count = (int) ctx_omni->session.unit_history.size();

    ctx_omni->session.unit_history.clear();
    ctx_omni->session.next_unit_id                 = 0;
    ctx_omni->session.pending_unit_id              = -1;
    ctx_omni->session.pending_unit_start_cache_len = 0;
    ctx_omni->session.prompt.system_preserve_length       = 0;
    ctx_omni->session.position_offset              = 0;
    ctx_omni->session.sliding_event_count          = 0;
    ctx_omni->session.total_dropped_tokens         = 0;
    ctx_omni->session.total_dropped_units          = 0;

    if (old_unit_count > 0) {
        print_with_timestamp("[SW] reset: cleared %d units, all sliding window state reset\n", old_unit_count);
    }
}

void sliding_window_reset_after_kvcache_clean(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return;
    }

    sliding_window_reset(ctx_omni);
    ctx_omni->session.prompt.system_preserve_length = ctx_omni->session.prompt.n_keep;
}

int sliding_window_register_unit_start(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return -1;
    }

    ctx_omni->session.pending_unit_id              = ctx_omni->session.next_unit_id;
    ctx_omni->session.pending_unit_start_cache_len = get_cache_length(ctx_omni);

    print_with_timestamp("[SW] unit_start: pending_unit_id=%d, cache_len=%d, preserve=%d, units=%zu\n",
                         ctx_omni->session.pending_unit_id, ctx_omni->session.pending_unit_start_cache_len,
                         ctx_omni->session.prompt.system_preserve_length, ctx_omni->session.unit_history.size());

    return ctx_omni->session.pending_unit_id;
}

void sliding_window_register_unit_end(struct omni_context *            ctx_omni,
                                      const std::string &              input_type,
                                      const std::vector<llama_token> & generated_tokens,
                                      bool                             is_listen) {
    if (ctx_omni == nullptr) {
        return;
    }

    if (ctx_omni->session.pending_unit_id < 0) {
        print_with_timestamp("[SW] WARNING: register_unit_end called without register_unit_start\n");
        return;
    }

    const int current_cache_len = get_cache_length(ctx_omni);
    const int unit_len          = current_cache_len - ctx_omni->session.pending_unit_start_cache_len;

    if (unit_len > 0) {
        UnitEntry entry;
        entry.unit_id          = ctx_omni->session.pending_unit_id;
        entry.length           = unit_len;
        entry.type             = input_type;
        entry.generated_tokens = generated_tokens;
        entry.is_listen        = is_listen;

        ctx_omni->session.unit_history.push_back(entry);

        print_with_timestamp(
            "[SW] unit_end: unit_id=%d type=%s len=%d gen_tokens=%zu is_listen=%d | cache=%d preserve=%d "
            "total_units=%zu\n",
            entry.unit_id, entry.type.c_str(), entry.length, entry.generated_tokens.size(), entry.is_listen,
            current_cache_len, ctx_omni->session.prompt.system_preserve_length, ctx_omni->session.unit_history.size());
    } else {
        print_with_timestamp(
            "[SW] WARNING: unit_end: unit_id=%d has zero length (start=%d, current=%d), not recorded\n",
            ctx_omni->session.pending_unit_id, ctx_omni->session.pending_unit_start_cache_len, current_cache_len);
    }

    ctx_omni->session.pending_unit_id              = -1;
    ctx_omni->session.pending_unit_start_cache_len = 0;
    ctx_omni->session.next_unit_id++;
}

void sliding_window_register_system_prompt(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return;
    }

    ctx_omni->session.prompt.system_preserve_length = get_cache_length(ctx_omni);
    print_with_timestamp("[SW] system_prompt registered: preserve_length=%d (will be protected from sliding)\n",
                         ctx_omni->session.prompt.system_preserve_length);
}

bool sliding_window_drop_tokens_from_cache(struct omni_context * ctx_omni, int length) {
    if (ctx_omni == nullptr || ctx_omni->ctx_llama == nullptr || length <= 0) {
        print_with_timestamp("[SW] drop_tokens: invalid params (length=%d)\n", length);
        return false;
    }

    const int cache_len_before = get_cache_length(ctx_omni);
    const int preserve         = ctx_omni->session.prompt.system_preserve_length;

    if (cache_len_before <= preserve) {
        print_with_timestamp("[SW] drop_tokens: cache_len=%d <= preserve=%d, nothing to drop\n", cache_len_before,
                             preserve);
        return false;
    }

    const int available = cache_len_before - preserve;
    if (available < length) {
        print_with_timestamp("[SW] drop_tokens: cannot drop %d tokens, only %d available (cache=%d, preserve=%d)\n",
                             length, available, cache_len_before, preserve);
        return false;
    }

    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_llama);
    if (!mem) {
        print_with_timestamp("[SW] drop_tokens: failed to get llama memory\n");
        return false;
    }

    const bool success = llama_memory_seq_rm(mem, 0, preserve, preserve + length);
    if (success) {
        ctx_omni->session.n_past = cache_len_before - length;
        ctx_omni->session.position_offset += length;

        print_with_timestamp("[SW] drop_tokens: SUCCESS, dropped %d tokens from [%d, %d), cache %d -> %d, offset=%d\n",
                             length, preserve, preserve + length, cache_len_before, ctx_omni->session.n_past,
                             ctx_omni->session.position_offset);
    } else {
        print_with_timestamp("[SW] drop_tokens: FAILED to drop %d tokens\n", length);
    }

    return success;
}

static bool sliding_window_drop_unit(struct omni_context * ctx_omni, int unit_id) {
    if (ctx_omni == nullptr) {
        return false;
    }

    auto it = std::find_if(ctx_omni->session.unit_history.begin(), ctx_omni->session.unit_history.end(),
                           [unit_id](const UnitEntry & entry) { return entry.unit_id == unit_id; });
    if (it == ctx_omni->session.unit_history.end()) {
        return false;
    }

    const int total_len = it->length;
    if (total_len <= 0) {
        print_with_timestamp("[SW] drop_unit: unit_id=%d has zero length, removing from history\n", unit_id);
        ctx_omni->session.unit_history.erase(it);
        return false;
    }

    const int cache_before = get_cache_length(ctx_omni);
    if (!sliding_window_drop_tokens_from_cache(ctx_omni, total_len)) {
        print_with_timestamp("[SW] drop_unit: failed to drop %d tokens for unit_id=%d\n", total_len, unit_id);
        return false;
    }

    const int cache_after = get_cache_length(ctx_omni);
    print_with_timestamp("[SW] 🗑️ DROPPED unit_id=%d type=%s len=%d gen_tokens=%zu | cache %d -> %d, offset=%d\n",
                         it->unit_id, it->type.c_str(), it->length, it->generated_tokens.size(), cache_before,
                         cache_after, ctx_omni->session.position_offset);

    ctx_omni->session.unit_history.erase(it);
    return true;
}

static bool sliding_window_drop_next_unit(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return false;
    }

    for (const auto & entry : ctx_omni->session.unit_history) {
        if (entry.type == "system") {
            print_with_timestamp("[SW] drop_next_unit: skipping system unit_id=%d\n", entry.unit_id);
            continue;
        }

        print_with_timestamp("[SW] drop_next_unit: attempting to drop unit_id=%d\n", entry.unit_id);
        if (sliding_window_drop_unit(ctx_omni, entry.unit_id)) {
            return true;
        }
    }

    print_with_timestamp("[SW] drop_next_unit: no droppable unit found in %zu units\n", ctx_omni->session.unit_history.size());
    return false;
}

bool sliding_window_enforce(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr) {
        return false;
    }

    const auto & cfg = ctx_omni->session.sliding_window_config;
    if (cfg.mode == "off") {
        return false;
    }

    const int cache_len_before = get_cache_length(ctx_omni);
    if (cache_len_before <= cfg.high_water_tokens) {
        return false;
    }

    print_with_timestamp("[SW] ⚡ SLIDING TRIGGERED: cache=%d > high_water=%d, target=low_water=%d\n", cache_len_before,
                         cfg.high_water_tokens, cfg.low_water_tokens);

    int dropped_count = 0;
    int cache_len     = cache_len_before;

    while (cache_len > cfg.low_water_tokens) {
        if (!sliding_window_drop_next_unit(ctx_omni)) {
            print_with_timestamp("[SW] enforce_window: no more units to drop, stopping\n");
            break;
        }
        dropped_count++;
        cache_len = get_cache_length(ctx_omni);
    }

    if (dropped_count > 0) {
        ctx_omni->session.sliding_event_count++;
        ctx_omni->session.total_dropped_tokens += cache_len_before - cache_len;
        ctx_omni->session.total_dropped_units += dropped_count;

        int expected = ctx_omni->session.prompt.system_preserve_length;
        for (const auto & unit : ctx_omni->session.unit_history) {
            expected += unit.length;
        }
        const bool is_consistent = expected == cache_len;

        print_with_timestamp(
            "[SW] ✅ SLIDING DONE: cache %d -> %d, dropped %d units, remaining %zu units | consistency: expected=%d "
            "actual=%d %s\n",
            cache_len_before, cache_len, dropped_count, ctx_omni->session.unit_history.size(), expected, cache_len,
            is_consistent ? "✓" : "✗ MISMATCH!");

        if (!is_consistent) {
            print_with_timestamp("[SW] ❌ CONSISTENCY ERROR! preserve=%d + sum(units)=%d != cache=%d, offset=%d\n",
                                 ctx_omni->session.prompt.system_preserve_length, expected - ctx_omni->session.prompt.system_preserve_length,
                                 cache_len, ctx_omni->session.position_offset);
        }
    }

    return dropped_count > 0;
}

void omni_finalize_decode_round(struct omni_context * ctx_omni) {
    if (ctx_omni->duplex_mode) {
        return;
    }

    const int reserved_space = 1024;
    const int n_ctx          = ctx_omni->params->n_ctx;

    if (ctx_omni->session.n_past > n_ctx - reserved_space) {
        print_with_timestamp("⚠️ Decode 结束滑窗检查: n_past=%d > n_ctx-reserved=%d，需要滑窗\n", ctx_omni->session.n_past,
                             n_ctx - reserved_space);
        kv_cache_slide_window(ctx_omni, ctx_omni->params, reserved_space);
    } else {
        print_with_timestamp("📍 Decode 结束: n_past=%d, 剩余空间=%d, 无需滑窗\n", ctx_omni->session.n_past,
                             n_ctx - ctx_omni->session.n_past);
    }

    ctx_omni->session.round_start_positions.push_back(ctx_omni->session.n_past);
    print_with_timestamp("📍 轮次 %zu 结束，记录边界于 n_past=%d\n", ctx_omni->session.round_start_positions.size(),
                         ctx_omni->session.n_past);

    const bool prefix_ok = omni_sliding_eval_string(ctx_omni, ctx_omni->params, "<|im_end|>\n<|im_start|>user\n",
                                                    ctx_omni->params->n_batch, &ctx_omni->session.n_past, false);
    if (!prefix_ok) {
        print_with_timestamp("⚠️ 为下一轮准备 user 前缀失败，n_past=%d\n", ctx_omni->session.n_past);
        return;
    }

    print_with_timestamp("📍 为下一轮准备: eval <|im_end|>\\n<|im_start|>user\\n, n_past=%d\n", ctx_omni->session.n_past);
}
