#pragma once

// think_speak.h — think-speak decode path: config, result types, entry points,
// and prompt/marker constants shared between omni.cpp and think_speak.cpp.

#include "llama.h"

#include <string>
#include <vector>

struct omni_context;

struct ThinkSpeakConfig {
    int  think_budget                = 25;   // token budget per think block
    int  answer_budget               = 4;    // token budget per (non-final) answer block
    int  endofthink_lookahead        = 12;   // extra probe steps to confirm <endofthink> after budget
    int  final_answer_max_new_tokens = 3000; // free-run budget for the final answer
    int  think_hard_limit            = 100;
    int  answer_hard_limit           = 100;
    int  final_answer_hard_limit     = 500;
    int  max_auto_blocks             = 100;
    bool manual_think_close          = true; // true = controller force-injects </think>
    bool use_tts_template            = true; // true = assistant prefix carries <|spk_*|><|tts_bos|>
};

enum class ThinkSpeakStop { Budget, HardLimit, StopToken, EndOfThink, Pad, ThinkOpen };

struct ThinkSpeakBlock {
    int         block_index = 0;
    std::string think;
    std::string answer;
    bool        endofthink = false;
};

struct ThinkSpeakResult {
    std::vector<ThinkSpeakBlock> blocks;
    std::string                  final_answer;
    std::string                  stopped_reason;
};

// TTS hook: caller provides a flush callback to receive per-token hidden states
// from answer/final phases. Think phase passes nullptr (silent).
struct ThinkSpeakTTSHook {
    int chunk_size = 10;
    void (*flush)(struct omni_context * ctx, std::string & text,
                  std::vector<llama_token> & ids, std::vector<float> & hidden,
                  bool is_final) = nullptr;
};

bool omni_think_speak_generate(struct omni_context * ctx, const ThinkSpeakConfig & cfg,
                               const std::string & question, ThinkSpeakResult * out);
bool think_speak_decode(struct omni_context * ctx, const ThinkSpeakConfig & cfg, int round_idx,
                        ThinkSpeakTTSHook * tts_hook = nullptr);

// Prompt/marker constants shared between omni.cpp and think_speak.cpp.
namespace think_speak {

inline constexpr const char * TRAIN_SYSTEM_PROMPT =
    "You are an AI assistant. You can receive video, audio, and text inputs and output speech and text.";

inline constexpr const char * SYSTEM_VOICE_CLONE_PREFIX = "克隆音频提示中的音色以生成语音。";
inline constexpr const char * SYSTEM_VOICE_CLONE_SUFFIX = "You are a helpful assistant with the above voice style.";

inline constexpr const char * TRAIN_USER_TEXT =
    "Listen to the question in the audio and answer by alternating between thinking and responding. "
    "Put intermediate reasoning inside <think>...</think>. For the final reasoning block, end it with "
    "<endofthink>, then write all remaining spoken answer directly without adding any further think or pad blocks.";

// {question} replaced by make_text_question().
inline constexpr const char * TEXT_QUESTION_TEMPLATE =
    "Please solve the following math problem from text.\n"
    "Problem: {question}\n"
    "Answer by alternating between thinking and responding. Put intermediate reasoning inside <think>...</think>. "
    "For the final reasoning block, end it with <endofthink>, then write all remaining spoken answer directly "
    "without adding any further think or pad blocks.";

// Must not be omitted — aligns first-turn distribution.
inline constexpr const char * EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n";

// Block 0 assistant prefix: TTS variant vs pure-text variant.
inline constexpr const char * ASSISTANT_THINK_BOS =
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|><think>";
inline constexpr const char * ASSISTANT_THINK_BOS_NOTTS =
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<think>";

inline constexpr const char * THINK_OPEN         = "<think>";
inline constexpr const char * THINK_CLOSE        = "</think>";
inline constexpr const char * END_OF_THINK       = "<endofthink>";
inline constexpr const char * PAD                = "<pad>";
inline constexpr const char * ANSWER_CONTINUE_BOS = "";

inline std::string make_text_question(const std::string & question) {
    std::string s = TEXT_QUESTION_TEMPLATE;
    const std::string ph = "{question}";
    const auto pos = s.find(ph);
    if (pos != std::string::npos) {
        s.replace(pos, ph.size(), question);
    }
    return s;
}

} // namespace think_speak
