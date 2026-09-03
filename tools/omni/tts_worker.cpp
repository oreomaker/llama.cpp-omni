// TTS worker layer (converted from the former omni_tts.inc include).
// Owns the TTS worker threads; the token-generation graph ops are in the
// compute layer (omni.cpp) via the omni_internal.h interface.
#include "omni-impl.h"
#include "omni.h"
#include "omni_internal.h"
#include "token2wav/token2wav-impl.h"
#include "common/common.h"
#include "common/sampling.h"

#include <cstdlib>

#ifndef _WIN32
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

// ==========================================================================
// TTS generation and worker implementation
// ==========================================================================

static void sm_tts_segment_cb(int segment_id, int segment_count, void * user_data) {
    if (segment_id + 1 >= segment_count || user_data == nullptr) return;
    sm_segment_boundary(*static_cast<SmSched *>(user_data), SM_BIT_TTS);
}


// Helper function to filter special tokens (matching Python file_loader.py logic)
static void filter_special_tokens(
    std::vector<llama_token>& token_ids,
    std::vector<float>& hidden_states,
    int n_embd
) {
    // Validate input: hidden_states should have size = token_ids.size() * n_embd
    if (hidden_states.size() != token_ids.size() * n_embd) {
        LOG_ERR("filter_special_tokens: hidden_states size (%zu) != token_ids.size() * n_embd (%zu * %d)\n",
                hidden_states.size(), token_ids.size(), n_embd);
        return;  // Don't filter if sizes don't match
    }
    
    size_t original_size = token_ids.size();
    
    // Filter out special tokens and tokens >= 150000
    // IMPORTANT: Keep token_ids and hidden_states aligned by filtering them together
    std::vector<llama_token> filtered_token_ids;
    std::vector<float> filtered_hidden_states;
    
    for (size_t i = 0; i < token_ids.size(); ++i) {
        llama_token tid = token_ids[i];
        
        // Use the shared is_valid_tts_token function
        if (!is_valid_tts_token(tid)) continue;
        
        // Keep this token and its corresponding hidden state
        filtered_token_ids.push_back(tid);
        // Copy corresponding hidden state (n_embd floats per token)
        for (int j = 0; j < n_embd; ++j) {
            filtered_hidden_states.push_back(hidden_states[i * n_embd + j]);
        }
    }
    
    // Filter out leading empty tokens (already handled by is_valid_tts_token for known_empty_token_ids)
    size_t start_idx = 0;
    for (size_t i = 0; i < filtered_token_ids.size(); ++i) {
        if (g_known_empty_token_ids.find(filtered_token_ids[i]) == g_known_empty_token_ids.end()) {
            start_idx = i;
            break;
        }
    }
    
    // Apply start_idx filter to both token_ids and hidden_states
    if (start_idx > 0) {
        filtered_token_ids.erase(filtered_token_ids.begin(), filtered_token_ids.begin() + start_idx);
        filtered_hidden_states.erase(filtered_hidden_states.begin(), 
                                     filtered_hidden_states.begin() + start_idx * n_embd);
    }
    
    // Validate alignment after filtering
    if (filtered_hidden_states.size() != filtered_token_ids.size() * n_embd) {
        LOG_ERR("filter_special_tokens: alignment error after filtering! token_ids.size()=%zu, hidden_states.size()=%zu, n_embd=%d\n",
                filtered_token_ids.size(), filtered_hidden_states.size(), n_embd);
        return;  // Don't update if alignment is broken
    }
    
    // Update input vectors
    token_ids = filtered_token_ids;
    hidden_states = filtered_hidden_states;
    
    if (original_size != token_ids.size()) {
    }
}

// ==============================================================================
// Local TTS Audio Token Generation

// Helper function to play WAV file
static void play_wav_file(const std::string& wav_file_path) {
#ifndef _WIN32
    // Play audio asynchronously using fork() to avoid blocking TTS thread
    pid_t pid = fork();
    if (pid == 0) {
        #ifdef __APPLE__
            execl("/usr/bin/afplay", "afplay", wav_file_path.c_str(), (char*)NULL);
        #else
            execl("/usr/bin/aplay", "aplay", wav_file_path.c_str(), (char*)NULL);
        #endif
        _exit(1);
    } else if (pid > 0) {
        // Parent process: continue without waiting
    } else {
        std::string play_cmd;
        #ifdef __APPLE__
            play_cmd = "afplay \"" + wav_file_path + "\" &";
        #else
            play_cmd = "aplay \"" + wav_file_path + "\" &";
        #endif
        LOG_WRN("TTS: fork() failed, using system() fallback for audio playback\n");
        system(play_cmd.c_str());
    }
#endif
    // Windows: no-op (audio playback handled by frontend)
}


// Helper function to move old output directory to old_output/<id>/
static void move_old_output_to_archive() {
    const std::string base_output_dir = "./tools/omni/output";
    const std::string old_output_base_dir = "./old_output";
    
    // Helper function to check if directory exists and has content
    auto dir_has_content = [](const std::string& dir_path) -> bool {
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0) {
            return false;  // Directory doesn't exist
        }
        if (!(info.st_mode & S_IFDIR)) {
            return false;  // Not a directory
        }
        
        // Check if directory has any files/subdirectories
#ifdef _WIN32
        std::string cmd = "dir /b \"" + dir_path + "\" 2>NUL | findstr /r \".\" >NUL 2>&1";
#else
        std::string cmd = "test -n \"$(ls -A " + dir_path + " 2>/dev/null)\"";
#endif
        int ret = system(cmd.c_str());
        return (ret == 0);  // Returns 0 if directory has content
    };
    
    // Helper function to create directory
    auto create_dir = [](const std::string& dir_path) -> bool {
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0) {
            // Directory doesn't exist, try to create it
            if (!cross_platform_mkdir_p(dir_path)) {
                LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
                return false;
            }
            return true;
        } else if (!(info.st_mode & S_IFDIR)) {
            LOG_ERR("Output path exists but is not a directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    };
    
    // Helper function to find next available ID in old_output directory
    auto get_next_output_id = [](const std::string& old_output_base) -> int {
        // Ensure old_output base directory exists
        cross_platform_mkdir_p(old_output_base);
        
        // Find maximum ID in old_output directory
        int max_id = -1;
#ifdef _WIN32
        std::string find_cmd = "dir /b \"" + old_output_base + "\" 2>NUL";
#else
        std::string find_cmd = "ls -1 " + old_output_base + " 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1";
#endif
        FILE* pipe = popen(find_cmd.c_str(), "r");
        if (pipe) {
            char buffer[128];
#ifdef _WIN32
            // On Windows, read all entries and find the max numeric ID
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                std::string result(buffer);
                while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                    result.pop_back();
                }
                if (!result.empty()) {
                    try {
                        int id = std::stoi(result);
                        if (id > max_id) max_id = id;
                    } catch (...) {}
                }
            }
#else
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                std::string result(buffer);
                // Remove trailing newline
                while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                    result.pop_back();
                }
                if (!result.empty()) {
                    try {
                        max_id = std::stoi(result);
                    } catch (...) {
                        max_id = -1;
                    }
                }
            }
#endif
            pclose(pipe);
        }
        
        return max_id + 1;  // Next ID is max_id + 1 (or 0 if no existing IDs)
    };
    
    // Check if output directory has content
    bool output_has_content = false;
    
    // Check if base output directory exists and has content
    if (dir_has_content(base_output_dir)) {
        output_has_content = true;
    } else {
        // Check subdirectories
        if (dir_has_content(base_output_dir + "/llm_debug") || 
            dir_has_content(base_output_dir + "/tts_txt") ||
            dir_has_content(base_output_dir + "/tts_wav")) {
            output_has_content = true;
        }
    }
    
    if (output_has_content) {
        int next_id = get_next_output_id(old_output_base_dir);
        std::string old_output_dir = old_output_base_dir + "/" + std::to_string(next_id);
        
        // Create old_output/<id> directory
        if (!create_dir(old_output_dir)) {
            LOG_ERR("Failed to create old_output directory: %s\n", old_output_dir.c_str());
        } else {
            // Move entire output directory contents to old_output/<id>/
            // Use find + xargs for more reliable moving
#ifdef _WIN32
            std::string move_cmd = "robocopy \"" + base_output_dir + "\" \"" + old_output_dir + "\" /E /MOVE >NUL 2>&1";
            system(move_cmd.c_str());
            // robocopy returns non-zero on success (exit codes < 8 are success)
            // Re-create the base output dir since robocopy /MOVE removes it
            cross_platform_mkdir_p(base_output_dir);
#else
            std::string move_cmd = "find " + base_output_dir + " -mindepth 1 -maxdepth 1 -exec mv {} " + old_output_dir + "/ \\; 2>/dev/null";
            int ret = system(move_cmd.c_str());
            if (ret == 0) {
            } else {
                // Fallback: try simple mv command
                std::string fallback_cmd = "sh -c 'cd " + base_output_dir + " && mv * " + old_output_dir + "/ 2>/dev/null || true'";
                ret = system(fallback_cmd.c_str());
                if (ret == 0) {
                } else {
                    LOG_WRN("Failed to move old output directory (may be empty or already moved)\n");
                }
            }
#endif
        }
    } else {
    }
}

// Helper function to merge all WAV files into a single file
static void merge_wav_files(const std::string& output_dir, int num_chunks) {
    if (num_chunks == 0) {
        LOG_WRN("TTS: no chunks to merge\n");
        return;
    }
    
    std::string merged_file = output_dir + "/tts_output_merged.wav";
    
    // Check all chunk files exist
    std::vector<std::string> chunk_files;
    for (int i = 0; i < num_chunks; ++i) {
        std::string chunk_file = output_dir + "/tts_output_chunk_" + std::to_string(i) + ".wav";
        struct stat st;
        if (stat(chunk_file.c_str(), &st) == 0 && st.st_size > 0) {
            chunk_files.push_back(chunk_file);
        } else {
            LOG_WRN("TTS: chunk file %s does not exist or is empty\n", chunk_file.c_str());
        }
    }
    
    if (chunk_files.empty()) {
        LOG_WRN("TTS: no valid WAV files to merge\n");
        return;
    }
    
    // Method 1: Use ffmpeg with concat demuxer (most reliable)
    std::string concat_list_file = output_dir + "/concat_list.txt";
    FILE* f_list = fopen(concat_list_file.c_str(), "w");
    if (f_list) {
        for (const auto& chunk_file : chunk_files) {
            fprintf(f_list, "file '%s'\n", chunk_file.c_str());
        }
        fclose(f_list);
        
        std::string ffmpeg_cmd = "ffmpeg -f concat -safe 0 -i \"" + concat_list_file + "\" -c copy \"" + merged_file + "\" -y -loglevel error 2>&1";
        int ret = system(ffmpeg_cmd.c_str());
        unlink(concat_list_file.c_str());  // Clean up temp file
        
        if (ret == 0) {
            struct stat st;
            if (stat(merged_file.c_str(), &st) == 0 && st.st_size > 0) {
                return;  // Success
            }
        }
    }
    
    // Method 2: Fallback to sox (if available)
    std::string sox_cmd = "sox";
    for (const auto& chunk_file : chunk_files) {
        sox_cmd += " \"" + chunk_file + "\"";
    }
    sox_cmd += " \"" + merged_file + "\"";
    int ret = system(sox_cmd.c_str());
    
    if (ret == 0) {
        struct stat st;
        if (stat(merged_file.c_str(), &st) == 0 && st.st_size > 0) {
        } else {
            LOG_WRN("TTS: merged file was not created or is empty (sox)\n");
        }
    } else {
        LOG_WRN("TTS: failed to merge WAV files (tried ffmpeg and sox). Please install ffmpeg or sox.\n");
    }
}

// ==============================================================================
// TTS Thread Function - Duplex Mode
// 双工模式专用的 TTS 线程函数
// 与单工版本的主要差异：
// 1. 不需要 simplex_round_idx 管理和 round_XXX 输出目录
// 2. TTS KV cache 跨 chunk 保持（由 is_end_of_turn 控制是否重置）
// 3. 使用 generate_audio_tokens_local（双工版本，max_audio_tokens=26）
// ==============================================================================
void tts_thread_func_duplex(struct omni_context * ctx_omni, common_params *params) {
    // TTS model state
    int tts_n_past = 0;
    std::vector<llama_token> audio_tokens;
    std::vector<llama_token> all_audio_tokens;
    std::string debug_dir = "";
    bool tts_finish = false;
    bool llm_finish = false;
    int chunk_idx = 0;
    std::string incomplete_bytes;
    
    // 双工模式：固定输出目录（不使用 round_XXX 子目录）
    // 多实例支持：使用可配置的 base_output_dir
    const std::string& base_output_dir = ctx_omni->base_output_dir;
    const std::string tts_output_dir = base_output_dir + "/tts_txt";
    const std::string llm_debug_output_dir = base_output_dir + "/llm_debug";
    const std::string tts_wav_output_dir = base_output_dir + "/tts_wav";
    const bool write_debug = !OMNI_REPLACEMENT_TTS_T2W || std::getenv("OMNI_TTS_DIAGNOSTICS") != nullptr;
    
    // Helper function to create directory
    auto create_dir = [](const std::string& dir_path) {
        if (!cross_platform_mkdir_p(dir_path)) {
            LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    };
    
    // 创建输出目录
    create_dir(tts_output_dir);
    create_dir(llm_debug_output_dir);
    create_dir(tts_wav_output_dir);
    
    // TTS model constants
    const int audio_bos_token_id = 151687;
    const int audio_eos_token_id = audio_bos_token_id + 6561;
    const int text_eos_token_id = 151692;
    const int num_audio_tokens = 6562;
    
    // 用于追踪所有 decode 调用
    int decode_call_idx = 0;
    
    // 标志位：当前 turn 是否已执行 turn_eos flush（修复重复生成问题）
    bool turn_eos_flushed = false;

    print_with_timestamp("TTS thread (duplex mode) started\n");

    // Multi Round Persistent Loop
    while(tts_thread_running) {
        if (!tts_thread_running) {
            break;
        }
        
        // 打断检测
        if (ctx_omni->break_event.load()) {
            // 清空 TTS 队列
            {
                std::lock_guard<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
                auto& queue = ctx_omni->tts_thread_info->queue;
                while (!queue.empty()) {
                    LLMOut *llm_out = queue.front();
                    queue.pop();
                    delete llm_out;
                }
            }
            // 重置状态
            llm_finish = false;
            tts_finish = false;
            chunk_idx = 0;
            tts_n_past = 0;
            audio_tokens.clear();
            all_audio_tokens.clear();
            incomplete_bytes.clear();
            // 清除 TTS KV cache
            llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
            if (mem) {
                llama_memory_seq_rm(mem, 0, 0, -1);
                print_with_timestamp("TTS Duplex: break_event - cleared TTS KV cache\n");
            }
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            continue;
        }

        std::string llm_text = "";
        std::vector<llama_token> current_chunk_token_ids;
        std::vector<float> current_chunk_hidden_states;
        int current_chunk_n_embd = 0;
        std::chrono::steady_clock::time_point current_chunk_queue_enter{};
        uint64_t current_frame_id = 0;
        int current_source_chunk_idx = -1;
        auto stamp_t2w_identity = [&](T2WOut * out) {
            if (out == nullptr) return;
            out->frame_id = current_frame_id;
            out->chunk_idx = current_source_chunk_idx >= 0 ? current_source_chunk_idx : chunk_idx;
        };
        
        // 从 LLMOut 获取 is_end_of_turn 状态（修复双工缺字：状态随数据传递，避免全局时序问题）
        bool accumulated_is_end_of_turn = false;
            
        // Wait for queue
        if (!llm_finish || (llm_finish && llm_text.empty())) {
            std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
            auto& queue = ctx_omni->tts_thread_info->queue;
            ctx_omni->tts_thread_info->cv.wait(lock, [&] { 
                return !queue.empty() || !tts_thread_running || ctx_omni->break_event.load(); 
            });
            
            if (ctx_omni->break_event.load()) {
                lock.unlock();
                continue;
            }
                
            if (!tts_thread_running) {
                break;
            }
            
            // 清空 current_chunk 数据
            current_chunk_token_ids.clear();
            current_chunk_hidden_states.clear();
            current_chunk_n_embd = 0;
            accumulated_is_end_of_turn = false;
            
            // 累积所有队列中的数据
            while (!queue.empty()) {
                LLMOut *llm_out = queue.front();
                if (llm_out != nullptr && current_frame_id != 0 && llm_out->frame_id != 0 &&
                        llm_out->frame_id != current_frame_id) {
                    break;
                }
                if (current_chunk_queue_enter.time_since_epoch().count() == 0 &&
                        llm_out != nullptr && llm_out->enqueue_time.time_since_epoch().count() != 0) {
                    current_chunk_queue_enter = llm_out->enqueue_time;
                }
                if (llm_out != nullptr) {
                    if (current_frame_id == 0) current_frame_id = llm_out->frame_id;
                    if (llm_out->chunk_idx >= 0) current_source_chunk_idx = llm_out->chunk_idx;
                }
                llm_finish |= llm_out->llm_finish;
                bool item_is_eot = llm_out->is_end_of_turn;
                accumulated_is_end_of_turn |= item_is_eot;
                
                if (!ctx_omni->speek_done || ctx_omni->duplex_mode) {
                    llm_text += llm_out->text;
                    debug_dir = llm_out->debug_dir;
                }
                
                // 累积数据
                if (!llm_out->token_ids.empty() && !llm_out->hidden_states.empty()) {
                    current_chunk_token_ids.insert(current_chunk_token_ids.end(), 
                                                   llm_out->token_ids.begin(), 
                                                   llm_out->token_ids.end());
                    current_chunk_hidden_states.insert(current_chunk_hidden_states.end(), 
                                                       llm_out->hidden_states.begin(), 
                                                       llm_out->hidden_states.end());
                    current_chunk_n_embd = llm_out->n_embd;
                }
                delete llm_out;
                queue.pop();

                // 遇到 EOT 边界立即停止，下一轮的 item 留给下次迭代处理
                if (item_is_eot) {
                    break;
                }
            }
            lock.unlock();
            ctx_omni->tts_thread_info->cv.notify_all();
            
            // 双工模式：如果有新数据，继续处理
            // 每次取出 LLMOut 后都打印状态
            // 使用 accumulated_is_end_of_turn 而非全局状态（修复双工缺字）
            print_with_timestamp("TTS Duplex: after queue - speek_done=%d, llm_finish=%d, token_ids.size=%zu, is_end_of_turn=%d, llm_text.len=%zu\n",
                                ctx_omni->speek_done ? 1 : 0,
                                llm_finish ? 1 : 0,
                                current_chunk_token_ids.size(),
                                accumulated_is_end_of_turn ? 1 : 0,
                                llm_text.size());
            
            if (ctx_omni->speek_done && llm_finish) {
                if (ctx_omni->duplex_mode && !current_chunk_token_ids.empty()) {
                    // 新一轮 SPEAK：重置 TTS 状态，防止上轮残留污染
                    if (chunk_idx > 0) {
                        chunk_idx = 0;
                        tts_n_past = 0;
                        audio_tokens.clear();
                        llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                        if (mem) {
                            llama_memory_seq_rm(mem, 0, 0, -1);
                        }
                        ctx_omni->tts_n_past_accumulated = 0;
                        ctx_omni->tts_all_generated_tokens.clear();
                        ctx_omni->tts_condition_saved = false;
                    }
                    ctx_omni->speek_done = false;
                } else if (ctx_omni->duplex_mode && accumulated_is_end_of_turn) {
                    ctx_omni->speek_done = false;
                    print_with_timestamp("TTS Duplex: is_end_of_turn=true, will call TTS to flush buffer\n");
                } else {
                    // LISTEN/CHUNK_EOS 且没有实际文本
                    decode_call_idx++;
                    llm_finish = false;
                    llm_text.clear();
                    
                    if (ctx_omni->t2w_thread_info) {
                        T2WOut *t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final = false;
                        t2w_out->is_chunk_end = true;
                        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                        stamp_t2w_identity(t2w_out);
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    continue;
                }
            }
        }
            
        std::string& response = llm_text;
        // 处理不完整的 UTF-8 字节
        if (!incomplete_bytes.empty()) {
            response = incomplete_bytes + response;
            incomplete_bytes.clear();
        }
        size_t incomplete_len = findIncompleteUtf8(response);
        if (incomplete_len > 0) {
            incomplete_bytes = response.substr(response.size() - incomplete_len, incomplete_len);
            response = response.substr(0, response.size() - incomplete_len);
        }
        
        // Skip empty responses
        if (response.empty() && !llm_finish) {
            if (ctx_omni->speek_done) {
                llm_finish = false;
                if (ctx_omni->duplex_mode && !accumulated_is_end_of_turn) {
                    // 保持状态
                } else {
                    // 轮次结束，重置 TTS 状态
                    chunk_idx = 0;
                    tts_n_past = 0;
                    audio_tokens.clear();
                    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                    if (mem) {
                        llama_memory_seq_rm(mem, 0, 0, -1);
                    }
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                }
            }
            continue;
        }
            
        // Tokenize text input
        std::vector<llama_token> text_tokens = common_tokenize(ctx_omni->ctx_tts_llama, response, false, true);
        
        if (text_tokens.empty() && !llm_finish) {
            continue;
        }
        
        // Handle empty final chunk
        if (text_tokens.empty() && response.empty() && llm_finish) {
            ctx_omni->speek_done = true;
            ctx_omni->warmup_done = true;
            speek_cv.notify_all();
            
            if (ctx_omni->duplex_mode && !accumulated_is_end_of_turn) {
                // LISTEN/CHUNK_EOS
                if (ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens.clear();
                    t2w_out->is_final = false;
                    t2w_out->is_chunk_end = true;
                    t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                    stamp_t2w_identity(t2w_out);
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                }
                tts_finish = false;
                llm_finish = false;
                continue;
            } else if (ctx_omni->duplex_mode && accumulated_is_end_of_turn) {
                // turn 结束，继续调用 TTS flush buffer
                print_with_timestamp("TTS Duplex: empty final chunk but is_end_of_turn=true, will call TTS to flush buffer\n");
            } else {
                // 非双工模式
                if (ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens.clear();
                    t2w_out->is_final = true;
                    t2w_out->is_chunk_end = false;
                    t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                }
                tts_finish = false;
                llm_finish = false;
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                if (mem && !ctx_omni->duplex_mode) {
                    llama_memory_seq_rm(mem, 0, 0, -1);
                }
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                continue;
            }
        }
        
        // Check for LLM data
        bool has_llm_data = (!current_chunk_token_ids.empty() && !current_chunk_hidden_states.empty() && current_chunk_n_embd > 0);
        
        if (has_llm_data) {
            int current_chunk_idx = chunk_idx;
            
            // 收到有效 LLM 数据，重置 turn_eos_flushed 标志位
            if (turn_eos_flushed) {
                turn_eos_flushed = false;
            }
            
            print_with_timestamp("TTS Duplex: processing chunk_idx=%d, n_tokens=%zu, is_end_of_turn=%d\n",
                                chunk_idx, current_chunk_token_ids.size(), accumulated_is_end_of_turn ? 1 : 0);
            
            // 安全检查
            if (current_chunk_n_embd <= 0 || current_chunk_n_embd > 16384) {
                LOG_ERR("TTS Duplex: invalid current_chunk_n_embd=%d\n", current_chunk_n_embd);
                continue;
            }
            
            size_t expected_hidden_size = current_chunk_token_ids.size() * current_chunk_n_embd;
            if (current_chunk_hidden_states.size() != expected_hidden_size) {
                LOG_ERR("TTS Duplex: hidden_states size mismatch\n");
                continue;
            }
            
            // Filter special tokens
            int n_tokens_orig = (int)(current_chunk_hidden_states.size() / current_chunk_n_embd);
            std::vector<llama_token> filtered_token_ids = current_chunk_token_ids;
            std::vector<float> filtered_hidden_states = current_chunk_hidden_states;
            filter_special_tokens(filtered_token_ids, filtered_hidden_states, current_chunk_n_embd);
            int n_tokens_filtered = (int)(filtered_hidden_states.size() / current_chunk_n_embd);
            
            if (n_tokens_filtered <= 0) {
                continue;
            }
            
            // Compute merged embeddings
            const int tts_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
            std::vector<float> merged_embeddings;
            bool merged_success = false;
            
            if (ctx_omni->emb_text_weight && ctx_omni->projector_semantic_linear1_weight) {
                // Step 1: emb_text
                std::vector<float> llm_embeds(n_tokens_filtered * tts_n_embd, 0.0f);
                bool emb_text_success = true;
                for (int i = 0; i < n_tokens_filtered; i++) {
                    if (!tts_emb_text(ctx_omni, filtered_token_ids[i], llm_embeds.data() + i * tts_n_embd, tts_n_embd)) {
                        emb_text_success = false;
                        break;
                    }
                }
                
                if (emb_text_success) {
                    // Step 2: projector_semantic
                    std::vector<float> projected_hidden(n_tokens_filtered * tts_n_embd, 0.0f);
                    bool projector_success = tts_projector_semantic(ctx_omni, filtered_hidden_states.data(),
                                                                     n_tokens_filtered, current_chunk_n_embd,
                                                                     projected_hidden.data(), tts_n_embd);
                    
                    if (projector_success) {
                        // Step 3: Normalize
                        normalize_l2_per_token(projected_hidden.data(), n_tokens_filtered, tts_n_embd);
                        
                        // Step 4: Merge
                        size_t merge_size = (size_t)n_tokens_filtered * tts_n_embd;
                        merged_embeddings.resize(merge_size);
                        for (size_t i = 0; i < merge_size; i++) {
                            merged_embeddings[i] = llm_embeds[i] + projected_hidden[i];
                        }
                        
                        // Add audio_bos_embed
                        std::vector<float> audio_bos_embed(tts_n_embd, 0.0f);
                        if (tts_emb_text(ctx_omni, audio_bos_token_id, audio_bos_embed.data(), tts_n_embd)) {
                            merged_embeddings.insert(merged_embeddings.end(), audio_bos_embed.begin(), audio_bos_embed.end());
                            n_tokens_filtered += 1;
                        }
                        
                        merged_success = true;
                    }
                }
            }
            
            // 保存 LLM debug 数据（追加模式，统一放在 llm_debug 目录）
            // Save LLM debug data: text, token_ids, hidden_states, and merged embeddings
            const bool write_debug = !OMNI_REPLACEMENT_TTS_T2W || std::getenv("OMNI_TTS_DIAGNOSTICS") != nullptr;
            if (write_debug) {
                // 1. Save LLM text output (追加模式，只记录纯文本，不记录 special tokens)
                {
                    // 从 llm_text 中过滤掉 special tokens
                    std::string clean_text = llm_text;
                    // 移除所有 [[XXX]] 格式的 special token 标记
                    size_t pos = 0;
                    while ((pos = clean_text.find("[[")) != std::string::npos) {
                        size_t end_pos = clean_text.find("]]", pos);
                        if (end_pos != std::string::npos) {
                            clean_text.erase(pos, end_pos - pos + 2);
                        } else {
                            break;
                        }
                    }
                    // 移除多余的空格
                    while (clean_text.find("  ") != std::string::npos) {
                        size_t space_pos = clean_text.find("  ");
                        clean_text.erase(space_pos, 1);
                    }
                    // 去除首尾空格
                    while (!clean_text.empty() && clean_text[0] == ' ') clean_text.erase(0, 1);
                    while (!clean_text.empty() && clean_text.back() == ' ') clean_text.pop_back();
                    
                    // 只有非空文本才记录
                    if (!clean_text.empty()) {
                        std::string text_file = llm_debug_output_dir + "/llm_text.txt";
                        FILE *f_text = fopen(text_file.c_str(), "a");
                        if (f_text) {
                            fprintf(f_text, "%s\n", clean_text.c_str());
                            fclose(f_text);
                        }
                    }
                    decode_call_idx++;
                }
                
                // 2. Save LLM token IDs (追加模式)
                std::string token_ids_file = llm_debug_output_dir + "/llm_token_ids.txt";
                FILE *f_tokens = fopen(token_ids_file.c_str(), "a");
                if (f_tokens) {
                    fprintf(f_tokens, "[chunk_%d] ", current_chunk_idx);
                    for (size_t i = 0; i < current_chunk_token_ids.size(); ++i) {
                        fprintf(f_tokens, "%d", current_chunk_token_ids[i]);
                        if (i < current_chunk_token_ids.size() - 1) fprintf(f_tokens, " ");
                    }
                    fprintf(f_tokens, "\n");
                    fclose(f_tokens);
                }
                
                // 3. Save LLM hidden states (追加模式，文本格式)
                int n_tokens_orig = (int)(current_chunk_hidden_states.size() / current_chunk_n_embd);
                std::string hidden_txt_file = llm_debug_output_dir + "/llm_hidden_states.txt";
                FILE *f_hidden_txt = fopen(hidden_txt_file.c_str(), "a");
                if (f_hidden_txt) {
                    fprintf(f_hidden_txt, "[chunk_%d] Hidden States (shape: [%d, %d]):\n", current_chunk_idx, n_tokens_orig, current_chunk_n_embd);
                    for (int i = 0; i < n_tokens_orig; ++i) {
                        fprintf(f_hidden_txt, "  Token %d: %.6f %.6f %.6f ... (first 3 values)\n", i,
                                current_chunk_hidden_states[i * current_chunk_n_embd + 0],
                                current_chunk_hidden_states[i * current_chunk_n_embd + 1],
                                current_chunk_hidden_states[i * current_chunk_n_embd + 2]);
                    }
                    fclose(f_hidden_txt);
                }
                
                // 4. Save merged embeddings (追加模式) if successfully computed
                if (merged_success && !merged_embeddings.empty()) {
                    std::string merged_txt_file = llm_debug_output_dir + "/merged_embeddings.txt";
                    FILE *f_merged_txt = fopen(merged_txt_file.c_str(), "a");
                    if (f_merged_txt) {
                        fprintf(f_merged_txt, "[chunk_%d] Merged Embeddings (shape: [%d, %d]):\n", current_chunk_idx, n_tokens_filtered, tts_n_embd);
                        for (int i = 0; i < n_tokens_filtered; ++i) {
                            fprintf(f_merged_txt, "  Token %d: %.6f %.6f %.6f ... (first 3 values)\n", i,
                                    merged_embeddings[i * tts_n_embd + 0],
                                    merged_embeddings[i * tts_n_embd + 1],
                                    merged_embeddings[i * tts_n_embd + 2]);
                        }
                        fclose(f_merged_txt);
                    }
                }
            }
            
            // Generate audio tokens using duplex version
            // 当 is_end_of_turn=true 时，即使 merged_embeddings 为空也要调用 TTS 生成（只添加 text_eos_embed），
            // 让 TTS flush 其 buffer（修复尾音问题）
            bool should_call_tts = (merged_success && !merged_embeddings.empty()) || 
                                   (accumulated_is_end_of_turn && ctx_omni->duplex_mode);
            
            if (should_call_tts) {
                std::vector<int32_t> audio_tokens_out;
                bool is_end_of_turn = accumulated_is_end_of_turn;
                
                if (merged_embeddings.empty() && is_end_of_turn) {
                    print_with_timestamp("TTS Duplex: is_end_of_turn=true with empty embeddings, calling TTS to flush\n");
                    n_tokens_filtered = 0;
                }
                
                int tts_selected = SM_PLAN_GLOBAL;
                sm_enter(ctx_omni->sm_sched, SM_BIT_TTS);
                tts_selected = ctx_omni->sm_sched.applied_plan;
                bool tts_gen_success = generate_audio_tokens_local(ctx_omni, params, merged_embeddings,
                                                n_tokens_filtered, tts_n_embd, current_chunk_idx,
                                                audio_tokens_out, is_end_of_turn,
                                                write_debug ? tts_wav_output_dir : std::string(),
                                                current_chunk_queue_enter, current_frame_id,
                                                ctx_omni->sm_sched.enabled ? sm_tts_segment_cb : nullptr,
                                                ctx_omni->sm_sched.enabled ? static_cast<void *>(&ctx_omni->sm_sched) : nullptr);
                sm_exit(ctx_omni->sm_sched, SM_BIT_TTS);
                sm_task_finish(ctx_omni->sm_sched, tts_selected, SM_BIT_TTS, "tts", current_frame_id);
                
                if (tts_gen_success) {
                    all_audio_tokens.insert(all_audio_tokens.end(), audio_tokens_out.begin(), audio_tokens_out.end());
                    
                    if (is_end_of_turn && ctx_omni->duplex_mode) {
                        turn_eos_flushed = true;
                    }
                }
            }
            
            ++chunk_idx;
            llm_text.clear();
            response.clear();
            
            // Handle final chunk
            if (llm_finish) {
                tts_finish = true;
                ctx_omni->speek_done = true;
                ctx_omni->warmup_done = true;
                speek_cv.notify_all();
                
                if (write_debug) {
                    merge_wav_files(tts_wav_output_dir, chunk_idx + 1);
                }
                
                if (ctx_omni->duplex_mode && !accumulated_is_end_of_turn) {
                    // LISTEN/CHUNK_EOS: 保持 TTS 状态
                    if (ctx_omni->t2w_thread_info) {
                        T2WOut *t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final = false;
                        t2w_out->is_chunk_end = true;
                        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                        stamp_t2w_identity(t2w_out);
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    llm_finish = false;
                    tts_finish = false;
                } else {
                    // 真正的轮次结束
                    if (ctx_omni->t2w_thread_info && !turn_eos_flushed) {
                        T2WOut *t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final = true;
                        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                        stamp_t2w_identity(t2w_out);
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    // 清除 TTS KV cache
                    llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
                    if (mem) {
                        llama_memory_seq_rm(mem, 0, 0, -1);
                    }
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                    tts_n_past = 0;
                    audio_tokens.clear();
                    all_audio_tokens.clear();
                    llm_finish = false;
                    tts_finish = false;
                }
            }
            continue;
        } else if (ctx_omni->duplex_mode && accumulated_is_end_of_turn && llm_finish) {
            // turn 结束但没有新数据，调用 TTS flush buffer
            if (turn_eos_flushed) {
                print_with_timestamp("TTS Duplex: turn_eos already flushed, skipping TTS generation\n");
            } else {
                print_with_timestamp("TTS Duplex: no LLM data but is_end_of_turn=true, calling TTS to flush buffer\n");
                
                const int tts_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
                std::vector<float> empty_embeddings;
                std::vector<int32_t> audio_tokens_out;
                int n_tokens_for_tts = 0;
                int current_chunk_idx = chunk_idx;
                
                int tts_selected = SM_PLAN_GLOBAL;
                sm_enter(ctx_omni->sm_sched, SM_BIT_TTS);
                tts_selected = ctx_omni->sm_sched.applied_plan;
                bool tts_gen_success = generate_audio_tokens_local(ctx_omni, params, empty_embeddings,
                                                n_tokens_for_tts, tts_n_embd, current_chunk_idx,
                                                audio_tokens_out, true,
                                                write_debug ? tts_wav_output_dir : std::string(),
                                                current_chunk_queue_enter, current_frame_id,
                                                ctx_omni->sm_sched.enabled ? sm_tts_segment_cb : nullptr,
                                                ctx_omni->sm_sched.enabled ? static_cast<void *>(&ctx_omni->sm_sched) : nullptr);
                sm_exit(ctx_omni->sm_sched, SM_BIT_TTS);
                sm_task_finish(ctx_omni->sm_sched, tts_selected, SM_BIT_TTS, "tts", current_frame_id);
                
                if (tts_gen_success) {
                    all_audio_tokens.insert(all_audio_tokens.end(), audio_tokens_out.begin(), audio_tokens_out.end());
                } else {
                    if (ctx_omni->t2w_thread_info) {
                        T2WOut *t2w_out = new T2WOut();
                        t2w_out->audio_tokens.clear();
                        t2w_out->is_final = true;
                        t2w_out->is_chunk_end = false;
                        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                        stamp_t2w_identity(t2w_out);
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                }
                
                turn_eos_flushed = true;
            }
            
            // 重置 TTS 状态
            if (write_debug) {
                merge_wav_files(tts_wav_output_dir, chunk_idx + 1);
            }
            llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
            if (mem) {
                llama_memory_seq_rm(mem, 0, 0, -1);
            }
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            tts_n_past = 0;
            audio_tokens.clear();
            all_audio_tokens.clear();
            llm_finish = false;
            tts_finish = false;
            ctx_omni->speek_done = true;
            ctx_omni->warmup_done = true;
            speek_cv.notify_all();
            
            llm_text.clear();
            response.clear();
            continue;
        } else {
            llm_text.clear();
            response.clear();
            continue;
        }
    }
    
    print_with_timestamp("TTS thread (duplex mode) stopped\n");
}

// 遗留：单工（非 duplex）模式 TTS 线程。活跃路径是 tts_thread_func_duplex（见上）。
void tts_thread_func(struct omni_context * ctx_omni, common_params *params) {
    // TTS model state
    int tts_n_past = 0;  // TTS model's n_past counter
    std::vector<llama_token> audio_tokens;  // Collected audio tokens
    std::vector<llama_token> all_audio_tokens;  // All audio tokens collected across all chunks
    std::string debug_dir = "";
    bool tts_finish = false;
    bool llm_finish = false;
    int chunk_idx = 0;
    std::string incomplete_bytes;
    
    // 多实例支持：使用可配置的 base_output_dir
    const std::string& base_output_dir = ctx_omni->base_output_dir;
    
    // Helper function to create directory
    struct stat info;
    auto create_dir = [](const std::string& dir_path) {
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0) {
            // Directory doesn't exist, try to create it
            if (!cross_platform_mkdir_p(dir_path)) {
                LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
                return false;
            }
            return true;
        } else if (!(info.st_mode & S_IFDIR)) {
            LOG_ERR("Output path exists but is not a directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    };
    
    // Helper function to get round-specific output directory
    // 单工模式下返回 output/round_XXX，双工模式下返回 output
    auto get_round_output_dir = [&]() -> std::string {
        if (!ctx_omni->duplex_mode) {
            // 单工模式：使用 round_XXX 子目录
            char round_dir[512];
            snprintf(round_dir, sizeof(round_dir), "%s/round_%03d", base_output_dir.c_str(), ctx_omni->simplex_round_idx);
            return std::string(round_dir);
        } else {
            // 双工模式：直接使用 base_output_dir
            return base_output_dir;
        }
    };
    
    // 动态目录路径（每轮开始时更新）
    std::string current_round_dir = get_round_output_dir();
    std::string tts_output_dir = current_round_dir + "/tts_txt";
    std::string llm_debug_output_dir = current_round_dir + "/llm_debug";
    std::string tts_wav_output_dir = current_round_dir + "/tts_wav";
    const bool write_debug = !OMNI_REPLACEMENT_TTS_T2W || std::getenv("OMNI_TTS_DIAGNOSTICS") != nullptr;
    
    // 记录上一次创建目录的 round_idx，避免重复创建
    int last_created_round_idx = -1;
    
    // Helper function to update output directories for current round
    auto update_output_dirs = [&]() {
        current_round_dir = get_round_output_dir();
        tts_output_dir = current_round_dir + "/tts_txt";
        llm_debug_output_dir = current_round_dir + "/llm_debug";
        tts_wav_output_dir = current_round_dir + "/tts_wav";
        
        // 只在新的 round 时创建目录
        if (ctx_omni->simplex_round_idx != last_created_round_idx || ctx_omni->duplex_mode) {
            create_dir(tts_output_dir);
            create_dir(llm_debug_output_dir);
            create_dir(tts_wav_output_dir);
            last_created_round_idx = ctx_omni->simplex_round_idx;
            
            if (!ctx_omni->duplex_mode) {
                print_with_timestamp("TTS: 创建单工模式输出目录: %s\n", current_round_dir.c_str());
            }
        }
    };
    
    // 初始创建目录
    update_output_dirs();
    
    // TTS model constants from config.json
    const int audio_bos_token_id = 151687;
    // Audio EOS token: relative index 6561, absolute ID = 151687 + 6561 = 158248
    const int audio_eos_token_id = audio_bos_token_id + 6561;  // 158248
    const int text_eos_token_id = 151692;  // 用于文本结束检测
    const int spk_emb_token_id = 21143;
    const int num_audio_tokens = 6562;
    const int max_audio_tokens = 1000;  // Maximum audio tokens to generate per text chunk
    
    // WAV timing log file (will be created in current round directory)
    auto create_wav_timing_file = [&]() {
        std::string wav_timing_file = tts_wav_output_dir + "/wav_timing.txt";
        FILE* f_timing = fopen(wav_timing_file.c_str(), "w");
        if (f_timing) {
            fprintf(f_timing, "# WAV file generation timing log\n");
            fprintf(f_timing, "# Format: chunk_index, elapsed_time_ms (since stream_decode start), file_size_bytes, request_duration_ms\n");
            fprintf(f_timing, "# Time 0 is when stream_decode() function starts\n");
            fclose(f_timing);
        }
    };
    if (write_debug) {
        create_wav_timing_file();
    }

    print_with_timestamp("TTS thread started\n");

    // 标志位：当前 break_event 是否已经递增过 round_idx
    // 防止在等待 T2W 线程清除 break_event 期间重复递增
    bool break_round_incremented = false;

    // Multi Round Persistent Loop
    while(tts_thread_running) {
        
        if (!tts_thread_running) {
            break;
        }
        
        // 打断检测：检测 break_event 并清空队列、重置状态
        if (ctx_omni->break_event.load()) {
            // 清空 TTS 队列（队列元素类型是 LLMOut*）
            {
                std::lock_guard<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
                auto& queue = ctx_omni->tts_thread_info->queue;
                while (!queue.empty()) {
                    LLMOut *llm_out = queue.front();
                    queue.pop();
                    delete llm_out;
                }
            }
            
            // 打断时递增 round 索引（只递增一次）
            if (!ctx_omni->duplex_mode && !break_round_incremented) {
                ctx_omni->simplex_round_idx++;
                // wav_turn_base 的递增已移到 T2W 线程的打断处理中
                // 原因：避免竞态条件，确保 T2W 先处理完当前轮次的所有数据再递增
                // ctx_omni->wav_turn_base += 1000;  // 已移到 T2W 线程
                break_round_incremented = true;  // 标记已递增，防止重复
                print_with_timestamp("TTS: 打断触发，下一轮 round_idx=%d\n", ctx_omni->simplex_round_idx);
            }
            
            // 重置状态
            llm_finish = false;
            tts_finish = false;
            chunk_idx = 0;
            tts_n_past = 0;
            audio_tokens.clear();
            all_audio_tokens.clear();
            incomplete_bytes.clear();  // 清理不完整的 UTF-8 字节（多轮对话）
            // 清理 TTS 跨 chunk 累积状态（避免下一轮串扰）
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            // 不清除 break_event，让 T2W 线程也能检测到
            continue;
        }
        
        // break_event 被 T2W 线程清除后，重置标志位
        if (break_round_incremented && !ctx_omni->break_event.load()) {
            break_round_incremented = false;
        }

        std::string llm_text = "";
        // 保存当前chunk的token IDs和hidden states用于TTS条件生成
        std::vector<llama_token> current_chunk_token_ids;
        std::vector<float> current_chunk_hidden_states;
        int current_chunk_n_embd = 0;
            
        // Always wait for queue if not finished, or if finished but need to reset state
        if (!llm_finish || (llm_finish && llm_text.empty())) {
            std::unique_lock<std::mutex> lock(ctx_omni->tts_thread_info->mtx);
                
            auto& queue = ctx_omni->tts_thread_info->queue;
            // 打断检测：在等待时也监听 break_event
            ctx_omni->tts_thread_info->cv.wait(lock, [&] { 
                return !queue.empty() || !tts_thread_running || ctx_omni->break_event.load(); 
            });
            
            // 检测到 break_event 时跳过当前处理
            if (ctx_omni->break_event.load()) {
                lock.unlock();
                continue;
            }
                
            if (!tts_thread_running) {
                break;
            }
            // 每次只处理一个 chunk，不要一次性取出所有
            // 之前的问题：while (!queue.empty()) 会取出所有 chunk，导致：
            // - llm_text 累积了多个 chunk 的文本
            // - 但 token_ids/hidden_states 被覆盖，只保留最后一个 chunk
            // - TTS 用 10 token 的 condition 去生成 20 token 文本的语音，内容丢失
            //
            // Python 逻辑：每次只处理一个 chunk，生成对应的 audio tokens
            if (!queue.empty()) {
                LLMOut *llm_out = queue.front();
                llm_finish |= llm_out->llm_finish;
                // 只取一个 chunk 的数据
                if (!ctx_omni->speek_done || ctx_omni->duplex_mode) {
                    llm_text = llm_out->text;  // 注意：= 而不是 +=
                    debug_dir = llm_out->debug_dir;
                }
                // 保存这一个 chunk 的 token IDs 和 hidden states
                if (!llm_out->token_ids.empty() && !llm_out->hidden_states.empty()) {
                    current_chunk_token_ids = llm_out->token_ids;
                    current_chunk_hidden_states = llm_out->hidden_states;
                    current_chunk_n_embd = llm_out->n_embd;
                    
                    // 打印 TTS 接收到的数据
                    std::string token_ids_str = "";
                    for (size_t i = 0; i < current_chunk_token_ids.size() && i < 20; i++) {
                        token_ids_str += std::to_string(current_chunk_token_ids[i]);
                        if (i < current_chunk_token_ids.size() - 1 && i < 19) token_ids_str += " ";
                    }
                    if (current_chunk_token_ids.size() > 20) token_ids_str += "...";
                    
                    print_with_timestamp("TTS<-LLM: chunk_idx=%d, text='%s', n_tokens=%zu, hidden_size=%zu, token_ids=[%s]\n",
                                        chunk_idx,
                                        llm_text.c_str(),
                                        current_chunk_token_ids.size(),
                                        current_chunk_hidden_states.size(),
                                        token_ids_str.c_str());
                }
                delete llm_out;
                queue.pop();
            }
            lock.unlock();
            ctx_omni->tts_thread_info->cv.notify_all();
            
            // 打印取出数据后的关键状态
            print_with_timestamp("TTS: after queue pop - speek_done=%d, llm_finish=%d, llm_text.empty=%d, token_ids.size=%zu\n",
                                ctx_omni->speek_done, llm_finish, llm_text.empty(), current_chunk_token_ids.size());
            
            // If speek_done is true but we received llm_finish=true, handle state transition
            if (ctx_omni->speek_done && llm_finish) {
                print_with_timestamp("TTS: speek_done=true and llm_finish=true, resetting state for next round\n");
                // 在双工模式下，如果有新数据，应该继续处理（与 Python 对齐）
                // Python: 每次 streaming_generate 调用都会独立处理，不会因为之前的状态跳过
                if (ctx_omni->duplex_mode && !current_chunk_token_ids.empty()) {
                    // 双工模式下有新数据，重置 speek_done 并继续处理
                    ctx_omni->speek_done = false;
                    // 不执行 continue，继续后续的 TTS 处理
                } else {
                    // 没有新数据，或者非双工模式：重置状态并跳过
                    llm_finish = false;
                    llm_text.clear();
                    chunk_idx = 0;
                    tts_n_past = 0;
                    audio_tokens.clear();
                    // 清理 TTS 跨 chunk 累积状态（避免下一轮串扰）
                    ctx_omni->tts_n_past_accumulated = 0;
                    ctx_omni->tts_all_generated_tokens.clear();
                    ctx_omni->tts_condition_saved = false;
                    continue;  // Skip processing and wait for next input
                }
            }
        }
            
        std::string& response = llm_text;
        // 拼接上一个循环保存的不完整字节
        if (!incomplete_bytes.empty()) {
            print_with_timestamp("TTS: prepending incomplete_bytes (len=%zu) to response (len=%zu)\n", 
                                incomplete_bytes.length(), response.length());
            response = incomplete_bytes + response;
            incomplete_bytes.clear();
        }
        // 检查当前响应是否有不完整的 UTF-8 序列
        size_t incomplete_len = findIncompleteUtf8(response);
        if (incomplete_len > 0) {
            print_with_timestamp("TTS: detected incomplete UTF-8 sequence at end: incomplete_len=%zu, response_len=%zu\n", 
                                incomplete_len, response.length());
            // 保存不完整的字节
            incomplete_bytes = response.substr(response.size() - incomplete_len, incomplete_len);
            // 仅处理完整的部分
            response = response.substr(0, response.size() - incomplete_len);
            print_with_timestamp("TTS: after truncation: response_len=%zu, incomplete_bytes_len=%zu\n", 
                                response.length(), incomplete_bytes.length());
        } else {
            // 确保没有残留的不完整字节
            incomplete_bytes.clear();
        }
        
        // 打印 response 状态
        print_with_timestamp("TTS: before empty check - response.empty=%d, response='%s', llm_finish=%d\n",
                            response.empty(), response.substr(0, 50).c_str(), llm_finish);
        
        // Skip empty responses (but allow processing if llm_finish is true to handle end of generation)
        if (response.empty() && !llm_finish) {
            // If speek_done is true but we're still getting empty responses, just continue waiting
            // Don't reset speek_done here - let stream_prefill reset it after it wakes up
            if (ctx_omni->speek_done) {
                print_with_timestamp("TTS: speek_done=true with empty response, keeping state (waiting for stream_prefill)\n");
                llm_finish = false;
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                // 清理 TTS 跨 chunk 累积状态（避免下一轮串扰）
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                // NOTE: 不重置 speek_done，让 stream_prefill 完成等待后重置
            }
            continue;
        }
        fflush(stdout);
            
        // 空 response + llm_finish 的提前检查
        // 必须在 tokenize 之前检查，因为空字符串 tokenize 可能返回非空结果（BOS token）
        if (response.empty() && llm_finish) {
            // 当收到 llm_finish=true 但没有新数据时，
            // 需要 flush tts_token_buffer 中剩余的 tokens，并发送 is_final=true 到 T2W
            print_with_timestamp("TTS: received llm_finish=true with no data, finalizing (tts_token_buffer=%zu)\n",
                                ctx_omni->tts_token_buffer.size());
            
            // Flush tts_token_buffer 中剩余的 tokens 到 T2W
            if (ctx_omni->t2w_thread_info && !ctx_omni->tts_token_buffer.empty()) {
                T2WOut *t2w_out = new T2WOut();
                t2w_out->audio_tokens.assign(ctx_omni->tts_token_buffer.begin(), 
                                             ctx_omni->tts_token_buffer.end());
                t2w_out->is_final = false;  // 先发送剩余 tokens
                t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                {
                    std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    ctx_omni->t2w_thread_info->queue.push(t2w_out);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
                print_with_timestamp("TTS: flushed %zu remaining tokens from tts_token_buffer\n", 
                                    ctx_omni->tts_token_buffer.size());
                ctx_omni->tts_token_buffer.clear();
            }
            
            // 发送 is_final=true 到 T2W，让 T2W 写入 generation_done.flag
            if (ctx_omni->t2w_thread_info) {
                // 保存当前 round_idx 用于 T2W（递增前的值）
                int current_round_idx = ctx_omni->simplex_round_idx;
                
                // 单工模式：递增 round_idx
                if (!ctx_omni->duplex_mode) {
                    ctx_omni->simplex_round_idx++;
                    print_with_timestamp("TTS: 单工模式轮次结束，下一轮 round_idx=%d\n", ctx_omni->simplex_round_idx);
                }
                
                T2WOut *t2w_final = new T2WOut();
                t2w_final->audio_tokens.clear();
                t2w_final->is_final = true;
                t2w_final->round_idx = current_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                {
                    std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    ctx_omni->t2w_thread_info->queue.push(t2w_final);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
                print_with_timestamp("TTS: sent is_final=true to T2W (llm_finish with no data)\n");
            }
            
            ctx_omni->speek_done = true;
            ctx_omni->warmup_done = true;  // 第一轮对话结束，后续 prefill 需要等待
            speek_cv.notify_all();
            print_with_timestamp("TTS: finished processing all chunks (llm_finish with no data path)\n");
            // 非双工模式或真正的轮次结束：完全重置
            tts_finish = false;
            llm_finish = false;
            chunk_idx = 0;
            tts_n_past = 0;
            audio_tokens.clear();
            // 清理 TTS 跨 chunk 累积状态（避免下一轮串扰）
            ctx_omni->tts_n_past_accumulated = 0;
            ctx_omni->tts_all_generated_tokens.clear();
            ctx_omni->tts_condition_saved = false;
            continue;
        }
        
        // 2. 根据Python实现，将LLM输出转换为TTS条件（hidden_text_merge模式）
        // Python流程：
        //   1. 使用TTS的emb_text层（152064词表）处理LLM token: llm_embeds = self.tts.emb_text(yield_chunk_token_ids)
        //   2. 使用projector_semantic将LLM hidden states（4096维）投影到TTS hidden_dim（768维）:
        //      hidden_embeds = self.tts.projector_semantic(output.last_hidden_states)
        //   3. 归一化: hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        //   4. 合并: tts_embeds = llm_embeds + hidden_embeds
        //   5. 添加audio_bos_token的embedding
        //   6. 将tts_embeds作为condition传递给TTS（使用batch.embd而不是token IDs）
        fflush(stdout);
        
        // 打印 current_chunk_n_embd 的值，确认数据是否正确传递
        print_with_timestamp("TTS: DEBUG before has_llm_data - token_ids.size=%zu, hidden_states.size=%zu, n_embd=%d\n",
                            current_chunk_token_ids.size(), current_chunk_hidden_states.size(), current_chunk_n_embd);
        
        // 使用从队列中获取的token IDs和hidden states
        bool has_llm_data = (!current_chunk_token_ids.empty() && !current_chunk_hidden_states.empty() && current_chunk_n_embd > 0);
        
        // 之前这里有一段"等待 50ms 并累积更多消息"的逻辑
        // 这会导致多个 chunk 被合并，破坏了"每次只处理一个 chunk"的逻辑
        // Python 不会累积多个 chunk，而是每个 chunk 独立处理
        // llm_finish 标志会在下一次从队列取数据时正确捕获
        
        // 打印关键数据状态
        print_with_timestamp("TTS: has_llm_data=%d, token_ids=%zu, hidden_states=%zu, n_embd=%d, llm_finish=%d\n",
                             has_llm_data, current_chunk_token_ids.size(), 
                             current_chunk_hidden_states.size(), current_chunk_n_embd, llm_finish);
        
        
        if (has_llm_data) {
            // 在每轮第一个 chunk 时更新输出目录
            // 确保每轮数据保存在独立的 round_XXX 子目录下
            if (chunk_idx == 0 && !ctx_omni->duplex_mode) {
                update_output_dirs();
                if (write_debug) {
                    create_wav_timing_file();
                }
            }
            
            // Save current chunk_idx to ensure consistent directory naming
            // This is important because chunk_idx will be incremented after processing
            int current_chunk_idx = chunk_idx;
            
            // 防止除零和异常值导致的崩溃
            if (current_chunk_n_embd <= 0 || current_chunk_n_embd > 16384) {
                LOG_ERR("TTS: invalid current_chunk_n_embd=%d, skipping chunk %d\n", 
                        current_chunk_n_embd, current_chunk_idx);
                continue;
            }
            
            // 验证 hidden_states 大小合理性
            if (current_chunk_hidden_states.size() > 100000000) {  // 100M floats = ~400MB，作为上限
                LOG_ERR("TTS: hidden_states size too large (%zu), possible corruption, skipping chunk %d\n", 
                        current_chunk_hidden_states.size(), current_chunk_idx);
                continue;
            }
            
            // 验证 token_ids 和 hidden_states 的对齐关系
            size_t expected_hidden_size = current_chunk_token_ids.size() * current_chunk_n_embd;
            if (current_chunk_hidden_states.size() != expected_hidden_size) {
                LOG_ERR("TTS: hidden_states size mismatch: got %zu, expected %zu (tokens=%zu * n_embd=%d), skipping chunk %d\n",
                        current_chunk_hidden_states.size(), expected_hidden_size, 
                        current_chunk_token_ids.size(), current_chunk_n_embd, current_chunk_idx);
                continue;
            }
            
            // 先过滤特殊 token 再计算 merged_embeddings（与 Python 行为一致）
            // 调试文件保存原始数据，merged_embeddings 使用过滤后的数据
            int n_tokens_orig = (int)(current_chunk_hidden_states.size() / current_chunk_n_embd);
            
            std::vector<llama_token> filtered_token_ids = current_chunk_token_ids;
            std::vector<float> filtered_hidden_states = current_chunk_hidden_states;
            filter_special_tokens(filtered_token_ids, filtered_hidden_states, current_chunk_n_embd);
            
            int n_tokens_filtered = (int)(filtered_hidden_states.size() / current_chunk_n_embd);
            
            // 打印过滤前后的 token 数量
            print_with_timestamp("TTS: n_tokens_orig=%d, n_tokens_filtered=%d (filtered %d special tokens)\n",
                                 n_tokens_orig, n_tokens_filtered, n_tokens_orig - n_tokens_filtered);
            
            // 注意：C++ 的 hidden state 收集方式和 Python 不同：
            // - Python: forward(T_{i-1}) → 收集 H_{i-1} → 采样 T_i（需要延迟调整）
            // - C++: 采样 T_i → forward(T_i) → 收集 H_i（不需要延迟！）
            // 所以 C++ 的 T_i 直接对应 H_i = forward(T_i)，对齐已经正确
            
            
            // 如果所有 token 都被过滤掉了，跳过处理
            if (n_tokens_filtered <= 0) {
                LOG_WRN("TTS: all tokens filtered out, skipping chunk %d\n", current_chunk_idx);
                continue;  // 跳过这个 chunk
            }
            
            // Compute merged embeddings (llm_embeds + projected_hidden) for saving
            // This matches Python implementation: tts_embeds = llm_embeds + hidden_embeds
            // IMPORTANT: Use filtered data to match Python's behavior
            const int tts_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
            std::vector<float> merged_embeddings;  // 合并后的embedding
            bool merged_success = false;
            
            // is_final_text_chunk 需要在外层声明，供后面调用使用（与 Python 对齐）
            // 当 llm_finish=true 时，这是最后一个 text chunk，设置 is_final_text_chunk=true
            // 这样 TTS 会 flush 剩余的 tts_token_buffer，并正确处理 text_eos_embed
            bool is_final_text_chunk = llm_finish;
            
            if (ctx_omni->emb_text_weight && ctx_omni->projector.initialized) {
                if (!ctx_omni->tts_condition_graph.initialized) {
                    tts_condition_graph_init(ctx_omni);
                }

                int graph_tts_n_embd = 0;
                if (ctx_omni->tts_condition_graph.initialized &&
                    tts_condition_graph_forward(ctx_omni,
                                                filtered_token_ids.data(),
                                                filtered_hidden_states.data(),
                                                n_tokens_filtered,
                                                current_chunk_n_embd,
                                                merged_embeddings,
                                                graph_tts_n_embd)) {
                    if (graph_tts_n_embd == tts_n_embd) {
                        merged_success = true;
                    } else {
                        LOG_ERR("TTS: condition graph tts_n_embd mismatch: got %d, expected %d\n",
                                graph_tts_n_embd, tts_n_embd);
                        merged_embeddings.clear();
                    }
                } else {
                    print_with_timestamp("TTS: WARNING - condition graph failed, skipping merged embedding computation\n");
                }
            } else {
                print_with_timestamp("TTS: WARNING - TTS condition graph weights not loaded, skipping merged embedding computation\n");
            }
            
            // Save LLM debug data: text, token_ids, hidden_states, and merged embeddings
            if (write_debug) {
                // Create chunk-specific directory
                std::string chunk_dir = llm_debug_output_dir + "/chunk_" + std::to_string(current_chunk_idx);
                create_dir(chunk_dir);
                
                // 1. Save LLM text output
                std::string text_file = chunk_dir + "/llm_text.txt";
                FILE *f_text = fopen(text_file.c_str(), "w");
                if (f_text) {
                    fprintf(f_text, "%s", response.c_str());
                    fclose(f_text);
                }
                
                // 2. Save LLM token IDs (original, before filtering)
                std::string token_ids_file = chunk_dir + "/llm_token_ids.txt";
                FILE *f_tokens = fopen(token_ids_file.c_str(), "w");
                if (f_tokens) {
                    for (size_t i = 0; i < current_chunk_token_ids.size(); ++i) {
                        fprintf(f_tokens, "%d", current_chunk_token_ids[i]);
                        if (i < current_chunk_token_ids.size() - 1) fprintf(f_tokens, " ");
                    }
                    fprintf(f_tokens, "\n");
                    fclose(f_tokens);
                }
                
                // 3. Save LLM hidden states (binary format, original before filtering)
                std::string hidden_file = chunk_dir + "/llm_hidden_states.bin";
                FILE *f_hidden = fopen(hidden_file.c_str(), "wb");
                if (f_hidden) {
                    // Write header: n_tokens, n_embd
                    int32_t header[2] = {n_tokens_orig, current_chunk_n_embd};
                    fwrite(header, sizeof(int32_t), 2, f_hidden);
                    // Write hidden states data
                    fwrite(current_chunk_hidden_states.data(), sizeof(float), current_chunk_hidden_states.size(), f_hidden);
                    fclose(f_hidden);
                }
                
                // 4. Save LLM hidden states as text (for easy inspection)
                std::string hidden_txt_file = chunk_dir + "/llm_hidden_states.txt";
                FILE *f_hidden_txt = fopen(hidden_txt_file.c_str(), "w");
                if (f_hidden_txt) {
                    fprintf(f_hidden_txt, "Hidden States (shape: [%d, %d]):\n", n_tokens_orig, current_chunk_n_embd);
                    for (int i = 0; i < n_tokens_orig; ++i) {
                        fprintf(f_hidden_txt, "Token %d: ", i);
                        for (int j = 0; j < current_chunk_n_embd; ++j) {
                            fprintf(f_hidden_txt, "%.6f", current_chunk_hidden_states[i * current_chunk_n_embd + j]);
                            if (j < current_chunk_n_embd - 1) fprintf(f_hidden_txt, " ");
                        }
                        fprintf(f_hidden_txt, "\n");
                    }
                    fclose(f_hidden_txt);
                }
                
                // 5. Save merged embeddings (binary format) if successfully computed
                if (merged_success && !merged_embeddings.empty()) {
                    std::string merged_file = chunk_dir + "/merged_embeddings.bin";
                    FILE *f_merged = fopen(merged_file.c_str(), "wb");
                    if (f_merged) {
                        // Write header: n_tokens, tts_n_embd (using filtered token count)
                        int32_t header[2] = {n_tokens_filtered, tts_n_embd};
                        fwrite(header, sizeof(int32_t), 2, f_merged);
                        // Write merged embeddings data
                        fwrite(merged_embeddings.data(), sizeof(float), merged_embeddings.size(), f_merged);
                        fclose(f_merged);
                    }
                    
                    // 6. Save merged embeddings as text (for easy inspection)
                    std::string merged_txt_file = chunk_dir + "/merged_embeddings.txt";
                    FILE *f_merged_txt = fopen(merged_txt_file.c_str(), "w");
                    if (f_merged_txt) {
                        fprintf(f_merged_txt, "Merged Embeddings (shape: [%d, %d]):\n", n_tokens_filtered, tts_n_embd);
                        fprintf(f_merged_txt, "# Formula: merged_embeds = emb_text(filtered_token_ids) + normalize(projector_semantic(filtered_hidden_states))\n");
                        fprintf(f_merged_txt, "# Note: Special tokens have been filtered before computation (matching Python behavior)\n");
                        for (int i = 0; i < n_tokens_filtered; ++i) {
                            fprintf(f_merged_txt, "Token %d: ", i);
                            for (int j = 0; j < tts_n_embd; ++j) {
                                fprintf(f_merged_txt, "%.6f", merged_embeddings[i * tts_n_embd + j]);
                                if (j < tts_n_embd - 1) fprintf(f_merged_txt, " ");
                            }
                            fprintf(f_merged_txt, "\n");
                        }
                        fclose(f_merged_txt);
                        print_with_timestamp("TTS: saved merged embeddings (text) to %s\n", merged_txt_file.c_str());
                    }
                } else {
                    print_with_timestamp("TTS: skipped saving merged embeddings (computation failed or weights not available)\n");
                }
            }
            
            // Generate audio tokens using local TTS model (if merged_embeddings available)
            // or fall back to TTS server
            std::string wav_file_path;
            bool is_final_chunk = llm_finish;
            bool tts_success = false;
            
            // Use local TTS model if merged_embeddings were successfully computed
            if (merged_success && !merged_embeddings.empty()) {
                std::vector<int32_t> audio_tokens;
                
                // 单工路径：使用 generate_audio_tokens_local_simplex 生成
                bool tts_gen_success = false;
                // 传递 is_final_text_chunk，用于 flush buffer（与 Python 对齐）
                tts_gen_success = generate_audio_tokens_local_simplex(ctx_omni, params, merged_embeddings,
                                                n_tokens_filtered, tts_n_embd, current_chunk_idx,
                                                audio_tokens, write_debug ? tts_wav_output_dir : std::string(), is_final_text_chunk);
                if (tts_gen_success) {
                    tts_success = true;
                    
                    // Save audio tokens for external token2wav processing (backup)
                    // token2wav expects relative token IDs (0-6561)
                    if (write_debug) {
                        std::string tokens_txt_file = tts_wav_output_dir + "/audio_tokens_chunk_" +
                                                      std::to_string(current_chunk_idx) + ".txt";
                        FILE* f_tokens = fopen(tokens_txt_file.c_str(), "w");
                        if (f_tokens) {
                            for (size_t i = 0; i < audio_tokens.size(); ++i) {
                                fprintf(f_tokens, "%d", audio_tokens[i]);
                                if (i < audio_tokens.size() - 1) fprintf(f_tokens, ",");
                            }
                            fprintf(f_tokens, "\n");
                            fclose(f_tokens);
                        }
                    }
                    
                    // Accumulate all audio tokens for final WAV generation
                    all_audio_tokens.insert(all_audio_tokens.end(), 
                                           audio_tokens.begin(), audio_tokens.end());
                    
                    // tokens 已在 generate_audio_tokens_local 内部流式推送（与 Python 流式双工对齐）
                    // T2W 端的 buffer 会持续累积，并按滑动窗口处理，保留 lookahead
                    // Python 逻辑：buffer 滑动 CHUNK_SIZE (25)，保留 pre_lookahead (3)
                    
                    // Also save audio tokens to file for the chunk (for debugging)
                    // This is separate from the token2wav sliding window
                } else {
                    LOG_ERR("TTS Local: failed for chunk %d\n", current_chunk_idx);
                }
            }
            
            // Always increment chunk_idx after processing
            // This ensures each chunk gets its own directory and data is not overwritten
            ++chunk_idx;
            
            // Clear processed text
            llm_text.clear();
            response.clear();
            
            // If this is the final chunk, mark as finished and merge all WAV files
            if (llm_finish) {
                // LLM 完成时，flush 剩余的 tts_token_buffer（与 Python 对齐）
                // 因为 is_final_text_chunk 的判断可能不准确（队列时序问题）
                if (!ctx_omni->tts_token_buffer.empty() && ctx_omni->t2w_thread_info) {
                    print_with_timestamp("TTS: llm_finish=true, flushing remaining %zu tokens from tts_token_buffer\n",
                                        ctx_omni->tts_token_buffer.size());
                    
                    T2WOut *t2w_out = new T2WOut();
                    if (t2w_out) {
                        t2w_out->audio_tokens.assign(
                            ctx_omni->tts_token_buffer.begin(),
                            ctx_omni->tts_token_buffer.end()
                        );
                        t2w_out->is_final = false;  // 还不是最后一个，后面还有 turn_end 的 is_final
                        t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                        
                        {
                            std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                            ctx_omni->t2w_thread_info->queue.push(t2w_out);
                        }
                        ctx_omni->t2w_thread_info->cv.notify_one();
                    }
                    ctx_omni->tts_token_buffer.clear();
                }
                
                tts_finish = true;
                ctx_omni->speek_done = true;
                ctx_omni->warmup_done = true;  // 第一轮对话结束，后续 prefill 需要等待
                speek_cv.notify_all();
                print_with_timestamp("TTS: finished processing all chunks\n");
                
                // Merge all WAV files into a single file
                if (write_debug) {
                    merge_wav_files(tts_wav_output_dir, chunk_idx + 1);
                }
                // Python: end_of_turn = last_id in turn_terminator_token_ids
                
                // 保存当前 round_idx 用于 T2W（递增前的值）
                int current_round_idx = ctx_omni->simplex_round_idx;
                
                // 先递增 round 索引，再发送 is_final
                // 这样 T2W 线程在处理完 is_final 后能立即检测到新的 round_idx
                // 避免竞态条件导致下一轮数据写入旧目录
                if (!ctx_omni->duplex_mode) {
                    ctx_omni->simplex_round_idx++;
                    print_with_timestamp("TTS: 单工模式轮次结束，下一轮 round_idx=%d\n", ctx_omni->simplex_round_idx);
                }
                
                // 发送 is_final=true 到 T2W 队列，通知 T2W 重置 buffer
                // 注意：T2W 端只在双工模式下调用 Token2WavSession::reset()
                // 单工模式下只重置 token_buffer 为静音 tokens，不调用 reset()
                // 不检查 all_audio_tokens.empty()：is_final 必须发送，否则 T2W 不会 flush 最后的 buffer（修复最后一个字没说完）
                if (ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens.clear();  // 空tokens，只是通知final
                    t2w_out->is_final = true;
                    t2w_out->round_idx = current_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                    {
                        std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                    print_with_timestamp("TTS: sent is_final=true to T2W queue (turn end)\n");
                }
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                // 不在这里重置 tts_condition_saved（避免竞态条件）
                // 新一轮的 generate_audio_tokens_local 会在 chunk_idx == 0 时重新设置
                // 如果在这里重置，可能与新一轮的 TTS 处理产生竞态
                // 旧版单工代码也没有在这里重置
                if (ctx_omni->duplex_mode) {
                    ctx_omni->tts_condition_saved = false;
                }
                
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                all_audio_tokens.clear();
                llm_finish = false;
                tts_finish = false;
                // wav_turn_base 的递增已移到 T2W 线程的 is_final 处理中
                // 原因：避免竞态条件，确保 T2W 先处理完当前轮次的所有数据再递增
                // ctx_omni->wav_turn_base += 1000;  // 已移到 T2W 线程
            }
            
            continue;  // Skip the rest of the TTS processing
        } else {
            // Clear processed text and continue
            llm_text.clear();
            response.clear();
            continue;
        }
        
        // OLD TTS PROCESSING CODE BELOW (DISABLED)
        // 以下是被禁用的旧 TTS 处理实现（#if 0 关闭），仅供历史参考
        // All code below is disabled and replaced by TTS service calls above
        #if 0
        // 准备TTS输入：使用合并后的embedding（hidden_text_merge模式）
        const int tts_n_embd = llama_n_embd(llama_get_model(ctx_omni->ctx_tts_llama));
        std::vector<float> tts_condition_embeddings;  // 合并后的embedding
        std::vector<llama_token> input_tokens;  // Fallback: 如果无法生成embedding，使用token IDs
        
        if (has_llm_data) {
            // 步骤1: 使用TTS的emb_text层处理LLM token IDs
            int n_tokens = (int)current_chunk_token_ids.size();
            std::vector<float> llm_embeds(n_tokens * tts_n_embd, 0.0f);
            bool emb_text_success = true;
            
            for (int i = 0; i < n_tokens; i++) {
                llama_token token_id = current_chunk_token_ids[i];
                float * emb = llm_embeds.data() + i * tts_n_embd;
                if (!tts_emb_text(ctx_omni, token_id, emb, tts_n_embd)) {
                    emb_text_success = false;
                    break;
                }
            }
            
            if (emb_text_success) {
                // 步骤2: 使用projector_semantic投影LLM hidden states
                // 验证hidden states的shape
                
                // 打印前几个hidden state的值用于调试
                if (current_chunk_hidden_states.size() >= 5) {
                }
                
                std::vector<float> projected_hidden(n_tokens * tts_n_embd, 0.0f);
                bool projector_success = tts_projector_semantic(ctx_omni,
                                                                 current_chunk_hidden_states.data(),
                                                                 n_tokens,
                                                                 current_chunk_n_embd,
                                                                 projected_hidden.data(),
                                                                 tts_n_embd);
                
                if (projector_success) {
                    // 打印投影后的前几个值
                    if (projected_hidden.size() >= 5) {
                    }
                    
                    // 步骤3: 归一化projected hidden states
                    normalize_l2_per_token(projected_hidden.data(), n_tokens, tts_n_embd);
                    
                    // 打印归一化后的前几个值
                    if (projected_hidden.size() >= 5) {
                    }
                    
                    // 步骤4: 合并: tts_embeds = llm_embeds + hidden_embeds
                    // 🔧 [安全检查] 防止创建过大的 vector 导致崩溃
                    size_t cond_size = (size_t)n_tokens * tts_n_embd;
                    if (n_tokens <= 0 || n_tokens > 10000 || 
                        tts_n_embd <= 0 || tts_n_embd > 10000 ||
                        cond_size > 100000000) {  // 100M elements max
                        LOG_ERR("TTS: invalid condition size: n_tokens=%d, tts_n_embd=%d, cond_size=%zu\n",
                                n_tokens, tts_n_embd, cond_size);
                        break;  // 跳过这个 chunk
                    }
                    tts_condition_embeddings.resize(cond_size);
                    for (size_t i = 0; i < cond_size; i++) {
                        tts_condition_embeddings[i] = llm_embeds[i] + projected_hidden[i];
                    }
                    
                    // 打印合并后的前几个值
                    if (tts_condition_embeddings.size() >= 5) {
                    }
                    
                    // Save LLM debug data: text, hidden states, and merged embeddings
                    if (write_debug) {
                        // Create chunk-specific directory
                        std::string chunk_dir = llm_debug_output_dir + "/chunk_" + std::to_string(chunk_idx);
                        create_dir(chunk_dir);
                        
                        // 1. Save LLM text output
                        std::string text_file = chunk_dir + "/llm_text.txt";
                        FILE *f_text = fopen(text_file.c_str(), "w");
                        if (f_text) {
                            fprintf(f_text, "%s", response.c_str());
                            fclose(f_text);
                        }
                        
                        // 2. Save LLM token IDs
                        std::string token_ids_file = chunk_dir + "/llm_token_ids.txt";
                        FILE *f_tokens = fopen(token_ids_file.c_str(), "w");
                        if (f_tokens) {
                            fprintf(f_tokens, "Token IDs (%zu):\n", current_chunk_token_ids.size());
                            for (size_t i = 0; i < current_chunk_token_ids.size(); ++i) {
                                fprintf(f_tokens, "%d", current_chunk_token_ids[i]);
                                if (i < current_chunk_token_ids.size() - 1) fprintf(f_tokens, " ");
                            }
                            fprintf(f_tokens, "\n");
                            fclose(f_tokens);
                        }
                        
                        // 3. Save LLM hidden states (binary format for precision)
                        std::string hidden_file = chunk_dir + "/llm_hidden_states.bin";
                        FILE *f_hidden = fopen(hidden_file.c_str(), "wb");
                        if (f_hidden) {
                            // Write header: n_tokens, n_embd
                            int32_t header[2] = {n_tokens, current_chunk_n_embd};
                            fwrite(header, sizeof(int32_t), 2, f_hidden);
                            // Write hidden states data
                            fwrite(current_chunk_hidden_states.data(), sizeof(float), current_chunk_hidden_states.size(), f_hidden);
                            fclose(f_hidden);
                        }
                        
                        // 4. Save LLM hidden states as text (for easy inspection)
                        std::string hidden_txt_file = chunk_dir + "/llm_hidden_states.txt";
                        FILE *f_hidden_txt = fopen(hidden_txt_file.c_str(), "w");
                        if (f_hidden_txt) {
                            fprintf(f_hidden_txt, "Hidden States (shape: [%d, %d]):\n", n_tokens, current_chunk_n_embd);
                            for (int i = 0; i < n_tokens; ++i) {
                                fprintf(f_hidden_txt, "Token %d: ", i);
                                for (int j = 0; j < current_chunk_n_embd; ++j) {
                                    fprintf(f_hidden_txt, "%.6f", current_chunk_hidden_states[i * current_chunk_n_embd + j]);
                                    if (j < current_chunk_n_embd - 1) fprintf(f_hidden_txt, " ");
                                }
                                fprintf(f_hidden_txt, "\n");
                            }
                            fclose(f_hidden_txt);
                        }
                        
                        // 5. Save merged embeddings (binary format)
                        std::string merged_file = chunk_dir + "/merged_embeddings.bin";
                        FILE *f_merged = fopen(merged_file.c_str(), "wb");
                        if (f_merged) {
                            // Write header: n_tokens, tts_n_embd
                            int32_t header[2] = {n_tokens, tts_n_embd};
                            fwrite(header, sizeof(int32_t), 2, f_merged);
                            // Write merged embeddings data
                            fwrite(tts_condition_embeddings.data(), sizeof(float), tts_condition_embeddings.size(), f_merged);
                            fclose(f_merged);
                        }
                        
                        // 6. Save merged embeddings as text (for easy inspection)
                        std::string merged_txt_file = chunk_dir + "/merged_embeddings.txt";
                        FILE *f_merged_txt = fopen(merged_txt_file.c_str(), "w");
                        if (f_merged_txt) {
                            fprintf(f_merged_txt, "Merged Embeddings (shape: [%d, %d]):\n", n_tokens, tts_n_embd);
                            for (int i = 0; i < n_tokens; ++i) {
                                fprintf(f_merged_txt, "Token %d: ", i);
                                for (int j = 0; j < tts_n_embd; ++j) {
                                    fprintf(f_merged_txt, "%.6f", tts_condition_embeddings[i * tts_n_embd + j]);
                                    if (j < tts_n_embd - 1) fprintf(f_merged_txt, " ");
                                }
                                fprintf(f_merged_txt, "\n");
                            }
                            fclose(f_merged_txt);
                        }
                        
                        // 7. Save intermediate results: llm_embeds and projected_hidden
                        std::string llm_embeds_file = chunk_dir + "/llm_embeds.txt";
                        FILE *f_llm_embeds = fopen(llm_embeds_file.c_str(), "w");
                        if (f_llm_embeds) {
                            fprintf(f_llm_embeds, "LLM Embeddings from emb_text (shape: [%d, %d]):\n", n_tokens, tts_n_embd);
                            for (int i = 0; i < n_tokens; ++i) {
                                fprintf(f_llm_embeds, "Token %d: ", i);
                                for (int j = 0; j < tts_n_embd; ++j) {
                                    fprintf(f_llm_embeds, "%.6f", llm_embeds[i * tts_n_embd + j]);
                                    if (j < tts_n_embd - 1) fprintf(f_llm_embeds, " ");
                                }
                                fprintf(f_llm_embeds, "\n");
                            }
                            fclose(f_llm_embeds);
                        }
                        
                        std::string projected_file = chunk_dir + "/projected_hidden.txt";
                        FILE *f_projected = fopen(projected_file.c_str(), "w");
                        if (f_projected) {
                            fprintf(f_projected, "Projected Hidden States (shape: [%d, %d], after normalization):\n", n_tokens, tts_n_embd);
                            for (int i = 0; i < n_tokens; ++i) {
                                fprintf(f_projected, "Token %d: ", i);
                                for (int j = 0; j < tts_n_embd; ++j) {
                                    fprintf(f_projected, "%.6f", projected_hidden[i * tts_n_embd + j]);
                                    if (j < tts_n_embd - 1) fprintf(f_projected, " ");
                                }
                                fprintf(f_projected, "\n");
                            }
                            fclose(f_projected);
                        }
                    }
                    
                    // 🔧 [修复] 不在 merge embed 阶段添加 audio_bos
                    // audio_bos 将在 prefill 之前动态添加
                } else {
                    emb_text_success = false;
                }
            }
            
            if (!emb_text_success) {
                // Fallback: 使用token IDs
                input_tokens.insert(input_tokens.end(), current_chunk_token_ids.begin(), current_chunk_token_ids.end());
                input_tokens.push_back(audio_bos_token_id);
            }
        } else {
            // 没有LLM数据，使用text_tokens作为fallback
            input_tokens.insert(input_tokens.end(), text_tokens.begin(), text_tokens.end());
            input_tokens.push_back(audio_bos_token_id);
        }
        
        // 3. Reset TTS KV cache ONLY when starting a completely new round (speek_done was true and we're starting fresh)
        // IMPORTANT: Python's TTSStreamingGenerator maintains past_key_values across chunks:
        //   - First call: past_key_values = None (reset)
        //   - Subsequent chunks: past_key_values = self.past_key_values (continue)
        //   - Only reset on end_of_turn or new round
        // We should NOT clear KV cache for each new chunk from LLM if LLM is still generating
        // Only clear when speek_done was true (indicating a new round)
        fflush(stdout);
        
        // Only reset if we're starting a new round (speek_done was true, meaning previous round finished)
        // AND we have previous state (tts_n_past > 0)
        // This matches Python behavior: reset only on end_of_turn or new round
        static bool last_speek_done = false;
        bool should_reset = (last_speek_done && ctx_omni->speek_done == false && tts_n_past > 0);
        last_speek_done = ctx_omni->speek_done;
        
        if (should_reset) {
            fflush(stdout);
            // Clear KV cache to start fresh for new round
            llama_memory_t mem = llama_get_memory(ctx_omni->ctx_tts_llama);
            if (mem) {
                llama_memory_seq_rm(mem, 0, 0, -1);
            } else {
            }
            fflush(stdout);
            tts_n_past = 0;
        } else {
            fflush(stdout);
        }
        
        // 4. Evaluate input (使用embedding或token IDs)
        // Python: condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
        // Python: condition_length = current_condition.shape[1]
        // Python: pos_ids = torch.arange(self.text_start_pos, self.text_start_pos + condition_length)
        // Python: self.text_start_pos += condition_length + len(chunk_generated_tokens)
        if (!tts_condition_embeddings.empty()) {
            // 🔧 [修复] 在 prefill 之前动态添加 audio_bos embedding
            // Python 中 audio_bos 是在 TTS 类内部（TTSStreamingGenerator.generate_with_buffer）添加的
            // 每个 chunk 都会加 audio_bos: condition = torch.cat([condition, self.audio_bos_embeds], dim=1)
            std::vector<float> audio_bos_embed(tts_n_embd, 0.0f);
            if (tts_emb_text(ctx_omni, audio_bos_token_id, audio_bos_embed.data(), tts_n_embd)) {
                // Append audio_bos to tts_condition_embeddings
                tts_condition_embeddings.insert(tts_condition_embeddings.end(), 
                                                audio_bos_embed.begin(), audio_bos_embed.end());
                print_with_timestamp("TTS: 在 prefill 前添加 audio_bos (chunk_idx=%d, new_size=%zu)\n", 
                                    chunk_idx, tts_condition_embeddings.size() / tts_n_embd);
            } else {
                LOG_ERR("TTS: failed to get audio_bos embedding for prefill\n");
            }
            
            // 使用合并后的embedding作为输入（正确的实现方式）
            int condition_length = (int)(tts_condition_embeddings.size() / tts_n_embd);
            fflush(stdout);
            
            // 使用prefill_emb_with_hidden类似的逻辑，但针对TTS模型
            // Python: prefill时，position_ids从text_start_pos开始，到text_start_pos + condition_length
            // C++: batch.pos[j] = tts_n_past + j，其中tts_n_past对应text_start_pos
            int n_pos = condition_length;
            int text_start_pos_before = tts_n_past;  // 保存prefill前的text_start_pos
            if (!prefill_with_emb_tts(ctx_omni, params, tts_condition_embeddings.data(), n_pos, params->n_batch, &tts_n_past)) {
                LOG_ERR("Failed to evaluate TTS input embeddings\n");
                // Clear the processed text to prevent infinite loop
                llm_text.clear();
                response.clear();
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                // 清理 TTS 跨 chunk 累积状态（避免下一轮串扰）
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                fflush(stdout);
                continue;
            }
                int condition_length_processed = tts_n_past - text_start_pos_before;
        } else {
            // Fallback: 使用token IDs作为输入
            fflush(stdout);
            fflush(stdout);
            if (!eval_tokens_tts(ctx_omni, params, input_tokens, params->n_batch, &tts_n_past)) {
                LOG_ERR("Failed to evaluate TTS input tokens\n");
                // Clear the processed text to prevent infinite loop
                llm_text.clear();
                response.clear();
                chunk_idx = 0;
                tts_n_past = 0;
                audio_tokens.clear();
                // 清理 TTS 跨 chunk 累积状态（避免下一轮串扰）
                ctx_omni->tts_n_past_accumulated = 0;
                ctx_omni->tts_all_generated_tokens.clear();
                ctx_omni->tts_condition_saved = false;
                fflush(stdout);
                continue;
            }
        }
        
        // 4. Generate audio tokens using TTS model
        // Following Python flow: decode 1 token at a time, accumulate to buffer, yield when buffer reaches 25 tokens
        audio_tokens.clear();
        bool audio_gen_finish = false;
        int audio_token_count = 0;
        
        // Limit audio tokens per chunk: 25 tokens per chunk (each chunk corresponds to ~10 LLM tokens)
        // 每一轮的限制：LLM生成10个token → TTS生成25个token
        // 接下来LLM再生成10个token → TTS再生成25个token，以此类推
        int text_token_count = (int)text_tokens.size();
        // Fixed at 25 tokens per chunk (matching Python's audio_token_chunk_size=25)
        int max_audio_tokens_for_text = 25;
        
        // Buffer for accumulating audio tokens (yield when reaching 25 tokens, matching Python chunk_size=25)
        const int audio_token_chunk_size = 25;  // Yield size matching Python
        std::vector<llama_token> audio_token_buffer;
        
        // 🔧 [差异3修复] 当前 chunk 的 tokens，用于 repetition penalty
        std::vector<llama_token> chunk_generated_tokens_tts;
        
        // Generate audio tokens one by one (matching Python decode flow)
        while (audio_token_count < max_audio_tokens_for_text && !audio_gen_finish && !ctx_omni->speek_done) {
            fflush(stdout);
            
            // Decode 1 audio token at a time (matching Python: each decode generates 1 token)
            // 🔧 [差异2&3修复] 传入：
            // - all_audio_tokens: 用于判断 is_first_token_overall
            // - chunk_generated_tokens_tts: 用于 repetition penalty（只用当前 chunk 的 tokens）
            // - audio_token_count: token_index_in_chunk，用于判断 is_chunk_first_token
            // - force_no_eos: false（这个路径已有 25 tokens 的限制，不需要额外阻止 EOS）
            llama_token audio_token = sample_tts_token_simplex(ctx_omni->ctx_tts_sampler, ctx_omni, params, &tts_n_past, &all_audio_tokens, &chunk_generated_tokens_tts, audio_token_count, false);
            fflush(stdout);
            
            audio_token_buffer.push_back(audio_token);
            audio_tokens.push_back(audio_token);
            all_audio_tokens.push_back(audio_token);
            chunk_generated_tokens_tts.push_back(audio_token);  // 🔧 [差异3修复]
            audio_token_count++;
            
            // Check for end of audio generation
            bool is_audio = is_audio_token(audio_token, audio_bos_token_id, num_audio_tokens);
            int relative_idx = audio_token - audio_bos_token_id;
            fflush(stdout);
            
            // Check for audio EOS token (relative index 6561, absolute 158248)
            // Note: EOS token should only end audio generation for current chunk, not set speek_done
            // speek_done should only be set when llm_finish is true
            if (audio_token == audio_eos_token_id || relative_idx == 6561) {
                fflush(stdout);
                audio_gen_finish = true;
                // Only set tts_finish if LLM has also finished, otherwise continue to next chunk
                if (llm_finish) {
                    tts_finish = true;
                }
                break;
            }
            
            // Yield when buffer reaches chunk_size (25 tokens), matching Python yield logic
            if (audio_token_buffer.size() >= audio_token_chunk_size) {
                fflush(stdout);
                
                // Push audio tokens to T2W queue for token2wav processing
                if (ctx_omni->t2w_thread_info) {
                    T2WOut *t2w_out = new T2WOut();
                    t2w_out->audio_tokens = audio_token_buffer;  // Copy the 25 tokens
                    t2w_out->is_final = false;  // Not final, more chunks may come
                    t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                    
                    {
                        std::unique_lock<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                        // Check if queue has space
                        while (ctx_omni->t2w_thread_info->queue.size() >= ctx_omni->t2w_thread_info->MAX_QUEUE_SIZE) {
                            // Wait for space in queue (with timeout to avoid deadlock)
                            lock.unlock();
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            lock.lock();
                        }
                        ctx_omni->t2w_thread_info->queue.push(t2w_out);
                    }
                    ctx_omni->t2w_thread_info->cv.notify_one();
                }
                
                // Yield this chunk (create unit_buffer and save)
                // Note: In Python, each yield creates a chunk, but here we create one buffer per 25-token chunk
                struct unit_buffer *res = new unit_buffer();
                res->text = response;
                res->unit_n_past = ctx_omni->n_past;
                res->completed = false;  // Not completed yet, more tokens may come
                
                // TODO: Convert audio tokens to audio waveform
                // For now, create a placeholder buffer
                res->buffer.resize(audio_token_chunk_size * 100);  // Placeholder size
                
                if (!res->buffer.empty()) {
                    // 读写使用 mutex 保护
                    std::unique_lock<std::mutex> lock(buffer_mutex);
                    ctx_omni->omni_output->output.push_back(res);
                    ctx_omni->omni_output->idx += 1;
                } else {
                    delete res;
                }
                
                // Clear buffer after yield, matching Python behavior
                audio_token_buffer.clear();
            }
        }
        
        // If we exited the loop because we reached max_audio_tokens_for_text (25 tokens per chunk),
        // mark audio_gen_finish as true to indicate this chunk is complete
        // This allows the next iteration to process the next chunk from LLM (if LLM is still generating)
        if (!audio_gen_finish && audio_token_count >= max_audio_tokens_for_text && !ctx_omni->speek_done) {
            fflush(stdout);
            audio_gen_finish = true;
        }
        fflush(stdout);
        
        // Save audio tokens to file for debugging/analysis
        // Note: Save relative indices (0-6561) for token2wav compatibility
        if (write_debug && !audio_tokens.empty()) {
            // Always save to fixed output directory
            const int audio_bos_token_id = 151687;
            std::string token_file = tts_output_dir + "/tts_audio_tokens_chunk_" + std::to_string(chunk_idx) + ".txt";
            FILE *f = fopen(token_file.c_str(), "w");
            if (f) {
                fprintf(f, "Text: %s\n", response.c_str());
                fprintf(f, "Audio tokens (%zu) [relative_index]:\n", audio_tokens.size());
                for (size_t i = 0; i < audio_tokens.size(); ++i) {
                    int absolute_id = audio_tokens[i];
                    int relative_idx = absolute_id - audio_bos_token_id;
                    // Verify token is in valid range
                    if (absolute_id < audio_bos_token_id || relative_idx >= 6562) {
                        LOG_ERR("TTS: WARNING - token %d (relative_idx=%d) is outside valid range [%d, %d)\n", 
                                absolute_id, relative_idx, audio_bos_token_id, audio_bos_token_id + 6562);
                        // Still save relative index, but mark as invalid
                        relative_idx = -1;  // Mark as invalid
                    }
                    // Save relative index only (for token2wav compatibility)
                    fprintf(f, "%d", relative_idx);
                    if (i < audio_tokens.size() - 1) fprintf(f, ",");
                }
                fprintf(f, "\n");
                fclose(f);
            } else {
                LOG_ERR("Failed to open file for writing: %s\n", token_file.c_str());
            }
        }
        
        // Create final unit_buffer for remaining tokens in buffer (if any)
        if (!audio_token_buffer.empty()) {
            // Push remaining tokens to T2W queue (final chunk)
            if (ctx_omni->t2w_thread_info) {
                T2WOut *t2w_out = new T2WOut();
                t2w_out->audio_tokens = audio_token_buffer;  // Copy remaining tokens
                t2w_out->is_final = (audio_gen_finish || llm_finish);  // Mark as final if generation finished
                t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                
                {
                    std::unique_lock<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    // Check if queue has space
                    while (ctx_omni->t2w_thread_info->queue.size() >= ctx_omni->t2w_thread_info->MAX_QUEUE_SIZE) {
                        lock.unlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        lock.lock();
                    }
                    ctx_omni->t2w_thread_info->queue.push(t2w_out);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
            }
            
            struct unit_buffer *res = new unit_buffer();
            res->text = response;
            res->unit_n_past = ctx_omni->n_past;
            res->completed = (audio_gen_finish || llm_finish);
            
            // TODO: Convert audio tokens to audio waveform
            res->buffer.resize(audio_token_buffer.size() * 100);  // Placeholder size
            
            if (!res->buffer.empty()) {
                // 读写使用 mutex 保护
                std::unique_lock<std::mutex> lock(buffer_mutex);
                ctx_omni->omni_output->output.push_back(res);
                ctx_omni->omni_output->idx += 1;
            } else {
                delete res;
            }
        }
        
        // Clear processed text after generating audio tokens for this chunk
        // This ensures we wait for new text in the next iteration instead of reprocessing the same text
        // Python: self.text_start_pos += condition_length + len(chunk_generated_tokens)
        // C++: tts_n_past is already updated during prefill (condition_length) and decode (len(chunk_generated_tokens))
        // So tts_n_past already equals text_start_pos after this chunk
        int condition_length_final = (int)(tts_condition_embeddings.empty() ? input_tokens.size() : (tts_condition_embeddings.size() / tts_n_embd));
        int chunk_generated_tokens_count = (int)audio_tokens.size();
        fflush(stdout);
        
        llm_text.clear();
        response.clear();
        ++chunk_idx;
        
        if (ctx_omni->speek_done) {
            // 可能是外部强行打断了，清除内部所有状态
            printf("speek done setted by outside, clear all internal states\n");
            tts_finish = true;
            // Reset TTS model state by clearing KV cache
            // Note: This might need llama_kv_cache_clear or similar function
            tts_n_past = 0;
        }
        
        // Only finish and set speek_done when BOTH LLM and TTS have finished
        // If only audio_gen_finish is true but llm_finish is false, continue to next chunk
        if (llm_finish && (tts_finish || audio_gen_finish)) {
            // Save all collected audio tokens to a summary file
            if (write_debug && !all_audio_tokens.empty()) {
                // Always save to fixed output directory
                // Note: Save relative indices (0-6561) for token2wav compatibility
                const int audio_bos_token_id = 151687;
                std::string summary_file = tts_output_dir + "/tts_all_audio_tokens_summary.txt";
                FILE *f = fopen(summary_file.c_str(), "w");
                if (f) {
                    fprintf(f, "Total audio tokens: %zu\n", all_audio_tokens.size());
                    fprintf(f, "Audio token range: relative [0, %d) (absolute [%d, %d))\n", 
                            6562, audio_bos_token_id, audio_bos_token_id + 6562);
                    fprintf(f, "Audio tokens [relative_index]:\n");
                    int invalid_count = 0;
                    for (size_t i = 0; i < all_audio_tokens.size(); ++i) {
                        int absolute_id = all_audio_tokens[i];
                        int relative_idx = absolute_id - audio_bos_token_id;
                        // Verify token is in valid range
                        if (absolute_id < audio_bos_token_id || relative_idx >= 6562 || relative_idx < 0) {
                            LOG_ERR("TTS: WARNING - token %d (relative_idx=%d) is outside valid range [%d, %d)\n", 
                                    absolute_id, relative_idx, audio_bos_token_id, audio_bos_token_id + 6562);
                            relative_idx = -1;  // Mark as invalid
                            invalid_count++;
                        }
                        // Save relative index only (for token2wav compatibility)
                        fprintf(f, "%d", relative_idx);
                        if (i < all_audio_tokens.size() - 1) {
                            if ((i + 1) % 20 == 0) fprintf(f, "\n");
                            else fprintf(f, " ");
                        }
                    }
                    fprintf(f, "\n");
                    if (invalid_count > 0) {
                        fprintf(f, "WARNING: Found %d tokens outside valid range [%d, %d)\n", 
                                invalid_count, audio_bos_token_id, audio_bos_token_id + 6562);
                    }
                    fclose(f);
                    fflush(stdout);
                } else {
                    LOG_ERR("Failed to open summary file for writing: %s\n", summary_file.c_str());
                }
            }
            
            // 🔧 [与 Python 流式双工对齐] 发送 is_final=true 到 T2W 队列，flush 剩余 buffer
            // Python: is_last_chunk=True 时会 flush 所有剩余 tokens
            // T2W 端 buffer 已经累积了所有 tokens，这里只是通知它 flush
            if (ctx_omni->t2w_thread_info && !all_audio_tokens.empty()) {
                T2WOut *t2w_out = new T2WOut();
                t2w_out->audio_tokens.clear();  // 空tokens，只是通知final
                t2w_out->is_final = true;
                t2w_out->round_idx = ctx_omni->simplex_round_idx;  // 轮次索引（TTS→T2W 输出目录归属）
                
                {
                    std::lock_guard<std::mutex> lock(ctx_omni->t2w_thread_info->mtx);
                    ctx_omni->t2w_thread_info->queue.push(t2w_out);
                }
                ctx_omni->t2w_thread_info->cv.notify_one();
            }
            
            // 旧代码（已禁用，改用T2W线程处理）
            /*
            if (ctx_omni->token2wav_initialized && ctx_omni->token2wav_session && 
                !ctx_omni->token2wav_buffer.empty()) {
                // ... 旧的同步处理代码 ...
            }
            */
            
            // 兼容旧代码的变量定义（避免编译错误）
            bool _unused_final_wav_placeholder = false;
            if (_unused_final_wav_placeholder && ctx_omni->token2wav_initialized && ctx_omni->token2wav_session && 
                !ctx_omni->token2wav_buffer.empty()) {
                std::vector<float> final_wav;
                if (ctx_omni->token2wav_session->feed_window(ctx_omni->token2wav_buffer, true, final_wav)) {
                    if (!final_wav.empty()) {
                        const int sample_rate = omni::flow::Token2Wav::kSampleRate;
                        std::string wav_path = tts_wav_output_dir + "/wav_" + 
                                               std::to_string(ctx_omni->token2wav_wav_idx) + ".wav";
                        
                        // Convert float to int16 and write WAV
                        const int16_t num_channels = 1;
                        const int16_t bits_per_sample = 16;
                        const int16_t block_align = num_channels * (bits_per_sample / 8);
                        const int32_t byte_rate = sample_rate * block_align;
                        
                        std::vector<int16_t> pcm(final_wav.size());
                        for (size_t i = 0; i < final_wav.size(); ++i) {
                            float x = final_wav[i];
                            if (!std::isfinite(x)) x = 0.0f;
                            x = std::max(-1.0f, std::min(1.0f, x));
                            float y = x * 32767.0f;
                            if (y >= 32767.0f) pcm[i] = 32767;
                            else if (y <= -32768.0f) pcm[i] = -32768;
                            else pcm[i] = (int16_t)y;
                        }
                        
                        uint32_t data_bytes = (uint32_t)(pcm.size() * sizeof(int16_t));
                        uint32_t riff_size = 36u + data_bytes;
                        
                        FILE* f_wav = fopen(wav_path.c_str(), "wb");
                        if (f_wav) {
                            fwrite("RIFF", 1, 4, f_wav);
                            fwrite(&riff_size, 4, 1, f_wav);
                            fwrite("WAVE", 1, 4, f_wav);
                            fwrite("fmt ", 1, 4, f_wav);
                            uint32_t fmt_size = 16;
                            uint16_t audio_format = 1;
                            fwrite(&fmt_size, 4, 1, f_wav);
                            fwrite(&audio_format, 2, 1, f_wav);
                            fwrite(&num_channels, 2, 1, f_wav);
                            fwrite(&sample_rate, 4, 1, f_wav);
                            fwrite(&byte_rate, 4, 1, f_wav);
                            fwrite(&block_align, 2, 1, f_wav);
                            fwrite(&bits_per_sample, 2, 1, f_wav);
                            fwrite("data", 1, 4, f_wav);
                            fwrite(&data_bytes, 4, 1, f_wav);
                            fwrite(pcm.data(), 1, data_bytes, f_wav);
                            fclose(f_wav);
                        }
                        ctx_omni->token2wav_wav_idx++;
                    }
                }
                
                // Reset buffer for next round (re-initialize with 3 silence tokens)
                ctx_omni->token2wav_buffer.clear();
                ctx_omni->token2wav_buffer = {4218, 4218, 4218};
                ctx_omni->token2wav_wav_idx = 0;
            }
            
            tts_finish = false;
            chunk_idx = 0;
            llm_finish = false;
            tts_n_past = 0;  // Reset TTS model state
            audio_tokens.clear();
            all_audio_tokens.clear();  // Clear for next round
            printf("\ntts finished\n");

            ctx_omni->speek_done = true;
            ctx_omni->warmup_done = true;  // 第一轮对话结束，后续 prefill 需要等待
            speek_cv.notify_all();
        } else if (audio_gen_finish && !llm_finish) {
            // Current chunk finished (either reached 25 tokens or EOS token) but LLM is still generating
            // Reset audio_gen_finish to allow processing next chunk from LLM
            // This ensures: LLM生成10个token → TTS生成25个token → LLM再生成10个token → TTS再生成25个token
            fflush(stdout);
            audio_gen_finish = false;  // Reset for next chunk
            // Note: llm_text and response are already cleared above, so we'll wait for next chunk in the next iteration
        }
        #endif  // End of disabled old TTS processing code
    }
}
