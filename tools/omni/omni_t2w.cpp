// ==========================================================================
#include "omni_internal.h"
#include "token2wav/token2wav-impl.h"

#include "common/common.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
    #include <signal.h>
    #include <sys/wait.h>
    #include <unistd.h>
#else
    #include <windows.h>
    #include <io.h>
    #include <process.h>
#endif

// Token2Wav service and worker implementation
// ==========================================================================

// ======================= Python Token2Wav 服务管理函数 =======================

// 启动 Python Token2Wav 服务进程
bool start_python_t2w_service(struct omni_context * ctx_omni) {
    if (ctx_omni->python_t2w_initialized) {
        return true;  // 已经启动
    }
    
    std::string script_path = ctx_omni->python_t2w_script_dir + "/token2wav_service.py";
    
    FILE* check = fopen(script_path.c_str(), "r");
    if (!check) {
        LOG_ERR("Python T2W: 脚本不存在: %s\n", script_path.c_str());
        return false;
    }
    fclose(check);
    
    print_with_timestamp("Python T2W: 启动服务进程 %s\n", script_path.c_str());
    
#ifdef _WIN32
    // Windows: use CreateProcess with pipes for bidirectional communication
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;
    
    HANDLE hStdinRead, hStdinWrite, hStdoutRead, hStdoutWrite;
    
    if (!CreatePipe(&hStdinRead, &hStdinWrite, &sa, 0)) {
        LOG_ERR("Python T2W: CreatePipe (stdin) 失败\n");
        return false;
    }
    SetHandleInformation(hStdinWrite, HANDLE_FLAG_INHERIT, 0);
    
    if (!CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, 0)) {
        LOG_ERR("Python T2W: CreatePipe (stdout) 失败\n");
        CloseHandle(hStdinRead);
        CloseHandle(hStdinWrite);
        return false;
    }
    SetHandleInformation(hStdoutRead, HANDLE_FLAG_INHERIT, 0);
    
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.hStdInput = hStdinRead;
    si.hStdOutput = hStdoutWrite;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    si.dwFlags |= STARTF_USESTDHANDLES;
    
    ZeroMemory(&pi, sizeof(pi));
    
    if (!ctx_omni->python_t2w_gpu_id.empty()) {
        _putenv_s("CUDA_VISIBLE_DEVICES", ctx_omni->python_t2w_gpu_id.c_str());
    }
    
    std::string win_cmd = "python \"" + script_path + "\"";
    char cmd_buf[2048];
    strncpy(cmd_buf, win_cmd.c_str(), sizeof(cmd_buf) - 1);
    cmd_buf[sizeof(cmd_buf) - 1] = '\0';
    
    if (!CreateProcessA(NULL, cmd_buf, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        LOG_ERR("Python T2W: CreateProcess 失败, error=%lu\n", GetLastError());
        CloseHandle(hStdinRead);
        CloseHandle(hStdinWrite);
        CloseHandle(hStdoutRead);
        CloseHandle(hStdoutWrite);
        return false;
    }
    
    CloseHandle(hStdinRead);
    CloseHandle(hStdoutWrite);
    CloseHandle(pi.hThread);
    
    ctx_omni->python_t2w_pid = (int)(intptr_t)pi.hProcess;
    
    int stdin_fd = _open_osfhandle((intptr_t)hStdinWrite, 0);
    int stdout_fd = _open_osfhandle((intptr_t)hStdoutRead, 0);
    
    if (stdin_fd < 0 || stdout_fd < 0) {
        LOG_ERR("Python T2W: _open_osfhandle 失败\n");
        TerminateProcess(pi.hProcess, 1);
        CloseHandle(pi.hProcess);
        return false;
    }
    
    ctx_omni->python_t2w_stdin = _fdopen(stdin_fd, "w");
    ctx_omni->python_t2w_stdout = _fdopen(stdout_fd, "r");
    
#else
    // POSIX implementation using fork/pipe
    int stdin_pipe[2];
    int stdout_pipe[2];
    
    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
        LOG_ERR("Python T2W: 创建管道失败\n");
        return false;
    }
    
    pid_t pid = fork();
    if (pid < 0) {
        LOG_ERR("Python T2W: fork 失败\n");
        return false;
    }
    
    if (pid == 0) {
        // 子进程
        close(stdin_pipe[1]);   // 关闭写端
        close(stdout_pipe[0]);  // 关闭读端
        
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        
        // 设置 CUDA_VISIBLE_DEVICES 环境变量（必须在 import torch 之前）
        // 这样 Python 子进程只能看到指定的 GPU
        if (!ctx_omni->python_t2w_gpu_id.empty()) {
            setenv("CUDA_VISIBLE_DEVICES", ctx_omni->python_t2w_gpu_id.c_str(), 1);
        }
        
        // 执行 Python 脚本 (使用 conda Python)
        execlp("python3", "python3", script_path.c_str(), (char*)NULL);
        
        // 如果 execlp 失败
        _exit(1);
    }
    
    // 父进程
    close(stdin_pipe[0]);   // 关闭读端
    close(stdout_pipe[1]);  // 关闭写端
    
    ctx_omni->python_t2w_pid = pid;
    ctx_omni->python_t2w_stdin = fdopen(stdin_pipe[1], "w");
    ctx_omni->python_t2w_stdout = fdopen(stdout_pipe[0], "r");
#endif
    
    if (!ctx_omni->python_t2w_stdin || !ctx_omni->python_t2w_stdout) {
        LOG_ERR("Python T2W: fdopen 失败\n");
        stop_python_t2w_service(ctx_omni);
        return false;
    }
    
    // 设置为行缓冲
    setvbuf(ctx_omni->python_t2w_stdin, NULL, _IOLBF, 0);
    setvbuf(ctx_omni->python_t2w_stdout, NULL, _IOLBF, 0);
    
    // 等待服务就绪
    char buffer[4096];
    if (fgets(buffer, sizeof(buffer), ctx_omni->python_t2w_stdout)) {
        print_with_timestamp("Python T2W: 服务响应: %s", buffer);
        if (strstr(buffer, "\"status\":\"ready\"") || strstr(buffer, "\"status\": \"ready\"")) {
            ctx_omni->python_t2w_initialized = true;
            print_with_timestamp("Python T2W: 服务就绪\n");
            return true;
        }
    }
    
    LOG_ERR("Python T2W: 服务未能正常启动\n");
    stop_python_t2w_service(ctx_omni);
    return false;
}

// 停止 Python Token2Wav 服务进程
void stop_python_t2w_service(struct omni_context * ctx_omni) {
    if (ctx_omni->python_t2w_stdin) {
        fprintf(ctx_omni->python_t2w_stdin, "{\"cmd\":\"quit\"}\n");
        fflush(ctx_omni->python_t2w_stdin);
        fclose(ctx_omni->python_t2w_stdin);
        ctx_omni->python_t2w_stdin = nullptr;
    }
    
    if (ctx_omni->python_t2w_stdout) {
        fclose(ctx_omni->python_t2w_stdout);
        ctx_omni->python_t2w_stdout = nullptr;
    }
    
    if (ctx_omni->python_t2w_pid > 0) {
#ifdef _WIN32
        HANDLE hProcess = (HANDLE)(intptr_t)ctx_omni->python_t2w_pid;
        // Wait briefly for process to exit gracefully
        if (WaitForSingleObject(hProcess, 500) == WAIT_TIMEOUT) {
            // Force terminate if still running
            TerminateProcess(hProcess, 1);
            WaitForSingleObject(hProcess, 1000);
        }
        CloseHandle(hProcess);
#else
        int status;
        waitpid(ctx_omni->python_t2w_pid, &status, WNOHANG);
        
        // 如果还没退出，发送 SIGTERM
        if (kill(ctx_omni->python_t2w_pid, 0) == 0) {
            kill(ctx_omni->python_t2w_pid, SIGTERM);
            usleep(100000);  // 等待 100ms
            
            // 如果还没退出，发送 SIGKILL
            if (kill(ctx_omni->python_t2w_pid, 0) == 0) {
                kill(ctx_omni->python_t2w_pid, SIGKILL);
            }
        }
        
        waitpid(ctx_omni->python_t2w_pid, &status, 0);
#endif
        ctx_omni->python_t2w_pid = -1;
    }
    
    ctx_omni->python_t2w_initialized = false;
    print_with_timestamp("Python T2W: 服务已停止\n");
}

// 发送命令到 Python 服务并获取响应
bool send_python_t2w_command(struct omni_context * ctx_omni, const std::string& cmd_json, std::string& response) {
    if (!ctx_omni->python_t2w_initialized || !ctx_omni->python_t2w_stdin || !ctx_omni->python_t2w_stdout) {
        return false;
    }
    
    fprintf(ctx_omni->python_t2w_stdin, "%s\n", cmd_json.c_str());
    fflush(ctx_omni->python_t2w_stdin);
    
    char buffer[8192];
    if (fgets(buffer, sizeof(buffer), ctx_omni->python_t2w_stdout)) {
        response = buffer;
        while (!response.empty() && (response.back() == '\n' || response.back() == '\r')) {
            response.pop_back();
        }
        return true;
    }
    
    return false;
}

// 初始化 Python Token2Wav 模型
bool init_python_t2w_model(struct omni_context * ctx_omni, const std::string& device) {
    if (!ctx_omni->python_t2w_initialized) {
        if (!start_python_t2w_service(ctx_omni)) {
            return false;
        }
    }
    
    // 设备格式转换："gpu:0" -> "cuda:0"，"cpu" -> "cpu"。
    // CUDA_VISIBLE_DEVICES 已在 fork 时设置，Python 子进程只能看到一张卡，所以统一用 cuda:0
    std::string python_device = "cuda:0";
    if (device.find("cpu") != std::string::npos) {
        python_device = "cpu";
    }
    
    // 使用 float16 以节省显存
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), 
             "{\"cmd\":\"init\",\"model_dir\":\"%s\",\"device\":\"%s\",\"float16\":true,\"n_timesteps\":5}",
             ctx_omni->python_t2w_model_dir.c_str(), python_device.c_str());
    
    std::string response;
    if (!send_python_t2w_command(ctx_omni, cmd, response)) {
        LOG_ERR("Python T2W: init 命令发送失败\n");
        return false;
    }
    
    print_with_timestamp("Python T2W init 响应: %s\n", response.c_str());
    return response.find("\"status\":\"ok\"") != std::string::npos || 
           response.find("\"status\": \"ok\"") != std::string::npos;
}

// 设置参考音频
bool set_python_t2w_ref_audio(struct omni_context * ctx_omni, const std::string& ref_audio_path) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"set_ref_audio\",\"ref_audio_path\":\"%s\"}", ref_audio_path.c_str());
    
    std::string response;
    if (!send_python_t2w_command(ctx_omni, cmd, response)) {
        LOG_ERR("Python T2W: set_ref_audio 命令发送失败\n");
        return false;
    }
    
    print_with_timestamp("Python T2W set_ref_audio 响应: %s\n", response.c_str());
    return response.find("\"status\":\"ok\"") != std::string::npos ||
           response.find("\"status\": \"ok\"") != std::string::npos;
}

// 处理 tokens 并生成 WAV
bool process_python_t2w_tokens(struct omni_context * ctx_omni, 
                                const std::vector<int32_t>& tokens, 
                                bool last_chunk, 
                                const std::string& output_path,
                                double& inference_time_ms,
                                double& audio_duration) {
    std::string tokens_json = "[";
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) tokens_json += ",";
        tokens_json += std::to_string(tokens[i]);
    }
    tokens_json += "]";
    
    char cmd[8192];
    snprintf(cmd, sizeof(cmd), 
             "{\"cmd\":\"process\",\"tokens\":%s,\"last_chunk\":%s,\"output_path\":\"%s\"}",
             tokens_json.c_str(), 
             last_chunk ? "true" : "false",
             output_path.c_str());
    
    std::string response;
    if (!send_python_t2w_command(ctx_omni, cmd, response)) {
        LOG_ERR("Python T2W: process 命令发送失败\n");
        return false;
    }
    
    // 解析响应中的时间信息
    // 简单解析，不使用 JSON 库
    size_t pos = response.find("\"inference_time_ms\":");
    if (pos != std::string::npos) {
        inference_time_ms = atof(response.c_str() + pos + 20);
    }
    pos = response.find("\"audio_duration\":");
    if (pos != std::string::npos) {
        audio_duration = atof(response.c_str() + pos + 17);
    }
    
    return response.find("\"status\":\"ok\"") != std::string::npos ||
           response.find("\"status\": \"ok\"") != std::string::npos;
}

// 重置 Python T2W 缓存
bool reset_python_t2w_cache(struct omni_context * ctx_omni) {
    std::string response;
    if (!send_python_t2w_command(ctx_omni, "{\"cmd\":\"reset\"}", response)) {
        LOG_ERR("Python T2W: reset 命令发送失败\n");
        return false;
    }
    return response.find("\"status\":\"ok\"") != std::string::npos ||
           response.find("\"status\": \"ok\"") != std::string::npos;
}

// ======================= Token2Wav 线程函数 =======================

// Python Token2Wav thread function
static void t2w_thread_func_python(struct omni_context * ctx_omni, common_params *params) {
    print_with_timestamp("T2W thread (Python) started\n");
    fflush(stdout);
    
    auto& queue = ctx_omni->t2w_thread_info->queue;
    auto& mtx = ctx_omni->t2w_thread_info->mtx;
    auto& cv = ctx_omni->t2w_thread_info->cv;
    
    // Token2Wav sliding window parameters
    constexpr int32_t CHUNK_SIZE = 25;      // Main chunk size (25 tokens = 1s audio)
    constexpr int32_t PRE_LOOKAHEAD = 3;    // Lookahead for overlap
    constexpr int32_t WINDOW_SIZE = CHUNK_SIZE + PRE_LOOKAHEAD;
    
    // Buffer to accumulate tokens (for sliding window)
    std::vector<int32_t> token_buffer = {4218, 4218, 4218};
    
    // 使用可配置的 base_output_dir 以支持多实例
    const std::string& base_output_dir = ctx_omni->base_output_dir;
    
    // 获取轮次对应的输出目录（单工用 round_XXX 子目录，双工用统一目录）
    auto get_wav_output_dir = [&]() -> std::string {
        if (!ctx_omni->duplex_mode) {
            char round_dir[512];
            snprintf(round_dir, sizeof(round_dir), "%s/round_%03d/tts_wav", 
                     base_output_dir.c_str(), ctx_omni->simplex_round_idx);
            return std::string(round_dir);
        } else {
            return base_output_dir + "/tts_wav";
        }
    };
    
    int last_round_idx = ctx_omni->simplex_round_idx;
    std::string tts_wav_output_dir = get_wav_output_dir();
    int wav_idx = 0;
    const int sample_rate = 24000;  // Python Token2Wav 输出采样率
    
    {
        cross_platform_mkdir_p(tts_wav_output_dir);
    }
    
    while (t2w_thread_running) {
        if (ctx_omni->break_event.load()) {
            std::lock_guard<std::mutex> lock(mtx);
            while (!queue.empty()) {
                T2WOut *t2w_out = queue.front();
                queue.pop();
                delete t2w_out;
            }
            ctx_omni->break_event = false;
            token_buffer = {4218, 4218, 4218};
            wav_idx = 0;
            
            if (!ctx_omni->duplex_mode) {
                ctx_omni->wav_turn_base += 1000;
            }
            
            if (!ctx_omni->duplex_mode && ctx_omni->simplex_round_idx != last_round_idx) {
                last_round_idx = ctx_omni->simplex_round_idx;
                tts_wav_output_dir = get_wav_output_dir();
                cross_platform_mkdir_p(tts_wav_output_dir);
            }
            
            reset_python_t2w_cache(ctx_omni);
            continue;
        }
        
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return !queue.empty() || !t2w_thread_running; });
        
        if (!t2w_thread_running && queue.empty()) {
            break;
        }
        
        if (ctx_omni->break_event.load()) {
            lock.unlock();
            continue;
        }
        
        std::vector<llama_token> new_tokens;
        bool is_final = false;
        bool is_chunk_end = false;
        int received_round_idx = -1;  // 队列消息携带的 round_idx
        
        while (!queue.empty()) {
            T2WOut *t2w_out = queue.front();
            queue.pop();
            new_tokens.insert(new_tokens.end(), t2w_out->audio_tokens.begin(), t2w_out->audio_tokens.end());
            is_final = is_final || t2w_out->is_final;
            is_chunk_end = is_chunk_end || t2w_out->is_chunk_end;
            // 保存最后一个有效的 round_idx
            if (t2w_out->round_idx >= 0) {
                received_round_idx = t2w_out->round_idx;
            }
            delete t2w_out;
        }
        
        lock.unlock();

        if (new_tokens.empty() && !is_chunk_end && !is_final) {
            continue;
        }
        
        // 通过 T2WOut 传递 round_idx，用传入值确定输出目录，
        // 比从 ctx_omni->simplex_round_idx 读取更可靠，避免竞态条件
        if (!ctx_omni->duplex_mode && received_round_idx >= 0 && received_round_idx != last_round_idx) {
            print_with_timestamp("T2W(Python): round_idx 变化 %d -> %d（来自T2WOut），更新输出目录\n",
                                last_round_idx, received_round_idx);
            last_round_idx = received_round_idx;
            tts_wav_output_dir = get_wav_output_dir();
            cross_platform_mkdir_p(tts_wav_output_dir);
            // 重置 wav 索引，因为是新的轮次
            wav_idx = 0;
        }
        
        token_buffer.insert(token_buffer.end(), new_tokens.begin(), new_tokens.end());
        
        if (!ctx_omni->python_t2w_initialized) {
            continue;
        }
        
        bool need_flush = false;
        size_t min_process_threshold = WINDOW_SIZE;
        
        if (!ctx_omni->duplex_mode) {
            need_flush = is_final || is_chunk_end;
        } else {
            need_flush = is_final;
        }

        // is_final=true 但 token_buffer 为空时也要重置 T2W 流式缓存，
        // 否则下一个 turn 的音频会与上一个 turn 的尾音混在一起
        if (is_final && token_buffer.empty()) {
            print_with_timestamp("T2W(Python): is_final=true but token_buffer empty, calling reset directly\n");
            reset_python_t2w_cache(ctx_omni);
            continue;
        }
        
        // Process windows using sliding window
        while (token_buffer.size() >= min_process_threshold || (need_flush && !token_buffer.empty())) {
            size_t process_size = std::min(token_buffer.size(), (size_t)WINDOW_SIZE);
            bool is_last_window = need_flush && (token_buffer.size() <= WINDOW_SIZE);
            
            std::vector<int32_t> window(token_buffer.begin(), token_buffer.begin() + process_size);
            
            std::string wav_path = tts_wav_output_dir + "/wav_" + std::to_string(ctx_omni->wav_turn_base + wav_idx) + ".wav";
            
            double inference_time_ms = 0;
            double audio_duration = 0;
            
            if (process_python_t2w_tokens(ctx_omni, window, is_last_window, wav_path, inference_time_ms, audio_duration)) {
                // Audio output callback (for WS protocol) — read the WAV and pass float32 samples
                if (ctx_omni->audio_output_cb && audio_duration > 0) {
                    std::vector<int16_t> pcm;
                    if (read_wav_pcm16_data(wav_path, pcm)) {
                        std::vector<float> float_samples(pcm.size());
                        for (size_t i = 0; i < pcm.size(); ++i) {
                            float_samples[i] = static_cast<float>(pcm[i]) / 32768.0f;
                        }
                        ctx_omni->audio_output_cb(float_samples.data(), static_cast<int>(float_samples.size()), 24000, is_last_window);
                    } else {
                        LOG_ERR("Python T2W: failed to read WAV data chunk from %s\n", wav_path.c_str());
                    }
                }

                if (audio_duration > 0) {
                    auto wav_complete_time = std::chrono::high_resolution_clock::now();
                    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        wav_complete_time - ctx_omni->stream_decode_start_time).count();
                    
                    if (wav_idx == 0) {
                        print_with_timestamp("🎉 首响时间 (First Audio Response): %lldms\n", (long long)elapsed_ms);
                    }
                    
                    float rtf = (float)(inference_time_ms / 1000.0) / audio_duration;
                    print_with_timestamp("T2W(Python): wav_%d.wav | %.2fs audio | %.1fms inference | RTF=%.2f | t=%lldms\n",
                                        ctx_omni->wav_turn_base + wav_idx, audio_duration, inference_time_ms, rtf, (long long)elapsed_ms);
                    wav_idx++;
                    
                    // 注意：不要在中途重置缓存！Python 原版在整个对话中保持缓存连续
                    // 只在对话/轮次结束时（is_final=true）才重置缓存
                    // 使用独立 GPU 后，显存不再是问题
                }
            } else {
                LOG_ERR("T2W(Python): process 失败\n");
            }
            
            if (!ctx_omni->duplex_mode) {
                if (token_buffer.size() > CHUNK_SIZE) {
                    token_buffer.erase(token_buffer.begin(), token_buffer.begin() + CHUNK_SIZE);
                } else {
                    token_buffer.clear();
                }
            } else {
                size_t slide_amount;
                if (is_last_window) {
                    slide_amount = token_buffer.size();
                } else if (token_buffer.size() > CHUNK_SIZE) {
                    slide_amount = CHUNK_SIZE;
                } else if (token_buffer.size() > PRE_LOOKAHEAD) {
                    slide_amount = token_buffer.size() - PRE_LOOKAHEAD;
                } else {
                    slide_amount = 0;
                }
                
                if (slide_amount > 0 && slide_amount <= token_buffer.size()) {
                    token_buffer.erase(token_buffer.begin(), token_buffer.begin() + slide_amount);
                } else if (slide_amount > token_buffer.size()) {
                    token_buffer.clear();
                }
            }
            
            if (is_last_window) {
                if (is_final) {
                    std::string done_flag_path = tts_wav_output_dir + "/generation_done.flag";
                    FILE* flag_file = fopen(done_flag_path.c_str(), "w");
                    if (flag_file) {
                        int last_wav_idx = (wav_idx > 0) ? (ctx_omni->wav_turn_base + wav_idx - 1) : 0;
                        fprintf(flag_file, "%d\n", last_wav_idx);
                        fclose(flag_file);
                    }
                    
                    token_buffer = {4218, 4218, 4218};
                    
                    reset_python_t2w_cache(ctx_omni);
                    
                    if (!ctx_omni->duplex_mode) {
                        wav_idx = 0;
                        ctx_omni->wav_turn_base += 1000;
                    }
                    
                    if (!ctx_omni->duplex_mode && ctx_omni->simplex_round_idx != last_round_idx) {
                        last_round_idx = ctx_omni->simplex_round_idx;
                        tts_wav_output_dir = get_wav_output_dir();
                        cross_platform_mkdir_p(tts_wav_output_dir);
                    }
                }
                break;
            }
        }
    }
    
    print_with_timestamp("T2W(Python) 线程: 停止\n");
    fflush(stdout);
}

// C++ Token2Wav 线程函数（不使用 Python 服务时走原生 token2wav 实现）
// ==============================================================================
// T2W Thread Function (C++ Token2Wav)
// ==============================================================================
// 
// 轮次管理说明（单工模式）：
// - simplex_round_idx 是轮次索引的唯一来源（全局变量），初始为 0，每轮结束 +1。
//   更新时机：stream_decode 开始时用传入的 round_idx 同步；TTS 线程每轮结束（发送 is_final=true 前）递增。
// - last_round_idx 是 T2W 线程本地记录的「上一次使用的轮次」，用于检测 simplex_round_idx
//   是否变化，变化时更新输出目录（单工每轮 WAV 输出到不同的 round_XXX 子目录）。
// - 轮次同步流程：Python 调用 stream_decode(round_idx) 同步 simplex_round_idx；
//   T2W 线程检测到变化后更新 tts_wav_output_dir。
// ==============================================================================
static void t2w_thread_func_cpp(struct omni_context * ctx_omni, common_params *params) {
    print_with_timestamp("T2W thread (C++) started\n");
    fflush(stdout);
    
    auto& queue = ctx_omni->t2w_thread_info->queue;
    auto& mtx = ctx_omni->t2w_thread_info->mtx;
    auto& cv = ctx_omni->t2w_thread_info->cv;
    
    // Token2Wav sliding window parameters
    constexpr int32_t CHUNK_SIZE = 25;      // Main chunk size (25 tokens = 1s audio)
    constexpr int32_t PRE_LOOKAHEAD = 3;    // Lookahead for overlap
    constexpr int32_t WINDOW_SIZE = CHUNK_SIZE + PRE_LOOKAHEAD;
    
    // Buffer to accumulate tokens (for sliding window)
    // Python: buffer = [4218] * 3  # 预先放入3个前缀静音token
    std::vector<int32_t> token_buffer = {4218, 4218, 4218};
    
    // 使用可配置的 base_output_dir 以支持多实例
    const std::string& base_output_dir = ctx_omni->base_output_dir;
    
    // 获取轮次对应的输出目录（单工用 round_XXX 子目录，双工用统一目录）
    auto get_wav_output_dir = [&]() -> std::string {
        if (!ctx_omni->duplex_mode) {
            // 单工模式：使用 round_XXX 子目录
            char round_dir[512];
            snprintf(round_dir, sizeof(round_dir), "%s/round_%03d/tts_wav", 
                     base_output_dir.c_str(), ctx_omni->simplex_round_idx);
            return std::string(round_dir);
        } else {
            // 双工模式：直接使用 base_output_dir
            return base_output_dir + "/tts_wav";
        }
    };
    
    // last_round_idx：T2W 线程本地记录的「上一次使用的轮次」，
    // 用于检测 simplex_round_idx 是否变化，变化则更新输出目录。
    // 初始化为 -1，确保第一轮也触发目录创建
    int last_round_idx = -1;

    // WAV output settings (与 tts_thread_func 保持一致)
    std::string tts_wav_output_dir = get_wav_output_dir();
    int wav_idx = 0;
    const int sample_rate = omni::flow::Token2Wav::kSampleRate;
    
    while (t2w_thread_running) {
        // 检测 break_event 并清空队列
        if (ctx_omni->break_event.load()) {
            std::lock_guard<std::mutex> lock(mtx);
            while (!queue.empty()) {
                T2WOut *t2w_out = queue.front();
                queue.pop();
                delete t2w_out;
            }
            ctx_omni->break_event = false;
            token_buffer = {4218, 4218, 4218};
            wav_idx = 0;
            
            // 在 T2W 线程处理完打断后再递增 wav_turn_base，
            // 确保当前轮次的所有 wav 文件使用旧编号，然后再切换到新编号
            if (!ctx_omni->duplex_mode) {
                ctx_omni->wav_turn_base += 1000;
            }
            
            // 单工：打断后更新输出目录（TTS 线程已递增 simplex_round_idx）
            if (!ctx_omni->duplex_mode && ctx_omni->simplex_round_idx != last_round_idx) {
                last_round_idx = ctx_omni->simplex_round_idx;
                tts_wav_output_dir = get_wav_output_dir();
                print_with_timestamp("T2W线程: 打断后更新输出目录为 %s\n", tts_wav_output_dir.c_str());
            }
            continue;
        }
        
        std::unique_lock<std::mutex> lock(mtx);
        
        cv.wait(lock, [&] { return !queue.empty() || !t2w_thread_running || ctx_omni->break_event.load(); });
        auto dequeue_time = std::chrono::steady_clock::now();
        
        if (!t2w_thread_running && queue.empty()) {
            break;
        }
        
        // 检测到 break_event 时跳过当前数据
        if (ctx_omni->break_event.load()) {
            lock.unlock();
            continue;
        }
        
        std::vector<llama_token> new_tokens;
        bool is_final = false;
        bool is_chunk_end = false;  // 标记 TTS chunk 结束
        int received_round_idx = -1;  // 队列消息携带的 round_idx
        uint64_t received_frame_id = 0;
        int received_chunk_idx = -1;
        
        std::chrono::steady_clock::time_point oldest_enqueue_time = dequeue_time;
        bool have_enqueue_time = false;
        while (!queue.empty()) {
            T2WOut *t2w_out = queue.front();
            queue.pop();
            if (!have_enqueue_time || t2w_out->enqueue_time < oldest_enqueue_time) {
                oldest_enqueue_time = t2w_out->enqueue_time;
                have_enqueue_time = true;
            }
            
            new_tokens.insert(new_tokens.end(), t2w_out->audio_tokens.begin(), t2w_out->audio_tokens.end());
            is_final = is_final || t2w_out->is_final;
            is_chunk_end = is_chunk_end || t2w_out->is_chunk_end;
            // 保存最后一个有效的 round_idx（优先使用非负值）
            if (t2w_out->round_idx >= 0) {
                received_round_idx = t2w_out->round_idx;
            }
            if (t2w_out->frame_id != 0) received_frame_id = t2w_out->frame_id;
            if (t2w_out->chunk_idx >= 0) received_chunk_idx = t2w_out->chunk_idx;
            delete t2w_out;
        }

        lock.unlock();
        
        if (new_tokens.empty() && !is_chunk_end && !is_final) {
            continue;
        }
        
        // 用 T2WOut 传入的 round_idx 判断轮次切换，而不是直接读 simplex_round_idx。
        // TTS 线程在发送 is_final 前会递增 simplex_round_idx，导致竞态条件；
        // T2WOut.round_idx 保存的是递增前的值，确保 WAV 写入正确的目录
        int effective_round_idx = (received_round_idx >= 0) ? received_round_idx : ctx_omni->simplex_round_idx;
        
        if (!ctx_omni->duplex_mode && effective_round_idx != last_round_idx) {
            print_with_timestamp("T2W线程(C++): 轮次切换 (%d -> %d)，更新输出目录\n",
                                last_round_idx, effective_round_idx);
            
            // 更新本地记录的轮次
            last_round_idx = effective_round_idx;
            
            // 更新输出目录（基于 effective_round_idx）
            tts_wav_output_dir = base_output_dir + "/round_" + 
                                 (effective_round_idx < 100 ? (effective_round_idx < 10 ? "00" : "0") : "") + 
                                 std::to_string(effective_round_idx) + "/tts_wav";
            
            wav_idx = 0;                                                    // WAV 文件编号从 0 开始
            ctx_omni->wav_turn_base = effective_round_idx * 1000;           // 更新全局 WAV 编号基数
            token_buffer = {4218, 4218, 4218};                              // 重置 token buffer（3个静音前缀）
            
            print_with_timestamp("T2W线程(C++): 新输出目录 %s\n", tts_wav_output_dir.c_str());
            
            cross_platform_mkdir_p(tts_wav_output_dir);
        }
        
        size_t buffer_before = token_buffer.size();
        token_buffer.insert(token_buffer.end(), new_tokens.begin(), new_tokens.end());
        const double queue_wait_ms = have_enqueue_time
            ? std::chrono::duration<double, std::milli>(dequeue_time - oldest_enqueue_time).count()
            : 0.0;
        
        if (new_tokens.size() > 0) {
            std::string tokens_str = "[";
            for (size_t i = 0; i < std::min(new_tokens.size(), (size_t)10); i++) {
                tokens_str += std::to_string(new_tokens[i]);
                if (i < std::min(new_tokens.size(), (size_t)10) - 1) tokens_str += ",";
            }
            if (new_tokens.size() > 10) {
                tokens_str += "...";
                for (size_t i = new_tokens.size() - 3; i < new_tokens.size(); i++) {
                    tokens_str += "," + std::to_string(new_tokens[i]);
                }
            }
            tokens_str += "]";
        }
        
        if (!ctx_omni->token2wav_initialized || !ctx_omni->token2wav_session) {
            continue;
        }
        
        // 处理逻辑（单工/双工分开）
        bool need_flush = false;
        size_t min_process_threshold = WINDOW_SIZE;
        
        if (!ctx_omni->duplex_mode) {
            need_flush = is_final || is_chunk_end;
        } else {
            // 双工：只在 is_final（LISTEN→SPEAK 切换时 LLM 线程设置）时 flush。
            // chunk_end 保持原语义"累积等下个 chunk"，避免把一轮 turn 切成多段 wav 产生播放 gap。
            need_flush = is_final;
        }
        
        // Process windows using sliding window
        int process_count = 0;
        while (token_buffer.size() >= min_process_threshold || (need_flush && !token_buffer.empty())) {
            size_t process_size = std::min(token_buffer.size(), (size_t)WINDOW_SIZE);
            // is_last_window：只有 is_final 才算轮次真正的"最后窗口"（触发 token2wav 终结 + buffer 重置）。
            // 双工 chunk_end 不能视为 last_window，否则 token2wav 会被重置，丢失跨 chunk 的状态
            bool is_last_window = is_final && (token_buffer.size() <= WINDOW_SIZE);
            
            std::vector<int32_t> window(token_buffer.begin(), token_buffer.begin() + process_size);
            print_with_timestamp("[DUPLEX CONTRACT] frame=%llu edge=T2WOut consumer=T2W window=%zu main=%d lookahead=%d chunk_end=%d final=%d\n",
                                 (unsigned long long) received_frame_id, window.size(),
                                 CHUNK_SIZE, PRE_LOOKAHEAD, is_chunk_end ? 1 : 0,
                                 is_final ? 1 : 0);
            
            auto t2w_start = std::chrono::high_resolution_clock::now();

            std::vector<float> chunk_wav;
            bool _t2w_fed = ctx_omni->token2wav_session->feed_window(window, is_last_window, chunk_wav);
            if (_t2w_fed) {
                auto t2w_end = std::chrono::high_resolution_clock::now();
                double t2w_ms = std::chrono::duration<double, std::milli>(t2w_end - t2w_start).count();
                
                if (!chunk_wav.empty()) {
                    // Audio output callback (for WS protocol)
                    if (ctx_omni->audio_output_cb) {
                        ctx_omni->audio_output_cb(chunk_wav.data(), static_cast<int>(chunk_wav.size()),
                                                  sample_rate, is_last_window);
                    }

                    std::string wav_path = tts_wav_output_dir + "/wav_" + std::to_string(ctx_omni->wav_turn_base + wav_idx) + ".wav";
                    
                    const int16_t num_channels = 1;
                    const int16_t bits_per_sample = 16;
                    const int16_t block_align = num_channels * (bits_per_sample / 8);
                    const int32_t byte_rate = sample_rate * block_align;
                    
                    std::vector<int16_t> pcm(chunk_wav.size());
                    for (size_t i = 0; i < chunk_wav.size(); ++i) {
                        float x = chunk_wav[i];
                        if (!std::isfinite(x)) x = 0.0f;
                        x = std::max(-1.0f, std::min(1.0f, x));
                        pcm[i] = (int16_t)(x * 32767.0f);
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
                        
                        float audio_duration = chunk_wav.size() / (float)sample_rate;
                        float rtf = (float)(t2w_ms / 1000.0) / audio_duration;
                        
                        auto wav_complete_time = std::chrono::high_resolution_clock::now();
                        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            wav_complete_time - ctx_omni->stream_decode_start_time).count();
                        
                        if (wav_idx == 0) {
                            print_with_timestamp("🎉 首响时间 (First Audio Response): %lldms\n", (long long)elapsed_ms);
                        }
                        // 记录每个音频窗口完成的绝对 wall 时间（供测试计算 TTFA/FAR：
                        // 帧结果到达后第一个 wav 的完成时刻 - 该帧 push 时间）。
                        {
                            const double wall_ms = std::chrono::duration<double, std::milli>(
                                wav_complete_time.time_since_epoch()).count();
                            std::lock_guard<std::mutex> lk(ctx_omni->t2w_wav_mtx);
                            ctx_omni->t2w_wav_walls.push_back(wall_ms);
                        }
                        print_with_timestamp("T2W线程: wav_%d.wav | %.2fs audio | %.1fms inference | RTF=%.2f | t=%lldms | queue_wait=%.1fms\n",
                                            ctx_omni->wav_turn_base + wav_idx, audio_duration, t2w_ms, rtf, (long long)elapsed_ms, queue_wait_ms);
                        wav_idx++;
                    }
                }
            } else {
                LOG_ERR("T2W线程: feed_window 失败\n");
            }

            // Slide window by CHUNK_SIZE (25), keep last PRE_LOOKAHEAD (3) for overlap
            size_t buffer_before_slide = token_buffer.size();
            
            if (!ctx_omni->duplex_mode) {
                if (token_buffer.size() > CHUNK_SIZE) {
                    token_buffer.erase(token_buffer.begin(), token_buffer.begin() + CHUNK_SIZE);
                } else {
                    token_buffer.clear();
                }
            } else {
                // 双工模式，滑动逻辑与 Python 版对齐：
                // Python: self.token2wav_buffer = self.token2wav_buffer[min(CHUNK_SIZE, chunk_to_process - self.pre_lookahead):]
                size_t slide_amount;
                if (is_last_window) {
                    slide_amount = token_buffer.size();  // 最后一个 window，清空 buffer
                } else if (token_buffer.size() > CHUNK_SIZE) {
                    slide_amount = CHUNK_SIZE;
                } else if (token_buffer.size() > PRE_LOOKAHEAD) {
                    slide_amount = token_buffer.size() - PRE_LOOKAHEAD;  // 有 chunk_eos 时保留 pre_lookahead
                } else {
                    slide_amount = 0;  // 剩余太少，不滑动
                }
                
                if (slide_amount > 0 && slide_amount <= token_buffer.size()) {
                    token_buffer.erase(token_buffer.begin(), token_buffer.begin() + slide_amount);
                } else if (slide_amount > token_buffer.size()) {
                    token_buffer.clear();
                }
            }
            process_count++;
            
            if (is_last_window) {
                // 与 Python 对齐：只有 is_final（轮次结束）时才重置
                // Python: if is_last_chunk: stream(..., last_chunk=True); buffer = []
                // 普通 chunk 结束时，剩余 tokens 保留在 buffer 中等待下一个 chunk
                if (is_final) {
                    // 写入结束标记文件，通知 Python 立即结束（无需等待超时）。
                    // 标记文件包含最后一个 wav 的编号，方便 Python 验证
                    {
                        std::string done_flag_path = tts_wav_output_dir + "/generation_done.flag";
                        FILE* flag_file = fopen(done_flag_path.c_str(), "w");
                        if (flag_file) {
                            // 写入最后一个 wav 的编号（wav_idx - 1，因为 wav_idx 已经指向下一个）
                            int last_wav_idx = (wav_idx > 0) ? (ctx_omni->wav_turn_base + wav_idx - 1) : 0;
                            fprintf(flag_file, "%d\n", last_wav_idx);
                            fclose(flag_file);
                            print_with_timestamp("T2W线程: 写入结束标记 %s (last_wav=%d)\n", done_flag_path.c_str(), last_wav_idx);
                        }
                    }
                    
                    // 不调用 Token2WavSession::reset()：
                    // reset() 会把 stream_started_ 置 false，导致下一轮 feed_window 失败。
                    // 单工和双工都只重置 token_buffer，保持 Token2Wav 的 stream 状态
                    token_buffer = {4218, 4218, 4218};
                    
                    // 在 T2W 线程处理完 is_final 后再递增 wav_turn_base，
                    // 确保当前轮次的所有 wav 文件使用旧编号再切换，避免 TTS 线程提前递增导致编号跳跃
                    if (!ctx_omni->duplex_mode) {
                        wav_idx = 0;  // 单工：重置 wav_idx 用于下一轮
                        ctx_omni->wav_turn_base += 1000;
                    }
                    
                    // 单工：is_final 后检查是否需要更新目录（simplex_round_idx 已在 TTS 线程中递增）
                    if (!ctx_omni->duplex_mode && ctx_omni->simplex_round_idx != last_round_idx) {
                        last_round_idx = ctx_omni->simplex_round_idx;
                        tts_wav_output_dir = get_wav_output_dir();
                        print_with_timestamp("T2W线程: 轮次结束后更新输出目录为 %s\n", tts_wav_output_dir.c_str());
                        cross_platform_mkdir_p(tts_wav_output_dir);
                    }
                }
                // 注意：is_chunk_end 时不重置 buffer，剩余 tokens 保留给下一个 chunk
                break;
            }
        }
    }

    print_with_timestamp("T2W(C++) 线程: 停止\n");
    fflush(stdout);
}

// Token2Wav 线程入口函数（根据配置选择 Python 或 C++ 实现）
void t2w_thread_func(struct omni_context * ctx_omni, common_params *params) {
    if (ctx_omni->use_python_token2wav) {
        t2w_thread_func_python(ctx_omni, params);
    } else {
        t2w_thread_func_cpp(ctx_omni, params);
    }
}

