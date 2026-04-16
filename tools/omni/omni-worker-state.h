#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

struct OmniWorkerState {
    std::mutex              speek_mtx;
    std::condition_variable speek_cv;

    std::atomic<bool> llm_thread_running{ true };
    std::atomic<bool> tts_thread_running{ true };
    std::atomic<bool> t2w_thread_running{ true };

    std::mutex buffer_mutex;
};
