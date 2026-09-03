#include "omni_internal.h"

#include <chrono>
#include <cstdarg>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>

#ifdef _WIN32
    #include <direct.h>
    #include <sys/stat.h>
#else
    #include <sys/stat.h>
    #include <unistd.h>
#endif

void print_with_timestamp(const char * format, ...) {
    const auto now = std::chrono::system_clock::now();
    const auto in_time_t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm buf;
#ifdef _WIN32
    localtime_s(&buf, &in_time_t);
#else
    localtime_r(&in_time_t, &buf);
#endif

    // 整行一次性 fwrite 到 stderr：与其它诊断（sm_log / [pb-prof] 等）统一走
    // stderr 单次写入，避免 stdout(全缓冲) 与 stderr(无缓冲) 在 `2>&1` 合并时
    // 行被从中间切断（曾导致 T2W wav 行 / LLM prefill 行被 [SM SEG] 打印插入）。
    char ts[64];
    strftime(ts, sizeof(ts), "%H:%M:%S", &buf);

    char line[2048];
    const int n = snprintf(line, sizeof(line), "%s.%03d ", ts, (int) ms.count());
    if (n < 0) return;
    va_list args;
    va_start(args, format);
    const int m = vsnprintf(line + n, sizeof(line) - (size_t) n, format, args);
    va_end(args);
    if (m < 0) return;

    const size_t avail   = sizeof(line) - (size_t) n;
    const size_t written = (size_t) m < avail ? (size_t) m : avail - 1;
    fwrite(line, 1, (size_t) n + written, stderr);
}

bool cross_platform_mkdir_p(const std::string & path) {
    if (path.empty()) return false;

    std::string normalized = path;
#ifdef _WIN32
    for (char & c : normalized) {
        if (c == '/') c = '\\';
    }
    size_t pos = 0;
    if (normalized.size() >= 2 && normalized[1] == ':') {
        pos = 2;
        if (normalized.size() > 2 && normalized[2] == '\\') pos = 3;
    }
    while (pos < normalized.size() && normalized[pos] == '\\') ++pos;

    while (pos < normalized.size()) {
        pos = normalized.find('\\', pos);
        if (pos == std::string::npos) pos = normalized.size();
        const std::string sub = normalized.substr(0, pos);
        if (!sub.empty()) {
            struct _stat info;
            if (_stat(sub.c_str(), &info) != 0 &&
                    _mkdir(sub.c_str()) != 0 && errno != EEXIST) {
                return false;
            }
        }
        if (pos < normalized.size()) ++pos;
    }
    return true;
#else
    for (char & c : normalized) {
        if (c == '\\') c = '/';
    }
    size_t pos = normalized.empty() || normalized[0] != '/' ? 0 : 1;
    while (pos < normalized.size()) {
        pos = normalized.find('/', pos);
        if (pos == std::string::npos) pos = normalized.size();
        const std::string sub = normalized.substr(0, pos);
        if (!sub.empty()) {
            struct stat info;
            if (::stat(sub.c_str(), &info) != 0 &&
                    mkdir(sub.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
        if (pos < normalized.size()) ++pos;
    }
    return true;
#endif
}

bool read_wav_pcm16_data(const std::string & wav_path, std::vector<int16_t> & pcm) {
    FILE * f = fopen(wav_path.c_str(), "rb");
    if (!f) return false;

    auto read_u32_le = [](FILE * fp, uint32_t & out) {
        unsigned char b[4];
        if (fread(b, 1, 4, fp) != 4) return false;
        out = (uint32_t) b[0] | ((uint32_t) b[1] << 8) |
              ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
        return true;
    };

    char riff[4];
    char wave[4];
    uint32_t riff_size = 0;
    if (fread(riff, 1, 4, f) != 4 || !read_u32_le(f, riff_size) ||
            fread(wave, 1, 4, f) != 4 || memcmp(riff, "RIFF", 4) != 0 ||
            memcmp(wave, "WAVE", 4) != 0) {
        fclose(f);
        return false;
    }
    (void) riff_size;

    while (true) {
        char chunk_id[4];
        uint32_t chunk_size = 0;
        if (fread(chunk_id, 1, 4, f) != 4 || !read_u32_le(f, chunk_size)) break;
        if (memcmp(chunk_id, "data", 4) == 0) {
            const size_t n_samples = chunk_size / sizeof(int16_t);
            pcm.assign(n_samples, 0);
            pcm.resize(fread(pcm.data(), sizeof(int16_t), n_samples, f));
            fclose(f);
            return !pcm.empty();
        }
        const long skip = (long) chunk_size + (chunk_size & 1u);
        if (fseek(f, skip, SEEK_CUR) != 0) break;
    }

    fclose(f);
    return false;
}

