#include "omni-output.h"

#include "omni-impl.h"
#include "omni-session-state.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#include <sys/types.h>
#define popen  _popen
#define pclose _pclose
#define unlink _unlink
#define stat   _stat
#define S_IFDIR _S_IFDIR
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

bool cross_platform_mkdir_p(const std::string & path) {
    if (path.empty()) {
        return false;
    }

    std::string normalized = path;
#ifdef _WIN32
    for (char & c : normalized) {
        if (c == '/') {
            c = '\\';
        }
    }

    size_t pos = 0;
    if (normalized.size() >= 2 && normalized[1] == ':') {
        pos = 2;
        if (normalized.size() > 2 && normalized[2] == '\\') {
            pos = 3;
        }
    }
    while (pos < normalized.size() && normalized[pos] == '\\') {
        ++pos;
    }

    while (pos < normalized.size()) {
        pos = normalized.find('\\', pos);
        if (pos == std::string::npos) {
            pos = normalized.size();
        }

        const std::string sub = normalized.substr(0, pos);
        if (!sub.empty()) {
            struct _stat info;
            if (_stat(sub.c_str(), &info) != 0) {
                if (_mkdir(sub.c_str()) != 0 && errno != EEXIST) {
                    return false;
                }
            }
        }

        if (pos < normalized.size()) {
            ++pos;
        }
    }

    return true;
#else
    for (char & c : normalized) {
        if (c == '\\') {
            c = '/';
        }
    }

    size_t pos = 0;
    if (!normalized.empty() && normalized[0] == '/') {
        pos = 1;
    }

    while (pos < normalized.size()) {
        pos = normalized.find('/', pos);
        if (pos == std::string::npos) {
            pos = normalized.size();
        }

        const std::string sub = normalized.substr(0, pos);
        if (!sub.empty()) {
            struct stat info;
            if (::stat(sub.c_str(), &info) != 0) {
                if (mkdir(sub.c_str(), 0755) != 0 && errno != EEXIST) {
                    return false;
                }
            }
        }

        if (pos < normalized.size()) {
            ++pos;
        }
    }

    return true;
#endif
}

bool omni_ensure_directory(const std::string & dir_path) {
    struct stat info;
    if (stat(dir_path.c_str(), &info) != 0) {
        if (!cross_platform_mkdir_p(dir_path)) {
            LOG_ERR("Failed to create output directory: %s\n", dir_path.c_str());
            return false;
        }
        return true;
    }

    if (!(info.st_mode & S_IFDIR)) {
        LOG_ERR("Output path exists but is not a directory: %s\n", dir_path.c_str());
        return false;
    }

    return true;
}

std::string omni_round_output_dir(const std::string & base_output_dir, const OmniRoundMeta & round_meta) {
    if (round_meta.duplex_mode) {
        return base_output_dir;
    }

    char round_dir[512];
    snprintf(round_dir, sizeof(round_dir), "%s/round_%03d", base_output_dir.c_str(), round_meta.round_idx);
    return std::string(round_dir);
}

std::string omni_round_tts_wav_output_dir(const std::string & base_output_dir, const OmniRoundMeta & round_meta) {
    return omni_round_output_dir(base_output_dir, round_meta) + "/tts_wav";
}

static bool omni_dir_has_content(const std::string & dir_path) {
    struct stat info;
    if (stat(dir_path.c_str(), &info) != 0) {
        return false;
    }
    if (!(info.st_mode & S_IFDIR)) {
        return false;
    }

#ifdef _WIN32
    const std::string cmd = "dir /b \"" + dir_path + "\" 2>NUL | findstr /r \".\" >NUL 2>&1";
#else
    const std::string cmd = "test -n \"$(ls -A " + dir_path + " 2>/dev/null)\"";
#endif
    return system(cmd.c_str()) == 0;
}

static int omni_get_next_archive_id(const std::string & archive_root) {
    cross_platform_mkdir_p(archive_root);

    int max_id = -1;
#ifdef _WIN32
    const std::string find_cmd = "dir /b \"" + archive_root + "\" 2>NUL";
#else
    const std::string find_cmd = "ls -1 " + archive_root + " 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1";
#endif

    FILE * pipe = popen(find_cmd.c_str(), "r");
    if (pipe == nullptr) {
        return max_id + 1;
    }

    char buffer[128];
#ifdef _WIN32
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string result(buffer);
        while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
            result.pop_back();
        }
        if (!result.empty()) {
            try {
                const int id = std::stoi(result);
                if (id > max_id) {
                    max_id = id;
                }
            } catch (...) {
            }
        }
    }
#else
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string result(buffer);
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
    return max_id + 1;
}

void omni_archive_output_dir(const std::string & base_output_dir, const std::string & archive_root) {
    bool output_has_content = omni_dir_has_content(base_output_dir);
    if (!output_has_content) {
        output_has_content = omni_dir_has_content(base_output_dir + "/llm_debug") ||
                             omni_dir_has_content(base_output_dir + "/tts_txt") ||
                             omni_dir_has_content(base_output_dir + "/tts_wav");
    }

    if (!output_has_content) {
        return;
    }

    const int next_id = omni_get_next_archive_id(archive_root);
    const std::string archive_dir = archive_root + "/" + std::to_string(next_id);
    if (!omni_ensure_directory(archive_dir)) {
        LOG_ERR("Failed to create old_output directory: %s\n", archive_dir.c_str());
        return;
    }

#ifdef _WIN32
    const std::string move_cmd = "robocopy \"" + base_output_dir + "\" \"" + archive_dir + "\" /E /MOVE >NUL 2>&1";
    system(move_cmd.c_str());
    cross_platform_mkdir_p(base_output_dir);
#else
    std::string move_cmd = "find " + base_output_dir + " -mindepth 1 -maxdepth 1 -exec mv {} " + archive_dir + "/ \\; 2>/dev/null";
    int ret = system(move_cmd.c_str());
    if (ret != 0) {
        move_cmd = "sh -c 'cd " + base_output_dir + " && mv * " + archive_dir + "/ 2>/dev/null || true'";
        ret = system(move_cmd.c_str());
        if (ret != 0) {
            LOG_WRN("Failed to move old output directory (may be empty or already moved)\n");
        }
    }
#endif
}

void omni_merge_wav_files(const std::string & output_dir, int num_chunks) {
    const std::string merged_file = output_dir + "/tts_output_merged.wav";
    std::vector<std::pair<int, std::string>> indexed_chunk_files;

    // 当前运行链路实际由 T2W 生成 wav_*.wav，这里按数字后缀排序收集。
    try {
        if (fs::exists(output_dir) && fs::is_directory(output_dir)) {
            for (const auto & entry : fs::directory_iterator(output_dir)) {
                if (!entry.is_regular_file()) {
                    continue;
                }

                const std::string filename = entry.path().filename().string();
                if (filename.rfind("wav_", 0) != 0 || entry.path().extension() != ".wav") {
                    continue;
                }

                const std::string stem = entry.path().stem().string();
                const std::string index_str = stem.substr(4);
                if (index_str.empty()) {
                    continue;
                }

                const bool is_numeric = std::all_of(index_str.begin(), index_str.end(), [](unsigned char ch) {
                    return std::isdigit(ch) != 0;
                });
                if (!is_numeric) {
                    continue;
                }

                std::error_code ec;
                const auto file_size = fs::file_size(entry.path(), ec);
                if (ec || file_size == 0) {
                    continue;
                }

                indexed_chunk_files.emplace_back(std::stoi(index_str), entry.path().string());
            }
        }
    } catch (const std::exception & e) {
        LOG_WRN("TTS: failed to scan wav files in %s: %s\n", output_dir.c_str(), e.what());
    }

    // 兼容旧目录格式：如果没有 wav_*.wav，再回退到 tts_output_chunk_*.wav。
    if (indexed_chunk_files.empty() && num_chunks > 0) {
        for (int i = 0; i < num_chunks; ++i) {
            const std::string chunk_file = output_dir + "/tts_output_chunk_" + std::to_string(i) + ".wav";
            struct stat st;
            if (stat(chunk_file.c_str(), &st) == 0 && st.st_size > 0) {
                indexed_chunk_files.emplace_back(i, chunk_file);
            }
        }
    }

    if (indexed_chunk_files.empty()) {
        LOG_WRN("TTS: no valid WAV files to merge\n");
        return;
    }

    std::sort(indexed_chunk_files.begin(), indexed_chunk_files.end(),
              [](const auto & lhs, const auto & rhs) {
                  return lhs.first < rhs.first;
              });

    std::vector<std::string> chunk_files;
    chunk_files.reserve(indexed_chunk_files.size());
    for (const auto & indexed_file : indexed_chunk_files) {
        chunk_files.push_back(indexed_file.second);
    }

    const std::string concat_list_file = output_dir + "/concat_list.txt";
    FILE * f_list = fopen(concat_list_file.c_str(), "w");
    if (f_list != nullptr) {
        for (const auto & chunk_file : chunk_files) {
            fprintf(f_list, "file '%s'\n", chunk_file.c_str());
        }
        fclose(f_list);

        const std::string ffmpeg_cmd = "ffmpeg -f concat -safe 0 -i \"" + concat_list_file + "\" -c copy \"" + merged_file + "\" -y -loglevel error 2>&1";
        const int ret = system(ffmpeg_cmd.c_str());
        unlink(concat_list_file.c_str());

        if (ret == 0) {
            struct stat st;
            if (stat(merged_file.c_str(), &st) == 0 && st.st_size > 0) {
                return;
            }
        }
    }

    std::string sox_cmd = "sox";
    for (const auto & chunk_file : chunk_files) {
        sox_cmd += " \"" + chunk_file + "\"";
    }
    sox_cmd += " \"" + merged_file + "\"";

    const int ret = system(sox_cmd.c_str());
    if (ret == 0) {
        struct stat st;
        if (stat(merged_file.c_str(), &st) != 0 || st.st_size <= 0) {
            LOG_WRN("TTS: merged file was not created or is empty (sox)\n");
        }
        return;
    }

    LOG_WRN("TTS: failed to merge WAV files (tried ffmpeg and sox). Please install ffmpeg or sox.\n");
}
