#pragma once

#include <string>

struct OmniRoundMeta;

bool cross_platform_mkdir_p(const std::string & path);
bool omni_ensure_directory(const std::string & dir_path);
std::string omni_round_output_dir(const std::string & base_output_dir, const OmniRoundMeta & round_meta);
std::string omni_round_tts_wav_output_dir(const std::string & base_output_dir, const OmniRoundMeta & round_meta);
void omni_archive_output_dir(const std::string & base_output_dir, const std::string & archive_root = "./old_output");
void omni_merge_wav_files(const std::string & output_dir, int num_chunks);
