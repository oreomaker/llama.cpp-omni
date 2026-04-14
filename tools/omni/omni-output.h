#pragma once

#include <string>
#include <vector>

struct OmniRoundMeta;

bool cross_platform_mkdir_p(const std::string & path);
bool omni_ensure_directory(const std::string & dir_path);
std::string omni_round_output_dir(const std::string & base_output_dir, const OmniRoundMeta & round_meta);
std::string omni_round_tts_wav_output_dir(const std::string & base_output_dir, const OmniRoundMeta & round_meta);
bool omni_ensure_round_tts_wav_output_dir(
        const std::string & base_output_dir,
        const OmniRoundMeta & round_meta,
        std::string * out_dir = nullptr);
std::string omni_tts_wav_file_name(const OmniRoundMeta & round_meta, int chunk_idx);
std::string omni_tts_wav_file_path(const std::string & output_dir, const OmniRoundMeta & round_meta, int chunk_idx);
bool omni_write_generation_done_flag(const std::string & output_dir, const std::string & last_wav_name);
bool omni_write_wav_file_f32_mono_s16(const std::string & wav_path, const std::vector<float> & samples, int sample_rate);
void omni_archive_output_dir(const std::string & base_output_dir, const std::string & archive_root = "./old_output");
void omni_merge_wav_files(const std::string & output_dir, int num_chunks);
