#pragma once

#include "omni.h"

#include <string>
#include <unordered_map>
#include <vector>

// WAV output cadence analysis
struct WavCadenceMetrics {
    std::vector<double> inter_wav_intervals_ms;
    double mean_interval_ms  = 0.0;
    double stdev_interval_ms = 0.0;
    double max_interval_ms   = 0.0;
    double min_interval_ms   = 0.0;
    int    stall_count       = 0;      // intervals > stall threshold
    double smoothness_score  = 0.0;    // 1 - CV, clamped [0,1]
    int    wav_count         = 0;
};

// Per-stage dispatch cadence
struct StageCadenceMetrics {
    std::vector<double> dispatch_intervals_ms;
    double stdev_ms       = 0.0;
    double tokens_per_sec = 0.0;
    double rtf            = 0.0;      // real-time factor (> 1.0 means slower than real-time)
    int    dispatch_count = 0;
};

// Per-turn (speak event) SLO metrics
struct TurnSLOMetrics {
    int    chunk_idx       = -1;
    double ttfa_ms         = -1.0;
    double e2e_latency_ms  = -1.0;

    WavCadenceMetrics   wav_cadence;
    StageCadenceMetrics llm_cadence;
    StageCadenceMetrics tts_cadence;

    double      jitter_amplification_factor = 0.0;
    std::string bottleneck_stage;    // "llm", "tts", "t2w", or "n/a"
};

// Session-level aggregated SLO report
struct SessionSLOReport {
    std::vector<TurnSLOMetrics> turns;

    // TTFA percentiles
    double ttfa_p50_ms = 0.0;
    double ttfa_p95_ms = 0.0;
    double ttfa_p99_ms = 0.0;

    // Inter-WAV interval percentiles (across all turns)
    double interval_p50_ms = 0.0;
    double interval_p95_ms = 0.0;
    double interval_p99_ms = 0.0;

    // E2E latency percentiles
    double e2e_p50_ms = 0.0;
    double e2e_p95_ms = 0.0;
    double e2e_p99_ms = 0.0;

    int         total_stalls       = 0;
    double      avg_smoothness     = 0.0;
    std::string dominant_bottleneck;
};

// Compute WAV cadence metrics from a timestamp series (relative timestamps in ms)
WavCadenceMetrics compute_wav_cadence_metrics(const std::vector<double> & wav_timestamps_ms,
                                              double                      stall_floor_ms = 500.0);

// Compute stage dispatch cadence from timestamp series
StageCadenceMetrics compute_stage_cadence_metrics(const std::vector<double> & dispatch_timestamps_ms);

// Compute per-turn SLO metrics from a single chunk's timing data
TurnSLOMetrics compute_turn_slo_metrics(int chunk_idx, const omni_duplex_chunk_timing & timing,
                                        double stall_floor_ms = 500.0);

// Compute session-level SLO report from all chunk timings (speak turns only)
SessionSLOReport compute_session_slo_report(const std::unordered_map<int, omni_duplex_chunk_timing> & timings,
                                            const std::vector<int> & speak_chunk_indices,
                                            double                   stall_floor_ms = 500.0);

// Compute the Pth percentile of a sorted vector
double compute_percentile(std::vector<double> & sorted_values, double p);

// Print session SLO report to stdout
void print_slo_report(const SessionSLOReport & report);
