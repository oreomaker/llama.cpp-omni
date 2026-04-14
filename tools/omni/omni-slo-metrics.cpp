#include "omni-slo-metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<double> compute_intervals(const std::vector<double> & timestamps_ms) {
    std::vector<double> intervals;
    if (timestamps_ms.size() < 2) {
        return intervals;
    }
    intervals.reserve(timestamps_ms.size() - 1);
    for (size_t i = 1; i < timestamps_ms.size(); ++i) {
        intervals.push_back(timestamps_ms[i] - timestamps_ms[i - 1]);
    }
    return intervals;
}

static double compute_mean(const std::vector<double> & v) {
    if (v.empty()) {
        return 0.0;
    }
    return std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size();
}

static double compute_stdev(const std::vector<double> & v, double mean) {
    if (v.size() < 2) {
        return 0.0;
    }
    double sum_sq = 0.0;
    for (double x : v) {
        const double diff = x - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / (double) (v.size() - 1));
}

double compute_percentile(std::vector<double> & sorted_values, double p) {
    if (sorted_values.empty()) {
        return 0.0;
    }
    std::sort(sorted_values.begin(), sorted_values.end());
    if (sorted_values.size() == 1) {
        return sorted_values[0];
    }
    const double rank = (p / 100.0) * (double) (sorted_values.size() - 1);
    const size_t lo   = (size_t) std::floor(rank);
    const size_t hi   = std::min(lo + 1, sorted_values.size() - 1);
    const double frac = rank - (double) lo;
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo]);
}

// ---------------------------------------------------------------------------
// WAV Cadence
// ---------------------------------------------------------------------------

WavCadenceMetrics compute_wav_cadence_metrics(const std::vector<double> & wav_timestamps_ms,
                                              double                      stall_floor_ms) {
    WavCadenceMetrics m;
    m.wav_count = (int) wav_timestamps_ms.size();

    m.inter_wav_intervals_ms = compute_intervals(wav_timestamps_ms);
    if (m.inter_wav_intervals_ms.empty()) {
        m.smoothness_score = 1.0;
        return m;
    }

    m.mean_interval_ms  = compute_mean(m.inter_wav_intervals_ms);
    m.stdev_interval_ms = compute_stdev(m.inter_wav_intervals_ms, m.mean_interval_ms);

    m.max_interval_ms = *std::max_element(m.inter_wav_intervals_ms.begin(), m.inter_wav_intervals_ms.end());
    m.min_interval_ms = *std::min_element(m.inter_wav_intervals_ms.begin(), m.inter_wav_intervals_ms.end());

    // Stall detection: interval > max(2 * mean, stall_floor)
    const double stall_threshold = std::max(2.0 * m.mean_interval_ms, stall_floor_ms);
    m.stall_count                = 0;
    for (double interval : m.inter_wav_intervals_ms) {
        if (interval > stall_threshold) {
            m.stall_count++;
        }
    }

    // Smoothness: 1 - CV (coefficient of variation), clamped [0, 1]
    if (m.mean_interval_ms > 0.0) {
        const double cv    = m.stdev_interval_ms / m.mean_interval_ms;
        m.smoothness_score = std::max(0.0, std::min(1.0, 1.0 - cv));
    } else {
        m.smoothness_score = 1.0;
    }

    return m;
}

// ---------------------------------------------------------------------------
// Stage Cadence
// ---------------------------------------------------------------------------

StageCadenceMetrics compute_stage_cadence_metrics(const std::vector<double> & dispatch_timestamps_ms) {
    StageCadenceMetrics m;
    m.dispatch_count = (int) dispatch_timestamps_ms.size();

    m.dispatch_intervals_ms = compute_intervals(dispatch_timestamps_ms);
    if (m.dispatch_intervals_ms.empty()) {
        return m;
    }

    const double mean = compute_mean(m.dispatch_intervals_ms);
    m.stdev_ms        = compute_stdev(m.dispatch_intervals_ms, mean);

    // Throughput: dispatches per second (based on total time span)
    if (dispatch_timestamps_ms.size() >= 2) {
        const double span_ms = dispatch_timestamps_ms.back() - dispatch_timestamps_ms.front();
        if (span_ms > 0.0) {
            m.tokens_per_sec = (double) (dispatch_timestamps_ms.size() - 1) / (span_ms / 1000.0);
        }
    }

    return m;
}

// ---------------------------------------------------------------------------
// Per-Turn SLO
// ---------------------------------------------------------------------------

TurnSLOMetrics compute_turn_slo_metrics(int chunk_idx, const omni_duplex_chunk_timing & timing,
                                        double stall_floor_ms) {
    TurnSLOMetrics m;
    m.chunk_idx      = chunk_idx;
    m.ttfa_ms        = timing.ttfa_ms;
    m.e2e_latency_ms = timing.e2e_latency_ms;

    // WAV cadence
    m.wav_cadence = compute_wav_cadence_metrics(timing.wav_timestamps_ms, stall_floor_ms);

    // LLM dispatch cadence
    m.llm_cadence = compute_stage_cadence_metrics(timing.llm_dispatch_timestamps_ms);

    // TTS dispatch cadence
    m.tts_cadence = compute_stage_cadence_metrics(timing.tts_dispatch_timestamps_ms);

    // Jitter amplification factor: WAV stdev / LLM stdev
    if (m.llm_cadence.stdev_ms > 0.0 && !m.wav_cadence.inter_wav_intervals_ms.empty()) {
        m.jitter_amplification_factor = m.wav_cadence.stdev_interval_ms / m.llm_cadence.stdev_ms;
    }

    // Bottleneck: compare total stage times
    // Use existing per-stage durations from the timing struct
    double tts_total  = std::max(timing.tts_audio_token_ms, 0.0);
    double t2w_total  = std::max(timing.token2wav_ms, 0.0);

    // Estimate LLM decode contribution from the gap between chunk_start and first LLM dispatch
    double llm_decode_est = 0.0;
    if (!timing.llm_dispatch_timestamps_ms.empty()) {
        llm_decode_est = timing.llm_dispatch_timestamps_ms.back();
        // Subtract encode time to isolate decode
        if (timing.audio_embedding_ms > 0.0) {
            llm_decode_est -= timing.audio_embedding_ms;
        }
        if (timing.vit_embedding_ms > 0.0) {
            llm_decode_est -= timing.vit_embedding_ms;
        }
        llm_decode_est = std::max(llm_decode_est, 0.0);
    }

    if (llm_decode_est >= tts_total && llm_decode_est >= t2w_total) {
        m.bottleneck_stage = "llm";
    } else if (tts_total >= llm_decode_est && tts_total >= t2w_total) {
        m.bottleneck_stage = "tts";
    } else if (t2w_total > 0.0) {
        m.bottleneck_stage = "t2w";
    } else {
        m.bottleneck_stage = "n/a";
    }

    return m;
}

// ---------------------------------------------------------------------------
// Session-Level Report
// ---------------------------------------------------------------------------

SessionSLOReport compute_session_slo_report(const std::unordered_map<int, omni_duplex_chunk_timing> & timings,
                                            const std::vector<int> & speak_chunk_indices,
                                            double                   stall_floor_ms) {
    SessionSLOReport report;

    std::vector<double> all_ttfa;
    std::vector<double> all_intervals;
    std::vector<double> all_e2e;
    double              total_smoothness = 0.0;
    int                 smoothness_count = 0;
    int                 bottleneck_llm   = 0;
    int                 bottleneck_tts   = 0;
    int                 bottleneck_t2w   = 0;

    for (int idx : speak_chunk_indices) {
        auto it = timings.find(idx);
        if (it == timings.end()) {
            continue;
        }

        TurnSLOMetrics turn = compute_turn_slo_metrics(idx, it->second, stall_floor_ms);
        report.turns.push_back(turn);

        if (turn.ttfa_ms > 0.0) {
            all_ttfa.push_back(turn.ttfa_ms);
        }
        if (turn.e2e_latency_ms > 0.0) {
            all_e2e.push_back(turn.e2e_latency_ms);
        }

        for (double iv : turn.wav_cadence.inter_wav_intervals_ms) {
            all_intervals.push_back(iv);
        }

        report.total_stalls += turn.wav_cadence.stall_count;

        if (turn.wav_cadence.wav_count > 1) {
            total_smoothness += turn.wav_cadence.smoothness_score;
            smoothness_count++;
        }

        if (turn.bottleneck_stage == "llm") {
            bottleneck_llm++;
        } else if (turn.bottleneck_stage == "tts") {
            bottleneck_tts++;
        } else if (turn.bottleneck_stage == "t2w") {
            bottleneck_t2w++;
        }
    }

    // TTFA percentiles
    if (!all_ttfa.empty()) {
        report.ttfa_p50_ms = compute_percentile(all_ttfa, 50.0);
        report.ttfa_p95_ms = compute_percentile(all_ttfa, 95.0);
        report.ttfa_p99_ms = compute_percentile(all_ttfa, 99.0);
    }

    // Inter-WAV interval percentiles
    if (!all_intervals.empty()) {
        report.interval_p50_ms = compute_percentile(all_intervals, 50.0);
        report.interval_p95_ms = compute_percentile(all_intervals, 95.0);
        report.interval_p99_ms = compute_percentile(all_intervals, 99.0);
    }

    // E2E percentiles
    if (!all_e2e.empty()) {
        report.e2e_p50_ms = compute_percentile(all_e2e, 50.0);
        report.e2e_p95_ms = compute_percentile(all_e2e, 95.0);
        report.e2e_p99_ms = compute_percentile(all_e2e, 99.0);
    }

    // Average smoothness
    report.avg_smoothness = smoothness_count > 0 ? total_smoothness / (double) smoothness_count : 1.0;

    // Dominant bottleneck
    if (bottleneck_llm >= bottleneck_tts && bottleneck_llm >= bottleneck_t2w) {
        report.dominant_bottleneck = "llm";
    } else if (bottleneck_tts >= bottleneck_llm && bottleneck_tts >= bottleneck_t2w) {
        report.dominant_bottleneck = "tts";
    } else {
        report.dominant_bottleneck = "t2w";
    }

    return report;
}

// ---------------------------------------------------------------------------
// Report Printing
// ---------------------------------------------------------------------------

static const char * fmt_ms(double ms, char * buf, size_t buf_size) {
    if (ms < 0.0) {
        snprintf(buf, buf_size, "n/a");
    } else {
        snprintf(buf, buf_size, "%.1f ms", ms);
    }
    return buf;
}

void print_slo_report(const SessionSLOReport & report) {
    char b1[64], b2[64], b3[64];

    printf("\n========================================\n");
    printf("  SLO Metrics Report\n");
    printf("========================================\n");

    // TTFA
    printf("\n  TTFA (Time to First Audio):\n");
    if (!report.turns.empty()) {
        printf("    P50: %s | P95: %s | P99: %s\n",
               fmt_ms(report.ttfa_p50_ms, b1, sizeof(b1)),
               fmt_ms(report.ttfa_p95_ms, b2, sizeof(b2)),
               fmt_ms(report.ttfa_p99_ms, b3, sizeof(b3)));
        printf("    Per-turn:");
        for (const auto & t : report.turns) {
            if (t.ttfa_ms > 0.0) {
                printf(" [chunk %d: %.1fms]", t.chunk_idx, t.ttfa_ms);
            }
        }
        printf("\n");
    } else {
        printf("    No speak turns recorded.\n");
    }

    // Streaming Continuity
    printf("\n  Streaming Continuity:\n");
    bool has_intervals = false;
    for (const auto & t : report.turns) {
        if (!t.wav_cadence.inter_wav_intervals_ms.empty()) {
            has_intervals = true;
            break;
        }
    }
    if (has_intervals) {
        // Aggregate interval stats across all turns
        double total_mean = 0.0, total_stdev = 0.0, total_max = 0.0, total_min = 1e18;
        int    count = 0;
        for (const auto & t : report.turns) {
            if (!t.wav_cadence.inter_wav_intervals_ms.empty()) {
                total_mean += t.wav_cadence.mean_interval_ms;
                total_stdev += t.wav_cadence.stdev_interval_ms;
                if (t.wav_cadence.max_interval_ms > total_max) {
                    total_max = t.wav_cadence.max_interval_ms;
                }
                if (t.wav_cadence.min_interval_ms < total_min) {
                    total_min = t.wav_cadence.min_interval_ms;
                }
                count++;
            }
        }
        if (count > 0) {
            printf("    Mean inter-WAV interval: %.1f ms\n", total_mean / count);
            printf("    Stdev: %.1f ms | Max: %.1f ms | Min: %.1f ms\n",
                   total_stdev / count, total_max, total_min);
        }
        printf("    P50: %s | P95: %s | P99: %s\n",
               fmt_ms(report.interval_p50_ms, b1, sizeof(b1)),
               fmt_ms(report.interval_p95_ms, b2, sizeof(b2)),
               fmt_ms(report.interval_p99_ms, b3, sizeof(b3)));
        printf("    Stalls (>2x mean): %d\n", report.total_stalls);
        printf("    Smoothness score: %.3f\n", report.avg_smoothness);
    } else {
        printf("    No WAV output data (TTS may be disabled or no speak turns).\n");
    }

    // Two-Stage Cascade Analysis
    printf("\n  Two-Stage Cascade Analysis:\n");
    bool has_cascade = false;
    for (const auto & t : report.turns) {
        if (t.llm_cadence.dispatch_count > 1 || t.tts_cadence.dispatch_count > 1) {
            has_cascade = true;
            break;
        }
    }
    if (has_cascade) {
        double avg_llm_stdev = 0.0, avg_tts_stdev = 0.0;
        double avg_llm_tps = 0.0, avg_tts_tps = 0.0;
        double avg_jaf = 0.0;
        int    llm_count = 0, tts_count = 0, jaf_count = 0;

        for (const auto & t : report.turns) {
            if (t.llm_cadence.dispatch_count > 1) {
                avg_llm_stdev += t.llm_cadence.stdev_ms;
                avg_llm_tps += t.llm_cadence.tokens_per_sec;
                llm_count++;
            }
            if (t.tts_cadence.dispatch_count > 1) {
                avg_tts_stdev += t.tts_cadence.stdev_ms;
                avg_tts_tps += t.tts_cadence.tokens_per_sec;
                tts_count++;
            }
            if (t.jitter_amplification_factor > 0.0) {
                avg_jaf += t.jitter_amplification_factor;
                jaf_count++;
            }
        }

        if (llm_count > 0) {
            printf("    LLM dispatch jitter (stdev): %.1f ms | throughput: %.1f dispatch/s\n",
                   avg_llm_stdev / llm_count, avg_llm_tps / llm_count);
        }
        if (tts_count > 0) {
            printf("    TTS dispatch jitter (stdev): %.1f ms | throughput: %.1f dispatch/s\n",
                   avg_tts_stdev / tts_count, avg_tts_tps / tts_count);
        }
        if (jaf_count > 0) {
            const double jaf = avg_jaf / jaf_count;
            printf("    Jitter amplification factor: %.2f (%s)\n", jaf,
                   jaf > 1.0 ? "downstream amplifies jitter" : "queues dampen jitter");
        }
    }

    // Bottleneck distribution
    {
        int bn_llm = 0, bn_tts = 0, bn_t2w = 0;
        for (const auto & t : report.turns) {
            if (t.bottleneck_stage == "llm") bn_llm++;
            else if (t.bottleneck_stage == "tts") bn_tts++;
            else if (t.bottleneck_stage == "t2w") bn_t2w++;
        }
        const int total = (int) report.turns.size();
        if (total > 0) {
            printf("    Bottleneck: %s (%d/%d turns)",
                   report.dominant_bottleneck.c_str(),
                   report.dominant_bottleneck == "llm" ? bn_llm :
                   report.dominant_bottleneck == "tts" ? bn_tts : bn_t2w,
                   total);
            if (bn_llm > 0 && report.dominant_bottleneck != "llm") {
                printf(" | llm: %d/%d", bn_llm, total);
            }
            if (bn_tts > 0 && report.dominant_bottleneck != "tts") {
                printf(" | tts: %d/%d", bn_tts, total);
            }
            if (bn_t2w > 0 && report.dominant_bottleneck != "t2w") {
                printf(" | t2w: %d/%d", bn_t2w, total);
            }
            printf("\n");
        }
    }

    // E2E Latency
    printf("\n  E2E Latency:\n");
    if (!report.turns.empty()) {
        printf("    P50: %s | P95: %s | P99: %s\n",
               fmt_ms(report.e2e_p50_ms, b1, sizeof(b1)),
               fmt_ms(report.e2e_p95_ms, b2, sizeof(b2)),
               fmt_ms(report.e2e_p99_ms, b3, sizeof(b3)));
    }

    printf("\n========================================\n");
}
