#!/bin/bash
#
# Mega Kernel SM Partitioning — All Benchmarks Runner
#
# Runs all four component benchmarks in sequence:
#   E1: TTS   — memory-bound verification (batch-size + SM scaling)
#   E2: LLM   — prefill/decode latency + SM scaling
#   E3: Vision — SM scaling (compute-bound verification)
#   E4: T2W   — SM scaling (mixed-bottleneck verification)
#
# Usage:
#   ./run-all-bench.sh \
#       --tts-model   <tts_model.gguf> \
#       --llm-model   <llm_model.gguf> \
#       --vision-model <vision_model.gguf> \
#       --t2w-dir     <token2wav-gguf-dir> \
#       [--output-dir <results_dir>] \
#       [--skip-tts] [--skip-llm] [--skip-vision] [--skip-t2w]
#
# Prerequisites:
#   - All benchmark binaries built: cmake --build build --target llama-omni-bench-{tts,llm,t2w,vision}
#   - For SM scaling experiments: CUDA MPS daemon (sudo nvidia-cuda-mps-control -d)
#
# Output:
#   results/
#   ├── tts/          — TTS batch-size + SM scaling results
#   ├── llm/          — LLM prefill/decode + SM scaling results
#   ├── vision/       — Vision SM scaling results
#   ├── t2w/          — Token2Wav SM scaling results
#   └── summary.txt   — Aggregated results and SM partitioning recommendations

set -euo pipefail

# ---- Parse args ----
TTS_MODEL=""
LLM_MODEL=""
VISION_MODEL=""
T2W_DIR=""
OUTDIR="results"

SKIP_TTS=false
SKIP_LLM=false
SKIP_VISION=false
SKIP_T2W=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tts-model)   TTS_MODEL="$2";   shift 2 ;;
        --llm-model)   LLM_MODEL="$2";   shift 2 ;;
        --vision-model) VISION_MODEL="$2"; shift 2 ;;
        --t2w-dir)     T2W_DIR="$2";     shift 2 ;;
        --output-dir)  OUTDIR="$2";      shift 2 ;;
        --skip-tts)    SKIP_TTS=true;    shift ;;
        --skip-llm)    SKIP_LLM=true;    shift ;;
        --skip-vision) SKIP_VISION=true; shift ;;
        --skip-t2w)    SKIP_T2W=true;    shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --tts-model <gguf> --llm-model <gguf> --vision-model <gguf> --t2w-dir <dir> [--output-dir <dir>]"
            exit 1
            ;;
    esac
done

# ---- Environment defaults ----
STEPS="${STEPS:-50}"
WARMUP="${WARMUP:-10}"
TTS_STEPS="${TTS_STEPS:-200}"
TTS_WARMUP="${TTS_WARMUP:-20}"
SM_FRACTIONS="${SM_FRACTIONS:-0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.75,1.0}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16,32,64}"
PREFILL_LENS="${PREFILL_LENS:-64,128}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
REPEAT="${REPEAT:-3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTDIR"

echo "================================================================"
echo " Mega Kernel SM Partitioning — Full Benchmark Suite"
echo "================================================================"
echo " Output dir:      $OUTDIR"
echo " SM fractions:    $SM_FRACTIONS"
echo ""
echo " Models:"
echo "   TTS:           ${TTS_MODEL:-<skipped>}"
echo "   LLM:           ${LLM_MODEL:-<skipped>}"
echo "   Vision:        ${VISION_MODEL:-<skipped>}"
echo "   T2W dir:       ${T2W_DIR:-<skipped>}"
echo "================================================================"
echo ""

# ---- Check MPS availability (shared) ----
MPS_AVAILABLE=false
if ! nvidia-cuda-mps-control --version &>/dev/null; then
    echo "MPS: not available (nvidia-cuda-mps-control not found)"
elif pgrep -x "nvidia-cuda-mps" > /dev/null || pgrep -x "nvidia-cuda-mps-c" > /dev/null; then
    MPS_AVAILABLE=true
    echo "MPS: daemon running — SM scaling enabled for all components"
else
    echo "MPS: binary found but daemon NOT running. SM scaling will be skipped."
    echo "     Start with: sudo nvidia-cuda-mps-control -d"
fi
echo ""

# ---- Run counter ----
TOTAL=0
PASSED=0

# ============================================================
# E1: TTS Memory-Bound Verification
# ============================================================
run_tts() {
    if [ "$SKIP_TTS" = true ] || [ -z "$TTS_MODEL" ]; then
        echo ">>> TTS: SKIPPED (no model or --skip-tts)"
        return
    fi
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "================================================================"
    echo " E1: TTS Memory-Bound Verification"
    echo "================================================================"
    echo ""

    TTS_OUT="$OUTDIR/tts"
    mkdir -p "$TTS_OUT"

    TTS_BIN="${TTS_BIN:-llama-omni-bench-tts}"

    if ! command -v "$TTS_BIN" &>/dev/null && [ ! -x "$TTS_BIN" ]; then
        echo "ERROR: $TTS_BIN not found. Build: cmake --build build --target llama-omni-bench-tts"
        return
    fi

    # E2: Batch-size scaling
    echo "--- TTS Batch-Size Scaling ---"
    "$TTS_BIN" \
        -m "$TTS_MODEL" \
        --steps "$TTS_STEPS" \
        --warmup "$TTS_WARMUP" \
        --batch-sizes "$BATCH_SIZES" \
        --output "$TTS_OUT/batch_results.csv" \
        2>&1 | tee "$TTS_OUT/batch_scaling.log"

    echo ""
    echo "TTS batch scaling: $TTS_OUT/batch_results.csv"

    # E3: SM scaling (if MPS available)
    if [ "$MPS_AVAILABLE" = true ]; then
        echo ""
        echo "--- TTS SM Scaling ---"

        IFS=',' read -ra FRACS <<< "$SM_FRACTIONS"
        for frac in "${FRACS[@]}"; do
            pct=$(echo "$frac * 100" | bc | cut -d. -f1)
            pct=$(( pct < 1 ? 1 : pct ))
            echo "  SM fraction $frac (${pct}%)..."
            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct" \
            "$TTS_BIN" \
                -m "$TTS_MODEL" \
                --steps "$TTS_STEPS" \
                --warmup "$TTS_WARMUP" \
                --batch-sizes 1 \
                --output /dev/null \
                2>&1 | grep -E "^\s+1\s" || echo "  (parse failed for $frac)"
        done
        echo "TTS SM scaling complete."
    else
        echo "  SM scaling skipped (MPS not available)"
    fi

    PASSED=$((PASSED + 1))
}

# ============================================================
# E2: LLM Prefill/Decode + SM Scaling
# ============================================================
run_llm() {
    if [ "$SKIP_LLM" = true ] || [ -z "$LLM_MODEL" ]; then
        echo ">>> LLM: SKIPPED (no model or --skip-llm)"
        return
    fi
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "================================================================"
    echo " E2: LLM Prefill/Decode + SM Scaling"
    echo "================================================================"
    echo ""

    LLM_OUT="$OUTDIR/llm"
    mkdir -p "$LLM_OUT"

    LLM_BIN="${LLM_BIN:-llama-omni-bench-llm}"

    if ! command -v "$LLM_BIN" &>/dev/null && [ ! -x "$LLM_BIN" ]; then
        echo "ERROR: $LLM_BIN not found. Build: cmake --build build --target llama-omni-bench-llm"
        return
    fi

    # E1+E2: Analytical + Prefill/Decode latency
    echo "--- LLM Analytical + Prefill/Decode Latency ---"
    "$LLM_BIN" \
        -m "$LLM_MODEL" \
        --steps "$STEPS" \
        --warmup "$WARMUP" \
        --prefill-lens "$PREFILL_LENS" \
        --decode-tokens "$DECODE_TOKENS" \
        --repeat "$REPEAT" \
        --output "$LLM_OUT/results.csv" \
        2>&1 | tee "$LLM_OUT/full_bench.log"

    # E3: SM scaling
    if [ "$MPS_AVAILABLE" = true ]; then
        echo ""
        echo "--- LLM SM Scaling ---"
        "$LLM_BIN" \
            -m "$LLM_MODEL" \
            --steps "$STEPS" \
            --warmup "$WARMUP" \
            --prefill-lens "$PREFILL_LENS" \
            --decode-tokens "$DECODE_TOKENS" \
            --repeat 1 \
            --sm-fractions "$SM_FRACTIONS" \
            --output "$LLM_OUT/sm_scaling.csv" \
            2>&1 | tee "$LLM_OUT/sm_scaling.log"
        echo "LLM SM scaling: $LLM_OUT/sm_scaling.csv"
    else
        echo "  SM scaling skipped (MPS not available)"
    fi

    PASSED=$((PASSED + 1))
}

# ============================================================
# E3: Vision Encoder SM Scaling
# ============================================================
run_vision() {
    if [ "$SKIP_VISION" = true ] || [ -z "$VISION_MODEL" ]; then
        echo ">>> Vision: SKIPPED (no model or --skip-vision)"
        return
    fi
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "================================================================"
    echo " E3: Vision Encoder SM Scaling (Compute-Bound Verification)"
    echo "================================================================"
    echo ""

    VISION_OUT="$OUTDIR/vision"
    mkdir -p "$VISION_OUT"

    VISION_BIN="${VISION_BIN:-llama-omni-bench-vision}"

    if ! command -v "$VISION_BIN" &>/dev/null && [ ! -x "$VISION_BIN" ]; then
        echo "ERROR: $VISION_BIN not found. Build: cmake --build build --target llama-omni-bench-vision"
        return
    fi

    # Single run (always)
    echo "--- Vision Baseline ---"
    "$VISION_BIN" \
        -m "$VISION_MODEL" \
        --steps "$STEPS" \
        --warmup "$WARMUP" \
        2>&1 | tee "$VISION_OUT/baseline.log"

    # SM scaling (if MPS available)
    if [ "$MPS_AVAILABLE" = true ]; then
        echo ""
        echo "--- Vision SM Scaling ---"
        "$VISION_BIN" \
            -m "$VISION_MODEL" \
            --steps "$STEPS" \
            --warmup "$WARMUP" \
            --sm-fractions "$SM_FRACTIONS" \
            2>&1 | tee "$VISION_OUT/sm_scaling.log"
        echo "Vision SM scaling: $VISION_OUT/sm_scaling.log"
    else
        echo "  SM scaling skipped (MPS not available)"
    fi

    PASSED=$((PASSED + 1))
}

# ============================================================
# E4: Token2Wav SM Scaling
# ============================================================
run_t2w() {
    if [ "$SKIP_T2W" = true ] || [ -z "$T2W_DIR" ]; then
        echo ">>> T2W: SKIPPED (no model dir or --skip-t2w)"
        return
    fi
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "================================================================"
    echo " E4: Token2Wav SM Scaling"
    echo "================================================================"
    echo ""

    T2W_OUT="$OUTDIR/t2w"
    mkdir -p "$T2W_OUT"

    T2W_BIN="${T2W_BIN:-llama-omni-bench-t2w}"

    if ! command -v "$T2W_BIN" &>/dev/null && [ ! -x "$T2W_BIN" ]; then
        echo "ERROR: $T2W_BIN not found. Build: cmake --build build --target llama-omni-bench-t2w"
        return
    fi

    if [ ! -f "$T2W_DIR"/encoder.gguf ]; then
        echo "ERROR: GGUF files not found in $T2W_DIR (expected encoder.gguf)"
        return
    fi

    # Single run
    echo "--- T2W Baseline ---"
    TIMESTEPS="${TIMESTEPS:-10}"
    "$T2W_BIN" \
        --t2w-dir "$T2W_DIR" \
        --steps "$STEPS" \
        --warmup "$WARMUP" \
        --timesteps "$TIMESTEPS" \
        2>&1 | tee "$T2W_OUT/baseline.log"

    # SM scaling (if MPS available)
    if [ "$MPS_AVAILABLE" = true ]; then
        echo ""
        echo "--- T2W SM Scaling ---"

        IFS=',' read -ra FRACS <<< "$SM_FRACTIONS"
        echo "SM_Frac  ~SMs    Mean(ms)   P95(ms)    Windows/s"
        echo "-------  ------  ---------  ---------  ----------"

        for frac in "${FRACS[@]}"; do
            pct=$(echo "$frac * 100" | bc | cut -d. -f1)
            pct=$(( pct < 1 ? 1 : pct ))

            OUTPUT=$(
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct" \
                CUDA_VISIBLE_DEVICES=0 \
                "$T2W_BIN" \
                    --t2w-dir "$T2W_DIR" \
                    --steps "$STEPS" \
                    --warmup "$WARMUP" \
                    --timesteps "$TIMESTEPS" \
                    2>/dev/null
            )

            DATA_LINE=$(echo "$OUTPUT" | grep "^T2W_DATA:" || true)
            if [ -n "$DATA_LINE" ]; then
                read -r _ f mean_ms p95_ms wps <<< "$DATA_LINE"
                APPROX_SMS=$(echo "$f * 128" | bc | cut -d. -f1)
                printf "%-7s  %-6s  %-9s  %-9s  %-10s\n" \
                    "$f" "$APPROX_SMS" "$mean_ms" "$p95_ms" "$wps"
            else
                printf "%-7s  %-6s  %s\n" "$frac" "?" "(parse failed)"
            fi
        done
        echo ""
    else
        echo "  SM scaling skipped (MPS not available)"
    fi

    PASSED=$((PASSED + 1))
}

# ---- Run all ----
run_tts
run_llm
run_vision
run_t2w

# ---- Summary ----
echo ""
echo "================================================================"
echo " RESULTS SAVED TO: $OUTDIR/"
echo "================================================================"
echo ""
echo "Directory layout:"
[ -d "$OUTDIR/tts"    ] && echo "  tts/       — TTS batch-size + SM scaling results"
[ -d "$OUTDIR/llm"    ] && echo "  llm/       — LLM prefill/decode + SM scaling results"
[ -d "$OUTDIR/vision" ] && echo "  vision/    — Vision encoder SM scaling results"
[ -d "$OUTDIR/t2w"    ] && echo "  t2w/       — Token2Wav SM scaling results"

# Write summary file
SUMMARY="$OUTDIR/summary.txt"
{
    echo "Mega Kernel SM Partitioning — Benchmark Summary"
    echo "================================================"
    echo "Date: $(date)"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
    echo "MPS: $MPS_AVAILABLE"
    echo "SM Fractions: $SM_FRACTIONS"
    echo ""
    echo "Components run: $PASSED/$TOTAL"
    echo ""
    echo "Expected SM partitioning (from experiments):"
    echo "  TTS:    ~28 SM (22%) — memory-bound, saturates at ~15-20%"
    echo "  LLM:    ~100 SM (78%) — memory-bound, needs most bandwidth"
    echo "  Vision: full SM (yield) — compute-bound, benefits from all SMs"
    echo "  T2W:    full SM (yield) — mixed bottleneck, independent launch"
} > "$SUMMARY"

echo ""
echo "Summary written to: $SUMMARY"
echo ""
echo "Key takeaways for mega-kernel SM partitioning:"
echo "  - TTS saturates at ~20 SM (15-20%), allocate ~28 SM for safety"
echo "  - LLM needs remaining ~100 SM (78%) for memory bandwidth"
echo "  - Vision & T2W get full SM via yield (independent launches)"
echo ""
echo "See docs/megakernel-sm-partitioning-experiment.md for full analysis."
