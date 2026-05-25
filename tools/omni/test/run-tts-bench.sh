#!/bin/bash
#
# TTS Memory-Bound Verification — Automated Experiment Runner
#
# Runs batch-size scaling and SM scaling experiments to verify the hypothesis
# that TTS decode is memory-bandwidth-bound and to determine the minimum SM
# count needed for TTS compute.
#
# Usage:
#   ./run-tts-bench.sh <tts_model.gguf> [output_dir]
#
# Prerequisites:
#   - llama-omni-bench-tts binary built and in PATH or same directory
#   - For SM scaling: CUDA MPS control daemon (nvidia-cuda-mps-control)
#
# Output:
#   - Console: formatted results with interpretation
#   - output_dir/results.csv: raw data
#   - output_dir/summary.txt: final report

set -euo pipefail

MODEL="${1:?Usage: $0 <tts_model.gguf> [output_dir]}"
OUTDIR="${2:-tts_bench_results}"

BENCH_BIN="${BENCH_BIN:-llama-omni-bench-tts}"
BENCH_BIN="${BENCH_BIN:-./llama-omni-bench-tts}"

STEPS="${STEPS:-200}"
WARMUP="${WARMUP:-20}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16,32,64}"
SM_FRACTIONS="${SM_FRACTIONS:-0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.75,1.0}"

mkdir -p "$OUTDIR"

echo "================================================================"
echo " TTS Memory-Bound Hypothesis Verification"
echo "================================================================"
echo " Model:       $MODEL"
echo " Output dir:  $OUTDIR"
echo " Steps:       $STEPS"
echo " Warmup:      $WARMUP"
echo " Batch sizes: $BATCH_SIZES"
echo " SM fractions: $SM_FRACTIONS"
echo "================================================================"
echo ""

# ---- Sanity checks ----
if ! command -v "$BENCH_BIN" &>/dev/null && [ ! -x "$BENCH_BIN" ]; then
    echo "ERROR: benchmark binary not found: $BENCH_BIN"
    echo "  Build with: cmake --build build --target llama-omni-bench-tts"
    echo "  Or set BENCH_BIN=/path/to/llama-omni-bench-tts"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL"
    exit 1
fi

# ---- Experiment 2: Batch-Size Scaling (always runs) ----
echo ""
echo "=== Experiment 2: Batch-Size Scaling ==="
echo ""

"$BENCH_BIN" \
    -m "$MODEL" \
    --steps "$STEPS" \
    --warmup "$WARMUP" \
    --batch-sizes "$BATCH_SIZES" \
    --output "$OUTDIR/batch_results.csv" \
    2>&1 | tee "$OUTDIR/batch_scaling.log"

echo ""
echo "Batch scaling results saved to: $OUTDIR/batch_results.csv"
echo ""

# ---- Experiment 3: SM Scaling (MPS required) ----
echo ""
echo "=== Experiment 3: SM Scaling ==="
echo ""

MPS_AVAILABLE=false
MPS_REASON=""

if ! nvidia-cuda-mps-control --version &>/dev/null; then
    MPS_REASON="nvidia-cuda-mps-control not found (CUDA MPS tools not installed)"
else
    # Check if MPS daemon is running (look for the daemon process)
    if pgrep -x "nvidia-cuda-mps" > /dev/null; then
        MPS_AVAILABLE=true
    elif pgrep -x "nvidia-cuda-mps-c" > /dev/null; then
        MPS_AVAILABLE=true
    else
        MPS_REASON="MPS binary found but daemon is NOT running. Start it with:  sudo nvidia-cuda-mps-control -d"
    fi
fi

if [ "$MPS_AVAILABLE" = true ]; then
    echo "MPS daemon detected. Running SM scaling experiment..."
    echo ""

    # Run with each SM fraction via MPS
    IFS=',' read -ra FRACS <<< "$SM_FRACTIONS"
    for frac in "${FRACS[@]}"; do
        pct=$(echo "$frac * 100" | bc | cut -d. -f1)
        pct=$(( pct < 1 ? 1 : pct ))

        echo "--- SM fraction: $frac (${pct}%) ---"

        # Re-exec with MPS limit
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct" \
        "$BENCH_BIN" \
            -m "$MODEL" \
            --steps "$STEPS" \
            --warmup "$WARMUP" \
            --batch-sizes 1 \
            --output /dev/null \
            2>&1 | grep -E "^\s+1\s" || echo "  (failed to parse output)"

        echo ""
    done

    echo "SM scaling experiment complete."
else
    echo "MPS not available. Reason: $MPS_REASON"
    echo ""
    echo "To enable SM scaling:"
    echo "  1. Start MPS daemon:  sudo nvidia-cuda-mps-control -d"
    echo "  2. Verify:            ps aux | grep nvidia-cuda-mps"
    echo "  3. Re-run this script"
    echo ""
    echo "Alternatively, use batch-size scaling results (Experiment 2)"
    echo "to estimate SM requirements — the bandwidth utilization at"
    echo "batch=1 directly indicates the fraction of GPU needed."
fi

# ---- Report ----
echo ""
echo "================================================================"
echo " RESULTS SAVED TO: $OUTDIR/"
echo "================================================================"
echo ""
echo "Files:"
echo "  batch_results.csv    — Batch-size scaling data"
echo "  batch_scaling.log    — Full console output"
echo ""
echo "To interpret results:"
echo "  - Compare batch=1 bandwidth utilization vs peak"
echo "  - Low utilization at batch=1 → memory-bound"
echo "  - bandwith utilization at batch=1 ≈ SM fraction needed for TTS"
echo ""
echo "For mega-kernel SM partitioning:"
echo "  - TTS needs ~(batch=1 BW util %) of total SMs"
echo "  - Remaining SMs available for LLM in mega kernel"
