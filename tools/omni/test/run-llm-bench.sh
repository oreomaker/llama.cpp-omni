#!/bin/bash
#
# LLM SM Partitioning Benchmark — Automated Runner
#
# Runs prefill+decode latency measurement and SM scaling experiments
# to determine how LLM performance changes with SM count for mega-kernel
# SM partitioning design.
#
# Usage:
#   ./run-llm-bench.sh <llm_model.gguf> [output_dir]
#
# Prerequisites:
#   - llama-omni-bench-llm binary built and in PATH or same directory
#   - For SM scaling: CUDA MPS control daemon (nvidia-cuda-mps-control)
#
# Output:
#   - Console: formatted results with interpretation
#   - output_dir/results.csv: raw data
#   - output_dir/full_bench.log: full console output

set -euo pipefail

MODEL="${1:?Usage: $0 <llm_model.gguf> [output_dir]}"
OUTDIR="${2:-llm_bench_results}"

BENCH_BIN="${BENCH_BIN:-llama-omni-bench-llm}"

STEPS="${STEPS:-50}"
WARMUP="${WARMUP:-10}"
PREFILL_LENS="${PREFILL_LENS:-64,128}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
REPEAT="${REPEAT:-3}"
SM_FRACTIONS="${SM_FRACTIONS:-0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.75,1.0}"

mkdir -p "$OUTDIR"

echo "================================================================"
echo " LLM SM Partitioning Benchmark"
echo "================================================================"
echo " Model:       $MODEL"
echo " Output dir:  $OUTDIR"
echo " Steps:       $STEPS"
echo " Warmup:      $WARMUP"
echo " Prefill:     $PREFILL_LENS"
echo " Decode tok:  $DECODE_TOKENS"
echo " Repeat:      $REPEAT"
echo " SM fractions: $SM_FRACTIONS"
echo "================================================================"
echo ""

# ---- Sanity checks ----
if ! command -v "$BENCH_BIN" &>/dev/null && [ ! -x "$BENCH_BIN" ]; then
    echo "ERROR: benchmark binary not found: $BENCH_BIN"
    echo "  Build with: cmake --build build --target llama-omni-bench-llm"
    echo "  Or set BENCH_BIN=/path/to/llama-omni-bench-llm"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL"
    exit 1
fi

# ---- Experiment 1+2: Analytical + Prefill/Decode Latency (always runs) ----
echo ""
echo "=== Experiment 1+2: Analytical + Prefill/Decode Latency ==="
echo ""

"$BENCH_BIN" \
    -m "$MODEL" \
    --steps "$STEPS" \
    --warmup "$WARMUP" \
    --prefill-lens "$PREFILL_LENS" \
    --decode-tokens "$DECODE_TOKENS" \
    --repeat "$REPEAT" \
    --output "$OUTDIR/results.csv" \
    2>&1 | tee "$OUTDIR/full_bench.log"

echo ""
echo "E1+E2 results saved to: $OUTDIR/full_bench.log"
echo "CSV data saved to: $OUTDIR/results.csv"
echo ""

# ---- Experiment 3: SM Scaling (MPS required) ----
echo ""
echo "=== Experiment 3: SM Scaling ==="
echo ""

MPS_AVAILABLE=false

if ! nvidia-cuda-mps-control --version &>/dev/null; then
    echo "MPS not available: nvidia-cuda-mps-control not found"
else
    if pgrep -x "nvidia-cuda-mps" > /dev/null || pgrep -x "nvidia-cuda-mps-c" > /dev/null; then
        MPS_AVAILABLE=true
    else
        echo "MPS binary found but daemon is NOT running."
        echo "Start it with:  sudo nvidia-cuda-mps-control -d"
    fi
fi

if [ "$MPS_AVAILABLE" = true ]; then
    echo "MPS daemon detected. Running SM scaling experiment..."
    echo ""

    # Run SM scaling via re-exec
    "$BENCH_BIN" \
        -m "$MODEL" \
        --steps "$STEPS" \
        --warmup "$WARMUP" \
        --prefill-lens "$PREFILL_LENS" \
        --decode-tokens "$DECODE_TOKENS" \
        --repeat 1 \
        --sm-fractions "$SM_FRACTIONS" \
        --output "$OUTDIR/sm_scaling.csv" \
        2>&1 | tee "$OUTDIR/sm_scaling.log"

    echo ""
    echo "SM scaling results saved to: $OUTDIR/sm_scaling.log"
    echo "CSV data saved to: $OUTDIR/sm_scaling.csv"
else
    echo "SM scaling skipped — MPS not available."
    echo ""
    echo "To enable SM scaling:"
    echo "  1. Start MPS daemon:  sudo nvidia-cuda-mps-control -d"
    echo "  2. Verify:            ps aux | grep nvidia-cuda-mps"
    echo "  3. Re-run this script"
    echo ""
    echo "Alternatively, the decode BW utilization from Experiment 2"
    echo "directly indicates the relative SM requirement for LLM."
fi

# ---- Report ----
echo ""
echo "================================================================"
echo " RESULTS SAVED TO: $OUTDIR/"
echo "================================================================"
echo ""
echo "Files:"
echo "  full_bench.log    — Analytical + Prefill/Decode latency output"
echo "  results.csv       — Prefill/Decode latency CSV"
echo "  sm_scaling.log    — SM scaling output (if MPS available)"
echo "  sm_scaling.csv    — SM scaling CSV (if MPS available)"
echo ""
echo "Interpretation:"
echo "  - Compare BW utilization at decode: low = memory-bound, needs more SMs"
echo "  - SM scaling knee point = minimum SM fraction for near-optimal LLM perf"
echo "  - Unlike TTS which saturates at ~15% SM, LLM typically needs 50-80% SMs"
echo ""
echo "For mega-kernel SM partitioning:"
echo "  - LLM decode is memory-bound (large model = large memory traffic)"
echo "  - LLM needs significant SM allocation to saturate memory bandwidth"
echo "  - TTS needs only ~15-22% SMs → allocate ~78% to LLM"
