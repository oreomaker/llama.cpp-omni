#!/bin/bash
#
# Token2Wav SM Scaling Benchmark Runner
#
# Usage:
#   ./run-t2w-bench.sh <token2wav-gguf-dir> [output_dir]
#
# Example:
#   ./run-t2w-bench.sh ./token2wav-gguf results/

set -euo pipefail

T2W_DIR="${1:?Usage: $0 <token2wav-gguf-dir> [output_dir]}"
OUTDIR="${2:-t2w_bench_results}"

BENCH_BIN="${BENCH_BIN:-llama-omni-bench-t2w}"
BENCH_BIN="${BENCH_BIN:-./llama-omni-bench-t2w}"

STEPS="${STEPS:-50}"
WARMUP="${WARMUP:-5}"
TIMESTEPS="${TIMESTEPS:-10}"
SM_FRACTIONS="${SM_FRACTIONS:-0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.75,1.0}"

mkdir -p "$OUTDIR"

echo "================================================================"
echo " Token2Wav SM Scaling Benchmark"
echo "================================================================"
echo " T2W dir:     $T2W_DIR"
echo " Output dir:  $OUTDIR"
echo " Timesteps:   $TIMESTEPS"
echo " Steps:       $STEPS"
echo " Warmup:      $WARMUP"
echo " SM fractions: $SM_FRACTIONS"
echo "================================================================"
echo ""

# ---- Sanity checks ----
if ! command -v "$BENCH_BIN" &>/dev/null && [ ! -x "$BENCH_BIN" ]; then
    echo "ERROR: benchmark binary not found: $BENCH_BIN"
    echo "  Build with: cmake --build build --target llama-omni-bench-t2w"
    echo "  Or set BENCH_BIN=/path/to/llama-omni-bench-t2w"
    exit 1
fi

if [ ! -f "$T2W_DIR"/encoder.gguf ]; then
    echo "ERROR: GGUF files not found in $T2W_DIR"
    exit 1
fi

# ---- Check MPS ----
MPS_OK=false
MPS_MSG=""

if ! nvidia-cuda-mps-control --version &>/dev/null; then
    MPS_MSG="MPS tools not installed"
elif ! pgrep -x "nvidia-cuda-mps" > /dev/null && ! pgrep -x "nvidia-cuda-mps-c" > /dev/null; then
    MPS_MSG="MPS binary found but daemon NOT running. Start: sudo nvidia-cuda-mps-control -d"
else
    MPS_OK=true
fi

if [ "$MPS_OK" = false ]; then
    echo "=== Single Run (no SM scaling, MPS unavailable) ==="
    echo "Reason: $MPS_MSG"
    echo ""

    "$BENCH_BIN" \
        --t2w-dir "$T2W_DIR" \
        --steps "$STEPS" \
        --warmup "$WARMUP" \
        --timesteps "$TIMESTEPS" \
        2>&1 | tee "$OUTDIR/t2w_single_run.log"

    echo ""
    echo "To enable SM scaling, start MPS daemon and re-run."
    exit 0
fi

echo "MPS daemon running. Starting SM scaling experiment..."
echo ""

# ---- SM Scaling via MPS ----
IFS=',' read -ra FRACS <<< "$SM_FRACTIONS"

echo "SM_Frac ~SMs   Mean(ms)  P95(ms)   Windows/s"
echo "------- ------ --------- --------- ----------"

for frac in "${FRACS[@]}"; do
    pct=$(echo "$frac * 100" | bc | cut -d. -f1)
    pct=$(( pct < 1 ? 1 : pct ))

    OUTPUT=$(
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct" \
        CUDA_VISIBLE_DEVICES=0 \
        "$BENCH_BIN" \
            --t2w-dir "$T2W_DIR" \
            --steps "$STEPS" \
            --warmup "$WARMUP" \
            --timesteps "$TIMESTEPS" \
            2>/dev/null
    )

    # Parse T2W_DATA line
    DATA_LINE=$(echo "$OUTPUT" | grep "^T2W_DATA:" || true)
    if [ -n "$DATA_LINE" ]; then
        # Format: T2W_DATA: fraction mean_ms p95_ms windows_per_sec
        read -r _ f mean_ms p95_ms wps <<< "$DATA_LINE"
        APPROX_SMS=$(echo "$f * $(nvidia-smi --query-gpu=cuda.cores --format=csv,noheader -i 0 2>/dev/null | head -1 | tr -d ' ')" | bc)
        APPROX_SMS=$(( $(nvidia-smi --query-gpu=cuda.cores --format=csv,noheader -i 0 | head -1) / 128 ))
        APPROX_SMS=$(echo "$f * $APPROX_SMS" | bc | cut -d. -f1)
        printf "%-7s %-6s %-9s %-9s %-10s\n" \
            "$f" "$APPROX_SMS" "$mean_ms" "$p95_ms" "$wps"
    else
        printf "%-7s %-6s %s\n" "$frac" "?" "(parse failed)"
    fi
done

echo ""
echo "Results saved to: $OUTDIR/"
echo ""
echo "Interpretation:"
echo "  - T2W saturation SM count = lowest SM fraction within 5% of full-SM performance"
echo "  - Remaining SMs can be allocated to LLM/TTS in mega kernel"
