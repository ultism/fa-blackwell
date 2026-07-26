#!/usr/bin/env bash
# 5090 evidence collection: fa2 (flashinfer wrapper) vs s3 mxfp8 operator.
# Per SHAPE x SIDE captures:
#   1) event-timed avg-of-100 (x5 alternating A/B rounds, report min) -> prof/5090/events_<shape>.log
#   2) one ncu --set full trace of a steady-state launch -> tensor-pipe %, gpu__time_duration, SOL
#
# Shapes are "name|Hq|Hkv|comma-lens" below; both sides run the SAME batch.
# Usage:  bash prof/trace_5090_ab.sh [events|ncu]
# Env overrides: PY=/path/python NCU=/path/ncu NVCC=/path/nvcc FI_SRC=/path/flashinfer
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (prof/ -> ..)

NCU=${NCU:-/usr/local/cuda/bin/ncu}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
PY=${PY:-python3}
export FI_SRC=${FI_SRC:-/root/fa-blackwell/flashinfer}
export PYTHONPATH="$FI_SRC:${PYTHONPATH:-}"
BIN=tests/bench_ragged_s9f
SRC=tests/bench_ragged.cu
OUT=prof/5090
mkdir -p "$OUT"

DEV_LENS="512,1024,768,1536,2048,640,1280,896,384,1792,512,2560,1024,768,2048,1100"
SHAPES=(
  "dev8x2|8|2|$DEV_LENS"
  "dev32x8|32|8|$DEV_LENS"
  "qwen8b|32|8|2048,2048,2048,2048,2048,2048,2048,2048"
  "long1|16|4|16384"
)

mode="${1:-all}"

echo ">>> building $BIN from $SRC ..."
$NVCC -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I tmp/cutlass/include -I include "$SRC" -o "$BIN"

for entry in "${SHAPES[@]}"; do
  IFS='|' read -r name HQ HKV LENS <<<"$entry"
  echo "================ shape $name (Hq=$HQ Hkv=$HKV) ================"

  if [[ "$mode" == "all" || "$mode" == "events" ]]; then
    : > "$OUT/events_$name.log"
    for round in 1 2 3 4 5; do
      echo "round $round fa2:" >> "$OUT/events_$name.log"
      LENS="$LENS" HQ=$HQ HKV=$HKV FA2_BENCH=1 "$PY" prof/fa2_ab_prof.py 2>/dev/null | grep FA2_MS >> "$OUT/events_$name.log"
      echo "round $round ours:" >> "$OUT/events_$name.log"
      S3_LENS="$LENS" S3_QH=$HQ S3_KH=$HKV "./$BIN" | grep -E "^[ 0-9]+," >> "$OUT/events_$name.log"
    done
    echo "--- events_$name.log:"; cat "$OUT/events_$name.log"
  fi

  if [[ "$mode" == "all" || "$mode" == "ncu" ]]; then
    # ours: single-cfg run = 10 warmup + 100 timed -> launch 50 is steady state.
    echo ">>> ncu ours $name ..."
    S3_LENS="$LENS" S3_QH=$HQ S3_KH=$HKV \
      "$NCU" --set full --clock-control none \
      -k regex:s3_kernel --launch-skip 50 --launch-count 1 \
      -f -o "$OUT/ours_$name" "./$BIN"
    # fa2: 20 warmup + 100 timed -> launch 60 is steady state.
    echo ">>> ncu fa2 $name ..."
    LENS="$LENS" HQ=$HQ HKV=$HKV \
      "$NCU" --set full --clock-control none \
      -k regex:BatchPrefill --launch-skip 60 --launch-count 1 \
      -f -o "$OUT/fa2_$name" "$PY" prof/fa2_ab_prof.py
  fi
done

echo "================ SUMMARY ================"
METRICS=gpu__time_duration.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed
for entry in "${SHAPES[@]}"; do
  IFS='|' read -r name HQ HKV LENS <<<"$entry"
  for side in ours fa2; do
    rep="$OUT/${side}_$name.ncu-rep"
    [[ -f "$rep" ]] || continue
    echo "--- $side $name ---"
    "$NCU" --import "$rep" --page raw --metrics "$METRICS" 2>/dev/null | grep -E "gpu__time|pipe_tensor|sm__throughput|lts__throughput|dram__throughput"
  done
done
echo ">>> event logs + ncu reps under $OUT/"
