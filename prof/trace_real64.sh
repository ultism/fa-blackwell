#!/usr/bin/env bash
# Full ncu trace of the CURRENT real-64 operator (branch experiment/s8-real-n64, kBlockN=64).
# WRITTEN 2026-06-24, NOT yet run (GPU busy). Run when the GPU frees up.
#
# Captures the production GQA causal launch (8 qo_heads / 2 kv_heads, varlen ragged batch)
# with the full ncu section set, so the L2/SOL + spill picture can be compared apples-to-apples
# against the archived HEAD-128 baseline (memory: HEAD-128 SOL L2 86%, local 1.66GB spill, L1 hit 1.5%).
#
# Expectation to verify in the rep: real-64 should show L2 SOL DROPPED (spill 780B->20B at ptxas),
# long_scoreboard ~0.23 (was 2.76), issue_active ~40% (was 29%), local traffic collapsed.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (prof/ -> ..)

NCU=/usr/local/cuda/bin/ncu
BIN=tests/bench_ragged_real64
SRC=tests/bench_ragged.cu

# --- 1. build a CLEAN binary straight from the branch source (guarantees real-64, no stale binary) ---
# kBlockN=64 / kSFBlockN=128 / static_assert(kBlockN==64) are baked into tests/s3_kernel.cuh on this branch.
echo ">>> building $BIN from $SRC (real-64) ..."
nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I tmp/cutlass/include -I include \
  "$SRC" -o "$BIN"

# --- 2. launch indexing (from launch_timed: 10 warmup + 100 timed = 110 launches/config) ---
#   {1,1}  = launch   0..109   (skip)
#   {8,2}  = launch 110..219   <-- production GQA causal; 120 = a steady-state TIMED launch
#   {32,8} = launch 220..329
# All three configs instantiate the SAME kernel (Causal=true, qh/kh are runtime params) ->
# ncu --launch-skip counts globally across them. 120 lands squarely in the (8,2) timed region.
mkdir -p prof

# --clock-control none = let the card boost (matches the deep-dig methodology that produced the
#   "real 2.63GHz, L2 86%" baseline). Switch to `base` if you want clock-locked reproducibility instead.
echo ">>> ncu --set full on (8,2) GQA causal (launch 120) ..."
"$NCU" --set full --clock-control none \
  -k regex:s3_kernel --launch-skip 120 --launch-count 1 \
  -f -o prof/real64_8x2_causal \
  "$BIN"

# --- 3. (optional) also grab (32,8); uncomment to run. launch 230 = first timed (32,8). ---
# echo ">>> ncu --set full on (32,8) GQA causal (launch 230) ..."
# "$NCU" --set full --clock-control none \
#   -k regex:s3_kernel --launch-skip 230 --launch-count 1 \
#   -f -o prof/real64_32x8_causal \
#   "$BIN"

echo ">>> DONE. Rep at prof/real64_8x2_causal.ncu-rep"
echo ">>> quick SOL/spill peek:"
echo "    $NCU --import prof/real64_8x2_causal.ncu-rep --page details \\"
echo "        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\"
echo "lts__throughput.avg.pct_of_peak_sustained_elapsed,\\"
echo "l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\\"
echo "smsp__average_warp_latency_per_inst_issued.ratio,\\"
echo "smsp__issue_active.avg.pct_of_peak_sustained_elapsed,\\"
echo "memory_l2_theoretical_sectors_local"
