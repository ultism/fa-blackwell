import os
import sys

import torch

# fa2 baseline runner for A/B evidence collection. Same varlen batch as
# tests/bench_ragged.cu (env-overridable), flashinfer ragged wrapper backend=fa2.
#   FA2_BENCH=1  -> CUDA-event timing mode: prints "FA2_MS <ms>" (avg of FA2_ITERS)
#   otherwise    -> plain run (ncu capture target): 20 warmup + FA2_ITERS launches
#
# Env: LENS="512,1024,..." HQ=8 HKV=8 CAUSAL=1 FA2_ITERS=100
def _patch_cutlass_dsl_operand_major_mode():
    try:
        import cutlass.cute as cute
        from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
    except ImportError:
        return
    if not hasattr(cute.nvgpu, "OperandMajorMode"):
        cute.nvgpu.OperandMajorMode = OperandMajorMode


_patch_cutlass_dsl_operand_major_mode()

sys.path.insert(0, os.environ.get("FI_SRC", "/root/fa-blackwell/flashinfer"))
import flashinfer  # noqa: E402

LENS = [int(x) for x in os.environ.get("LENS", "512,1024,768,1536,2048,640,1280,896,384,1792,512,2560,1024,768,2048,1100").split(",")]
HQ = int(os.environ.get("HQ", "8"))
HKV = int(os.environ.get("HKV", "2"))
CAUSAL = os.environ.get("CAUSAL", "1") == "1"
ITERS = int(os.environ.get("FA2_ITERS", "100"))
BENCH = os.environ.get("FA2_BENCH", "0") == "1"

dev = "cuda"
torch.manual_seed(0)
D = 128
qo_indptr = torch.tensor([0] + list(torch.tensor(LENS).cumsum(0)), dtype=torch.int32, device=dev)
kv_indptr = qo_indptr.clone()
total = sum(LENS)
ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
q = torch.randn(total, HQ, D, dtype=torch.bfloat16, device=dev)
k = torch.randn(total, HKV, D, dtype=torch.bfloat16, device=dev)
v = torch.randn(total, HKV, D, dtype=torch.bfloat16, device=dev)
wrapper.plan(qo_indptr, kv_indptr, HQ, HKV, D, causal=CAUSAL, q_data_type=torch.bfloat16)

for _ in range(20):
    o = wrapper.run(q, k, v)
torch.cuda.synchronize()

if BENCH:
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(ITERS):
        o = wrapper.run(q, k, v)
    e1.record()
    torch.cuda.synchronize()
    print(f"FA2_MS {e0.elapsed_time(e1) / ITERS:.4f}")
else:
    for _ in range(ITERS):
        o = wrapper.run(q, k, v)
    torch.cuda.synchronize()
    print("FA2_DONE", o.shape)
