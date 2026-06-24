import sys; sys.path.insert(0, "/root/fa-blackwell/tmp/flashinfer")
import torch, flashinfer
dev="cuda"; torch.manual_seed(0)
# SAME varlen batch as tests/bench_ragged.cu, (8,2) GQA causal only.
LENS = [512,1024,768,1536,2048,640,1280,896,384,1792,512,2560,1024,768,2048,1100]
D=128; Hq,Hkv=8,2
qo_indptr = torch.tensor([0]+list(torch.tensor(LENS).cumsum(0)), dtype=torch.int32, device=dev)
kv_indptr = qo_indptr.clone(); total=sum(LENS)
ws = torch.empty(256*1024*1024, dtype=torch.uint8, device=dev)
wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
q = torch.randn(total, Hq, D, dtype=torch.bfloat16, device=dev)
k = torch.randn(total, Hkv, D, dtype=torch.bfloat16, device=dev)
v = torch.randn(total, Hkv, D, dtype=torch.bfloat16, device=dev)
wrapper.plan(qo_indptr, kv_indptr, Hq, Hkv, D, causal=True, q_data_type=torch.bfloat16)
for _ in range(20): o = wrapper.run(q, k, v)     # warmup
torch.cuda.synchronize()
import os
N = int(os.environ.get("FA2_ITERS","6"))
for _ in range(N): o = wrapper.run(q, k, v)      # the launches ncu targets
torch.cuda.synchronize()
print("FA2_DONE", o.shape)
