import sys, os
SAGE="/root/fa-blackwell/tmp/SageAttention/sageattention3_blackwell"
sys.path.insert(0, SAGE); os.chdir(SAGE)
import torch
from sageattn3.api import sageattn3_blackwell
dev="cuda"; torch.manual_seed(0)
# head_dim=128 causal, steady-state compute-bound. Dense (Sage api has no GQA); compare STALL REGIME.
B,H,L,D = 2,16,4096,128
q = torch.randn(B,H,L,D, dtype=torch.bfloat16, device=dev)
k = torch.randn(B,H,L,D, dtype=torch.bfloat16, device=dev)
v = torch.randn(B,H,L,D, dtype=torch.bfloat16, device=dev)
for _ in range(10): o = sageattn3_blackwell(q,k,v,is_causal=True)   # warmup
torch.cuda.synchronize()
N = int(os.environ.get("SAGE_ITERS","6"))
for _ in range(N): o = sageattn3_blackwell(q,k,v,is_causal=True)    # ncu targets compute_attn_ws here
torch.cuda.synchronize()
print("SAGE_DONE", o.shape)
