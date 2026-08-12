"""fp8flash example: mirrors the flash-attn-2 flash_attn_func call shape with
uint8 e4m3 inputs + per-tensor scales. Run on sm90 (H20) with torch+cu130."""
import torch

from fp8flash import flash_attn_func

torch.manual_seed(0)
batch_size = 2
seqlen = 1024
nheads = 8
headdim = 128

q16 = torch.randn(batch_size, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
k16 = torch.randn(batch_size, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
v16 = torch.randn(batch_size, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)


def quant_per_tensor(x):
    s = x.abs().max().clamp_min(1e-12) / 448.0
    return (x / s).to(torch.float8_e4m3fn).view(torch.uint8), float(s)


q, q_scale = quant_per_tensor(q16.float())
k, k_scale = quant_per_tensor(k16.float())
v, v_scale = quant_per_tensor(v16.float())

out = flash_attn_func(q, k, v, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale, causal=True)
print(f"Output shape: {out.shape}")  # (batch_size, seqlen, nheads, headdim)

# reference: dequantize -> fp32 SDPA (causal)
qd = q.view(torch.float8_e4m3fn).float() * q_scale
kd = k.view(torch.float8_e4m3fn).float() * k_scale
vd = v.view(torch.float8_e4m3fn).float() * v_scale
ref = torch.nn.functional.scaled_dot_product_attention(
    qd.permute(0, 2, 1, 3), kd.permute(0, 2, 1, 3), vd.permute(0, 2, 1, 3),
    is_causal=True).permute(0, 2, 1, 3)
err = (out - ref).abs().max().item()
print(f"max abs err vs fp32 SDPA: {err:.5f}  (fp8 e4m3 quantization granularity)")
assert err < 5e-2, "MISMATCH"
print("fp8flash example OK")
