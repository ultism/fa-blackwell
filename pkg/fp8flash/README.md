# fp8flash — FA2-compatible fp8 (e4m3) flash-attention for sm90 (H20)

`fp8flash` gives a **flash-attn-2 compatible call shape** backed by the s3
ragged prefill kernel (plain e4m3 `mma.sync`, warp-specialized + TMA,
persistent tile scheduler). Inputs are **uint8 e4m3 bytes** with per-tensor
dequant scales (vllm-style fp8 KV cache layout).

Validated on NVIDIA H20-3e inside `vllm/vllm-openai:v0.26.0` (torch 2.11.0+cu130,
CUDA 13.0). Kernel numerics: O max|abs| = **2.4e-3** vs an fp64 oracle
(`tests/test_ragged_ext.py`, gate 5e-3).

## Requirements

- sm90 GPU (H100/H20), driver with CUDA ≥ 13.0
- python 3.12, **torch == 2.11.0+cu130 exactly** — the bundled extension is a
  torch C++ extension (no cross-version ABI); e.g. torch 2.12/2.13 fails at
  import with `undefined symbol: _ZN3c104impl3cow23materialize_cow_storage...`.
  The `vllm/vllm-openai:v0.26.0` image matches out of the box.
- linux x86_64

## Install & run

```bash
# torch must be pinned BEFORE installing the wheel (uv run auto-syncs to the
# latest torch unless the pin is in pyproject.toml / requirements):
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install fp8flash-0.1.0-py3-none-linux_x86_64.whl
python example.py     # smoke test incl. fp32-SDPA cross-check
```

## API

```python
from fp8flash import flash_attn_func

out = flash_attn_func(q, k, v,                      # uint8 e4m3, (batch, seqlen, nheads, headdim)
                      q_scale=qs, k_scale=ks, v_scale=vs,   # per-tensor fp8 dequant scalars
                      softmax_scale=None,           # default 1/sqrt(headdim)
                      causal=True)
# out: fp32, (batch, seqlen, nheads, headdim)
```

- `q/k/v`: `torch.uint8` CUDA tensors holding e4m3 bit patterns
  (`x_fp8.view(torch.uint8)`), dense `(batch, seqlen, nheads, headdim)`
- GQA/MQA supported: `k/v` may have fewer heads (`nheads % nheads_k == 0`)
- `causal=True/False`; any `seqlen` (128-padding handled internally)
- `headdim` 128 (64 via rebuild)
- output fp32 (cast yourself if you want bf16)

## Example (mirrors the FA2 call in the issue)

```python
import torch
from fp8flash import flash_attn_func

batch_size, seqlen, nheads, headdim = 2, 1024, 8, 128
q16 = torch.randn(batch_size, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
k16 = torch.randn_like(q16); v16 = torch.randn_like(q16)

def quant_per_tensor(x):
    s = x.abs().max().clamp_min(1e-12) / 448.0
    return (x / s).to(torch.float8_e4m3fn).view(torch.uint8), float(s)

q, qs = quant_per_tensor(q16.float())
k, ks = quant_per_tensor(k16.float())
v, vs = quant_per_tensor(v16.float())

out = flash_attn_func(q, k, v, q_scale=qs, k_scale=ks, v_scale=vs, causal=True)
print(out.shape)   # torch.Size([2, 1024, 8, 128])
```

Full runnable version with an fp32-SDPA accuracy check: `example.py`
(attached to the release). Expected printout on H20:

```
Output shape: torch.Size([2, 1024, 8, 128])
max abs err vs fp32 SDPA: 0.04243  (fp8 e4m3 quantization granularity)
fp8flash example OK
```

## Internals

- Kernel source: `tests/s3_kernel.cuh` (sm90 build via
  `-gencode arch=compute_90a,code=sm_90a`); design + port record:
  `docs/sm90_fp8_port_plan.md`
- The wheel bundles the prebuilt extension (`fp8flash/mxfp8_ragged_ext.so`);
  no CUDA toolkit needed at install time
- Ragged/varlen form is the native one; the dense FA2 shape is a thin shim
  (`fp8flash/__init__.py`): pads each request to 128, packs
  `(qo|kv)_indptr`, transposes V to DIM-major `[Hkv, D, Sk]`, folds
  `q_scale*k_scale` into `sm_scale` and `v_scale` into the epilogue
- Known limits: sm90 only (sm120 build exists but is not shipped in this
  wheel); mma.sync fp8 runs through the fp16 pipe on Hopper (wgmma upgrade is
  the v2 path, see the port doc); per-tensor scales only
