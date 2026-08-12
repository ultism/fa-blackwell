"""fp8flash: FA2-compatible fp8 (e4m3) flash-attention prefill for sm90 (H20).

The CUDA kernel (s3 ragged varlen + GQA + per-tensor fp8, plain e4m3 mma.sync)
is prebuilt and bundled as mxfp8_ragged_ext.so. flash_attn_func mirrors the
flash-attn-2 call shape, with uint8 e4m3 inputs + per-tensor scales.
"""
import importlib.util
import pathlib

import torch

_spec = importlib.util.spec_from_file_location(
    "mxfp8_ragged_ext", pathlib.Path(__file__).with_name("mxfp8_ragged_ext.so"))
_C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_C)

__all__ = ["flash_attn_func"]


def _pad128(n):
    return (n + 127) // 128 * 128


def flash_attn_func(q, k, v, q_scale=1.0, k_scale=1.0, v_scale=1.0,
                    softmax_scale=None, causal=False):
    """flash-attn-2 compatible call: out = softmax(q k^T * scale) v, causal option.

    q, k, v : uint8 CUDA tensors of e4m3 bytes, (batch, seqlen, nheads, headdim);
              k/v may use fewer heads (GQA, nheads % nheads_k == 0).
    q_scale / k_scale / v_scale : per-tensor dequant scalars (value = e4m3 * scale).
    softmax_scale : defaults to 1/sqrt(headdim).
    Returns fp32 (batch, seqlen, nheads, headdim) on the same device.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be CUDA tensors"
    assert q.dtype == k.dtype == v.dtype == torch.uint8, "q/k/v must be uint8 e4m3 bytes"
    B, L, Hq, D = q.shape
    Bk, Lk, Hkv, Dk = k.shape
    assert Bk == B and Lk == L and Dk == D, "q/k/v batch, seqlen, headdim must match"
    assert Hq % Hkv == 0, "GQA group must divide evenly"
    assert v.shape[2] == Hkv and v.shape[3] == D
    dev = q.device

    Lp = _pad128(L)
    indptr = torch.arange(0, (B + 1) * Lp, Lp, dtype=torch.int32, device=dev)
    lens = torch.full((B,), L, dtype=torch.int32, device=dev)

    Qp = torch.zeros(B * Lp, Hq, D, dtype=torch.uint8, device=dev)
    Kp = torch.zeros(B * Lp, Hkv, D, dtype=torch.uint8, device=dev)
    Vp = torch.zeros(Hkv, D, B * Lp, dtype=torch.uint8, device=dev)
    for b in range(B):
        s = slice(b * Lp, b * Lp + L)
        Qp[s] = q[b]
        Kp[s] = k[b]
        Vp[:, :, s] = v[b].permute(1, 2, 0)

    sm_scale = softmax_scale if softmax_scale is not None else 1.0 / D ** 0.5
    sm_scale = float(sm_scale) * float(q_scale) * float(k_scale)
    O, _LSE, _L = _C.s3_ragged_fp8_attn(Qp, Kp, Vp, indptr, indptr, lens, lens,
                                        Hq, Hkv, sm_scale, float(v_scale), causal, 0)
    out = torch.empty(B, L, Hq, D, dtype=torch.float32, device=dev)
    for b in range(B):
        out[b] = O[b * Lp: b * Lp + L]
    return out
