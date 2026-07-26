# PR 草案：MXFP8 ragged prefill attention for SM120/121（待审核，勿直接提交）

> 状态：**草案待审**。2026-07-25 误发的 draft PR flashinfer-ai/flashinfer#4147 已被作者关闭；
> 审核通过本文档后再重新提交。分支 `ultism/flashinfer:mxfp8-prefill-sm120` 仍在（含全部改动，
> 已合并上游 main@fa2672f，测试 5/5 通过）。

---

## PR 元信息

- **仓库**：flashinfer-ai/flashinfer（base: `main`）
- **head**：`ultism:mxfp8-prefill-sm120`
- **标题**：`feat(attention): MXFP8 ragged prefill attention for SM120/121`
- **形式**：Draft PR
- **改动量**：17 文件，+3028 行（2 commits：feature + merge upstream main）
- **关联 RFC**：**flashinfer-ai/flashinfer#3628** "[RFC] MXFP8 block-scaled prefill
  attention for SM120a (consumer Blackwell)"（2026-06-13 由 ultism 提交，open，
  needs-triage，无评论）。PR 正文首行写 `Implements RFC #3628.`（正文只链 issue，
  不收 RFC 全文进 PR——已按审核意见移除 design_docs 文件。）

## 改动清单

| 类别 | 文件 |
|---|---|
| kernel 头（新） | `include/flashinfer/attention/sm120/mxfp8_attention_sm120/{kernel,mxfp8_mma,tile_scheduler,named_barrier}.cuh` |
| binding（新） | `csrc/mxfp8_attention_sm120/mxfp8_attention_sm120_binding.cu`（TVM-FFI，无 torch 头） |
| JIT（新） | `flashinfer/jit/mxfp8_attention_sm120.py` |
| Python API（新） | `flashinfer/mxfp8_attention_sm120.py`（`mxfp8_attention_sm120_fwd`） |
| trace（新） | `flashinfer/trace/templates/mxfp8_attention_sm120.py`、`tests/trace/fi_trace_out/mxfp8_attention_sm120_{fwd,run}_head_dim128.json` |
| 测试（新） | `tests/attention/test_mxfp8_attention_sm120.py`（7 例）、`benchmarks/bench_mxfp8_attention_sm120.py` |
| 注册（改） | `flashinfer/__init__.py`、`flashinfer/aot.py`、`flashinfer/jit/__init__.py`、`tests/trace/example.py`、`docs/api/attention.rst` |

## Commit message（feature commit）

```
feat(attention): MXFP8 ragged prefill attention for SM120/121

Add a warp-specialized persistent MXFP8 prefill kernel for consumer
Blackwell (SM120a/SM121a), exposed as mxfp8_attention_sm120_fwd: ragged
varlen FP8 (e4m3) Q/K/V with per-tensor scales (q/k folded into the score
scale, v into the PV output), GQA, and slice-3 (append) causal masking,
which covers prefix-cache/chunked-prefill continuation natively.

The block-scaled tensor instruction on consumer Blackwell sustains full
throughput with FP32 accumulate while legacy warp-MMA FP32-acc paths are
halved (measurements in RFC flashinfer-ai/flashinfer#3628), so the
mainloop keeps the PV GEMM on the SM120 block-scaled MMA atom.

Structure follows nvfp4_attention_sm120: framework-agnostic headers under
include/flashinfer/attention/sm120/mxfp8_attention_sm120/, a TVM-FFI
binding, JIT module, Python API (host-side 128-row padding + persistent
LPT work-list build), trace template, AOT registration, API docs, a
microbenchmark, and oracle tests replaying the kernel's P requantization.

AI-assisted change (kernel developed and validated in an external
worktree, including ncu-guided optimization and vLLM end-to-end checks
against Qwen3 before upstreaming).
```

---

## PR 正文（body）

### Summary

Implements RFC #3628.

Add an **MXFP8 ragged prefill attention kernel for SM120a/SM121a** (consumer Blackwell — RTX 50 series, GB20x), exposed as `flashinfer.mxfp8_attention_sm120_fwd`. FlashInfer currently has no block-scaled FP8 attention that runs on SM120; this PR adds one, following the standalone-op structure of `nvfp4_attention_sm120`.

### Motivation (from the RFC)

On consumer Blackwell, FP32-accumulate tensor throughput is halved for the legacy warp-MMA instructions — but the **block-scaled tensor instruction is not throttled** (register-resident MMA microbenchmark, RTX 5060 Ti, pure issue rate):

| inputs / accumulate | SASS | TFLOP/s |
|---|---|---:|
| FP16 / FP32 acc | `HMMA.16816.F32` | ~51 |
| FP8 e4m3 / FP32 acc (plain) | `QMMA.16832.F32.E4M3.E4M3` | ~102 |
| **MXFP8 e4m3 + ue8m0 block-scaled / FP32 acc** | **`QMMA.SF.16832.F32.E4M3.E4M3.E8`** | **~202** |

The block-scaled MXFP8 MMA is the only way to get full-rate tensor throughput *with* FP32 accumulation on these parts, so the mainloop keeps both GEMMs on the SM120 block-scaled MMA atom (`SM120_16x8x32_TN_VS`).

### What

- **Kernel** (`include/flashinfer/attention/sm120/mxfp8_attention_sm120/`): warp-specialized producer/consumer prefill kernel, TMA-fed, persistent with a host-built LPT (longest-processing-time) tile work-list (`BatchPrefillPersistentTileScheduler`). Framework-agnostic headers, raw pointers.
- **Contract** (`mxfp8_attention_sm120_fwd`): ragged varlen FP8 (`float8_e4m3fn`) Q/K/V `[total_tokens, heads, 128]`, per-tensor scales (`q_scale·k_scale` fold into `sm_scale` on the host, `v_scale` into the PV output in-kernel — the kernel's `kUniformFp8` mode), GQA (`num_qo_heads % num_kv_heads == 0`), and FlashInfer slice-3 causal semantics (query `m` attends keys `[0, m + kv_len - qo_len]`), which covers prefix-cache hits and chunked-prefill continuation natively. Returns `(out [total_q, Hq, 128] fp16/bf16, lse [total_q, Hq] fp32)`.
- **Plumbing** mirroring `nvfp4_attention_sm120`: TVM-FFI binding (`csrc/mxfp8_attention_sm120/`), JIT module with `sm120a_nvcc_flags`, AOT registration, trace templates + `fi_trace_out` examples + `tests/trace/example.py` entry, API docs, and a standalone microbenchmark (`benchmarks/bench_mxfp8_attention_sm120.py`).
- **Plan/run split** (`MXFP8AttentionSM120Wrapper`): `plan()` performs the batch-composition host work once per step (128-row padding layout, persistent LPT work lists) and `run()` is tensors-only per layer — matching the calling convention of the other FlashInfer prefill wrappers, so per-layer calls do not repeat host work. Scales/`sm_scale` are run-time (per-layer quantization); `causal` is plan-time (scheduler cost model). `mxfp8_attention_sm120_fwd` remains as a one-shot plan+run convenience.

### Testing

- `tests/attention/test_mxfp8_attention_sm120.py` (SM120/SM121-gated): ragged GQA causal (bf16/fp16 out), ragged MHA non-causal, single long request (2048), prefix-append (`kv_len > qo_len`), wrapper plan-reuse (one plan, multiple runs), and the one-shot functional path. The reference replays the kernel's exact numerics — unnormalized P requantized to e4m3 at a fixed scale of 256, normalized by the unquantized row sum — so the gates are tight against masking/stride/GQA indexing bugs while tolerating only accumulation-order noise. **7/7 passing on RTX 5060 Ti (SM120).**
- Trace-template consistency tests pass (auto-discovered).
- ruff + clang-format (19.1.1) clean.
- Before upstreaming, the same kernel was validated end-to-end in vLLM 0.21 (Qwen3-8B-AWQ, fp8 KV cache, `enable_prefix_caching=True`, chunked prefill): long-context outputs matched the stock bf16-prefill backend token-for-token.

### Performance

RTX 5090 (sm_120), same ragged batch on both sides, event timing (min of 5 alternating A/B rounds x 100 iters) cross-checked against single-launch `gpu__time_duration` from `ncu --set full` (agreement <3%). Baseline: flashinfer `BatchPrefillWithRaggedKVCacheWrapper(backend="fa2")`, bf16.

| shape (identical batch both sides) | fa2 | ours | **speedup** | tensor-pipe active (fa2 / ours) |
|---|---:|---:|---:|---:|
| 16 mixed varlen requests, Hq8/Hkv2, causal | 0.388 ms | 0.141 ms | **2.74x** | 32.5% / 46.9% |
| same batch, Hq32/Hkv8, causal | 1.165 ms | 0.566 ms | **2.06x** | 42.7% / 49.3% |
| 8x2048 (Qwen3-8B-like), Hq32/Hkv8, causal | 1.317 ms | 0.614 ms | **2.14x** | 44.0% / 51.5% |
| single 16384-token request, Hq16/Hkv4, causal | 5.434 ms | 2.234 ms | **2.43x** | 41.5% / 54.8% |

Effective compute: fa2 saturates at ~200-209 TFLOP/s on the larger shapes — i.e. it already sits at 83-87% of the consumer-Blackwell `HMMA.F32` (bf16, FP32-accumulate) hardware ceiling. This kernel reaches 413-492 TFLOP/s with the same FP32-accumulate semantics on the block-scaled path, while its tensor pipe is only 47-55% busy — roughly 2x of headroom remains toward the block-scaled ceiling. Full methodology + raw reps: [`ultism/fa-blackwell` docs/5090_evidence.md](https://github.com/ultism/fa-blackwell/blob/master/docs/5090_evidence.md).

Note: these are kernel-level numbers for the attention itself; the op-level Python path additionally does the host-side padding/LPT work described above.

### Follow-ups (not in this PR)

- `BatchPrefillWithRaggedKVCacheWrapper` dispatch integration (this PR is the standalone op only).
- Drop the internal V transpose by moving to an HD-major V smem atom (the current Sk-major atom requires Sk-contiguous gmem; documented in the binding).
- Move the plan-time host work (padding layout + LPT lists) on-device to remove the D2H sync.
- Split-KV scheduling for single-long-prompt batches, RTX 5090 benchmarks, and vLLM e2e wiring.

AI-assisted development (kernel authored and validated in an external worktree with ncu-guided optimization before upstreaming).

---

1. ~~"2.8×" 数据要不要贴~~ → **已定：换 5090 实测表**（2.06–2.74×，tensor-pipe 47–55% vs 32–44%，fa2 钉在 HMMA.F32 上界 200–209 TFLOP/s；正文已更新，方法学链 `ultism/fa-blackwell` 的 `docs/5090_evidence.md`）。
2. Follow-ups 清单是否删减（写多了可能被要求先完成再合入）？
3. vLLM e2e 那段（"token-for-token"）是否保留——是我们自己环境的结果，上游无法直接复现。
4. ~~RFC 收不收进 PR~~ → **已定：只链 #3628，不收全文**（已执行）。
5. 审核通过后：重开 #4147 还是新建 PR？
6. commit message 里有一句 `(measured in docs/design_docs/mxfp8_prefill_sm120.md)`，
   文件移除后该引用悬空——提交前是否 squash/改写掉（需要 amend + force push fork 分支）？
