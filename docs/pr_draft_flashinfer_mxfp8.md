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
| trace（新） | `flashinfer/trace/templates/mxfp8_attention_sm120.py`、`tests/trace/fi_trace_out/mxfp8_attention_sm120_fwd_head_dim128.json` |
| 测试（新） | `tests/attention/test_mxfp8_attention_sm120.py`（5 例）、`benchmarks/bench_mxfp8_attention_sm120.py` |
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
halved (measured in docs/design_docs/mxfp8_prefill_sm120.md), so the
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
- **Plumbing** mirroring `nvfp4_attention_sm120`: TVM-FFI binding (`csrc/mxfp8_attention_sm120/`), JIT module with `sm120a_nvcc_flags`, AOT registration, trace template + `fi_trace_out` example + `tests/trace/example.py` entry, API docs, and a standalone microbenchmark (`benchmarks/bench_mxfp8_attention_sm120.py`).
- The Python API currently does the per-request 128-row padding and the LPT work-list build on the host (documented in the docstring).

### Testing

- `tests/attention/test_mxfp8_attention_sm120.py` (SM120/SM121-gated): ragged GQA causal (bf16/fp16 out), ragged MHA non-causal, single long request (2048), and prefix-append (`kv_len > qo_len`) cases. The reference replays the kernel's exact numerics — unnormalized P requantized to e4m3 at a fixed scale of 256, normalized by the unquantized row sum — so the gates are tight against masking/stride/GQA indexing bugs while tolerating only accumulation-order noise. **5/5 passing on RTX 5060 Ti (SM120).**
- Trace-template consistency tests pass (auto-discovered).
- ruff + clang-format (19.1.1) clean.
- Before upstreaming, the same kernel was validated end-to-end in vLLM 0.21 (Qwen3-8B-AWQ, fp8 KV cache, `enable_prefix_caching=True`, chunked prefill): long-context outputs matched the stock bf16-prefill backend token-for-token.

### Performance

Correctness is the focus of this PR; detailed performance tuning/benchmarking is intentionally left for follow-up (see below). The kernel itself is a persistent TMA/warp-specialized design whose microbenchmarks during development showed ~2.8× over the FA2-path prefill it replaces on SM120; op-level numbers currently also include the host-side padding/LPT work.

### Follow-ups (not in this PR)

- `BatchPrefillWithRaggedKVCacheWrapper` dispatch integration (this PR is the standalone op only).
- Drop the internal V transpose by moving to an HD-major V smem atom (the current Sk-major atom requires Sk-contiguous gmem; documented in the binding).
- Move padding/LPT work-list construction off the Python hot path (plan/run split).
- Split-KV scheduling for single-long-prompt batches, RTX 5090 benchmarks, and vLLM e2e wiring.

AI-assisted development (kernel authored and validated in an external worktree with ncu-guided optimization before upstreaming).

---

1. PR 正文里 "2.8× over FA2-path prefill" 的微基准结论要不要保留（5060Ti 开发期数据，可能被要求贴证据）？
   → **证据采集方案已备好（待 5090 执行）**：`prof/trace_5090_ab.sh`（本机已逐项冒烟通过）。
     - 4 个形状 × 双侧同 batch：dev8x2 / dev32x8（开发期 16 请求 varlen batch）、qwen8b（8×2048，Hq32/Hkv8）、long1（单请求 16384，Hq16/Hkv4）。
     - 事件计时：双侧各 20 warmup + 100 timed ×5 交替 A/B 轮取 min（抗 WSL/热节流）。
     - ncu `--set full --clock-control none` 各抓一个稳态 launch：ours=`s3_kernel` skip 50，fa2=`BatchPrefill` skip 60。
     - 对比指标：`gpu__time_duration.sum`（算子实际运行时间）、`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`（tensorcore 活跃度）、`sm__throughput`/`lts__throughput`（SOL）。
     - 运行：`PY=<5090上的python> bash prof/trace_5090_ab.sh`（需 flashinfer 可导入 + nvcc/ncu；fa2 走 `prof/fa2_ab_prof.py`，ours 走 `tests/bench_ragged.cu` 的 S3_LENS/S3_QH/S3_KH env 覆盖）。
     - 存量资产：5060Ti 留档 `prof/{fa2,real64,s9f}_8x2*.ncu-rep`、RFC issue #3628 的 QMMA issue-rate 表（`bench/mma_peak*.cu`）。
2. Follow-ups 清单是否删减（写多了可能被要求先完成再合入）？
3. vLLM e2e 那段（"token-for-token"）是否保留——是我们自己环境的结果，上游无法直接复现。
4. ~~RFC 收不收进 PR~~ → **已定：只链 #3628，不收全文**（已执行）。
5. 审核通过后：重开 #4147 还是新建 PR？
6. commit message 里有一句 `(measured in docs/design_docs/mxfp8_prefill_sm120.md)`，
   文件移除后该引用悬空——提交前是否 squash/改写掉（需要 amend + force push fork 分支）？
