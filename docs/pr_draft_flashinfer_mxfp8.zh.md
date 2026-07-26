# PR 草案（中文版）：SM120/121 的 MXFP8 ragged prefill attention

> 本文档是 `docs/pr_draft_flashinfer_mxfp8.md` 的中文翻译版，供审核用。
> 正式提交以英文版为准。状态：**草案待审**，勿直接提交。
> 误发的 flashinfer-ai/flashinfer#4147 已关闭；分支 `ultism/flashinfer:mxfp8-prefill-sm120`
> 保留全部改动（已合并上游 main@fa2672f，测试 5/5 通过）。

---

## PR 元信息

- **目标仓库**：flashinfer-ai/flashinfer（base 分支：`main`）
- **源分支**：`ultism:mxfp8-prefill-sm120`
- **标题**：`feat(attention): MXFP8 ragged prefill attention for SM120/121`
- **形式**：Draft PR
- **改动量**：17 个文件，+3028 行（2 个 commit：feature + 合并上游 main）
- **关联 RFC**：**flashinfer-ai/flashinfer#3628**「[RFC] MXFP8 block-scaled prefill
  attention for SM120a (consumer Blackwell)」（2026-06-13 由 ultism 提交，open，
  needs-triage，无评论）。PR 正文首行写 `Implements RFC #3628.`（正文只链 issue，
  不把 RFC 全文收进 PR。）

## 改动清单

| 类别 | 文件 |
|---|---|
| kernel 头文件（新增） | `include/flashinfer/attention/sm120/mxfp8_attention_sm120/{kernel,mxfp8_mma,tile_scheduler,named_barrier}.cuh` |
| binding（新增） | `csrc/mxfp8_attention_sm120/mxfp8_attention_sm120_binding.cu`（TVM-FFI，不含 torch 头） |
| JIT 模块（新增） | `flashinfer/jit/mxfp8_attention_sm120.py` |
| Python API（新增） | `flashinfer/mxfp8_attention_sm120.py`（`mxfp8_attention_sm120_fwd`） |
| trace（新增） | `flashinfer/trace/templates/mxfp8_attention_sm120.py`、`tests/trace/fi_trace_out/mxfp8_attention_sm120_{fwd,run}_head_dim128.json` |
| 测试/benchmark（新增） | `tests/attention/test_mxfp8_attention_sm120.py`（7 例）、`benchmarks/bench_mxfp8_attention_sm120.py` |
| 注册（修改） | `flashinfer/__init__.py`、`flashinfer/aot.py`、`flashinfer/jit/__init__.py`、`tests/trace/example.py`、`docs/api/attention.rst` |

## Commit message（feature commit，提交用英文原文，下附中文）

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

中文大意：

> 为消费级 Blackwell（SM120a/SM121a）新增一个 warp 特化、持久化的 MXFP8 prefill
> kernel，对外暴露为 `mxfp8_attention_sm120_fwd`：ragged 变长 FP8（e4m3）Q/K/V，
> per-tensor scale（q/k 折进分数缩放，v 折进 PV 输出），支持 GQA 和 slice-3（追加式）
> causal mask，天然覆盖 prefix cache / chunked prefill 续算场景。
>
> 消费级 Blackwell 上块缩放张量指令在 FP32 累加下不受限（传统 warp-MMA 的 FP32 累加
> 通路减半，测量见 RFC #3628），因此 mainloop 把 PV GEMM 保持在 SM120 块缩放 MMA
> 指令上。
>
> 结构仿照 `nvfp4_attention_sm120`：框架无关头文件、TVM-FFI binding、JIT 模块、
> Python API（host 侧 128 行 padding + 持久化 LPT work-list 构建）、trace 模板、
> AOT 注册、API 文档、微基准、以及回放 kernel P 重量化的 oracle 测试。
>
> AI 辅助开发（kernel 在外部 worktree 完成开发与验证，含 ncu 指导的优化和基于
> Qwen3 的 vLLM 端到端校验，然后才上游化）。

---

## PR 正文（body 中文翻译）

### 概述（Summary）

实现 RFC #3628。

新增一个面向 **SM120a/SM121a（消费级 Blackwell——RTX 50 系，GB20x）的 MXFP8
ragged prefill attention kernel**，以 `flashinfer.mxfp8_attention_sm120_fwd` 暴露。
FlashInfer 目前没有任何能在 SM120 上运行的块缩放 FP8 attention；本 PR 按
`nvfp4_attention_sm120` 的独立算子结构补上一个。

### 动机（Motivation，引自 RFC）

在消费级 Blackwell 上，传统 warp-MMA 指令的 FP32 累加张量吞吐被减半——但**块缩放
张量指令不受此限**（寄存器驻留 MMA 微基准，RTX 5060 Ti，纯指令发射速率）：

| 输入 / 累加 | SASS | TFLOP/s |
|---|---|---:|
| FP16 / FP32 累加 | `HMMA.16816.F32` | ~51 |
| FP8 e4m3 / FP32 累加（普通） | `QMMA.16832.F32.E4M3.E4M3` | ~102 |
| **MXFP8 e4m3 + ue8m0 块缩放 / FP32 累加** | **`QMMA.SF.16832.F32.E4M3.E4M3.E8`** | **~202** |

块缩放 MXFP8 MMA 是在这些芯片上同时拿到全速张量吞吐*和* FP32 累加的唯一途径，
因此 mainloop 的两个 GEMM 都保持在 SM120 块缩放 MMA atom（`SM120_16x8x32_TN_VS`）上。

### 内容（What）

- **Kernel**（`include/flashinfer/attention/sm120/mxfp8_attention_sm120/`）：warp 特化
  生产者/消费者 prefill kernel，TMA 喂数，持久化调度，work-list 由 host 端按 LPT
  （最长处理时间优先）构建（`BatchPrefillPersistentTileScheduler`）。框架无关头文件、
  裸指针接口。
- **算子契约**（`mxfp8_attention_sm120_fwd`）：ragged 变长 FP8（`float8_e4m3fn`）
  Q/K/V，形状 `[total_tokens, heads, 128]`；per-tensor scale（`q_scale·k_scale` 在 host
  折进 `sm_scale`，`v_scale` 在 kernel 内折进 PV 输出——即 kernel 的 `kUniformFp8`
  模式）；GQA（`num_qo_heads % num_kv_heads == 0`）；FlashInfer slice-3 causal 语义
  （第 `m` 个查询 attend keys `[0, m + kv_len - qo_len]`），天然覆盖 prefix cache 命中
  和 chunked prefill 续算。返回 `(out [total_q, Hq, 128] fp16/bf16, lse [total_q, Hq] fp32)`。
- **配套工程**：仿 `nvfp4_attention_sm120`：TVM-FFI binding（`csrc/mxfp8_attention_sm120/`）、
  带 `sm120a_nvcc_flags` 的 JIT 模块、AOT 注册、trace 模板 + `fi_trace_out` 示例 +
  `tests/trace/example.py` 条目、API 文档、独立微基准
  （`benchmarks/bench_mxfp8_attention_sm120.py`）。
- **plan/run 拆分**（`MXFP8AttentionSM120Wrapper`）：`plan()` 每步一次完成 batch 组成的
  host 工作（128 行 padding 布局、持久化 LPT work-list），`run()` 每层一次、纯张量——
  与 flashinfer 其他 prefill wrapper 的调用惯例一致，按层调用不重复 host 工作。
  scales/`sm_scale` 在 run 时传入（per-layer 量化），`causal` 在 plan 时固定（影响调度
  成本模型）。`mxfp8_attention_sm120_fwd` 保留为一次性 plan+run 的便捷入口。

### 测试（Testing）

- `tests/attention/test_mxfp8_attention_sm120.py`（SM120/SM121 门控）：ragged GQA
  causal（bf16/fp16 输出）、ragged MHA 非 causal、单长请求（2048）、prefix 追加
  （`kv_len > qo_len`）、wrapper plan 复用（一次 plan 多次 run）、一次性函数式路径。
  参考实现精确回放 kernel 的数值行为——对**未归一化**的 P 以固定 scale 256 重量化到
  e4m3，再用**未量化**的 row sum 归一——因此门限对 mask/stride/GQA 索引类 bug 很紧，
  只容忍累加顺序噪声。**RTX 5060 Ti（SM120）7/7 通过。**
- trace 模板一致性测试通过（自动发现）。
- ruff + clang-format（19.1.1）干净。
- 上游化之前，同一 kernel 在 vLLM 0.21 中做过端到端验证（Qwen3-8B-AWQ、fp8 KV
  cache、`enable_prefix_caching=True`、chunked prefill）：长上下文输出与 stock bf16
  prefill 后端逐 token 一致。

### 性能（Performance）

RTX 5090（sm_120），双侧同一 ragged batch；事件计时（5 轮交替 A/B × 100 次迭代取
min）与 `ncu --set full` 的单 launch `gpu__time_duration` 互验（偏差 <3%）。基线：
flashinfer `BatchPrefillWithRaggedKVCacheWrapper(backend="fa2")`，bf16。

| 形状（双侧同 batch） | fa2 | 本算子 | **加速比** | tensor pipe 活跃（fa2 / 本算子） |
|---|---:|---:|---:|---:|
| 16 个混合 varlen 请求，Hq8/Hkv2，causal | 0.388 ms | 0.141 ms | **2.74x** | 32.5% / 46.9% |
| 同 batch，Hq32/Hkv8，causal | 1.165 ms | 0.566 ms | **2.06x** | 42.7% / 49.3% |
| 8×2048（类 Qwen3-8B），Hq32/Hkv8，causal | 1.317 ms | 0.614 ms | **2.14x** | 44.0% / 51.5% |
| 单请求 16384 token，Hq16/Hkv4，causal | 5.434 ms | 2.234 ms | **2.43x** | 41.5% / 54.8% |

有效算力：fa2 在较大形状下饱和于 ~200–209 TFLOP/s——即它已经顶到消费级 Blackwell
`HMMA.F32`（bf16、FP32 累加）硬件上限的 83–87%。本 kernel 以同样的 FP32 累加语义
在块缩放通路上达到 413–492 TFLOP/s，而 tensor pipe 仅 47–55% 占用——距块缩放上界
还有约一倍余量。完整方法学 + 原始 ncu reps：见
[`ultism/fa-blackwell` 的 docs/5090_evidence.md](https://github.com/ultism/fa-blackwell/blob/master/docs/5090_evidence.md)。

注：以上为 attention 本体的 kernel 级数字；算子级 Python 路径还包含前述 host 侧
padding/LPT 工作。

### 后续工作（Follow-ups，不在本 PR）

- `BatchPrefillWithRaggedKVCacheWrapper` 调度集成（本 PR 仅独立算子）。

AI 辅助开发（kernel 在外部 worktree 完成编写与验证，含 ncu 指导的优化，然后上游化）。

---

## 待审核的开放问题

1. ~~"2.8×" 数据要不要贴~~ → **已定：换 5090 实测表**（2.06–2.74×，tensor-pipe
   47–55% vs 32–44%，fa2 钉在 HMMA.F32 上界 200–209 TFLOP/s；正文已更新，方法学
   链 `ultism/fa-blackwell` 的 `docs/5090_evidence.md`）。
2. ~~Follow-ups 清单是否删减~~ → **已定：只留 wrapper 调度集成一条**（vLLM 等由他人跑）。
3. ~~vLLM e2e 段~~ → **保留**（已完成的工作证据，非承诺）。
4. ~~RFC 收不收进 PR~~ → **已定：只链 #3628，不收全文**（已执行）。
5. 审核通过后：重开 #4147 还是新建 PR？
6. ~~commit message 引用被删文件路径~~ → **已定：改为引用 RFC issue #3628**（英文草案已同步）。
