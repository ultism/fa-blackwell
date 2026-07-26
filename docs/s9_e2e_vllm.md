# S9 e2e: vLLM 端到端跑通验证（prefill 走 s3 ragged fp8 kernel）

**日期**：2026-07-22 · **环境**：WSL2 / RTX 5060 Ti 16GB（仅跑通验证；实测在 5090 做）

> **2026-07-25 更新**：该 kernel 已正式集成到 flashinfer（`/root/fa-blackwell/flashinfer`，
> op 名 `flashinfer.mxfp8_attention_sm120_fwd`，ragged per-tensor-fp8，sm120/121），
> 结构仿 nvfp4_attention_sm120（include/csrc/jit/python-api/trace/test/bench 全链路），
> `pytest tests/attention/test_mxfp8_attention_sm120.py` 5/5 通过。本文档的猴子补丁
> 路径仍可用于 vllm 对照实验；上游 PR 以 flashinfer 仓库为准。

## 结论（2026-07-22 修正版）

我们的 MXFP8 prefill kernel（kUniformFp8 per-tensor fp8 路径）接入 vllm 0.21
`FlashInferImpl.forward`（prefill → s3 kernel，decode → 原样 fa2），端到端**跑通且输出合理**：

| 场景 | stock vs ours（真实 patch 生效后） |
|---|---|
| 长 prompt（791 tok） | **48/48 逐 token 一致** |
| chunked prefill（1870 tok → offset 追加路径） | **32/32 逐 token 一致** |
| prefix cache 命中（r2 共享 r1 的 ~800 tok 前缀，qo_len<kv_len） | r1 48/48 一致；r2 45/48（token 45 才分叉） |
| 短 prompts（6~40 tok × 4） | 语义等价、部分 token 分叉（fp8-vs-bf16 attention 的正常表现） |

**split-KV（prefix cache）场景 = kernel 原生 slice-3**：vllm 前缀命中后 Q 只含新 token、
KV 为全量（缓存前缀+新写入），kernel 内 `offset_q = kv_len - qo_len`，查询 m  attend
keys `[0, m+offset_q]`（s3_kernel.cuh 两处同步修改点，见 gotcha.md）。缓存前缀的 fp8 KV
由 cache 写入路径统一编码（与 prefill kernel 无关），gather 时与新 KV 无差别——无需任何
特殊处理。验证脚本 `bench/qwen3_e2e_prefix.py`（两请求共享前缀，enable_prefix_caching=True）。

## ⚠️ 关键教训：vllm 0.21 的 spawn worker 会让主进程猴子补丁失效（假阳性）

第一版对比显示"全部逐 token 一致"——**是假的**：vllm 强制 `spawn` EngineCore
（fork 请求被否决："We must use the spawn multiprocessing start method"），worker 进程
重新解释主脚本，主进程里 patch 的 `FlashInferImpl.forward` 在 worker 里根本没生效，
"ours" 和 "stock" 跑的是同一份 stock。识别方法：在 patch 函数里打印（计数器在主进程
不可见！），确认 worker 进程真的走了 patch 分支。

**正确姿势：venv 里放 `sitecustomize.py`**（每个 python 进程启动自动导入，spawn 的
worker 也覆盖），用环境变量 `S3_PATCH=1` 门控调用 `bench.vllm_s3_patch.install()`。
另注意：worker 里的补丁把 vllm 的 output 当 `[tokens, Hq*D]` 会挂——实际是
`[tokens, Hq, D]`，写回前 `reshape(tokens, -1)`。

## 数据通路（每层、每个 prefill chunk）

1. Q（RoPE 后 bf16）→ per-tensor 量化（amax/448，整个 chunk 一个 scale）→ e4m3；
2. K/V 从 paged cache 按 block table gather（fp8 字节原样搬运）→ 打包成 ragged [Sk,Hkv,D]；
3. V 转置为 kernel 需要的 [Hkv, D, Sk]（TMA_V 描述符换 stride 可省，见下）；
4. 每请求 128-pad + LPT work-list + 持久 kernel（`s3_kernel<…, Causal=true, kUniformFp8>`），
   `sm_scale = (1/√D)·q_scale·k_scale`，`o_scale = v_scale`；
5. O fp32 → 按 qo_indptr 取真实行 → cast 回 bf16 写 output。

## 环境适配记录（5090 上不会遇到）

- torch 2.11 c10 头文件 vs nvcc EDG 前端 → 扩展拆 TU：kernel 纯 nvcc（`mxfp8_ragged_kernel.cu`），
  pybind 胶水走 host gcc + `-fpermissive`（`mxfp8_ragged_ext.cpp`）。
- vllm 0.21 可选依赖 `humming` 未装 → venv 里打了 stub 包。
- WSL 下 vllm HTTP server 的 loopback 监听异常 → 一律用离线 `LLM()` API。
- 16GB 显存：14B（哪怕 int4）+ KV 预算挤不下 → 本机用 Qwen3-8B-AWQ（5.7GB），
  `kv_cache_memory_bytes=1GB` 限制 KV 预占。
- `max_num_batched_tokens=1024` 压低 profiling 激活峰值。

## 给 5090 测评的准备清单

1. **性能版数据通路**：现在的 gather/pad/转置是 per-layer 的 python 循环（每层 ~ms 级
   host 开销），实测前必须 fused——方案 A：一个 gather+quantize+transpose 的融合 CUDA
   kernel；方案 B（更优）：**教 s3 kernel 直接按页表读 paged cache**（TMA 按 page 索引
   取坐标，k/v 各加一个 indirection），Q 量化也融进去。方案 B 同时消掉 V 转置
   （TMA_V 描述符换成 token-major stride 即可）。
2. 模型：Qwen3-14B-FP8（15GB）或 14B-AWQ（9.5GB）在 32GB 上均可；14B 的 group=5
   是新配置（本机只测过 group 1/2/4/8），到 5090 先跑一遍 `tests/test_ragged_ext.py`
   改 (Hq,Hkv)=(40,8) 的 oracle 验证。
3. 基准：TTFT / prefill throughput vs fa2（FLASHINFER backend 即同栈基线）+ vs
   vllm 的 trtllm-gen（若 5090 可用 cubin）。
4. 计时注意：本补丁的 decode 没动（stock fa2 decode），对比时把 prefill 阶段单独计时。
