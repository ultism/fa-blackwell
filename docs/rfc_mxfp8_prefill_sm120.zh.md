# [RFC] 面向 SM120a（消费级 Blackwell）的 MXFP8 block-scaled prefill attention

> 中文版，供自己审阅。对外发布用同目录的 `rfc_mxfp8_prefill_sm120.md`（英文）。
> 末尾「内部备注」一节**不要**发出去。

## 摘要

为 **SM120a / SM121a**（消费级 Blackwell —— GeForce RTX 50 系列，GB20x）新增一个
开源、源码级的 **MXFP8** prefill / ragged attention kernel。这里的 MXFP8 指 OCP 微缩放
FP8 格式：`e4m3` 数据 + 每 32 元素一个 `ue8m0` block scale，FP32 累加。

FlashInfer 目前没有任何能跑在 SM120 上的 block-scaled FP8 attention。本 RFC 提出补上它，
动机是下面这条在消费级 Blackwell 上**实测**到的硬件特性。

## 动机

### 1. block-scaled 张量路径绕开了消费级 FP32 累加 throttle（实测）

消费级 Blackwell 的 FP32 累加张量吞吐被砍半 —— 但**只砍 legacy warp-MMA 指令**，
**block-scaled 张量指令不被砍**。

寄存器内 MMA microbench，RTX 5060 Ti（SM120，GB206，36 SM，CUDA 13.3），纯张量核发射率、
零访存：

| 输入 / 累加 | SASS 指令 | TFLOP/s |
|---|---|---:|
| FP16 / FP16 累加 | `HMMA.16816.F16` | ~103 |
| FP16 / FP32 累加 | `HMMA.16816.F32` | ~51 |
| FP8 e4m3 / FP32 累加（plain） | `QMMA.16832.F32.E4M3.E4M3` | ~102 |
| FP8 e4m3 / FP16 累加（plain） | `QMMA.16832.F16.E4M3.E4M3` | ~206 |
| **MXFP8 e4m3 + ue8m0 block-scaled / FP32 累加** | **`QMMA.SF.16832.F32.E4M3.E4M3.E8`** | **~202** |

关键比值：

- **MXFP8 block-scaled / BF16(FP32 累加) = 3.95×**（202 vs 51 TFLOP/s）。
- legacy FP32 累加 throttle = 2.0×（`HMMA.16816.F16` vs `.F32`），它同样把 *plain* FP8
  砍半（`QMMA...F32` 102 vs `...F16` 206）—— **唯独不砍 block-scaled 路径**
  （`QMMA.SF` 在 FP32 累加下仍 ~202）。

也就是说：消费级 Blackwell 上，**只有 block-scaled MXFP8 MMA 能同时拿到满速张量吞吐
和 FP32 累加**。per-tensor / plain FP8 attention（走 plain `mma.sync`）因为吃 FP32 累加
throttle，只能到 2×。

该测量是验证过的，不是推断：
- SASS 显示 `QMMA.SF.16832.F32.E4M3.E4M3.E8` 在计时循环内、累加器有真实读改写依赖链
  （排除 DCE）。
- 正确性校验（A=B=1.0，scale=1.0）所有输出 lane 返回精确的 `K = 32` 规约。
- 在 ILP（8 / 16 个独立累加器）与不同循环次数下稳定一致。

复现代码见文末。

### 2. 今天 SM120 上没有源码级的 block-scaled FP8 attention

| 现有路径 | 源码？ | 跑 SM120？ | 缩放方式 |
|---|---|---|---|
| `hopper/quantization/prefill_sm90` | 源码 | 否（仅 SM90） | dequant→BF16，非原生 FP8 MMA |
| `fmha_v2` `e4m3_fp32_*_sm120` | 源码 | kernel 能编，但 **Python API 禁用** | per-tensor 标量（`scale_bmm1/2`） |
| trtllm-gen FMHA | 预编译 cubin | **否** —— runner 断言 `mSM == kSM_100 \|\| kSM_103` | block-scaled，仅数据中心 |
| `mxfp8_gemm_cutlass_sm120` | 源码 | 是 | `ue8m0` block-32 —— **仅 GEMM，从未接到 attention** |

所以庞大的消费级 Blackwell 用户群没有开源、可调、block-scaled 的 FP8 attention——尽管
MMA atom、block-scaled layout、`ue8m0` scale 处理都已在树内（MXFP8 GEMM 里），而且有一个
可工作的 SM120a block-scaled attention 参考（SageAttention3，NVFP4 版）。

## 设计方案

**目标：** 仅 SM120a / SM121a。warp 级 block-scaled
`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0`
（SASS `QMMA.SF`），cluster 1×1×1 —— 消费级 Blackwell 的 warp 路径，**不是** SM100/103 的
tcgen05 / tensor-memory 路径。

### 实现底座：CUTLASS C++ (CuTe)，不是 CuTe DSL

我们要的 atom 在 CUTLASS C++ 里现成存在：

```cpp
cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t,   // A = Q / P
    cutlass::float_e4m3_t,   // B = K / V
    float,                   // FP32 累加
    cutlass::float_ue8m0_t,  // block scale
    /*SFVecSize=*/32>
```

（`cute/arch/mma_sm120.hpp`，含完整 `{e2m1,e2m3,e3m2,e4m3,e5m2}²` 矩阵 +
`sm120_make_smem_layout_sfa/sfb` helper。）CuTe **DSL** 在 SM120 *warp* 路径只接了 FP4 的
block-scaled op（`MmaMXF4Op`、`MmaMXF4NVF4Op`）；它的 `MmaMXF8Op` 在 SM100 *tcgen05* 路径上，
SM120 调不到。所以 MXFP8 atom 在 C++ 里能调、DSL 里调不到 —— kernel 走 C++ CuTe。

**Toolchain 已在消费级 Blackwell 验证：** 上游 SM120 MXFP8 GEMM 示例
（`79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm`，用 `mx_float8_t<e4m3>` +
`OpClassBlockScaledTensorOp` + `Sm120` + `LayoutSFA/SFB` 交错 scale 布局）在 RTX 5060 Ti +
CUDA 13.3 上编译并 `Disposition: Passed` —— atom、`ue8m0` scale 布局、元素类型在目标硬件上都已证明可用。

### 架构：hybrid（block-scaled 芯 + FlashInfer 壳）

| 层 | 来源 | 说明 |
|---|---|---|
| block-scaled 融合 mainloop —— TMA warp-specialized load → QKᵀ 块缩放 MMA → online softmax → **kernel 内把 P requant 成 `e4m3` + `ue8m0`** → PV 块缩放 MMA | 改编 **SageAttention3** 的 SM120a attention mainloop | 唯一已有的 SM120a block-scaled *attention*；NVFP4 → MXFP8 retarget |
| MMA atom + SF smem 布局 | **上游 CUTLASS**（`SM120_16x8x32_TN_VS<e4m3,…>`、`sm120_make_smem_layout_sfa/sfb`）| 替换 SageAttention 手写的内联 PTX NVFP4 atom |
| K/V load、scheduler/plan、算子注册、paged-KV（后续）| **FlashInfer**（`blackwell/collective/*load*`、`gather_tensor.hpp`、`plan.cuh`）| SageAttention 缺的部分；paged-KV 推到 Phase 2 |

- **数据类型：** Q/K/V 用 `e4m3` + `ue8m0` block-32 scale；P 在 kernel 内量化为
  `e4m3` + `ue8m0` block scale；全程 FP32 累加（QKᵀ 与 PV 都走 `QMMA.SF` —— 这正是 kernel
  保住满速吞吐的原因）。
- **retiling 提示：** atom 形状与 SageAttention3 的 FP4 路不同（FP4 是 `16x32x64`,
  `scale_vec::4X`；MXFP8 是 `16x8x32`, `scale_vec::1X`），所以 `TiledMMA`、SF 布局、K 循环
  都要重新推，不是改类型。
- **SMEM 预算：** SM120 只有 ~99 KB shared memory（SM100/A100 是 160+ KB）；tile 跟 SageAttention3
  的 SM120a 调优形状走，别照搬 SM100。

## API 与落地

按 `CONTRIBUTING.md`：

- kernel：`include/flashinfer/attention/blackwell/quantization/`
- 注册 / 绑定：`csrc/`
- Python 接口：`flashinfer/`
- JIT module 注册 `supported_major_versions=[12]`
- 测试 `tests/`，benchmark `benchmarks/`

## 精度

对 BF16 参考做验证，覆盖 head_dim ∈ {64, 128}、causal 与 non-causal（相对误差 / 余弦
相似度）。MX block-32 缩放预期对 K/V outlier 的跟踪显著优于现有 `fmha_v2` 的 per-tensor
标量缩放。

## 范围 / 非目标（Phase 1）

- **包含：** single + ragged prefill，`e4m3`，head_dim 128，SM120a / SM121a。
- **后续：** paged-KV，SM100/103，decode，FP6/FP4 混合输入。

## 待定问题

- **两个 matmul 之间的 P 量化** —— 主要工程风险：把 softmax 概率 P requant 成 `e4m3` +
  `ue8m0` 的粒度与 block 布局，使 scale 向量与 PV MMA 的 SFB 布局对齐；以及 kernel 内
  requant + 算 scale 的开销。
- **`16x8x32` MXFP8 atom 的 K 循环 / SF 布局重推**（SageAttention3 是围绕 `16x32x64` FP4 atom 建的）。
- **与 FlashInfer prefill scheduler / plan kernel 及 ragged/varlen KV-load 的集成**；K/V load
  这一 stage 能否抽出来，让 paged-KV（Phase 2）通过 `gather_tensor` + page table 直接插入。
- **~99 KB SMEM 预算下 head_dim 64 vs 128 的 tile 形状。**

## microbench 复现

```cuda
// nvcc -gencode arch=compute_120a,code=sm_120a -O3 mma_peak.cu -o mma_peak
// 注意：-arch=sm_120a 不会把 ptxas target 设成 sm_120a，block-scaled 指令会报
//       "not supported on .target 'sm_120'"。必须用 -gencode arch=compute_120a,code=sm_120a。
asm volatile(
  "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col"
  ".f32.e4m3.e4m3.f32.ue8m0 "
  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
  "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
  : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
  : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
    "r"(sfa),"h"(bidA),"h"(tidA), "r"(sfb),"h"(bidB),"h"(tidB));
```

---

## 内部备注（不要发出去）

- **为什么标题/动机只说"MMA 吃 4×"而不提端到端 kernel：** 维护者一看 MMA 层的
  `QMMA.SF` 4× + FP32 累加，就立刻明白对 attention 的价值；真实 kernel 因为有 softmax /
  TMA / online rescale / P 量化等非张量核开销吃不满 4×，这是常识，他们清楚，不必在 RFC 里
  自我设限或解释。我们只陈述实测到的 MMA 事实，价值让他们自己推。
- **本机环境：** RTX 5060 Ti / sm_120 / GB206 / 36 SM / CUDA 13.3 / nvcc 13.3，可直接编译跑。
- **实验产物：** `bench/mma_peak.cu`（五路对比，用 `-gencode arch=compute_120a,code=sm_120a`
  编）、`bench/mxf8_verify.cu`（正确性=32）、`bench/mma_mxf8.cu`（单路）。
- **本来差点写错的结论：** 中途按 plain `mma.sync` 推出"只有 2×"。陷阱在于真实 MXFP8
  kernel 走的是 block-scaled `QMMA.SF`，正好绕开 throttle，所以是 4×。你最初的直觉是对的。
- **框架决策（C++ CuTe，不是 DSL）：** DSL 在 sm120 warp 路径只接了 FP4；`MmaMXF8Op` 在
  tcgen05(sm100) 那侧，sm120 调不到。C++ 有现成 `SM120_16x8x32_TN_VS<e4m3,e4m3,f32,ue8m0,32>`
  (`mma_sm120.hpp:2736`)。详见记忆 `design-decision-cpp-cute-not-dsl`。
- **toolchain 已验证：** CUTLASS example `79c`（sm120 MXFP8 GEMM）在本卡编译+跑+校验通过，
  4096³ = 159 TFLOPS。证明官方 atom + SF 布局 + 元素类型在硬件上可用。注意：别用大尺寸开
  verification 跑，host 端窄精度参考校验会卡死。编译产物在 `bench/cutlass_ex/`。
- **本地源码：** `tmp/cutlass`（@b46b16d）；wheel 自带 CuTeDSL + blackwell 示例在
  `~/vllm-omni/.venv/.../flashinfer/data/cutlass/`。example 79c 的 `mx_float8_t<e4m3>` /
  `LayoutSFA/SFB` / `Sm1xxBlkScaledConfig` 可当类型/布局模板。
- **下一步（真正第一个编码动作）：** 用 `make_tiled_mma(SM120_16x8x32_TN_VS<e4m3,...>{}, ...)`
  搭 QK 的 TiledMMA，喂单个 `[128×hd]×[hd×128]` tile，确认 attention 形状下能编、数值对。
- **下一步可选：** 若要更硬的支撑，可补一条带访存的真 GEMM 对比（树里 `mxfp8_gemm_cutlass_sm120`
  vs cuBLAS bf16），作为 RFC 的 Phase 0 验证；但你已说明维护者清楚这点，非必需。
