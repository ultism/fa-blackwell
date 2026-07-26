# 5090 证据：s3 MXFP8 prefill vs fa2（RTX 5090, sm_120）

**日期**：2026-07-26 · **机器**：vast-node-5090（RTX 5090 32GB, driver 580.95.05, CUDA 13.0, ncu 13.x）
**方法**：`prof/trace_5090_ab.sh`（双侧同一 varlen batch；事件计时 = 20 warmup + 100 timed × 5 交替 A/B 轮取 min；ncu `--set full --clock-control none` 抓稳态单 launch）
**fa2 基线**：flashinfer（fork 分支 main@fa2672f 合并版）`BatchPrefillWithRaggedKVCacheWrapper(backend="fa2")` JIT，`prof/fa2_ab_prof.py`
**ours**：`tests/bench_ragged.cu`（s3_kernel S9f，kUniformFp8 per-tensor fp8）

## 结果

| shape（同 batch 双侧） | ours ms (ncu µs) | fa2 ms (ncu µs) | **加速比** | ours TFLOP/s | fa2 TFLOP/s | ours tensor% | fa2 tensor% |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev8x2：16 请求混合 varlen, Hq8/Hkv2 causal | 0.1414 (144.96) | 0.3881 (387.17) | **2.74×** | 413 | 150 | 46.9 | 32.5 |
| dev32x8：同 batch, Hq32/Hkv8 causal | 0.5662 (575.33) | 1.1648 (1180) | **2.06×** | 413 | 201 | 49.3 | 42.7 |
| qwen8b：8×2048, Hq32/Hkv8 causal | 0.6143 (627.26) | 1.3173 (1320) | **2.14×** | 447 | 209 | 51.5 | 44.0 |
| long1：单请求 16384, Hq16/Hkv4 causal | 2.2341 (2160) | 5.4336 (5400) | **2.43×** | 492 | 202 | 54.8 | 41.5 |

- 事件计时（min-of-5×avg100）与 ncu `gpu__time_duration` 互洽（偏差 <3%）。
- TFLOP/s = 有效 FLOPs（4·Hq·D·L²，causal 减半）/ 事件计时。
- tensor% = `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`。

## 结论（对应 RFC #3628 的论点）

1. **fa2 已顶到它自己的硬件天花板**：fa2 三档大形状都落在 ~200–209 TFLOP/s，
   即其 `HMMA.16816.F32`（FP32-acc bf16）通路上限的 84–88%。该上限经两项独立
   测量坐实（不再依赖 5060Ti 外推）：
   - **mma_peak 寄存器微基准（GB202 实测）**：FP16/F32-acc **253.0** TFLOP/s，
     FP16/F16-acc 505.3，plain FP8/F32-acc 505.8，**MXFP8 块缩放/F32-acc 992.9**
     ——FP32-累加减半节流在 GB202 同样存在（比值 3.92×，与 GB206 相同）。
   - **cuBLAS（8192³，GB202 实测）**：bf16→F32 **237.4** TFLOP/s（= 节流通路
     roof 的 94%，生产级 GEMM 确认）；fp8 e4m3→F32 826.1 TFLOP/s。
2. **ours 同精度语义（FP32 累加）下走块缩放通路，413–492 TFLOP/s = fa2 的 2.0–2.4×**，
   tensor pipe 占用 47–55%——离块缩放上界（~993 TFLOP/s @GB202）还有近一倍余量。
3. 加速比随 batch/并行度收敛（2.74×→2.06×）：大 batch 下 fa2 填满 SM 后只能靠
   指令速率差（~4× 上限比 × 双方利用率比 ~50%/85% ≈ 2.2×）。
4. **注意（metric 语义，已用 ncu + mma_peak 双向验证）**：FP32-acc 节流在两个 die 上
   机制不同。纯寄存器 MMA 满速循环的 `sm__pipe_tensor_cycles_active` / `issue`：
   - GB206（5060 Ti）：HMMA.F32 = **96.5% / 48.2%（占用 : 发射 ≈ 2:1）**——数据通路
     减半，每条指令在 pipe 里占 2 拍；HMMA.F16 = 97.3/97.3。
   - GB202（5090）：HMMA.F32 = **49.8% / 49.8%（1:1）**——发射配给，指令 1 拍走完但
     隔拍才放行；HMMA.F16 与 QMMA.SF = 99.6/99.6。
   - 两 die 净效果相同（F32-acc 吞吐减半），QMMA.SF 块缩放在两 die 均全速。
   因此 `pipe_tensor_cycles_active` 对 HMMA.F32 跨 die 不可直接比（fa2 在 GB206 显示
   ~87%、GB202 显示 ~43%，对应的是同一相对水平 ~85% roof）；同 die 双侧对比有效。
   我们的 QMMA.SF 路径该指标跨 die 可直接比（两机均 ~50%）。

## 原始数据

- mma_peak（GB202）：`bench/mma_peak.cu`，输出见上（170 SMs, grid=680×256thr）
- cuBLAS（GB202）：`bench/cublas_peak.cu`（bf16→F32 用 `cublasGemmEx/CUBLAS_COMPUTE_32F`；
  fp8→F32 用 `cublasLt/CUDA_R_8F_E4M3`）

- ncu reps：5090 机 `fa-blackwell/prof/5090/{ours,fa2}_{dev8x2,dev32x8,qwen8b,long1}.ncu-rep`
- 事件日志：5090 机 `fa-blackwell/prof/5090/events_*.log`
- 5060Ti 留档（同 dev8x2 batch）：本仓 `prof/{fa2_8x2_causal,real64_8x2_causal,s9f_8x2}.ncu-rep`（未入库，*.ncu-rep 被 gitignore）
