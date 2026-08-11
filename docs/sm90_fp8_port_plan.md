# SM90 FP8 port — s3 ragged prefill kernel on H20 (as-built record)

Port of the SM120a MXFP8 ragged prefill kernel (`tests/s3_kernel.cuh`, the
ragged varlen + GQA + per-tensor-fp8 `kUniformFp8` path driven by
`tests/csrc/mxfp8_ragged_kernel.cu` / `tests/test_ragged_ext.py`) to **SM90a**
(H20), with the block-scaled QMMA compute core replaced by a **plain FP8 (e4m3)
`mma.sync`** atom. Single source serves both arches; the sm120 build is
bit-exact unchanged.

**Status: DONE, validated.** H20 (8×H20-3e, vllm/vllm-openai:v0.26.0 image,
torch 2.11.0+cu130): `TEST_RAGGED_EXT PASS`, O max|abs| vs fp64 oracle =
**2.41e-3** (sm120 build on 5060 Ti: 2.28e-3, unchanged). Gate 5e-3.

Workflow: cross-compile on the dev box (nvcc 13.3, torch cu130 venv,
`-gencode arch=compute_90a,code=sm_90a`), upload the `.so`, run inside the
vllm 0.26.0 image via `S3_EXT_SO` (see tests/test_ragged_ext.py). Local 5060 Ti
(sm120) cannot execute sm90a cubins; H20 cannot build (no toolkit) — hence the
split.

---

## 1. Compute-core swap (the only functional change)

```
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
using AtomMXF8 = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;   // mma.sync m16n8k32 e4m3, sm89+ PTX
#else
using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<e4m3,e4m3,f32,ue8m0,32>;
#endif
```

Why `mma.sync`, not wgmma: the mainloop is already warp-level shaped (LDSM +
interleaved copies + `cute::gemm` on register fragments). wgmma would be a
rewrite (async warpgroup MMA, smem descriptors). v2 upgrade path, out of scope.

**Key enabler (verified in cutlass traits): the two atoms' A/B/C thread-value
layouts are bit-identical** — `SM89_16x8x32_F32E4M3E4M3F32_TN`
(mma_traits_sm89.hpp:54) and the SM120 VS atom (inherited from
`SM80_16x8x32_S32S8S8S32_TN`, mma_traits_sm120.hpp:121) share:
- ALayout `((4,8),(4,2,2)) : ((64,1),(16,8,256))`
- BLayout `((4,8),(4,2)) : ((32,1),(8,128))`
- CLayout `SM80_16x8_Row`

Consequence: **no LayoutP re-derivation** (contra the original plan §2.4) — the
S5 quad-shuffle that redistributes QK-accumulator P into the PV A-operand works
verbatim, as do all LDSM retiles, masking, softmax and the epilogue.

Host-pass note: `__CUDA_ARCH__` is undefined in the host pass, which therefore
keeps the SM120 types (unguarded in cutlass); the SM89 atom/traits exist in all
passes, so the dispatch needs no build flag.

## 2. SF layer excision (`#if !S3_PLAIN_FP8_MMA` guards, sm120 path untouched)

Removed from the sm90 device pass only: `tSrSFQ/tSrSFK/tOrSFP/tOrSFV` fragments,
SF tiled copies, `sfp_coord`, `subSFK`, all `make_zip_tensor(data, SF)` gemms
(→ plain `cute::gemm(mma, A, B, C)`), the kUniformFp8 constant-byte SF fills,
the V-SF finite-ize block. Producer/consumer pipeline skeleton, TMA loads,
transaction-byte contract, tile scheduler, kFillZero V-data zeroing: unchanged.
The sm90 build `static_assert`s `Src == kUniformFp8`.

## 3. The one numerics gotcha: P's 2^-8 scale folding

P is quantized with a FIXED scale 256 (`kPScaleExp = -8`). On sm120 the SF byte
119 (=2^-8) is applied *inside* the block-scaled MMA. The plain e4m3 MMA has no
SF operand, so `accO = Σ(p·256)·V` comes out **256× too large**; sm120's
degenerate-SF trick does not carry over. Fix: fold `/256` into the epilogue
normalization (linear, once):

```cpp
float inv = o_scale / (row_sum[mi] * 256.f);   // S3_PLAIN_FP8_MMA only
```

Symptom before the fix: outputs exactly ~256× too large (first-H20-run error
772 ≈ 256 × 3). Q/K/V SFs are all 127 (2^0) on the kUniformFp8 path — P is the
only place the SF byte is semantically load-bearing.

## 4. What ptxas does on sm90 (perf note)

sm90 SASS has no FP8 HMMA encoding; ptxas lowers
`mma.sync.m16n8k32.f32.e4m3.e4m3.f32` to `F2FP.F16.E4M3.UNPACK_B` (exact
e4m3→f16 hardware unpack) + 2× `HMMA.16816.F32` (fp16 tensor cores, f32
accumulate). **Numerically exact** for e4m3 operands (e4m3 ⊂ f16, products
exact in f32) — the 2.41e-3 vs 2.28e-3 delta vs sm120 is the fp32-rounding
tail, not a format error. Perf: FP8 mma.sync on Hopper runs through the fp16
pipe with unpack overhead; full FP8 tensor throughput requires wgmma (v2).

## 5. Build & run

```bash
# dev box: cross-compile (torch cpp_extension; S3_GENCODE selects the arch)
S3_GENCODE="arch=compute_90a,code=sm_90a" TORCH_EXTENSIONS_DIR=~/.cache/torch_ext_sm90 \
  /root/vllm-omni/.venv/bin/python -c "import test_ragged_ext as t; t.ext()"

# H20: upload .so + test, run in the vllm 0.26.0 image (entrypoint is the vllm CLI)
docker run --rm --entrypoint python3 -v /root/sm90_test:/work \
  -e S3_EXT_SO=/work/mxfp8_ragged_ext.so vllm/vllm-openai:v0.26.0 /work/test_ragged_ext.py
```

ABI: local venv python 3.12/torch 2.11.0+cu130 == image python 3.12/torch
2.11.0+cu130 → the prebuilt pybind `.so` imports directly. nvcc 13.3 vs cu130
runtime: torch cpp_extension only hard-errors on CUDA **major** mismatch.

## 6. Regression guarantee

sm120 build (`tests/test_ragged_ext.py` default): PASS, max|abs| **2.28e-3 —
bit-identical** before/after the port (all sm90 edits are `#if`-guarded; the
only shared-code change is the epilogue `inv` ternary whose sm120 branch is the
original expression).

## 7. Follow-ups

- wgmma SS/RS compute core for real FP8 throughput (drop the S5 shuffle; P
  transits smem instead — FA3-fp8 / flashinfer hopper mainloop pattern)
- Bench on H20 vs vllm's fp8 prefill (bench_ragged harness) once wgmma lands
- sm120 kMxFp8 path is unaffected; sm90 kMxFp8 (real block-scaled SF) would
  need per-block SF applied in registers before the plain MMA — not planned
