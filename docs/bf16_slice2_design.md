> **⚠️ SUPERSEDED (2026-06-21, user-confirmed).** This fused in-attention bf16→mxfp8 quant
> design is PARKED. Rationale: (1) in-attention KV quant has no production-prefill precedent
> (only P→fp8 does — Sage/FA3/FlashInfer all pre-quantize K/V); (2) we own prefill only, and a
> quantized KV cache is a prefill+decode contract we can't introduce unilaterally. Decision:
> bf16→mxfp8 quant lives OUTSIDE attention (a separate op / K/V projection-GEMM epilogue, à la
> vLLM #32220); our existing `kMxFp8` reads the result. Kept for the staging/quant-arithmetic
> analysis, which a standalone quant op may reuse. See memory `vllm-nvfp4-kv-cache-landscape`.

# S6b slice-2 design — bf16 KV cache via in-kernel dynamic block-quant

Design note for the second (and harder) half of #22 (S6b: KV dtype contract). Slice-1
(`kUniformFp8`, committed `078183a`) made the **strict-superset** claim concrete: a
per-tensor fp8 cache is eaten with zero data conversion. Slice-2 adds the
**differentiator**: a **bf16** KV cache that the kernel quantizes to mxfp8 *in-kernel*,
per-32 block, calibration-free. No compile yet — this fixes the architecture before
coding, in the style of `docs/mainloop_design.md`.

References studied first (so we know what is borrowed vs novel):

| Reference | What it does with two KV dtypes | Relevance |
|---|---|---|
| **FlashInfer FA2** prefill (`prefill.cuh`) | fp8 KV is **dequantized UP to bf16** (`repack_fp8_tile_to_bf16` staging buffer, or in-loop), then a **16-bit MMA** runs. Scales come from OUTSIDE via `maybe_k/v_cache_sf`. **Never quantizes in-kernel.** | Opposite direction (our MMA is mxfp8-only). Borrow only the **staging-buffer structure** (load wide tile → smem → convert per-tile → standard downstream). |
| **SageAttention3** Blackwell nvfp4 (`mainloop_tma_ws.h`, `softmax_fused.h`) | K/V arrive **already nvfp4** — pre-quantized by a SEPARATE kernel `fp4_quantization_4d.cu`. Only **P** (softmax output) is quantized in-kernel (`online_softmax_with_quant`), staying in **registers** (no smem round-trip). | Confirms: **no production kernel does in-kernel KV quant.** Our bf16→mxfp8 KV block-quant is **novel**. Sage3's P-quant is the closest mechanism, but its amax needs `__shfl_xor` (P lives in MMA-accum layout); ours is **simpler** (staged contiguous). |

**Net:** the in-kernel KV quant has no template — it is exactly mxfp8's edge over
per-tensor fp8 (memory `kv-cache-dtype-compat-mxfp8-superset`). But every *piece* we need
already exists and is validated, so this is assembly, not invention:

* quant arithmetic — `quant_e4m3(v, scale_exp)` + `mx_scale_exp(amax)` (`s3_kernel.cuh:256-267`),
  already torchao-validated via the dynamic-P path.
* SF-into-fragment — the dynamic-P **linear-buffer + identity-coord gather** pattern
  (`sSFP[q*NKB+sfi]` written at `:604`, gathered via `sfp_coord` at `:626`).
* `partition_SFA` / `partition_SFB` for the gather coords (`sm120_mxfp8_mma.cuh:91,99`).

---

## 1. Axis: `SFSource::kDynamicBf16` (DTypeKV folded onto the existing switch)

Slice-1 added `enum class SFSource { kMxFp8, kUniformFp8 }`. Slice-2 adds a third value
`kDynamicBf16`. It is more than an SF source (the DATA element changes e4m3→bf16), but
folding it onto the same template axis keeps **one** dispatch and one kernel.

```cpp
enum class SFSource : int { kMxFp8 = 0, kUniformFp8 = 1, kDynamicBf16 = 2 };

template <SFSource S>
using DTypeKVFor = cute::conditional_t<S == SFSource::kDynamicBf16,
                                       cutlass::bfloat16_t, Element /*e4m3*/>;

constexpr bool kLoadSF  = (Src == SFSource::kMxFp8);        // gate the SF TMA  (unchanged)
constexpr bool kQuantKV = (Src == SFSource::kDynamicBf16);  // gate the bf16 staging + quant
```

Three SF regimes (consumer K/V/Q SF fill becomes a 3-way `if constexpr`):

| Src | data TMA | SF TMA | SF in fragment |
|---|---|---|---|
| `kMxFp8` | e4m3 | yes | copy from BlkSF smem |
| `kUniformFp8` | e4m3 | no | fill byte 127 |
| `kDynamicBf16` | **bf16** | no | **gather from in-kernel `sSF*_lin`** |

**Params stays un-templated.** Add bf16 TMA descriptors as *parallel fields*:

```cpp
struct Params {
  TMA_Q  tma_q;  TMA_K  tma_k;  TMA_V  tma_v;     // e4m3 (kMxFp8 / kUniformFp8)
  TMA_SFQ tma_sfq; TMA_SFK tma_sfk; TMA_SFV tma_sfv;
  TMA_Qb tma_qb;  TMA_Kb tma_kb;  TMA_Vb tma_vb;  // bf16  (kDynamicBf16), default-ctor'd otherwise
  ...
};
```

Cute TMA copy atoms are default-constructible (POD `CUtensorMap` inside), so `s3_e2e.cu`
and `s6a_ragged.cu` (field-assignment, e4m3 only) leave the bf16 fields zeroed and
**compile unchanged** — verified earlier that those callers use field-assignment, not
positional aggregate init. The kernel reads `params.tma_kb` only under `if constexpr (kQuantKV)`.

---

## 2. smem plan — reuse the `sK` region as a universal bf16 staging (zero added smem)

Measured: `sizeof(SharedStorage)=84992 B (83 KB)`; `sharedMemPerBlockOptin=101376 B (99 KB)`.
A full `128×128` bf16 tile is **32768 B = exactly the `sK` region** (the 2-stage e4m3 K ring).

In the default **S5** build the `sP` storage (16 KB) is **dead** — only its *layout type*
is used (`partition_fragment_A(sP)` shapes a register fragment; the data path is the
register `__shfl`, `s3_kernel.cuh:630-685`). That frees 16 KB. So:

```
bf16 path (Src == kDynamicBf16), reusing existing buffers — NOTHING added:
  sK region  (32 KB) ← bf16 staging   (loads K, then reused for V; and Q at start)
  sP region  (16 KB) ← e4m3 K  (QK B-operand source — the ONLY repointed read)
  sV region  (16 KB) ← e4m3 V  (PV B-operand — unchanged home)
  sQ region  (16 KB) ← e4m3 Q  (QK A-operand — unchanged home)
  sSF*_lin   ~1.5 KB ← union over the bf16-idle BlkSF SF buffers (sSFK/sSFV/sSFQ)
  --------------------------------------------------------------------------
  sizeof unchanged = 83 KB  (e4m3 path byte-identical; bf16 path well under cap)
```

Rejected alternatives:
* **Dedicated 32 KB staging** → `sizeof` hits 101376 = optin exactly (no slack for the
  linear SF, launch-margin risk). 
* **bf16 smem ring (register-quant, no e4m3 dest)** → bf16 storage everywhere doubles the
  data footprint (K+V+Q rings) far past 99 KB.

**v1 cost:** the shared staging is single-buffered, so the bf16 K-load / K-quant /
V-load / V-quant serialize (no load↔compute overlap). Acceptable for first-correct; the
perf follow-up is a 2-deep **half-tile** ring (2×16 KB in the same `sK` region) restoring
overlap. Constraint: bf16 path supports the **S5/default** build only (the `S3_P_SMEM`
oracle path actually uses `sP`, so it stays e4m3-only — `static_assert` it).

---

## 3. The producer/consumer handshake (the deadlock-prone core)

The shared staging forces a strict order: the producer must not overwrite the staging
with V's bf16 until the consumer has finished quantizing K out of it. This maps cleanly
onto a **1-stage pipeline on the staging buffer** — the standard
`producer_acquire`/`consumer_release` handshake *is* the serialization:

```
producer (wg0)                         consumer (wg1)
──────────────                         ──────────────
Q: acquire stg; TMA bf16 Q→stg; commit
                                       wait stg; quant stg→sQ + SFQ_lin; release stg
for nb in [0, n_block_max):            for nb in [0, n_block_max):
  acquire stg; TMA bf16 K→stg; commit
                                         wait stg; quant stg→sP + SFK_lin; release stg
                                         QK from sQ·sP  (downstream UNCHANGED)
  acquire stg; TMA bf16 V→stg; commit
                                         wait stg; quant stg→sV + SFV_lin; release stg
                                         PV from sV     (downstream UNCHANGED)
```

**[staging-trip contract]** — same hazard class as `[n_block_max contract]` /
`[SF-bytes contract]`: the producer commits and the consumer waits **the same number of
times** on the staging pipeline = `1 + 2·n_block_max`. Diverge → deadlock (100% GPU,
no output), not a wrong number. The bf16 path replaces the separate `PipeK`(2)/`PipeV`(1)
with **one** `PipeStage`(1); `transaction_bytes = bf16-tile bytes` (data only, no SF).

`n_block_max` already carries the causal/offset bound and is shared producer↔consumer
(`s3_kernel.cuh:363`), so causal/varlen/GQA ride along unchanged — only the *load body*
inside the loop changes for bf16.

---

## 4. The quant pass (consumer, cooperative, no cross-thread shuffle)

After `wait stg`, the 256 consumer threads cooperatively quantize the staged bf16 tile
into the e4m3 dest + the linear SF. Each thread owns whole **(row, 32-block)** units →
amax is a **serial scan of 32 contiguous** staged elements, **no `__shfl`** (this is the
simplification over Sage3's P-quant, whose 16-elem block straddles 2 threads):

```cpp
// kQuantKV, per operand. `Stg` = bf16 view over the sK-region; `Dst` = e4m3 view (sP/sV/sQ);
// `sSF_lin` = linear ue8m0 [row * NBLK_dim + block].   NBLK along the contraction dim.
for (int u = tid; u < kRows * NBLK_dim; u += NumMmaThreads) {
  int row = u / NBLK_dim, blk = u % NBLK_dim;
  float amax = 0.f;
  for (int j = 0; j < SFVecSize; ++j) amax = fmaxf(amax, fabsf(float(Stg(row, blk*32+j))));
  int se = mx_scale_exp(amax);                                   // ue8m0 exponent (validated)
  for (int j = 0; j < SFVecSize; ++j) Dst(row, blk*32+j) = quant_e4m3(float(Stg(...)), se);
  sSF_lin[row * NBLK_dim + blk] = ElementSF::bitcast(uint8_t(se + 127));
}
cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();  // Dst + SF visible
```

* **K** contraction = head_dim → `NBLK_dim = kHeadDim/32 = NBLK`, `kRows = kBlockN` keys.
* **V** (Vt `[head_dim, keys]`) contraction = keys → `NBLK_dim = kBlockN/32 = NKB`,
  `kRows = kHeadDim`. Staged-contiguous along keys (Vt inner dim) → still serial amax.
* **Q** contraction = head_dim, `kRows = kBlockM`.

`Dst` write uses the e4m3 swizzled layout (sP/sV/sQ), so the existing
`partition_fragment_B/A` + ldmatrix downstream is **byte-identical** to the mxfp8 path.

**SF feed = 3rd branch of the existing K/V/Q SF fill** (alongside `kLoadSF`→copy,
`kUniformFp8`→byte127). Gather from `sSF*_lin` by logical coord, exactly like dynamic-P:

```cpp
else if constexpr (kQuantKV) {                       // K example (SFB in QK)
  Tensor sfk_coord = mxfp8::partition_SFB(
      make_identity_tensor(make_shape(Int<kBlockN>{}, Int<kSFPadHD>{})), thr_qk);
  for (int i = 0; i < size(tSrSFK); ++i) {
    auto c = sfk_coord(i);
    tSrSFK(i) = ss.sSFK_lin[int(get<0>(c)) * NBLK + int(get<1>(c)) / SFVecSize];
  }
}
```

Q uses `partition_SFA` over `(kBlockM, kSFPadHD)`; V uses `partition_SFB` over
`(kHeadDim, kBlockN)` into `sSFV_lin` (`head_dim * NKB`). `kSFPadHD` keeps hd<128 working
(memory `s3-head-dim-64-sf-padding`) — the gather just reads the first `NBLK` real blocks.

---

## 5. Scales, masking, LSE — fold-through (mostly free)

* **No per-tensor scalar for bf16.** bf16 carries full dynamic range; the per-32 block
  exponent IS the scale. `sm_scale` unchanged, `o_scale = 1.0` (slice-1's epilogue knob
  stays at its default). bf16 is *calibration-free* — that is the whole point.
* **Partial-block / masking:** the `S3_V_KFILLZERO` guards (memory
  `s6a-v-kfillzero-and-nan-blind-test`) still apply to the bf16 V DATA tail. But note the
  bf16 staging tail is **finite bf16** (zero-padded by TMA), so the `0*NaN` SF hazard is
  weaker — the computed SF of an all-zero block is `mx_scale_exp(0)=-127` (finite), no
  poisoned NaN. Keep the kFillZero data-zeroing; the SF-finite-ize becomes a no-op.
* **LSE** unchanged (`row_max*sm_scale + log(row_sum)`); the in-kernel quant only perturbs
  the data numerically, not the softmax bookkeeping.

---

## 6. Validation plan (oracle + differential + regression)

1. **torchao bf16 oracle** (`tests/test_mxfp8_prefill.py`): take bf16 Q/K/V, run the
   reference as `to_mx`-quantize each per-32 (the SAME quant the kernel does) → dequant →
   attention. Assert the kernel matches (this is a *self-consistent* oracle — the in-kernel
   quant must equal torchao `to_mx`, which `quant_e4m3`/`mx_scale_exp` already passes for P).
   Dense non-causal hd128 first, then causal, then partial-kv.
2. **`s6a_ragged.cu` differential:** add a bf16 mode; the bf16 result is NOT bit-exact to
   e4m3 (different quant), so compare bf16-kernel vs a **per-request bf16 reference replay**,
   and keep the **e4m3 path bit-exact 24/24** (the new template value must not perturb it).
3. **Regression:** mxfp8 torchao 7/7 + per-tensor 3/3 unchanged (e4m3/uniform paths byte-identical).

Slice order: 2a (plumbing, e4m3 stays green) → 2b (quant + gather, dense) → 2c (oracle +
extend causal/partial/varlen/GQA). Commit only when asked.

---

## 7. Open risks

* **`Dst` layout compat.** e4m3 K written into `sP` (`SmemLayoutP`) must be MMA-read-compatible
  when QK-B repoints `sK→sP`. Both are `sm120_rr_smem_selector<e4m3,128>` over `(128,128)` →
  expected identical swizzle, but **verify the QK numerics** on the very first dense run
  (a swizzle mismatch shows as garbage O, not a deadlock).
* **Staging-trip contract.** Mis-counting the `1+2·n_block_max` handshake deadlocks. Mirror
  the `[n_block_max contract]` discipline: edit producer + consumer trip counts together.
* **bf16 TMA descriptor.** New `TMA_Kb/Vb/Qb` over a bf16 gmem tensor + a bf16 staging
  `SmemLayout` (swizzle from `sm120_rr_smem_selector<bfloat16_t, …>`); box = full tile.
* **Perf.** v1 serializes load↔quant↔compute (single-buffered staging). Expect a kernel-time
  regression vs the pre-quantized e4m3 path; the half-tile ring (§2) is the follow-up. Only
  the MMA-hits-4x claim is load-bearing for the RFC — never claim bf16 end-to-end parity.
