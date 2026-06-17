# Gotchas — MXFP8 prefill attention for SM120a

Engineering notes for non-obvious traps found while building the kernel. Newest
first. Each entry: what bit us, the measured evidence, and the actionable rule.

---

## Measuring O error: normalize by max|O|, never per-element `clamp_min` (it explodes on near-zero O)

**TL;DR:** Attention output O has entries that are legitimately ~0 (rows where the
softmax-weighted V sum cancels). A per-element "relative error"
`(O - O_ref).abs() / O_ref.abs().clamp_min(1e-3)` divides a real ~0.05 absolute
quant error by the `1e-3` floor at those near-zero entries → a **bogus 49.5 ("4950%")**
that looks like catastrophic failure when the kernel is actually fine. We hit exactly
this comparing fixed-vs-dynamic P quant: the headline number was `49.5`, the honest
number was `~1%`.

**The trap:** `clamp_min(floor)` turns "small error at a small-and-unimportant O
entry" into a huge ratio, because the denominator stops shrinking but the numerator
doesn't. The near-zero O entries it amplifies are the ones that **don't matter
downstream** (they're dominated by accumulation noise and feed a near-zero
contribution to the next layer).

**Rule:** report the max-abs O error **normalized by the global scale `max|O|`**
(`(O-O_ref).abs().max() / O_ref.abs().max()`), or just the raw abs error alongside
the O range. That gives the true "fraction of the signal" error (~1%), and it's the
metric that actually predicts downstream impact. Reserve per-element relative error
for quantities with no near-zero entries (e.g. row_sum, LSE). Same reasoning is why
the kernel's own asserts use `assert_close(rtol, atol)` (atol catches the near-zero
entries) rather than a pure rtol.

---

## Softmax P: a fixed scale 256 beats per-block dynamic MX (dynamic MX saturates the top 1/8 of each binade)

**TL;DR:** The PV A-operand is the softmax output P. Post-max-subtraction `p =
exp2((S - m_cur)*sm)` with `m_cur` the running max ⇒ **every P ≤ 1.0 and the row
argmax = exactly 1.0**. So quantize P with a **fixed scalar scale = 256.0**
(`se = -8`, `P_e4m3 = P*256`) — drop the per-32-block `amax`/`quad_reduce` and the
SF smem store/gather entirely (`tOrSFP` = the compile-time constant byte `-8+127 =
119`). `1.0*256 = 256 ≤ 448` (e4m3 max) ⇒ **never saturates**.

**The surprise — fixed is MORE accurate than dynamic, not just cheaper.** Measured
(`test_attn_matches_torchao`, max-abs O error vs un-requantized P, normalized by
max|O|): fixed `0.55–1.08%` vs dynamic per-block `0.86–2.56%` across hd64/128 ×
causal. Dynamic OCP-MX picks `se = floor(log2(amax)) - 8`, mapping amax into
`[256, 512)` — but **e4m3 max = 448 = 1.75·256, not 512**. Any block whose amax is
in the top 1/8 of its binade (e.g. amax=0.9 → 0.9·512 = 460.8) **saturates to 448**,
clipping the *largest* (most important) P values. Fixed-256 puts the max at exactly
256 and never clips; it only loses the tiny elements of small-amax blocks (below
~2^-17 → 0), which contribute <0.01% to `O = ΣP·V/ΣP`. For any quantity bounded by
1, a static scale dodges MX's systematic top-of-binade saturation.

**Rule:** for softmax P (or any [0,1]-bounded operand) use a fixed scale, not MX
dynamic. Toggle `S3_P_DYNAMIC_SCALE` keeps the dynamic path for A/B only. (When
reporting the precision delta, normalize by `max|O|` — see the measurement-trap
entry above; per-element relative error gave a bogus `49.5` vs the honest ~1%.)

**Measured — but trust ncu, not wall-clock, on a 0.1 ms kernel.** `bench/bench_p_quant.cu`,
RTX 5060 Ti. The clock swings 1537↔2820 MHz between runs (timing sd up to 0.27 ms),
so a min-of-200 read 10%/23% with the causal number inflated by clock noise. The
honest evidence is **ncu `--clock-control base`**, deterministic `sm__cycles_elapsed`
(4096²): non-causal **−10%** (291.6k → 262.0k cycles), causal **−17%** (310.9k →
256.9k); instructions −15%/−12%; **occupancy identical** (18.7%, 168 reg, 1 CTA/SM) so
it's not an occupancy effect. Mechanism: the `amax` reduction is a warp `SHFL`
(`quad_reduce`) whose result is consumed immediately → a `short_scoreboard` stall
(0.53→0.32) that, at this kernel's low ~18.7% occupancy, isn't hidden and lands on the
QK→PV critical path. A 3rd build (`-DS3_P_CONST_SF=0`, fixed scale but SF still via
smem) shows the whole win is the amax removal — dropping the SF smem store/gather adds
~0. Two takeaways: (1) on sub-ms kernels with boosting clocks, profile cycles with ncu,
don't trust wall-clock min; (2) bench needs the debug dumps off — guard the full-P gmem
write with `if (params.out_Ppre != nullptr)` or it dominates the time.

---

## head_dim < 128: the block-scaled SF smem atom / TMA box is always 128 (pad it, don't shrink it)

**TL;DR:** The cutlass block-scaled SF tile is inherently **128 along its tiled
dim** (`Blk_MN=128`, `Blk_SF=4` ue8m0 per 128-element K-block). Hand-rolling the SF
smem atom with the real `head_dim=64` makes `sSF_shapeK = K/SFVecSize/Blk_SF =
64/32/4 = 0` → a malformed layout → the TMA fires `"CTA_Tile and SLayout top-level
size equivalence"` / `"Could not find a common tile-gmem vectorization"`
static_asserts. There is **no sub-128 SF atom** in cutlass — normal GEMMs always
tile K to 128, and the collective only special-cases *N* < 128 (`logical_divide`),
never *K* < 128.

**The fix is to PAD, not shrink.** `tile_atom_to_shape_SFA(make_shape(128,128,64))`
already returns the **128-K-padded** layout (cosize 512, byte-identical type to the
K=128 case); `tile_atom_to_shape_SFB` pads N<128 the same way. So use
`kSFPadHD = max(kHeadDim,128)` (=128) for the **SF smem atom, `LayoutSF`, and all
three TMA SF boxes** (SFQ/SFK K-dim, SFV N-dim), while the **data path**
(`SmemLayoutQ/K/Vt`, data TMA boxes, the QK `TiledMma` Tile) keeps the real
`kHeadDim`. The QK MMA contracts only 64 (`size<2>(tSrQ)=2`), so the gemm loop reads
the first 2 of the 4 SF blocks; the host fills `NBLK=kHeadDim/32` real blocks and
cutlass auto-pads the gmem buffer (the upper blocks are loaded-but-unused). The V-SF
box becomes (128,128) and the same `SmemLayoutSFV` serves both head dims.

**Trap:** the TMA-SF box is built in THREE files that must agree on the Params type —
`tests/s3_kernel.cuh` (the `TMA_SF*` aliases), `tests/s3_e2e.cu` main, and
`tests/csrc/mxfp8_attn_ext.cu`. Updating only two of three compiles fine at
head_dim=128 (everything is 128 anyway) and fails **only** at head_dim=64 with a TMA
type mismatch — and only in the torch-extension TU, not the plain-nvcc self-test.

---

## FMA contraction in the online softmax → max P = 0.99999988 → e4m3 saturates the max to 7/8

**TL;DR:** Writing the softmax as `p = exp2f(accS*sm_log2 - max_scaled)` with
`max_scaled = m_cur*sm_log2` lets nvcc contract it to `fma(accS, sm_log2, -max_scaled)`.
At the argmax `accS == m_cur`, but the FMA subtracts the **rounded** `max_scaled`
from the **unrounded** product `accS*sm_log2`, leaving the rounding error of
`m_cur*sm_log2` (~ -1.7e-7) instead of 0. So the softmax max comes out **0.99999988,
not 1.0**. Then the P-requant block scale `mx_scale_exp(0.99999988)` =
`floor(log2)-8` = `-9` (one too low vs `-8` for exactly 1.0), the max element scales
to `1.0·2^9 = 512`, **saturates to e4m3 max 448**, and dequantizes to
`448·2^-9 = 0.875`. Every row whose softmax-argmax 32-key block has amax == 1.0 comes
out uniformly **×7/8** (~37/128 rows on random data).

**Fix:** subtract the max BEFORE scaling — `p = exp2f((accS - m_sub)*sm_log2)`,
`m_sub = (m_cur==-inf)?0:m_cur`. `(a-b)*c` is **not** an FMA pattern (the sub feeds
the mul, not mul→add), so at the argmax `(m_cur-m_cur)*sm = 0` exactly →
`exp2f(0)=1.0` → se=-8 → no saturation. **`a*c - b*c` is NOT FMA-safe even when a==b.**

### Why the bit-exact self-replay was BLIND to it (the meta-lesson)

The self-replay reference re-quantizes the **device's own dumped P** with the
**identical** `mx_scale_exp` rule: device computes 0.99999988 → se=-9 → 0.875, host
re-quant of that same 0.99999988 → se=-9 → 0.875 → **they agree → "PASS"**. A test
that shares the quantization rule with the kernel cannot catch a bug in that rule's
*input*. Only the **independent torchao oracle** (`tests/csrc/mxfp8_attn_ext.cu` +
`test_mxfp8_prefill.py::test_attn_matches_torchao`, fp64 softmax where the max is
exactly 1.0) exposed it — at **25% of elements wrong**. Rule: a self-consistency test
and an independent-oracle test catch **disjoint** bug classes; S3 needed both.
(Non-argmax blocks with amax mantissa in `[1.75,2)` also saturate, but torchao and
the kernel both use the floor rule and saturate identically, so they agree —
`cutlass::float_e4m3_t` matches `to_mx` element-for-element. Only the argmax block was
special because FMA nudged amax just *below* the 1.0 power-of-2 boundary.)

---

## V's scale-factor (SFV) layout: K's SF layout only *coincides* with V's at a single 128×128 block

**TL;DR:** In PV, V is the **B operand** shaped `[head_dim, keys]`, block-scaled
**along keys**. Its scale-factor (ue8m0) tensor must be laid out as
`tile_atom_to_shape_SFB(_, head_dim, keys)` = `tile_to_shape(SfAtom, (head_dim, keys))`.
Do **not** reuse K's SF layout (`tile_atom_to_shape_SFA(keys, _, head_dim)` =
`(keys, head_dim)`, scale along head_dim) for V. For one 128×128 block (head_dim ==
kBlockN) the two are byte-identical, so a single-tile test passes — but they are
**transposed**, and the moment V spans more than one n_block the per-block SFV
slice for keys ≥ 128 reads the wrong gmem region → ue8m0 byte `0` = `2^-127` ≈ 0 →
`V · scale ≈ 0` → that block's PV ≈ 0.

### The symptom (and how it hides)

Multi-n_block: **only the first n_block's PV produces output; blocks 1+ give
accB ≈ 0.** Everything upstream is fine — QK, softmax, `row_sum`, LSE, P-requant,
the PV *data* operands `tOrP`/`tOrV`, and **P's** scale `tOrSFP` are all correct
for every block. Only **`tOrSFV` is nonzero for block 0 and ≈ 1e-35 for blocks
1,2,3** (per-block dump). The whole thing is an **input-side gmem-layout bug**, not
a quantization-math or consumer-smem bug — the consumer's per-block `SmemLayoutSFV`
is fine precisely because head_dim == kBlockN makes the per-block shape coincide.

The reason this single bug ate a long debugging session is a **test-data trap**:
adversarial data that makes the running softmax max grow fast (to "exercise" the
online rescale) drives `scores_scale = exp2((m_prev−m_cur)·scale)` to **fp32
underflow**, collapsing the output onto a single n_block — and if that surviving
block is block 0 (the only one whose SFV is correct), the end-to-end check
**passes** and the bug stays invisible. Switching to FLAT data (every block
contributes materially) exposed it immediately.

### The rule

- Build each operand's SF from the function that matches its **role and
  scale-axis**: SFA (`tile_to_shape(SfAtom,(M,K))`) for an A operand scaled along
  its K; SFB (`(N,K)`) for a B operand scaled along its K. V-in-PV is B,
  scaled along keys → SFB over `(head_dim, keys)`. Never borrow another operand's
  SF layout just because the per-tile shapes look equal.
- A single-tile (one n_block, head_dim == kBlockN) test **cannot** catch a
  transposed-SF bug — the layouts coincide there. Always validate with **≥2
  n_blocks** and data where **every block contributes** to O (not max-collapsed
  onto one block). Dump per-block `sum|operand|` / `sum|SF|` to localize which
  operand of which block is zero.
- This is the same generalization needed for head_dim ≠ kBlockN (where V's SF
  shape stops coinciding with K's even for a single block) — do it once, correctly,
  for V.

---

## P→PV requant: QK-C and PV-A are NOT thread-local aligned (and nvfp4 only *looks* like it is)

**TL;DR:** In the fused kernel, the softmax probabilities P come out of QK as the
**C accumulator** and must go back in as the **PV A operand**. For the mxf8
`SM120_16x8x32_TN_VS` atom these two layouts do **not** match per thread, so P
cannot be reused in-register by a simple relabel — it has to be physically
redistributed. The current kernel does this through an smem transpose buffer
(`tests/s2_pv.cu`). Don't assume P "just stays in registers."

### The mismatch

For the m16n8k32 atom, per thread:
- **QK-C** N(key) micro-layout = "2 adjacent keys, stride-8 comb" (FP32 accumulate,
  n8 → 2 cols/thread).
- **PV-A** K(key) micro-layout = "4 adjacent keys" (FP8 A operand, k32 → 4 bytes/thread).

The 2-vs-4 granularity is intrinsic to the hardware instruction's TV layouts.
Probe (`/tmp/locality.cu`, `partition_C` vs `partition_A` over the same
(q,key) identity tensor): **256/256 threads have different {(q,key)} sets.**

### The nvfp4 trap — it looks aligned, it isn't

Reading SageAttention you'll see `SM120_16x32x64_TN_VS_NVFP4` and may conclude
"their n32 atom makes QK-C and PV-A line up, so P is register-resident for free,
and mxf8 is uniquely unlucky." **Not true on either count:**

- `SM120_16x32x64_TN_VS_NVFP4` is **not a hardware MMA**. It's a hand-written
  *macro* atom (`cute_extension.h`) that issues **4× the real
  `mma.sync…m16n8k64` instruction** stacked in N, sharing one A operand and one
  `scale_vec::4X` SFA. Every hardware blockscaled MMA on sm120 is m16n8
  (`cute/arch/mma_sm120.hpp`: only `SM120_16x8x32` / `SM120_16x8x64`). Stacking
  instructions tiles the per-instruction TV layout — it cannot change it.
- The same probe on the nvfp4 macro: **also 256/256 mismatching.** nvfp4 is
  *not* thread-local aligned either. SageAttention keeps P in registers via the
  **V-permutation trick** (fill `tOrP` thread-locally, which permutes the key
  axis by some π, then load V in that same π order so `Σ P[π(k)]·V[π(k)]` is
  still correct) plus its quad `__shfl_xor` on the SF — not free alignment.

### The upside (and the rule)

The probe's useful half: for **both** mxf8 and nvfp4 the QK-C→PV-A redistribution
is **entirely within a 4-lane quad** (same q-rows; the 128 keys of those rows just
re-partition among the 4 quad lanes). So **smem is not required** — an intra-quad
`__shfl` (move only P) or the V-permutation trick (P stays thread-local, V loaded
key-permuted) both work and skip the smem write + named-barrier + ldmatrix.

**Rule:** the smem transpose buffer is the simplest *correct* path — keep it as the
oracle. The intra-quad shuffle / V-permutation is a real perf win (drops a
warpgroup-serializing barrier) but is an optimization to layer on *after* the
pipeline is numerically correct, validated bit-exact against the smem version.

---

## Persistent tile scheduling is neutral-to-harmful for a non-warp-specialized mainloop

**TL;DR:** Don't add a persistent tile scheduler to the option-B (non-WS, cp.async)
mainloop. Launch plain non-persistent (`grid = num_q_blocks × heads × batch`) and
let the hardware block scheduler balance the load. It already balances causal
(triangular) work as well as a software dynamic scheduler — for free.

### Why this is a trap

SageAttention3 / FlashAttention-3 use a **persistent** kernel (grid = #SMs, each
CTA loops over tiles via a `tile_scheduler`). It's tempting to copy that. But their
persistent design exists for reasons that **don't apply to a non-WS kernel**:

1. Their warp-specialized CTAs already occupy **1 block/SM**, so `grid = #SMs` *is*
   full occupancy.
2. The persistent loop **amortizes an expensive prologue** (TMA descriptor setup,
   mbarrier init, producer/consumer warp setup) across many tiles.

A non-WS cp.async kernel has a cheap prologue (nothing to amortize) and occupancy
> 1 block/SM — so persistent scheduling buys nothing and can actively hurt.

### Measured A/B (RTX 5060 Ti, 36 SMs) — `bench/sched_ab.cu`

Same tunable per-tile compute kernel under three schedulers from SageAttention's
`tile_scheduler.h`, swept over 1–32 waves and {uniform, causal-triangular} load:

| scheduler | uniform | causal (typical) | causal (worst) |
|---|---|---|---|
| non-persistent (`grid=num_tiles`, HW scheduler) | 1.00x (baseline) | 1.00x | 1.00x |
| **static**-persistent (`grid=num_sm×occ`, round-robin) | ~1.1x | ~1.1x | **1.46x slower** |
| dynamic-persistent (`grid=num_sm×occ`, atomicAdd) | ~1.0x | ~1.0x | ~1.0x |

- **non-persistent ≈ dynamic-persistent ≈ optimal**, robustly, across all configs.
  The HW block scheduler load-balances triangular causal work for free.
- **static-persistent is strictly dominated** — never faster, up to 1.46x slower.
  SageAttention wires in exactly the *static* one (`launch.h:42`).
- The 1.46x worst case is **period aliasing**: when the causal work-period `G`
  (q-blocks per head) divides the persistent grid stride (`G=8 | grid 216`), every
  tile a given CTA processes lands on the same qblock, so one CTA inherits the
  entire heavy tail and gates the launch. Non-divisor `G` (7/5/13) → only ~1.1x.
  non-/dynamic-persistent are immune (verified by sweeping `G`).

### ncu evidence — why dynamic-persistent doesn't beat non-persistent, and why static is slower

Profiled the three kernels at the heavy causal config (1152 tiles, 32 waves,
`./bench/sched_ab 8 profile`), RTX 5060 Ti:

| kernel | duration | SM throughput (% peak, elapsed) | achieved occupancy (elapsed) | instructions |
|---|---|---|---|---|
| non-persistent | 2.328 ms | **86.8%** | **83.0%** | 747,424,512 |
| static-persistent | 3.252 ms (1.40x) | **62.1%** | **47.6%** | 747,465,408 |
| dynamic-persistent | 2.297 ms | **87.9%** | **84.6%** | 747,524,376 |

- **All three execute the same instruction count** (within 0.02%). The runtime
  difference is *entirely* whether the SMs stay fed, not extra work. The dynamic
  scheduler's atomic-counter overhead is negligible (+0.01% instructions).
- **non-persistent ≈ dynamic-persistent**: both hold ~83–85% achieved occupancy and
  ~87% SM throughput — the SMs are already saturated and balanced. The HW block
  scheduler *is* a dynamic greedy load-balancer; a software atomic scheduler just
  re-does what the hardware already does → no slack to recover → ~1.0x. (dynamic is
  a marginal 1.3% faster here only because 216 long-lived CTAs pay less launch/retire
  turnover than 1152 short ones at high wave counts; at low waves it's *slower*.)
- **static-persistent is slower purely from idle tail**: same work, but achieved
  occupancy collapses 83%→48% and throughput 87%→62%. The duration ratio
  (3.252/2.328 = 1.40x) equals the throughput ratio (86.8/62.1 = 1.40x) almost
  exactly — the lost time is SMs sitting idle after their light tiles finish while
  heavy-tile CTAs grind on; static round-robin can't hand idle SMs the heavy work.

(The FMA pipe sits at ~43% because the microbench's burn loop is a single-thread
dependency chain, latency-bound; irrelevant to the scheduling conclusion — read the
SM-throughput and occupancy rows, which show the SMs are kept busy issuing.)

### Real-kernel confirmation (WS+TMA) — persistent IS worth ~15% on balanced load

The FMA microbench above **explicitly could not** measure persistent's real
upside: amortizing the expensive WS+TMA prologue (TMA descriptor prefetch,
mbarrier init, producer/consumer reg dealloc/alloc) across many tiles. So I
built the **actual SageAttention3 nvfp4 WS+TMA kernel** and A/B'd it for real —
toggling `launch.h:41/42` (`SingleTileScheduler` vs `StaticPersistentTileScheduler`)
and rebuilding `fp4attn_cuda` between runs. `bench/bench_sage3_real.py`, RTX
5060 Ti, B=4 H=32 D=128 fp16. Output is **bit-identical** between the two
schedulers (1-cos vs SDPA = 0.0181 non-causal / 0.254 causal, same to 4 digits)
— confirms the scheduler is numerically transparent, so the timing delta is
purely scheduling.

Full-op TFLOPS (higher = better):

| workload | seqlen | persistent | non-persistent | winner |
|---|---|---|---|---|
| non-causal (balanced)   | 8192  | 152.1 | 143.1 | **persistent +6.3%** |
| non-causal (balanced)   | 16384 | 170.3 | 160.6 | **persistent +6.0%** |
| causal (triangular)     | 8192  | 110.9 | 114.1 | non-persistent +2.9% |
| causal (triangular)     | 16384 | 135.2 | 140.8 | non-persistent +4.0% |

ncu, **attention kernel only** (non-causal, L=4096) — the smoking gun for
prologue amortization:

| metric | persistent | non-persistent |
|---|---|---|
| grid | **36 CTAs** (= #SM) | **4096 CTAs** (m×h×b) |
| waves / SM | 1 | 113.8 |
| achieved occupancy | 20.7% | 20.0% |
| SM throughput | **57.2%** | 48.7% |
| kernel duration | **5.09 ms** | 5.97 ms (**+17%**) |

- **Occupancy is identical (~20% = 1 block/SM)** — both run one heavy WS CTA per
  SM. So unlike the microbench's *static* collapse (occupancy 83→48%), here the
  difference is **not** occupancy. Same work, same residency.
- The win is **SM throughput: 57.2% vs 48.7%.** The ratio 57.2/48.7 = 1.17 equals
  the duration ratio 5.97/5.09 = 1.17 exactly. The persistent kernel keeps the
  SMs ~17% busier because it pays the prologue **36 times** (one per resident CTA,
  then loops ~114 tiles each) instead of **4096 times** (a fresh CTA launch +
  prologue per tile, streamed 113.8 waves deep). That is prologue + launch/retire
  amortization, made visible.
- **Causal flips the sign by a few %:** static round-robin hands one CTA the heavy
  triangular tail (the microbench's idle-tail effect), costing ≤4% vs the HW
  scheduler — but nowhere near the synthetic 1.46x, because real per-tile work is
  far heavier than a burn loop. A **dynamic** persistent scheduler would keep the
  prologue win *and* balance the tail; SageAttention ships *static*, leaving the
  causal few-% on the table.

**This does not contradict the non-WS finding above — it completes it.** The
microbench verdict ("persistent neutral-to-harmful") is correct *and scoped to a
cheap-prologue non-WS kernel*. For a heavy-prologue WS+TMA kernel the prologue
amortization is real and dominates on balanced load (~15% at the kernel level,
~6% end-to-end once the identical quant/group-mean pre-kernels dilute it).

### Why static-persistent *loses* on causal — it's two orthogonal effects

Counter-intuitive: causal (triangular) load is exactly where you'd expect a
scheduler to help, yet static-persistent loses ~2.5–4% there. Profiled the
controlled 2×2 (`bench/profile_sched_2x2.sh`, swaps prebuilt .so variants, L=4096)
plus a causal L-sweep. **The clean, clock-independent probe is `sm__cycles_active`
— the fraction of elapsed time an SM has *any* warp resident, i.e. "not idle".**
(Ignore ncu's *absolute* kernel ms here: it locks clocks to base, and for a ~3%
effect its single-shot durations are unreliable — at L=8192 they even flipped the
sign vs the averaged benchmark. Trust the averaged benchmark for timing and ncu
only for the `pct_of_peak_sustained_elapsed` *fractions*, which are ratios within
one locked-clock run and so clock-independent.)

| variant | mask | sm__cycles_active (not-idle) | sm__throughput |
|---|---|---|---|
| persistent     | non-causal | **99.5%** | 57.3% |
| non-persistent | non-causal | 98.2% | 48.2% |
| persistent     | causal | **89.0%** | 45.9% |
| non-persistent | causal | **94.4%** | 47.9% |

Causal, sweeping L (cycles_active / throughput, %):

| L | persistent | non-persistent |
|---|---|---|
| 4096  | 89.0 / 45.9 | 94.4 / 47.9 |
| 8192  | 92.4 / 49.0 | 94.2 / 51.0 |
| 16384 | 97.5 / 52.3 | 99.1 / 55.9 |

Read it as **two orthogonal effects** that the non-causal/causal split cleanly
separates:

1. **Prologue / pipeline-fill amortization** — always favors persistent. Visible
   non-causal as throughput 57.3 vs 48.2 (+9 pts) at ~equal, ~99% occupancy-of-time.
   36 prologues vs 4096.
2. **Load balancing** — favors whoever can rebalance the heavy triangular tail.
   Static round-robin **can't**: it fixes each CTA's tile set up front, so when a
   CTA exhausts its (lighter) tiles it goes idle while the CTA holding the heavy
   tail grinds on. That's the **`sm__cycles_active` drop to 89–97% on causal**,
   always below non-persistent's 94–99%. The hardware block scheduler (what
   non-persistent rides) is a greedy dynamic balancer — it hands an idle SM the
   next pending tile, heavy or light, and stays 94–99% fed.

Non-causal: only effect (1) is in play → persistent wins (~15% kernel). Causal:
effect (2)'s idle tail costs static-persistent more than effect (1)'s prologue
edge saves → persistent nets ~2.5–4% slower (and the throughput gap widens with
L: 1.9 → 2.0 → 3.6 pts). The user's intuition was right — causal *is* a
load-balancing problem — but **static-persistent is the wrong tool for it; you
need a dynamic scheduler to claim that win.**

Caveat that matters for us: SageAttention's `DynamicPersistentTileScheduler` is a
**non-functional stub** — its `get_next_work` is identical to the static one
(`tile_idx + gridDim.x`) and the `tile_count_semaphore` it stores is never read.
So the "best of both" (prologue amortization + greedy balancing) isn't actually
available in this codebase without implementing the atomic tile fetch ourselves.

### Rules

- **Option B (non-WS) mainloop: start non-persistent.** Zero-gain to do otherwise,
  plus the static-persistent aliasing tail risk.
- If we later go **option A (WS + TMA)**, persistent is justified — **measured**
  ~15% kernel-level / ~6% end-to-end on balanced load from prologue amortization
  at occupancy=1 (see real-kernel section above). But a *static* persistent
  scheduler gives the prologue win back on causal via the idle tail (−2.5–4%).
- **FA3 confirms our 2×2 exactly** (`flash-attention/hopper/flash_fwd_launch_
  template.h:61-76`): it picks **StaticPersistent only for dense non-causal**, the
  **real DynamicPersistent for dense causal/local**, **VarlenDynamic for varlen**,
  and SingleTile for split/low-work. So StaticPersistent is not vestigial — its
  niche is precisely the no-imbalance case we measured it winning. Mirror this map.
- The **real dynamic scheduler is more than an atomic counter**: FA3's does **LPT**
  (issue heaviest tiles first, `block = num_m_blocks-1-block`) + **L2 swizzling**
  (head×batch sections sized to keep K/V in L2). SageAttention's `Dynamic…` is a
  stub (no atomic, no LPT) — don't copy it; copy FA3's. For FlashInfer's **ragged
  BatchPrefill the relevant one is the varlen dynamic scheduler** (batch prefix-sums
  + LPT), the hardest of the four — plan it early. If causal dominates and we want
  to defer that work, plain non-persistent is within a few % and far simpler.

### Two methodology gotchas (caught while running the A/B)

1. **A persistent kernel must launch `grid = num_sm × cudaOccupancyMaxActiveBlocks
   PerMultiprocessor`, not `num_sm`.** A light kernel has occupancy > 1 (here 6
   blocks/SM); using `grid = num_sm` starves it to 1/6 load, and you end up
   measuring occupancy loss (a fake ~2x slowdown), not scheduling. FA3/SageAttention
   get away with `grid = num_sm` only because their heavy WS CTAs are already 1/SM.

2. **This microbench cannot measure persistent's real benefit** for FA3/SageAttention
   (amortizing the WS+TMA prologue). It therefore argues against persistent for a
   *cheap-prologue non-WS* kernel **only** — not against option A's persistent.

### Side fix

SageAttention's `sageattn3/blackwell/launch.h:89` hardcoded `num_sm = 170` (the
RTX 5090's SM count). Patched to query
`getCurrentDeviceProperties()->multiProcessorCount` so the persistent grid is
correct on any GPU (36 on the 5060 Ti) — otherwise persistent over-subscribes
170 CTAs onto 36 SMs and the A/B is polluted before it starts.
