# Architecture audit — what warp-spec + persistent + TMA actually buy on SM120, and the `wait` floor

**Scope.** A from-the-data audit of the three big architectural choices in our MXFP8 prefill kernel
(warp specialization, persistent scheduling, TMA loads) and where the *residual* bottleneck lives, on
consumer Blackwell (RTX 5060 Ti, sm120a, 36 SM). Every claim below is grounded in an `ncu --set full`
trace, **profiled on the production scenario: causal + variable-length + GQA** — which is exactly where
warp-spec and the persistent scheduler are supposed to earn their keep.

**Reproduce.**
- Ours (real-64, `kBlockN=64`): `prof/trace_real64.sh` → `prof/real64_8x2_causal.ncu-rep`
  (launch 120 = the (8,2) GQA causal config of the 16-request varlen batch in `tests/bench_ragged.cu`).
- flashinfer fa2 (bf16, the realistic ceiling — sm120 has no fp8 prefill path): `prof/fa2_8x2_prof.py`
  → `prof/fa2_8x2_causal.ncu-rep` (same 16-request varlen batch, Hq=8/Hkv=2, causal).
- SageAttention3 (fp4, the *structural* reference at the identical 1-CTA/SM persistent-WS design):
  `prof/sage_prof.py` (`compute_attn_ws`, dense head_dim=128 causal).

The workload is `lens = {512,1024,768,1536,2048,640,1280,896,384,1792,512,2560,1024,768,2048,1100}`
(16 requests, lengths 384–2560 ⇒ 6.7× spread), driven by `BatchPrefillPersistentTileScheduler` with a
host LPT (longest-processing-time-first) work-list.

---

## Scorecard — both halves of the design deliver; the residual is one ISA wall

| component | what it should buy | evidence on the causal-varlen-GQA trace | verdict |
|---|---|---|---|
| **WS + TMA** | hide K/V memory latency | `long_scoreboard` **0.23**; K/V ride the TMA path | ✅ |
| **persistent + LPT** | balance the varlen-causal load across SMs | 36 CTA / **1 wave**; SM active-cycle spread **2.7%**; **98.4%** active; no tail | ✅ |
| per-CTA compute | feed the tensor pipe | tensor pipe **42.7%** (half-idle); `wait` floor | ⛔ ISA wall (below) |

### 1. WS + TMA — memory latency is genuinely hidden

Warp specialization splits the CTA into a **producer** warpgroup (128 thr, `setmaxnreg.dec<24>`, drives
TMA) and a **consumer** warpgroup (256 thr, `setmaxnreg.inc<232>`, does MMA + online softmax). The
producer prefetches K/V tiles through the async pipeline while the consumer computes. It works:

| signal | value | reading |
|---|---|---|
| K/V via TMA vs via LSU | TMA-ld **8.88M** sectors vs global-LSU-ld **0.06M** | K/V load almost entirely on the dedicated TMA path (`xbar2l1tex…op_tma_ld`), not LSU/`LDG` |
| `long_scoreboard` /issue | **0.23** | K/V load latency is *not* a stall — the producer/TMA overlap hides it |
| DRAM SOL | **33%** | tiny fp8 + TMA ⇒ K/V stay L2-resident, never HBM-bound |

The key logical point: **the reason this kernel is not memory-latency-bound is precisely that TMA
succeeded.** Had TMA failed, K/V would appear as `LDG`-wait on `long_scoreboard`; instead long_sb is
0.23. The downstream `wait` bottleneck only surfaces *because* the memory side was cleaned up first.

The register reallocation half of WS also delivered: the consumer's large register budget makes the
big-tile operands register-resident, so the fast fp8 path runs without operand-fetch spill (real-64
ptxas spill = **20 B**).

### 2. Persistent + LPT — the varlen-causal load is balanced, no straggler tail

Variable lengths (6.7× spread) × causal (triangular per-tile work) is naturally imbalanced. A
one-CTA-per-tile launch under the HW scheduler would leave a straggler tail (a few SMs chewing the big
requests while the rest idle). The persistent design avoids it:

| signal | value | reading |
|---|---|---|
| `launch__grid_size` / `waves_per_multiprocessor` | **36 / 1** | one resident CTA per SM, looping over its assigned work-items — not multi-wave one-CTA-per-tile |
| `sm__cycles_active` min/avg/max | 1.990M / 2.019M / **2.044M** | cross-SM *useful-work* spread is only **2.7%** (max/min) |
| `sm__cycles_elapsed` spread | **0.19%** | all 36 SMs start and finish together |
| active / elapsed | **98.4%** | SMs are busy essentially the whole kernel, almost zero idle |

A 2.7% active-cycle spread on a 6.7×-length-spread causal batch is near-perfect balance — the host LPT
"heaviest-first" work-list did its job. (A single trace proves the balance is *good*; attributing it
specifically to LPT-over-HW is the separate S4 A/B, which measured LPT winning 10–20% on varlen loads.)

### 3. Per-CTA compute — the `wait` floor (the one thing software can't touch)

With memory hidden and the load balanced, the kernel's remaining bottleneck is intra-CTA compute-
dependency latency:

| metric | real-64 | note |
|---|---|---|
| tensor subpipe active (per-active) | **42.7%** | the fast fp8 QMMA leaves the tensor pipe **half-idle** |
| dominant warp stall | **`wait` 1.26** | fixed-latency MMA/FFMA dependency, not memory |
| `long_scoreboard` / `short_scoreboard` | 0.23 / 0.26 | both tiny — not memory-stalled |
| `issue_active` | **39.7%** | |
| `warps_eligible` /cycle | **0.59** | **< 1 eligible warp per scheduler-cycle** ⇒ ~60% of cycles issue nothing |

This is analyzed in full below; it is a platform ISA limit, not a tuning miss.

---

## The fa2 contrast — the opposite bottleneck regime on the same silicon

fa2 (bf16, the only sm120 prefill path flashinfer offers) is the mirror image of ours:

| metric (8,2 GQA causal varlen) | fa2 bf16 | real-64 mxfp8 |
|---|---|---|
| **tensor subpipe active (per-active)** | **96.4%** | 42.7% |
| SM (Compute) SOL | 87.4% | 45.8% |
| L2 SOL | 16.2% | 76.3% |
| DRAM SOL | 18.1% | 33.3% |
| LSU pipe active | 13.7% | 46.5% |
| `issue_active` | 13.7% | 39.7% |
| dominant warp stall | **`math_pipe_throttle` 9.10** | `wait` 1.26 |
| regs/thread | 244 | 168 |
| ncu single-launch | **1.73 ms** | **0.77 ms** (≈2.25×) |

fa2 is **tensor-pipe-SATURATED** (96.4% + math-pipe-throttle): every CUDA-core pipe < 8%, L2/DRAM ~17%
— textbook compute-bound FlashAttention, and it has *no* sm120 software headroom (no fp8 tensor-core
path exists in flashinfer). We are the opposite: the tensor pipe is half-idle because fp8 QMMA is so
fast, and the limiter is on-chip data movement + the `wait` floor. **Absolute tensor-busy time: fa2
~1.51 ms vs ours ~0.32 ms ≈ 4.7×** — the fp8-vs-bf16 MMA advantage made directly visible (consistent
with the standing "MMA ~4×" claim; the kernel dilutes it to ~2.25× single-launch).

---

## Roofline framing — is FlashAttention memory-bound on this device?

**On the DRAM/HBM roofline: no.** FA's arithmetic intensity ∝ S (it never materializes the S×S matrix
to HBM: HBM traffic = O(S·D), FLOPs = O(S²·D)), so at our sizes FA is compute-bound on DRAM — confirmed
both ways (fa2 DRAM 18%, ours 33%; neither HBM-bound; fa2 is the textbook 96%-tensor point).

The "memory" we see is the **on-chip** roofline (L2 / LSU / smem), and it has two layers worth keeping
straight:

- **`LSU` pipe ≠ `L2` traffic.** LSU 46.5% (an instruction-issue pipe) contains `LDS` (smem operand
  re-read) + `LDL`/`STL` (spill) + `STG` (O-write). **smem (`LDS`) is served by L1TEX banks and
  generates *zero* L2 traffic.**
- **L2 76% (28.35M sectors) is *not* binding on real-64.** Decomposition: spill-STORE write-through
  ~14.3M (**50%**, fire-and-forget, doesn't stall warps), K/V TMA loads 8.88M (**31%**), O-write
  4.89M (**17%**), spill-read-miss 0.23M (0.8%), smem **0%**. The spill *reloads* (the latency-critical
  path) hit L1 at **99.3%** → only 0.23M reach L2 (that is why `long_scoreboard` is 0.23). So L2 76% is
  high write-*throughput*, not a binding latency stall — the binding stall is `wait`.

The deeper general fact: **fp8 QMMA raises the compute ceiling ~4× → the roofline ridge point shifts
right → the same algorithm slides from compute-bound (bf16) toward memory-leaning (fp8).** But here it
does not land on a memory *roofline*; it lands on `wait` (MMA-dependency latency). bf16 FA = solidly
tensor-bound; fp8 FA on consumer Blackwell = tensor pipe half-idle, pinned by MMA-dependency latency,
with DRAM/L2 loud but non-binding.

---

## Root cause of the `wait` floor — SM120 has no asynchronous MMA

This is the one wall software cannot move, and it is a hardware/ISA fact about consumer Blackwell.

**SM120's block-scaled MMA is synchronous, warp-level `mma.sync`** — the cute atom
`SM120_16x8x32_TN_VS` (16×8×32, 32-thread operand distribution, results written to registers with a
fixed latency; reading the result stalls on `wait`). It is the Ampere/Ada programming model.

**It has neither async MMA primitive:**
- no `wgmma.mma_async` (that is sm90a / Hopper only), and
- no `tcgen05.mma` (that is sm100 / datacenter Blackwell, and needs TMEM — sm120 has no TMEM; see
  `docs/gotcha.md`, "the SF 128-granularity is the sm100 TMEM staging format").

Why this is *the* cause:

- An **async** MMA (`wgmma`/`tcgen05`) lets the issuing warp fire the MMA and immediately continue —
  the tensor core computes in the background, the warp only synchronizes at an explicit `wait_group`.
  The MMA latency is hidden by the *instruction semantics*, so a low-occupancy warp-spec kernel (few
  warps) can still keep the tensor pipe fed. This is the assumption the WS/persistent design (Hopper
  FA3 lineage) is built on.
- A **synchronous** `mma.sync` has no background overlap: the only ways to hide its latency are **TLP**
  (another warp issues while this one waits) or **ILP** (issue an independent `mma.sync` back-to-back).
  Our design starves both:
  - **TLP is starved.** WS + 1 CTA/SM ⇒ the producer's 4 warps are asleep on the mbarrier and the
    consumer's 8 warps spread over 4 SMSPs = **~2 consumer warps per scheduler**, running the same code
    and hitting their MMAs in lockstep. `warps_eligible = 0.59` (< 1 per scheduler-cycle) is the direct
    proof: most cycles there is no other eligible warp to cover the stall, so the issue slot empties —
    that *is* the `wait`. You cannot add warps (register-bound at 1 CTA/SM).
  - **ILP is register-walled.** A second accumulator chain would keep a warp eligible through its own
    MMA shadow, but it costs +32–64 registers over the 168 ceiling → re-spill. The only independent MMA
    work is cross-`n_block` (next key block's QK ⟂ this block's PV), and the `kStages=2` mainloop
    already pipelines that cheap part.

**Sage proves it is the floor, not our miss.** At the *identical* structure (1 CTA/SM, 384 thr,
persistent WS, 168 regs), SageAttention3 (fp4) has **`wait` 1.45 > our 1.26**, `warps_eligible` 0.58 ≈
our 0.59, `issue_active` 42.0% ≈ our 39.7%. The best-in-class fp4 reference hits the same wall — and is
*worse* on `wait` than we are. The residual ~6% issue gap to Sage is **format, not implementation**:
nvfp4's k64 = 2 operand k-tiles vs mxfp8's k32 = 4 ⇒ Sage issues the same MMA work in half the
instructions (denser tensor pipe, 51.7% vs 42.7%). That is the immutable mxfp8-vs-nvfp4 choice, not a
lever.

---

## Bottom line

The WS + persistent + TMA machinery is the Hopper-FA3 design lineage, which implicitly assumes async
MMA. Porting it to sm120, **every part that does *not* depend on async MMA transferred and delivered**,
demonstrably so on the production causal-varlen-GQA workload:

- **TMA + producer** → K/V memory latency hidden (`long_scoreboard` 0.23, K/V on the TMA path).
- **WS register realloc** → register-resident big-tile fp8 operands, spill 20 B.
- **persistent + LPT** → varlen-causal load balanced to 2.7% cross-SM spread, 98.4% active, no tail.

The one part that did *not* transfer is the design's implicit reliance on async MMA to hide compute
latency at low occupancy — and sm120 simply lacks the instruction. The result is the `wait` floor:
**every software-solvable bottleneck has been solved, and the kernel is pinned against an ISA-level
platform ceiling that the SOTA reference (Sage) hits even harder.** We still beat fa2 ~2× because fp8
math is cheap enough that even with `wait` exposed we finish well ahead of fa2's saturated bf16 pipe.

*See also:* `docs/gotcha.md` (SF-128 = TMEM artifact / no TMEM on sm120; the spill diagnosis arc).

---

## Addendum (2026-07-21, S9) — two claims in this audit were wrong; the "ISA wall" is ~23% softer than concluded

Re-examination of the same `real64_8x2_causal.ncu-rep` + SASS disassembly produced two
corrections, and fixing them made the kernel **1.21–1.25× faster** (bench_ragged min-of-N,
all configs; ncu single-launch 0.770→0.625 ms; vs fa2 now ~2.8×):

1. **"Spill write-through is fire-and-forget, non-binding" — wrong on the issue axis.** The
   1.5GB local traffic was NOT ptxas spill (20B); it was the real-64 `nb&1` SF half-slice
   demoting the SF register fragments to a 160B stack frame (64× LDL.U8 + 32× STL.U8 per
   n_block). The reloads do hit L1 (latency fine), but ~96 byte-granular local instructions
   per thread per block steal issue slots at 2 warps/scheduler — LSU was the #1 pipe (45.7%
   > tensor 42.1%). Fix (even/odd unroll, `cute::Int<h>`): local sectors −98%, LSU 30.9%.
2. **"WS register realloc delivered" — CORRECT after all, but the wall is subtler.** On
   sm120, `setmaxnreg` compiles to `USETMAXREG` (not `SETMAXNREG` — an early grep for the
   latter false-negatived and briefly produced the opposite, wrong claim). ptxas allocates
   per-region: producer ≤24 (R20 observed), consumer ≤232 (R229 observed);
   `res-usage REG:168` is only the launch-static count. So the consumer really runs at 232 —
   and full QK-rotation pipelining STILL spilled (accO+accB+accS+operands ≈ 220–240 > 232;
   STACK 200–296B, net regression). The productive use of the budget was the accB fold
   (direct `accO += P*V`, −64 regs) + cheap hoists, not rotation.

What S9 actually shipped (all validated by the torchao oracle, O ≤ 9e-5, 12/12):
- **S9a** even/odd static SF halves (above): ~1.14–1.17×.
- **S9e** accB fold + early V release (smem drained at ldmatrix, not after gemm): +1–2%,
  STACK 48B. Not bit-exact vs the accB+telescope add order (~1e-7, inside oracle tol).
- **S9f** causal full-block mask skip (one bound check vs 32 FSETP/FSEL per full block) +
  O-write float2 vectorization (`AutoVectorizingCopyWithAssumedAlignment<64>`, was scalar
  STG.E at 16.1/32B per sector): +4–5%.

Post-S9 profile (ncu, same launch): tensor 42.1→**50.9%**, L2 SOL 76→**25%**, wait
1.26→1.30 (per-issue ratio flat because instruction count fell 8%; absolute wait-cycles
dropped), issue 39.7→~46%. The residual `wait` ~1.3 at tensor ~51% with 0.65 eligible
warps IS the synchronous-mma.sync floor this audit describes — but ~23% of kernel time was
software-removable on top of it. Also fixed: the in-repo torchao oracle replayed the online
algo at 128-key blocks (stale since real-64; false-failed ~0.3%) — now replays at 64-key
(`tests/test_mxfp8_prefill.py`), agreeing with the kernel at ~5e-6 as the gotcha predicted.
