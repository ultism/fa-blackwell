# #6 Mainloop design — MXFP8 block-scaled prefill for FlashInfer sm120a

Locked design for the attention mainloop. Target is a **FlashInfer-grade** sm120a
MXFP8 prefill kernel (not a throwaway), so the architecture is laid out so the
optimized form is reached by *layering*, not rewriting. Decisions here are fixed
before coding; open risks are called out at the end.

Validated substrate we build on (see memory `qk-smoke-test-cute-validated`):
QK block-scaled MMA through `cute::gemm` on `SM120_16x8x32_TN_VS<e4m3,e4m3,f32,
ue8m0,32>` emits `QMMA.SF.16832` and is numerically correct incl. the K-loop and
per-(row/col,block) ue8m0 scales; SF source layout for a tile is
`(MN,(32,NBLK)):(NBLK,(0,1))`; SF partition helpers live in
`include/flashinfer/attention/blackwell/quantization/sm120_mxfp8_mma.cuh`.

---

## 1. Architecture — scheduler decoupled from the mainloop

Mirror SageAttention/FA3: the tile scheduler is a **template parameter**, the
mainloop touches it only through a 4-method interface. This lets one mainloop run
under any scheduler and lets us **A/B + validate a real dynamic scheduler on our
own kernel** (SageAttention's `DynamicPersistentTileScheduler` is a stub — see
`docs/gotcha.md` — so this is also where we contribute the real one).

```
template <typename Ktraits, bool Causal, typename TileScheduler>
__global__ void mxfp8_prefill(Params, MainloopParams, EpilogueParams, SchedParams);
```

**Align the interface to FlashInfer's, not SageAttention's** (this is the key
correction): FlashInfer's hopper prefill (`include/flashinfer/attention/hopper/
prefill_sm90.cuh:44`) already templates on `typename TileScheduler` and uses the
*fuller* contract below. Adopt it verbatim so FlashInfer's own schedulers drop
straight into our mainloop:
- `get_grid_dim(Arguments, num_sm) -> dim3`
- `using SharedStorage = …` (scheduler reserves its own smem; `int` for simple,
  `int4` for the broadcast ones — the mainloop must union in `TileScheduler::SharedStorage`)
- `get_initial_work(Params) -> WorkTileInfo`
- `WorkTileInfo::is_valid(Params)` / `get_block_coord(Params)` — **note FlashInfer's
  coord is an 8-tuple** `(q_tile_idx, qo_head, kv_head, qo_indptr, kv_indptr, qo_len,
  kv_len, batch_idx)` carrying the ragged per-request offsets, not FA3's 4-tuple.
- `init_consumer()`, `prefetch_next_work(Params, WorkTileInfo&)`,
  `broadcast_next_work(WorkTileInfo&)`, `get_next_work<bool is_producer>(Params,
  WorkTileInfo) -> WorkTileInfo` (producer warp fetches + broadcasts; consumers read).

We ship these implementations (and pick per-workload, exactly as FA3 does):
1. **SingleTileScheduler** — non-persistent, `grid = (num_m_blocks, H, B)`. Default
   for the non-WS correctness core, and for low-work / split cases.
2. **StaticPersistentTileScheduler** — `grid = num_sm × occ`, `tile_idx += gridDim.x`.
3. **DynamicPersistentTileScheduler (REAL one)** — `grid = num_sm × occ`,
   `get_next_work` does `atomicAdd(params.tile_count_semaphore, 1)` from one elected
   lane and broadcasts the index (smem + named barrier for the WS case, à la FA3).
   **Must also do FA3's two tricks, not just the atomic fetch:** **LPT**
   (longest-processing-time-first — issue the heaviest tiles, i.e. largest `m_block`
   with the most keys, *first*, via `block = num_m_blocks-1-block`, so the tail is
   minimal) and **L2 swizzling** (schedule in head×batch "sections" sized to keep
   K/V resident in L2). Atomic fetch alone is *not* the win.
4. **Varlen / ragged** — **we do NOT write this for FlashInfer.** FlashInfer
   already provides `BatchPrefillTileScheduler` / `BatchPrefillPersistentTileScheduler`
   (`hopper/tile_scheduler.cuh`), whose device side is a *trivial work-queue walker*
   (`work_indptr[blockIdx.x .. +1]` → look up precomputed `qo_tile_indices/qo_indptr/
   kv_indptr/…`). The balancing/LPT lives **host-side in `plan()`/`scheduler.cuh`**,
   not in the kernel — the opposite of FA3, which computes ragged prefix-sums +
   LPT + L2-swizzle *on device* (`VarlenDynamicPersistentTileScheduler`). Since our
   mainloop adopts FlashInfer's interface, these plug in **for free**; the hard
   varlen lift is already done upstream. (#2/#3 above are for the standalone /
   SinglePrefill path and for our own scheduler A/B research.)

**FA3's selection map = our measured 2×2 (external confirmation).** `flash_fwd_
launch_template.h:61-76` picks:

| regime | FA3 scheduler | matches our finding |
|---|---|---|
| dense, **non-causal** | **StaticPersistent** | non-causal: static wins (prologue, no imbalance) ✓ |
| dense, **causal/local** | **DynamicPersistent** (LPT + L2 swizzle) | causal: static loses the idle tail ✓ |
| **varlen / ragged** | **VarlenDynamicPersistent** | — |
| split / low-work | SingleTile (non-persistent) | "not enough work for persistent" |
(`UsePersistentScheduler = Arch≥90 ? !(Split && !Varlen) : ((Is_causal && !Varlen)
|| (Varlen && Split))`.) So **StaticPersistent is NOT vestigial — its niche is
exactly dense-non-causal**, the one cell where it has no imbalance to lose. Causal
never uses static.

**Implication for FlashInfer (revised — varlen is NOT our lift):** BatchPrefill is
ragged, but FlashInfer's idiom puts the ragged work-partitioning in the host
`plan()` step (`scheduler.cuh`), which emits a balanced work queue; the device
scheduler just walks it. So if our mainloop speaks FlashInfer's scheduler interface
(§1) and consumes its 8-tuple block coord, the batch/persistent schedulers and the
whole varlen path come **for free** from upstream. What we owe is the *mainloop*
(block-scaled QK/softmax/requant/PV) behind that interface — not a varlen scheduler.

## 2. Numerical core (the per-tile compute pipeline)

One CTA owns one `(m_block, head, batch)` query tile of `kBlockM` rows; it streams
over `kBlockN`-wide key blocks (`n_block = 0 .. n_block_max(m_block)`), causal
trip count from the scheduler's block coord. Per key block:

1. **QK**: `S = Q_tile @ K_block^T` via the block-scaled atom (A=Q[M,hd],
   B=K[N,hd], head_dim = K-contraction, 1 atom = 32 = one ue8m0 block; K-loop over
   head_dim/32 atoms). Q SFA + K SFB are ue8m0 from the quant kernel.
2. **online softmax** (§5): row max/sum update, rescale running `O`.
3. **P requant** (§6, highest risk): `P = exp(S - m)` (fp32) → e4m3 data +
   ue8m0 block scale along the **seq_k** axis (the PV contraction axis).
4. **PV**: `O += P_block @ V_block` via a second block-scaled atom (A=P[M,seq_k],
   B=V[seq_k,hd]; seq_k = K-contraction; P's SFA + V's SFB feed SFB/SFA slots).
5. After the n-loop: finalize `O = O / l`, write `O` and `LSE = m + log(l)`.

## 3. Tile / warp layout

- `kBlockM = 128`, `kBlockN = 128`, `head_dim ∈ {64,128}` (256 later).
- 2 consumer warpgroups split `kBlockM` spatially (64 rows each) → each warp owns
  disjoint output rows ⇒ **softmax reduction is warp-shuffle only, no cross-warp**
  (SageAttention's trick; keeps online softmax cheap).
- QK accumulator `S` and PV accumulator `O` partitioned by the two TiledMMAs;
  `O` is held in registers across the whole n-loop (flash-style).

## 4. smem layout

- Q staged once per tile; K/V multistage (`kStages = 3`) ring buffer.
- **Data tiles**: non-WS core uses `cp.async`; A-form swaps to TMA. Same smem
  shapes so the swap is local.
- **SF tiles**: reuse the validated `(MN,(32,NBLK)):(NBLK,(0,1))` hand-rolled
  layout for Q/K/V/P scales. CAVEAT (memory `qk-smoke-test-cute-validated`): this
  is *not* cutlass `Sm1xxBlockScaledConfig`. It's correct for `cp.async`/register
  staging. **If A-form TMA-loads the K/V scales from gmem, those SFs must match the
  TMA descriptor's layout** → revisit only the gmem-SF path then; P's SF stays
  hand-rolled (computed in-kernel, never TMA'd).

## 5. Online softmax

Port SageAttention's per-row scheme: each warp keeps `m_i` (running max) and `l_i`
(running sum) for its rows; per key block compute row max of `S`, `correction =
exp(m_old - m_new)`, rescale `O *= correction` and `l`, `P = exp(S - m_new)`,
`l += rowsum(P)`. Reductions stay within a warp (row ownership from §3). Fold the
`softmax_scale * log2e` into the exponent (use `exp2`) — and, for requant, fold the
P-quant scale into the same exponent (SageAttention's `softmax_scale_log2`
coupling) so requant is ~free.

## 6. P requant — highest-risk item

`P` (fp32, in registers after softmax) must become e4m3 + ue8m0 **blocked along
seq_k** (the PV contraction axis), with the SF laid out to feed PV's atom exactly
as the A-operand scale.
- Block size 32 along seq_k; `AbsMaxP` per 32-block computed during the softmax
  pass (cheap, P already in regs).
- `scale = AbsMaxP / e4m3_max`, rounded to a ue8m0 (power-of-two) exponent; data =
  `round_to_e4m3(P / scale)`.
- The 8-scales-per-128 (vs FA-nvfp4's layout) SF tile is written so PV's
  `partition_fragment_SFA` reads cosize-1 fragments — validated pattern from the QK
  SF-smem test, now on the PV side.
- De-risked at single-atom scale already (QK SF-smem test); the new work is the
  multi-atom seq_k tiling + doing it from live softmax registers.

## 7. FlashInfer interface alignment

Match the SM90 FP8 prefill precedent (`hopper/quantization/prefill_sm90.cuh`):
- **AttentionVariant CRTP**: `LogitsTransform / LogitsMask / update_m_d /
  OutputTransform` — our requant hooks into `update_m_d`/output path.
- **LSE** output `[total_qo_tokens, num_qo_heads]` f32 — non-negotiable shape.
- **BatchPrefill**: ragged `qo_indptr / kv_indptr` + `paged_kv_t`. Work-partition is
  owned by FlashInfer's host `plan()` (`scheduler.cuh`) → work queue → the device
  scheduler's 8-tuple block coord carries the per-request offsets (see §1). We
  consume that coord; we don't compute ragged scheduling. (SinglePrefill = degenerate.)
- JIT: `gen_*_prefill_module` + jinja + `modules.py`.
Scales `scale_q/k/v` flow in as kernel params; ours are *block* ue8m0 (not the
SM90 per-tensor scalar) — the variant carries the SF pointers.

## 8. Build sequence (layering, not rewriting)

1. **Non-WS numerical core** + `SingleTileScheduler`: QK→softmax→requant→PV in one
   CTA, `cp.async` loads. Validate numerics vs the torchao oracle (head_dim 64/128,
   causal + non-causal). This is the correctness milestone; scheduler abstraction
   is already live here.
2. **Scheduler validation**: drop in Static + the real Dynamic; A/B on our own
   kernel. (On the cheap-prologue non-WS core we expect ≈ non-persistent — that's
   the control; the payoff shows up after step 3.)
3. **WS + TMA wrap**: producer/consumer warpgroups, TMA loads, mbarrier pipeline.
   Same numerics, same scheduler interface. Re-run the scheduler A/B here — this is
   where Dynamic's "best of both" should finally beat both Static and non-persistent.
4. **FlashInfer integration**: AttentionVariant + batch/paged + LSE + JIT; perf
   pass.

Layouts (SF, smem, accumulator partitions) are A-form from step 1 so steps 3–4 add
code, not rewrites.

## 9. Open risks

- **P-requant from live registers** at multi-atom seq_k scale (§6) — the one piece
  not yet validated end-to-end.
- **gmem SF layout vs TMA** (§4) — only bites if/when A-form TMA-loads K/V scales.
- **Numerics of in-exponent requant coupling** (§5) — verify the folded scale
  doesn't lose precision vs a separate quant step, against the oracle.
- **head_dim 256** and GQA (`H_kv < H`) — deferred but keep the indexing general.
