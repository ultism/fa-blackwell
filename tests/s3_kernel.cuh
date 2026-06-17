#pragma once
// S3 MXFP8 prefill kernel + all CuTe types -- shared by the bit-exact self-test
// (tests/s3_e2e.cu) and the torchao independent oracle (tests/csrc/mxfp8_prefill_ext.cu).
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "flashinfer/attention/blackwell/quantization/sm120_mxfp8_mma.cuh"
#include "flashinfer/attention/blackwell/prefill/flashinfer_tile_scheduler.cuh"

using namespace cute;
namespace mxfp8 = flashinfer::sm120_mxfp8;

#define CK(call)                                                              \
  do { cudaError_t e_ = (call);                                               \
    if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n",               \
        cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

using Element   = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
#ifndef S3_HEAD_DIM
#define S3_HEAD_DIM 128
#endif
constexpr int kHeadDim = S3_HEAD_DIM, kBlockM = 128, kBlockN = 128, SFVecSize = 32, kStages = 2;
// The block-scaled SF smem atom + TMA box are inherently 128 along the contraction
// (cutlass Blk_SF granularity); the gmem SF layout pads K to 128 for head_dim < 128
// too. So all SF tiles use kSFPadHD=128 while the DATA path uses the real kHeadDim
// (the QK MMA contracts only kHeadDim, reading the first kHeadDim/32 of the 4 SF blocks).
constexpr int kSFPadHD = (kHeadDim < 128 ? 128 : kHeadDim);
// P is the softmax output: post-max-subtraction it is bounded in [0,1] with the row argmax
// exactly 1.0, so a FIXED scale 256.0 (se=-8) maps it into e4m3's normal range and NEVER
// saturates (1.0*256=256 <= 448). Dropping the per-32-block amax/quad_reduce + the SF
// smem store/gather: tOrSFP becomes the compile-time constant byte (-8+127)=119. The dynamic
// per-block path is kept behind this flag only so the oracle can A/B the precision delta.
#ifndef S3_P_DYNAMIC_SCALE
#define S3_P_DYNAMIC_SCALE 0
#endif
// Decoupled so the bench can isolate the two costs separately: (a) the per-block
// amax/quad_reduce, (b) the SF smem store+gather. kPConstSF=1 (fill tOrSFP with the
// constant byte, no smem) is only valid when the scale is fixed; a dynamic scale
// MUST transit smem. Default: fixed scale -> const SF.
#ifndef S3_P_CONST_SF
#define S3_P_CONST_SF (S3_P_DYNAMIC_SCALE ? 0 : 1)
#endif
constexpr bool kPDynamicScale = (S3_P_DYNAMIC_SCALE != 0);
constexpr bool kPConstSF      = (S3_P_CONST_SF != 0);
constexpr int  kPScaleExp     = -8;            // fixed: scale = 2^-se = 256.0
constexpr int NBLK = kHeadDim / SFVecSize;        // SF blocks along head_dim (QK contraction)
constexpr int NKB  = kBlockN / SFVecSize;         // SF blocks along keys (PV contraction) = 4
constexpr int kNWarps = 12, kNThreads = kNWarps * 32;     // 384
constexpr int NumMmaThreads = 256, NumCopyThreads = 128;
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kQuantBarrier = 0;                  // named barrier id for P-smem handoff

using AtomMXF8 = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    Element, Element, float, ElementSF, SFVecSize>;
using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
using TiledMmaQK = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, Int<kHeadDim>>{}));
// PV: O[M=q, N=head_dim] = P[q, K=keys] * Vt[head_dim, K=keys]. TileK = kBlockN.
using TiledMmaPV = decltype(make_tiled_mma(
    AtomMXF8{}, Layout<Shape<_8, _1, _1>>{}, Tile<_128, _32, Int<kBlockN>>{}));

namespace ccd = cutlass::gemm::collective::detail;
using SmemLayoutAtomQ = decltype(ccd::sm120_rr_smem_selector<Element, Int<kHeadDim>>());
using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
// K rides a kStages ring (keys x head_dim x stage).
using SmemLayoutK = decltype(tile_to_shape(
    SmemLayoutAtomQ{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kStages>{})));
// P-smem transpose buffer [q, key] (single, per-nb scratch) and V [head_dim, keys].
// V stays depth-1 (the ring depth is a perf knob, orthogonal to S3's online-O goal):
// K(2 stages)+V(2 stages) = 96KB data > the 99KB sm120 opt-in smem limit.
using SmemLayoutAtomKeys = decltype(ccd::sm120_rr_smem_selector<Element, Int<kBlockN>>());
using SmemLayoutP  = decltype(tile_to_shape(SmemLayoutAtomKeys{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomKeys{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));

// Canonical block-scaled SF smem atom (128x128 operand; MMA_NSF=1). Reused for SFQ,
// and (head_dim==kBlockN) for the K and V rings.
using BlkSF = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
static constexpr int MMA_NSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVecSize;
using Blk_MN = typename BlkSF::Blk_MN; using Blk_SF = typename BlkSF::Blk_SF;
using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
using mnBasicBlockShape = Shape<_32, _4>; using mnBasicBlockStride = Stride<_16, _4>;
using kBasicBlockShape = Shape<Int<SFVecSize>, Int<MMA_NSF>>; using kBasicBlockStride = Stride<_0, _1>;
using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));
using sSF_shapeK = decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{},
                                               Int<kSFPadHD>{} / Int<SFVecSize>{} / Blk_SF{}), kBasicBlockShape{}));
using sSFA_shapeM = decltype(prepend(Int<kBlockM>{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFA_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, Int<kBlockM>{} / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
using SmemLayoutSFQ = decltype(make_layout(make_shape(sSFA_shapeM{}, sSF_shapeK{}), make_stride(sSF_strideMN{}, sSFA_strideK{})));
using sSFBTileShape_N = Int<cute::max(int(kBlockN), 128)>;
using sSFB_shapeN = decltype(prepend(sSFBTileShape_N{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFB_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, sSFBTileShape_N{} / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
using SmemLayoutAtomSFK = decltype(make_layout(make_shape(sSFB_shapeN{}, sSF_shapeK{}), make_stride(sSF_strideMN{}, sSFB_strideK{})));
// add stage dim (S1 pattern).
using SmemLayoutSFK = decltype(make_layout(
    append(shape(SmemLayoutAtomSFK{}), Int<kStages>{}),
    append(stride(SmemLayoutAtomSFK{}), size(filter_zeros(SmemLayoutAtomSFK{})))));
using SmemLayoutSFV = SmemLayoutAtomSFK;     // V depth-1

using SmemCopyAtomData = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomSF   = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

using LayoutSF = decltype(BlkSF::tile_atom_to_shape_SFA(make_shape(int(kBlockM), int(kBlockN), int(kSFPadHD))));
// V is the PV B operand [head_dim, keys], block-scaled along KEYS -> SFB layout (head_dim, keys),
// NOT K's SFA (keys, head_dim). Per-block type for the TMA descriptor; the full
// (head_dim x seqlen_k) runtime layout has the SAME C++ type (all extents dynamic int).
using LayoutSFV = decltype(BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), int(kHeadDim), int(kBlockN))));

using TMA_Q = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
        make_shape(int(kBlockM), int(kHeadDim)), make_stride(int(kHeadDim), _1{})),
    SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{}));
using TMA_K = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
        make_shape(int(8 * kBlockN), int(kHeadDim)), make_stride(int(kHeadDim), _1{})),
    SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}), _1{}));
using TMA_V = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
        make_shape(int(kHeadDim), int(8 * kBlockN)), make_stride(int(8 * kBlockN), _1{})),
    SmemLayoutVt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), _1{}));
using TMA_SFQ = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSF{}),
    SmemLayoutSFQ{}, make_shape(Int<kBlockM>{}, Int<kSFPadHD>{}), _1{}));
using TMA_SFK = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSF{}),
    SmemLayoutSFK{}(_, _, _0{}), make_shape(Int<kBlockN>{}, Int<kSFPadHD>{}), _1{}));
using TMA_SFV = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSFV{}),
    SmemLayoutSFV{}, make_shape(Int<kSFPadHD>{}, Int<kBlockN>{}), _1{}));

using PipeQ  = cutlass::PipelineTmaAsync<1>;
using PipeK  = cutlass::PipelineTmaAsync<kStages>;
using PipeV  = cutlass::PipelineTmaAsync<1>;
using StateQ  = cutlass::PipelineState<1>;
using StateK  = cutlass::PipelineState<kStages>;
using StateV  = cutlass::PipelineState<1>;

struct Params {
  TMA_Q tma_q; TMA_K tma_k; TMA_V tma_v; TMA_SFQ tma_sfq; TMA_SFK tma_sfk; TMA_SFV tma_sfv;
  LayoutSF layout_sf;           // 128x128 (per-block K)
  LayoutSF layout_sfq;          // seqlen_q x 128 (Q spans all m_blocks)
  LayoutSFV layout_sfv;         // head_dim x seqlen_k (V spans all n_blocks, scale along keys)
  int seqlen_q, seqlen_k, n_block_total;
  int const* tile_kv_len;   // OPTIONAL [num_q_tiles]: per-q-tile key count (variable-cost / varlen proxy); nullptr -> seqlen_k
  float sm_scale;
  float* out_O;     // [seqlen_q, head_dim]
  float* out_lse;   // [seqlen_q]
  float* out_l;     // [seqlen_q] row_sum (cross-check)
  float* out_Ppre;  // [seqlen_q, seqlen_k] device pre-quant fp32 P (host re-quantizes the SAME P)
  float* out_Mnb;   // [seqlen_q, n_block_total] running max after each block (-inf if skipped)
  float* out_dbg;   // [seqlen_q, seqlen_k] OPTIONAL (nullptr to skip): device dequantized requant-P
};

struct SharedStorage {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>  sQ;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>>  sK;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutVt>> sV;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutP>>  sP;
  alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFQ>> sSFQ;
  alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFK>> sSFK;
  alignas(128)  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFV>> sSFV;
  alignas(128)  ElementSF sSFP[kBlockM * NKB];   // simple [q_local][keyblock]
  struct {
    alignas(16) PipeQ::SharedStorage pipeline_q;
    alignas(16) PipeK::SharedStorage pipeline_k;
    alignas(16) PipeV::SharedStorage pipeline_v;
  };
};

static constexpr uint32_t TmaBytesQ =
    cute::cosize_v<SmemLayoutQ> * sizeof(Element) + cute::cosize_v<SmemLayoutSFQ> * sizeof(ElementSF);
static constexpr uint32_t TmaBytesK =
    cute::cosize_v<SmemLayoutK> / kStages * sizeof(Element) + cute::cosize_v<SmemLayoutSFK> / kStages * sizeof(ElementSF);
static constexpr uint32_t TmaBytesV =
    cute::cosize_v<SmemLayoutVt> * sizeof(Element) + cute::cosize_v<SmemLayoutSFV> * sizeof(ElementSF);

// e4m3 + ue8m0 quant given the block's ue8m0 scale-exponent.
__device__ __forceinline__ Element quant_e4m3(float v, int scale_exp) {
  return Element(v * exp2f(float(-scale_exp)));
}
// OCP-MX scale exponent for an e4m3 block: floor(log2(amax)) - 8 (e4m3 emax=8), bit-extracted.
__host__ __device__ __forceinline__ int mx_scale_exp_bits(uint32_t b) {
  int e = int((b >> 23) & 0xFF) - 127;
  int se = e - 8;
  return se < -127 ? -127 : (se > 127 ? 127 : se);
}
__device__ __forceinline__ int mx_scale_exp(float amax) {
  if (!(amax > 0.f)) return -127;
  return mx_scale_exp_bits(__float_as_uint(amax));
}

template <class Op>
__device__ __forceinline__ float quad_reduce(float v, Op op) {
  v = op(v, __shfl_xor_sync(uint32_t(-1), v, 2));
  v = op(v, __shfl_xor_sync(uint32_t(-1), v, 1));
  return v;
}

template <typename TileScheduler, bool Causal>
__global__ void __launch_bounds__(kNThreads, 1)
s3_kernel(CUTE_GRID_CONSTANT Params const params,
          CUTE_GRID_CONSTANT typename TileScheduler::Params const sched_params) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorage*>(smem_raw);

  int const wg = cutlass::canonical_warp_group_idx();        // 0=producer
  int const warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
  int const elect = cute::elect_one_sync();

  PipeQ::Params pq; pq.role = (wg == 0) ? PipeQ::ThreadCategory::Producer : PipeQ::ThreadCategory::Consumer;
  pq.is_leader = (threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0); pq.num_consumers = NumMmaThreads;
  pq.transaction_bytes = TmaBytesQ; PipeQ pipeline_q(ss.pipeline_q, pq, Shape<_1, _1, _1>{});
  PipeK::Params pk; pk.role = (wg == 0) ? PipeK::ThreadCategory::Producer : PipeK::ThreadCategory::Consumer;
  pk.is_leader = pq.is_leader; pk.num_consumers = NumMmaThreads;
  pk.transaction_bytes = TmaBytesK; PipeK pipeline_k(ss.pipeline_k, pk, Shape<_1, _1, _1>{});
  PipeV::Params pv; pv.role = (wg == 0) ? PipeV::ThreadCategory::Producer : PipeV::ThreadCategory::Consumer;
  pv.is_leader = pq.is_leader; pv.num_consumers = NumMmaThreads;
  pv.transaction_bytes = TmaBytesV; PipeV pipeline_v(ss.pipeline_v, pv, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sQ   = make_tensor(make_smem_ptr(ss.sQ.begin()),  SmemLayoutQ{});
  Tensor sK   = make_tensor(make_smem_ptr(ss.sK.begin()),  SmemLayoutK{});
  Tensor sV   = make_tensor(make_smem_ptr(ss.sV.begin()),  SmemLayoutVt{});
  Tensor sP   = make_tensor(make_smem_ptr(ss.sP.begin()),  SmemLayoutP{});
  Tensor sSFQ = make_tensor(make_smem_ptr(ss.sSFQ.begin()), SmemLayoutSFQ{});
  Tensor sSFK = make_tensor(make_smem_ptr(ss.sSFK.begin()), SmemLayoutSFK{});
  Tensor sSFV = make_tensor(make_smem_ptr(ss.sSFV.begin()), SmemLayoutSFV{});

  int const n_block_total = params.n_block_total;
  float const sm_scale = params.sm_scale;
  float const sm_scale_log2 = sm_scale * kLog2e;
  TileScheduler scheduler;

  if (wg == 0) {
    // -------- producer --------
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (warp_in_wg == 0 && elect) {
      Tensor mQ   = params.tma_q.get_tma_tensor(make_shape(int(params.seqlen_q), int(kHeadDim)));
      Tensor mK   = params.tma_k.get_tma_tensor(make_shape(int(params.seqlen_k), int(kHeadDim)));
      Tensor mV   = params.tma_v.get_tma_tensor(make_shape(int(kHeadDim), int(params.seqlen_k)));
      Tensor mSFQ = params.tma_sfq.get_tma_tensor(shape(params.layout_sfq));
      Tensor mSFK = params.tma_sfk.get_tma_tensor(shape(params.layout_sf));
      Tensor mSFV = params.tma_sfv.get_tma_tensor(shape(params.layout_sfv));
      auto bq = params.tma_q.get_slice(_0{}); auto bsq = params.tma_sfq.get_slice(_0{});
      auto bk = params.tma_k.get_slice(_0{}); auto bsk = params.tma_sfk.get_slice(_0{});
      auto bv = params.tma_v.get_slice(_0{}); auto bsv = params.tma_sfv.get_slice(_0{});
      StateQ wq = cutlass::make_producer_start_state<PipeQ>();
      StateK wk = cutlass::make_producer_start_state<PipeK>();
      StateV wv = cutlass::make_producer_start_state<PipeV>();

      for (auto work = scheduler.get_initial_work(sched_params); work.is_valid(sched_params);
           work = scheduler.get_next_work(sched_params, work)) {
        int const m_block = get<0>(work.get_block_coord(sched_params));
        int const nb_tile = params.tile_kv_len ? (params.tile_kv_len[m_block] + kBlockN - 1) / kBlockN : n_block_total;
        int const n_block_max = Causal
            ? cute::min(nb_tile, ((m_block + 1) * kBlockM + kBlockN - 1) / kBlockN)
            : nb_tile;

        Tensor gQ   = local_tile(mQ,   select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        Tensor gSFQ = local_tile(mSFQ, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        Tensor gK   = local_tile(mK,   select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));            // (N,K,nb)
        Tensor gSFK = local_tile(mSFK, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gV   = local_tile(mV,   make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_coord(_0{}, _));  // (hd,N,nb)
        Tensor gSFV = local_tile(mSFV, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_coord(_0{}, _));  // (hd,N,nb)
        Tensor tKgK = group_modes<0, 3>(bk.partition_S(gK)); Tensor tKsK = group_modes<0, 3>(bk.partition_D(sK));
        Tensor tKgSFK = group_modes<0, 3>(bsk.partition_S(gSFK)); Tensor tKsSFK = group_modes<0, 3>(bsk.partition_D(sSFK));
        Tensor tVgV = group_modes<0, 3>(bv.partition_S(gV)); Tensor tVsV = group_modes<0, 3>(bv.partition_D(sV));   // V depth-1: dest has no stage
        Tensor tVgSFV = group_modes<0, 3>(bsv.partition_S(gSFV)); Tensor tVsSFV = group_modes<0, 3>(bsv.partition_D(sSFV));

        pipeline_q.producer_acquire(wq);
        copy(params.tma_q.with(*pipeline_q.producer_get_barrier(wq), 0), bq.partition_S(gQ), bq.partition_D(sQ));
        copy(params.tma_sfq.with(*pipeline_q.producer_get_barrier(wq), 0), bsq.partition_S(gSFQ), bsq.partition_D(sSFQ));
        ++wq;
        for (int nb = 0; nb < n_block_max; ++nb) {
          pipeline_k.producer_acquire(wk);
          copy(params.tma_k.with(*pipeline_k.producer_get_barrier(wk), 0), tKgK(_, nb), tKsK(_, wk.index()));
          copy(params.tma_sfk.with(*pipeline_k.producer_get_barrier(wk), 0), tKgSFK(_, nb), tKsSFK(_, wk.index()));
          ++wk;
          pipeline_v.producer_acquire(wv);
          copy(params.tma_v.with(*pipeline_v.producer_get_barrier(wv), 0), tVgV(_, nb), tVsV);
          copy(params.tma_sfv.with(*pipeline_v.producer_get_barrier(wv), 0), tVgSFV(_, nb), tVsSFV);
          ++wv;
        }
      }
    }
  } else {
    // -------- consumers --------
    cutlass::arch::warpgroup_reg_alloc<232>();
    int const tid = threadIdx.x - NumCopyThreads;     // 0..255
    int const warp = tid / 32, lane = tid % 32;
    TiledMmaQK mma_qk; TiledMmaPV mma_pv;
    auto thr_qk = mma_qk.get_thread_slice(tid);
    auto thr_pv = mma_pv.get_thread_slice(tid);

    // QK operands
    Tensor tSrQ = thr_qk.partition_fragment_A(sQ);
    Tensor tSrK = thr_qk.partition_fragment_B(sK(_, _, _0{}));
    Tensor tSrSFQ = mxfp8::partition_fragment_SFA(sSFQ, thr_qk);
    Tensor tSrSFK = mxfp8::partition_fragment_SFB(sSFK(_, _, _0{}), thr_qk);
    auto scQ = make_tiled_copy_A(SmemCopyAtomData{}, mma_qk); auto tscQ = scQ.get_thread_slice(tid);
    auto scK = make_tiled_copy_B(SmemCopyAtomData{}, mma_qk); auto tscK = scK.get_thread_slice(tid);
    auto ts_qk = tile_shape(mma_qk);
    auto scSFQ = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mma_qk), make_shape(size<0>(ts_qk), size<2>(ts_qk)));
    auto scSFK = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma_qk), make_shape(size<1>(ts_qk), size<2>(ts_qk)));
    auto tscSFQ = scSFQ.get_thread_slice(tid); auto tscSFK = scSFK.get_thread_slice(tid);

    // PV operands
    Tensor tOrP  = thr_pv.partition_fragment_A(sP);
    Tensor tOrV  = thr_pv.partition_fragment_B(sV);
    Tensor tOrSFP = mxfp8::partition_fragment_SFA(sSFQ, thr_pv);     // canonical SFA layout (cosize-1)
    Tensor tOrSFV = mxfp8::partition_fragment_SFB(sSFV, thr_pv);
    auto scP = make_tiled_copy_A(SmemCopyAtomData{}, mma_pv); auto tscP = scP.get_thread_slice(tid);
    auto scV = make_tiled_copy_B(SmemCopyAtomData{}, mma_pv); auto tscV = scV.get_thread_slice(tid);
    auto ts_pv = tile_shape(mma_pv);
    auto scSFV = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma_pv), make_shape(size<1>(ts_pv), size<2>(ts_pv)));
    auto tscSFV = scSFV.get_thread_slice(tid);
    Tensor sfp_coord = mxfp8::partition_SFA(make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kBlockN>{})), thr_pv);

    auto max_op = [](float a, float b) { return fmaxf(a, b); };
    auto add_op = [](float a, float b) { return a + b; };

    Tensor mO = make_tensor(make_gmem_ptr(params.out_O),
                            make_layout(make_shape(int(params.seqlen_q), int(kHeadDim)),
                                        make_stride(int(kHeadDim), _1{})));
    Tensor mPpre = make_tensor(make_gmem_ptr(params.out_Ppre),
                               make_layout(make_shape(int(params.seqlen_q), int(params.seqlen_k)),
                                           make_stride(int(params.seqlen_k), _1{})));

    StateQ rq; StateK rk; StateV rv;
    for (auto work = scheduler.get_initial_work(sched_params); work.is_valid(sched_params);
         work = scheduler.get_next_work(sched_params, work)) {
      int const m_block = get<0>(work.get_block_coord(sched_params));
      int const nb_tile = params.tile_kv_len ? (params.tile_kv_len[m_block] + kBlockN - 1) / kBlockN : n_block_total;
      int const n_block_max = Causal
          ? cute::min(nb_tile, ((m_block + 1) * kBlockM + kBlockN - 1) / kBlockN)
          : nb_tile;

      // online-softmax row state (this thread owns 2 rows).
      float row_max[2] = {-INFINITY, -INFINITY};
      float row_sum[2] = {0.f, 0.f};
      // PV output accumulator (q x head_dim), running-rescaled across n_blocks.
      Tensor accO = partition_fragment_C(mma_pv, select<0, 2>(TileShape_MNK{}));
      Tensor accO_rc = make_tensor(accO.data(), make_layout(
          make_layout(get<0, 1>(accO.layout()), get<1>(accO.layout())),
          make_layout(get<0, 0>(accO.layout()), get<2>(accO.layout()))));
      clear(accO);

      { auto t = pipeline_q.consumer_try_wait(rq); pipeline_q.consumer_wait(rq, t);
        copy(scQ, tscQ.partition_S(as_position_independent_swizzle_tensor(sQ)), tscQ.retile_D(tSrQ));
        copy(scSFQ, tscSFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ)), tscSFQ.retile_D(tSrSFQ));
        pipeline_q.consumer_release(rq); ++rq; }

      for (int nb = 0; nb < n_block_max; ++nb) {
        // ---- QK ----
        { auto t = pipeline_k.consumer_try_wait(rk); pipeline_k.consumer_wait(rk, t);
          int stage = rk.index();
          copy(scK, tscK.partition_S(as_position_independent_swizzle_tensor(sK(_, _, stage))), tscK.retile_D(tSrK));
          copy(scSFK, tscSFK.partition_S(as_position_independent_swizzle_tensor(sSFK(_, _, stage))), tscSFK.retile_D(tSrSFK)); }
        Tensor accS = partition_fragment_C(mma_qk, select<0, 1>(TileShape_MNK{}));   // ((2,2),1,16)
        clear(accS);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tSrQ); ++k)
          cute::gemm(mma_qk, make_zip_tensor(tSrQ(_, _, k), tSrSFQ(_, _, k)),
                     make_zip_tensor(tSrK(_, _, k), tSrSFK(_, _, k)), accS);
        pipeline_k.consumer_release(rk); ++rk;

        // reduction view: ((row=2,MMA_M=1),(col=2,MMA_N=16)) = 2 rows x 32 cols.
        Tensor accS_rc = make_tensor(accS.data(), make_layout(
            make_layout(get<0, 1>(accS.layout()), get<1>(accS.layout())),
            make_layout(get<0, 0>(accS.layout()), get<2>(accS.layout()))));
        constexpr int kNRow = 2, kNCol = 32;

        if constexpr (Causal) {
          CUTLASS_PRAGMA_UNROLL
          for (int mi = 0; mi < kNRow; ++mi) {
            int m_global = m_block * kBlockM + warp * 16 + (lane / 4) + mi * 8;
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < kNCol; ++ni) {
              int col = (ni / 2) * 8 + (lane % 4) * 2 + (ni % 2);
              int n_global = nb * kBlockN + col;
              if (n_global > m_global) accS_rc(mi, ni) = -INFINITY;
            }
          }
        }

        // ---- online softmax + accO rescale factor ----
        float scores_scale[2];
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < kNRow; ++mi) {
          float m_prev = row_max[mi];
          float m_cur = m_prev;
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < kNCol; ++ni) m_cur = fmaxf(m_cur, accS_rc(mi, ni));
          m_cur = quad_reduce(m_cur, max_op);
          row_max[mi] = m_cur;
          float ss_mi = exp2f((m_prev - m_cur) * sm_scale_log2);   // first tile: m_prev=-inf -> 0
          scores_scale[mi] = ss_mi;
          // Subtract the max BEFORE scaling so the argmax gives exp2(0)=1.0 EXACTLY.
          // The `accS*sm - m_cur*sm` form lets the compiler contract to fma(accS, sm, -m_cur*sm),
          // which at the argmax yields the rounding error of m_cur*sm (~ -1.7e-7), not 0 -- so the
          // max P came out 0.99999988 < 1.0, dropping floor(log2) to -1, forcing the block scale
          // exponent one too low, and SATURATING the max element to 448*2^-9 = 0.875 (the 7/8 bug).
          float m_sub = (m_cur == -INFINITY) ? 0.f : m_cur;   // fully-masked row: keep -inf entries -> 0
          row_sum[mi] *= ss_mi;
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < kNCol; ++ni) {
            float p = exp2f((accS_rc(mi, ni) - m_sub) * sm_scale_log2);
            accS_rc(mi, ni) = p;
            row_sum[mi] += p;
          }
        }

        // dump pre-quant float P (host re-quantizes the IDENTICAL P) and running max --
        // the bit-exact reference replays the online algo from these device-side dumps.
        // Guarded by the pointer so a timing build (out_Ppre=nullptr) skips the full-P
        // gmem write, which otherwise dominates the kernel time.
        if (params.out_Ppre != nullptr) {
          Tensor gPre = local_tile(mPpre, select<0, 1>(TileShape_MNK{}), make_coord(m_block, nb));
          copy(accS, thr_qk.partition_C(gPre));
          CUTLASS_PRAGMA_UNROLL
          for (int mi = 0; mi < kNRow; ++mi) {
            int q = m_block * kBlockM + warp * 16 + (lane / 4) + mi * 8;
            if ((lane % 4) == 0) params.out_Mnb[q * n_block_total + nb] = row_max[mi];
          } }

        // ---- quantize P (e4m3 + ue8m0 per 32-key block) -> smem transpose buffer ----
        cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();   // prev nb's PV readers done
        Tensor rP = make_fragment_like<Element>(accS);
        Tensor rP_rc = make_tensor(rP.data(), accS_rc.layout());
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < kNRow; ++mi) {
          int q_local = warp * 16 + (lane / 4) + mi * 8;     // 0..127 within this m_block tile
          CUTLASS_PRAGMA_UNROLL
          for (int sfi = 0; sfi < NKB; ++sfi) {
            int se;
            if constexpr (kPDynamicScale) {
              float amax = 0.f;
              CUTLASS_PRAGMA_UNROLL
              for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(accS_rc(mi, sfi * 8 + j)));
              amax = quad_reduce(amax, max_op);
              se = mx_scale_exp(amax);
            } else {
              se = kPScaleExp;   // P<=1.0 guaranteed -> constant scale 256.0, no per-block amax
            }
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 8; ++j) rP_rc(mi, sfi * 8 + j) = quant_e4m3(accS_rc(mi, sfi * 8 + j), se);
            if (!kPConstSF && (lane % 4) == 0) ss.sSFP[q_local * NKB + sfi] = ElementSF::bitcast(uint8_t(se + 127));
            if (params.out_dbg) {     // dequantized requant-P, indexed by logical (q, key)
              int q = m_block * kBlockM + warp * 16 + (lane / 4) + mi * 8;
              CUTLASS_PRAGMA_UNROLL
              for (int j = 0; j < 8; ++j) {
                int ni = sfi * 8 + j;
                int col = (ni / 2) * 8 + (lane % 4) * 2 + (ni % 2);
                params.out_dbg[q * params.seqlen_k + nb * kBlockN + col] =
                    float(rP_rc(mi, ni)) * exp2f(float(se));
              }
            }
          }
        }
        copy(rP, thr_qk.partition_C(as_position_independent_swizzle_tensor(sP)));
        cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();   // P/SF visible

        // ---- load PV operands ----
        copy(scP, tscP.partition_S(as_position_independent_swizzle_tensor(sP)), tscP.retile_D(tOrP));
        if constexpr (kPConstSF) {   // constant P scale -> no smem SF, no gather
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFP); ++i) tOrSFP(i) = ElementSF::bitcast(uint8_t(kPScaleExp + 127));
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFP); ++i) {
            auto c = sfp_coord(i);
            tOrSFP(i) = ss.sSFP[int(get<0>(c)) * NKB + int(get<1>(c)) / SFVecSize];
          }
        }
        { auto t = pipeline_v.consumer_try_wait(rv); pipeline_v.consumer_wait(rv, t);
          copy(scV, tscV.partition_S(as_position_independent_swizzle_tensor(sV)), tscV.retile_D(tOrV));
          copy(scSFV, tscSFV.partition_S(as_position_independent_swizzle_tensor(sSFV)), tscSFV.retile_D(tOrSFV)); }

        // ---- PV into a FRESH block accumulator, then telescope onto the running accO ----
        // accO = accO*scores_scale + accB (accO is never itself a gemm target). accB native
        // ((2,2),1,MMA_N): coord ((a,b),0,c) with b = get<0,1> = M-row (scores_scale is per
        // M-row), a = N column-pair.
        Tensor accB = partition_fragment_C(mma_pv, select<0, 2>(TileShape_MNK{}));
        clear(accB);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tOrP); ++k)
          cute::gemm(mma_pv, make_zip_tensor(tOrP(_, _, k), tOrSFP(_, _, k)),
                     make_zip_tensor(tOrV(_, _, k), tOrSFV(_, _, k)), accB);
        pipeline_v.consumer_release(rv); ++rv;   // release V AFTER the gemm has consumed it
        CUTLASS_PRAGMA_UNROLL
        for (int a = 0; a < 2; ++a) {
          CUTLASS_PRAGMA_UNROLL
          for (int b = 0; b < 2; ++b) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < size<2>(accO); ++c) {
              auto coord = make_coord(make_coord(a, b), _0{}, c);
              accO(coord) = accO(coord) * scores_scale[b] + accB(coord);
            }
          }
        }
      }

      // ---- epilogue: finalize row_sum, normalize O, write LSE ----
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < 2; ++mi) row_sum[mi] = quad_reduce(row_sum[mi], add_op);
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(accO_rc); ++mi) {
        float inv = (row_sum[mi] == 0.f) ? 0.f : 1.f / row_sum[mi];
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(accO_rc); ++ni) accO_rc(mi, ni) *= inv;
      }
      Tensor gO = local_tile(mO, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
      copy(accO, thr_pv.partition_C(gO));
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < 2; ++mi) {
        int q = m_block * kBlockM + warp * 16 + (lane / 4) + mi * 8;
        if (q < params.seqlen_q && (lane % 4) == 0) {
          params.out_l[q] = row_sum[mi];
          params.out_lse[q] = (row_sum[mi] == 0.f) ? -INFINITY : (row_max[mi] * sm_scale + logf(row_sum[mi]));
        }
      }
    }
  }
}
