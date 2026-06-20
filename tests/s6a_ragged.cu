// S6a slice-1: ragged / variable-length prefill addressing (single kv-head).
//
// Validates the ragged TMA addressing added to tests/s3_kernel.cuh (the kernel now
// consumes the full BatchPrefillPersistentTileScheduler 8-tuple: qo_indptr/kv_indptr/
// qo_len/kv_len + request-local q_tile, instead of just the global m_block).
//
// ORACLE = the already-bit-exact-validated DENSE kernel, run per request. For a batch
// of B variable-length requests (each 128-padded along its sequence, qo_len==kv_len =
// full prefill), we:
//   (1) run the dense kernel (SingleTileScheduler) on each request alone -> O_dense[r],
//   (2) pack all requests into 128-padded global Q/K/V/SF buffers, build a HOST LPT
//       work-list, run the ragged kernel (BatchPrefillPersistentTileScheduler) once,
//   (3) assert O_ragged[request r's rows] == O_dense[r] (and LSE) bit-for-bit.
// The grid is forced small (S3_NUM_SM, default 2) so a single persistent CTA serially
// processes q-tiles from DIFFERENT requests -- the ragged-addressing correctness trap.
//
// Build:
//   nvcc -std=c++17 -O2 -gencode arch=compute_120a,code=sm_120a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -I tmp/cutlass/include -I include tests/s6a_ragged.cu -o tests/s6a_ragged

#include <cstdlib>
#include <algorithm>
#include <queue>
#include <vector>
#include "s3_kernel.cuh"

using namespace flashinfer;

static uint8_t e4m3_byte(float v) { return cutlass::float_e4m3_t(v).storage; }
static uint8_t ue8m0_byte_pow2(int e) { return uint8_t(e + 127); }
static int query_num_sm() {
  if (const char* e = std::getenv("S3_NUM_SM")) return std::atoi(e);
  int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); return n;
}
static int cdiv(int a, int b) { return (a + b - 1) / b; }

// ---- per-request logical data (deterministic, salted by request so requests differ) ----
struct ReqData {
  int qo_len, kv_len, qo_pad, kv_pad;
  std::vector<uint8_t> Q, K, V;              // e4m3 bytes; Q[qo_pad,HD] K[kv_pad,HD] V[HD,kv_pad]
  std::vector<int> qexp, kexp, vexp;         // ue8m0 exponents: q[qo_pad,NBLK] k[kv_pad,NBLK] v[HD,kv_pad/32]
};

static ReqData gen_req(int r, int qo_len, int kv_len, int HD) {
  ReqData d;
  d.qo_len = qo_len; d.kv_len = kv_len;
  d.qo_pad = cdiv(qo_len, kBlockM) * kBlockM;
  d.kv_pad = cdiv(kv_len, kBlockN) * kBlockN;
  const int NBLK = HD / SFVecSize, NVK = d.kv_pad / SFVecSize;
  const std::vector<float> dv = {0.5f, 1.f, 2.f, -0.5f, -1.f, -2.f};
  d.Q.assign(d.qo_pad * HD, 0); d.K.assign(d.kv_pad * HD, 0); d.V.assign(HD * d.kv_pad, 0);
  d.qexp.assign(d.qo_pad * NBLK, -127); d.kexp.assign(d.kv_pad * NBLK, -127); d.vexp.assign(HD * NVK, -127);
  // only the REAL (un-padded) region is populated; padding stays zero / exp -127 (masked out).
  for (int m = 0; m < qo_len; ++m) {
    for (int b = 0; b < NBLK; ++b) d.qexp[m * NBLK + b] = ((m + b + r) % 4) - 1;
    for (int k = 0; k < HD; ++k) d.Q[m * HD + k] = e4m3_byte(dv[(m * HD + k + r * 13) % dv.size()]);
  }
  for (int n = 0; n < kv_len; ++n) {
    for (int b = 0; b < NBLK; ++b) d.kexp[n * NBLK + b] = (n + 2 * b + r) % 3;
    for (int k = 0; k < HD; ++k) d.K[n * HD + k] = e4m3_byte(dv[(n * 7 + k * 3 + r * 5) % dv.size()]);
  }
  for (int h = 0; h < HD; ++h) {
    for (int n = 0; n < kv_len; ++n) {
      d.V[h * d.kv_pad + n] = e4m3_byte(dv[(h * 5 + n * 2 + r * 11) % dv.size()]);
      d.vexp[h * NVK + n / SFVecSize] = ((h + n / SFVecSize + r) % 3) - 1;
    }
  }
  return d;
}

// scatter a logical [rows,blocks] exponent table into a cutlass SF tiled layout, at a
// global ROW offset (the request's 128-padded base). row_off is a multiple of 128 so it
// lands on a Blk_MN block boundary of the blocked layout.
template <class Layout>
static void place_sf(std::vector<uint8_t>& dst, Layout layout, const std::vector<int>& exp,
                     int rows, int blocks, int row_off) {
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < blocks; ++b)
      dst[layout(make_coord(row_off + r, b * SFVecSize))] = ue8m0_byte_pow2(exp[r * blocks + b]);
}

// Launch s3_kernel for any scheduler; copies O/LSE back to host.
template <typename Scheduler, bool Causal>
static void launch(Params params, typename Scheduler::Arguments sa, dim3 grid,
                   std::vector<float>& hO, std::vector<float>& hLSE, int rowsO, int HD) {
  typename Scheduler::Params sp = Scheduler::to_underlying_arguments(sa);
  int smem = int(sizeof(SharedStorage));
  if (cudaFuncSetAttribute(s3_kernel<Scheduler, Causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
    { printf("smem attr fail\n"); std::exit(1); }
  s3_kernel<Scheduler, Causal><<<grid, kNThreads, smem>>>(params, sp);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
    { printf("launch fail: %s\n", cudaGetErrorString(cudaGetLastError())); std::exit(1); }
  hO.resize(rowsO * HD); hLSE.resize(rowsO);
  cudaMemcpy(hO.data(), params.out_O, hO.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hLSE.data(), params.out_lse, hLSE.size() * sizeof(float), cudaMemcpyDeviceToHost);
}

// Build TMA params over device tensors of the given (padded) global dims.
static Params make_params(Element* dQ, Element* dK, Element* dV, ElementSF* dSFQ, ElementSF* dSFK,
                          ElementSF* dSFV, float* dO, float* dLSE, float* dL, float* dMnb,
                          int Sq_pad, int Sk_pad, int HD, float sm_scale) {
  auto layoutSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(Sq_pad, int(kBlockN), HD));
  auto layoutSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(Sk_pad, int(kBlockN), HD));
  auto layoutSFV = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), HD, Sk_pad));
  Tensor mQ = make_tensor(make_gmem_ptr(dQ), make_shape(Sq_pad, HD), make_stride(HD, _1{}));
  Tensor mK = make_tensor(make_gmem_ptr(dK), make_shape(Sk_pad, HD), make_stride(HD, _1{}));
  Tensor mV = make_tensor(make_gmem_ptr(dV), make_shape(HD, Sk_pad), make_stride(Sk_pad, _1{}));
  Tensor mSFQ = make_tensor(make_gmem_ptr(dSFQ), layoutSFQ);
  Tensor mSFK = make_tensor(make_gmem_ptr(dSFK), layoutSFK);
  Tensor mSFV = make_tensor(make_gmem_ptr(dSFV), layoutSFV);
  Params p;
  p.tma_q   = make_tma_copy(SM90_TMA_LOAD{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{});
  p.tma_k   = make_tma_copy(SM90_TMA_LOAD{}, mK, SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}), _1{});
  p.tma_v   = make_tma_copy(SM90_TMA_LOAD{}, mV, SmemLayoutVt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), _1{});
  p.tma_sfq = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFQ, SmemLayoutSFQ{}, make_shape(Int<kBlockM>{}, Int<kSFPadHD>{}), _1{});
  p.tma_sfk = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFK, SmemLayoutSFK{}(_, _, _0{}), make_shape(Int<kBlockN>{}, Int<kSFPadHD>{}), _1{});
  p.tma_sfv = make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFV, SmemLayoutSFV{}, make_shape(Int<kSFPadHD>{}, Int<kBlockN>{}), _1{});
  p.layout_sfq = layoutSFQ; p.layout_sfv = layoutSFV;
  p.seqlen_q = Sq_pad; p.seqlen_k = Sk_pad; p.n_block_total = Sk_pad / kBlockN; p.sm_scale = sm_scale;
  p.out_O = dO; p.out_lse = dLSE; p.out_l = dL; p.out_Ppre = nullptr; p.out_Mnb = dMnb; p.out_dbg = nullptr;
  p.tile_kv_len = nullptr;
  return p;
}

int main() {
  const int HD = kHeadDim;
  const float sm_scale = 1.0f / std::sqrt((float)HD);
  // varlen batch: distinct lengths; 200/360 are NON-128-multiples -> exercise the partial
  // last-tile key mask (padded keys past kv_len must be dropped, esp. non-causal).
  std::vector<std::pair<int,int>> lens = {{128,128}, {256,256}, {200,200}, {384,384}, {360,360}};
  const int B = lens.size();
  printf("S6a: ragged varlen prefill (single kv-head), B=%d requests, head_dim=%d\n", B, HD);

  std::vector<ReqData> reqs;
  std::vector<int> qo_base(B), kv_base(B);
  int Sq_pad = 0, Sk_pad = 0;
  for (int r = 0; r < B; ++r) {
    reqs.push_back(gen_req(r, lens[r].first, lens[r].second, HD));
    qo_base[r] = Sq_pad; kv_base[r] = Sk_pad;
    Sq_pad += reqs[r].qo_pad; Sk_pad += reqs[r].kv_pad;
    printf("  req %d: qo_len=%d kv_len=%d (qo_pad=%d kv_pad=%d) @ qo_base=%d kv_base=%d\n",
           r, reqs[r].qo_len, reqs[r].kv_len, reqs[r].qo_pad, reqs[r].kv_pad, qo_base[r], kv_base[r]);
  }
  const int NBLK = HD / SFVecSize;

  // ============================ (1) per-request DENSE oracle ============================
  std::vector<std::vector<float>> Odense(B), LSEdense(B);
  for (bool causal : {false, true}) {
    for (int r = 0; r < B; ++r) {
      auto& d = reqs[r];
      const int NVK = d.kv_pad / SFVecSize;
      auto lSFQ = BlkSF::tile_atom_to_shape_SFA(make_shape(d.qo_pad, int(kBlockN), HD));
      auto lSFK = BlkSF::tile_atom_to_shape_SFA(make_shape(d.kv_pad, int(kBlockN), HD));
      auto lSFV = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), HD, d.kv_pad));
      std::vector<uint8_t> hSFQ(cosize(lSFQ), 0), hSFK(cosize(lSFK), 0), hSFV(cosize(lSFV), 0);
      place_sf(hSFQ, lSFQ, d.qexp, d.qo_pad, NBLK, 0);
      place_sf(hSFK, lSFK, d.kexp, d.kv_pad, NBLK, 0);
      place_sf(hSFV, lSFV, d.vexp, HD, NVK, 0);

      Element *dQ,*dK,*dV; ElementSF *dSFQ,*dSFK,*dSFV; float *dO,*dLSE,*dL,*dMnb;
      cudaMalloc(&dQ, d.Q.size()); cudaMalloc(&dK, d.K.size()); cudaMalloc(&dV, d.V.size());
      cudaMalloc(&dSFQ, hSFQ.size()); cudaMalloc(&dSFK, hSFK.size()); cudaMalloc(&dSFV, hSFV.size());
      cudaMalloc(&dO, d.qo_pad * HD * sizeof(float)); cudaMalloc(&dLSE, d.qo_pad * sizeof(float));
      cudaMalloc(&dL, d.qo_pad * sizeof(float)); cudaMalloc(&dMnb, size_t(d.qo_pad) * (d.kv_pad / kBlockN) * sizeof(float));
      cudaMemcpy(dQ, d.Q.data(), d.Q.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(dK, d.K.data(), d.K.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(dV, d.V.data(), d.V.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(dSFQ, hSFQ.data(), hSFQ.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(dSFK, hSFK.data(), hSFK.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(dSFV, hSFV.data(), hSFV.size(), cudaMemcpyHostToDevice);

      Params p = make_params(dQ, dK, dV, dSFQ, dSFK, dSFV, dO, dLSE, dL, dMnb, d.qo_pad, d.kv_pad, HD, sm_scale);
      using Sched = SingleTileScheduler;
      Sched::Arguments sa{cdiv(d.qo_len, kBlockM), 1, d.qo_len, d.kv_len, cutlass::FastDivmod(1)};
      dim3 grid = Sched::get_grid_dim(sa, query_num_sm());
      std::vector<float> hO, hLSE;
      if (causal) launch<Sched, true >(p, sa, grid, hO, hLSE, d.qo_pad, HD);
      else        launch<Sched, false>(p, sa, grid, hO, hLSE, d.qo_pad, HD);
      Odense[r] = hO; LSEdense[r] = hLSE;
      cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dSFQ); cudaFree(dSFK); cudaFree(dSFV);
      cudaFree(dO); cudaFree(dLSE); cudaFree(dL); cudaFree(dMnb);
    }

    // ============================ (2) packed RAGGED run ============================
    auto gSFQ_l = BlkSF::tile_atom_to_shape_SFA(make_shape(Sq_pad, int(kBlockN), HD));
    auto gSFK_l = BlkSF::tile_atom_to_shape_SFA(make_shape(Sk_pad, int(kBlockN), HD));
    auto gSFV_l = BlkSF::tile_atom_to_shape_SFB(make_shape(int(kBlockM), HD, Sk_pad));
    std::vector<uint8_t> gQ(Sq_pad * HD, 0), gK(Sk_pad * HD, 0), gV(HD * Sk_pad, 0);
    std::vector<uint8_t> gSFQ(cosize(gSFQ_l), 0), gSFK(cosize(gSFK_l), 0), gSFV(cosize(gSFV_l), 0);
    for (int r = 0; r < B; ++r) {
      auto& d = reqs[r]; const int NVK = d.kv_pad / SFVecSize;
      // data: copy request's padded block into the global packed buffer at its base.
      std::copy(d.Q.begin(), d.Q.end(), gQ.begin() + size_t(qo_base[r]) * HD);
      std::copy(d.K.begin(), d.K.end(), gK.begin() + size_t(kv_base[r]) * HD);
      for (int h = 0; h < HD; ++h)  // V is [HD, Sk_pad]: per-row copy into the global column range
        std::copy(d.V.begin() + size_t(h) * d.kv_pad, d.V.begin() + size_t(h + 1) * d.kv_pad,
                  gV.begin() + size_t(h) * Sk_pad + kv_base[r]);
      // SF: re-place by GLOBAL row/col offset (blocked layout, 128-aligned base).
      place_sf(gSFQ, gSFQ_l, d.qexp, d.qo_pad, NBLK, qo_base[r]);
      place_sf(gSFK, gSFK_l, d.kexp, d.kv_pad, NBLK, kv_base[r]);
      // V SF spans keys; place at global key-column base (head-row offset 0).
      for (int h = 0; h < HD; ++h)
        for (int b = 0; b < NVK; ++b)
          gSFV[gSFV_l(make_coord(h, kv_base[r] + b * SFVecSize))] = ue8m0_byte_pow2(d.vexp[h * NVK + b]);
    }

    Element *dQ,*dK,*dV; ElementSF *dSFQ,*dSFK,*dSFV; float *dO,*dLSE,*dL,*dMnb;
    cudaMalloc(&dQ, gQ.size()); cudaMalloc(&dK, gK.size()); cudaMalloc(&dV, gV.size());
    cudaMalloc(&dSFQ, gSFQ.size()); cudaMalloc(&dSFK, gSFK.size()); cudaMalloc(&dSFV, gSFV.size());
    cudaMalloc(&dO, size_t(Sq_pad) * HD * sizeof(float)); cudaMalloc(&dLSE, Sq_pad * sizeof(float));
    cudaMalloc(&dL, Sq_pad * sizeof(float)); cudaMalloc(&dMnb, size_t(Sq_pad) * (Sk_pad / kBlockN) * sizeof(float));
    cudaMemset(dO, 0, size_t(Sq_pad) * HD * sizeof(float));
    cudaMemcpy(dQ, gQ.data(), gQ.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, gK.data(), gK.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, gV.data(), gV.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dSFQ, gSFQ.data(), gSFQ.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dSFK, gSFK.data(), gSFK.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dSFV, gSFV.data(), gSFV.size(), cudaMemcpyHostToDevice);
    Params p = make_params(dQ, dK, dV, dSFQ, dSFK, dSFV, dO, dLSE, dL, dMnb, Sq_pad, Sk_pad, HD, sm_scale);

    // ---- HOST LPT work-list: enumerate (request, q-tile), heaviest-first onto CTAs ----
    int num_sm = query_num_sm();
    struct W { int req, qtile; long cost; };
    std::vector<W> works;
    for (int r = 0; r < B; ++r) {
      int nqt = cdiv(reqs[r].qo_len, kBlockM), nkt = cdiv(reqs[r].kv_len, kBlockN);
      for (int qt = 0; qt < nqt; ++qt) {
        int eff = causal ? std::min(nkt, cdiv((qt + 1) * kBlockM, kBlockN)) : nkt;
        works.push_back({r, qt, (long)eff});
      }
    }
    std::stable_sort(works.begin(), works.end(), [](const W& a, const W& b) { return a.cost > b.cost; });
    using PQ = std::pair<long, int>;
    std::priority_queue<PQ, std::vector<PQ>, std::greater<PQ>> heap;
    for (int c = 0; c < num_sm; ++c) heap.push({0, c});
    std::vector<std::vector<W>> cta(num_sm);
    for (auto& w : works) { auto [load, c] = heap.top(); heap.pop(); cta[c].push_back(w); heap.push({load + w.cost, c}); }
    std::vector<int> work_indptr(num_sm + 1, 0), head_i, qo_tile_i, qo_ip, kv_ip, qo_l, kv_l, batch_i;
    for (int c = 0; c < num_sm; ++c) work_indptr[c + 1] = work_indptr[c] + cta[c].size();
    for (int c = 0; c < num_sm; ++c) for (auto& w : cta[c]) {
      head_i.push_back(0); qo_tile_i.push_back(w.qtile);
      qo_ip.push_back(qo_base[w.req]); kv_ip.push_back(kv_base[w.req]);
      qo_l.push_back(reqs[w.req].qo_len); kv_l.push_back(reqs[w.req].kv_len); batch_i.push_back(w.req);
    }
    int total_works = work_indptr[num_sm];
    auto up = [](const std::vector<int>& v) { int* d; cudaMalloc(&d, v.size() * 4); cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice); return d; };
    using Sched = BatchPrefillPersistentTileScheduler<int>;
    Sched::Arguments sa;
    sa.work_indptr = up(work_indptr); sa.head_indices = up(head_i); sa.qo_tile_indices = up(qo_tile_i);
    sa.qo_indptr = up(qo_ip); sa.kv_indptr = up(kv_ip); sa.qo_lens = up(qo_l); sa.kv_lens = up(kv_l);
    sa.batch_indices = up(batch_i); sa.group_size_fastdiv = cutlass::FastDivmod(1); sa.num_qo_heads = 1;
    dim3 grid = Sched::get_grid_dim(sa, num_sm);
    printf("  [%s] grid=%u CTAs, %d work-items (LPT)\n", causal ? "causal" : "non-causal", grid.x, total_works);

    std::vector<float> rO, rLSE;
    if (causal) launch<Sched, true >(p, sa, grid, rO, rLSE, Sq_pad, HD);
    else        launch<Sched, false>(p, sa, grid, rO, rLSE, Sq_pad, HD);

    // ============================ (3) compare ragged vs per-request dense ============================
    double max_abs = 0, max_lse = 0; int bad = 0, fr = -1, fm = -1, fh = -1;
    for (int r = 0; r < B; ++r) {
      for (int m = 0; m < reqs[r].qo_len; ++m) {
        int gm = qo_base[r] + m;
        double dl = std::abs((double)rLSE[gm] - (double)LSEdense[r][m]);
        max_lse = std::max(max_lse, dl);
        for (int h = 0; h < HD; ++h) {
          double a = rO[size_t(gm) * HD + h], b = Odense[r][size_t(m) * HD + h];
          double e = std::abs(a - b);
          if (e > max_abs) max_abs = e;
          if (e > 1e-5) { if (!bad) { fr = r; fm = m; fh = h; } ++bad; }
        }
      }
    }
    printf("  [%s] ragged-vs-dense O max|abs|=%.3g | LSE max|abs|=%.3g | bad(>1e-5)=%d\n",
           causal ? "causal" : "non-causal", max_abs, max_lse, bad);
    if (bad) printf("    first bad: req=%d q=%d h=%d  ragged=%.6f dense=%.6f\n",
                    fr, fm, fh, rO[size_t(qo_base[fr] + fm) * HD + fh], Odense[fr][size_t(fm) * HD + fh]);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dSFQ); cudaFree(dSFK); cudaFree(dSFV);
    cudaFree(dO); cudaFree(dLSE); cudaFree(dL); cudaFree(dMnb);
    cudaFree(sa.work_indptr); cudaFree(sa.head_indices); cudaFree(sa.qo_tile_indices); cudaFree(sa.qo_indptr);
    cudaFree(sa.kv_indptr); cudaFree(sa.qo_lens); cudaFree(sa.kv_lens); cudaFree(sa.batch_indices);
    if (bad) { printf("S6a FAIL (%s)\n", causal ? "causal" : "non-causal"); return 1; }
  }
  printf("S6a PASS\n");
  return 0;
}
