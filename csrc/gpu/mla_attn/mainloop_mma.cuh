// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ATTENTION_HOPPER_MAINLOOP_MMA_CUH_
#define ATTENTION_HOPPER_MAINLOOP_MMA_CUH_

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "named_barrier.cuh"

// #define DEBUG_MLA

namespace mla_attn {

template <typename Ktraits, bool CAUSAL, typename Params, typename MainloopPipeline, typename MainloopPipelineQ,
          typename PipelineState, typename PipelineStateQ, typename SharedStorage, typename FrgTensorO, typename AttentionUpdater>
CUTLASS_DEVICE void mma_f16(const Params& mainloop_params,
                            MainloopPipelineQ pipeline_q,
                            PipelineStateQ& smem_pipe_read_q,
                            MainloopPipeline pipeline_kv,
                            PipelineState& smem_pipe_read_kv,
                            FrgTensorO& tOrO, 
                            AttentionUpdater& attention_updater,
                            const int thread_idx, 
                            const int bid,
                            const int kv_len,
                            const int qo_len,
                            const int tile_idx,
                            SharedStorage& shared_storage) {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeMD = typename Ktraits::DTypeO;
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutP = typename Ktraits::SmemLayoutP;
  using SmemLayoutRow = typename Ktraits::SmemLayoutRow;
  using SmemCopyAtom = typename Ktraits::SmemCopyAtom;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;
  using SmemLayoutVtOneStage = typename Ktraits::SmemLayoutVtOneStage;
  static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

  const int chunk_num_this_seq = cute::ceil_div(kv_len, mainloop_params.chunk_size);

  static constexpr int BLOCK_SHAPE_Q = get<0>(TileShape_QKD{});
  static constexpr int BLOCK_SHAPE_KV = get<1>(TileShape_QKD{});

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutK{});
  Tensor sVt_s1 = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutVtOneStage{});
  Tensor sVt_s2 = make_tensor(make_smem_ptr(shared_storage.smem_kv.data() + Ktraits::NUM_PER_STAGE), SmemLayoutVtOneStage{});
  Tensor sPSS = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{});
  Tensor s_scale = make_tensor(make_smem_ptr(shared_storage.smem_scale.data()), SmemLayoutRow{});
  Tensor mM = make_tensor(make_gmem_ptr(mainloop_params.m_ptr), mainloop_params.layout_MD)(tile_idx, _); // (bsz * draft_token_num * num_head)
  Tensor mD = make_tensor(make_gmem_ptr(mainloop_params.d_ptr), mainloop_params.layout_MD)(tile_idx, _);

  typename Ktraits::TiledMmaQK tiled_mma_qk;
  auto threadMmaQK = tiled_mma_qk.get_thread_slice(thread_idx);
  auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtom{}, tiled_mma_qk);
  auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);
  Tensor tPsP = smem_thr_copy_P.partition_D(sPSS);
  Tensor tScalesScale = s_scale(_, thread_idx % cutlass::NumThreadsPerWarpGroup);

  typename Ktraits::TiledMmaPVSS tiled_mma_pv_ss;
  auto threadMmaPVSS = tiled_mma_pv_ss.get_thread_slice(thread_idx);
  Tensor tOrV1 = threadMmaPVSS.partition_fragment_B(sVt_s1);
  Tensor tOrV2 = threadMmaPVSS.partition_fragment_B(sVt_s2);
  Tensor tOrP_CS2 = threadMmaPVSS.partition_fragment_A(sPSS);

  const int start_len = tile_idx * mainloop_params.chunk_size;
  const int start_tile_idx = start_len / BLOCK_SHAPE_KV;
  const int end_tile_idx =cute::ceil_div(min(start_len + mainloop_params.chunk_size, kv_len), BLOCK_SHAPE_KV) - 1;
  int kv_tile_idx = end_tile_idx;

  auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
  };

  int warp_group_idx = cutlass::canonical_warp_group_idx();
  if (warp_group_idx == 1) {
    // consumer 0, compute qk
    Tensor tSrQ = threadMmaQK.partition_fragment_A(sQ);
    Tensor tSrK = threadMmaQK.partition_fragment_B(sK);

    constexpr int n_masking_steps = !CAUSAL ? 1 : cute::ceil_div(BLOCK_SHAPE_Q, BLOCK_SHAPE_KV) + 1;
    auto col_limit_right = [&](int qo_idx) { return qo_idx + 1 + kv_len - qo_len; };
    bool is_first_step = true;
    // wait q
    consumer_wait(pipeline_q, smem_pipe_read_q);
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
#pragma unroll 1
    for (int masking_step = n_masking_steps; kv_tile_idx >= start_tile_idx; --masking_step, --kv_tile_idx) {
      // wait kv
      consumer_wait(pipeline_kv, smem_pipe_read_kv);
      // gemm qk
      gemm</*init=*/true, /*wg_wait=*/0>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_kv.index()),
                                         tSrS);
      // mask
      if (masking_step > 0) {
        Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_QKD{}));
        Tensor tScS = threadMmaQK.partition_C(cS);
#pragma unroll
        for (int i = 0; i < size(tSrS); ++i) {
          int qo_idx = get<0>(tScS(i)) / Ktraits::GROUP_SIZE;
          int kv_idx = get<1>(tScS(i)) + kv_tile_idx * BLOCK_SHAPE_KV;
          if constexpr (!CAUSAL) {  // Just masking based on col
            if (kv_idx >= kv_len) {
              tSrS(i) = AttentionUpdater::fill_value;
            }
          } else {
            if (kv_idx >= std::min(kv_len, col_limit_right(qo_idx))) {
              tSrS(i) = AttentionUpdater::fill_value;
            }
          }
        }
      }

      // update s (exp(s - m))
      Tensor scale_o = is_first_step ? attention_updater.update</*init=*/true>(tSrS) : attention_updater.update</*init=*/false>(tSrS);
      is_first_step = false;

      Tensor convert_tSrS = convert_type<DTypeKV>(tSrS);
      Tensor tPrP = smem_thr_copy_P.retile_S(convert_tSrS);

      // gather qk gemm res
      cute::copy(smem_tiled_copy_P, tPrP, tPsP);
      cute::copy(scale_o, tScalesScale);
      // make sure r2s all done
      cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWarpSchedulerWG1));

      attention_updater.rescale_o(tOrO, scale_o);

      // pv gemm
      if (smem_pipe_read_kv.index() == 0) {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_CS2,
                                            tOrV1(_, _, _, _0{}), tOrO);
      } else {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_CS2,
                                            tOrV2(_, _, _, _0{}), tOrO);
      }

      pipeline_kv.consumer_release(smem_pipe_read_kv);
      ++smem_pipe_read_kv;
      // sync WG1 WG2
      cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWG1WG2Sync));
    } 
    // release q
    pipeline_q.consumer_release(smem_pipe_read_q);
    ++smem_pipe_read_q;

    // normalize
    Tensor scale_o = attention_updater.finalize(tSrS); // warp reduce row sum
    if (chunk_num_this_seq == 1) {
      // norm
      cute::copy(scale_o, tScalesScale);

      cutlass::arch::NamedBarrier::arrive(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWarpSchedulerWG2));
      attention_updater.rescale_o(tOrO, scale_o);
    }

    // WG1 write m,d back to gmem
    if (chunk_num_this_seq > 1 && thread_idx % 4 == 0) { // 16 rows per warp, eg. t0->row0 row8，t4->row1 row9
      const int warp_idx = thread_idx / 32;
#pragma unroll
      for (int w_i = 0; w_i < 2; ++w_i) {
        const int token_group_idx = warp_idx * 16 + thread_idx / 4 + 8 * w_i;
        const int token_idx = token_group_idx / Ktraits::GROUP_SIZE;

        if (token_idx < qo_len) {
          const int head_idx = token_group_idx % Ktraits::GROUP_SIZE;
          const int bid_offset = mainloop_params.max_draft_token_num * Ktraits::GROUP_SIZE;
          const int write_idx = bid * bid_offset + token_idx * Ktraits::GROUP_SIZE + head_idx;
          mM(write_idx) = static_cast<DTypeMD>(attention_updater.row_max(w_i));
          mD(write_idx) = static_cast<DTypeMD>(attention_updater.row_sum(w_i));
        }
      }
    }
  } else if (warp_group_idx == 2) {
    // consumer 1, compute pv
    Tensor scale_o = make_tensor<DTypeQKAccum>(Shape<_2>{});
    for (; kv_tile_idx >= start_tile_idx; --kv_tile_idx) {
      // wait kv
      consumer_wait(pipeline_kv, smem_pipe_read_kv);
      cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWarpSchedulerWG1));

      // A: tPsP
      cute::copy(tScalesScale, scale_o);

      // rescale
      attention_updater.rescale_o(tOrO, scale_o);
      if (smem_pipe_read_kv.index() == 0) {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_CS2,
                                            tOrV1(_, _, _, _0{}), tOrO);
      } else {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_CS2,
                                            tOrV2(_, _, _, _0{}), tOrO);
      }

      pipeline_kv.consumer_release(smem_pipe_read_kv);
      ++smem_pipe_read_kv;
      // sync WG1 WG2
      cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWG1WG2Sync));
    }
    if (chunk_num_this_seq == 1) {
      // norm
      cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWarpSchedulerWG2));
      cute::copy(tScalesScale, scale_o);
      attention_updater.rescale_o(tOrO, scale_o);
    }
  }
  return;
}

template <typename Ktraits, bool CAUSAL, typename Params, typename MainloopPipeline, typename MainloopPipelineQ,
          typename PipelineState, typename PipelineStateQ, typename SharedStorage, typename FrgTensorO, typename AttentionUpdater>
CUTLASS_DEVICE void mma_f16_two_stages(const Params& mainloop_params,
                                       MainloopPipelineQ pipeline_q,
                                       PipelineStateQ& smem_pipe_read_q,
                                       MainloopPipeline pipeline_kv,
                                       PipelineState& smem_pipe_read_kv,
                                       FrgTensorO& tOrO, 
                                       AttentionUpdater& attention_updater,
                                       const int thread_idx, 
                                       const int bid,
                                       const int kv_len,
                                       const int qo_len,
                                       const int tile_idx,
                                       SharedStorage& shared_storage) {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeMD = typename Ktraits::DTypeO; // !!! bf16
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutP = typename Ktraits::SmemLayoutP;
  using SmemLayoutRow = typename Ktraits::SmemLayoutRow;
  using SmemCopyAtom = typename Ktraits::SmemCopyAtom;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;
  using SmemLayoutVtOneStage = typename Ktraits::SmemLayoutVtOneStage;
  static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

  const int chunk_num_this_seq = cute::ceil_div(kv_len, mainloop_params.chunk_size);

  static constexpr int BLOCK_SHAPE_Q = get<0>(TileShape_QKD{});
  static constexpr int BLOCK_SHAPE_KV = get<1>(TileShape_QKD{});

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutK{});
  Tensor sVt_s1 = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutVtOneStage{});
  Tensor sVt_s2 = make_tensor(make_smem_ptr(shared_storage.smem_kv.data() + Ktraits::NUM_PER_STAGE), SmemLayoutVtOneStage{});
  Tensor sVt_s3 = make_tensor(make_smem_ptr(shared_storage.smem_kv.data() + 2 * Ktraits::NUM_PER_STAGE), SmemLayoutVtOneStage{});
  Tensor sVt_s4 = make_tensor(make_smem_ptr(shared_storage.smem_kv.data() + 3 * Ktraits::NUM_PER_STAGE), SmemLayoutVtOneStage{});
  Tensor sPSS = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{});
  Tensor mM = make_tensor(make_gmem_ptr(mainloop_params.m_ptr), mainloop_params.layout_MD)(tile_idx, _);
  Tensor mD = make_tensor(make_gmem_ptr(mainloop_params.d_ptr), mainloop_params.layout_MD)(tile_idx, _);

  Tensor s_scale = make_tensor(make_smem_ptr(shared_storage.smem_scale.data()), SmemLayoutRow{});

  typename Ktraits::TiledMmaQK tiled_mma_qk;
  auto threadMmaQK = tiled_mma_qk.get_thread_slice(thread_idx);
  auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtom{}, tiled_mma_qk);
  auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);
  Tensor tPsP = smem_thr_copy_P.partition_D(sPSS);
  Tensor tScalesScale = s_scale(_, thread_idx % cutlass::NumThreadsPerWarpGroup, _);

  typename Ktraits::TiledMmaPVSS tiled_mma_pv_ss;
  auto threadMmaPVSS = tiled_mma_pv_ss.get_thread_slice(thread_idx);
  Tensor tOrV1 = threadMmaPVSS.partition_fragment_B(sVt_s1);
  Tensor tOrV2 = threadMmaPVSS.partition_fragment_B(sVt_s2);
  Tensor tOrV3 = threadMmaPVSS.partition_fragment_B(sVt_s3);
  Tensor tOrV4 = threadMmaPVSS.partition_fragment_B(sVt_s4);
  Tensor tOrP_CS2 = threadMmaPVSS.partition_fragment_A(sPSS);

  const int start_len = tile_idx * mainloop_params.chunk_size;
  const int start_tile_idx = start_len / BLOCK_SHAPE_KV;
  const int end_tile_idx = cute::ceil_div(min(start_len + mainloop_params.chunk_size, kv_len), BLOCK_SHAPE_KV) - 1;
  int kv_tile_idx = end_tile_idx;

  auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
  };

  int warp_group_idx = cutlass::canonical_warp_group_idx();
  if (warp_group_idx == 1) {
    // consumer 0, compute qk
    Tensor tSrQ = threadMmaQK.partition_fragment_A(sQ);
    Tensor tSrK = threadMmaQK.partition_fragment_B(sK);
    auto col_limit_right = [&](int qo_idx) { return qo_idx + 1 + kv_len - qo_len; };
    // wait q
    consumer_wait(pipeline_q, smem_pipe_read_q);
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
    // wait k
    consumer_wait(pipeline_kv, smem_pipe_read_kv);
    // first qk gemm
    gemm</*init=*/true, /*wg_wait=*/0>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_kv.index()),
                                       tSrS);
    // mask
    {
      Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_QKD{}));
      Tensor tScS = threadMmaQK.partition_C(cS);
#pragma unroll
      for (int i = 0; i < size(tSrS); ++i) {
        int qo_idx = get<0>(tScS(i)) / Ktraits::GROUP_SIZE;
        int kv_idx = get<1>(tScS(i)) + kv_tile_idx * BLOCK_SHAPE_KV;
        if constexpr (!CAUSAL) {  // Just masking based on col
          if (kv_idx >= kv_len) {
            tSrS(i) = AttentionUpdater::fill_value;
          }
        } else {
          if (kv_idx >= std::min(kv_len, col_limit_right(qo_idx))) {
            tSrS(i) = AttentionUpdater::fill_value;
          }
        }
      }
    }

    Tensor scale_o = attention_updater.update</*init=*/true>(tSrS);
    Tensor tPrP = smem_thr_copy_P.retile_S(convert_type<DTypeKV>(tSrS));
    // gather qk gemm res
    cute::copy(smem_tiled_copy_P, tPrP, tPsP(_, _, _, smem_pipe_read_kv.index() % 2));
    cute::copy(scale_o, tScalesScale(_, smem_pipe_read_kv.index() % 2));
    cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWarpSchedulerWG1));

    constexpr int n_masking_steps = CAUSAL ? cute::ceil_div(BLOCK_SHAPE_Q, BLOCK_SHAPE_KV) : 0;
    --kv_tile_idx;
    for (int masking_step = n_masking_steps; kv_tile_idx >= start_tile_idx; --masking_step, --kv_tile_idx) {
      Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
      PipelineState smem_pipe_read_kv_cur = smem_pipe_read_kv;
      ++smem_pipe_read_kv;
      // wait next kv
      consumer_wait(pipeline_kv, smem_pipe_read_kv);

      // gemm next qk
      gemm</*init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_kv.index()),
                                          tSrS);
      attention_updater.rescale_o(tOrO);
      // last pv gemm
      if (smem_pipe_read_kv_cur.index() == 0) {
        gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, smem_pipe_read_kv_cur.index() % 2),
                                             tOrV1(_, _, _, _0{}), tOrO);
      } else if (smem_pipe_read_kv_cur.index() == 1) {
        gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, smem_pipe_read_kv_cur.index() % 2),
                                             tOrV2(_, _, _, _0{}), tOrO);
      } else if (smem_pipe_read_kv_cur.index() == 2) {
        gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, smem_pipe_read_kv_cur.index() % 2),
                                             tOrV3(_, _, _, _0{}), tOrO);
      } else {
        gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, smem_pipe_read_kv_cur.index() % 2),
                                             tOrV4(_, _, _, _0{}), tOrO);
      }
      // wait cur qk gemm
      warpgroup_wait<1>();
      // mask p
      if (masking_step > 0) {
        Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_QKD{}));
        Tensor tScS = threadMmaQK.partition_C(cS);
#pragma unroll
        for (int i = 0; i < size(tSrS); ++i) {
          int qo_idx = get<0>(tScS(i)) / Ktraits::GROUP_SIZE;
          int kv_idx = get<1>(tScS(i)) + kv_tile_idx * BLOCK_SHAPE_KV;
          if constexpr (!CAUSAL) {  // Just masking based on col
            if (kv_idx >= kv_len) {
              tSrS(i) = AttentionUpdater::fill_value;
            }
          } else {
            if (kv_idx >= std::min(kv_len, col_limit_right(qo_idx))) {
              tSrS(i) = AttentionUpdater::fill_value;
            }
          }
        }
      }
      // update s (exp(s - m))
      Tensor scale_o = attention_updater.update</*init=*/false>(tSrS);
      Tensor tPrP = smem_thr_copy_P.retile_S(convert_type<DTypeKV>(tSrS));

      // gather qk gemm res
      cute::copy(smem_tiled_copy_P, tPrP, tPsP(_, _, _, smem_pipe_read_kv.index() % 2));
      cute::copy(scale_o, tScalesScale(_, smem_pipe_read_kv.index() % 2));
      // make sure tSrS r2s done
      cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWarpSchedulerWG1));
      // wait last pv gemm
      warpgroup_wait<0>();
      // release last kv
      pipeline_kv.consumer_release(smem_pipe_read_kv_cur);
    }
    // release q
    pipeline_q.consumer_release(smem_pipe_read_q);
    ++smem_pipe_read_q;
    // compute last pv
    attention_updater.rescale_o(tOrO);
    if (smem_pipe_read_kv.index() == 0) {
      gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, smem_pipe_read_kv.index() % 2),
                                           tOrV1(_, _, _, _0{}), tOrO);
    } else if (smem_pipe_read_kv.index() == 1) {
      gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, smem_pipe_read_kv.index() % 2),
                                           tOrV2(_, _, _, _0{}), tOrO);
    } else if (smem_pipe_read_kv.index() == 2) {
      gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, smem_pipe_read_kv.index() % 2),
                                           tOrV3(_, _, _, _0{}), tOrO);
    } else {
      gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, smem_pipe_read_kv.index() % 2),
                                           tOrV4(_, _, _, _0{}), tOrO);
    }
    scale_o = attention_updater.finalize(tSrS);
    warpgroup_wait<0>();
    // release last kv
    pipeline_kv.consumer_release(smem_pipe_read_kv);
    ++smem_pipe_read_kv;
    if (chunk_num_this_seq == 1) {
      // norm
      cute::copy(scale_o, tScalesScale(_, smem_pipe_read_kv.index() % 2));

      cutlass::arch::NamedBarrier::arrive(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWG1WG2LastSync));
      attention_updater.rescale_o(tOrO);
    }
    // WG1 write m,d back to gmem
    if (chunk_num_this_seq > 1 && thread_idx % 4 == 0) { // 16 rows per warp, eg. t0->row0 row8，t4->row1 row9
      const int warp_idx = thread_idx / 32;
#pragma unroll
      for (int w_i = 0; w_i < 2; ++w_i) {
        const int token_group_idx = warp_idx * 16 + thread_idx / 4 + 8 * w_i;
        const int token_idx = token_group_idx / Ktraits::GROUP_SIZE;

        if (token_idx < qo_len) {
          const int head_idx = token_group_idx % Ktraits::GROUP_SIZE;
          const int bid_offset = mainloop_params.max_draft_token_num * Ktraits::GROUP_SIZE;
          const int write_idx = bid * bid_offset + token_idx * Ktraits::GROUP_SIZE + head_idx;
          mM(write_idx) = static_cast<DTypeMD>(attention_updater.row_max(w_i));
          mD(write_idx) = static_cast<DTypeMD>(attention_updater.row_sum(w_i));
        }
      }
    }
  } else if (warp_group_idx == 2) {
    // consumer 1, compute pv
    Tensor scale_o = make_tensor<DTypeQKAccum>(Shape<_2>{});
    for (; kv_tile_idx >= start_tile_idx; --kv_tile_idx) {
      consumer_wait(pipeline_kv, smem_pipe_read_kv);
      cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWarpSchedulerWG1));
      // A: tPsP
      cute::copy(tScalesScale(_, smem_pipe_read_kv.index() % 2), scale_o);
      // rescale
      attention_updater.rescale_o(tOrO, scale_o);
      if (smem_pipe_read_kv.index() == 0) {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, smem_pipe_read_kv.index() % 2),
                                            tOrV1(_, _, _, _0{}), tOrO);
      } else if (smem_pipe_read_kv.index() == 1) {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, smem_pipe_read_kv.index() % 2),
                                            tOrV2(_, _, _, _0{}), tOrO);
      } else if (smem_pipe_read_kv.index() == 2) {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, smem_pipe_read_kv.index() % 2),
                                            tOrV3(_, _, _, _0{}), tOrO);
      } else {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, smem_pipe_read_kv.index() % 2),
                                            tOrV4(_, _, _, _0{}), tOrO);
      }
      pipeline_kv.consumer_release(smem_pipe_read_kv);
      ++smem_pipe_read_kv;
    }
    if (chunk_num_this_seq == 1) {
      // norm
      cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWG1WG2LastSync));
      cute::copy(tScalesScale(_, smem_pipe_read_kv.index() % 2), scale_o);
      attention_updater.rescale_o(tOrO, scale_o);
    }
  }
  return;
}

}  // namespace mla_attn

#endif  // ATTENTION_HOPPER_MAINLOOP_MMA_CUH_
