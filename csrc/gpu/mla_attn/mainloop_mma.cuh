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

#pragma once

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
                            const int q_tile_idx,
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

  const int q_group_offset = q_tile_idx * BLOCK_SHAPE_Q;

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutK{});
  // todo: split rope/norope for sVt, not use s1~s2.
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
          int qo_idx = (get<0>(tScS(i)) + q_group_offset) / Ktraits::GROUP_SIZE;
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
      // r2s fence wgmma
      cutlass::arch::fence_view_async_shared();
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
        const int token_group_idx = warp_idx * 16 + (thread_idx % 32) / 4 + 8 * w_i + q_group_offset;
        const int token_idx = token_group_idx / Ktraits::GROUP_SIZE;

        if (token_idx < qo_len) {
          // const int head_idx = token_group_idx % Ktraits::GROUP_SIZE;
          const int bid_offset = mainloop_params.draft_total_token_num * Ktraits::GROUP_SIZE;
          const int write_idx = bid * bid_offset + token_group_idx;
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
                                       const int q_tile_idx,
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

  const int q_group_offset = q_tile_idx * BLOCK_SHAPE_Q;

  cutlass::FastDivmod stage_div(2);
  int quotient, remainder;

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutK{});
  // TODO: split rope/norope for sVt, not use s1~s4.
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
        int qo_idx = (get<0>(tScS(i)) + q_group_offset) / Ktraits::GROUP_SIZE;
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
    stage_div.fast_divmod(quotient, remainder, smem_pipe_read_kv.index());
    cute::copy(smem_tiled_copy_P, tPrP, tPsP(_, _, _, remainder));
    cute::copy(scale_o, tScalesScale(_, remainder));
    // r2s fence wgmma
    cutlass::arch::fence_view_async_shared();
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
      stage_div.fast_divmod(quotient, remainder, smem_pipe_read_kv_cur.index());
      // last pv gemm
      if (smem_pipe_read_kv_cur.index() == 0) {
        gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, remainder),
                                             tOrV1(_, _, _, _0{}), tOrO);
      } else if (smem_pipe_read_kv_cur.index() == 1) {
        gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, remainder),
                                             tOrV2(_, _, _, _0{}), tOrO);
      } else if (smem_pipe_read_kv_cur.index() == 2) {
        gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, remainder),
                                             tOrV3(_, _, _, _0{}), tOrO);
      } else {
        gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, remainder),
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
          int qo_idx = (get<0>(tScS(i)) + q_group_offset) / Ktraits::GROUP_SIZE;
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
      stage_div.fast_divmod(quotient, remainder, smem_pipe_read_kv.index());
      cute::copy(smem_tiled_copy_P, tPrP, tPsP(_, _, _, remainder));
      cute::copy(scale_o, tScalesScale(_, remainder));
      // r2s fence wgmma
      cutlass::arch::fence_view_async_shared();
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
    stage_div.fast_divmod(quotient, remainder, smem_pipe_read_kv.index());
    if (smem_pipe_read_kv.index() == 0) {
      gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, remainder),
                                           tOrV1(_, _, _, _0{}), tOrO);
    } else if (smem_pipe_read_kv.index() == 1) {
      gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, remainder),
                                           tOrV2(_, _, _, _0{}), tOrO);
    } else if (smem_pipe_read_kv.index() == 2) {
      gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, remainder),
                                           tOrV3(_, _, _, _0{}), tOrO);
    } else {
      gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, remainder),
                                           tOrV4(_, _, _, _0{}), tOrO);
    }
    scale_o = attention_updater.finalize(tSrS);
    warpgroup_wait<0>();
    // release last kv
    pipeline_kv.consumer_release(smem_pipe_read_kv);
    ++smem_pipe_read_kv;
    if (chunk_num_this_seq == 1) {
      // norm
      stage_div.fast_divmod(quotient, remainder, smem_pipe_read_kv.index());
      cute::copy(scale_o, tScalesScale(_, remainder));
      cutlass::arch::NamedBarrier::arrive(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWG1WG2LastSync));
      attention_updater.rescale_o(tOrO);
    }
    // WG1 write m,d back to gmem
    if (chunk_num_this_seq > 1 && thread_idx % 4 == 0) { // 16 rows per warp, eg. t0->row0 row8，t4->row1 row9
      const int warp_idx = thread_idx / 32;
#pragma unroll
      for (int w_i = 0; w_i < 2; ++w_i) {
        const int token_group_idx = warp_idx * 16 + (thread_idx % 32) / 4 + 8 * w_i + q_group_offset;
        const int token_idx = token_group_idx / Ktraits::GROUP_SIZE;

        if (token_idx < qo_len) {
          // const int head_idx = token_group_idx % Ktraits::GROUP_SIZE;
          const int bid_offset = mainloop_params.draft_total_token_num * Ktraits::GROUP_SIZE;
          const int write_idx = bid * bid_offset + token_group_idx;
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
      stage_div.fast_divmod(quotient, remainder, smem_pipe_read_kv.index());
      cute::copy(tScalesScale(_, remainder), scale_o);
      // rescale
      attention_updater.rescale_o(tOrO, scale_o);
      if (smem_pipe_read_kv.index() == 0) {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, remainder),
                                            tOrV1(_, _, _, _0{}), tOrO);
      } else if (smem_pipe_read_kv.index() == 1) {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, remainder),
                                            tOrV2(_, _, _, _0{}), tOrO);
      } else if (smem_pipe_read_kv.index() == 2) {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, remainder),
                                            tOrV3(_, _, _, _0{}), tOrO);
      } else {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_CS2(_, _, _, remainder),
                                            tOrV4(_, _, _, _0{}), tOrO);
      }
      pipeline_kv.consumer_release(smem_pipe_read_kv);
      ++smem_pipe_read_kv;
    }
    if (chunk_num_this_seq == 1) {
      // norm
      stage_div.fast_divmod(quotient, remainder, smem_pipe_read_kv.index());
      cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWG1WG2LastSync));
      cute::copy(tScalesScale(_, remainder), scale_o);
      attention_updater.rescale_o(tOrO, scale_o);
    }
  }
  return;
}

template <typename Ktraits, bool CAUSAL, typename Params, typename MainloopPipeline, typename MainloopPipelineQ, typename MainloopPipelineQK,
          typename PipelineState, typename PipelineStateQ, typename PipelineStateQK, typename SharedStorage, typename AttentionUpdater>
CUTLASS_DEVICE void mma_qk_one_stages(const Params& mainloop_params,
                                      MainloopPipelineQ pipeline_q,
                                      PipelineStateQ& smem_pipe_read_q,
                                      MainloopPipelineQK pipeline_qk,
                                      PipelineStateQK& smem_pipe_write_qk,
                                      MainloopPipeline pipeline_kv,
                                      PipelineState& smem_pipe_read_kv,
                                      AttentionUpdater& attention_updater,
                                      const int& thread_idx, 
                                      const int& bid,
                                      const int kv_len,
                                      const int qo_len,
                                      const int tile_idx,
                                      const int q_tile_idx,
                                      SharedStorage& shared_storage) {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeMD = typename Ktraits::DTypeO; 
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutPSS = typename Ktraits::SmemLayoutPSS;
  using SmemLayoutRow = typename Ktraits::SmemLayoutRow;
  using SmemCopyAtom = typename Ktraits::SmemCopyAtom;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;

  static constexpr int BLOCK_SHAPE_Q = get<0>(TileShape_QKD{});
  static constexpr int BLOCK_SHAPE_KV = get<1>(TileShape_QKD{});

  const int q_group_offset = q_tile_idx * BLOCK_SHAPE_Q;

  const int chunk_num_this_seq = cute::ceil_div(kv_len, mainloop_params.chunk_size);

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutK{});

  Tensor sPSS = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutPSS{});
  Tensor s_scale = make_tensor(make_smem_ptr(shared_storage.smem_scale.data()), SmemLayoutRow{});

  typename Ktraits::TiledMmaQK tiled_mma_qk;
  auto threadMmaQK = tiled_mma_qk.get_thread_slice(thread_idx);
  Tensor tSrQ = threadMmaQK.partition_fragment_A(sQ);
  Tensor tSrK = threadMmaQK.partition_fragment_B(sK);
  auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtom{}, tiled_mma_qk);
  auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);
  Tensor tPsP = smem_thr_copy_P.partition_D(sPSS);
  Tensor tScalesScale = s_scale(_, thread_idx, _);
  Tensor mM = make_tensor(make_gmem_ptr(mainloop_params.m_ptr), mainloop_params.layout_MD)(tile_idx, _); // (bsz * draft_token_num * num_head)
  Tensor mD = make_tensor(make_gmem_ptr(mainloop_params.d_ptr), mainloop_params.layout_MD)(tile_idx, _);

  const int start_len = tile_idx * mainloop_params.chunk_size;
  const int start_tile_idx = start_len / BLOCK_SHAPE_KV;
  const int end_tile_idx = cute::ceil_div(min(start_len + mainloop_params.chunk_size, kv_len), BLOCK_SHAPE_KV) - 1;
  int kv_tile_idx = end_tile_idx;

  auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
  };
  Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));

  // DO QK gemm
  auto col_limit_right = [&](int qo_idx) { return qo_idx + 1 + kv_len - qo_len; };
  // wait q
  consumer_wait(pipeline_q, smem_pipe_read_q);
  constexpr int n_masking_steps = CAUSAL ? cute::ceil_div(BLOCK_SHAPE_Q, BLOCK_SHAPE_KV) : 0;
  int masking_step = n_masking_steps;
  for (; kv_tile_idx >= start_tile_idx; --masking_step, --kv_tile_idx) {
    // Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
    consumer_wait(pipeline_kv, smem_pipe_read_kv);
    // gemm next qk
    gemm</*init=*/true, /*wg_wait=*/0>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_kv.index()),
                                        tSrS);
    // release last kv
    pipeline_kv.consumer_release(smem_pipe_read_kv);
    ++smem_pipe_read_kv;
    // mask
    if (masking_step > 0) {
      Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_QKD{}));
      Tensor tScS = threadMmaQK.partition_C(cS);
#pragma unroll
      for (int i = 0; i < size(tSrS); ++i) {
        int qo_idx = (get<0>(tScS(i)) + q_group_offset) / Ktraits::GROUP_SIZE;
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
    Tensor scale_o = (kv_tile_idx == end_tile_idx ? attention_updater.update<true>(tSrS) : attention_updater.update<false>(tSrS));
    Tensor tPrP = smem_thr_copy_P.retile_S(convert_type<DTypeKV>(tSrS));

    // r2s
    pipeline_qk.producer_acquire(smem_pipe_write_qk);
    cute::copy(smem_tiled_copy_P, tPrP, tPsP(_, _, _, smem_pipe_write_qk.index()));
    cute::copy(scale_o, tScalesScale(_, smem_pipe_write_qk.index()));
    // r2s fence wgmma
    cutlass::arch::fence_view_async_shared();
    pipeline_qk.producer_commit(smem_pipe_write_qk);
    ++smem_pipe_write_qk;
  }
  // wg1 release q
  pipeline_q.consumer_release(smem_pipe_read_q);
  ++smem_pipe_read_q;

  // update last softmax scales
  Tensor scale_o = attention_updater.finalize(tSrS);
  if (chunk_num_this_seq == 1) {
    // norm
    pipeline_qk.producer_acquire(smem_pipe_write_qk);
    cute::copy(scale_o, tScalesScale(_, smem_pipe_write_qk.index()));
    pipeline_qk.producer_commit(smem_pipe_write_qk);
    ++smem_pipe_write_qk;
  }

  // WG1 write m,d back to gmem
  if (chunk_num_this_seq > 1 && thread_idx % 4 == 0) { // 16 rows per warp, eg. t0->row0 row8，t4->row1 row9
    const int warp_idx = thread_idx / 32;
#pragma unroll
    for (int w_i = 0; w_i < 2; ++w_i) {
      const int token_group_idx = warp_idx * 16 + (thread_idx % 32) / 4 + 8 * w_i + q_group_offset;
      const int token_idx = token_group_idx / Ktraits::GROUP_SIZE;
      if (token_idx < qo_len) {
        // const int head_idx = token_group_idx % Ktraits::GROUP_SIZE;
        const int bid_offset = mainloop_params.draft_total_token_num * Ktraits::GROUP_SIZE;
        const int write_idx = bid * bid_offset + token_group_idx;
        mM(write_idx) = static_cast<DTypeMD>(attention_updater.row_max(w_i));
        mD(write_idx) = static_cast<DTypeMD>(attention_updater.row_sum(w_i));
      }
    }
  }
  return;
}

template <typename Ktraits, bool CAUSAL, typename Params, typename MainloopPipeline, typename MainloopPipelineQ, typename MainloopPipelineQK,
          typename PipelineState, typename PipelineStateQ, typename PipelineStateQK, typename SharedStorage, typename AttentionUpdater>
CUTLASS_DEVICE void mma_qk_two_stages(const Params& mainloop_params,
                                      MainloopPipelineQ pipeline_q,
                                      PipelineStateQ& smem_pipe_read_q,
                                      MainloopPipelineQK pipeline_qk,
                                      PipelineStateQK& smem_pipe_write_qk,
                                      MainloopPipeline pipeline_kv,
                                      PipelineState& smem_pipe_read_kv,
                                      AttentionUpdater& attention_updater,
                                      const int& thread_idx, 
                                      const int& bid,
                                      const int kv_len,
                                      const int qo_len,
                                      const int tile_idx,
                                      const int q_tile_idx,
                                      SharedStorage& shared_storage) {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeMD = typename Ktraits::DTypeO; 
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutPSS = typename Ktraits::SmemLayoutPSS;
  using SmemLayoutRow = typename Ktraits::SmemLayoutRow;
  using SmemCopyAtom = typename Ktraits::SmemCopyAtom;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;

  static constexpr int BLOCK_SHAPE_Q = get<0>(TileShape_QKD{});
  static constexpr int BLOCK_SHAPE_KV = get<1>(TileShape_QKD{});

  const int q_group_offset = q_tile_idx * BLOCK_SHAPE_Q;

  const int chunk_num_this_seq = cute::ceil_div(kv_len, mainloop_params.chunk_size);

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutK{});

  Tensor sPSS = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutPSS{});
  Tensor s_scale = make_tensor(make_smem_ptr(shared_storage.smem_scale.data()), SmemLayoutRow{});

  typename Ktraits::TiledMmaQK tiled_mma_qk;
  auto threadMmaQK = tiled_mma_qk.get_thread_slice(thread_idx);
  Tensor tSrQ = threadMmaQK.partition_fragment_A(sQ);
  Tensor tSrK = threadMmaQK.partition_fragment_B(sK);
  auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtom{}, tiled_mma_qk);
  auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);
  Tensor tPsP = smem_thr_copy_P.partition_D(sPSS);
  Tensor tScalesScale = s_scale(_, thread_idx, _);
  Tensor mM = make_tensor(make_gmem_ptr(mainloop_params.m_ptr), mainloop_params.layout_MD)(tile_idx, _); // (bsz * draft_token_num * num_head)
  Tensor mD = make_tensor(make_gmem_ptr(mainloop_params.d_ptr), mainloop_params.layout_MD)(tile_idx, _);

  const int start_len = tile_idx * mainloop_params.chunk_size;
  const int start_tile_idx = start_len / BLOCK_SHAPE_KV;
  int kv_tile_idx = cute::ceil_div(min(start_len + mainloop_params.chunk_size, kv_len), BLOCK_SHAPE_KV) - 1;

  auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
  };

  Tensor tSrS1 = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
  Tensor tSrS2 = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));

  // DO QK gemm
  auto col_limit_right = [&](int qo_idx) { return qo_idx + 1 + kv_len - qo_len; };
  // wait q
  consumer_wait(pipeline_q, smem_pipe_read_q);
  // wait k
  consumer_wait(pipeline_kv, smem_pipe_read_kv);
  // first qk gemm
  if (smem_pipe_write_qk.index() == 0) {
    gemm</*init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_kv.index()),
                                        tSrS1);
  } else {
    gemm</*init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_kv.index()),
                                        tSrS2);
  }
  constexpr int n_masking_steps = CAUSAL ? cute::ceil_div(BLOCK_SHAPE_Q, BLOCK_SHAPE_KV) : 0;
  int masking_step = n_masking_steps;
  --kv_tile_idx;
  for (; kv_tile_idx >= start_tile_idx; --masking_step, --kv_tile_idx) {
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
    PipelineState smem_pipe_read_kv_cur = smem_pipe_read_kv;
    ++smem_pipe_read_kv;
    // wait next kv
    consumer_wait(pipeline_kv, smem_pipe_read_kv);
    // gemm next qk
    if (smem_pipe_write_qk.index() == 1) {
      gemm</*init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_kv.index()),
                                          tSrS1);
    } else {
      gemm</*init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_kv.index()),
                                          tSrS2);
    }
    // wait last qk done
    warpgroup_wait<1>();
    // release last kv
    pipeline_kv.consumer_release(smem_pipe_read_kv_cur);
    // mask now
    if (masking_step > 0) {
      Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_QKD{}));
      Tensor tScS = threadMmaQK.partition_C(cS);
#pragma unroll
      for (int i = 0; i < size(tSrS1); ++i) {
        int qo_idx = (get<0>(tScS(i)) + q_group_offset) / Ktraits::GROUP_SIZE;
        int kv_idx = get<1>(tScS(i)) + kv_tile_idx * BLOCK_SHAPE_KV;
        if (kv_idx >= std::min(kv_len, col_limit_right(qo_idx))) {
          if (smem_pipe_write_qk.index() == 0) {
            tSrS1(i) = AttentionUpdater::fill_value;
          } else {
            tSrS2(i) = AttentionUpdater::fill_value;
          }
        }
      }
    }
    // update s (exp(s - m))
    if (smem_pipe_write_qk.index() == 0) {
      Tensor scale_o = attention_updater.update</*init=*/false>(tSrS1);
      Tensor tPrP = smem_thr_copy_P.retile_S(convert_type<DTypeKV>(tSrS1));
      // r2s
      pipeline_qk.producer_acquire(smem_pipe_write_qk);
      cute::copy(smem_tiled_copy_P, tPrP, tPsP(_, _, _, smem_pipe_write_qk.index()));
      cute::copy(scale_o, tScalesScale(_, smem_pipe_write_qk.index()));
    } else {
      Tensor scale_o = attention_updater.update</*init=*/false>(tSrS2);
      Tensor tPrP = smem_thr_copy_P.retile_S(convert_type<DTypeKV>(tSrS2));
      // r2s
      pipeline_qk.producer_acquire(smem_pipe_write_qk);
      cute::copy(smem_tiled_copy_P, tPrP, tPsP(_, _, _, smem_pipe_write_qk.index()));
      cute::copy(scale_o, tScalesScale(_, smem_pipe_write_qk.index()));
    }
    // r2s fence wgmma
    cutlass::arch::fence_view_async_shared();
    pipeline_qk.producer_commit(smem_pipe_write_qk);
    ++smem_pipe_write_qk;
  }
  // wg1 release q
  pipeline_q.consumer_release(smem_pipe_read_q);
  ++smem_pipe_read_q;
  // wait last qk
  warpgroup_wait<0>();
  // wg1 release kv
  pipeline_kv.consumer_release(smem_pipe_read_kv);
  ++smem_pipe_read_kv;
  // update s (exp(s - m)) last
  if (smem_pipe_write_qk.index() == 0) {
    Tensor scale_o = attention_updater.update</*init=*/false>(tSrS1);
    Tensor tPrP = smem_thr_copy_P.retile_S(convert_type<DTypeKV>(tSrS1));
    // r2s
    pipeline_qk.producer_acquire(smem_pipe_write_qk);
    cute::copy(smem_tiled_copy_P, tPrP, tPsP(_, _, _, smem_pipe_write_qk.index()));
    cute::copy(scale_o, tScalesScale(_, smem_pipe_write_qk.index()));
  } else {
    Tensor scale_o = attention_updater.update</*init=*/false>(tSrS2);
    Tensor tPrP = smem_thr_copy_P.retile_S(convert_type<DTypeKV>(tSrS2));
    // r2s
    pipeline_qk.producer_acquire(smem_pipe_write_qk);
    cute::copy(smem_tiled_copy_P, tPrP, tPsP(_, _, _, smem_pipe_write_qk.index()));
    cute::copy(scale_o, tScalesScale(_, smem_pipe_write_qk.index()));
  }
  // r2s fence wgmma
  cutlass::arch::fence_view_async_shared();
  pipeline_qk.producer_commit(smem_pipe_write_qk);
  ++smem_pipe_write_qk;

  // update last softmax scales
  Tensor scale_o = attention_updater.finalize(tSrS1);
  if (chunk_num_this_seq == 1) {
    // norm
    pipeline_qk.producer_acquire(smem_pipe_write_qk);
    cute::copy(scale_o, tScalesScale(_, smem_pipe_write_qk.index()));
    pipeline_qk.producer_commit(smem_pipe_write_qk);
    ++smem_pipe_write_qk;
  }

  // WG1 write m,d back to gmem
  if (chunk_num_this_seq > 1 && thread_idx % 4 == 0) { // 16 rows per warp, eg. t0->row0 row8，t4->row1 row9
    const int warp_idx = thread_idx / 32;
#pragma unroll
    for (int w_i = 0; w_i < 2; ++w_i) {
      const int token_group_idx = warp_idx * 16 + (thread_idx % 32) / 4 + 8 * w_i + q_group_offset;
      const int token_idx = token_group_idx / Ktraits::GROUP_SIZE;
      if (token_idx < qo_len) {
        const int bid_offset = mainloop_params.draft_total_token_num * Ktraits::GROUP_SIZE;
        const int write_idx = bid * bid_offset + token_group_idx;
        mM(write_idx) = static_cast<DTypeMD>(attention_updater.row_max(w_i));
        mD(write_idx) = static_cast<DTypeMD>(attention_updater.row_sum(w_i));
      }
    }
  }
  return;
}

template <typename Ktraits, bool CAUSAL, typename Params, typename MainloopPipeline, typename MainloopPipelineQK,
          typename PipelineState, typename PipelineStateQK, typename SharedStorage, typename FrgTensorO, typename AttentionUpdater>
CUTLASS_DEVICE void mma_pv_one_stages(const Params& mainloop_params,
                                      MainloopPipelineQK pipeline_qk,
                                      PipelineStateQK& smem_pipe_read_qk,
                                      MainloopPipeline pipeline_kv,
                                      PipelineState& smem_pipe_read_kv,
                                      AttentionUpdater& attention_updater,
                                      FrgTensorO& tOrO, 
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
  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutPSS = typename Ktraits::SmemLayoutPSS;
  using SmemLayoutRow = typename Ktraits::SmemLayoutRow;
  using SmemCopyAtom = typename Ktraits::SmemCopyAtom;
  using SmemLayoutVtOneStage = typename Ktraits::SmemLayoutVtOneStage;
  static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

  static constexpr int BLOCK_SHAPE_Q = get<0>(TileShape_QKD{});
  static constexpr int BLOCK_SHAPE_KV = get<1>(TileShape_QKD{});

  const int chunk_num_this_seq = cute::ceil_div(kv_len, mainloop_params.chunk_size);
  // todo: split rope/norope for sVt, not use s1~s4.
  Tensor sVt_s1 = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutVtOneStage{});
  Tensor sVt_s2 = make_tensor(make_smem_ptr(shared_storage.smem_kv.data() + Ktraits::NUM_PER_STAGE), SmemLayoutVtOneStage{});
  Tensor sVt_s3 = make_tensor(make_smem_ptr(shared_storage.smem_kv.data() + 2 * Ktraits::NUM_PER_STAGE), SmemLayoutVtOneStage{});
  Tensor sVt_s4 = make_tensor(make_smem_ptr(shared_storage.smem_kv.data() + 3 * Ktraits::NUM_PER_STAGE), SmemLayoutVtOneStage{});

  Tensor sPSS = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutPSS{});
  Tensor s_scale = make_tensor(make_smem_ptr(shared_storage.smem_scale.data()), SmemLayoutRow{});
  Tensor tScalesScale = s_scale(_, thread_idx % cutlass::NumThreadsPerWarpGroup, _);

  typename Ktraits::TiledMmaPVSS tiled_mma_pv_ss;
  auto threadMmaPVSS = tiled_mma_pv_ss.get_thread_slice(thread_idx);
  Tensor tOrV1 = threadMmaPVSS.partition_fragment_B(sVt_s1);
  Tensor tOrV2 = threadMmaPVSS.partition_fragment_B(sVt_s2);
  Tensor tOrV3 = threadMmaPVSS.partition_fragment_B(sVt_s3);
  Tensor tOrV4 = threadMmaPVSS.partition_fragment_B(sVt_s4);
  Tensor tOrP_S = threadMmaPVSS.partition_fragment_A(sPSS);
  const int start_len = tile_idx * mainloop_params.chunk_size;
  const int start_tile_idx = start_len / BLOCK_SHAPE_KV;
  const int end_tile_idx = cute::ceil_div(min(start_len + mainloop_params.chunk_size, kv_len), BLOCK_SHAPE_KV) - 1;
  int kv_tile_idx = end_tile_idx;

  auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
  };

  Tensor scale_o = make_tensor<DTypeQKAccum>(Shape<_2>{});
  for (; kv_tile_idx >= start_tile_idx; --kv_tile_idx) {
    // wait v
    consumer_wait(pipeline_kv, smem_pipe_read_kv);
    // wait p
    consumer_wait(pipeline_qk, smem_pipe_read_qk);
    // A: tPsP
    cute::copy(tScalesScale(_, smem_pipe_read_qk.index()), scale_o);
    // rescale
    attention_updater.rescale_o(tOrO, scale_o);
    // gemm pv
    if (smem_pipe_read_kv.index() == 0) {
      gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_S(_, _, _, smem_pipe_read_qk.index()),
                                          tOrV1(_, _, _, _0{}), tOrO);
    } else if (smem_pipe_read_kv.index() == 1) {
      gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_S(_, _, _, smem_pipe_read_qk.index()),
                                          tOrV2(_, _, _, _0{}), tOrO);
    } else if (smem_pipe_read_kv.index() == 2) {
      gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_S(_, _, _, smem_pipe_read_qk.index()),
                                          tOrV3(_, _, _, _0{}), tOrO);
    } else if (smem_pipe_read_kv.index() == 3) {
      gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv_ss, tOrP_S(_, _, _, smem_pipe_read_qk.index()),
                                          tOrV4(_, _, _, _0{}), tOrO);
    }
    pipeline_qk.consumer_release(smem_pipe_read_qk);
    ++smem_pipe_read_qk;
    pipeline_kv.consumer_release(smem_pipe_read_kv);
    ++smem_pipe_read_kv;
  }
  if (chunk_num_this_seq == 1) {
    // norm
    consumer_wait(pipeline_qk, smem_pipe_read_qk);
    cute::copy(tScalesScale(_, smem_pipe_read_qk.index()), scale_o);
    pipeline_qk.consumer_release(smem_pipe_read_qk);
    ++smem_pipe_read_qk;
    attention_updater.rescale_o(tOrO, scale_o);
  }
  return;
}

}  // namespace mla_attn

#endif  // ATTENTION_HOPPER_MAINLOOP_MMA_CUH_
