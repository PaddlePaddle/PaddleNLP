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

/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */

#pragma once

#ifndef ATTENTION_HOPPER_EPILOGUE_CUH_
#define ATTENTION_HOPPER_EPILOGUE_CUH_

#include <cutlass/cutlass.h>

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "named_barrier.cuh"
#include "utils.cuh"

#ifdef DEBUG_MLA
#undef DEBUG_MLA
#endif
// #define DEBUG_MLA

namespace mla_attn {

using namespace cute;

template <typename Ktraits>
struct CollectiveEpilogue {
  using DTypeO = typename Ktraits::DTypeO;
  static constexpr int BLOCK_SHAPE_Q = Ktraits::BLOCK_SHAPE_Q;
  static constexpr int BLOCK_SHAPE_KV = Ktraits::BLOCK_SHAPE_KV;
  static constexpr int HEAD_DIM_VO = Ktraits::HEAD_DIM_VO;
  using TileShape_PDV = Shape<Int<BLOCK_SHAPE_Q>, Int<HEAD_DIM_VO>, Int<BLOCK_SHAPE_KV>>;

  static constexpr int NUM_WARPS = Ktraits::NUM_WARPS;
  static constexpr int NUM_THREADS = NUM_WARPS * cutlass::NumThreadsPerWarp;

  static constexpr int NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;

  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeO, decltype(cute::get<0>(TileShape_PDV{})),
                                   decltype(cute::get<1>(TileShape_PDV{}))>());
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_PDV{})));

  using SmemCopyAtomO = Copy_Atom<cute::SM90_U32x4_STSM_N, DTypeO>;
  using SharedStorage = cute::array_aligned<DTypeO, cute::cosize_v<SmemLayoutO>>;

  using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideT = cute::Shape<int32_t, _1, int32_t>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeTmpT = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideTmpT = cute::Shape<int32_t, _1, int32_t, int32_t>;
  using LayoutTmpT = cute::Layout<ShapeTmpT, StrideTmpT>;

  using ShapeNTMAT = cute::Shape<int32_t, int32_t>;
  using StrideNTMAT = cute::Shape<int32_t, _1>;
  using LayoutNTMAT = cute::Layout<ShapeNTMAT, StrideNTMAT>;

  using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;
  using TMA_O = decltype(make_tma_copy(
      GmemTiledCopyOTMA{},
      make_tensor(make_gmem_ptr(static_cast<DTypeO*>(nullptr)), ShapeT{}, StrideT{}), SmemLayoutO{},
      select<0, 1>(TileShape_PDV{}), _1{}));  // no mcast for O

  static constexpr int VEC_SIZE = cute::ceil_div(128, sizeof_bits_v<DTypeO>); // 8
  static_assert(HEAD_DIM_VO % VEC_SIZE == 0);
  static constexpr int NUM_THREADS_PER_ROW = HEAD_DIM_VO / VEC_SIZE; // 64
  static_assert(NUM_MMA_THREADS % NUM_THREADS_PER_ROW == 0);
  static constexpr int NUM_ROWS = NUM_MMA_THREADS / NUM_THREADS_PER_ROW;
  using TiledCopyOAtom = cute::Copy_Atom<cute::UniversalCopy<cutlass::uint128_t>, DTypeO>;
  using TiledCopyOThrLayout = decltype(cute::make_layout(
      cute::make_shape(Int<NUM_ROWS>{}, Int<NUM_THREADS_PER_ROW>{}), LayoutRight{}));
  using TiledCopyOValLayout =
      decltype(cute::make_layout(cute::make_shape(_1{}, Int<VEC_SIZE>{}), LayoutRight{}));
  using TiledCopyO =
      decltype(make_tiled_copy(TiledCopyOAtom{}, TiledCopyOThrLayout{},  // Thr layout
                               TiledCopyOValLayout{}                     // Val layout
                               ));
  struct Arguments {
    DTypeO* O_ptr;
    LayoutNTMAT const layout_O;
    DTypeO* O_ptr_tmp;
    LayoutNTMAT const layout_O_tmp;
  };

  // Device side kernel params
  struct Params {
    DTypeO* O_ptr;
    LayoutNTMAT const layout_O;
    DTypeO* O_ptr_tmp;
    LayoutNTMAT const layout_O_tmp;
  };

  static Params to_underlying_arguments_ntma(Arguments const& args) {
    return {args.O_ptr, args.layout_O, args.O_ptr_tmp, args.layout_O_tmp};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& epilogue_params) {}

  template <typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE,
            typename TiledMma>
  CUTLASS_DEVICE void store(Params const& epilogue_params, 
                            FrgTensorO const& tOrO,
                            FrgTensorLSE const& lse, 
                            SharedStorage& shared_storage,
                            TiledMma tiled_mma, 
                            const int thread_idx,
                            const int bid,
                            const int bsz,
                            const int seq_len_now,
                            const int start_token_idx,
                            const int tile_idx,
                            const int q_tile_idx,
                            const int kv_len,
                            const int chunk_size,
                            const int draft_total_token_num,
                            const int o_stride_head) {
    const int q_group_offset = q_tile_idx * BLOCK_SHAPE_Q;
    const int num_chunks = cute::ceil_div(kv_len, chunk_size);
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);

    Tensor tOrO_out = convert_type<DTypeO>(tOrO);
    Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);  // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
    // make sure gemm done
    cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS,
                                      /*id=*/static_cast<int>(NamedBarriers::kValueEmpty));
    // r2s
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    // make sure r2s done
    cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS,
                                      /*id=*/static_cast<int>(NamedBarriers::kValueEmpty));
    TiledCopyO gmem_tiled_copy_O;
    auto O_ptr = num_chunks == 1 ? epilogue_params.O_ptr + (start_token_idx * Ktraits::GROUP_SIZE + q_group_offset) * o_stride_head : epilogue_params.O_ptr_tmp + (((tile_idx * bsz + bid) * draft_total_token_num) * Ktraits::GROUP_SIZE + q_group_offset) * o_stride_head;
    Tensor mO = make_tensor(make_gmem_ptr(O_ptr), epilogue_params.layout_O);
    Tensor gO = local_tile(mO, select<0, 1>(TileShape_PDV{}), make_coord(_, _0{}))(_, _, _0{});
    Tensor cO = make_identity_tensor(gO.shape());  // (O, D) -> (o_idx, d_idx)
    ThrCopy thr_copy_O = gmem_tiled_copy_O.get_slice(thread_idx);
    Tensor tOgO = thr_copy_O.partition_D(gO);  // (CPY, CPY_O, CPY_D)
    Tensor tOsO = thr_copy_O.partition_S(sO);  // (CPY, CPY_O, CPY_D)
    Tensor tOcO = thr_copy_O.partition_D(cO);  // (CPY, CPY_O, CPY_D)
    Tensor tOgOGroup = flatten_1(tOgO);        // (CPY, (CPY_O, CPY_D))
    Tensor tOsOGroup = flatten_1(tOsO);        // (CPY, (CPY_O, CPY_D))
    Tensor tOcOGroup = flatten_1(tOcO);        // (CPY, (CPY_O, CPY_D))

    // copy if not out of bound
    auto predicate_fn = [&](auto coords) {
      auto s_coords = tOcOGroup(_0{}, coords);
      return elem_less((get<0>(s_coords) + q_group_offset) / Ktraits::GROUP_SIZE, seq_len_now);
    };
    copy_if(gmem_tiled_copy_O, predicate_fn, tOsOGroup, tOgOGroup);
  }
};

}  // namespace mla_attn

#endif  // ATTENTION_HOPPER_EPILOGUE_CUH_
