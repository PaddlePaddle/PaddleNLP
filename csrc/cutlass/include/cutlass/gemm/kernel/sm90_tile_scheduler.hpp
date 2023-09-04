/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/fast_math.h"
#include "cute/layout.hpp"

namespace cutlass::gemm::kernel::detail {

///////////////////////////////////////////////////////////////////////////////

// Persistent Thread Block (TB) scheduler
class PersistentTileSchedulerSm90 {
  //
  // Data members
  //

private:
  uint32_t blocks_per_problem_;
  uint32_t current_work_linear_idx_;
  uint32_t grid_blocks_total_;

  FastDivmod divmod_batch_;
  FastDivmod divmod_grid_y_;
  FastDivmod divmod_blk_m_;

  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;
    uint32_t is_valid_tile = false;
  };

  //
  // Methods
  //

public:

  template<class ProblemShapeMNKL, class TileShape, class ClusterShape>
  CUTLASS_DEVICE
  PersistentTileSchedulerSm90(ProblemShapeMNKL problem_shape_mnkl, TileShape tile_shape, ClusterShape cluster_shape) {
    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(is_static<TileShape>::value);
    static_assert(is_static<ClusterShape>::value);

    // Round up to nearest multiple of cluster dim along each mode
    auto [problem_blocks_m, problem_blocks_n, problem_blocks_l] = get_tiled_blk_shape_mnl(
        problem_shape_mnkl, tile_shape, cluster_shape);

    blocks_per_problem_ = problem_blocks_m * problem_blocks_n * problem_blocks_l;
    current_work_linear_idx_ = (int(blockIdx.x) * int(gridDim.y)) + int(blockIdx.y);
    grid_blocks_total_ = int(gridDim.x) * int(gridDim.y);

    // Pre-compute our fast div/mods for rasterization so we don't have to pay for DIVs
    divmod_batch_  = FastDivmod(problem_blocks_m * problem_blocks_n);
    divmod_grid_y_ = FastDivmod(size<1>(cluster_shape));
    divmod_blk_m_  = FastDivmod(problem_blocks_m);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    // Map worker's linear index into the CTA tiled problem shape to the corresponding MNL indices
    int work_idx_l, remainder;
    divmod_batch_(work_idx_l, remainder, current_work_linear_idx_);

    int blk_per_grid_dim, dontcare;
    divmod_grid_y_(blk_per_grid_dim, dontcare, remainder);

    int block_idx_m, block_idx_n;
    divmod_blk_m_(block_idx_n, block_idx_m, blk_per_grid_dim);
    int work_idx_m = block_idx_m;
    int work_idx_n = (block_idx_n * gridDim.y) + blockIdx.y;

    return {work_idx_m, work_idx_n, work_idx_l, current_work_linear_idx_ < blocks_per_problem_};
  }

  CUTLASS_DEVICE 
  void
  advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += grid_blocks_total_ * advance_count;
  }

  // Given the inputs, computes the total number of output blocks this problem will compute over
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE constexpr static
  dim3
  get_tiled_blk_shape_mnl(ProblemShapeMNKL problem_shape_mnkl, BlockShape blk_shape, ClusterShape cluster_shape) {
    // Across M and N is our Cluster tile, so we must round up the blocks to the nearest whole number of Cluster tiles
    auto blk_m = cute::size(cute::ceil_div(cute::shape<0>(problem_shape_mnkl), cute::shape<0>(blk_shape)));
    auto blk_n = cute::size(cute::ceil_div(cute::shape<1>(problem_shape_mnkl), cute::shape<1>(blk_shape)));

    // Round up to nearest multiple of cluster dim along each mode
    int problem_blocks_m = round_up(blk_m, cute::size<0>(cluster_shape));
    int problem_blocks_n = round_up(blk_n, cute::size<1>(cluster_shape));

    // Cluster tile does not span the batch mode, so no extra rounding up required for it
    int problem_blocks_l = int(cute::size<3>(problem_shape_mnkl));
    return {uint32_t(problem_blocks_m), uint32_t(problem_blocks_n), uint32_t(problem_blocks_l)};
  }
};

} // namespace cutlass::gemm::kernel::detail
