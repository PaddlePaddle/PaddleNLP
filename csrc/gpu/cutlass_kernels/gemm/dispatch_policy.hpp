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

#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm {

//////////////////////////////////////////////////////////////////////////////

// FP8 related policies (including Blocked Scaled Accumulation)
//  `ScaleGranularityM` specifies scaling granularity along M, while zero-value
//  `ScaleGranularityM` indicates that scaling granularity is
//  `size<0>(TileShape_MNK{})` along M.
template <int ScaleGranularityM = 0>
struct KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum
    : KernelTmaWarpSpecializedCooperative {};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp
// specialized dynamic schedule For FP8 kernels with Block Scaling
template <int Stages_, class ClusterShape_ = Shape<_1, _1, _1>,
          class KernelSchedule = KernelTmaWarpSpecialized,
          int ScaleGranularityM =
              0  // `ScaleGranularityM` specifies scaling granularity along M,
                 // while zero-value `ScaleGranularityM` indicates that scaling
                 // granularity is `size<0>(TileShape_MNK{})` along M.
          >
struct MainloopSm90TmaGmmaWarpSpecializedBlockScalingSubGroupMFP8
    : MainloopSm90TmaGmmaWarpSpecialized<Stages_, ClusterShape_,
                                         KernelSchedule> {
  static_assert(
      cute::is_same_v<
          KernelSchedule,
          KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum<
              ScaleGranularityM>>,
      "KernelSchedule must be one of the warp specialized policies");
};

//////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm