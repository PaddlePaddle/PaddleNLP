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

#ifndef ATTENTION_HOPPER_NAMED_BARRIERS_CUH_
#define ATTENTION_HOPPER_NAMED_BARRIERS_CUH_

#include <cuda_runtime.h>

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"

namespace mla_attn {

enum class NamedBarriers {
  kQueryEmpty = 0,
  kValueEmpty = 1,
  kWarpSchedulerWG1 = 2,
  kWarpSchedulerWG2 = 3,
  kWarpSchedulerWG3 = 4,
  kPrefetchIndices = 5,
  kOdone = 6,
  kWG1WG2Sync = 7,
  kWG0WG1WG2Sync = 8,
  kWG1WG2LastSync = 9,
  kAllWGSync = 10,
};

}  // namespace mla_attn

#endif  // ATTENTION_HOPPER_NAMED_BARRIERS_CUH_
