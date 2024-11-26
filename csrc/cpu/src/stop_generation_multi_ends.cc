// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdlib.h>
#include <string.h>

#include "helper.h"
#include <stdio.h>

void set_value_by_flags(const bool* stop_flags,
                        const int64_t* end_ids,
                        int64_t* topk_ids,
                        bool* stop_flags_out,
                        const int bs,
                        int end_length) {
  for (int bi = 0; bi < bs; bi++) {
    topk_ids[bi] = stop_flags[bi] ? end_ids[0] : topk_ids[bi];
    if (is_in_end(topk_ids[bi], end_ids, end_length)) {
      stop_flags_out[bi] = true;
    }
  }
}


std::vector<paddle::Tensor> GetStopFlagsMulti(const paddle::Tensor& topk_ids,
                                              const paddle::Tensor& stop_flags,
                                              const paddle::Tensor& end_ids) {
  PD_CHECK(topk_ids.dtype() == paddle::DataType::INT64);
  PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);

  std::vector<int64_t> shape = topk_ids.shape();
  int64_t bs_now = shape[0];
  int64_t end_length = end_ids.shape()[0];
  auto topk_ids_out = topk_ids.copy_to(topk_ids.place(), false);
  auto stop_flags_out = stop_flags.copy_to(stop_flags.place(), false);
  set_value_by_flags(stop_flags.data<bool>(),
                     end_ids.data<int64_t>(),
                     topk_ids_out.data<int64_t>(),
                     stop_flags_out.data<bool>(),
                     bs_now,
                     end_length);

  return {topk_ids_out, stop_flags_out};
}

std::vector<std::vector<int64_t>> GetStopFlagsMultiInferShape(
    const std::vector<int64_t>& topk_ids_shape,
    const std::vector<int64_t>& stop_flags_shape,
    const std::vector<int64_t>& end_ids_shape) {
  return {topk_ids_shape, stop_flags_shape};
}

std::vector<paddle::DataType> GetStopFlagsMultiInferDtype(
    const paddle::DataType& topk_ids_dtype,
    const paddle::DataType& stop_flags_dtype,
    const paddle::DataType& end_ids_dtype) {
  return {topk_ids_dtype, stop_flags_dtype};
}

PD_BUILD_OP(set_stop_value_multi_ends)
    .Inputs({"topk_ids", "stop_flags", "end_ids"})
    .Outputs({"topk_ids_out", "stop_flags_out"})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMulti))
    .SetInferShapeFn(PD_INFER_SHAPE(GetStopFlagsMultiInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetStopFlagsMultiInferDtype));