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

#include "helper.h"

__global__ void update_this_time(int* seq_lens_this_time,
                                 const int* seq_lens_encoder,
                                 const int* seq_lens_decoder,
                                 int real_bsz,
                                 int value
                                 ) {
    int linear_idx = threadIdx.x;
    // verify and set stop flags
    for (;linear_idx < real_bsz; linear_idx += blockDim.x) {
        if (seq_lens_encoder[linear_idx] == 0 && seq_lens_decoder[linear_idx] != 0) {
            seq_lens_this_time[linear_idx] = value;
        } else if (seq_lens_encoder[linear_idx] == 0 && seq_lens_decoder[linear_idx] == 0) {
            seq_lens_this_time[linear_idx] = 0;
        }
    }

}

void UpdateThisTime(const paddle::Tensor& seq_lens_this_time,
                   const paddle::Tensor& seq_lens_encoder,
                   const paddle::Tensor& seq_lens_decoder,
                   const int real_bsz,
                   const int value
                   ) {

  constexpr int BlockSize = 512;

  update_this_time<<<1, BlockSize, 0, seq_lens_this_time.stream()>>>(
    const_cast<int*>(seq_lens_this_time.data<int>()),
    seq_lens_encoder.data<int>(),
    seq_lens_decoder.data<int>(),
    real_bsz,
    value
  );
}

PD_BUILD_OP(speculate_update_seq_lens_this_time)
    .Inputs({"seq_lens_this_time", 
             "seq_lens_encoder",
             "seq_lens_decoder"})
    .Outputs({"seq_lens_this_time_out"})
    .Attrs({"real_bsz: int", "value: int"})
    .SetInplaceMap({{"seq_lens_this_time", "seq_lens_this_time_out"}})
    .SetKernelFn(PD_KERNEL(UpdateThisTime));