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

#include "helper.h"

__global__ void speculate_clear_accept_nums_kernel(
                                 int* accept_num,
                                 const int* seq_lens_decoder,
                                 const int max_bsz
                                 ) {
    const int bid = threadIdx.x;
    if (bid >= max_bsz) return;
    accept_num[bid] = seq_lens_decoder[bid] == 0 ? 0 : accept_num[bid];

}

void SpeculateClearAcceptNums(const paddle::Tensor& accept_num,
                   const paddle::Tensor& seq_lens_decoder
                   ) {
    // printf("enter clear \n");
    const int max_bsz = seq_lens_decoder.shape()[0];
    speculate_clear_accept_nums_kernel<<<1, 1024, 0, accept_num.stream()>>>(const_cast<int*>(accept_num.data<int>()),
                                                                            seq_lens_decoder.data<int>(), max_bsz);
}

PD_BUILD_OP(speculate_clear_accept_nums)
    .Inputs({"accept_num", 
             "seq_lens_decoder"})
    .Outputs({"seq_lens_decoder_out"})
    .SetInplaceMap({{"seq_lens_decoder", "seq_lens_decoder_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateClearAcceptNums));