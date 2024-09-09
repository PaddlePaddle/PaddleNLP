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

#include "paddle/extension.h"

__global__ void SpeculateGetSeqLensOutputKernel(
                              int* seq_lens_output,
                              const int *seq_lens_this_time,
                              const int *seq_lens_encoder,
                              const int *seq_lens_decoder,
                              const int real_bsz) {
    for (int bid = threadIdx.x; bid < real_bsz; bid += blockDim.x) {
        if (seq_lens_this_time[bid] == 0) {
            continue;
        } else if (seq_lens_this_time[bid] == 1) {
            seq_lens_output[bid] = 1;
        } else if (seq_lens_encoder[bid] != 0) {
            seq_lens_output[bid] = 1;
        } else {
            seq_lens_output[bid] = seq_lens_this_time[bid];
        }
    }
}

std::vector<paddle::Tensor> SpeculateGetSeqLensOutput(const paddle::Tensor& seq_lens_this_time,
                                             const paddle::Tensor& seq_lens_encoder,
                                             const paddle::Tensor& seq_lens_decoder) {
    auto cu_stream = seq_lens_this_time.stream();
    std::vector<int64_t> seq_lens_this_time_shape = seq_lens_this_time.shape();
    const int bsz = seq_lens_this_time_shape[0];

    auto seq_lens_output = paddle::full({bsz}, 0, paddle::DataType::INT32, seq_lens_this_time.place());

    SpeculateGetSeqLensOutputKernel<<<1, 256, 0, cu_stream>>>(seq_lens_output.data<int>(),
                                                              seq_lens_this_time.data<int>(),
                                                              seq_lens_encoder.data<int>(),
                                                              seq_lens_decoder.data<int>(),
                                                              bsz);
    
    return {seq_lens_output};
}

std::vector<std::vector<int64_t>> SpeculateGetSeqLensOutputInferShape(const std::vector<int64_t>& seq_lens_this_time_shape,
                                                             const std::vector<int64_t>& seq_lens_encoder_shape,
                                                             const std::vector<int64_t>& seq_lens_decoder_shape) {
    int64_t bsz = seq_lens_this_time_shape[0];
    return {{bsz}};
}

std::vector<paddle::DataType> SpeculateGetSeqLensOutputInferDtype(const paddle::DataType& seq_lens_this_time_dtype,
                                                         const paddle::DataType& seq_lens_encoder_dtype,
                                                         const paddle::DataType& seq_lens_decoder_dtype) {
    return {seq_lens_this_time_dtype};
}

PD_BUILD_OP(speculate_get_seq_lens_output)
    .Inputs({"seq_lens_this_time", "seq_lens_encoder", "seq_lens_decoder"})
    .Outputs({"seq_lens_output"})
    .SetKernelFn(PD_KERNEL(SpeculateGetSeqLensOutput))
    .SetInferShapeFn(PD_INFER_SHAPE(SpeculateGetSeqLensOutputInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SpeculateGetSeqLensOutputInferDtype));