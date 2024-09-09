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

__global__ void SpeculateGetOutputPaddingOffsetKernel(
                              int* output_padding_offset,
                              int* output_cum_offsets,
                              const int *output_cum_offsets_tmp,
                              const int *seq_lens_output,
                              const int max_seq_len) {
    // get padding offset of each batch
  const int bi = blockIdx.x;
  const int ti = threadIdx.x;
  int cum_offset = bi == 0 ? 0 : output_cum_offsets_tmp[bi - 1];
  for (int i = ti; i < seq_lens_output[bi]; i += blockDim.x) {
    output_padding_offset[bi * max_seq_len - cum_offset + i] = cum_offset;
  }
  if (ti == 0) {
    output_cum_offsets[bi] = cum_offset;
  }
}

std::vector<paddle::Tensor> SpeculateGetOutputPaddingOffset(const paddle::Tensor& output_cum_offsets_tmp,
                                             const paddle::Tensor& out_token_num,
                                             const paddle::Tensor& seq_lens_output,
                                             const int max_seq_len) {
    auto cu_stream = output_cum_offsets_tmp.stream();
    std::vector<int64_t> output_cum_offsets_tmp_shape = output_cum_offsets_tmp.shape();
    const int bsz = output_cum_offsets_tmp_shape[0];
    auto cpu_out_token_num = out_token_num.copy_to(paddle::CPUPlace(), false);

    auto output_padding_offset = paddle::full({cpu_out_token_num}, 0, paddle::DataType::INT32, output_cum_offsets_tmp.place());
    auto output_cum_offsets = output_cum_offsets_tmp.copy_to(output_cum_offsets_tmp.place(), false);

    SpeculateGetOutputPaddingOffsetKernel<<<bsz, 256, 0, cu_stream>>>(output_padding_offset.data<int>(),
                                                              output_cum_offsets.data<int>(),
                                                              output_cum_offsets_tmp.data<int>(),
                                                              seq_lens_output.data<int>(),
                                                              max_seq_len);
    
    return {output_padding_offset, output_cum_offsets};
}

std::vector<std::vector<int64_t>> SpeculateGetOutputPaddingOffsetInferShape(const std::vector<int64_t>& output_cum_offsets_tmp_shape,
                                                             const std::vector<int64_t>& out_token_num_shape,
                                                             const std::vector<int64_t>& seq_lens_output_shape) {
    int64_t bsz = output_cum_offsets_tmp_shape[0];
    return {{-1}, {bsz}};
}

std::vector<paddle::DataType> SpeculateGetOutputPaddingOffsetInferDtype(const paddle::DataType& output_cum_offsets_tmp_dtype,
                                                         const paddle::DataType& out_token_num_dtype,
                                                         const paddle::DataType& seq_lens_output_dtype) {
    return {output_cum_offsets_tmp_dtype, output_cum_offsets_tmp_dtype};
}

PD_BUILD_OP(speculate_get_output_padding_offset)
    .Inputs({"output_cum_offsets_tmp", "out_token_num", "seq_lens_output"})
    .Outputs({"output_padding_offset", "output_cum_offsets"})
    .Attrs({"max_seq_len: int"})
    .SetKernelFn(PD_KERNEL(SpeculateGetOutputPaddingOffset))
    .SetInferShapeFn(PD_INFER_SHAPE(SpeculateGetOutputPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SpeculateGetOutputPaddingOffsetInferDtype));