// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

__global__ void RemovePadding(int64_t *output_data,
                              const int64_t *input_data,
                              const int *seq_lens,
                              const int *cum_offsets,
                              const int sequence_length) {
  const int bi = blockIdx.x;
  const int tid = threadIdx.x;

  for (int i = tid; i < seq_lens[bi]; i += blockDim.x) {
    const int tgt_seq_id = bi * sequence_length - cum_offsets[bi] + i;
    const int src_seq_id = bi * sequence_length + i;
    output_data[tgt_seq_id] = input_data[src_seq_id];
  }
}

__global__ void GetCumOffsetKernel(int *token_num,
                                   int *enc_token_num,
                                   int *dec_token_num,
                                   int *cum_offsets,
                                   const int *sequence_lengths,
                                   const int *sequence_lengths_encoder,
                                   const int *sequence_lengths_decoder,
                                   const int batch_size,
                                   const int max_seq_len) {
  // get padding offset of each batch
  int total_seq_len = 0;
  int enc_total_seq_len = 0;
  int dec_total_seq_len = 0;
  int cum_offset = 0;
  int index = 0;
  
  for (int i = 0; i < batch_size; i++) {
    cum_offsets[i] = cum_offset;
    int seq_len = sequence_lengths[i];
    int seq_len_enc = sequence_lengths_encoder[i];
    int seq_len_dec = sequence_lengths_decoder[i];

    cum_offset += max_seq_len - seq_len;

    total_seq_len += seq_len;
    enc_total_seq_len += seq_len_enc;
    dec_total_seq_len += seq_len_dec;
  }
  token_num[0] = total_seq_len;
  enc_token_num[0] = enc_total_seq_len;
  dec_token_num[0] = dec_total_seq_len;
}

__global__ void GetPaddingOffsetKernel(int *padding_offset,
                                       int *cum_offsets_out,
                                       const int *cum_offsets,
                                       const int *seq_lens,
                                       const int max_seq_len) {
  // get padding offset of each batch
  const int bi = blockIdx.x;
  const int ti = threadIdx.x;
  if (ti == 0) {
    cum_offsets_out[bi] = bi == 0 ? 0 : cum_offsets[bi - 1];
  }
  int cum_offset = bi == 0 ? 0 : cum_offsets[bi - 1];
  for (int i = ti; i < seq_lens[bi]; i += blockDim.x) {
    padding_offset[bi * max_seq_len - cum_offset + i] = cum_offset;
  }
}


std::vector<paddle::Tensor> GetPaddingOffset(const paddle::Tensor& input_ids,
                                             const paddle::Tensor& cum_offsets,
                                             const paddle::Tensor& token_num,
                                             const paddle::Tensor& seq_len) {
    auto cu_stream = input_ids.stream();
    std::vector<int64_t> input_ids_shape = input_ids.shape();
    const int bsz = input_ids_shape[0];
    const int seq_length = input_ids_shape[1];
    auto cum_offsets_out = cum_offsets.copy_to(cum_offsets.place(), false);
    auto cpu_token_num = token_num.copy_to(paddle::CPUPlace(), false);
    const int token_num_data = cpu_token_num.data<int64_t>()[0];
    auto x_remove_padding = paddle::full({token_num_data}, 0, paddle::DataType::INT64, input_ids.place());
    auto padding_offset = paddle::full({token_num_data}, 0, paddle::DataType::INT32, input_ids.place());
    int blockSize = min((token_num_data + 32 - 1) / 32 * 32, 128);
    GetPaddingOffsetKernel<<<bsz, 128, 0, cu_stream>>>(
      padding_offset.data<int>(), 
      cum_offsets_out.data<int>(),
      cum_offsets.data<int>(),
      seq_len.data<int>(),
      seq_length);
    RemovePadding<<<bsz, blockSize, 0, cu_stream>>>(
      x_remove_padding.data<int64_t>(), 
      input_ids.data<int64_t>(), 
      seq_len.data<int>(),
      cum_offsets_out.data<int>(), 
      seq_length);
    return {x_remove_padding, cum_offsets_out, padding_offset}; // , enc_token_num, dec_token_num};
}

std::vector<std::vector<int64_t>> GetPaddingOffsetInferShape(const std::vector<int64_t>& input_ids_shape,
                                                             const std::vector<int64_t>& cum_offsets_shape,
                                                             const std::vector<int64_t>& token_num_shape,
                                                             const std::vector<int64_t>& seq_len_shape) {
    int64_t bsz = input_ids_shape[0];
    int64_t seq_len = input_ids_shape[1];
    return {{-1}, {bsz}, {-1}};
}

std::vector<paddle::DataType> GetPaddingOffsetInferDtype(const paddle::DataType& input_ids_dtype,
                                                         const paddle::DataType& cum_offsets_dtype,
                                                         const paddle::DataType& token_num_dtype,
                                                         const paddle::DataType& seq_len_dtype) {
    return {input_ids_dtype, seq_len_dtype, seq_len_dtype};
}

PD_BUILD_OP(get_padding_offset)
    .Inputs({"input_ids", "cum_offsets", "token_num", "seq_len"})
    .Outputs({"x_remove_padding", "cum_offsets_out", "padding_offset"})
    .SetKernelFn(PD_KERNEL(GetPaddingOffset))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPaddingOffsetInferDtype));