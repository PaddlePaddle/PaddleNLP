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
#include "helper.h"

__global__ void GetPaddingOffsetV2Kernel(int *padding_offset,
                                         int *cum_offsets_out,
                                         int *cu_seqlens_q,
                                         int *cu_seqlens_k,
                                         int64_t *output_data,
                                         const int64_t *input_data,
                                         const int *cum_offsets,
                                         const int *seq_lens,
                                         const int64_t *draft_tokens,
                                         const int *seq_lens_encoder,
                                         const int max_draft_tokens,
                                         const int max_seq_len) {
  // get padding offset of each batch
  const int bi = blockIdx.x;
  const int ti = threadIdx.x;
  int cum_offset = bi == 0 ? 0 : cum_offsets[bi - 1];
  for (int i = ti; i < seq_lens[bi]; i += blockDim.x) {
    padding_offset[bi * max_seq_len - cum_offset + i] = cum_offset;
    const int tgt_seq_id = bi * max_seq_len - cum_offset + i;
    if (draft_tokens == nullptr) {
      const int src_seq_id = bi * max_seq_len + i;
      output_data[tgt_seq_id] = input_data[src_seq_id];
    } else { // speculative decoding
      if (seq_lens_encoder[bi] > 0) {
        const int src_seq_id = bi * max_seq_len + i;
        output_data[tgt_seq_id] = input_data[src_seq_id];
      } else {
        const int src_seq_id = bi * max_draft_tokens + i;
        output_data[tgt_seq_id] = draft_tokens[src_seq_id];
      }
    }
  }
  if (ti == 0) {
    if (bi == 0) {
      cu_seqlens_q[0] = 0;
      cu_seqlens_k[0] = 0;
    }
    cum_offsets_out[bi] = cum_offset;
    int cum_seq_len = (bi + 1) * max_seq_len - cum_offsets[bi];
    cu_seqlens_q[bi + 1] = cum_seq_len;
    cu_seqlens_k[bi + 1] = cum_seq_len;
  }
}


std::vector<paddle::Tensor> GetPaddingOffsetV2(const paddle::Tensor& input_ids,
                                               const paddle::Tensor& cum_offsets,
                                               const paddle::Tensor& token_num,
                                               const paddle::Tensor& seq_len,
                                               const paddle::optional<paddle::Tensor>& draft_tokens,
                                               const paddle::optional<paddle::Tensor>& seq_lens_encoder) {
    auto cu_stream = input_ids.stream();
    std::vector<int64_t> input_ids_shape = input_ids.shape();
    const int bsz = seq_len.shape()[0];
    const int seq_length = input_ids_shape[1];
    auto cum_offsets_out = cum_offsets.copy_to(cum_offsets.place(), false);
    auto cpu_token_num = token_num.copy_to(paddle::CPUPlace(), false);

    const int token_num_data = cpu_token_num.data<int64_t>()[0];

    auto x_remove_padding = GetEmptyTensor({token_num_data}, paddle::DataType::INT64, input_ids.place());
    auto padding_offset = GetEmptyTensor({token_num_data}, paddle::DataType::INT32, input_ids.place());
    auto cu_seqlens_q = GetEmptyTensor({bsz + 1}, paddle::DataType::INT32, input_ids.place());
    auto cu_seqlens_k = GetEmptyTensor({bsz + 1}, paddle::DataType::INT32, input_ids.place());

    int max_draft_tokens = 0;
    if (draft_tokens) { // speculative decoding
      max_draft_tokens = draft_tokens.get().shape()[1];
      GetPaddingOffsetV2Kernel<<<bsz, 128, 0, cu_stream>>>(
        padding_offset.data<int>(), 
        cum_offsets_out.data<int>(),
        cu_seqlens_q.data<int>(),
        cu_seqlens_k.data<int>(),
        x_remove_padding.data<int64_t>(), 
        input_ids.data<int64_t>(), 
        cum_offsets.data<int>(),
        seq_len.data<int>(),
        draft_tokens.get_ptr()->data<int64_t>(),
        seq_lens_encoder.get_ptr()->data<int>(),
        max_draft_tokens,
        seq_length);
    } else {
      GetPaddingOffsetV2Kernel<<<bsz, 128, 0, cu_stream>>>(
        padding_offset.data<int>(), 
        cum_offsets_out.data<int>(),
        cu_seqlens_q.data<int>(),
        cu_seqlens_k.data<int>(),
        x_remove_padding.data<int64_t>(), 
        input_ids.data<int64_t>(), 
        cum_offsets.data<int>(),
        seq_len.data<int>(),
        nullptr,
        nullptr,
        max_draft_tokens,
        seq_length);
    }
    return {x_remove_padding, cum_offsets_out, padding_offset, cu_seqlens_q, cu_seqlens_k}; // , enc_token_num, dec_token_num};
}

std::vector<std::vector<int64_t>> GetPaddingOffsetV2InferShape(const std::vector<int64_t>& input_ids_shape,
                                                             const std::vector<int64_t>& cum_offsets_shape,
                                                             const std::vector<int64_t>& token_num_shape,
                                                             const std::vector<int64_t>& seq_len_shape,
                                                             const std::vector<int64_t>& draft_tokens_shape,
                                                             const std::vector<int64_t>& seq_lens_encoder_shape) {
    int64_t bsz = seq_len_shape[0];
    int64_t seq_len = input_ids_shape[1];
    return {{-1}, {bsz}, {-1}, {bsz + 1}, {bsz + 1}};
}

std::vector<paddle::DataType> GetPaddingOffsetV2InferDtype(const paddle::DataType& input_ids_dtype,
                                                         const paddle::DataType& cum_offsets_dtype,
                                                         const paddle::DataType& token_num_dtype,
                                                         const paddle::DataType& seq_len_dtype,
                                                         const paddle::DataType& draft_tokens_dtype,
                                                         const paddle::DataType& seq_lens_encoder_dtype) {
    return {input_ids_dtype, seq_len_dtype, seq_len_dtype, seq_len_dtype, seq_len_dtype};
}

PD_BUILD_OP(get_padding_offset_v2)
    .Inputs({"input_ids", "cum_offsets", "token_num", "seq_len", paddle::Optional("draft_tokens"), paddle::Optional("seq_lens_encoder"),})
    .Outputs({"x_remove_padding", "cum_offsets_out", "padding_offset", "cu_seqlens_q", "cu_seqlens_k"})
    .SetKernelFn(PD_KERNEL(GetPaddingOffsetV2))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPaddingOffsetV2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPaddingOffsetV2InferDtype));