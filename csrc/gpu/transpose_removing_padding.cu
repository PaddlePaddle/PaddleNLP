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

template <typename T, int VecSize>
__global__ void TransposeRemovingPadding(const T* input_data,
                                         const int* seq_lens,
                                         T* output_data,
                                         const int batch_size,
                                         const int num_head,
                                         const int max_len_this_time,
                                         const int seq_len,
                                         const int head_dim,
                                         const int token_num,
                                         const int elem_cnt,
                                         const int* padding_offset) {
  // transpose and remove padding
  // [batch_size, num_head, max_len_this_time, head_dim] -> [token_num, num_head,
  // head_dim]
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int dim_embed = num_head * head_dim;
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;

  for (int32_t linear_index = idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / dim_embed;
    const int ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int ori_batch_id = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_batch_id] == 0) continue;
    const int ori_seq_id = ori_token_idx % seq_len;
    const int ori_head_id = (linear_index % dim_embed) / head_dim;
    const int ori_head_lane = (linear_index % dim_embed) % head_dim;
    const int ori_idx = ori_batch_id * num_head * max_len_this_time * head_dim +
                        ori_head_id * max_len_this_time * head_dim +
                        ori_seq_id * head_dim + ori_head_lane;
    Load<T, VecSize>(&input_data[ori_idx], &src_vec);
    Store<T, VecSize>(src_vec, &output_data[linear_index]);
  }
}

template <typename T>
void InvokeTransposeRemovePadding(const T* input_data,
                                  const int* seq_lens,
                                  T* output_data,
                                  const int batch_size,
                                  const int num_head,
                                  const int max_len_this_time,
                                  const int seq_len,
                                  const int head_dim,
                                  const int token_num,
                                  const int* padding_offset,
#ifdef PADDLE_WITH_HIP
                                  hipStream_t cu_stream
#else
                                  cudaStream_t cu_stream
#endif
                                  ) {
  // [batch_size, num_head, max_len_this_time, head_dim] -> [token_num, num_head,
  // head_dim]
  constexpr int VEC_16B = 16;
  const int elem_cnt = token_num * num_head * head_dim;
  constexpr int PackSize = VEC_16B / sizeof(T);
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t block_size = 128;
  int32_t grid_size = (pack_num + block_size - 1) / block_size;
  TransposeRemovingPadding<T, PackSize>
      <<<grid_size, block_size, 0, cu_stream>>>(input_data,
                                                seq_lens,
                                                output_data,
                                                batch_size,
                                                num_head,
                                                max_len_this_time,
                                                seq_len,
                                                head_dim,
                                                token_num,
                                                elem_cnt,
                                                padding_offset);
}

template <paddle::DataType D>
std::vector<paddle::Tensor> apply_transpose_remove_padding(const paddle::Tensor& input, 
                                                           const paddle::Tensor& seq_lens, 
                                                           const paddle::Tensor& padding_offset) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    auto cu_stream = input.stream();
    std::vector<int64_t> input_shape = input.shape();
    const int bsz = input_shape[0];
    const int num_head = input_shape[1];
    const int seq_len = input_shape[2];
    const int dim_head = input_shape[3];
    const int token_num = padding_offset.shape()[0];

    auto out = paddle::full({token_num, num_head * dim_head}, 0, input.dtype(), input.place());
    InvokeTransposeRemovePadding(
        reinterpret_cast<DataType_*>(const_cast<data_t*>(input.data<data_t>())),
        seq_lens.data<int>(),
        reinterpret_cast<DataType_*>(out.data<data_t>()),
        bsz,
        num_head,
        seq_len,
        seq_len,
        dim_head,
        token_num,
        padding_offset.data<int>(),
        cu_stream
    );
    return {out};
}

std::vector<paddle::Tensor> ApplyTransposeRemovingPadding(const paddle::Tensor& input, 
                                                          const paddle::Tensor& seq_lens, 
                                                          const paddle::Tensor& padding_offset) {
    switch (input.type()) {
        case paddle::DataType::BFLOAT16: {
            return apply_transpose_remove_padding<paddle::DataType::BFLOAT16>(
                input,
                seq_lens,
                padding_offset
            );
        }
        case paddle::DataType::FLOAT16: {
            return apply_transpose_remove_padding<paddle::DataType::FLOAT16>(
                input,
                seq_lens,
                padding_offset
            );
        }
        case paddle::DataType::FLOAT32: {
            return apply_transpose_remove_padding<paddle::DataType::FLOAT32>(
                input,
                seq_lens,
                padding_offset
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16, bfloat16 and float32 are supported. ");
            break;
        }
    }
}

std::vector<std::vector<int64_t>> ApplyTransposeRemovingPaddingInferShape(
        const std::vector<int64_t>& input_shape, 
        const std::vector<int64_t>& seq_lens_shape,
        const std::vector<int64_t>& padding_offset_shape) {
    return {{padding_offset_shape[0], input_shape[1] * input_shape[3]}};
}

std::vector<paddle::DataType> ApplyTransposeRemovingPaddingInferDtype(
        const paddle::DataType& input_dtype, 
        const paddle::DataType& seq_lens_dtype,
        const paddle::DataType& padding_offset_dtype) {
    return {input_dtype};
}

PD_BUILD_OP(transpose_remove_padding)
    .Inputs({"input", "seq_lens", "padding_offset"})
    .Outputs({"fmha_out"})
    .SetKernelFn(PD_KERNEL(ApplyTransposeRemovingPadding))
    .SetInferShapeFn(PD_INFER_SHAPE(ApplyTransposeRemovingPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ApplyTransposeRemovingPaddingInferDtype));