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


template <typename T, int VecSize>
__global__ void RebuildAppendPaddingKernel(T *output_data,
                               const T *input_data,
                               const int *cum_offset,
                               const int *seq_len_decoder,
                               const int *seq_len_encoder,
                               const int *output_padding_offset,
                               const int max_seq_len,
                               const int dim_embed,
                               const int64_t output_elem_nums) {
  AlignedVector<T, VecSize> src_vec;
  const int64_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int64_t i = global_idx * VecSize; i < output_elem_nums; i += gridDim.x * blockDim.x * VecSize) {
    const int out_token_id = i / dim_embed;
    const int ori_token_id = out_token_id + output_padding_offset[out_token_id];
    const int bi = ori_token_id / max_seq_len;
    int seq_id = 0;

    if (seq_len_decoder[bi] == 0 && seq_len_encoder[bi] == 0) continue;
    else if (seq_len_encoder[bi] != 0) {
      seq_id = seq_len_encoder[bi] - 1;
    }

    const int input_token_id = ori_token_id - cum_offset[bi] + seq_id;
    const int bias_idx = i % dim_embed;
    
    Load<T, VecSize>(&input_data[input_token_id * dim_embed + bias_idx], &src_vec);
    Store<T, VecSize>(src_vec, &output_data[i]);
  }
}


template <paddle::DataType D>
std::vector<paddle::Tensor> LaunchRebuildAppendPadding(const paddle::Tensor& input,
                                                    const paddle::Tensor& cum_offsets,
                                                    const paddle::Tensor& seq_len_decoder,
                                                    const paddle::Tensor& seq_len_encoder,
                                                    const paddle::Tensor& output_padding_offset,
                                                    int max_seq_len,
                                                    int dim_embed) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    int need_delete_token_num = 0;
    const int bsz = seq_len_encoder.shape()[0];

    auto seq_len_encoder_cpu = seq_len_encoder.copy_to(paddle::CPUPlace(), true);
    for (int i = 0; i < bsz; ++i) {
        if (seq_len_encoder_cpu.data<int>()[i] > 0) {
            need_delete_token_num += seq_len_encoder_cpu.data<int>()[i] - 1;
        }
    }
    const int token_num = input.shape()[0];
    auto output = paddle::full({token_num - need_delete_token_num, dim_embed}, 0, D, input.place());
    int64_t output_elem_nums = (token_num - need_delete_token_num) * dim_embed;
    constexpr int PackSize = VEC_16B / sizeof(DataType_);
    if (dim_embed % PackSize != 0) {
        PD_THROW("dim_embed=%d must be divisible by vec_size=%d", dim_embed, PackSize);
    }

    int pack_num = output_elem_nums / PackSize;
    const int threads_per_block = 128;
    int grid_size = 1;
    GetNumBlocks(pack_num, &grid_size);
    RebuildAppendPaddingKernel<DataType_, PackSize><<<grid_size, threads_per_block, 0, input.stream()>>>(
        reinterpret_cast<DataType_*>(output.data<data_t>()),
        reinterpret_cast<const DataType_*>(input.data<data_t>()),
        cum_offsets.data<int>(),
        seq_len_decoder.data<int>(),
        seq_len_encoder.data<int>(),
        output_padding_offset.data<int>(),
        max_seq_len,
        dim_embed,
        output_elem_nums);
    return {output};
}


std::vector<paddle::Tensor> DispatchRebuildAppendPaddingWithDtype(const paddle::Tensor& input,
                                                                  const paddle::Tensor& cum_offsets,
                                                                  const paddle::Tensor& seq_len_decoder,
                                                                  const paddle::Tensor& seq_len_encoder,
                                                                  const paddle::Tensor& output_padding_offset,
                                                                  int max_seq_len,
                                                                  int dim_embed) {
    switch (input.type()) {
        case paddle::DataType::BFLOAT16: 
            return LaunchRebuildAppendPadding<paddle::DataType::BFLOAT16>(input, 
                                                                          cum_offsets, 
                                                                          seq_len_decoder,
                                                                          seq_len_encoder,
                                                                          output_padding_offset,
                                                                          max_seq_len,
                                                                          dim_embed);
            break;
        case paddle::DataType::FLOAT16: 
            return LaunchRebuildAppendPadding<paddle::DataType::FLOAT16>(input, 
                                                                          cum_offsets, 
                                                                          seq_len_decoder,
                                                                          seq_len_encoder,
                                                                          output_padding_offset,
                                                                          max_seq_len,
                                                                          dim_embed);
            break;
        case paddle::DataType::FLOAT32: 
            return LaunchRebuildAppendPadding<paddle::DataType::FLOAT32>(input, 
                                                                          cum_offsets, 
                                                                          seq_len_decoder,
                                                                          seq_len_encoder,
                                                                          output_padding_offset,
                                                                          max_seq_len,
                                                                          dim_embed);
            break;
        default:
            PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");
            break;
    }
}

std::vector<paddle::Tensor> RebuildAppendPadding(const paddle::Tensor& input,
                                                 const paddle::Tensor& cum_offsets,
                                                 const paddle::Tensor& seq_len_decoder,
                                                 const paddle::Tensor& seq_len_encoder,
                                                 const paddle::Tensor& output_padding_offset,
                                                 int max_seq_len,
                                                 int dim_embed) {
    return DispatchRebuildAppendPaddingWithDtype(input, 
                                                cum_offsets, 
                                                seq_len_decoder, 
                                                seq_len_encoder, 
                                                output_padding_offset, 
                                                max_seq_len,
                                                dim_embed);
}


std::vector<std::vector<int64_t>> RebuildAppendPaddingInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& seq_len_decoder_shape,
    const std::vector<int64_t>& seq_len_encoder_shape,
    const std::vector<int64_t>& output_padding_offset_shape) {
  int64_t dim_embed = input_shape[1];
  std::vector<int64_t> dynamic_shape = {-1, dim_embed};

  return {dynamic_shape};
}


std::vector<paddle::DataType> RebuildAppendPaddingInferDtype(const paddle::DataType& input_dtype, 
                                                            const paddle::DataType& cum_offsets_dtype,
                                                            const paddle::DataType& seq_len_decoder_dtype,
                                                            const paddle::DataType& seq_len_encoder_dtype,
                                                            const paddle::DataType& output_padding_offset_dtype) {
    return {input_dtype};
}

PD_BUILD_OP(rebuild_append_padding)
    .Inputs({"input","cum_offsets","seq_len_decoder","seq_len_encoder","output_padding_offset"})
    .Outputs({"output"})
    .Attrs({"max_seq_len: int", "dim_embed: int"})
    .SetKernelFn(PD_KERNEL(RebuildAppendPadding))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildAppendPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildAppendPaddingInferDtype));