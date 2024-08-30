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
__global__ void RebuildPaddingV2Kernel(T *output_data,
                                      const T *input_data,
                                      const int *cum_offsets,
                                      const int *seq_len_decoder,
                                      const int *seq_len_encoder,
                                      const int seq_len,
                                      const int dim_embed,
                                      const int elem_nums) {
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;
  const int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = global_idx * VecSize; i < elem_nums; i += gridDim.x * blockDim.x * VecSize) {
    const int bi = i / dim_embed;
    const int bias_idx = i % dim_embed;
    int seq_id = 0;
    // just encoder or stop, get last token; just decoder, get first token.
    if (seq_len_decoder[bi] == 0) {
        if (seq_len_encoder[bi] != 0) {
            seq_id = seq_len_encoder[bi] - 1;
        } else {
            return;
        }
    }
    const int ori_token_idx = bi * seq_len - cum_offsets[bi] + seq_id;
    const int src_offset = ori_token_idx * dim_embed + bias_idx;
    Load<T, VecSize>(&input_data[src_offset], &src_vec);
    Store<T, VecSize>(src_vec, &output_data[i]);
  }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> rebuild_padding_v2(const paddle::Tensor& tmp_out, // [token_num, dim_embed]
                                               const paddle::Tensor& cum_offsets, // [bsz, 1]
                                               const paddle::Tensor& seq_lens_decoder,
                                               const paddle::Tensor& seq_lens_encoder,
                                               int max_input_length) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    auto cu_stream = tmp_out.stream();
    std::vector<int64_t> tmp_out_shape = tmp_out.shape();
    const int token_num = tmp_out_shape[0];
    const int dim_embed = tmp_out_shape[1];
    const int bsz = cum_offsets.shape()[0];
    auto out = paddle::full({bsz, dim_embed}, 0, tmp_out.dtype(), tmp_out.place());
    constexpr int PackSize = VEC_16B / sizeof(DataType_);
    int elem_nums = out.numel();
    int pack_num = elem_nums / PackSize;
    const int blocksize = 128;
    const int grid_size = (pack_num + blocksize - 1) / blocksize;
    RebuildPaddingV2Kernel<DataType_, PackSize><<<grid_size, blocksize, 0, tmp_out.stream()>>>(
        reinterpret_cast<DataType_*>(out.data<data_t>()), 
        reinterpret_cast<DataType_*>(const_cast<data_t*>(tmp_out.data<data_t>())), 
        cum_offsets.data<int>(), 
        seq_lens_decoder.data<int>(), 
        seq_lens_encoder.data<int>(), 
        max_input_length, 
        dim_embed, 
        elem_nums);
    return {out};
}

std::vector<paddle::Tensor> RebuildPaddingV2(const paddle::Tensor& tmp_out, // [token_num, dim_embed]
                                             const paddle::Tensor& cum_offsets, // [bsz, 1]
                                             const paddle::Tensor& seq_lens_decoder,
                                             const paddle::Tensor& seq_lens_encoder,
                                             int max_input_length) {
    switch (tmp_out.type()) {
        case paddle::DataType::BFLOAT16: {
            return rebuild_padding_v2<paddle::DataType::BFLOAT16>(
                tmp_out,
                cum_offsets,
                seq_lens_decoder,
                seq_lens_encoder,
                max_input_length
            );
        }
        case paddle::DataType::FLOAT16: {
            return rebuild_padding_v2<paddle::DataType::FLOAT16>(
                tmp_out,
                cum_offsets,
                seq_lens_decoder,
                seq_lens_encoder,
                max_input_length
            );
        }
        case paddle::DataType::FLOAT32: {
            return rebuild_padding_v2<paddle::DataType::FLOAT32>(
                tmp_out,
                cum_offsets,
                seq_lens_decoder,
                seq_lens_encoder,
                max_input_length
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

std::vector<std::vector<int64_t>> RebuildPaddingV2InferShape(const std::vector<int64_t>& tmp_out_shape,
                                                             const std::vector<int64_t>& cum_offsets_shape,
                                                             const std::vector<int64_t>& seq_lens_decoder_shape,
                                                             const std::vector<int64_t>& seq_lens_encoder_shape) {
    int64_t bsz = cum_offsets_shape[0];
    int64_t dim_embed = tmp_out_shape[1];
    return {{bsz, dim_embed}};
}

std::vector<paddle::DataType> RebuildPaddingV2InferDtype(const paddle::DataType& tmp_out_dtype,
                                                         const paddle::DataType& cum_offsets_dtype,
                                                         const paddle::DataType& seq_lens_decoder_dtype,
                                                         const paddle::DataType& seq_lens_encoder_dtype) {
    return {tmp_out_dtype};
}

PD_BUILD_OP(rebuild_padding_v2)
    .Inputs({"tmp_out", "cum_offsets", "seq_lens_decoder", "seq_lens_encoder"})
    .Outputs({"out"})
    .Attrs({"max_input_length: int"})
    .SetKernelFn(PD_KERNEL(RebuildPaddingV2))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildPaddingV2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildPaddingV2InferDtype));