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
__global__ void RebuildPaddingKernel(T *output_data,
                                     const T *input_data,
                                     const int *cum_offsets,
                                     const int *seq_lens,
                                     const int max_seq_len,
                                     const int dim_embed,
                                     const int elem_nums) {
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;
  const int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = global_idx * VecSize; i < elem_nums; i += gridDim.x * blockDim.x * VecSize) {
    const int bi = i / dim_embed;
    const int bias_idx = i % dim_embed;
    int seq_id = seq_lens[bi] - 1;
    const int ori_token_idx = bi * max_seq_len - cum_offsets[bi] + seq_id;
    const int src_offset = ori_token_idx * dim_embed + bias_idx;
    Load<T, VecSize>(&input_data[src_offset], &src_vec);
    Store<T, VecSize>(src_vec, &output_data[i]);
  }
}

template <typename T>
__global__ void RebuildPaddingKernel(T *output_data,
                                    const T *input_data,
                                    const int *padding_offset,
                                    const int dim_embed) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int dst_seq_id = bid + padding_offset[bid];
  const int src_seq_id = bid;

  for (int i = tid; i < dim_embed; i += blockDim.x) {
    output_data[dst_seq_id * dim_embed + i] =
        input_data[src_seq_id * dim_embed + i];
  }
}

template <typename T>
void InvokeRebuildPadding(T *output_data,
                          const T *input_data,
                          const int *padding_offset,
                          const int token_num,
                          const int dim_embed,
#ifdef PADDLE_WITH_HIP
                          hipStream_t stream
#else
                          cudaStream_t stream
#endif
                          ) {
  // src: [token_num, dim_embed]
  // dst: [batch_size * max_seq_len, dim_embed]
  RebuildPaddingKernel<<<token_num, 256, 0, stream>>>(
      output_data, input_data, padding_offset, dim_embed);
}

template <paddle::DataType D>
std::vector<paddle::Tensor> rebuild_padding(const paddle::Tensor& tmp_out, // [token_num, dim_embed]
                                            const paddle::Tensor& padding_offset, // [bsz, 1]
                                            const paddle::Tensor& seq_lens,
                                            const paddle::Tensor& input_ids) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    auto cu_stream = tmp_out.stream();
    std::vector<int64_t> tmp_out_shape = tmp_out.shape();
    const int token_num = tmp_out_shape[0];
    const int dim_embed = tmp_out_shape[1];
    const int bsz = seq_lens.shape()[0];
    auto out = paddle::full({bsz, dim_embed}, 0, tmp_out.dtype(), tmp_out.place());
    constexpr int PackSize = VEC_16B / sizeof(DataType_);
    int elem_nums = out.numel();
    int pack_num = elem_nums / PackSize;
    const int blocksize = 128;
    const int grid_size = (pack_num + blocksize - 1) / blocksize;
    RebuildPaddingKernel<DataType_, PackSize><<<grid_size, blocksize, 0, tmp_out.stream()>>>(
        reinterpret_cast<DataType_*>(out.data<data_t>()), 
        reinterpret_cast<DataType_*>(const_cast<data_t*>(tmp_out.data<data_t>())), 
        padding_offset.data<int>(), 
        seq_lens.data<int>(), 
        input_ids.shape()[1], 
        dim_embed, 
        elem_nums);
    // InvokeRebuildPadding(
    //     reinterpret_cast<DataType_*>(out.data<data_t>()), 
    //     reinterpret_cast<DataType_*>(const_cast<data_t*>(tmp_out.data<data_t>())), 
    //     padding_offset.data<int>(),
    //     token_num,
    //     dim_embed,
    //     tmp_out.stream()
    // );
    return {out};
}

std::vector<paddle::Tensor> RebuildPadding(const paddle::Tensor& tmp_out, 
                                           const paddle::Tensor& padding_offset, 
                                           const paddle::Tensor& seq_lens,
                                           const paddle::Tensor& input_ids) {
    switch (tmp_out.type()) {
        case paddle::DataType::BFLOAT16: {
            return rebuild_padding<paddle::DataType::BFLOAT16>(
                tmp_out,
                padding_offset,
                seq_lens,
                input_ids
            );
        }
        case paddle::DataType::FLOAT16: {
            return rebuild_padding<paddle::DataType::FLOAT16>(
                tmp_out,
                padding_offset,
                seq_lens,
                input_ids
            );
        }
        case paddle::DataType::FLOAT32: {
            return rebuild_padding<paddle::DataType::FLOAT32>(
                tmp_out,
                padding_offset,
                seq_lens,
                input_ids
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

std::vector<std::vector<int64_t>> RebuildPaddingInferShape(const std::vector<int64_t>& tmp_out_shape,
                                                           const std::vector<int64_t>& padding_offset_shape,
                                                           const std::vector<int64_t>& seq_lens_shape,
                                                           const std::vector<int64_t>& input_ids_shape) {
    int64_t bsz = seq_lens_shape[0];
    int64_t dim_embed = tmp_out_shape[1];
    return {{bsz, dim_embed}};
}

std::vector<paddle::DataType> RebuildPaddingInferDtype(const paddle::DataType& tmp_out_dtype,
                                                       const paddle::DataType& padding_offset_dtype,
                                                       const paddle::DataType& seq_lens_dtype,
                                                       const paddle::DataType& input_ids_dtype) {
    return {tmp_out_dtype};
}

PD_BUILD_OP(rebuild_padding)
    .Inputs({"tmp_out", "padding_offset", "seq_lens", "input_ids"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(RebuildPadding))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildPaddingInferDtype));