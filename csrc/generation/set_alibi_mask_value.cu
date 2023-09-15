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

template <typename T>
__global__ void set_value_by_id(const int *seq_lens, 
                               const bool *stop_flags, 
                              const float *alibi_slopes, 
                              const int64_t *tgt_pos, 
                              T *output_data, 
                              int *sequence_lengths, 
                              int bs, 
                              int length,
                              int num_head) {
    int bs_id = blockIdx.x;                          
    int hid = threadIdx.x;
    if (bs_id < bs) {
        T *output_data_now = output_data + bs_id * num_head * length + hid * length;
        float tgt_pos_now = static_cast<float>(tgt_pos[bs_id]);
        output_data_now[seq_lens[bs_id]] = static_cast<T>(tgt_pos_now * alibi_slopes[hid]);
        if (stop_flags[bs_id]) {
            sequence_lengths[bs_id] = 0;
        }
    }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> set_mask_value(const paddle::Tensor& input_data, 
                                           const paddle::Tensor& stop_flags, 
                                          const paddle::Tensor& seq_lens,
                                          const paddle::Tensor& alibi_slopes,
                                          const paddle::Tensor& tgt_pos
                                          ) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    PD_CHECK(seq_lens.dtype() == paddle::DataType::INT32);
    PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);
    auto cu_stream = input_data.stream();
    std::vector<int64_t> input_data_shape = input_data.shape();
    std::vector<int64_t> seq_lens_shape = seq_lens.shape();
    auto sequence_lengths = seq_lens.copy_to(seq_lens.place(), false);

    int input_bs = input_data_shape[0];
    int length = input_data_shape[3];
    int seq_bs = seq_lens_shape[0];
    int num_head = alibi_slopes.shape()[0];

    int grid_size = input_bs;
    int block_size = num_head;
    set_value_by_id<<<grid_size, block_size, 0, cu_stream>>>(seq_lens.data<int>(), 
                                                     stop_flags.data<bool>(), 
                                                     alibi_slopes.data<float>(),
                                                     tgt_pos.data<int64_t>(),
                                                     reinterpret_cast<DataType_*>(const_cast<data_t*>(input_data.data<data_t>())), 
                                                     sequence_lengths.data<int>(), seq_bs, length, num_head);
    return {sequence_lengths};
}

std::vector<paddle::Tensor> SetMaskValue(const paddle::Tensor& input_data, 
                                          const paddle::Tensor& stop_flags, 
                                          const paddle::Tensor& seq_lens,
                                          const paddle::Tensor& alibi_slopes,
                                          const paddle::Tensor& tgt_pos) {
    switch (input_data.type()) {
        case paddle::DataType::BFLOAT16: {
            return set_mask_value<paddle::DataType::BFLOAT16>(
                input_data,
                stop_flags,
                seq_lens,
                alibi_slopes,
                tgt_pos
            );
        }
        case paddle::DataType::FLOAT16: {
            return set_mask_value<paddle::DataType::FLOAT16>(
                input_data,
                stop_flags,
                seq_lens,
                alibi_slopes,
                tgt_pos
            );
        }
        case paddle::DataType::FLOAT32: {
            return set_mask_value<paddle::DataType::FLOAT32>(
                input_data,
                stop_flags,
                seq_lens,
                alibi_slopes,
                tgt_pos
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

std::vector<std::vector<int64_t>> SetMaskValueInferShape(const std::vector<int64_t>& input_data_shape, 
                                                         const std::vector<int64_t>& stop_flags_shape, 
                                                         const std::vector<int64_t>& seq_lens_shape,
                                                         const std::vector<int64_t>& alibi_slopes_shape,
                                                         const std::vector<int64_t>& tgt_pos) {
    return {seq_lens_shape};
}

std::vector<paddle::DataType> SetMaskValueInferDtype(const paddle::DataType& input_data_dtype, 
                                                      const paddle::DataType& stop_flags_dtype, 
                                                      const paddle::DataType& seq_lens_dtype,
                                                      const paddle::DataType& alibi_slopes_dtype,
                                                      const paddle::DataType& tgt_pos_dtype) {
    return {seq_lens_dtype};
}

PD_BUILD_OP(set_alibi_mask_value)
    .Inputs({"input_data", "stop_flags", "seq_lens", "alibi_slopes", "tgt_pos"})
    .Outputs({"sequence_lengths"})
    .SetKernelFn(PD_KERNEL(SetMaskValue))
    .SetInferShapeFn(PD_INFER_SHAPE(SetMaskValueInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SetMaskValueInferDtype));