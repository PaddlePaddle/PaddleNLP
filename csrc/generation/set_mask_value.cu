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

#include "paddle/extension.h"

template <paddle::DataType D>
class PDTraits;

template <>
class PDTraits<paddle::DataType::FLOAT32> {
public:
  typedef float DataType;
  typedef float data_t;
};

template <>
class PDTraits<paddle::DataType::FLOAT16> {
public:
  typedef half DataType;
  typedef paddle::float16 data_t;
};

template <>
class PDTraits<paddle::DataType::BFLOAT16> {
public:
  typedef __nv_bfloat16 DataType;
  typedef paddle::bfloat16 data_t;
};

template <typename T>
__global__ void set_value_by_id(const int *seq_lens, const bool *stop_flags, T *output_data, int *sequence_lengths, int bs, int length) {
    int tid = threadIdx.x;
    if (tid < bs) {
        T *output_data_now = output_data + tid * length;
        output_data_now[seq_lens[tid]] = 1.0;
        if (stop_flags[tid]) {
            sequence_lengths[tid] = 0;
        }
    }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> set_mask_value(const paddle::Tensor& input_data, const paddle::Tensor& stop_flags, const paddle::Tensor& seq_lens) {
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

    int block_size = (input_bs + 32 - 1) / 32 * 32;
    set_value_by_id<<<1, block_size, 0, cu_stream>>>(seq_lens.data<int>(),
                                                     stop_flags.data<bool>(),
                                                     reinterpret_cast<DataType_*>(const_cast<data_t*>(input_data.data<data_t>())),
                                                     sequence_lengths.data<int>(), seq_bs, length);
    return {sequence_lengths};
}

std::vector<paddle::Tensor> SetMaskValue(const paddle::Tensor& input_data, const paddle::Tensor& stop_flags, const paddle::Tensor& seq_lens) {
    switch (input_data.type()) {
        case paddle::DataType::BFLOAT16: {
            return set_mask_value<paddle::DataType::BFLOAT16>(
                input_data,
                stop_flags,
                seq_lens
            );
        }
        case paddle::DataType::FLOAT16: {
            return set_mask_value<paddle::DataType::FLOAT16>(
                input_data,
                stop_flags,
                seq_lens
            );
        }
        case paddle::DataType::FLOAT32: {
            return set_mask_value<paddle::DataType::FLOAT32>(
                input_data,
                stop_flags,
                seq_lens
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

std::vector<std::vector<int64_t>> SetMaskValueInferShape(const std::vector<int64_t>& input_data_shape, const std::vector<int64_t>& stop_flags_shape, const std::vector<int64_t>& seq_lens_shape) {
    return {seq_lens_shape};
}

std::vector<paddle::DataType> SetMaskValueInferDtype(const paddle::DataType& input_data_dtype, const paddle::DataType& stop_flags_dtype, const paddle::DataType& seq_lens_dtype) {
    return {seq_lens_dtype};
}

PD_BUILD_OP(set_mask_value)
    .Inputs({"input_data", "stop_flags", "seq_lens"})
    .Outputs({"sequence_lengths"})
    .SetKernelFn(PD_KERNEL(SetMaskValue))
    .SetInferShapeFn(PD_INFER_SHAPE(SetMaskValueInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SetMaskValueInferDtype));
