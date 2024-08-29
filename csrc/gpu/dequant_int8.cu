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
#include<stdlib.h>
#include<string.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>
#include<stdio.h>
#include<algorithm>


constexpr int DequantKernelVecSize = 4;

template <typename data_t>
inline HOSTDEVICE data_t roundWithTiesToEven(data_t x) {
  data_t xLower = floor(x);
  data_t xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  data_t dLower = x - xLower;
  data_t dUpper = xUpper - x;
  return static_cast<data_t>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}

template <typename data_t, int VecSize>
__global__ void DequantKernel(data_t* output,
                              const int32_t* input,
                              const int m,  // batch size
                              const int n,  // hidden
                              const float* dequant_out_scale_data) {
  int numel = m * n;
  int stride = blockDim.x * gridDim.x * VecSize;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  int col_id = idx % n;

  AlignedVector<int32_t, VecSize> in_vec;
  AlignedVector<float, VecSize> out_scale_vec;
  AlignedVector<data_t, VecSize> out_vec;

  for (; idx < numel; idx += stride) {
    Load<int32_t, VecSize>(input + idx, &in_vec);
    Load<float, VecSize>(dequant_out_scale_data + col_id, &out_scale_vec);

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      out_vec[i] =
          static_cast<data_t>(static_cast<float>(in_vec[i]) * out_scale_vec[i]);
    }

    Store<data_t, VecSize>(out_vec, output + idx);
  }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> DispatchLaunchDequantInt8(const paddle::Tensor& input,
                                    const paddle::Tensor& scale) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    std::vector<int64_t> input_shape = input.shape();

    auto output=paddle::full(input_shape, 0, D, input.place());
    int64_t m = input_shape[0];
    int64_t n = input_shape[1];

    int64_t numel = m*n;
    constexpr int64_t thread_per_block = 512;
    int64_t block_per_grid = (numel / DequantKernelVecSize + thread_per_block - 1) / thread_per_block;
    auto stream = input.stream();

    DequantKernel<DataType_, DequantKernelVecSize>
        <<<block_per_grid, thread_per_block, 0, stream>>>(
            reinterpret_cast<DataType_*>(output.data<data_t>()),
            reinterpret_cast<const int32_t*>(input.data<int32_t>()), m, n, 
            reinterpret_cast<const float*>(scale.data<float>()));

    
    return {output};
    
}


std::vector<paddle::Tensor> LaunchDequantInt8(const paddle::Tensor& input,
                                              const paddle::Tensor& scale,
                                              std::string dtype) {
    paddle::DataType data_type;

    if (dtype == "float32")
        data_type = paddle::DataType::FLOAT32;
    else if (dtype == "bfloat16")
        data_type = paddle::DataType::BFLOAT16;
    else if (dtype ==  "float16")
        data_type = paddle::DataType::FLOAT16;
    else 
        PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");

    switch (data_type) {
        case paddle::DataType::BFLOAT16: 
            return DispatchLaunchDequantInt8<paddle::DataType::BFLOAT16>(input, scale);
            break;
        case paddle::DataType::FLOAT16: 
            return DispatchLaunchDequantInt8<paddle::DataType::FLOAT16>(input, scale);
            break;
        case paddle::DataType::FLOAT32: 
            return DispatchLaunchDequantInt8<paddle::DataType::FLOAT32>(input, scale);
            break;
        default:
            break;
    }
}

std::vector<paddle::Tensor> DequantInt8(const paddle::Tensor& input,
                                        const paddle::Tensor& out_scale,
                                        std::string dtype
                                        ) {
    return LaunchDequantInt8(input, out_scale, dtype);
}

std::vector<std::vector<int64_t>> DequantInt8Shape(const std::vector<int64_t>& input_shape) {
    return {input_shape};
}

std::vector<paddle::DataType> DequantInt8Dtype(const paddle::DataType& input_dtype, const paddle::DataType& out_scale_dtype, std::string dtype) {
    paddle::DataType data_type;
    if (dtype == "float32")
        data_type = paddle::DataType::FLOAT32;
    else if (dtype == "bfloat16")
        data_type = paddle::DataType::BFLOAT16;
    else if (dtype ==  "float16")
        data_type = paddle::DataType::FLOAT16;
    else 
        PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");

    return {data_type};
}

PD_BUILD_OP(dequant_int8)
    .Inputs({"intput","out_scale"})
    .Outputs({"output"})
    .Attrs({"dtype: std::string"})
    .SetKernelFn(PD_KERNEL(DequantInt8))
    .SetInferShapeFn(PD_INFER_SHAPE(DequantInt8Shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DequantInt8Dtype));