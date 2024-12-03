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
#ifdef PADDLE_WITH_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#else
#include<cuda_fp16.h>
#include<cuda_bf16.h>
#endif


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

template <typename T>
__forceinline__ __device__ T add_mul(T a, T b, T c) {
    return (a + b) * c;
}

template<>
__forceinline__ __device__ half add_mul<half>(half a, half b, half c) {
    return __hmul(__hadd(a, b), c);
}

#ifdef PADDLE_WITH_HIP
template<>
__forceinline__ __device__ hip_bfloat16 add_mul<hip_bfloat16>(hip_bfloat16 a, hip_bfloat16 b, hip_bfloat16 c) {
    return (a + b) * c;
}
#else
template<>
__forceinline__ __device__ __nv_bfloat16 add_mul<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
  #if __CUDA_ARCH__ >= 800
    return __hmul(__hadd(a, b), c);
  #else
    return (static_cast<float>(a) + static_cast<float>(b)) * static_cast<float>(c);
  #endif
}
#endif


template <typename data_t>
__forceinline__ __device__ int8_t quant_helper(const data_t input,
                                               const float scale,
                                               const int round_type,
                                               const float max_bound,
                                               const float min_bound) {
  float quant_value = max_bound * scale * static_cast<float>(input);

  if (round_type == 0) {
    quant_value = static_cast<float>(roundWithTiesToEven(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<int8_t>(quant_value);
}

template <typename data_t>
__forceinline__ __device__ int8_t quant_helper(const data_t input,
                                               const data_t shift,
                                               const data_t smooth,
                                               const float scale,
                                               const int round_type,
                                               const float max_bound,
                                               const float min_bound) {
  auto smooth_out = add_mul(input, shift, smooth);
  float quant_value = max_bound * scale * static_cast<float>(smooth_out);

  if (round_type == 0) {
    quant_value = static_cast<float>(roundWithTiesToEven(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<int8_t>(quant_value);
}

template <typename data_t>
__global__ void QuantKernel(const data_t* input,
                            char4* output,
                            const float scale,
                            const int m,
                            const int n,
                            const int round_type,
                            const float max_bound,
                            const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;
  bool check = ((m_id < m) && (n_id < n));

  if (check) {
    char4 tmp;
    tmp.x = quant_helper(
        input[m_id * n + n_id], scale, round_type, max_bound, min_bound);
    tmp.y = quant_helper(
        input[m_id * n + n_id + 1], scale, round_type, max_bound, min_bound);
    tmp.z = quant_helper(
        input[m_id * n + n_id + 2], scale, round_type, max_bound, min_bound);
    tmp.w = quant_helper(
        input[m_id * n + n_id + 3], scale, round_type, max_bound, min_bound);

    output[(m_id * n + n_id) >> 2] = tmp;
  }
}

template <typename data_t>
__global__ void QuantKernel(const data_t* input,
                            const data_t* shift,
                            const data_t* smooth,
                            char4* output,
                            const float scale,
                            const int m,
                            const int n,
                            const int round_type,
                            const float max_bound,
                            const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;
  bool check = ((m_id < m) && (n_id < n));

  if (check) {
    char4 tmp;
    tmp.x = quant_helper(
        input[m_id * n + n_id], shift[n_id], smooth[n_id], scale, round_type, max_bound, min_bound);
    tmp.y = quant_helper(
        input[m_id * n + n_id + 1], shift[n_id + 1], smooth[n_id + 1], scale, round_type, max_bound, min_bound);
    tmp.z = quant_helper(
        input[m_id * n + n_id + 2], shift[n_id + 2], smooth[n_id + 2], scale, round_type, max_bound, min_bound);
    tmp.w = quant_helper(
        input[m_id * n + n_id + 3], shift[n_id + 3], smooth[n_id + 3], scale, round_type, max_bound, min_bound);

    output[(m_id * n + n_id) >> 2] = tmp;
  }
}



template <paddle::DataType D>
std::vector<paddle::Tensor> LaunchQuantInt8(const paddle::Tensor& input,
                                      const paddle::optional<paddle::Tensor>& shift,
                                      const paddle::optional<paddle::Tensor>& smooth,
                                      float scale,
                                      int32_t round_type,
                                      float max_bound,
                                      float min_bound) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    std::vector<int64_t> input_shape = input.shape();
    auto output=paddle::full(input_shape, -1, paddle::DataType::INT8, input.place());
    int m = input_shape[0];
    int n = input_shape[1];
#ifdef PADDLE_WITH_HIP
    dim3 grid(((n >> 2) + 63) / 64, (m + 7) / 8);
    dim3 block(64, 8);
#else
    dim3 grid((n >> 2 + 31) / 32, (m + 31) / 32);
    dim3 block(32, 32);
#endif
    auto stream = input.stream();
    if (shift && smooth) {
        QuantKernel<DataType_><<<grid, block, 0, stream>>>(reinterpret_cast<const DataType_*>(input.data<data_t>()),
                                                       reinterpret_cast<const DataType_*>(shift.get().data<data_t>()),
                                                       reinterpret_cast<const DataType_*>(smooth.get().data<data_t>()),
                                                       reinterpret_cast<char4*>(output.data<int8_t>()),  // NOLINT
                                                       scale,
                                                       m,
                                                       n,
                                                       round_type,
                                                       max_bound,
                                                       min_bound);
    } else {
        QuantKernel<DataType_><<<grid, block, 0, stream>>>(reinterpret_cast<const DataType_*>(input.data<data_t>()),
                                                       reinterpret_cast<char4*>(output.data<int8_t>()),  // NOLINT
                                                       scale,
                                                       m,
                                                       n,
                                                       round_type,
                                                       max_bound,
                                                       min_bound);
    }
    return {output};

}

std::vector<paddle::Tensor> QuantInt8(const paddle::Tensor& input,
                                      const paddle::optional<paddle::Tensor>& shift,
                                      const paddle::optional<paddle::Tensor>& smooth,
                                      float scale,
                                      int32_t round_type,
                                      float max_bound,
                                      float min_bound) {
    // printf("#### quant int8 scale:%f \n",scale);
    switch (input.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchQuantInt8<paddle::DataType::BFLOAT16>(
                input, shift, smooth, scale, round_type, max_bound, min_bound
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchQuantInt8<paddle::DataType::FLOAT16>(
                input, shift, smooth, scale, round_type, max_bound, min_bound
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchQuantInt8<paddle::DataType::FLOAT32>(
                input, shift, smooth, scale, round_type, max_bound, min_bound
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");
            break;
        }
    }
}



std::vector<std::vector<int64_t>> QuantInt8Shape(const std::vector<int64_t>& input_shape,
                                                const paddle::optional<std::vector<int64_t>>& shift_shape,
                                                const paddle::optional<std::vector<int64_t>>& smooth_shape
                                                ) {
    return {input_shape};
}

std::vector<paddle::DataType> QuantInt8Dtype(const paddle::DataType& input_dtype,
                                            const paddle::optional<paddle::DataType>& shift_dtype,
                                            const paddle::optional<paddle::DataType>& smooth_dtype
                                            ) {
    return {paddle::DataType::INT8};
}

PD_BUILD_OP(quant_int8)
    .Inputs({"intput", paddle::Optional("shift"),paddle::Optional("smooth") })
    .Outputs({"output"})
    .Attrs({"scale: float","round_type: int","max_bound: float", "min_bound: float"})
    .SetKernelFn(PD_KERNEL(QuantInt8))
    .SetInferShapeFn(PD_INFER_SHAPE(QuantInt8Shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(QuantInt8Dtype));