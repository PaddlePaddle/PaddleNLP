// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include<string.h>
#include <cuda_runtime.h>

template <typename InType>
__global__ void
per_tensor_absmax_kernel(const InType* __restrict__ input, float* __restrict__ output_s, const int64_t num_elements) {
    float max_value = 0.0f;
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = blockDim.x * gridDim.x;

    constexpr uint32_t vec_size = 16 / sizeof(InType);
    using in_vec_t = AlignedVector<InType, vec_size>;

    const int32_t num_vec_elems = num_elements / vec_size;

    for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
        in_vec_t input_vec;
        Load<InType, vec_size>(input + i * vec_size, &input_vec);

        #pragma unroll
        for (uint32_t j = 0; j < vec_size; ++j) {
            float val = static_cast<float>(input_vec[j]);
            max_value = fmaxf(max_value, fabsf(val));
        }
    }

    const int32_t remaining_start = num_vec_elems * vec_size;
    for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
        float val = static_cast<float>(input[idx]);
        max_value = fmaxf(max_value, fabsf(val));
    }

    max_value = blockReduceMax(max_value);

    if (tid == 0) {
        atomicMaxFloat(output_s, max_value / 448);
    }
}

template <typename InType, typename OutType>
__global__ void per_tensor_quant_fp8_kernel(
    const InType* __restrict__ input,
    OutType* __restrict__ output,
    const float* __restrict__ scale,
    const int64_t num_elements) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_size = blockDim.x * gridDim.x;
  const float scale_val = 1.0f / (*scale);

  constexpr uint32_t vec_size = 16 / sizeof(InType);
  using in_vec_t = AlignedVector<InType, vec_size>;
  using out_vec_t = AlignedVector<OutType, vec_size>;
  in_vec_t input_vec;
  out_vec_t output_vec;
  const int32_t num_vec_elems = num_elements / vec_size;

  for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
    Load<InType, vec_size>(input + i * vec_size, &input_vec);
    #pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = fmax(fmin(static_cast<float>(input_vec[j]) * scale_val, 448), -448);
      output_vec[j] = static_cast<OutType>(val);
    }
    Store<OutType, vec_size>(output_vec, output + i * vec_size);
  }

  const int32_t remaining_start = num_vec_elems * vec_size;
  for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
    float val = fmax(-448, fmin(static_cast<float>(input[idx]) * scale_val, 448));
    output[idx] = static_cast<OutType>(val);
  }
}


template <paddle::DataType InType, paddle::DataType OutType>
std::vector<paddle::Tensor> LaunchPerTensorQuantFp8Kernel(const paddle::Tensor& x, const paddle::optional<paddle::Tensor>& scale) {
    typedef PDTraits<InType> in_traits;
    typedef typename in_traits::DataType InDataType;
    typedef typename in_traits::data_t in_data_t_pd;

    typedef PDTraits<OutType> out_traits;
    typedef typename out_traits::DataType OutDataType;
    typedef typename out_traits::data_t out_data_t_pd;

    paddle::Tensor out;
    paddle::Tensor scale_out;
    auto place = x.place();
    cudaStream_t stream = x.stream();
    int rank = x.dims().size();
    std::vector<int64_t> out_shape = x.shape();
    std::vector<int64_t> scale_shape = {1};

    out = paddle::empty(out_shape, OutType, place);
    if(scale){
        scale_out = scale.get();
    }else{
        scale_out = paddle::empty(scale_shape, paddle::DataType::FLOAT32, place);
    }

    const int block_size = 256;
    const int64_t num_elements = x.numel();
    const int64_t num_blocks = min((num_elements + block_size - 1) / block_size, static_cast<int64_t>(1024));

    dim3 grid(num_blocks);
    dim3 block(block_size);
    if(scale){
        per_tensor_absmax_kernel<InDataType><<<grid, block, 0, stream>>>(
          reinterpret_cast<const InDataType*>(x.data<in_data_t_pd>()), reinterpret_cast<float*>(scale_out.data<float>()), num_elements);
    }

    per_tensor_quant_fp8_kernel<InDataType, OutDataType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const InDataType*>(x.data<in_data_t_pd>()),
        reinterpret_cast<OutDataType*>(out.data<out_data_t_pd>()),
        reinterpret_cast<float*>(scale_out.data<float>()),
        num_elements);
    return {out, scale_out};
}
template <paddle::DataType InType>
std::vector<paddle::Tensor> LaunchPerTensorQuantFp8(const paddle::Tensor& x, const paddle::optional<paddle::Tensor>& scale) {
    return LaunchPerTensorQuantFp8Kernel<InType, paddle::DataType::FLOAT8_E4M3FN>(x, scale);
}


std::vector<paddle::Tensor> PerTensorQuantFp8(const paddle::Tensor& x, const paddle::optional<paddle::Tensor>& scale) {
    if(x.dtype() == paddle::DataType::FLOAT32){
        return LaunchPerTensorQuantFp8<paddle::DataType::FLOAT32>(x, scale);
    }else if(x.dtype() == paddle::DataType::FLOAT16){
        return LaunchPerTensorQuantFp8<paddle::DataType::FLOAT16>(x, scale);
    }else if(x.dtype() == paddle::DataType::BFLOAT16){
        return LaunchPerTensorQuantFp8<paddle::DataType::BFLOAT16>(x, scale);
    }else{
        PD_THROW("Unsupported data type.");
    }
}

std::vector<std::vector<int64_t>> PerTensorQuantFp8InferShape(const std::vector<int64_t>& input_shape, const paddle::optional<std::vector<int64_t>>& scale_shape) {
    std::vector<int64_t> scale_out_shape = {1};
    if(scale_shape){
        return {input_shape, scale_shape.get()};
    }
    return {input_shape, scale_out_shape};
}

std::vector<paddle::DataType> PerTensorQuantFp8InferDtype(const paddle::DataType& input_dtype, const paddle::optional<paddle::DataType>& scale_dtype) {
    return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32};
}

PD_BUILD_OP(per_tensor_quant_fp8)
    .Inputs({"x", paddle::Optional("scale")})
    .Outputs({"output", "scale_out"})
    .SetKernelFn(PD_KERNEL(PerTensorQuantFp8))
    .SetInferShapeFn(PD_INFER_SHAPE(PerTensorQuantFp8InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PerTensorQuantFp8InferDtype));