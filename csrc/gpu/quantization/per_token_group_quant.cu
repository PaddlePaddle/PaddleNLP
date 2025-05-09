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


__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = 0xffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

template <typename InType, typename OutType>
__global__ void PerTokenGroupQuantKernel(
    const InType* __restrict__ input,
    OutType* __restrict__ output_q,
    float* __restrict__ output_s,
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const float eps,
    const float quant_min_bound,
    const float quant_max_bound,
    bool transpose_scale = false,
    const int scale_num_rows = 0,
    const int scale_stride = 0) {
  const int threads_per_group = 16;
  const int local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int block_group_id = blockIdx.x * groups_per_block;
  const int global_group_id = block_group_id + local_group_id;
  const int block_group_offset = global_group_id * group_size;

  float local_absmax = eps;

  const InType* group_input = input + block_group_offset;
  OutType* group_output = output_q + block_group_offset;
  float* scale_output;

  if (transpose_scale) {
    const int row_idx = global_group_id / scale_num_rows;
    const int col_idx = global_group_id % scale_num_rows;
    scale_output = output_s + (col_idx * scale_stride + row_idx);
  } else {
    scale_output = output_s + global_group_id;
  }

  constexpr uint32_t vec_size = 16 / sizeof(InType);
  using vec_t = AlignedVector<InType, vec_size>;

  const int32_t num_vec_elems = group_size / vec_size;

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t input_vec;
    Load<InType, vec_size>(group_input + i * vec_size, &input_vec);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      float abs_val = fabsf(val);
      local_absmax = fmaxf(local_absmax, abs_val);
    }
  }

  local_absmax = GroupReduceMax(local_absmax, lane_id);

  const float y_s = local_absmax / quant_max_bound;

  if (lane_id == 0) {
    *scale_output = y_s;
  }

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t input_vec;
    Load<InType, vec_size>(group_input + i * vec_size, &input_vec);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      float q_val = fminf(fmaxf(val / y_s, quant_min_bound), quant_max_bound);
      group_output[i * vec_size + j] = static_cast<OutType>(q_val);
    }
  }
}


template <paddle::DataType InType, paddle::DataType OutType>
std::vector<paddle::Tensor> LaunchPerTokenGroupQuantKernel(const paddle::Tensor& x,
                                                   const int group_size,
                                                   const bool transpose_scale,
                                                   const float quant_max_bound,
                                                   const float quant_min_bound) {
    typedef PDTraits<InType> in_traits;
    typedef typename in_traits::DataType InDataType;
    typedef typename in_traits::data_t in_data_t;

    paddle::Tensor out;
    paddle::Tensor scale_out;
    auto place = x.place();
    cudaStream_t stream = x.stream();
    int rank = x.dims().size();
    std::vector<int64_t> out_shape = x.shape();
    std::vector<int64_t> scale_shape = x.shape();
    int64_t m = x.shape()[rank - 2];
    int64_t k = x.shape()[rank - 1];
    PD_CHECK(k % group_size == 0);
    int64_t scale_k = k / group_size;

    out = paddle::empty(out_shape, OutType, place);
    if(transpose_scale){
        scale_shape[rank - 2] = scale_k;
        scale_shape[rank - 1] = m;
    }else{
        scale_shape[rank - 1] = scale_k;
    }
    scale_out = paddle::empty(scale_shape, paddle::DataType::FLOAT32, place);
    int64_t numel = x.numel();
    const int num_groups = numel / group_size;
    constexpr int THREADS_PER_GROUP = 16;
    int groups_per_block = 1;

    if (num_groups % 16 == 0) {
        groups_per_block = 16;
    } else if (num_groups % 8 == 0) {
        groups_per_block = 8;
    } else if (num_groups % 4 == 0) {
        groups_per_block = 4;
    } else if (num_groups % 2 == 0) {
        groups_per_block = 2;
    }

    const int num_blocks = num_groups / groups_per_block;
    const int num_threads = groups_per_block * THREADS_PER_GROUP;

    int scale_num_rows = 0;
    int scale_stride = 0;
    if (transpose_scale){
        scale_num_rows = m;
        scale_stride = scale_k;
    }
    
    dim3 grid(num_blocks);
    dim3 block(num_threads);

    typedef PDTraits<OutType> out_traits;
    typedef typename out_traits::DataType OutDataType;
    typedef typename out_traits::data_t out_data_t;
    float eps = 0.000001f;

    PerTokenGroupQuantKernel<InDataType, OutDataType><<<grid, block, 0, stream>>>(reinterpret_cast<const InDataType*>(x.data<in_data_t>()),
    reinterpret_cast<OutDataType*>(out.data<out_data_t>()),
    reinterpret_cast<float*>(scale_out.data<float>()),
    group_size,
    num_groups,
    groups_per_block,
    eps,
    quant_min_bound,
    quant_max_bound,
    transpose_scale,
    scale_num_rows,
    scale_stride);
    
    return {out, scale_out};
}

template <paddle::DataType InType>
std::vector<paddle::Tensor> LaunchPerTokenGroupQuant(const paddle::Tensor& x,
                                             const int group_size,
                                             const bool transpose_scale,
                                             const float quant_max_bound,
                                             const float quant_min_bound) {

    if(fabs(quant_max_bound - 448.0f) < 0.000001){
        return LaunchPerTokenGroupQuantKernel<InType, paddle::DataType::FLOAT8_E4M3FN>(x, group_size, transpose_scale, quant_max_bound, quant_min_bound);
    }else if(fabs(quant_max_bound - 127.0f) < 0.000001){
        return LaunchPerTokenGroupQuantKernel<InType, paddle::DataType::INT8>(x, group_size, transpose_scale, quant_max_bound, quant_min_bound);
    }else{
        PD_THROW("Only supported float8_e4m3fn and int8 quantization.");
    }
    
}


std::vector<paddle::Tensor> PerTokenGroupQuant(const paddle::Tensor& x,
                                        const int group_size,
                                        const bool transpose_scale,
                                        const float quant_max_bound,
                                        const float quant_min_bound) {
    if(x.dtype() == paddle::DataType::FLOAT32){
        return LaunchPerTokenGroupQuant<paddle::DataType::FLOAT32>(x, group_size, transpose_scale, quant_max_bound, quant_min_bound);
    }else if(x.dtype() == paddle::DataType::FLOAT16){
        return LaunchPerTokenGroupQuant<paddle::DataType::FLOAT16>(x, group_size, transpose_scale, quant_max_bound, quant_min_bound);
    }else if(x.dtype() == paddle::DataType::BFLOAT16){
        return LaunchPerTokenGroupQuant<paddle::DataType::BFLOAT16>(x, group_size, transpose_scale, quant_max_bound, quant_min_bound);
    }else{
        PD_THROW("Unsupported data type.");
    }
}

std::vector<std::vector<int64_t>> PerTokenGroupQuantInferShape(const std::vector<int64_t>& input_shape, const int group_size, const bool transpose_scale, const float quant_max_bound,const float quant_min_bound) {
    std::vector<int64_t> scale_shape = input_shape;
    int rank = input_shape.size();
    PD_CHECK(scale_shape[rank-1] % group_size == 0);
    if(transpose_scale){
        scale_shape[rank - 1] = input_shape[rank - 2];
        scale_shape[rank - 2] = input_shape[rank - 1] / group_size;
    }else{
        scale_shape[rank - 1] = input_shape[rank - 1] / group_size;
    }
    return {input_shape, scale_shape};
}

std::vector<paddle::DataType> PerTokenGroupQuantInferDtype(const paddle::DataType& input_dtype, const int group_size, const bool transpose_scale, const float quant_max_bound,const float quant_min_bound) {
    
    if(fabs(quant_max_bound - 448.0f) < 0.000001){
        return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32};
    }else if(fabs(quant_max_bound - 127.0f) < 0.000001){
        return {paddle::DataType::INT8, paddle::DataType::FLOAT32};
    }else{
        PD_THROW("Only supported attr of quant_max_bound in [448.0, 127.0].");
    }
}

PD_BUILD_OP(per_token_group_quant)
    .Inputs({"x"})
    .Outputs({"output", "scale"})
    .Attrs({"group_size: int",
            "transpose_scale: bool",
            "quant_max_bound: float",
            "quant_min_bound: float"})
    .SetKernelFn(PD_KERNEL(PerTokenGroupQuant))
    .SetInferShapeFn(PD_INFER_SHAPE(PerTokenGroupQuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PerTokenGroupQuantInferDtype));