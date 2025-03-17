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

template <typename T>
__inline__ __device__ T WarpReduceAbsMax(T val, unsigned lane_mask) {
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1){
    val = max(val, __shfl_xor_sync(lane_mask, val, mask, 32));
  }
  return val;
}

template <typename InType, typename OutType, int GroupSize, int VecSize, bool transpose_scale>
__global__ void DynamicGroupQuantKernel(const InType* input,
                                   const int64_t numel,
                                   const int scale_rows,
                                   const int scale_cols,
                                   const float quant_max_bound,
                                   const float quant_min_bound,
                                   OutType* output,
                                   float* out_scale_data) {

  const int idx = (blockIdx.x * gridDim.x) + threadIdx.x * VecSize;

  if (idx >= numel) return;

  const int scale_row_idx = blockIdx.x / scale_cols;
  const int scale_col_idx = blockIdx.x % scale_cols;
  int scale_idx = 0;
  float abs_max_val = 0.000001;

  if constexpr (transpose_scale) {
    scale_idx = scale_col_idx * scale_rows + scale_row_idx;
  }else{
    scale_idx = scale_row_idx * scale_cols + scale_col_idx;
  }

  AlignedVector<InType, VecSize> in_vec;
  AlignedVector<OutType, VecSize> out_vec;

  Load<InType, VecSize>(input + idx, &in_vec);
  #pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    abs_max_val = max(abs_max_val, abs(static_cast<float>(in_vec[i])));
  }
  
  abs_max_val = WarpReduceAbsMax(abs_max_val, 0xffffffff);
  __syncthreads();
  float scale = __shfl_sync(0xffffffff, abs_max_val, 0) / quant_max_bound;
  
  #pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    float quant_val =  static_cast<float>(in_vec[i]) / scale;
    quant_val = quant_val > quant_max_bound ? quant_max_bound : quant_val;
    quant_val = quant_val < quant_min_bound ? quant_min_bound : quant_val;
    out_vec[i] = static_cast<OutType>(quant_val);
  }
  Store<OutType, VecSize>(out_vec, output + idx);
  if (threadIdx.x == 0) {
    out_scale_data[scale_idx] = scale;
  }
}

template <paddle::DataType InType, paddle::DataType OutType>
std::vector<paddle::Tensor> LaunchDynamicGroupQuantKernel(const paddle::Tensor& x,
                                                   const int group_size,
                                                   const bool transpose_scale,
                                                   const float quant_max_bound,
                                                   const float quant_min_bound) {
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
    std::vector<int64_t> scale_shape = x.shape();

    out = paddle::empty(out_shape, OutType, place);
    int64_t m = x.shape()[rank - 2];
    int64_t n = x.shape()[rank - 1];

    PD_CHECK(n % group_size == 0, "The last dimension[%d] of the input must be divisible by group_size %d.", n, group_size);
    int64_t scale_n = n / group_size;

    if(transpose_scale){
        scale_shape[rank - 2] = scale_n;
        scale_shape[rank - 1] = m;
    }else{
        scale_shape[rank - 1] = scale_n;
    }
    
    scale_out = paddle::empty(scale_shape, paddle::DataType::FLOAT32, place);

    int64_t numel = x.numel();
    constexpr int VecSize = 16 / sizeof(in_data_t_pd);
    int group_vec_num = group_size / VecSize;
    int thread_block_size = min(max(group_vec_num, 32), 1024)
    int thread_block_num = thread_block_size * VecSize;
    dim3 threadsPerBlock(thread_block_size, 1, 1);
    dim3 block_per_grid((numel + thread_block_num - 1) / thread_block_num, 1, 1);
    
    if(group_size == 128){
        DynamicGroupQuantKernel<InDataType, OutDataType, 128, VecSize, true><<<block_per_grid, threadsPerBlock, 0, stream>>>(reinterpret_cast<const InDataType*>(x.data<in_data_t_pd>()),
                        numel,
                        m,
                        n / 128,
                        quant_max_bound,
                        quant_min_bound,
                        reinterpret_cast<OutDataType*>(out.data<out_data_t_pd>()),
                        reinterpret_cast<float*>(scale_out.data<float>()));
    }else{
        PD_THROW("dynamic_group_quant's group_size only support 128.");
    }
    
    return {out, scale_out};
}
template <paddle::DataType InType>
std::vector<paddle::Tensor> LaunchDynamicGroupQuant(const paddle::Tensor& x,
                                             const int group_size,
                                             const bool transpose_scale,
                                             const float quant_max_bound,
                                             const float quant_min_bound) {

    if(fabs(quant_max_bound - 448.0f) < 0.000001){
        return LaunchDynamicGroupQuantKernel<InType, paddle::DataType::FLOAT8_E4M3FN>(x, group_size, transpose_scale, quant_max_bound, quant_min_bound);
    }else{
        PD_THROW("Only supported float8_e4m3fn quantization, please set quant_max_bound=448, quant_min_bound=-448.");
    }
    
}


std::vector<paddle::Tensor> DynamicGroupQuant(const paddle::Tensor& x,
                                        const int group_size,
                                        const bool transpose_scale,
                                        const float quant_max_bound,
                                        const float quant_min_bound) {
    if(x.dtype() == paddle::DataType::FLOAT32){
        return LaunchDynamicGroupQuant<paddle::DataType::FLOAT32>(x, group_size, transpose_scale, quant_max_bound, quant_min_bound);
    }else if(x.dtype() == paddle::DataType::FLOAT16){
        return LaunchDynamicGroupQuant<paddle::DataType::FLOAT16>(x, group_size, transpose_scale, quant_max_bound, quant_min_bound);
    }else if(x.dtype() == paddle::DataType::BFLOAT16){
        return LaunchDynamicGroupQuant<paddle::DataType::BFLOAT16>(x, group_size, transpose_scale, quant_max_bound, quant_min_bound);
    }else{
        PD_THROW("Unsupported data type.");
    }
}

std::vector<std::vector<int64_t>> DynamicGroupQuantInferShape(const std::vector<int64_t>& input_shape, const int group_size, const bool transpose_scale, const float quant_max_bound,const float quant_min_bound) {
    std::vector<int64_t> scale_shape = input_shape;
    int rank = input_shape.size();
    if(transpose_scale){
        scale_shape[rank - 1] = input_shape[rank - 2];
        scale_shape[rank - 2] = input_shape[rank - 1] / group_size;
    }else{
        scale_shape[rank - 1] = input_shape[rank - 1] / group_size;
    }
    return {input_shape, scale_shape};
}

std::vector<paddle::DataType> DynamicGroupQuantInferDtype(const paddle::DataType& input_dtype, const int group_size, const bool transpose_scale, const float quant_max_bound,const float quant_min_bound) {
    
    if(fabs(quant_max_bound - 448.0f) < 0.000001){
        return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32};
    }else{
        PD_THROW("Only supported attr of quant_max_bound in ['448.0'].");
    }
}

PD_BUILD_OP(dynamic_group_quant)
    .Inputs({"x"})
    .Outputs({"output", "scale"})
    .Attrs({"group_size: int",
            "transpose_scale: bool",
            "quant_max_bound: float",
            "quant_min_bound: float"})
    .SetKernelFn(PD_KERNEL(DynamicGroupQuant))
    .SetInferShapeFn(PD_INFER_SHAPE(DynamicGroupQuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DynamicGroupQuantInferDtype));