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

#ifdef PADDLE_WITH_HIP
constexpr int32_t WARP_SIZE = 64; 
constexpr int32_t HALF_WARP = 32; 
#else
constexpr int32_t WARP_SIZE = 32; 
constexpr int32_t HALF_WARP = 16; 
#endif
constexpr float QUANT_MAX_BOUND = 127.0;
constexpr float QUANT_MIN_BOUND = -127.0;

template <typename T>
inline __device__ __host__ T div_up(T m, T n) {
  return (m + n - 1) / n;
}

template<typename T>
struct QuantFunc{
  __host__ __device__ uint8_t operator()(T x, float quant_scale) {
    float tmp = static_cast<float>(x) * quant_scale;
    tmp = round(tmp);
    if (tmp > QUANT_MAX_BOUND)
      tmp = QUANT_MAX_BOUND;
    else if (tmp < QUANT_MIN_BOUND)
      tmp = QUANT_MIN_BOUND;
    return static_cast<uint8_t>(tmp + 128.0f);;
  }
};

template<typename T>
struct MaxFunc{
  __device__ T operator()(T a, T b){
    return max(a, b); 
  }
}; 

template<>
struct MaxFunc<half>{
  __device__ half operator()(half a, half b){
#if (__CUDA_ARCH__ >= 800) || defined(PADDLE_WITH_HIP)
    return __hmax(a, b); 
#else
    return max(static_cast<float>(a), static_cast<float>(b));
#endif
  }
}; 

#ifdef PADDLE_WITH_HIP
template<>
struct MaxFunc<hip_bfloat16>{
  __device__ hip_bfloat16 operator()(hip_bfloat16 a, hip_bfloat16 b){
    return static_cast<hip_bfloat16>(max(static_cast<float>(a), static_cast<float>(b)));
  }
}; 
#else
template<>
struct MaxFunc<__nv_bfloat16>{
  __device__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b){
#if __CUDA_ARCH__ >= 800
    return __hmax(a, b); 
#else
    return max(static_cast<float>(a), static_cast<float>(b));
#endif
  }
}; 
#endif

template<typename T>
struct AbsFunc{
  __device__ T operator()(T x){
    return abs(x); 
  }
}; 

template<>
struct AbsFunc<half>{
  __device__ half operator()(half x){
  #if (__CUDA_ARCH__ >= 800) || defined(PADDLE_WITH_HIP)
    return __habs(x); 
  #else
    return abs(static_cast<float>(x));
  #endif
  }
}; 

#ifdef PADDLE_WITH_HIP
template<>
struct AbsFunc<hip_bfloat16>{
  __device__ hip_bfloat16 operator()(hip_bfloat16 x) {
    return static_cast<hip_bfloat16>(abs(static_cast<float>(x)));
  }
}; 
#else
template<>
struct AbsFunc<__nv_bfloat16>{
  __device__ __nv_bfloat16 operator()(__nv_bfloat16 x){
  #if __CUDA_ARCH__ >= 800
    return __habs(x); 
  #else
    return abs(static_cast<float>(x));
  #endif
  }
}; 
#endif

template <typename T, typename Vec, int VecSize>
__inline__ __device__ T LocalReduceMax(Vec& vec) {
  T local_max = static_cast<T>(0.0);
  #pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    local_max = vec[i] > local_max ?  vec[i] : local_max;
  }
  return local_max;
}

template <typename T>
__inline__ __device__ T WarpReduceAbsMax(T val, unsigned lane_mask) {
  #pragma unroll
  for (int mask = HALF_WARP; mask > 0; mask >>= 1){
#ifdef PADDLE_WITH_HIP
    val = MaxFunc<T>()(val, static_cast<T>(__shfl_xor(static_cast<float>(val), mask, WARP_SIZE)));
#else
    val = MaxFunc<T>()(val, __shfl_xor_sync(lane_mask, val, mask, WARP_SIZE));
#endif
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceAbsMax(T val, unsigned mask) {
    static __shared__ T smem[WARP_SIZE];
    int32_t lane_id = threadIdx.x % WARP_SIZE;
    int32_t warp_id = threadIdx.x / WARP_SIZE;

    val = WarpReduceAbsMax(val, mask);

    if (lane_id == 0) {
        smem[warp_id] = val;
    }

    __syncthreads();

    T abs_max_val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? smem[threadIdx.x] : static_cast<T>(0.0f);
    abs_max_val = WarpReduceAbsMax(abs_max_val, mask);
    return abs_max_val;
}


template<typename T, int VecSize>
__global__ void write_cache_k_int8_kernel(const T* k, const int64_t num_head, const int64_t dim_head, const int64_t  seq_len, int  max_seq_len, uint8_t* cache, float* quant_scales, float* dequant_scales) {
    const int bi = blockIdx.y;
    const int hi = blockIdx.x;

    using InVec = AlignedVector<T, VecSize>;
    using OutVec = AlignedVector<uint8_t, VecSize>;

    InVec in_vec;
    OutVec out_vec;
    InVec abs_max_vec;
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      abs_max_vec[i] = static_cast<T>(0.0f);
    }

    T local_abs_max;

    for (int idx = threadIdx.x * VecSize; idx < seq_len * dim_head; idx += blockDim.x * VecSize) {
        int linear_idx = bi * num_head * seq_len * dim_head + hi * seq_len * dim_head + idx;
        Load<T, VecSize>(k + linear_idx, &in_vec);
#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
            abs_max_vec[i] = MaxFunc<T>()(abs_max_vec[i], AbsFunc<T>()(in_vec[i]));
        }
    }

    local_abs_max = LocalReduceMax<T, InVec, VecSize>(abs_max_vec);
    T abs_max_val = BlockReduceAbsMax<T>(local_abs_max, 0xffffffff);

    __shared__ float quant_scale;
    if (threadIdx.x == 0) {
      quant_scale = 127.0f / static_cast<float>(abs_max_val);
    }

    __syncthreads();
    
    for (int idx = threadIdx.x * VecSize; idx < seq_len * dim_head; idx += blockDim.x * VecSize) {
        int linear_idx = bi * num_head * seq_len * dim_head + hi * seq_len * dim_head + idx;
        // [bsz, num_head, seq_len, dim_head/x, x]
        Load<T, VecSize>(k + linear_idx, &in_vec);
#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
            out_vec[i] = QuantFunc<T>()(in_vec[i], quant_scale);
        }
        int dim_head_div_x = dim_head / VecSize;
        int seq_id = idx / dim_head;
        int vec_id = threadIdx.x % dim_head_div_x;
        //  [bsz, num_head, dim_head/x, max_seq_len, x]
        Store<uint8_t>(out_vec, cache + bi * num_head * max_seq_len * dim_head + hi * max_seq_len * dim_head + vec_id * max_seq_len * VecSize + seq_id * VecSize);
    }

    if (threadIdx.x == 0) {
        quant_scales[bi * num_head + hi] =  quant_scale;
        dequant_scales[bi * num_head + hi] = 1.0f / quant_scale;
    }
}

template<typename T, int VecSize>
__global__ void write_cache_v_int8_kernel(const T* v, const int64_t num_head, const int64_t dim_head, const int64_t  seq_len, int max_seq_len, uint8_t* cache, float* quant_scales, float* dequant_scales) {
    const int bi = blockIdx.y;
    const int hi = blockIdx.x;

    using InVec = AlignedVector<T, VecSize>;
    using OutVec = AlignedVector<uint8_t, VecSize>;

    InVec in_vec;
    OutVec out_vec;
    InVec abs_max_vec;
  #pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      abs_max_vec[i] = static_cast<T>(0.0f);
    }

    T local_abs_max;

    for (int idx = threadIdx.x * VecSize; idx < seq_len * dim_head; idx += blockDim.x * VecSize) {
        int linear_idx = bi * num_head * seq_len * dim_head + hi * seq_len * dim_head + idx;
        Load<T, VecSize>(v + linear_idx, &in_vec);
#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
            abs_max_vec[i] = MaxFunc<T>()(abs_max_vec[i], AbsFunc<T>()(in_vec[i]));
        }
    }

    local_abs_max = LocalReduceMax<T, InVec, VecSize>(abs_max_vec);
    T abs_max_val = BlockReduceAbsMax<T>(local_abs_max, 0xffffffff);

    __shared__ float quant_scale;
    if (threadIdx.x == 0) {
      quant_scale = 127.0f / static_cast<float>(abs_max_val);
    }

    __syncthreads();
    for (int idx = threadIdx.x * VecSize; idx < seq_len * dim_head; idx += blockDim.x * VecSize) {
        int linear_idx = bi * num_head * seq_len * dim_head + hi * seq_len * dim_head + idx;
        // [bsz, num_head, seq_len, dim_head/x, x]
        Load<T, VecSize>(v + linear_idx, &in_vec);
#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
            out_vec[i] = QuantFunc<T>()(in_vec[i], quant_scale);
        }
        int dim_head_div_x = dim_head / VecSize;
        int seq_id = idx / dim_head;
        int vec_id = threadIdx.x % dim_head_div_x;
        //  [bsz, num_head, max_seq_len, dim_head/x, x]
        Store<uint8_t>(out_vec, cache + bi * num_head * max_seq_len * dim_head + hi * max_seq_len * dim_head + seq_id * dim_head + vec_id * VecSize);
    }

    if (threadIdx.x == 0) {
        quant_scales[bi * num_head + hi] =  quant_scale;
        dequant_scales[bi * num_head + hi] = 1.0f / quant_scale;
    }
}

template <paddle::DataType D>
void LaunchWriteInt8CacheKV(const paddle::Tensor& input_k, 
                        const paddle::Tensor& input_v, 
                        const paddle::Tensor& cache_kv,
                        const paddle::Tensor& k_quant_scales,
                        const paddle::Tensor& v_quant_scales,
                        const paddle::Tensor& k_dequant_scales,
                        const paddle::Tensor& v_dequant_scales
                        ) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    const int64_t bsz = input_k.shape()[0];
    const int64_t seq_len = input_k.shape()[2]; 
    const int64_t cache_bsz = cache_kv.shape()[1]; 
    const int64_t num_head = cache_kv.shape()[2]; 
    const int64_t dim_head = cache_kv.shape()[4]; 

    auto cache_kv_out = paddle::full({1}, -1, paddle::DataType::UINT8, cache_kv.place());

    const DataType_ *k_ptr = reinterpret_cast<const DataType_*>(input_k.data<data_t>());
    const DataType_ *v_ptr = reinterpret_cast<const DataType_*>(input_v.data<data_t>());

    // [2, bsz, num_head, max_seq_len, head_dim]
    int max_seq_len = cache_kv.shape()[3];
    uint8_t *cache_kv_data = reinterpret_cast<uint8_t*>(const_cast<uint8_t*>(cache_kv.data<uint8_t>()));
    
    float* k_quant_scales_data = const_cast<float*>(k_quant_scales.data<float>());
    float* k_dequant_scales_data = const_cast<float*>(k_dequant_scales.data<float>());

    float* v_quant_scales_data = const_cast<float*>(v_quant_scales.data<float>());
    float* v_dequant_scales_data = const_cast<float*>(v_dequant_scales.data<float>());

    int64_t cache_k_size = cache_bsz * num_head * max_seq_len * dim_head;

    uint8_t *cache_k_ptr = cache_kv_data;
    uint8_t *cache_v_ptr = cache_kv_data + cache_k_size;

    constexpr int block_sz = 512;
    constexpr int VecSize = VEC_16B / sizeof(DataType_);

    assert(dim_head % VecSize == 0);
    // PD_CHECK((dim_head % x) == 0, "PD_CHECK returns ", false, ", dim_head must be divisible by vec_size.");

    dim3 grid(num_head, bsz);

    // transpose [bsz, num_head, seq_len, dim_head/x, x]->
    // [bsz, num_head, dim_head/x, max_seq_len, x]
    write_cache_k_int8_kernel<DataType_, VecSize><<<grid, block_sz, 0, input_k.stream()>>>(
        k_ptr,  num_head, dim_head, seq_len, max_seq_len, cache_k_ptr, k_quant_scales_data, k_dequant_scales_data);


    // copy [bsz, num_head, seq_len, dim_head/x, x]->
    // [bsz, num_head, max_seq_len, dim_head/x, x]
    write_cache_v_int8_kernel<DataType_, VecSize><<<grid, block_sz, 0, input_k.stream()>>>(
        v_ptr,  num_head, dim_head, seq_len, max_seq_len, cache_v_ptr, v_quant_scales_data, v_dequant_scales_data);

}


void WriteInt8CacheKV(const paddle::Tensor& input_k,
                  const paddle::Tensor& input_v,
                  const paddle::Tensor& cache_kv,
                  const paddle::Tensor& k_quant_scales,
                  const paddle::Tensor& v_quant_scales,
                  const paddle::Tensor& k_dequant_scales,
                  const paddle::Tensor& v_dequant_scales) {
    switch (input_k.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchWriteInt8CacheKV<paddle::DataType::BFLOAT16>(
                input_k, input_v, cache_kv, k_quant_scales, v_quant_scales, k_dequant_scales, v_dequant_scales
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchWriteInt8CacheKV<paddle::DataType::FLOAT16>(
                input_k, input_v, cache_kv, k_quant_scales, v_quant_scales, k_dequant_scales, v_dequant_scales
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchWriteInt8CacheKV<paddle::DataType::FLOAT32>(
                input_k, input_v, cache_kv, k_quant_scales, v_quant_scales, k_dequant_scales, v_dequant_scales
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

PD_BUILD_OP(write_int8_cache_kv)
    .Inputs({"input_k", "input_v", "cache_kv", "k_quant_scales", "v_quant_scales", "q_dequant_scales", "v_dequant_scales"})
    .Outputs({"cache_kv_out"})
    .SetInplaceMap({{"cache_kv", "cache_kv_out"}})
    .SetKernelFn(PD_KERNEL(WriteInt8CacheKV));