/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fastertransformer/cuda/cuda_int8_kernels.h"

namespace fastertransformer {

template <typename T>
__inline__ __device__ T gelu(T x) {
  float cdf =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));

  // NOTE: The precision of gelu with or without approximate formulation
  // may cause serious problem in some cases. If necessary, the following
  // comments can be opened to use the non-approximate formulation.
  // float cdf = 0.5f * (1.0f + erf((float)x / sqrt(2.0f)));
  return x * cdf;
}

template <>
__inline__ __device__ half2 gelu(half2 val) {
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp = __half22float2(val);

  tmp.x =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

template <typename T>
__global__ void transpose_cache_batch_major(T* k_dst,
                                            T* v_dst,
                                            const T* k_src,
                                            const T* v_src,
                                            const int* memory_seq_len,
                                            const int head_num,
                                            const int size_per_head,
                                            const int memory_max_seq_len,
                                            const int cache_max_len) {
  const int hidden_dim = head_num * size_per_head;
  const int x = (sizeof(T) == 4) ? 4 : 8;
  const int size_per_head_split = size_per_head / x;
  const int batch_id = blockIdx.x;
  const int seq_id = blockIdx.y;

  for (int id = threadIdx.x; id < head_num * size_per_head_split * x;
       id += blockDim.x) {
    int tmp_id = id;
    int x_id = tmp_id % x;
    tmp_id /= x;
    int size_id = tmp_id % size_per_head_split;
    tmp_id /= size_per_head_split;
    int head_id = tmp_id % head_num;

    int src_seq_id =
        (seq_id < memory_seq_len[batch_id])
            ? (seq_id + memory_max_seq_len - memory_seq_len[batch_id])
            : (seq_id - memory_seq_len[batch_id]);

    // key: [B, head_num, L, size_per_head / x, x] ->
    // [B, head_num, size_per_head / x, L, x]
    k_dst[batch_id * hidden_dim * cache_max_len +
          head_id * size_per_head * cache_max_len +
          size_id * cache_max_len * x + seq_id * x + x_id] =
        k_src[batch_id * hidden_dim * memory_max_seq_len +
              head_id * size_per_head * memory_max_seq_len +
              src_seq_id * size_per_head + size_id * x + x_id];

    // value: [B, head_num, L, size_per_head/x, x] ->
    // [B, head_num, L, size_per_head/x, x]
    v_dst[batch_id * hidden_dim * cache_max_len +
          head_id * size_per_head * cache_max_len + seq_id * size_per_head +
          size_id * x + x_id] =
        v_src[batch_id * hidden_dim * memory_max_seq_len +
              head_id * size_per_head * memory_max_seq_len +
              src_seq_id * size_per_head + size_id * x + x_id];
  }
}

template <typename T>
void transpose_cache_batch_major_kernelLauncher(T* k_dst,
                                                T* v_dst,
                                                const T* k_src,
                                                const T* v_src,
                                                const int* memory_seq_len,
                                                const int local_batch_size,
                                                const int memory_max_seq_len,
                                                const int cache_max_len,
                                                const int size_per_head,
                                                const int local_head_num,
                                                cudaStream_t stream) {
  constexpr int block_sz = 128;
  dim3 grid(local_batch_size, memory_max_seq_len);

  transpose_cache_batch_major<<<grid, block_sz, 0, stream>>>(k_dst,
                                                             v_dst,
                                                             k_src,
                                                             v_src,
                                                             memory_seq_len,
                                                             local_head_num,
                                                             size_per_head,
                                                             memory_max_seq_len,
                                                             cache_max_len);
}

template <typename T>
__global__ void quantize_channel_wise_kernel(
    int8_t* out, const T* in, const int n, T* scale, const T max_range) {
  int rowid = blockIdx.x;
  float local_i = -1e20f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    local_i = max(fabs((float)(in[rowid * n + i])), local_i);
  }
  float max_val = blockReduceMax<float>(local_i);

  if (threadIdx.x == 0) {
    scale[rowid] = (T)max_val;
  }
  __syncthreads();

  T scale_val = max_range / scale[rowid];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[rowid * n + i] = __float2int_rn((float)(in[rowid * n + i] * scale_val));
  }
}

template <typename T>
inline __host__ __device__ T inverse(T s) {
  T eps = static_cast<T>(1e-6);
  T one = static_cast<T>(1.0);
  return s <= static_cast<T>(1e-30) ? one / (s + eps) : one / s;
}

template <typename T>
__global__ void find_channel_abs_max_kernel_quant(const T* input,
                                                  const int n,
                                                  T* scale) {
  int tid = threadIdx.x;
  int channel_size = n;
  extern __shared__ float sh_max_data[];

  const T* in_c = input + blockIdx.x * channel_size;

  float local_max_data = float(0);

  for (int i = tid; i < channel_size; i += blockDim.x) {
    float tmp = fabs((float)in_c[i]);
    if (tmp > local_max_data) {
      local_max_data = tmp;
    }
  }
  sh_max_data[tid] = local_max_data;
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i && (sh_max_data[tid] < sh_max_data[tid + i])) {
      sh_max_data[tid] = sh_max_data[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    scale[blockIdx.x] = (T)sh_max_data[0];
  }
}

/*
// add bias to matrix of m * n, CUBLASLT_ORDER_COL32
// grid, thread = (m), (n/4)
// using char4
// for per-channel-quantization weight
// template <typename T>
__global__ void quantized_kernel(char4 *dst, const float4* src, const int
size_div_4, const int n, const float* scale_ptr)
{
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  if (tid < size_div_4){
    const float scale = __ldg(scale_ptr[tid / (n / 4)]);
    char4 tmp;
    const float4 floatTmp = __ldg(src + tid);
    tmp.x = __float2int_rn(floatTmp.x*scale);
    tmp.y = __float2int_rn(floatTmp.y*scale);
    tmp.z = __float2int_rn(floatTmp.z*scale);
    tmp.w = __float2int_rn(floatTmp.w*scale);
    dst[tid] = tmp;
  }
}

__global__ void quantized_kernel(char4 *dst, const half2* src, const int
size_div_4, const int n, const float* scale_ptr)
{
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  if (tid < size_div_4){
    const float scale = __ldg(scale_ptr[tid / (n / 4)]);
    char4 tmp;
    int src_id = tid << 1;

    const half2 half2Tmp = __ldg(src + src_id);
    tmp.x = __float2int_rn(static_cast<float>(half2Tmp.x) * scale);
    tmp.y = __float2int_rn(static_cast<float>(half2Tmp.y) * scale);

    const half2 half2Tmp2 = __ldg(src + src_id + 1);
    tmp.z = __float2int_rn(static_cast<float>(half2Tmp2.x) * scale);
    tmp.w = __float2int_rn(static_cast<float>(half2Tmp2.y) * scale);
    dst[tid] = tmp;
  }
}
*/

// transpose matrix & transfrom row-major to COL32
// input matrix is (n, m) row-major
// output matrix is (n, m) COL32
// m should be a mutiple of 32
// grid((m+31)/32, (n+31)/32)
// block(32, 32)
template <typename T>
__global__ void quantized_kernel(int8_t* dst,
                                 const T* src,
                                 const T* scale,
                                 const int m,  // hidden
                                 const int n,  // batch size
                                 const T max_range) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // hidden
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // batch size

  bool check = ((x < m) && (y < n));
  if (check) {
    // COL32_col = x >> 5 ; COL32_row = (y << 5) + (x & 31);
    // COL32_idx = (COL32_col << 5) * n + COL32_row = (x & 0xffffffe0)*n + (y <<
    // 5) + (x & 31)
    // float_to_int8_rn
    dst[(x & 0xffffffe0) * n + (y << 5) + (x & 31)] =
        __float2int_rn((float)(__ldg(src + y * m + x) / scale[y] * max_range));
  }
}


// transpose matrix & transfrom col-major to COL32 & quantize
// input matrix is (m, n) col-major
// output matrix is (n, m) COL32, using char4 to write out
// m should be a mutiple of 32
// grid((m+31)/32, (n+31)/32)
// block(8, 32)
template <typename T>
__global__ void transposeMatrix_COL32_quantize_kernel(char4* dst,
                                                      const T* src,
                                                      const int m,
                                                      const int n,
                                                      const T* scale_ptr,
                                                      const T max_range) {
  int x = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  const float scale = (float)(max_range / scale_ptr[y]);

  bool check = ((x < m) && (y < n));
  if (check) {
    char4 tmp4;
    tmp4.x = __float2int_rn(static_cast<float>(__ldg(src + y * m + x)) * scale);
    tmp4.y =
        __float2int_rn(static_cast<float>(__ldg(src + y * m + x + 1)) * scale);
    tmp4.z =
        __float2int_rn(static_cast<float>(__ldg(src + y * m + x + 2)) * scale);
    tmp4.w =
        __float2int_rn(static_cast<float>(__ldg(src + y * m + x + 3)) * scale);

    dst[((x & 0xffffffe0) * n + (y << 5) + (x & 31)) >> 2] = tmp4;
  }
}

template <typename T>
__global__ void channel_wise_quantize_kernel(char4* dst,
                                             const T* src,
                                             const int m,
                                             const int n,
                                             T* scale,
                                             const T max_range) {
  int bid = blockIdx.x;

  float local_i = -FLT_MAX;
  float val;

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    local_i = max(fabs((float)src[bid * n + tid]), local_i);
  }
  float max_val = blockReduceMax<float>(local_i);

  if (threadIdx.x == 0) {
    scale[bid] = (T)max_val;
  }
  __syncthreads();

  const float scale_val = (float)(max_range / scale[bid]);

  for (int tid = threadIdx.x; tid < n / 4; tid += blockDim.x) {
    char4 tmp4;
    int x = tid << 2;

    tmp4.x = __float2int_rn(static_cast<float>(src[bid * n + x]) * scale_val);
    tmp4.y =
        __float2int_rn(static_cast<float>(src[bid * n + x + 1]) * scale_val);
    tmp4.z =
        __float2int_rn(static_cast<float>(src[bid * n + x + 2]) * scale_val);
    tmp4.w =
        __float2int_rn(static_cast<float>(src[bid * n + x + 3]) * scale_val);

    dst[((x & 0xffffffe0) * m + (bid << 5) + (x & 31)) >> 2] = tmp4;
  }
}

template <typename T>
void channel_wise_quantize_kernelLauncher(const T* input,
                                          int8_t* ffn_quantize_input_buf_,
                                          T* scale,
                                          const int batch_size,
                                          const int hidden_units,
                                          cudaStream_t stream,
                                          bool use_COL32) {
  if (use_COL32) {
    dim3 grid(batch_size);
    dim3 block(min(1024, hidden_units));

    channel_wise_quantize_kernel<T><<<grid, block, 0, stream>>>(
        (char4*)ffn_quantize_input_buf_,
        input,
        batch_size,
        hidden_units,
        scale,
        (T)127.0f);
  } else {
    dim3 grid(batch_size);
    dim3 block(min(1024, hidden_units));

    quantize_channel_wise_kernel<<<grid, block, 0, stream>>>(
        ffn_quantize_input_buf_, input, hidden_units, scale, (T)127.0f);
  }
}

template <typename T>
__global__ void dequantized_kernel(const int32_t* input,
                                   const T* scale,
                                   const T* w_scale,
                                   const T max_range,
                                   int col,
                                   T* out) {
  int row_idx = blockIdx.x;
  for (int i = threadIdx.x; i < col; i += blockDim.x) {
    int idx = row_idx * col + i;
    out[idx] =
        (T)input[idx] * scale[row_idx] / max_range * w_scale[i] / max_range;
  }
}

template <typename T>
__global__ void dequantized_tc_kernel(T* dst,
                                      const int32_t* src,
                                      const T* scale,
                                      const T* w_scale,
                                      const int m,  // hidden
                                      const int n,  // batch size
                                      const T max_range) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // hidden
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // batch size

  bool check = ((x < m) && (y < n));
  if (check) {
    dst[y * m + x] = (T)(src[(x & 0xffffffe0) * n + (y << 5) + (x & 31)]) *
                     scale[y] / max_range * w_scale[x] / max_range;
  }
}

template <typename T>
void channel_wise_dequantize_kernelLauncher(const int32_t* input,
                                            const T* scale,
                                            const T* weight_scale,
                                            T* output,
                                            const int batch_size,
                                            const int hidden_units,
                                            cudaStream_t stream,
                                            bool use_COL32) {
  if (use_COL32) {
    dim3 grid((hidden_units + 31) / 32, (batch_size + 31) / 32);
    dim3 block(32, 32);

    dequantized_tc_kernel<<<grid, block, 0, stream>>>(output,
                                                      input,
                                                      scale,
                                                      weight_scale,
                                                      hidden_units,
                                                      batch_size,
                                                      (T)127.0f);
  } else {
    dim3 grid(batch_size);
    dim3 block(min(1024, hidden_units));
    dequantized_kernel<<<grid, block, 0, stream>>>(
        input, scale, weight_scale, (T)127.0f, hidden_units, output);
  }
}

template <typename T>
__global__ void qkv_dequantized_kernel(const int32_t* input,
                                       const T* scale,
                                       const T* qw_scale,
                                       const T* kw_scale,
                                       const T* vw_scale,
                                       const T max_range,
                                       int col,
                                       T* query,
                                       T* key,
                                       T* value) {
  int row_idx = blockIdx.x;
  for (int i = threadIdx.x; i < col; i += blockDim.x) {
    int idx = row_idx * col + i;
    query[idx] =
        (T)input[idx] * scale[row_idx] / max_range * qw_scale[i] / max_range;
    key[idx] = (T)input[gridDim.x * col + idx] * scale[row_idx] / max_range *
               kw_scale[i] / max_range;
    value[idx] = (T)input[2 * gridDim.x * col + idx] * scale[row_idx] /
                 max_range * vw_scale[i] / max_range;
  }
}

template <typename T>
__global__ void qkv_dequantized_tc_kernel(T* query,
                                          T* key,
                                          T* value,
                                          const int32_t* src,
                                          const T* scale,
                                          const T* qw_scale,
                                          const T* kw_scale,
                                          const T* vw_scale,
                                          const int m,  // hidden
                                          const int n,  // batch size
                                          const T max_range) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // hidden
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // batch size

  bool check = ((x < m) && (y < n));
  if (check) {
    query[y * m + x] = (T)(src[(x & 0xffffffe0) * n + (y << 5) + (x & 31)]) *
                       scale[y] / max_range * qw_scale[x] / max_range;
    key[y * m + x] =
        (T)(src[m * n + (x & 0xffffffe0) * n + (y << 5) + (x & 31)]) *
        scale[y] / max_range * kw_scale[x] / max_range;
    value[y * m + x] =
        (T)(src[2 * m * n + (x & 0xffffffe0) * n + (y << 5) + (x & 31)]) *
        scale[y] / max_range * vw_scale[x] / max_range;
  }
}

template <typename T>
void qkv_channel_wise_dequantize_kernelLauncher(const int32_t* input,
                                                const T* scale,
                                                const T* qw_scale,
                                                const T* kw_scale,
                                                const T* vw_scale,
                                                T* query,
                                                T* key,
                                                T* value,
                                                const int batch_size,
                                                const int hidden_units,
                                                cudaStream_t stream,
                                                bool use_COL32) {
  if (use_COL32) {
    dim3 grid((hidden_units + 31) / 32, (batch_size + 31) / 32);
    dim3 block(32, 32);

    qkv_dequantized_tc_kernel<<<grid, block, 0, stream>>>(query,
                                                          key,
                                                          value,
                                                          input,
                                                          scale,
                                                          qw_scale,
                                                          kw_scale,
                                                          vw_scale,
                                                          hidden_units,
                                                          batch_size,
                                                          (T)127.0f);

  } else {
    dim3 grid(batch_size);
    dim3 block(min(1024, hidden_units));

    qkv_dequantized_kernel<<<grid, block, 0, stream>>>(input,
                                                       scale,
                                                       qw_scale,
                                                       kw_scale,
                                                       vw_scale,
                                                       (T)127.0f,
                                                       hidden_units,
                                                       query,
                                                       key,
                                                       value);
  }
}

template <typename T>
__global__ void kv_dequantized_kernel(const int32_t* input,
                                      const T* scale,
                                      const T* kw_scale,
                                      const T* vw_scale,
                                      const T max_range,
                                      int col,
                                      T* key,
                                      T* value) {
  int row_idx = blockIdx.x;
  for (int i = threadIdx.x; i < col; i += blockDim.x) {
    int idx = row_idx * col + i;
    key[idx] =
        (T)input[idx] * scale[row_idx] / max_range * kw_scale[i] / max_range;
    value[idx] = (T)input[gridDim.x * col + idx] * scale[row_idx] / max_range *
                 vw_scale[i] / max_range;
  }
}

template <typename T>
__global__ void kv_dequantized_tc_kernel(T* key,
                                         T* value,
                                         const int32_t* src,
                                         const T* scale,
                                         const T* kw_scale,
                                         const T* vw_scale,
                                         const int m,  // hidden
                                         const int n,  // batch size
                                         const T max_range) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // hidden
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // batch size

  bool check = ((x < m) && (y < n));
  if (check) {
    key[y * m + x] = (T)(src[(x & 0xffffffe0) * n + (y << 5) + (x & 31)]) *
                     scale[y] / max_range * kw_scale[x] / max_range;
    value[y * m + x] =
        (T)(src[m * n + (x & 0xffffffe0) * n + (y << 5) + (x & 31)]) *
        scale[y] / max_range * vw_scale[x] / max_range;
  }
}

template <typename T>
void mem_kv_channel_wise_dequantize_kernelLauncher(const int32_t* input,
                                                   const T* scale,
                                                   const T* kw_scale,
                                                   const T* vw_scale,
                                                   T* key,
                                                   T* value,
                                                   const int m,
                                                   const int hidden_units,
                                                   cudaStream_t stream,
                                                   bool use_COL32) {
  if (use_COL32) {
    dim3 grid((hidden_units + 31) / 32, (m + 31) / 32);
    dim3 block(32, 32);

    kv_dequantized_tc_kernel<<<grid, block, 0, stream>>>(key,
                                                         value,
                                                         input,
                                                         scale,
                                                         kw_scale,
                                                         vw_scale,
                                                         hidden_units,
                                                         m,
                                                         (T)127.0f);

  } else {
    dim3 grid(m);
    dim3 block(min(1024, hidden_units));

    kv_dequantized_kernel<<<grid, block, 0, stream>>>(
        input, scale, kw_scale, vw_scale, (T)127.0f, hidden_units, key, value);
  }
}

// add bias to matrix of m * n, CUBLASLT_ORDER_COL32
// grid, thread = (m), (n)
template <typename T>
__global__ void dequant_add_bias_relu_quant_COL32_int32I_int8O(
    int8_t* out,
    const int32_t* input,
    const T* bias,
    T* ffn_inner,
    const int m,
    const int n,
    const T* w_scale,
    T* scale,
    const T max_range) {
  int bid = blockIdx.x;

  float local_i = -FLT_MAX;
  float val;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int outIdx = (tid & 0xffffffe0) * m + (bid << 5) + (tid & 31);

    val = (float)((T)input[outIdx] * scale[bid] / max_range * w_scale[tid] /
                      max_range +
                  bias[tid]);
    val = (val >= 0.0f) ? val : 0.0f;

    ffn_inner[outIdx] = (T)val;

    // no need fabs
    local_i = max(local_i, val);
  }
  float max_val = blockReduceMax<float>(local_i);

  if (threadIdx.x == 0) {
    scale[bid] = (T)max_val;
  }
  __syncthreads();

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int outIdx = (tid & 0xffffffe0) * m + (bid << 5) + (tid & 31);
    out[outIdx] =
        __float2int_rn((float)(ffn_inner[outIdx] / scale[bid] * max_range));
  }
}

// add bias to matrix of m * n, CUBLASLT_ORDER_COL32
// grid, thread = (m), (n)
template <typename T>
__global__ void dequant_add_bias_gelu_quant_COL32_int32I_int8O(
    int8_t* out,
    const int32_t* input,
    const T* bias,
    T* ffn_inner,
    const int m,
    const int n,
    const T* w_scale,
    T* scale,
    const T max_range) {
  int bid = blockIdx.x;

  float local_i = -FLT_MAX;
  float val;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int outIdx = (tid & 0xffffffe0) * m + (bid << 5) + (tid & 31);

    val = (float)((T)input[outIdx] * scale[bid] / max_range * w_scale[tid] /
                      max_range +
                  bias[tid]);
    val = gelu<float>(val);

    ffn_inner[outIdx] = (T)val;

    // no need fabs
    local_i = max(local_i, val);
  }
  float max_val = blockReduceMax<float>(local_i);

  if (threadIdx.x == 0) {
    scale[bid] = (T)max_val;
  }
  __syncthreads();

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int outIdx = (tid & 0xffffffe0) * m + (bid << 5) + (tid & 31);
    out[outIdx] =
        __float2int_rn((float)(ffn_inner[outIdx] / scale[bid] * max_range));
  }
}

template <typename T>
void dequant_add_bias_act_quant_COL32_int32I_int8O_kernelLauncher(
    int8_t* out,
    const int32_t* input,
    const T* bias,
    T* ffn_inner,
    const int batch_size,
    const int hidden_units,
    const T* weight_scale,
    T* scale,
    ActivationType activation_type,
    cudaStream_t stream) {
  dim3 grid(batch_size);
  dim3 block(min(hidden_units, 1024));

  if (activation_type == ActivationType::RELU) {
    dequant_add_bias_relu_quant_COL32_int32I_int8O<<<grid, block, 0, stream>>>(
        out,
        input,
        bias,
        ffn_inner,
        batch_size,
        hidden_units,
        weight_scale,
        scale,
        (T)127.0f);
  } else if (activation_type == ActivationType::GELU) {
    dequant_add_bias_gelu_quant_COL32_int32I_int8O<<<grid, block, 0, stream>>>(
        out,
        input,
        bias,
        ffn_inner,
        batch_size,
        hidden_units,
        weight_scale,
        scale,
        (T)127.0f);
  }
}

template void channel_wise_quantize_kernelLauncher(
    const float* input,
    int8_t* ffn_quantize_input_buf_,
    float* scale,
    const int batch_size,
    const int hidden_units,
    cudaStream_t stream,
    bool use_COL32);

template void channel_wise_quantize_kernelLauncher(
    const half* input,
    int8_t* ffn_quantize_input_buf_,
    half* scale,
    const int batch_size,
    const int hidden_units,
    cudaStream_t stream,
    bool use_COL32);

template void channel_wise_dequantize_kernelLauncher(const int32_t* input,
                                                     const float* scale,
                                                     const float* weight_scale,
                                                     float* output,
                                                     const int batch_size,
                                                     const int hidden_units,
                                                     cudaStream_t stream,
                                                     bool use_COL32);

template void channel_wise_dequantize_kernelLauncher(const int32_t* input,
                                                     const half* scale,
                                                     const half* weight_scale,
                                                     half* output,
                                                     const int batch_size,
                                                     const int hidden_units,
                                                     cudaStream_t stream,
                                                     bool use_COL32);

template void dequant_add_bias_act_quant_COL32_int32I_int8O_kernelLauncher(
    int8_t* out,
    const int32_t* input,
    const float* bias,
    float* ffn_inner,
    const int batch_size,
    const int hidden_units,
    const float* weight_scale,
    float* scale,
    ActivationType activation_type,
    cudaStream_t stream);

template void dequant_add_bias_act_quant_COL32_int32I_int8O_kernelLauncher(
    int8_t* out,
    const int32_t* input,
    const half* bias,
    half* ffn_inner,
    const int batch_size,
    const int hidden_units,
    const half* weight_scale,
    half* scale,
    ActivationType activation_type,
    cudaStream_t stream);

template void qkv_channel_wise_dequantize_kernelLauncher(const int32_t* input,
                                                         const float* scale,
                                                         const float* qw_scale,
                                                         const float* kw_scale,
                                                         const float* vw_scale,
                                                         float* query,
                                                         float* key,
                                                         float* value,
                                                         const int batch_size,
                                                         const int hidden_units,
                                                         cudaStream_t stream,
                                                         bool use_COL32);

template void qkv_channel_wise_dequantize_kernelLauncher(const int32_t* input,
                                                         const half* scale,
                                                         const half* qw_scale,
                                                         const half* kw_scale,
                                                         const half* vw_scale,
                                                         half* query,
                                                         half* key,
                                                         half* value,
                                                         const int batch_size,
                                                         const int hidden_units,
                                                         cudaStream_t stream,
                                                         bool use_COL32);

template void mem_kv_channel_wise_dequantize_kernelLauncher(
    const int32_t* input,
    const float* scale,
    const float* kw_scale,
    const float* vw_scale,
    float* key,
    float* value,
    const int m,
    const int hidden_units,
    cudaStream_t stream,
    bool use_COL32);

template void mem_kv_channel_wise_dequantize_kernelLauncher(
    const int32_t* input,
    const half* scale,
    const half* kw_scale,
    const half* vw_scale,
    half* key,
    half* value,
    const int m,
    const int hidden_units,
    cudaStream_t stream,
    bool use_COL32);

void transpose_general_kernelLauncher(T* dst,
                                      T* src,
                                      const int batch_size,
                                      const int seq_len,
                                      const int head_num,
                                      const int size_per_head,
                                      cudaStream_t stream) {
  dim3 grid, block;
  int grid_size = batch_size * head_num * seq_len;
  if (sizeof(T) == 2) {
    int seq_per_block = grid_size % 4 == 0 ? 4 : 1;
    grid.x = grid_size / seq_per_block;
    block.x = seq_per_block * size_per_head / 2;
    transpose<T><<<grid, block, 0, stream>>>(
        src, dst, batch_size, seq_len, head_num, size_per_head / 2);
  } else {
    const int seq_per_block = 1;
    grid.x = grid_size / seq_per_block;
    block.x = seq_per_block * size_per_head;
    transpose<T><<<grid, block, 0, stream>>>(
        src, dst, batch_size, seq_len, head_num, size_per_head);
  }
}

template void transpose_cache_batch_major_kernelLauncher(
    float* k_dst,
    float* v_dst,
    const float* k_src,
    const float* v_src,
    const int* memory_seq_len,
    const int local_batch_size,
    const int memory_max_seq_len,
    const int cache_max_len,
    const int size_per_head,
    const int local_head_num,
    cudaStream_t stream);

template void transpose_cache_batch_major_kernelLauncher(
    half* k_dst,
    half* v_dst,
    const half* k_src,
    const half* v_src,
    const int* memory_seq_len,
    const int local_batch_size,
    const int memory_max_seq_len,
    const int cache_max_len,
    const int size_per_head,
    const int local_head_num,
    cudaStream_t stream);

template void transpose_general_kernelLauncher(float* dst,
                                               float* src,
                                               const int batch_size,
                                               const int seq_len,
                                               const int head_num,
                                               const int size_per_head,
                                               cudaStream_t stream);

template void transpose_general_kernelLauncher(half* dst,
                                               half* src,
                                               const int batch_size,
                                               const int seq_len,
                                               const int head_num,
                                               const int size_per_head,
                                               cudaStream_t stream);
}
