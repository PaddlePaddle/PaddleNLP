/*
 * Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


namespace fastertransformer {

template <typename T>
__global__ void quantize_channel_wise_kernel(
    int8_t* out, const T* in, const int n, T* scale, const float max_range) {
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

  T scale_val = max_range / (float)scale[rowid];
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
                                 const float max_range) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // hidden
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // batch size

  bool check = ((x < m) && (y < n));
  if (check) {
    // COL32_col = x >> 5 ; COL32_row = (y << 5) + (x & 31);
    // COL32_idx = (COL32_col << 5) * n + COL32_row = (x & 0xffffffe0)*n + (y <<
    // 5) + (x & 31)
    // float_to_int8_rn
    dst[(x & 0xffffffe0) * n + (y << 5) + (x & 31)] = __float2int_rn(
        (float)__ldg(src + y * m + x) / (float)scale[y] * max_range);
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
                                                      const float max_range) {
  int x = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  const float scale = max_range / (float)scale_ptr[y];

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
                                             const float max_range) {
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

  const float scale_val = max_range / (float)scale[bid];

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
        127.0f);
  } else {
    dim3 grid(batch_size);
    dim3 block(min(1024, hidden_units));

    quantize_channel_wise_kernel<<<grid, block, 0, stream>>>(
        ffn_quantize_input_buf_, input, hidden_units, scale, 127.0f);
  }
}

template <typename T>
__global__ void dequantized_kernel(const int32_t* input,
                                   const T* scale,
                                   const T* w_scale,
                                   const float max_range,
                                   int col,
                                   T* out) {
  int row_idx = blockIdx.x;
  for (int i = threadIdx.x; i < col; i += blockDim.x) {
    int idx = row_idx * col + i;
    out[idx] = (T)((float)input[idx] * (float)scale[row_idx] / max_range *
                   (float)w_scale[i] / max_range);
  }
}

template <typename T>
__global__ void dequantized_tc_kernel(T* dst,
                                      const int32_t* src,
                                      const T* scale,
                                      const T* w_scale,
                                      const int m,  // hidden
                                      const int n,  // batch size
                                      const float max_range) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // hidden
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // batch size

  bool check = ((x < m) && (y < n));
  if (check) {
    dst[y * m + x] =
        (T)((float)(src[(x & 0xffffffe0) * n + (y << 5) + (x & 31)]) *
            (float)scale[y] / max_range * (float)w_scale[x] / max_range);
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

    dequantized_tc_kernel<<<grid, block, 0, stream>>>(
        output, input, scale, weight_scale, hidden_units, batch_size, 127.0f);
  } else {
    dim3 grid(batch_size);
    dim3 block(min(1024, hidden_units));
    dequantized_kernel<<<grid, block, 0, stream>>>(
        input, scale, weight_scale, 127.0f, hidden_units, output);
  }
}

template <typename T>
__global__ void qkv_dequantized_kernel(const int32_t* input,
                                       const T* scale,
                                       const T* qw_scale,
                                       const T* kw_scale,
                                       const T* vw_scale,
                                       const float max_range,
                                       int col,
                                       T* query,
                                       T* key,
                                       T* value) {
  int row_idx = blockIdx.x;
  for (int i = threadIdx.x; i < col; i += blockDim.x) {
    int idx = row_idx * col + i;
    query[idx] = (T)((float)input[idx] * (float)scale[row_idx] / max_range *
                     (float)qw_scale[i] / max_range);
    key[idx] = (T)((float)input[gridDim.x * col + idx] * (float)scale[row_idx] /
                   max_range * (float)kw_scale[i] / max_range);
    value[idx] =
        (T)((float)input[2 * gridDim.x * col + idx] * (float)scale[row_idx] /
            max_range * (float)vw_scale[i] / max_range);
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
                                          const float max_range) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // hidden
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // batch size

  bool check = ((x < m) && (y < n));
  if (check) {
    query[y * m + x] =
        (T)((float)(src[(x & 0xffffffe0) * n + (y << 5) + (x & 31)]) *
            (float)scale[y] / max_range * (float)qw_scale[x] / max_range);
    key[y * m + x] =
        (T)((float)(src[m * n + (x & 0xffffffe0) * n + (y << 5) + (x & 31)]) *
            (float)scale[y] / max_range * (float)kw_scale[x] / max_range);
    value[y * m + x] = (T)(
        (float)(src[2 * m * n + (x & 0xffffffe0) * n + (y << 5) + (x & 31)]) *
        (float)scale[y] / max_range * (float)vw_scale[x] / max_range);
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
                                                          127.0f);

  } else {
    dim3 grid(batch_size);
    dim3 block(min(1024, hidden_units));

    qkv_dequantized_kernel<<<grid, block, 0, stream>>>(input,
                                                       scale,
                                                       qw_scale,
                                                       kw_scale,
                                                       vw_scale,
                                                       127.0f,
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
                                      const float max_range,
                                      int col,
                                      T* key,
                                      T* value) {
  int row_idx = blockIdx.x;
  for (int i = threadIdx.x; i < col; i += blockDim.x) {
    int idx = row_idx * col + i;
    key[idx] = (T)((float)input[idx] * (float)scale[row_idx] / max_range *
                   (float)kw_scale[i] / max_range);
    value[idx] =
        (T)((float)input[gridDim.x * col + idx] * (float)scale[row_idx] /
            max_range * (float)vw_scale[i] / max_range);
  }
}

template <typename T>
__global__ void kv_dequantized_COL32_kernel(T* key,
                                            T* value,
                                            const int32_t* src,
                                            const T* scale,
                                            const T* kw_scale,
                                            const T* vw_scale,
                                            const int m,  // hidden
                                            const int n,  // batch size
                                            const float max_range) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // hidden
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // batch size

  bool check = ((x < m) && (y < n));
  if (check) {
    key[y * m + x] =
        (T)((float)(src[(x & 0xffffffe0) * n + (y << 5) + (x & 31)]) *
            (float)scale[y] / max_range * (float)kw_scale[x] / max_range);
    value[y * m + x] =
        (T)((float)(src[m * n + (x & 0xffffffe0) * n + (y << 5) + (x & 31)]) *
            (float)scale[y] / max_range * (float)vw_scale[x] / max_range);
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

    kv_dequantized_COL32_kernel<<<grid, block, 0, stream>>>(
        key, value, input, scale, kw_scale, vw_scale, hidden_units, m, 127.0f);

  } else {
    dim3 grid(m);
    dim3 block(min(1024, hidden_units));

    kv_dequantized_kernel<<<grid, block, 0, stream>>>(
        input, scale, kw_scale, vw_scale, 127.0f, hidden_units, key, value);
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
    const float max_range) {
  int bid = blockIdx.x;

  float local_i = -FLT_MAX;
  float val;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int outIdx = (tid & 0xffffffe0) * m + (bid << 5) + (tid & 31);

    val = (float)input[outIdx] * (float)scale[bid] / max_range *
              (float)w_scale[tid] / max_range +
          (float)bias[tid];
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
    out[outIdx] = __float2int_rn((float)ffn_inner[outIdx] / (float)scale[bid] *
                                 max_range);
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
    const float max_range) {
  int bid = blockIdx.x;

  float local_i = -FLT_MAX;
  float val;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    int outIdx = (tid & 0xffffffe0) * m + (bid << 5) + (tid & 31);

    val = (float)input[outIdx] * (float)scale[bid] / max_range *
              (float)w_scale[tid] / max_range +
          (float)bias[tid];
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
    out[outIdx] = __float2int_rn((float)ffn_inner[outIdx] / (float)scale[bid] *
                                 max_range);
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
        127.0f);
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
        127.0f);
  }
}

template <typename T>
__global__ void dequant_add_bias_relu_quant(int32_t* input,
                                            int8_t* output,
                                            T* scale,
                                            T* ffn_inner,
                                            const T* __restrict bias,
                                            const T* __restrict w_scale,
                                            int n,
                                            const float max_range) {
  float local_i = -1e20f;
  int row_idx = blockIdx.x;

  for (int id = threadIdx.x; id < n; id += blockDim.x) {
    T reg_bias = __ldg(&bias[id]);

    T val = (T)((float)input[row_idx * n + id] * (float)scale[row_idx] /
                max_range * (float)w_scale[id] / max_range);
    val = val + reg_bias;
    val = val > (T)0.0f ? val : (T)0.0f;

    ffn_inner[row_idx * n + id] = val;

    local_i = max(fabs((float)val), local_i);
  }
  float max_val = blockReduceMax<float>(local_i);

  if (threadIdx.x == 0) {
    scale[row_idx] = (T)max_val;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
    output[row_idx * n + idx] =
        __float2int_rn((float)ffn_inner[row_idx * n + idx] /
                       (float)scale[row_idx] * max_range);
  }
}

template <typename T>
__global__ void dequant_add_bias_gelu_quant(int32_t* input,
                                            int8_t* output,
                                            T* scale,
                                            T* ffn_inner,
                                            const T* __restrict bias,
                                            const T* __restrict w_scale,
                                            int n,
                                            const float max_range) {
  float local_i = -1e20f;
  int row_idx = blockIdx.x;

  for (int id = threadIdx.x; id < n; id += blockDim.x) {
    float reg_bias = (float)__ldg(&bias[id]);

    float val = (float)input[row_idx * n + id] *
                ((float)scale[row_idx] / max_range) *
                ((float)w_scale[id] / max_range);
    val = val + reg_bias;
    val = gelu<float>(val);

    ffn_inner[row_idx * n + id] = val;

    local_i = max(fabs((float)val), local_i);
  }
  float max_val = blockReduceMax<float>(local_i);

  if (threadIdx.x == 0) {
    scale[row_idx] = (T)max_val;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
    output[row_idx * n + idx] =
        __float2int_rn((float)ffn_inner[row_idx * n + idx] /
                       (float)scale[row_idx] * max_range);
  }
}

template <typename T>
void dequant_add_bias_act_quant_kernelLauncher(int32_t* out_quant_buf,
                                               int8_t* input_quant_buf,
                                               T* scale,
                                               T* ffn_inner,
                                               const T* bias,
                                               const T* w_scale,
                                               int m,
                                               int n,
                                               ActivationType activation_type,
                                               cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));

  if (activation_type == ActivationType::RELU) {
    dequant_add_bias_relu_quant<T><<<grid, block, 0, stream>>>(out_quant_buf,
                                                               input_quant_buf,
                                                               scale,
                                                               ffn_inner,
                                                               bias,
                                                               w_scale,
                                                               n,
                                                               127.0f);
  } else if (activation_type == ActivationType::GELU) {
    dequant_add_bias_gelu_quant<T><<<grid, block, 0, stream>>>(out_quant_buf,
                                                               input_quant_buf,
                                                               scale,
                                                               ffn_inner,
                                                               bias,
                                                               w_scale,
                                                               n,
                                                               127.0f);
  }
}

template <typename T>
__global__ void dequant_add_bias_input_layernorm_2_quant_COL32(
    const int32_t* quant_input,
    const T* input,
    const T* __restrict gamma,
    const T* __restrict beta,
    const T* __restrict bias,
    const T* __restrict w_scale,
    T* output,
    T* norm_output,
    char4* quant_out,
    T* scale,
    const int m,
    const int n,
    const float max_range) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    output[bid * n + i] =
        (T)((float)quant_input[(i & 0xffffffe0) * m + (bid << 5) + (i & 31)] *
            (float)scale[bid] / max_range * (float)w_scale[i] / max_range);

    float local_out = (float)(__ldg(&input[bid * n + i]));
    local_out += (float)(output[bid * n + i]);
    local_out += (float)(__ldg(&bias[i]));
    output[bid * n + i] = (T)local_out;
    local_sum += local_out;
  }

  mean = blockReduceSum<float>(local_sum);

  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  float local_var_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float diff = (float)(__ldg(&output[bid * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum<float>(local_var_sum);

  if (threadIdx.x == 0) s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  float local_i = -FLT_MAX;
  for (int i = tid; i < n; i += blockDim.x) {
    norm_output[bid * n + i] =
        (T)((((float)output[bid * n + i] - s_mean) * s_variance) *
                (float)(__ldg(&gamma[i])) +
            (float)(__ldg(&beta[i])));
    local_i = max(local_i, fabs((float)norm_output[bid * n + i]));
  }

  float max_val = blockReduceMax<float>(local_i);
  if (tid == 0) {
    scale[bid] = (T)max_val;
  }
  __syncthreads();

  const float scale_val = max_range / (float)scale[bid];

  for (int tid = threadIdx.x; tid < n / 4; tid += blockDim.x) {
    char4 tmp4;
    int x = tid << 2;

    tmp4.x = __float2int_rn(static_cast<float>(norm_output[bid * n + x]) *
                            scale_val);
    tmp4.y = __float2int_rn(static_cast<float>(norm_output[bid * n + x + 1]) *
                            scale_val);
    tmp4.z = __float2int_rn(static_cast<float>(norm_output[bid * n + x + 2]) *
                            scale_val);
    tmp4.w = __float2int_rn(static_cast<float>(norm_output[bid * n + x + 3]) *
                            scale_val);

    quant_out[((x & 0xffffffe0) * m + (bid << 5) + (x & 31)) >> 2] = tmp4;
  }
}

template <typename T>
__global__ void dequant_add_bias_input_layernorm_2_quant(
    const int32_t* quant_input,
    const T* input,
    const T* __restrict gamma,
    const T* __restrict beta,
    const T* __restrict bias,
    const T* __restrict w_scale,
    T* output,
    T* norm_output,
    int8_t* quant_out,
    T* scale,
    const int m,
    const int n,
    const float max_range) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    output[bid * n + i] =
        (T)((float)quant_input[bid * n + i] * (float)scale[bid] / max_range *
            (float)w_scale[i] / max_range);

    float local_out = (float)(__ldg(&input[bid * n + i]));
    local_out += (float)(output[bid * n + i]);
    local_out += (float)(__ldg(&bias[i]));
    output[bid * n + i] = (T)local_out;
    local_sum += local_out;
  }

  mean = blockReduceSum<float>(local_sum);

  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  float local_var_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float diff = (float)(__ldg(&output[bid * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum<float>(local_var_sum);

  if (threadIdx.x == 0) s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  float local_i = -FLT_MAX;
  for (int i = tid; i < n; i += blockDim.x) {
    norm_output[bid * n + i] =
        (T)((((float)output[bid * n + i] - s_mean) * s_variance) *
                (float)(__ldg(&gamma[i])) +
            (float)(__ldg(&beta[i])));
    local_i = max(local_i, fabs((float)norm_output[bid * n + i]));
  }

  float max_val = blockReduceMax<float>(local_i);
  if (tid == 0) {
    scale[bid] = (T)max_val;
  }
  __syncthreads();

  const float scale_val = max_range / (float)scale[bid];

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    quant_out[bid * n + tid] =
        __float2int_rn((float)norm_output[bid * n + tid] * scale_val);
  }
}

template <typename T>
void dequant_add_bias_input_layernorm_2_quant_kernelLauncher(
    const int32_t* quant_output_buf,
    const T* input,
    const T* gamma,
    const T* beta,
    const T* bias,
    const T* w_scale,
    T* output,
    T* norm_output,
    int8_t* quantize_input_buf,
    T* scale,
    int m,
    int n,
    cudaStream_t stream,
    bool use_COL32) {
  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
  Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */

  if (n % 32 != 0) block.x = 1024;

  block.x =
      block.x / (4 / sizeof(T));  // if using half, only need half of block.x

  /* should pay attention to the rsqrt precision*/
  if (use_COL32) {
    dequant_add_bias_input_layernorm_2_quant_COL32<
        T><<<grid, block, 0, stream>>>(quant_output_buf,
                                       input,
                                       gamma,
                                       beta,
                                       bias,
                                       w_scale,
                                       output,
                                       norm_output,
                                       (char4*)quantize_input_buf,
                                       scale,
                                       m,
                                       n,
                                       127.0f);
  } else {
    dequant_add_bias_input_layernorm_2_quant<T><<<grid, block, 0, stream>>>(
        quant_output_buf,
        input,
        gamma,
        beta,
        bias,
        w_scale,
        output,
        norm_output,
        quantize_input_buf,
        scale,
        m,
        n,
        127.0f);
  }
}

template <typename T>
__global__ void dequant_add_bias_input_layernorm_2_kernel(
    const int32_t* quant_input,
    const T* input,
    const T* __restrict gamma,
    const T* __restrict beta,
    const T* __restrict bias,
    const T* __restrict w_scale,
    T* output,
    T* norm_output,
    char4* quant_out,
    T* scale,
    const int m,
    const int n,
    const float max_range,
    bool use_COL32) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    if (use_COL32) {
      output[bid * n + i] =
          (T)((float)quant_input[(i & 0xffffffe0) * m + (bid << 5) + (i & 31)] *
              (float)scale[bid] / max_range * (float)w_scale[i] / max_range);
    } else {
      output[bid * n + i] =
          (T)((float)quant_input[bid * n + i] * (float)scale[bid] / max_range *
              (float)w_scale[i] / max_range);
    }

    float local_out = (float)(__ldg(&input[bid * n + i]));
    local_out += (float)(output[bid * n + i]);
    local_out += (float)(__ldg(&bias[i]));
    output[bid * n + i] = (T)local_out;
    local_sum += local_out;
  }

  mean = blockReduceSum<float>(local_sum);

  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  float local_var_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float diff = (float)(__ldg(&output[bid * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum<float>(local_var_sum);

  if (threadIdx.x == 0) s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x) {
    norm_output[bid * n + i] =
        (T)((((float)output[bid * n + i] - s_mean) * s_variance) *
                (float)(__ldg(&gamma[i])) +
            (float)(__ldg(&beta[i])));
  }
}

template <typename T>
void dequant_add_bias_input_layernorm_2_kernelLauncher(
    const int32_t* quant_output_buf,
    const T* input,
    const T* gamma,
    const T* beta,
    const T* bias,
    const T* w_scale,
    T* output,
    T* norm_output,
    int8_t* quantize_input_buf,
    T* scale,
    int m,
    int n,
    cudaStream_t stream,
    bool use_COL32) {
  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
  Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */

  if (n % 32 != 0) block.x = 1024;

  block.x =
      block.x / (4 / sizeof(T));  // if using half, only need half of block.x

  /* should pay attention to the rsqrt precision*/
  dequant_add_bias_input_layernorm_2_kernel<T><<<grid, block, 0, stream>>>(
      quant_output_buf,
      input,
      gamma,
      beta,
      bias,
      w_scale,
      output,
      norm_output,
      (char4*)quantize_input_buf,
      scale,
      m,
      n,
      127.0f,
      use_COL32);
}

template <typename T>
__global__ void add_bias_input_layernorm_2_quant(const int32_t* quant_input,
                                                 const T* input,
                                                 const T* __restrict gamma,
                                                 const T* __restrict beta,
                                                 const T* __restrict bias,
                                                 const T* __restrict w_scale,
                                                 T* output,
                                                 T* norm_output,
                                                 char4* quant_out,
                                                 T* scale,
                                                 const int m,
                                                 const int n,
                                                 const float max_range,
                                                 bool use_COL32) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float local_out = (float)(__ldg(&input[bid * n + i]));
    local_out += (float)(output[bid * n + i]);
    local_out += (float)(__ldg(&bias[i]));
    output[bid * n + i] = (T)local_out;
    local_sum += local_out;
  }

  mean = blockReduceSum<float>(local_sum);

  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  float local_var_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float diff = (float)(__ldg(&output[bid * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum<float>(local_var_sum);

  if (threadIdx.x == 0) s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  float local_i = -FLT_MAX;
  for (int i = tid; i < n; i += blockDim.x) {
    norm_output[bid * n + i] =
        (T)((((float)output[bid * n + i] - s_mean) * s_variance) *
                (float)(__ldg(&gamma[i])) +
            (float)(__ldg(&beta[i])));
    local_i = max(local_i, fabs((float)norm_output[bid * n + i]));
  }

  float max_val = blockReduceMax<float>(local_i);
  if (tid == 0) {
    scale[bid] = (T)max_val;
  }
  __syncthreads();

  const float scale_val = max_range / (float)scale[bid];

  for (int tid = threadIdx.x; tid < n / 4; tid += blockDim.x) {
    char4 tmp4;
    int x = tid << 2;

    tmp4.x = __float2int_rn(static_cast<float>(norm_output[bid * n + x]) *
                            scale_val);
    tmp4.y = __float2int_rn(static_cast<float>(norm_output[bid * n + x + 1]) *
                            scale_val);
    tmp4.z = __float2int_rn(static_cast<float>(norm_output[bid * n + x + 2]) *
                            scale_val);
    tmp4.w = __float2int_rn(static_cast<float>(norm_output[bid * n + x + 3]) *
                            scale_val);

    if (use_COL32) {
      quant_out[((x & 0xffffffe0) * m + (bid << 5) + (x & 31)) >> 2] = tmp4;
    } else {
      quant_out[(bid * n + x) >> 2] = tmp4;
    }
  }
}

template <typename T>
void add_bias_input_layernorm_2_quant_kernelLauncher(
    const int32_t* quant_output_buf,
    const T* input,
    const T* gamma,
    const T* beta,
    const T* bias,
    const T* w_scale,
    T* output,
    T* norm_output,
    int8_t* quantize_input_buf,
    T* scale,
    int m,
    int n,
    cudaStream_t stream,
    bool use_COL32) {
  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
  Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */

  if (n % 32 != 0) block.x = 1024;

  block.x =
      block.x / (4 / sizeof(T));  // if using half, only need half of block.x

  /* should pay attention to the rsqrt precision*/
  add_bias_input_layernorm_2_quant<T><<<grid, block, 0, stream>>>(
      quant_output_buf,
      input,
      gamma,
      beta,
      bias,
      w_scale,
      output,
      norm_output,
      (char4*)quantize_input_buf,
      scale,
      m,
      n,
      127.0f,
      use_COL32);
}

template <typename T>
__global__ void layer_norm_quant_COL32_kernel(const T* __restrict input,
                                              const T* __restrict gamma,
                                              const T* __restrict beta,
                                              T* scale,
                                              T* output,
                                              char4* quant_out,
                                              int m,
                                              int n,
                                              float max_range) {
  const int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    local_sum += (float)(__ldg(&input[blockIdx.x * n + i]));
  }

  mean = blockReduceSum<float>(local_sum);

  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  float local_var_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float diff = (float)(__ldg(&input[blockIdx.x * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum<float>(local_var_sum);

  if (threadIdx.x == 0) s_variance = rsqrtf(variance / n + 1e-6);

  __syncthreads();

  float local_i = -FLT_MAX;
  for (int i = tid; i < n; i += blockDim.x) {
    output[blockIdx.x * n + i] =
        (T)((((float)input[blockIdx.x * n + i] - s_mean) * s_variance) *
                (float)(__ldg(&gamma[i])) +
            (float)(__ldg(&beta[i])));
    local_i = max(local_i, fabs((float)output[blockIdx.x * n + i]));
  }

  float max_val = blockReduceMax<float>(local_i);
  if (tid == 0) {
    scale[blockIdx.x] = (T)max_val;
  }
  __syncthreads();

  const float scale_val = max_range / (float)scale[blockIdx.x];

  for (int tid = threadIdx.x; tid < n / 4; tid += blockDim.x) {
    char4 tmp4;
    int x = tid << 2;

    tmp4.x = __float2int_rn(static_cast<float>(output[blockIdx.x * n + x]) *
                            scale_val);
    tmp4.y = __float2int_rn(static_cast<float>(output[blockIdx.x * n + x + 1]) *
                            scale_val);
    tmp4.z = __float2int_rn(static_cast<float>(output[blockIdx.x * n + x + 2]) *
                            scale_val);
    tmp4.w = __float2int_rn(static_cast<float>(output[blockIdx.x * n + x + 3]) *
                            scale_val);

    quant_out[((x & 0xffffffe0) * m + (blockIdx.x << 5) + (x & 31)) >> 2] =
        tmp4;
  }
}

template <typename T>
__global__ void layer_norm_quant_kernel(const T* __restrict input,
                                        const T* __restrict gamma,
                                        const T* __restrict beta,
                                        T* scale,
                                        T* output,
                                        int8_t* quant_out,
                                        int m,
                                        int n,
                                        float max_range) {
  const int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    local_sum += (float)(__ldg(&input[blockIdx.x * n + i]));
  }

  mean = blockReduceSum<float>(local_sum);

  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  float local_var_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float diff = (float)(__ldg(&input[blockIdx.x * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum<float>(local_var_sum);

  if (threadIdx.x == 0) s_variance = rsqrtf(variance / n + 1e-6);

  __syncthreads();

  float local_i = -FLT_MAX;
  for (int i = tid; i < n; i += blockDim.x) {
    output[blockIdx.x * n + i] =
        (T)((((float)input[blockIdx.x * n + i] - s_mean) * s_variance) *
                (float)(__ldg(&gamma[i])) +
            (float)(__ldg(&beta[i])));
    local_i = max(local_i, fabs((float)output[blockIdx.x * n + i]));
  }

  float max_val = blockReduceMax<float>(local_i);
  if (tid == 0) {
    scale[blockIdx.x] = (T)max_val;
  }
  __syncthreads();

  const float scale_val = max_range / (float)scale[blockIdx.x];

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    quant_out[blockIdx.x * n + tid] =
        __float2int_rn((float)output[blockIdx.x * n + tid] * scale_val);
  }
}

template <typename T>
void layer_norm_quant(const T* from_tensor,
                      const T* gamma,
                      const T* beta,
                      T* norm_from_tensor_buf_,
                      T* scale,
                      int8_t* quantize_input_buf_,
                      const int m,
                      const int n,
                      cudaStream_t stream,
                      bool use_COL32) {
  dim3 grid(m);
  dim3 block(min(n, 1024));

  if (n % 32 != 0) block.x = 1024;

  block.x = block.x / (4 / sizeof(T));

  if (use_COL32) {
    layer_norm_quant_COL32_kernel<T><<<grid, block, 0, stream>>>(
        from_tensor,
        gamma,
        beta,
        scale,
        norm_from_tensor_buf_,
        (char4*)quantize_input_buf_,
        m,
        n,
        127.0f);
  } else {
    layer_norm_quant_kernel<T><<<grid, block, 0, stream>>>(
        from_tensor,
        gamma,
        beta,
        scale,
        norm_from_tensor_buf_,
        quantize_input_buf_,
        m,
        n,
        127.0f);
  }
}

template <typename T>
__global__ void dequant_add_bias_input(const int32_t* quant_in,
                                       const T* scale,
                                       const T* weight_scale,
                                       T* output,
                                       const T* input,
                                       const T* bias,
                                       const int m,
                                       const int n,
                                       const float max_range,
                                       const bool use_COL32) {
  const int bid = blockIdx.x;

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    T bias_val = __ldg(&bias[tid]);
    if (use_COL32) {
      output[bid * n + tid] =
          (T)((float)
                  quant_in[(tid & 0xffffffe0) * m + (bid << 5) + (tid & 31)] *
              (float)scale[bid] / max_range * (float)weight_scale[tid] /
              max_range) +
          input[bid * n + tid] + bias_val;
    } else {
      output[bid * n + tid] =
          (T)((float)quant_in[bid * n + tid] * (float)scale[bid] / max_range *
              (float)weight_scale[tid] / max_range) +
          input[bid * n + tid] + bias_val;
    }
  }
}

template <typename T>
void dequant_add_bias_input_kernelLauncher(const int32_t* quant_in,
                                           const T* scale,
                                           const T* weight_scale,
                                           T* output,
                                           const T* bias,
                                           const T* input,
                                           const int m,
                                           const int n,
                                           cudaStream_t stream,
                                           bool use_COL32) {
  dim3 grid(min(m, 65536));
  dim3 block(min(n, 1024));

  dequant_add_bias_input<<<grid, block, 0, stream>>>(quant_in,
                                                     scale,
                                                     weight_scale,
                                                     output,
                                                     input,
                                                     bias,
                                                     m,
                                                     n,
                                                     127.0f,
                                                     use_COL32);
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

template void dequant_add_bias_act_quant_kernelLauncher<float>(
    int32_t* out_quant_buf,
    int8_t* input_quant_buf,
    float* scale,
    float* ffn_inner,
    const float* bias,
    const float* w_scale,
    int m,
    int n,
    ActivationType activation_type,
    cudaStream_t stream);

template void dequant_add_bias_act_quant_kernelLauncher<half>(
    int32_t* out_quant_buf,
    int8_t* input_quant_buf,
    half* scale,
    half* ffn_inner,
    const half* bias,
    const half* w_scale,
    int m,
    int n,
    ActivationType activation_type,
    cudaStream_t stream);

template void dequant_add_bias_input_layernorm_2_kernelLauncher(
    const int32_t* quant_output_buf,
    const float* input,
    const float* gamma,
    const float* beta,
    const float* bias,
    const float* w_scale,
    float* output,
    float* norm_output,
    int8_t* quantize_input_buf_,
    float* scale,
    int m,
    int n,
    cudaStream_t stream,
    bool use_COL32);

template void dequant_add_bias_input_layernorm_2_kernelLauncher(
    const int32_t* quant_output_buf,
    const half* input,
    const half* gamma,
    const half* beta,
    const half* bias,
    const half* w_scale,
    half* output,
    half* norm_output,
    int8_t* quantize_input_buf_,
    half* scale,
    int m,
    int n,
    cudaStream_t stream,
    bool use_COL32);

template void dequant_add_bias_input_layernorm_2_quant_kernelLauncher(
    const int32_t* quant_output_buf,
    const float* input,
    const float* gamma,
    const float* beta,
    const float* bias,
    const float* w_scale,
    float* output,
    float* norm_output,
    int8_t* quantize_input_buf_,
    float* scale,
    int m,
    int n,
    cudaStream_t stream,
    bool use_COL32);

template void dequant_add_bias_input_layernorm_2_quant_kernelLauncher(
    const int32_t* quant_output_buf,
    const half* input,
    const half* gamma,
    const half* beta,
    const half* bias,
    const half* w_scale,
    half* output,
    half* norm_output,
    int8_t* quantize_input_buf_,
    half* scale,
    int m,
    int n,
    cudaStream_t stream,
    bool use_COL32);

template void add_bias_input_layernorm_2_quant_kernelLauncher(
    const int32_t* quant_output_buf,
    const float* input,
    const float* gamma,
    const float* beta,
    const float* bias,
    const float* w_scale,
    float* output,
    float* norm_output,
    int8_t* quantize_input_buf_,
    float* scale,
    int m,
    int n,
    cudaStream_t stream,
    bool use_COL32);

template void add_bias_input_layernorm_2_quant_kernelLauncher(
    const int32_t* quant_output_buf,
    const half* input,
    const half* gamma,
    const half* beta,
    const half* bias,
    const half* w_scale,
    half* output,
    half* norm_output,
    int8_t* quantize_input_buf_,
    half* scale,
    int m,
    int n,
    cudaStream_t stream,
    bool use_COL32);

template void dequant_add_bias_input_kernelLauncher(const int32_t* quant_in,
                                                    const float* scale,
                                                    const float* weight_scale,
                                                    float* output,
                                                    const float* bias,
                                                    const float* input,
                                                    const int m,
                                                    const int n,
                                                    cudaStream_t stream,
                                                    bool use_COL32);

template void dequant_add_bias_input_kernelLauncher(const int32_t* quant_in,
                                                    const half* scale,
                                                    const half* weight_scale,
                                                    half* output,
                                                    const half* bias,
                                                    const half* input,
                                                    const int m,
                                                    const int n,
                                                    cudaStream_t stream,
                                                    bool use_COL32);

template void layer_norm_quant(const float* from_tensor,
                               const float* gamma,
                               const float* beta,
                               float* norm_from_tensor_buf_,
                               float* scale,
                               int8_t* quantize_input_buf_,
                               const int m,
                               const int n,
                               cudaStream_t stream,
                               bool use_COL32);

template void layer_norm_quant(const half* from_tensor,
                               const half* gamma,
                               const half* beta,
                               half* norm_from_tensor_buf_,
                               half* scale,
                               int8_t* quantize_input_buf_,
                               const int m,
                               const int n,
                               cudaStream_t stream,
                               bool use_COL32);

}