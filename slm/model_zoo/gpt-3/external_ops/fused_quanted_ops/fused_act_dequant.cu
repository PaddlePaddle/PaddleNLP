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

#include "quant_utils.h"


// cuda结构体拷贝
template <typename T, int N>
struct alignas(16) VectorType {
  T data[N];
};

__global__ void FusedActDequant(const phi::float8_e4m3fn *__restrict__ Xin,
                                const float *__restrict__ Xscale,
                                phi::bfloat16 *__restrict__ out,
                                const int64_t rows,
                                const int cols) {
  const int64_t this_row_idx = blockIdx.x;
  if (this_row_idx >= rows) return;

  const int Xscale_stride = (cols + 127) / 128;  // 计算缩放因子的步长

  const int vector_size = 16;  // 向量的元素数量，处理16个元素

  // 每行的向量数量
  const int num_vectors = cols / vector_size;
  const int remaining_elements = cols % vector_size;

  const int tid = threadIdx.x;

  for (int vec_idx = tid; vec_idx < num_vectors; vec_idx += blockDim.x) {
    int x_offset = vec_idx * vector_size;
    int64_t X_idx = (int64_t)this_row_idx * (int64_t)cols + (int64_t)x_offset;

    // 加载16个 __nv_fp8_e4m3 元素到向量中
    const VectorType<__nv_fp8_e4m3, vector_size> *X_vec_ptr =
        reinterpret_cast<const VectorType<__nv_fp8_e4m3, vector_size> *>(Xin +
                                                                         X_idx);
    VectorType<__nv_fp8_e4m3, vector_size> X_vec = X_vec_ptr[0];

    // 获取对应的缩放因子
    int64_t scale_idx =
        (int64_t)this_row_idx * (int64_t)Xscale_stride + (x_offset / 128);
    float this_scale = Xscale[scale_idx];

    // 初始化输出向量
    VectorType<__nv_bfloat16, vector_size> out_vec;

// 逐元素处理向量中的数据
#pragma unroll
    for (int i = 0; i < vector_size; ++i) {
      // 将fp8转换为float
      float X_value = static_cast<float>(X_vec.data[i]);
      // 乘以缩放因子
      X_value *= this_scale;
      // 转换为bfloat16并存储
      out_vec.data[i] = __float2bfloat16(X_value);
    }

    // 将输出向量存储到全局内存
    VectorType<__nv_bfloat16, vector_size> *out_vec_ptr =
        reinterpret_cast<VectorType<__nv_bfloat16, vector_size> *>(out + X_idx);
    out_vec_ptr[0] = out_vec;
  }

  // 处理剩余不能被向量化的元素
  if (remaining_elements > 0) {
    int x_offset = num_vectors * vector_size;
    int64_t X_idx = (int64_t)this_row_idx * (int64_t)cols + (int64_t)x_offset;
    int64_t idx = X_idx + tid;
    if (tid < remaining_elements) {
      float X_value = static_cast<float>(Xin[idx]);
      X_value *= Xscale[(int64_t)this_row_idx * (int64_t)Xscale_stride +
                        (x_offset / 128)];
      out[idx] = __float2bfloat16(X_value);
    }
  }
}

void dispatch_fused_act_dequant(const paddle::Tensor &X,
                                const paddle::Tensor &Xscale,
                                paddle::Tensor &out,
                                const int64_t rows,
                                const int64_t cols) {
  dim3 grid;
  dim3 block;
  grid.x = X.shape()[0];
  block.x = 256;
  FusedActDequant<<<grid, block, 0, X.stream()>>>(X.data<phi::float8_e4m3fn>(),
                                                  Xscale.data<float>(),
                                                  out.data<phi::bfloat16>(),
                                                  rows,
                                                  cols);
}
std::vector<paddle::Tensor> fused_act_dequant(const paddle::Tensor &X,
                                              const paddle::Tensor &Xscale) {
  PD_CHECK(X.dtype() == paddle::DataType::FLOAT8_E4M3FN);
  PD_CHECK(Xscale.dtype() == paddle::DataType::FLOAT32);
  int64_t rows, cols;
  rows = X.shape()[0];
  cols = X.shape()[1];
  PADDLE_ENFORCE_LE(
      rows,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument(
          "rows should be less than INT_MAX, received rows: (%ld)", rows));
  PADDLE_ENFORCE_LE(
      cols,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument(
          "cols should be less than INT_MAX, received cols: (%ld)", cols));
  paddle::Tensor out;

  out = paddle::empty({rows, cols}, paddle::DataType::BFLOAT16, X.place());
  auto out_ptr = reinterpret_cast<void *>(out.data<phi::bfloat16>());
  cudaMemsetAsync(
      out_ptr, 0, sizeof(phi::bfloat16) * rows * cols, out.stream());
  dispatch_fused_act_dequant(X, Xscale, out, rows, cols);
  return {out};
}

PD_BUILD_OP(fused_act_dequant)
    .Inputs({"X", "Xscale"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(fused_act_dequant));
