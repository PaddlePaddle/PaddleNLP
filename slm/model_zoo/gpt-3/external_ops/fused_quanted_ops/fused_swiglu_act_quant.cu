#include "quant_utils.h"

#define LAUNCH_FUSED_SWIGLU_ACT_QUANT(                                         \
    __is_combine, __transpose, __using_pow2_scaling, __padding_last_dim_to_8x) \
  do {                                                                         \
    auto kernel = FusedSwigluActQuantKernel<outT,                              \
                                            __is_combine,                      \
                                            __transpose,                       \
                                            __using_pow2_scaling,              \
                                            __padding_last_dim_to_8x>;         \
    int smem_size = 128 * 129 * sizeof(float);                                 \
    PD_CHECK(cudaFuncSetAttribute(kernel,                                      \
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                  smem_size) == cudaSuccess);                  \
    kernel<<<grid, block, smem_size, X.stream()>>>(                            \
        X.data<phi::bfloat16>(),                                               \
        Y ? Y->data<phi::bfloat16>() : nullptr,                                \
        out.data<outT>(),                                                      \
        scale.data<float>(),                                                   \
        rows,                                                                  \
        cols,                                                                  \
        TYPE_MAX);                                                             \
  } while (0)


__device__ __forceinline__ float fast_swiglu(const __nv_bfloat16 x,
                                             const __nv_bfloat16 y) {
  const float x_f = __bfloat162float(x);
  const float y_f = __bfloat162float(y);
  const float silu = x_f * __frcp_rn(1.0f + __expf(-x_f));
  const float result = silu * y_f;
  return result;
}

constexpr int BLOCK_SIZE = 128;
template <typename OutT,
          bool is_combine,
          bool transpose_output,
          bool using_pow2_scaling,
          bool padding_last_dim_to_8x>
__global__ void FusedSwigluActQuantKernel(const phi::bfloat16 *__restrict__ Xin,
                                          const phi::bfloat16 *__restrict__ Yin,
                                          OutT *__restrict__ out,
                                          float *__restrict__ scales,
                                          const int rows,
                                          const int cols,
                                          const int TYPE_MAX) {
  // 如果is_combine,
  // 则Xin中存储了X和Y的数据，此时gridDim.x应为cols/BLOCK_SIZE的一半
  // 共享内存用于:1.计算后直接存储转置后的结果数据  2.用于1x128规约+量化
  // 共享内存布局,前128x128为计算结果，后128为scale
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  // extern __shared__  uint8_t smem_buffer[];
  float *smem_tile = reinterpret_cast<float *>(smem_buffer);
  float *smem_max =
      reinterpret_cast<float *>(smem_buffer) + BLOCK_SIZE * BLOCK_SIZE;

  const int g_block_y_offset = blockIdx.y * BLOCK_SIZE;  // 块内行坐标偏移
  const int g_block_x_offset = blockIdx.x * BLOCK_SIZE;  // 块内列坐标偏移
  const __nv_bfloat16 *X = reinterpret_cast<const __nv_bfloat16 *>(Xin);
  const __nv_bfloat16 *Y = reinterpret_cast<const __nv_bfloat16 *>(Yin);

  // 阶段1:
  // Elementwise加载数据、计算、并将结果直接以特定布局存入共享内存‌(32x32)
  // ------------------------------
  for (int y_offset = threadIdx.y; y_offset < BLOCK_SIZE;
       y_offset += blockDim.y) {
    for (int x_offset = threadIdx.x; x_offset < BLOCK_SIZE;
         x_offset += blockDim.x) {
      const int in_y_idx = g_block_y_offset + y_offset;
      const int in_x_idx = g_block_x_offset + x_offset;
      const int src_idx = in_y_idx * cols + in_x_idx;
      if constexpr (transpose_output) {  // shared-mem conflict free写数
        if constexpr (is_combine) {
          if (in_y_idx < rows && in_x_idx < cols / 2) {
            smem_tile[swizzled_2d_idx(x_offset, BLOCK_SIZE, y_offset)] =
                fast_swiglu(X[src_idx], X[src_idx + cols / 2]);
          }
        } else {
          if (in_y_idx < rows && in_x_idx < cols) {
            const int src_idx = in_y_idx * cols + in_x_idx;
            smem_tile[swizzled_2d_idx(x_offset, BLOCK_SIZE, y_offset)] =
                fast_swiglu(X[src_idx], Y[src_idx]);
          }
        }
      } else {
        if constexpr (is_combine) {
          if (in_y_idx < rows && in_x_idx < cols / 2) {
            smem_tile[swizzled_2d_idx(y_offset, BLOCK_SIZE, x_offset)] =
                fast_swiglu(X[src_idx], X[src_idx + cols / 2]);
          }
        } else {
          if (in_y_idx < rows && in_x_idx < cols) {
            smem_tile[swizzled_2d_idx(y_offset, BLOCK_SIZE, x_offset)] =
                fast_swiglu(X[src_idx], Y[src_idx]);
          }
        }
      }
    }
  }
  __syncthreads();  // smem_tile中的swiglu数据(按需transpose)已ready

  // 阶段2: ‌
  // Shared 两级reduce, 给出每行的absmax_f
  // ------------------------------------------------------
  float local_max = 0.0f;
  // 每个warp计算一行最大值，每个warp处理4行
  for (int y_offset = threadIdx.y; y_offset < BLOCK_SIZE;
       y_offset += blockDim.y) {
    // 行quantize max计算, 每行128个元素，每个线程处理4个元素
    for (int x_offset = threadIdx.x; x_offset < BLOCK_SIZE;
         x_offset += blockDim.x) {
      bool is_output_inner_OOB;
      if constexpr (transpose_output) {
        is_output_inner_OOB = (g_block_y_offset + x_offset) >= rows;
      } else {
        is_output_inner_OOB = (g_block_x_offset + x_offset) >= cols;
      }
      if (is_output_inner_OOB) break;  // 列越界则不取值、不影响max;
      local_max = fmaxf(
          local_max,
          fabsf(smem_tile[swizzled_2d_idx(
              y_offset,
              BLOCK_SIZE,
              x_offset)]));  // 正常情况下提供绝对值给其他线程,每个线程的local_max最多为4数最大值
    }
    bool is_output_outer_OOB;
    if constexpr (transpose_output) {
      is_output_outer_OOB = (g_block_x_offset + y_offset) >= cols;
    } else {
      is_output_outer_OOB = (g_block_y_offset + y_offset) >= rows;
    }
    if (is_output_outer_OOB) break;  // 行越界，不对该行做量化
    local_max =
        warpReduceMax(local_max);  // 无论该block有多少个有效线程，max均合法
    if (threadIdx.x == 0)
      smem_max[y_offset] = local_max;  // x0 顺序写，无conflict
  }
  __syncthreads();  // smem_max中的scale数据ready，128个元素对应128行的scale

  // 阶段3:
  // Output放缩强转 + Scale写回‌
  // ------------------------------------------------------------------
  /* 输出缓冲区伪代码：
  if(transpose_output){
      if(is_combine){
          out = {cols / 2, rows}, scale = {cols / 2, (rows + 127) / 128}
      }else{
          out = {cols, rows}, scale = {cols, (rows + 127) / 128}
      }
  }else{
      if(is_combine){
          out {rows, cols / 2}, scale = {rows, ((cols / 2) + 127) / 128}
      }else{
          out = {rows, cols}, scale = {rows, (cols + 127) / 128}
      }
  }
  */
  for (int y_offset = threadIdx.y; y_offset < BLOCK_SIZE;
       y_offset += blockDim.y) {
    for (int x_offset = threadIdx.x; x_offset < BLOCK_SIZE;
         x_offset += blockDim.x) {
      const float scale_on_fp32_to_outputT =
          ComputeScale<__nv_bfloat16, OutT, using_pow2_scaling>(
              smem_max[y_offset], 0.0f);
      const float scale_on_fp8_to_inputT = __frcp_rn(scale_on_fp32_to_outputT);
      float output_scaled_fp32 =
          smem_tile[swizzled_2d_idx(y_offset, BLOCK_SIZE, x_offset)] *
          scale_on_fp32_to_outputT;
      const OutT output_scaled_fp8 = static_cast<OutT>(output_scaled_fp32);
      if constexpr (transpose_output) {
        const int g_output_y_offset = g_block_x_offset + y_offset;
        const int g_output_x_offset = g_block_y_offset + x_offset;
        // 如果padding连续维为8的倍数，则将连续维的rank使用位运算向上取整为最近的8倍数
        const int g_output_inner_rank =
            (padding_last_dim_to_8x) ? (rows + 7) & -8 : rows;
        if constexpr (is_combine) {
          const int g_scale_inner_rank = (rows + 127) / 128;
          if (g_output_y_offset < cols / 2 &&
              g_output_x_offset < g_output_inner_rank) {
            // 如果超过了原有的rank，则必然是padding情况，顺手使用0进行padding
            out[g_output_y_offset * g_output_inner_rank + g_output_x_offset] =
                (g_output_x_offset < rows) ? output_scaled_fp8 : (OutT)0;
            scales[g_output_y_offset * g_scale_inner_rank +
                   g_output_x_offset / 128] = scale_on_fp8_to_inputT;
          }
        } else {
          const int g_scale_inner_rank = (rows + 127) / 128;
          if (g_output_y_offset < cols &&
              g_output_x_offset < g_output_inner_rank) {
            out[g_output_y_offset * g_output_inner_rank + g_output_x_offset] =
                (g_output_x_offset < rows) ? output_scaled_fp8 : (OutT)0;
            scales[g_output_y_offset * g_scale_inner_rank +
                   g_output_x_offset / 128] = scale_on_fp8_to_inputT;
          }
        }
      } else {
        const int g_output_y_offset = g_block_y_offset + y_offset;
        const int g_output_x_offset = g_block_x_offset + x_offset;
        if constexpr (is_combine) {
          const int g_scale_inner_rank = (cols / 2 + 127) / 128;
          const int g_output_inner_rank =
              (padding_last_dim_to_8x) ? ((cols / 2 + 7) & -8) : (cols / 2);
          if (g_output_y_offset < rows &&
              g_output_x_offset < g_output_inner_rank) {
            out[g_output_y_offset * g_output_inner_rank + g_output_x_offset] =
                (g_output_x_offset < cols / 2) ? output_scaled_fp8 : (OutT)0;
            scales[g_output_y_offset * g_scale_inner_rank +
                   g_output_x_offset / 128] = scale_on_fp8_to_inputT;
          }
        } else {
          const int g_scale_inner_rank = (cols + 127) / 128;
          const int g_output_inner_rank =
              (padding_last_dim_to_8x) ? ((cols + 7) & -8) : (cols);
          if (g_output_y_offset < rows &&
              g_output_x_offset < g_output_inner_rank) {
            out[g_output_y_offset * g_output_inner_rank + g_output_x_offset] =
                (g_output_x_offset < cols) ? output_scaled_fp8 : (OutT)0;
            scales[g_output_y_offset * g_scale_inner_rank +
                   g_output_x_offset / 128] = scale_on_fp8_to_inputT;
          }
        }
      }
    }
  }
}

template <typename outT>
void dispatch_fused_swiglu_act_quant(const paddle::Tensor &X,
                                     const paddle::optional<paddle::Tensor> &Y,
                                     paddle::Tensor &out,
                                     paddle::Tensor &scale,
                                     const int rows,
                                     const int cols,
                                     const bool &transpose_output,
                                     const bool &using_pow2_scaling,
                                     const bool &padding_last_dim_to_8x,
                                     const float TYPE_MAX) {
  const bool is_combine = Y ? false : true;
  dim3 grid;
  dim3 block(32, 32);
  DISPATCH_BOOL(
      is_combine,
      k_is_combine,
      DISPATCH_BOOL(
          transpose_output,
          k_transpose_output,
          DISPATCH_BOOL(
              using_pow2_scaling,
              k_using_pow2_scaling,
              DISPATCH_BOOL(
                  padding_last_dim_to_8x,
                  k_padding_last_dim_to_8x,
                  if constexpr (k_is_combine) {
                    grid.y = (rows + 127) / 128;
                    grid.x = ((cols / 2) + 127) / 128;
                  } else {
                    grid.y = (rows + 127) / 128;
                    grid.x = (cols + 127) / 128;
                  } LAUNCH_FUSED_SWIGLU_ACT_QUANT(k_is_combine,
                                                  k_transpose_output,
                                                  k_using_pow2_scaling,
                                                  k_padding_last_dim_to_8x);))))
}
std::vector<paddle::Tensor> fused_swiglu_act_quant(
    const paddle::Tensor &X,
    const paddle::optional<paddle::Tensor> &Y,
    const bool &transpose_output,
    const bool &to_e4m3,
    const bool &using_pow2_scaling,
    const bool &padding_last_dim_to_8x) {
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16);
  int64_t data_rows = size_to_dim(X.shape().size() - 1, X.shape());
  int64_t data_cols = X.shape().back();
  int64_t rows = data_rows, cols = data_cols;

  paddle::Tensor out;
  paddle::Tensor scale;

  if (Y) {
    // Y存在
    PD_CHECK(Y.get().dtype() == paddle::DataType::BFLOAT16);
    auto Xdims = X.dims();
    const auto &y_tensor = Y.get();
    const auto &y_dims = y_tensor.dims();
    PADDLE_ENFORCE_EQ(y_dims,
                      Xdims,
                      common::errors::InvalidArgument(
                          "The shape of Input(Y):[%s] must be equal "
                          "to the shape of Input(X):[%s].",
                          y_dims,
                          Xdims));
    if (transpose_output) {
      if (padding_last_dim_to_8x) {
        rows = ((data_rows + 7) / 8) *
               8;  // 向上padding到8的倍数, 因为128为8的倍数，不影响scale shape
      }
      if (to_e4m3) {
        out = paddle::empty(
            {cols, rows}, paddle::DataType::FLOAT8_E4M3FN, X.place());
      } else {
        out = paddle::empty(
            {cols, rows}, paddle::DataType::FLOAT8_E5M2, X.place());
      }
      scale = paddle::empty(
          {cols, (rows + 127) / 128}, paddle::DataType::FLOAT32, X.place());
    } else {
      if (padding_last_dim_to_8x) {
        cols = ((data_cols + 7) / 8) *
               8;  // 向上padding到8的倍数, 因为128为8的倍数，不影响scale shape
      }
      if (to_e4m3) {
        out = paddle::empty(
            {rows, cols}, paddle::DataType::FLOAT8_E4M3FN, X.place());
      } else {
        out = paddle::empty(
            {rows, cols}, paddle::DataType::FLOAT8_E5M2, X.place());
      }
      scale = paddle::empty(
          {rows, (cols + 127) / 128}, paddle::DataType::FLOAT32, X.place());
    }
  } else {
    // Y不存在时，X的column是输出column的两倍
    auto Xdims = X.dims();
    auto n = Xdims.at(Xdims.size() - 1);
    PADDLE_ENFORCE_EQ(n % 2,
                      0,
                      common::errors::InvalidArgument(
                          "The last dim of Input(X) should be exactly divided "
                          "by 2 when Input(Y) is None, but got %d",
                          n));
    if (transpose_output) {
      if (padding_last_dim_to_8x) {
        rows =
            ((data_rows + 7) / 8) * 8;  // rows 向上padding到8的倍数,
                                        // 因为128为8的倍数，不影响scale shape
      }
      if (to_e4m3) {
        out = paddle::empty(
            {cols / 2, rows}, paddle::DataType::FLOAT8_E4M3FN, X.place());
      } else {
        out = paddle::empty(
            {cols / 2, rows}, paddle::DataType::FLOAT8_E5M2, X.place());
      }
      scale = paddle::empty(
          {cols / 2, (rows + 127) / 128}, paddle::DataType::FLOAT32, X.place());
    } else {
      if (padding_last_dim_to_8x) {
        cols = ((data_cols / 2 + 7) / 8) *
               16;  // col/2 向上padding到8的倍数,
                    // 因为128为8的倍数，不影响scale shape
      }
      if (to_e4m3) {
        out = paddle::empty(
            {rows, cols / 2}, paddle::DataType::FLOAT8_E4M3FN, X.place());
      } else {
        out = paddle::empty(
            {rows, cols / 2}, paddle::DataType::FLOAT8_E5M2, X.place());
      }
      scale = paddle::empty({rows, ((cols / 2) + 127) / 128},
                            paddle::DataType::FLOAT32,
                            X.place());
    }
  }
  if (to_e4m3) {
    dispatch_fused_swiglu_act_quant<phi::float8_e4m3fn>(
        X,
        Y,
        out,
        scale,
        data_rows,
        data_cols,
        transpose_output,
        using_pow2_scaling,
        padding_last_dim_to_8x,
        F8LimitsTrait<__nv_fp8_e4m3>::max);
  } else {
    dispatch_fused_swiglu_act_quant<phi::float8_e5m2>(
        X,
        Y,
        out,
        scale,
        data_rows,
        data_cols,
        transpose_output,
        using_pow2_scaling,
        padding_last_dim_to_8x,
        F8LimitsTrait<__nv_fp8_e5m2>::max);
  }
  return {out, scale};
}

PD_BUILD_OP(fused_swiglu_act_quant)
    .Inputs({"X", paddle::Optional("Y")})
    .Outputs({"output", "scale"})
    .Attrs({"transpose_output: bool",
            "to_e4m3: bool",
            "using_pow2_scaling: bool",
            "padding_last_dim_to_8x: bool"})
    .SetKernelFn(PD_KERNEL(fused_swiglu_act_quant));
