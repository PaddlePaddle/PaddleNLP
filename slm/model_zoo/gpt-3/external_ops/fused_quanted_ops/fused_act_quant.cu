#include "quant_utils.h"

constexpr int64_t TILE_SIZE = 128;  // 每个 block 处理 128x128 的元素块
#define LAUNCH_FUSED_ACT_QUANT(                                                \
    __transpose, __using_pow2_scaling, __padding_last_dim_to_8x)               \
  do {                                                                         \
    auto kernel = FusedActQuant<outT,                                          \
                                __transpose,                                   \
                                __using_pow2_scaling,                          \
                                __padding_last_dim_to_8x>;                     \
    int smem_size = 128 * 129 * sizeof(float);                                 \
    PD_CHECK(cudaFuncSetAttribute(kernel,                                      \
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                  smem_size) == cudaSuccess);                  \
    kernel<<<grid, block, smem_size, X.stream()>>>(X.data<phi::bfloat16>(),    \
                                                   out.data<outT>(),           \
                                                   scale.data<float>(),        \
                                                   rows,                       \
                                                   cols,                       \
                                                   TYPE_MAX);                  \
  } while (0)


#define BLOCK_SIZE 128
template <typename OutT,
          bool transpose_output,
          bool using_pow2_scaling,
          bool padding_last_dim_to_8x>
__global__ void FusedActQuant(const phi::bfloat16 *__restrict__ Xin,
                              OutT *__restrict__ out,
                              float *__restrict__ scales,
                              const int rows,  // 此处的row未带padding
                              const int cols,  // 同上
                              const int TYPE_MAX) {
  // 共享内存用于:1.直接读取+存储转置后的数据  2.用于1x128规约+量化
  // 共享内存布局,前128x128为计算结果，后128为scale
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  float *smem_tile = reinterpret_cast<float *>(smem_buffer);
  float *smem_max =
      reinterpret_cast<float *>(smem_buffer) + BLOCK_SIZE * BLOCK_SIZE;

  const int g_block_y_offset = blockIdx.y * BLOCK_SIZE;  // 块内行坐标偏移
  const int g_block_x_offset = blockIdx.x * BLOCK_SIZE;  // 块内列坐标偏移
  const __nv_bfloat16 *X = reinterpret_cast<const __nv_bfloat16 *>(Xin);

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
        if (in_y_idx < rows && in_x_idx < cols) {
          const int src_idx = in_y_idx * cols + in_x_idx;
          smem_tile[swizzled_2d_idx(x_offset, BLOCK_SIZE, y_offset)] =
              X[src_idx];
        }
      } else {
        if (in_y_idx < rows && in_x_idx < cols) {
          smem_tile[swizzled_2d_idx(y_offset, BLOCK_SIZE, x_offset)] =
              X[src_idx];
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
  // 无论是否padding_last_dim_to_8x，输出连续维的stride均为合法值
  const int padded_rows = (padding_last_dim_to_8x) ? ((rows + 7) & -8) : rows;
  const int padded_cols = (padding_last_dim_to_8x) ? ((cols + 7) & -8) : cols;
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
        // 如果padding连续维为8的倍数，则将连续维的stride使用位运算向上取整为最近的8倍数
        const int g_output_inner_stride = padded_rows;
        const int g_scale_inner_stride = (rows + 127) / 128;
        if (g_output_y_offset < cols &&
            g_output_x_offset < g_output_inner_stride) {
          out[g_output_y_offset * g_output_inner_stride + g_output_x_offset] =
              (g_output_x_offset < rows) ? output_scaled_fp8 : (OutT)0;
          scales[g_output_y_offset * g_scale_inner_stride +
                 g_output_x_offset / 128] = scale_on_fp8_to_inputT;
        }
      } else {
        const int g_output_y_offset = g_block_y_offset + y_offset;
        const int g_output_x_offset = g_block_x_offset + x_offset;
        const int g_scale_inner_stride = (cols + 127) / 128;
        const int g_output_inner_stride = padded_cols;
        if (g_output_y_offset < rows &&
            g_output_x_offset < g_output_inner_stride) {
          out[g_output_y_offset * g_output_inner_stride + g_output_x_offset] =
              (g_output_x_offset < cols) ? output_scaled_fp8 : (OutT)0;
          scales[g_output_y_offset * g_scale_inner_stride +
                 g_output_x_offset / 128] = scale_on_fp8_to_inputT;
        }
      }
    }
  }
}

template <typename outT>
void dispatch_fused_act_quant(const paddle::Tensor &X,
                              paddle::Tensor &out,
                              paddle::Tensor &scale,
                              const int rows,
                              const int cols,
                              const bool &transpose_output,
                              const bool &using_pow2_scaling,
                              const bool &padding_last_dim_to_8x,
                              const float TYPE_MAX) {
  dim3 grid;
  dim3 block(32, 32);
  DISPATCH_BOOL(
      transpose_output,
      k_transpose_output,
      DISPATCH_BOOL(
          using_pow2_scaling,
          k_using_pow2_scaling,
          DISPATCH_BOOL(padding_last_dim_to_8x, k_padding_last_dim_to_8x, {
            grid.y = (rows + 127) / 128;
            grid.x = (cols + 127) / 128;
            LAUNCH_FUSED_ACT_QUANT(k_transpose_output,
                                   k_using_pow2_scaling,
                                   k_padding_last_dim_to_8x);
          })))
}
std::vector<paddle::Tensor> fused_act_quant(const paddle::Tensor &X,
                                            const bool &transpose_output,
                                            const bool &padding_last_dim_to_8x,
                                            const bool &using_pow2_scaling) {
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16);
  int64_t data_rows = size_to_dim(X.shape().size() - 1, X.shape());
  int64_t data_cols = X.shape().back();
  int64_t rows = data_rows, cols = data_cols;

  paddle::Tensor out;
  paddle::Tensor scale;

  if (transpose_output) {
    if (padding_last_dim_to_8x) {
      rows = ((data_rows + 7) / 8) *
             8;  // 向上padding到8的倍数, 因为128为8的倍数，不影响scale shape
    }
    out =
        paddle::empty({cols, rows}, paddle::DataType::FLOAT8_E4M3FN, X.place());
    scale = paddle::empty(
        {cols, (rows + 127) / 128}, paddle::DataType::FLOAT32, X.place());
  } else {
    if (padding_last_dim_to_8x) {
      cols = ((data_cols + 7) / 8) *
             8;  // 向上padding到8的倍数, 因为128为8的倍数，不影响scale shape
    }
    out =
        paddle::empty({rows, cols}, paddle::DataType::FLOAT8_E4M3FN, X.place());
    scale = paddle::empty(
        {rows, (cols + 127) / 128}, paddle::DataType::FLOAT32, X.place());
  }

  dispatch_fused_act_quant<phi::float8_e4m3fn>(
      X,
      out,
      scale,
      data_rows,
      data_cols,
      transpose_output,
      using_pow2_scaling,
      padding_last_dim_to_8x,
      F8LimitsTrait<__nv_fp8_e4m3>::max);
  return {out, scale};
}

PD_BUILD_OP(fused_act_quant)
    .Inputs({"X"})
    .Outputs({"output", "scale"})
    .Attrs({"transpose_output: bool",
            "padding_last_dim_to_8x: bool",
            "using_pow2_scaling: bool"})
    .SetKernelFn(PD_KERNEL(fused_act_quant));
