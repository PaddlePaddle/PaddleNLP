#include "quant_utils.h"

constexpr int64_t TILE_SIZE = 128;  // 每个 block 处理 128x128 的元素块
#define LAUNCH_FUSED_ACT_DEQUANT_TRANSPOSE_ACT_QUANT(__using_pow2_scaling,     \
                                                     __padding_last_dim_to_8x) \
  do {                                                                         \
    auto kernel = FusedActDequantTransposeActQuant<outT,                       \
                                                   __using_pow2_scaling,       \
                                                   __padding_last_dim_to_8x>;  \
    int smem_size = 128 * 129 * sizeof(float);                                 \
    PD_CHECK(cudaFuncSetAttribute(kernel,                                      \
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                  smem_size) == cudaSuccess);                  \
    kernel<<<grid, block, smem_size, X.stream()>>>(                            \
        X.data<phi::float8_e4m3fn>(),                                          \
        Xscale.data<float>(),                                                  \
        out.data<outT>(),                                                      \
        scale.data<float>(),                                                   \
        rows,                                                                  \
        cols,                                                                  \
        TYPE_MAX);                                                             \
  } while (0)


#define BLOCK_SIZE 128
template <typename OutT, bool using_pow2_scaling, bool padding_last_dim_to_8x>
__global__ void FusedActDequantTransposeActQuant(
    const phi::float8_e4m3fn *__restrict__ Xin,
    const float *__restrict__ Xscale,
    OutT *__restrict__ out,
    float *__restrict__ scales,
    const int rows,  // 此处的row未带padding
    const int cols,  // 同上
    const int TYPE_MAX) {
  // 共享内存用于:1.直接读取+存储转置后的数据  2.用于1x128规约+量化
  // 共享内存布局,前128x128为计算结果，后128为scale(复用)
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  float *smem_tile = reinterpret_cast<float *>(smem_buffer);
  float *smem_max =
      reinterpret_cast<float *>(smem_buffer) + BLOCK_SIZE * BLOCK_SIZE;

  const int g_block_y_offset = blockIdx.y * BLOCK_SIZE;  // 块内行坐标偏移
  const int g_block_x_offset = blockIdx.x * BLOCK_SIZE;  // 块内列坐标偏移
  const __nv_fp8_e4m3 *X = reinterpret_cast<const __nv_fp8_e4m3 *>(Xin);
  // 阶段0:
  // 原始fp8 scale读入smem_max，用于后续dequant
  // ------------------------------
  if (threadIdx.y == 0) {
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
      smem_max[i] = Xscale[i];
    }
  }
  __syncthreads();  // smem_tile中的Xscale数据已ready

  // 阶段1:
  // Elementwise加载数据、计算反量化、并将结果直接以特定布局存入共享内存‌(32x32)
  // ------------------------------
  for (int y_offset = threadIdx.y; y_offset < BLOCK_SIZE;
       y_offset += blockDim.y) {
    float row_scale = smem_max[y_offset];
    for (int x_offset = threadIdx.x; x_offset < BLOCK_SIZE;
         x_offset += blockDim.x) {
      const int in_y_idx = g_block_y_offset + y_offset;
      const int in_x_idx = g_block_x_offset + x_offset;
      const int src_idx = in_y_idx * cols + in_x_idx;
      // 类型提升、dequant、共享内存swizzle写数
      if (in_y_idx < rows && in_x_idx < cols) {
        const int src_idx = in_y_idx * cols + in_x_idx;
        smem_tile[swizzled_2d_idx(x_offset, BLOCK_SIZE, y_offset)] =
            static_cast<float>(X[src_idx]) * row_scale;
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
      is_output_inner_OOB = (g_block_y_offset + x_offset) >= rows;
      if (is_output_inner_OOB) break;  // 列越界则不取值、不影响max;
      local_max = fmaxf(
          local_max,
          fabsf(smem_tile[swizzled_2d_idx(
              y_offset,
              BLOCK_SIZE,
              x_offset)]));  // 正常情况下提供绝对值给其他线程,每个线程的local_max最多为4数最大值
    }
    bool is_output_outer_OOB;
    is_output_outer_OOB = (g_block_x_offset + y_offset) >= cols;
    if (is_output_outer_OOB) break;  // 行越界，不对该行做量化
    local_max =
        warpReduceMax(local_max);  // 无论该block有多少个有效线程，max均合法
    if (threadIdx.x == 0)
      smem_max[y_offset] = local_max;  // x0 顺序写，复用，无conflict
  }

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
    }
  }
}

template <typename outT>
void dispatch_fused_act_dequant_transpose_act_quant(
    const paddle::Tensor &X,
    const paddle::Tensor &Xscale,
    paddle::Tensor &out,
    paddle::Tensor &scale,
    const int rows,
    const int cols,
    const bool &using_pow2_scaling,
    const bool &padding_last_dim_to_8x,
    const float TYPE_MAX) {
  dim3 grid;
  dim3 block(32, 32);
  DISPATCH_BOOL(
      using_pow2_scaling,
      k_using_pow2_scaling,
      DISPATCH_BOOL(padding_last_dim_to_8x, k_padding_last_dim_to_8x, {
        grid.y = (rows + 127) / 128;
        grid.x = (cols + 127) / 128;
        LAUNCH_FUSED_ACT_DEQUANT_TRANSPOSE_ACT_QUANT(k_using_pow2_scaling,
                                                     k_padding_last_dim_to_8x);
      }))
}
std::vector<paddle::Tensor> fused_act_dequant_transpose_act_quant(
    const paddle::Tensor &X,
    const paddle::Tensor &Xscale,
    const bool &padding_last_dim_to_8x,
    const bool &using_pow2_scaling) {
  PD_CHECK(X.dtype() == paddle::DataType::FLOAT8_E4M3FN);
  PD_CHECK(Xscale.dtype() == paddle::DataType::FLOAT32);

  int64_t data_rows = size_to_dim(X.shape().size() - 1, X.shape());
  int64_t data_cols = X.shape().back();
  int64_t rows = data_rows, cols = data_cols;

  paddle::Tensor out;
  paddle::Tensor scale;

  if (padding_last_dim_to_8x) {
    rows = ((data_rows + 7) / 8) *
           8;  // 向上padding到8的倍数, 因为128为8的倍数，不影响scale shape
  }
  out = paddle::empty({cols, rows}, paddle::DataType::FLOAT8_E4M3FN, X.place());
  scale = paddle::empty(
      {cols, (rows + 127) / 128}, paddle::DataType::FLOAT32, X.place());

  dispatch_fused_act_dequant_transpose_act_quant<phi::float8_e4m3fn>(
      X,
      Xscale,
      out,
      scale,
      data_rows,
      data_cols,
      using_pow2_scaling,
      padding_last_dim_to_8x,
      F8LimitsTrait<__nv_fp8_e4m3>::max);
  return {out, scale};
}

PD_BUILD_OP(fused_act_dequant_transpose_act_quant)
    .Inputs({"X", "Xscale"})
    .Outputs({"output", "scale"})
    .Attrs({"padding_last_dim_to_8x: bool", "using_pow2_scaling: bool"})
    .SetKernelFn(PD_KERNEL(fused_act_dequant_transpose_act_quant));
