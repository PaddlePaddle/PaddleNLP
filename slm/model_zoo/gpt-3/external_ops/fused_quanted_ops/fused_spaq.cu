#include "quant_utils.h"

#define LAUNCH_FUSED_SPAQ(__using_pow2_scaling)                     \
  do {                                                              \
    auto kernel = FusedSPAQKernel<__using_pow2_scaling, with_prob>; \
    kernel<<<grid, block, 0, X.stream()>>>(                         \
        X.data<phi::bfloat16>(),                                    \
        prob ? prob->data<float>() : nullptr,            \
        out.data<phi::float8_e4m3fn>(),                             \
        scale.data<float>(),                                        \
        rows,                                                       \
        cols);                                                      \
  } while (0)


__device__ __forceinline__ float fast_swiglu(const __nv_bfloat16 x,
                                             const __nv_bfloat16 y) {
  const float x_f = __bfloat162float(x);
  const float y_f = __bfloat162float(y);
  const float silu = x_f * __frcp_rn(1.0f + __expf(-x_f));
  const float result = silu * y_f;
  return result;
}

template <bool using_pow2_scaling, bool with_prob>
__global__ void FusedSPAQKernel(const phi::bfloat16 *__restrict__ Xin,
                                const float *__restrict__ prob,
                                phi::float8_e4m3fn *__restrict__ out,
                                float *__restrict__ scales,
                                const int rows,
                                const int cols) {
  // Configure shared memory
  __shared__ float smem_tile[256];  // Shared memory for activation values
  __shared__ float warp_max[2][4];  // Shared memory for warp maxima (2 quant
                                    // blocks x 4 warps)
  __shared__ __nv_bfloat16
      quant_block_amax[2];  // Shared memory for quant block maxima

  const __nv_bfloat16 *X = reinterpret_cast<const __nv_bfloat16 *>(Xin);
  const int x_offset = threadIdx.x;
  const int quant_block_idx =
      threadIdx.x / 128;  // 0 or 1, two quant blocks per block
  const int in_y_idx = blockIdx.y;
  const int in_x_idx = blockIdx.x * blockDim.x + x_offset;
  const int src_idx = in_y_idx * cols + in_x_idx;

  // Load data and compute swiGLU activation
  if (in_x_idx < cols / 2) [[likely]] {
    __nv_bfloat16 x1 = X[src_idx];             // First half of the input
    __nv_bfloat16 x2 = X[src_idx + cols / 2];  // Second half of the input

    if constexpr (with_prob) {
      float row_prob = prob[in_y_idx];
      smem_tile[x_offset] = fast_swiglu(x1, x2) * row_prob;
    } else {
      smem_tile[x_offset] = fast_swiglu(x1, x2);
    }
  }

  __syncthreads();  // Ensure all threads have loaded their data

  // Phase 2: Block Reduction to find per-quant block absolute maximums
  float local_max = (in_x_idx < (cols / 2)) ? fabsf(smem_tile[x_offset]) : 0.0f;


  // Warp-level reduction
  unsigned int mask = 0xffffffff;
  int lane = threadIdx.x % 32;
  int warp_id =
      (threadIdx.x % 128) / 32;  // Warp ID within the quant block (0-3)

  // Reduce within the warp
  for (int offset = 16; offset > 0; offset /= 2) {
    float val = __shfl_down_sync(mask, local_max, offset);
    local_max = fmaxf(local_max, val);
  }

  // Store warp maxima
  if (lane == 0) {
    warp_max[quant_block_idx][warp_id] = local_max;
  }

  __syncthreads();

  // Reduce warp maxima to get quant block maxima
  if (warp_id == 0 && lane < 4) {
    if (threadIdx.x < 256) {  // Ensure only valid threads participate
      float block_max = warp_max[quant_block_idx][lane];
      // Reduce over the 4 warp maxima
      if (lane == 0) {
        block_max = fmaxf(block_max, warp_max[quant_block_idx][1]);
        block_max = fmaxf(block_max, warp_max[quant_block_idx][2]);
        block_max = fmaxf(block_max, warp_max[quant_block_idx][3]);
        quant_block_amax[quant_block_idx] = __float2bfloat16(block_max);
      }
    }
  }

  __syncthreads();

  // Phase 3: Compute scales and quantize the outputs
  const float block_max_float = (float)quant_block_amax[quant_block_idx];
  const int scale_stride = (cols / 2 + 127) / 128;

  float scale = ComputeScale<float, __nv_fp8_e4m3, using_pow2_scaling>(
      block_max_float, 0.0f);
  float inv_scale = __frcp_rn(scale);

  // Quantize
  float output_scaled_fp32 = smem_tile[x_offset] * scale;


  const int g_output_y_offset = in_y_idx;
  const int g_output_x_offset = in_x_idx;

  // Write output and scales
  if (g_output_y_offset < rows && g_output_x_offset < cols / 2) {
    out[g_output_y_offset * (cols / 2) + g_output_x_offset] =
        static_cast<phi::float8_e4m3fn>(output_scaled_fp32);
    if (x_offset % 128 == 0) {
      // Only one thread per quant block writes the scale
      scales[g_output_y_offset * scale_stride + in_x_idx / 128] = inv_scale;
    }
  }
}

template <bool with_prob>
void dispatch_fused_spaq(const paddle::Tensor &X,
                         const paddle::optional<paddle::Tensor> &prob,
                         paddle::Tensor &out,
                         paddle::Tensor &scale,
                         const int rows,
                         const int cols,
                         const bool &using_pow2_scaling) {
  dim3 grid;
  dim3 block;
  // parallel strategy:
  // each block processing a row of the input tensor.
  block.x = 256;
  DISPATCH_BOOL(using_pow2_scaling, k_using_pow2_scaling, grid.y = rows;
                grid.x = ((cols / 2) + block.x - 1) / block.x;
                LAUNCH_FUSED_SPAQ(k_using_pow2_scaling);)
}
/*
PD_BUILD_OP(fused_spaq)
    .Inputs({"X", "prob"})
    .Outputs({"output", "scale"})
    .Attrs({"using_pow2_scaling: bool"})
    .SetKernelFn(PD_KERNEL(fused_spaq));
*/
std::vector<paddle::Tensor> fused_spaq(
    const paddle::Tensor &X,
    const paddle::optional<paddle::Tensor> &prob,
    const bool &using_pow2_scaling) {
  // ---------------- Arguments check --------------------
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16);
  if (prob) PD_CHECK(prob.get().dtype() == paddle::DataType::FLOAT32);
  int64_t rows = size_to_dim(X.shape().size() - 1, X.shape());
  int64_t cols = X.shape().back();
  PADDLE_ENFORCE_EQ(cols % 2,
                    0,
                    common::errors::InvalidArgument(
                        "The last dim of Input(X) should be exactly divided "
                        "by 2 , but got %d",
                        cols));
  if(prob){
  PADDLE_ENFORCE_EQ(
      prob.get().shape()[0],
      rows,
      common::errors::InvalidArgument(
          "The first dim of Input(X) should be equal to the "
          "first dim of Input(prob) but got X.shape[0]: %d, prob.shape[0]: %d",
          rows,
          prob.get().shape()[0]));
  }

  paddle::Tensor out;
  paddle::Tensor scale;

  out = paddle::empty(
      {rows, cols / 2}, paddle::DataType::FLOAT8_E4M3FN, X.place());
  scale = paddle::empty(
      {rows, ((cols / 2) + 127) / 128}, paddle::DataType::FLOAT32, X.place());

  if (prob) {
    dispatch_fused_spaq<true>(
        X, prob, out, scale, rows, cols, using_pow2_scaling);
  }
  else {
    dispatch_fused_spaq<false>(
        X, prob, out, scale, rows, cols, using_pow2_scaling);
  }
    return {out, scale};
  }

  PD_BUILD_OP(fused_spaq)
      .Inputs({"X", paddle::Optional("prob")})
      .Outputs({"output", "scale"})
      .Attrs({"using_pow2_scaling: bool"})
      .SetKernelFn(PD_KERNEL(fused_spaq));
