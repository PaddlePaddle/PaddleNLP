#include "quant_utils.h"

template <typename T, int VecSize>
struct __align__(sizeof(T) * VecSize) VecType {
  T val[VecSize];
  __host__ __device__ inline T& operator[](size_t i) { return val[i]; }
  __host__ __device__ inline const T& operator[](size_t i) const {
    return val[i];
  }
};

template <typename OutT, int VecSize>
__device__ void StoreQuantResult(OutT* out,
                                 const OutT shm[][129],
                                 int64_t L,
                                 int64_t C) {
  for (int i = 0; i < 4; i++) {
    int64_t idx_n = blockIdx.z;
    int64_t idx_h =
        static_cast<int64_t>(blockIdx.x) * 128 + (i * 32 + threadIdx.y);
    int64_t idx_l = static_cast<int64_t>(blockIdx.y) * 128;
    int64_t idx = (idx_n * C + idx_h) * L + idx_l;

    for (int j = 0; j < 4; j += VecSize) {
      int64_t block_offset = j * 32 + threadIdx.x * VecSize;
      if (idx_l + block_offset < L) {
        using StoreT = VecType<OutT, VecSize>;
        StoreT data;
        for (int k = 0; k < VecSize; k++) {
          data[k] = shm[i * 32 + threadIdx.y][block_offset + k];
        }
        *reinterpret_cast<StoreT*>(out + (idx + block_offset)) = data;
      }
    }
  }
}

__device__ int GetVecSize(int64_t n) {
  return n % 4 == 0 ? 4 : (n % 2 == 0 ? 2 : 1);
}

template <typename OutT>
__global__ void __launch_bounds__(1024)
    FusedTransposeQuantKernel(const phi::bfloat16* __restrict__ X,
                              OutT* __restrict__ out,
                              float* __restrict__ scale,
                              int64_t L,
                              int64_t C) {
  __shared__ OutT shm[128][129];

  int64_t offset_n = blockIdx.z;
  int64_t offset_l = static_cast<int64_t>(blockIdx.y) * 128 + threadIdx.y;
  int64_t offset_h = static_cast<int64_t>(blockIdx.x) * 128 + threadIdx.x * 4;
  int64_t offset = (offset_n * L + offset_l) * C + offset_h;

  for (int i = 0; i < 4; i++) {
    int64_t idx = offset + i * 32 * C;

    if (offset_l + i * 32 < L) {
      // Load x [N, L, C], 128 elements per warp
      using LoadT = VecType<__nv_bfloat16, 4>;
      const LoadT input = *reinterpret_cast<const LoadT*>(X + idx);

      float input_fp32[4];
      for (int j = 0; j < 4; j++) {
        input_fp32[j] = static_cast<float>(input[j]);
      }

      // Find the maximum of every 128 elements in dim C.
      float max = fabsf(input_fp32[0]);
      for (int j = 1; j < 4; j++) {
        max = fmaxf(max, fabsf(input_fp32[j]));
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        max = fmaxf(max, __shfl_xor_sync(0xffffffff, max, offset));
      }

      // Compute the scale of max.
      const float scale_on_fp32_to_outputT =
          ComputeScale<__nv_bfloat16, OutT>(max, 0.0f);
      const float scale_on_fp8_to_inputT = __frcp_rn(scale_on_fp32_to_outputT);

      // Scale X and transpose into shared memory.
      for (int j = 0; j < 4; j++) {
        float output_scaled = input_fp32[j] * scale_on_fp8_to_inputT;
        shm[threadIdx.x * 4 + j][i * 32 + threadIdx.y] =
            static_cast<OutT>(output_scaled);
      }

      // Store scale [N, L, C/128].
      if (threadIdx.x == 0) {
        scale[idx / 128] = scale_on_fp32_to_outputT;
      }
    }
  }

  __syncthreads();

  // Store y [N, C, L]
  int vec_size = GetVecSize(L);
  if (vec_size == 4) {
    StoreQuantResult<OutT, 4>(out, shm, L, C);
  } else if (vec_size == 2) {
    StoreQuantResult<OutT, 2>(out, shm, L, C);
  } else {
    StoreQuantResult<OutT, 1>(out, shm, L, C);
  }
}

/**
 * Doing fused transpose and quant.
 *
 * Inputs:
 *   X    : [*, L, C], bfloat16
 *
 * Outputs:
 *   out  : [*, C, L], float8
 *   scale: [*, L, C/128], float32
 *
 * Equivalent python code:
 * def fused_transpose_quant(x):
 *     N, L, C = paddle.shape(x)
 *     x = x.reshape([N, L, C // 128, 128]).astype('float32')
 *     scale = ComputeScale(x.abs().max(axis=-1))
 *     x = (x / scale.unsqueeze(-1)).astype('float8_e4m3fn')
 *     out = x.reshape([N, L, C]).transpose(0, 2, 1)
 *     return out, scale
 */
std::vector<paddle::Tensor> fused_transpose_quant(const paddle::Tensor& X) {
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16);

  std::vector<int64_t> shape = X.shape();
  PD_CHECK(shape.size() >= 2);

  int64_t seq_len = shape[shape.size() - 2];
  int64_t hidden_size = shape[shape.size() - 1];
  int64_t batch_size = X.numel() / (seq_len * hidden_size);

  PADDLE_ENFORCE_LE(batch_size,
                    65535,
                    common::errors::InvalidArgument(
                        "The batch size (shape[0:-2]) of Input(X) must be no "
                        "larger than 65535."));
  PADDLE_ENFORCE_LE(seq_len,
                    65535 * 128,
                    common::errors::InvalidArgument(
                        "The sequence length (shape[-2]) of Input(X) must be "
                        "no larger than 65535 * 128."));
  PADDLE_ENFORCE_LE(hidden_size,
                    ((1L << 31) - 1) * 128,
                    common::errors::InvalidArgument(
                        "The hidden size (shape[-1]) of Input(X) must be no "
                        "larger than (2^31 - 1) * 128."));
  PADDLE_ENFORCE_EQ(hidden_size % 128,
                    0,
                    common::errors::InvalidArgument(
                        "The hidden size (shape[-1]) of Input(X) must be "
                        "multiple of 128."));

  // Allocate for out and scale
  std::vector<int64_t> out_shape = shape;
  out_shape[shape.size() - 2] = hidden_size;
  out_shape[shape.size() - 1] = seq_len;
  paddle::Tensor out =
      paddle::empty(out_shape, paddle::DataType::FLOAT8_E4M3FN, X.place());

  std::vector<int64_t> scale_shape = shape;
  scale_shape[shape.size() - 1] = hidden_size / 128;
  paddle::Tensor scale =
      paddle::empty(scale_shape, paddle::DataType::FLOAT32, X.place());

  // Skip 0-size
  if (batch_size == 0 || seq_len == 0 || hidden_size == 0) {
    return {out, scale};
  }

  // Launch kernel
  dim3 grid(hidden_size / 128, (seq_len + 127) / 128, batch_size);
  dim3 block(32, 32);

  FusedTransposeQuantKernel<<<grid, block>>>(X.data<phi::bfloat16>(),
                                             out.data<phi::float8_e4m3fn>(),
                                             scale.data<float>(),
                                             seq_len,
                                             hidden_size);

  return {out, scale};
}

PD_BUILD_OP(fused_transpose_quant)
    .Inputs({"X"})
    .Outputs({"output", "scale"})
    .SetKernelFn(PD_KERNEL(fused_transpose_quant));
