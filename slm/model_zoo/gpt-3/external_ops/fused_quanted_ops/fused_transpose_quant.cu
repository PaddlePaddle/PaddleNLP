#include "quant_utils.h"

template <typename T, int VecSize>
struct __align__(sizeof(T) * VecSize) VecType {
  T val[VecSize];
  __host__ __device__ inline T& operator[](size_t i) { return val[i]; }
  __host__ __device__ inline const T& operator[](size_t i) const {
    return val[i];
  }
};

template <int VecSize>
__device__ void BlockLoad(const phi::bfloat16* X,
                          __nv_bfloat16 input[4][4],
                          size_t M,
                          size_t K) {
  for (size_t i = 0; i < 4; i++) {
    size_t off_n = blockIdx.z;
    size_t off_m = blockIdx.y * 128 + threadIdx.y + i * 32;
    size_t off_k = blockIdx.x * 128 + threadIdx.x * VecSize;
    size_t offset = (off_n * M + off_m) * K + off_k;

    for (size_t j = 0; j < 4; j += VecSize) {
      if (off_k + j * 32 < K) {
        size_t idx = offset + j * 32;
        using LoadT = VecType<__nv_bfloat16, VecSize>;
        LoadT data = *reinterpret_cast<const LoadT*>(X + idx);
        for (int k = 0; k < VecSize; k++) {
          input[i][j + k] = data[k];
        }
      }
    }
  }
}

__device__ void BlockColumnMax(const __nv_bfloat16 input[4][4],
                               __nv_bfloat16 amax[4],
                               __nv_bfloat16* shm) {
  // Reduce [(4), 32, 32, 4] => [32, 32, 4]
  __nv_bfloat16 warp_max[4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      __nv_bfloat16 t = __habs(input[i][j]);
      warp_max[j] = i == 0 ? t : __hmax(warp_max[j], t);
    }
  }

  // Reduce [(32), 32, 4] => [32, 4]
  for (int i = 0; i < 4; i++) {
    shm[threadIdx.y * 128 + i * 32 + threadIdx.x] = warp_max[i];
  }
  __syncthreads();
  for (int offset = 16; offset > 0; offset /= 2) {
    if (threadIdx.y < offset) {
      for (int i = 0; i < 4; i++) {
        shm[threadIdx.y * 128 + i * 32 + threadIdx.x] =
            __hmax(shm[threadIdx.y * 128 + i * 32 + threadIdx.x],
                   shm[(threadIdx.y + offset) * 128 + i * 32 + threadIdx.x]);
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; i++) {
    amax[i] = shm[i * 32 + threadIdx.x];
  }
}

template <typename OutT, int VecSize>
__device__ void BlockStoreScale(float* scale,
                                __nv_bfloat16 amax[4],
                                float scale_inv[4],
                                size_t M,
                                size_t K) {
  float scale_out[4];
  for (int i = 0; i < 4; i++) {
    scale_inv[i] = ComputeScale<__nv_bfloat16, OutT>(amax[i], 0.0f);
    scale_out[i] = __frcp_rn(scale_inv[i]);
  }
  if (threadIdx.y == 0) {
    size_t off_n = blockIdx.z;
    size_t off_m = blockIdx.y;
    size_t off_k = blockIdx.x * 128 + threadIdx.x * VecSize;
    size_t offset = (off_n * (M / 128) + off_m) * K + off_k;

    for (size_t j = 0; j < 4; j += VecSize) {
      if (off_k + j * 32 < K) {
        size_t idx = offset + j * 32;
        using StoreT = VecType<float, VecSize>;
        StoreT data;
        for (int k = 0; k < VecSize; k++) {
          data[k] = scale_out[j + k];
        }
        *reinterpret_cast<StoreT*>(scale + idx) = data;
      }
    }
  }
}

template <typename OutT, int VecSize>
__device__ void BlockStoreOut(OutT* out,
                              const OutT shm[128][129],
                              size_t M,
                              size_t K) {
  for (size_t i = 0; i < 4; i++) {
    size_t idx_n = blockIdx.z;
    size_t idx_k = blockIdx.x * 128 + threadIdx.y + i * 32;
    size_t idx_m = blockIdx.y * 128 + threadIdx.x * 4;
    size_t idx = (idx_n * K + idx_k) * M + idx_m;

    if (idx_k < K) {
      using StoreT = VecType<OutT, VecSize>;
      StoreT data;
      for (int j = 0; j < VecSize; j++) {
        data[j] = shm[i * 32 + threadIdx.y][threadIdx.x * 4 + j];
      }
      *reinterpret_cast<StoreT*>(out + idx) = data;
    }
  }
}

template <typename OutT, int VecSize>
__global__ void __launch_bounds__(1024, 2)
    FusedTransposeQuantKernel(const phi::bfloat16* __restrict__ X,
                              OutT* __restrict__ out,
                              float* __restrict__ scale,
                              size_t M,
                              size_t K) {
  __shared__ OutT shm[128][129];

  // Load 128x128 elements from X
  __nv_bfloat16 input[4][4];
  BlockLoad<VecSize>(X, input, M, K);

  // Find the maximum of each 128 elements on the M axis
  __nv_bfloat16 amax[4];
  BlockColumnMax(input, amax, reinterpret_cast<__nv_bfloat16*>(shm));

  // Compute scale and scale_inv, then store scale back
  float scale_inv[4];
  BlockStoreScale<OutT, VecSize>(scale, amax, scale_inv, M, K);

  // Scale X and save into shared memory with transposed layout
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j += VecSize) {
      for (int k = 0; k < VecSize; k++) {
        float input_fp32 = static_cast<float>(input[i][j + k]);
        float output_scaled = input_fp32 * scale_inv[j + k];
        shm[threadIdx.x * VecSize + j * 32 + k][i * 32 + threadIdx.y] =
            static_cast<OutT>(output_scaled);
      }
    }
  }
  __syncthreads();

  // Store 128x128 elements back
  // Note: out is always 4x vectorizable.
  BlockStoreOut<OutT, 4>(out, shm, M, K);
}

/**
 * Doing quantization on dim[-2] of X, then transpose dim[-1] and dim[-2] of X.
 *
 * Inputs:
 *   X    : [*, M, K], bfloat16
 *
 * Outputs:
 *   out  : [*, K, M], float8_e4m3fn
 *   scale: [*, M/128, K], float32
 *
 * Requirements:
 *   1) batch_size <= 65535
 *   2) M <= 65535 * 128 and M % 128 == 0
 */
std::vector<paddle::Tensor> fused_transpose_quant(const paddle::Tensor& X) {
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16);

  std::vector<int64_t> shape = X.shape();
  PD_CHECK(shape.size() >= 2);

  int64_t M = shape[shape.size() - 2];
  int64_t K = shape[shape.size() - 1];
  int64_t N = X.numel() / (M * K);

  PADDLE_ENFORCE_LE(
      N,
      65535,
      common::errors::InvalidArgument("The batch size (X.shape[0:-2] in total) "
                                      "must be no larger than 65535."));
  PADDLE_ENFORCE_LE(M,
                    65535 * 128,
                    common::errors::InvalidArgument(
                        "X.shape[-2] must be no larger than 65535 * 128."));
  PADDLE_ENFORCE_EQ(
      M % 128,
      0,
      common::errors::InvalidArgument("X.shape[-2] must be multiple of 128."));

  // Allocate for out and scale
  std::vector<int64_t> out_shape = shape;
  out_shape[shape.size() - 2] = K;
  out_shape[shape.size() - 1] = M;
  paddle::Tensor out =
      paddle::empty(out_shape, paddle::DataType::FLOAT8_E4M3FN, X.place());

  std::vector<int64_t> scale_shape = shape;
  scale_shape[shape.size() - 2] = M / 128;
  paddle::Tensor scale =
      paddle::empty(scale_shape, paddle::DataType::FLOAT32, X.place());

  // Skip 0-size
  if (N == 0 || M == 0 || K == 0) {
    return {out, scale};
  }

  // Launch kernel
  dim3 grid((K + 127) / 128, M / 128, N);
  dim3 block(32, 32);

#define LAUNCH_KERNEL(VEC_SIZE)                           \
  FusedTransposeQuantKernel<phi::float8_e4m3fn, VEC_SIZE> \
      <<<grid, block>>>(X.data<phi::bfloat16>(),          \
                        out.data<phi::float8_e4m3fn>(),   \
                        scale.data<float>(),              \
                        M,                                \
                        K);
  if (K % 4 == 0) {
    LAUNCH_KERNEL(4);
  } else if (K % 2 == 0) {
    LAUNCH_KERNEL(2);
  } else {
    LAUNCH_KERNEL(1);
  }
#undef LAUNCH_KERNEL

  return {out, scale};
}

PD_BUILD_OP(fused_transpose_quant)
    .Inputs({"X"})
    .Outputs({"output", "scale"})
    .SetKernelFn(PD_KERNEL(fused_transpose_quant));
