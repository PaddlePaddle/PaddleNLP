// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

// This code is partially inspired by and references the implementation found
// in FlashInfer.Specifically, the implementation of Top-p Sampling functionality
// in this code is inspired by the logic of
// FlashInfer’s flashinfer.sampling.top_p_sampling_from_probs .
// For more details on FlashInfer’s documentation, please refer to:
// https://docs.flashinfer.ai/generated/flashinfer.sampling.top_p_sampling_from_probs.html

#pragma once

#include <cub/block/block_adjacent_difference.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <numeric>

#include "sample_kernels/utils.cuh"


namespace sampling {

using namespace cub;

#define DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, ...) \
  if (compute_capacity.first >= 8) {                                           \
    constexpr uint32_t BLOCK_THREADS = 1024;                                   \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    constexpr uint32_t BLOCK_THREADS = 512;                                    \
    __VA_ARGS__                                                                \
  }

constexpr BlockScanAlgorithm SCAN_ALGO = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120100)
#define SAMPLING_CUB_SUBTRACTLEFT_DEFINED
#endif

template <typename T>
struct Pair {
  T value;
  int count;

  __device__ Pair operator+(const Pair& other) const {
    return {value + other.value, count + other.count};
  }
  __device__ Pair& operator+=(const Pair& other) {
    value += other.value;
    count += other.count;
    return *this;
  }
};

struct BoolDiffOp {
  __device__ __forceinline__ bool operator()(const bool& lhs,
                                             const bool& rhs) const {
    return lhs != rhs;
  }
};

template <typename T,
          uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
struct SamplingTempStorage {
  union {
    T deterministic_scan[BLOCK_THREADS / 32];
    typename BlockScan<T, BLOCK_THREADS, SCAN_ALGORITHM>::TempStorage scan;
    typename BlockReduce<T, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage
        reduce;
    typename BlockReduce<Pair<T>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage
        reduce_pair;
    typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
  } block_prim;
  struct {
    int32_t sampled_id;
    union {
      T value;
      Pair<T> pair;
      T max_p;
    } block_aggregate;
  } data;
};

/*!
 * \brief Deterministic inclusive scan implementation, use Belloch scan
 * algorithm. \note This implementation is slower than the cub::BlockScan, but
 * it is deterministic.
 */
template <uint32_t VEC_SIZE,
          uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM,
          typename T>
__device__ __forceinline__ void DeterministicInclusiveSum(
    const T* in_data,
    T* out_data,
    SamplingTempStorage<T, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>*
        temp_storage) {
  T* smem_prefix_sum = temp_storage->block_prim.deterministic_scan;
  T thread_data[VEC_SIZE];
  T thread_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    thread_sum += in_data[i];
    thread_data[i] = thread_sum;
  }

  T thread_exclusive_prefix_sum = thread_sum;

#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    T tmp = __shfl_up_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum += tmp;
    }
  }

  T warp_sum = __shfl_sync(
      0xffffffff, thread_exclusive_prefix_sum, threadIdx.x | 0xffffffff);
  if (threadIdx.x % 32 == 31) {
    thread_exclusive_prefix_sum = 0;
  }

#pragma unroll
  for (uint32_t offset = 16; offset >= 1; offset /= 2) {
    T tmp = __shfl_xor_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum = tmp + thread_exclusive_prefix_sum;
    }
    if ((threadIdx.x + 1) % (offset * 2) == offset) {
      thread_exclusive_prefix_sum = tmp;
    }
  }

  smem_prefix_sum[threadIdx.x / 32] = warp_sum;
  __syncthreads();

  if (threadIdx.x < 32) {
    T warp_exclusive_prefix_sum =
        (threadIdx.x < BLOCK_THREADS / 32) ? smem_prefix_sum[threadIdx.x] : 0;

#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset *= 2) {
      T tmp = __shfl_up_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum += tmp;
      }
    }

    if (threadIdx.x % 32 == 31) {
      warp_exclusive_prefix_sum = 0;
    }

#pragma unroll
    for (uint32_t offset = 16; offset >= 1; offset /= 2) {
      T tmp = __shfl_xor_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum = tmp + warp_exclusive_prefix_sum;
      }
      if ((threadIdx.x + 1) % (offset * 2) == offset) {
        warp_exclusive_prefix_sum = tmp;
      }
    }
    if (threadIdx.x < BLOCK_THREADS / 32) {
      smem_prefix_sum[threadIdx.x] = warp_exclusive_prefix_sum;
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    out_data[i] = smem_prefix_sum[threadIdx.x / 32] +
                  thread_exclusive_prefix_sum + thread_data[i];
  }
}

template <uint32_t VEC_SIZE,
          uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM,
          bool DETERMINISTIC,
          typename T>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i,
    uint32_t d,
    T threshold,
    T u,
    vec_t<T, VEC_SIZE> prob_vec,
    T& aggregate,
    SamplingTempStorage<T, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>*
        temp_storage) {
  const uint32_t tx = threadIdx.x;
  T prob_greater_than_threshold[VEC_SIZE];
  T inclusive_cdf[VEC_SIZE];
  bool greater_than_u[VEC_SIZE], valid[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    prob_greater_than_threshold[j] =
        (prob_vec[j] > threshold) ? prob_vec[j] : T(0);
    valid[j] =
        prob_vec[j] > threshold && (i * BLOCK_THREADS + tx) * VEC_SIZE < d;
  }
  T aggregate_local = BlockReduce<T, BLOCK_THREADS, REDUCE_ALGORITHM>(
                          temp_storage->block_prim.reduce)
                          .Sum<VEC_SIZE>(prob_greater_than_threshold);
  if (tx == 0) {
    temp_storage->data.block_aggregate.value = aggregate_local;
  }
  __syncthreads();
  aggregate_local = temp_storage->data.block_aggregate.value;

  if (aggregate + aggregate_local > u) {
    if constexpr (DETERMINISTIC) {
      DeterministicInclusiveSum<VEC_SIZE,
                                BLOCK_THREADS,
                                SCAN_ALGORITHM,
                                REDUCE_ALGORITHM,
                                T>(
          prob_greater_than_threshold, inclusive_cdf, temp_storage);
    } else {
      BlockScan<T, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
          .InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);

      __syncthreads();
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      greater_than_u[j] = inclusive_cdf[j] + aggregate > u;
    }

    bool greater_than_u_diff[VEC_SIZE];
#ifdef SAMPLING_CUB_SUBTRACTLEFT_DEFINED
    BlockAdjacentDifference<bool, BLOCK_THREADS>(
        temp_storage->block_prim.adj_diff)
        .SubtractLeft<VEC_SIZE>(
            greater_than_u, greater_than_u_diff, BoolDiffOp());
#else
    BlockAdjacentDifference<bool, BLOCK_THREADS>(
        temp_storage->block_prim.adj_diff)
        .FlagHeads<VEC_SIZE>(
            greater_than_u_diff, greater_than_u, BoolDiffOp(), 0);
#endif
    __syncthreads();

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      if (greater_than_u_diff[j] && valid[j]) {
        if constexpr (DETERMINISTIC) {
          temp_storage->data.sampled_id =
              (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
        } else {
          // cub's block scan result might not be monotonic, so we need to find
          // the first element
          atomicMin(&(temp_storage->data.sampled_id),
                    (i * BLOCK_THREADS + tx) * VEC_SIZE + j);
        }
      }
    }
    __syncthreads();
  }
  aggregate += aggregate_local;
}

template <uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM,
          uint32_t VEC_SIZE,
          bool DETERMINISTIC,
          typename DType,
          typename IdType>
__global__ void TopPSamplingFromProbKernel(DType* probs,
                                           DType* uniform_samples,
                                           IdType* output,
                                           float* top_p_val,
                                           uint32_t d,
                                           uint32_t max_top_p_rounds) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  float top_p =  top_p_val[bx];

  extern __shared__ __align__(alignof(SamplingTempStorage<DType,
                                                          BLOCK_THREADS,
                                                          SCAN_ALGORITHM,
                                                          REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<DType,
                                           BLOCK_THREADS,
                                           SCAN_ALGORITHM,
                                           REDUCE_ALGORITHM>&>(smem_sampling);

  vec_t<DType, VEC_SIZE> probs_vec;
  DType aggregate;
  DType q = DType(1);
  DType pivot = DType(0);
  IdType sampled_id;
  for (uint32_t round = 0; round < max_top_p_rounds; ++round) {
    temp_storage.data.sampled_id = d - 1;
    __syncthreads();
    DType u = uniform_samples[round * batch_size + bx] * q;
    aggregate = DType(0);
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(DType(0));
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.load(probs + bx * d +
                       (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE,
                             BLOCK_THREADS,
                             SCAN_ALGORITHM,
                             REDUCE_ALGORITHM,
                             DETERMINISTIC,
                             DType>(
          i, d, pivot, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = temp_storage.data.sampled_id;
    pivot = max(pivot, probs[bx * d + sampled_id]);

    Pair<DType> aggregate_gt_pivot{DType(0), 0};
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(DType(0));
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.load(probs + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      Pair<DType> probs_gt_pivot[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot[j] = {(probs_vec[j] > pivot) ? probs_vec[j] : DType(0),
                             (probs_vec[j] > pivot && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
      }

      aggregate_gt_pivot += BlockReduce<Pair<DType>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                temp_storage.block_prim.reduce_pair)
                                .Sum<VEC_SIZE>(probs_gt_pivot);
      if (tx == 0) {
        temp_storage.data.block_aggregate.pair = aggregate_gt_pivot;
      }
      __syncthreads();
    }
    q = temp_storage.data.block_aggregate.pair.value;
    if (float(q) > 0 && float(q) < top_p) {
      // top_p is not 0
      break;
    } else {
      // top_p is 0, use top_k, k=1
      if (temp_storage.data.block_aggregate.pair.count < 1) {
        break;
      }
    }
  }
  __syncthreads();
  if (tx == 0) {
    output[bx] = sampled_id;
  }
}

template <typename T, typename IdType>
cudaError_t TopPSamplingFromProb(T* probs,
                                 T* uniform_samples,
                                 IdType* output,
                                 uint32_t batch_size,
                                 const T* top_p_val,
                                 uint32_t d,
                                 uint32_t max_top_p_rounds,
                                 bool deterministic,
                                 cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t smem_size =
      sizeof(SamplingTempStorage<T, BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs,
                  &uniform_samples,
                  &output,
                  &top_p_val,
                  &d,
                  &max_top_p_rounds};

  DISPATCH_ALIGNED_VEC_SIZE(
      vec_size,
      VEC_SIZE,
      {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
        auto kernel = TopPSamplingFromProbKernel<BLOCK_THREADS,
                                                 SCAN_ALGO,
                                                 REDUCE_ALGO,
                                                 VEC_SIZE,
                                                 DETERMINISTIC,
                                                 T,
                                                 IdType>;
        CUDA_CALL(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        CUDA_CALL(cudaLaunchKernel(
            (void*)kernel, nblks, nthrs, args, smem_size, stream));
      })});
  return cudaSuccess;
}

}  // namespace sampling