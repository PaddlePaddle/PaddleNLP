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
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

enum class SharedMemFillMode { kFillZero, kNoFill };

enum class PrefetchMode { kNoPrefetch, kPrefetch };

template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_impl(uint32_t* R, T* smem_ptr) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
}

template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans_impl(uint32_t* R,
                                                           T* smem_ptr) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
}

__device__ __forceinline__ void commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <size_t n>
__device__ __forceinline__ void wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

template <PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load_128b(T* smem_ptr, const T* gmem_ptr) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
            smem_int_ptr),
        "l"(gmem_ptr),
        "n"(16),
        "r"(16));
  } else {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
        "l"(gmem_ptr),
        "n"(16),
        "r"(16));
  }
}

template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load_128b(T* smem_ptr,
                                               const T* gmem_ptr,
                                               bool predicate) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    int src_in_bytes = predicate ? 16 : 0;
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
      asm volatile(
          "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
              smem_int_ptr),
          "l"(gmem_ptr),
          "n"(16),
          "r"(src_in_bytes));
    } else {
      asm volatile(
          "cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
          "l"(gmem_ptr),
          "n"(16),
          "r"(src_in_bytes));
    }
  } else {
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr),
          "l"(gmem_ptr),
          "n"(16));
    } else {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr),
          "l"(gmem_ptr),
          "n"(16));
    }
  }
}

template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load_64b(T* smem_ptr,
                                              const T* gmem_ptr,
                                              bool predicate) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    int src_in_bytes = predicate ? 8 : 0;
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
        "l"(gmem_ptr),
        "n"(8),
        "r"(src_in_bytes));
  } else {
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %0, 0;\n"
        " @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int)predicate),
        "r"(smem_int_ptr),
        "l"(gmem_ptr),
        "n"(8));
  }
}

template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load_32b(T* smem_ptr,
                                              const T* gmem_ptr,
                                              bool predicate) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    int src_in_bytes = predicate ? 4 : 0;
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
        "l"(gmem_ptr),
        "n"(4),
        "r"(src_in_bytes));
  } else {
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %0, 0;\n"
        " @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int)predicate),
        "r"(smem_int_ptr),
        "l"(gmem_ptr),
        "n"(4));
  }
}

template <size_t num_bits, PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load(T* smem_ptr, const T* gmem_ptr) {
  static_assert(num_bits == 128, "num_bits must be 128");
  load_128b<prefetch_mode>(smem_ptr, gmem_ptr);
}

template <size_t num_bits,
          PrefetchMode prefetch_mode,
          SharedMemFillMode fill_mode,
          typename T>
__device__ __forceinline__ void pred_load(T* smem_ptr,
                                          const T* gmem_ptr,
                                          bool predicate) {
  static_assert(num_bits == 128 || num_bits == 64 || num_bits == 32,
                "num_bits must be 128, 64 or 32.");
  if constexpr (num_bits == 128) {
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
  } else if constexpr (num_bits == 64) {
    pred_load_64b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
  } else if constexpr (num_bits == 32) {
    pred_load_32b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
  }
}

using b32_t = uint32_t;
using b64_t = uint2;
using b128_t = uint4;

template <typename T>
constexpr __host__ __device__ __forceinline__ uint32_t num_elems_per_128b() {
  return sizeof(b128_t) / sizeof(T);
}

struct smem_t {
  // The base pointer.
  b128_t* base;
  __device__ __forceinline__ smem_t() : base(nullptr) {}
  template <typename T>
  __device__ __forceinline__ smem_t(T* base) : base((b128_t*)base) {}


  template <uint32_t stride, uint32_t inv_stride = 0>
  static __device__ __forceinline__ uint32_t get_permuted_offset(uint32_t i,
                                                                 uint32_t j) {
    if constexpr (inv_stride <= 1) {
      return i * stride + (j ^ (i % 8));
    } else {
      return i / inv_stride * 8 + ((j + (i % inv_stride) * stride)) ^
             ((i / inv_stride) % 8);
    }
  }

  template <uint32_t step_size, uint32_t row_stride = 8>
  static __device__ __forceinline__ uint32_t
  advance_offset_by_column(uint32_t offset, uint32_t step_idx) {
    if constexpr (row_stride == 2) {
      static_assert(step_size == 2, "Unsupported step size");
      return offset + step_size;
    } else if constexpr (row_stride == 4) {
      static_assert(step_size == 2 || step_size == 4, "Unsupported step size");
      if constexpr (step_size == 2) {
        return (offset ^ 0x2) + (step_idx % 2 == 1) * 4;
      } else {
        return offset + step_size;
      }
    } else {
      static_assert(step_size == 2 || step_size == 4 || step_size % 8 == 0,
                    "Unsupported step size");
      if constexpr (step_size == 2) {
        return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) +
               (step_idx % 4 == 3) * 8;
      } else if constexpr (step_size == 4) {
        return (offset ^ 0x4) + (step_idx % 2 == 1) * 8;
      } else {
        // step_size % 8 == 0
        return offset + step_size;
      }
    }
  }

  template <uint32_t step_size, uint32_t row_stride>
  static __device__ __forceinline__ uint32_t
  advance_offset_by_row(uint32_t offset) {
    if constexpr (row_stride == 2) {
      static_assert(step_size == 16 || step_size % 32 == 0,
                    "Unsupported step size");
      if constexpr (step_size == 16) {
        return (offset ^ 0x4) + step_size * row_stride;
      } else {
        // step_size % 32 == 0
        return offset + step_size * row_stride;
      }
    } else if constexpr (row_stride == 4) {
      static_assert(step_size == 8 || step_size % 16 == 0,
                    "Unsupported step size");
      if constexpr (step_size == 8) {
        return (offset ^ 0x4) + step_size * row_stride;
      } else {
        // step_size % 16 == 0
        return offset + step_size * row_stride;
      }
    } else {
      static_assert(step_size == 4 || step_size % 8 == 0,
                    "Unsupported step size");
      if constexpr (step_size == 4) {
        return (offset ^ 0x4) + step_size * row_stride;
      } else {
        // step_size % 8 == 0
        return offset + step_size * row_stride;
      }
    }
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t offset,
                                                  uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    ldmatrix_m8n8x4_impl(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t offset,
                                                        uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    ldmatrix_m8n8x4_trans_impl(R, smem_ptr);
  }

  template <SharedMemFillMode fill_mode, typename T>
  __device__ __forceinline__ void load_128b_async(uint32_t offset,
                                                  const T* gptr,
                                                  bool predicate) {
    b128_t* smem_ptr = base + offset;
    pred_load_128b<PrefetchMode::kPrefetch, fill_mode>(
        smem_ptr, reinterpret_cast<const b128_t*>(gptr), predicate);
  }

  template <typename T>
  __device__ __forceinline__ void load_128b_async(uint32_t offset,
                                                  const T* gptr) {
    b128_t* smem_ptr = base + offset;
    load_128b<PrefetchMode::kPrefetch>(smem_ptr,
                                       reinterpret_cast<const b128_t*>(gptr));
  }

  template <typename T>
  __device__ __forceinline__ void store_128b(uint32_t offset, T* gptr) {
    *reinterpret_cast<b128_t*>(gptr) = *(base + offset);
  }
};
