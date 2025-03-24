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

#pragma once
#include "xpu/kernel/cluster.h"

static __device__ inline void vload2_lm_unordered_infer(
        const bfloat16* ptr,
        float32x16_t& veven,
        float32x16_t& vodd) {
    constexpr int mask = 0xaaaaaaaa;  // 0b10101010101010101010101010101010
    veven = reinterpret_cast<float32x16_t>(vload_lm_int16x32_mz(ptr, mask));
    vodd = reinterpret_cast<float32x16_t>(vshuffle2_float16x32(
            reinterpret_cast<float16x32_t>(vload_lm_int16x32_mz(ptr, (~mask)))));
}

static __device__ inline void vload2_lm_unordered_infer_mz(
        const bfloat16* ptr,
        float32x16_t& veven,
        float32x16_t& vodd,
        int mask = -1) {
    constexpr int mask_ = 0xaaaaaaaa;  // 0b10101010101010101010101010101010
    veven = reinterpret_cast<float32x16_t>(vload_lm_int16x32_mz(ptr, mask_ & mask));
    vodd = reinterpret_cast<float32x16_t>(vshuffle2_float16x32(
            reinterpret_cast<float16x32_t>(vload_lm_int16x32_mz(ptr, (~mask_) & mask))));
}

static __device__ inline void vstore2_lm_unordered_infer(
        bfloat16* ptr,
        float32x16_t veven,
        float32x16_t vodd) {
    veven = reinterpret_cast<float32x16_t>(
            svadd_uint32x16(0x8000, reinterpret_cast<uint32x16_t>(veven)));
    vodd = reinterpret_cast<float32x16_t>(
            svadd_uint32x16(0x8000, reinterpret_cast<uint32x16_t>(vodd)));
    constexpr int mask = 0xaaaaaaaa;  // 0b10101010101010101010101010101010
    vstore_lm_int16x32_mh(ptr, reinterpret_cast<int16x32_t>(veven), mask);
    vstore_lm_int16x32_mh(
            ptr,
            reinterpret_cast<int16x32_t>(
                    vshuffle2_float16x32(reinterpret_cast<float16x32_t>(vodd))),
            (~mask));
}

static __device__ inline void vstore2_lm_unordered_infer_mh(
        bfloat16* ptr,
        float32x16_t veven,
        float32x16_t vodd,
        int mask = -1) {
    veven = reinterpret_cast<float32x16_t>(
            svadd_uint32x16(0x8000, reinterpret_cast<uint32x16_t>(veven)));
    vodd = reinterpret_cast<float32x16_t>(
            svadd_uint32x16(0x8000, reinterpret_cast<uint32x16_t>(vodd)));
    constexpr int mask_ = 0xaaaaaaaa;  // 0b10101010101010101010101010101010
    vstore_lm_int16x32_mh(ptr, reinterpret_cast<int16x32_t>(veven), mask_ & mask);
    vstore_lm_int16x32_mh(
            ptr,
            reinterpret_cast<int16x32_t>(
                    vshuffle2_float16x32(reinterpret_cast<float16x32_t>(vodd))),
            (~mask_) & mask);
}

static __device__ inline void vstore2_lm_unordered_infer_rz(
        bfloat16* ptr,
        float32x16_t veven,
        float32x16_t vodd) {
    constexpr int mask = 0xaaaaaaaa;  // 0b10101010101010101010101010101010
    vstore_lm_int16x32_mh(ptr, reinterpret_cast<int16x32_t>(veven), mask);
    vstore_lm_int16x32_mh(
            ptr,
            reinterpret_cast<int16x32_t>(
                    vshuffle2_float16x32(reinterpret_cast<float16x32_t>(vodd))),
            (~mask));
}

static __device__ inline void vload2_sm_unordered_infer(
        const _shared_ptr_ bfloat16* ptr,
        float32x16_t& veven,
        float32x16_t& vodd) {
    constexpr int mask = 0xaaaaaaaa;  // 0b10101010101010101010101010101010
    veven = reinterpret_cast<float32x16_t>(vload_sm_int16x32_mz(ptr, mask));
    vodd = reinterpret_cast<float32x16_t>(vshuffle2_float16x32(
            reinterpret_cast<float16x32_t>(vload_sm_int16x32_mz(ptr, (~mask)))));
}

static __device__ inline float v_reduce(float32x16_t v) {
    auto v0 = vsrlp_float32x16(256, v);
    v = vvadd_float32x16(v0, v);
    v0 = vsrlp_float32x16(128, v);
    v = vvadd_float32x16(v0, v);
    v0 = vsrlp_float32x16(64, v);
    v = vvadd_float32x16(v0, v);
    v0 = vsrlp_float32x16(32, v);
    v = vvadd_float32x16(v0, v);
    float res;
    __asm__("vextract.f %0, %1{%2}" : "=&r"(res) : "v"(v), "M"(1));
    return res;
}

template <typename TID>
static inline __device__ void simd_copy_lm2sm(
        const TID* src_lm,
        __shared_ptr__ TID* dst_sm,
        const int len) {
    int32x16_t a0;
    int32x16_t a1;
    int i = 0;
    for (; i < len - 16; i += 32) {
        a0 = vload_lm_int32x16(src_lm + i);
        a1 = vload_lm_int32x16(src_lm + i + 16);
        vstore_sm_int32x16(dst_sm + i, a0);
        vstore_sm_int32x16(dst_sm + i + 16, a1);
    }
    if (i < len) {
        a0 = vload_lm_int32x16(src_lm + i);
        vstore_sm_int32x16(dst_sm + i, a0);
    }
    mfence_sm();
}

template <typename TID>
static inline __device__ void simd_copy_sm2lm(
        const __shared_ptr__ TID* src_sm,
        TID* dst_lm,
        const int len) {
    int32x16_t a0;
    int32x16_t a1;
    int i = 0;
    for (; i < len - 16; i += 32) {
        a0 = vload_sm_int32x16(src_sm + i);
        a1 = vload_sm_int32x16(src_sm + i + 16);
        vstore_lm_int32x16(dst_lm + i, a0);
        vstore_lm_int32x16(dst_lm + i + 16, a1);
    }
    if (i < len) {
        a0 = vload_sm_int32x16(src_sm + i);
        vstore_lm_int32x16(dst_lm + i, a0);
    }
    mfence_lm();
}

template <typename TID>
static inline __device__ void simd_copy_sm2sm(
        const __shared_ptr__ TID* src_sm,
        __shared_ptr__ TID* dst_sm,
        const int len) {
    int32x16_t a0;
    int32x16_t a1;
    int i = 0;
    for (; i < len - 16; i += 32) {
        a0 = vload_sm_int32x16(src_sm + i);
        a1 = vload_sm_int32x16(src_sm + i + 16);
        vstore_sm_int32x16(dst_sm + i, a0);
        vstore_sm_int32x16(dst_sm + i + 16, a1);
    }
    if (i < len) {
        a0 = vload_sm_int32x16(src_sm + i);
        vstore_sm_int32x16(dst_sm + i, a0);
    }
    mfence_sm();
}

template <typename TID>
static inline __device__ void simd_2SmVecAddToLm(
        const __shared_ptr__ TID* src0_sm,
        const __shared_ptr__ TID* src1_sm,
        TID* dst_lm,
        const int len) {
    int32x16_t a0;
    int32x16_t a1;
    int32x16_t b0;
    int32x16_t b1;
    int32x16_t c0;
    int32x16_t c1;
    int i = 0;
    for (; i < len - 16; i += 32) {
        a0 = vload_sm_int32x16(src0_sm + i);
        a1 = vload_sm_int32x16(src0_sm + i + 16);
        b0 = vload_sm_int32x16(src1_sm + i);
        b1 = vload_sm_int32x16(src1_sm + i + 16);
        c0 = vvadd_int32x16(a0, b0);
        c1 = vvadd_int32x16(a1, b1);
        vstore_lm_int32x16(dst_lm + i, c0);
        vstore_lm_int32x16(dst_lm + i + 16, c1);
    }
    if (i < len) {
        a0 = vload_sm_int32x16(src0_sm + i);
        b0 = vload_sm_int32x16(src1_sm + i);
        c0 = vvadd_int32x16(a0, b0);
        vstore_lm_int32x16(dst_lm + i, c0);
    }
    mfence_lm();
}

template <typename TID>
static inline __device__ void simd_2SmVecAddToSm(
        const __shared_ptr__ TID* src0_sm,
        const __shared_ptr__ TID* src1_sm,
        __shared_ptr__ TID* dst_sm,
        const int len) {
    int32x16_t a0;
    int32x16_t a1;
    int32x16_t b0;
    int32x16_t b1;
    int32x16_t c0;
    int32x16_t c1;
    int i = 0;
    for (; i < len - 16; i += 32) {
        a0 = vload_sm_int32x16(src0_sm + i);
        a1 = vload_sm_int32x16(src0_sm + i + 16);
        b0 = vload_sm_int32x16(src1_sm + i);
        b1 = vload_sm_int32x16(src1_sm + i + 16);
        c0 = vvadd_int32x16(a0, b0);
        c1 = vvadd_int32x16(a1, b1);
        vstore_sm_int32x16(dst_sm + i, c0);
        vstore_sm_int32x16(dst_sm + i + 16, c1);
    }
    if (i < len) {
        a0 = vload_sm_int32x16(src0_sm + i);
        b0 = vload_sm_int32x16(src1_sm + i);
        c0 = vvadd_int32x16(a0, b0);
        vstore_sm_int32x16(dst_sm + i, c0);
    }
    mfence_sm();
}