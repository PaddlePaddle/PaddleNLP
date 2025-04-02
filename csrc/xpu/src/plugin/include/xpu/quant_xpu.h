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
#include "xpu/infer_simd.h"
#include "xpu/kernel/xtdk_io.h"

KERNEL_NAMESPACE_BEGIN

#define CLIP_FP32(a_h, a_l)                \
    a_h = svmax_float32x16(int8_min, a_h); \
    a_h = svmin_float32x16(int8_max, a_h); \
    a_l = svmax_float32x16(int8_min, a_l); \
    a_l = svmin_float32x16(int8_max, a_l);

#define CLIP_DOUBLE_FP32(a_hh, a_hl, a_lh, a_ll) \
    /* x = clip(x) */                            \
    a_hh = svmax_float32x16(int8_min, a_hh);     \
    a_hh = svmin_float32x16(int8_max, a_hh);     \
    a_hl = svmax_float32x16(int8_min, a_hl);     \
    a_hl = svmin_float32x16(int8_max, a_hl);     \
    a_lh = svmax_float32x16(int8_min, a_lh);     \
    a_lh = svmin_float32x16(int8_max, a_lh);     \
    a_ll = svmax_float32x16(int8_min, a_ll);     \
    a_ll = svmin_float32x16(int8_max, a_ll);

// Calculate QX = clip(X*(qmax/fmax))
#define CALC_QUANT_VEC_FP32(scale, a_hh, a_hl, a_lh, a_ll) \
    /* x = x * scale */                                    \
    a_hh = svmul_float32x16(scale, a_hh);                  \
    a_hl = svmul_float32x16(scale, a_hl);                  \
    a_lh = svmul_float32x16(scale, a_lh);                  \
    a_ll = svmul_float32x16(scale, a_ll);                  \
    CLIP_DOUBLE_FP32(a_hh, a_hl, a_lh, a_ll)

#define CALC_QUANT_VEC_VEC_FP32(a_hh, a_hl, a_lh, a_ll, s_ll, s_lh, s_hl, s_hh) \
    /* x = x * scale */                                                         \
    a_hh = vvmul_float32x16(a_hh, s_hh);                                        \
    a_hl = vvmul_float32x16(a_hl, s_hl);                                        \
    a_lh = vvmul_float32x16(a_lh, s_lh);                                        \
    a_ll = vvmul_float32x16(a_ll, s_ll);                                        \
    CLIP_DOUBLE_FP32(a_hh, a_hl, a_lh, a_ll)

template <typename T, bool MFENCE = true>
static __device__ void primitive_reduce_max_lm(const T* x, float* y, int len) {
    unsigned int mask = -1;
    float32x16_t vx0;
    float32x16_t vx1;
    float thresh = 0.0f;
    float32x16_t vec_max = vset_zero();
    int roundsize32 = rounddown32(len);
    int remain = len - roundsize32;
    if (remain) {
        mask = ~(-1 << remain);
        vload2_lm_mz(x + roundsize32, vx0, vx1, mask);
        vx0 = vabs_float32x16(vx0);
        vx1 = vabs_float32x16(vx1);
        vec_max = vvmax_float32x16(vec_max, vx0);
        vec_max = vvmax_float32x16(vec_max, vx1);
    }
    for (int i = 0; i < roundsize32; i += 32) {
        vload2_lm(x + i, vx0, vx1);
        vx0 = vabs_float32x16(vx0);
        vx1 = vabs_float32x16(vx1);
        vec_max = vvmax_float32x16(vec_max, vx0);
        vec_max = vvmax_float32x16(vec_max, vx1);
    }
    thresh = vrmax_float32x16(vec_max);
    *y = thresh;
#if MFENCE 
        mfence_lm();
#endif    
}

template <>
__device__ void primitive_reduce_max_lm<bfloat16, true>(const bfloat16* x, float* y, int len) {
    float32x16_t vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7;
    float32x16_t vec_max = vset_zero();
    int roundsize128 = len / 128 * 128;
    for (int i = 0; i < roundsize128; i += 128) {
        vload2_lm_unordered_infer(x + i, vx0, vx1);
        vload2_lm_unordered_infer(x + i + 32, vx2, vx3);
        vload2_lm_unordered_infer(x + i + 64, vx4, vx5);
        vload2_lm_unordered_infer(x + i + 96, vx6, vx7);

        vx0 = vabs_float32x16(vx0);
        vx1 = vabs_float32x16(vx1);
        vx2 = vabs_float32x16(vx2);
        vx3 = vabs_float32x16(vx3);
        vx4 = vabs_float32x16(vx4);
        vx5 = vabs_float32x16(vx5);
        vx6 = vabs_float32x16(vx6);
        vx7 = vabs_float32x16(vx7);

        vx0 = vvmax_float32x16(vx0, vx1);
        vx2 = vvmax_float32x16(vx2, vx3);
        vx4 = vvmax_float32x16(vx4, vx5);
        vx6 = vvmax_float32x16(vx6, vx7);

        vx0 = vvmax_float32x16(vx0, vx2);
        vx4 = vvmax_float32x16(vx4, vx6);
        vec_max = vvmax_float32x16(vec_max, vx0);
        vec_max = vvmax_float32x16(vec_max, vx4);
    }

    unsigned int mask = -1;
    int roundsize32 = rounddown32(len - roundsize128) + roundsize128;
    int remain = len - roundsize32;
    if (remain) {
        mask = ~(-1 << remain);
        vload2_lm_mz(x + roundsize32, vx0, vx1, mask);
        vx0 = vabs_float32x16(vx0);
        vx1 = vabs_float32x16(vx1);
        vec_max = vvmax_float32x16(vec_max, vx0);
        vec_max = vvmax_float32x16(vec_max, vx1);
    }
    for (int i = roundsize128; i < len; i += 32) {
        vload2_lm(x + i, vx0, vx1);
        vx0 = vabs_float32x16(vx0);
        vx1 = vabs_float32x16(vx1);
        vec_max = vvmax_float32x16(vec_max, vx0);
        vec_max = vvmax_float32x16(vec_max, vx1);
    }
    float thresh = vrmax_float32x16(vec_max);
    *y = thresh;
    mfence_lm();
}

template <>
__device__ void primitive_reduce_max_lm<bfloat16, false>(const bfloat16* x, float* y, int len) {
    float32x16_t vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7;
    float32x16_t vec_max = vset_zero();
    int roundsize128 = len / 128 * 128;
    for (int i = 0; i < roundsize128; i += 128) {
        vload2_lm_unordered_infer(x + i, vx0, vx1);
        vload2_lm_unordered_infer(x + i + 32, vx2, vx3);
        vload2_lm_unordered_infer(x + i + 64, vx4, vx5);
        vload2_lm_unordered_infer(x + i + 96, vx6, vx7);

        vx0 = vabs_float32x16(vx0);
        vx1 = vabs_float32x16(vx1);
        vx2 = vabs_float32x16(vx2);
        vx3 = vabs_float32x16(vx3);
        vx4 = vabs_float32x16(vx4);
        vx5 = vabs_float32x16(vx5);
        vx6 = vabs_float32x16(vx6);
        vx7 = vabs_float32x16(vx7);

        vx0 = vvmax_float32x16(vx0, vx1);
        vx2 = vvmax_float32x16(vx2, vx3);
        vx4 = vvmax_float32x16(vx4, vx5);
        vx6 = vvmax_float32x16(vx6, vx7);

        vx0 = vvmax_float32x16(vx0, vx2);
        vx4 = vvmax_float32x16(vx4, vx6);
        vec_max = vvmax_float32x16(vec_max, vx0);
        vec_max = vvmax_float32x16(vec_max, vx4);
    }

    unsigned int mask = -1;
    int roundsize32 = rounddown32(len - roundsize128) + roundsize128;
    int remain = len - roundsize32;
    if (remain) {
        mask = ~(-1 << remain);
        vload2_lm_mz(x + roundsize32, vx0, vx1, mask);
        vx0 = vabs_float32x16(vx0);
        vx1 = vabs_float32x16(vx1);
        vec_max = vvmax_float32x16(vec_max, vx0);
        vec_max = vvmax_float32x16(vec_max, vx1);
    }
    for (int i = roundsize128; i < len; i += 32) {
        vload2_lm(x + i, vx0, vx1);
        vx0 = vabs_float32x16(vx0);
        vx1 = vabs_float32x16(vx1);
        vec_max = vvmax_float32x16(vec_max, vx0);
        vec_max = vvmax_float32x16(vec_max, vx1);
    }
    float thresh = vrmax_float32x16(vec_max);
    *y = thresh;
}

template <>
__device__ void primitive_reduce_max_lm<float16, true>(const float16* x, float* y, int len) {
    unsigned int mask_0 = -1;
    unsigned int mask_1 = -1;
    unsigned long long mask = -1;
    float16x32_t vx0;
    float16x32_t vx1;
    float16 thresh = float16(0.0f);
    float16x32_t vec_max = vset_zero_fp16();
    int roundsize64 = rounddown64(len);
    int remain = len - roundsize64;
    if (remain) {
        mask_0 = (remain >= 32) ? -1 : ~(-1 << remain);
        mask_1 = (remain > 32) ? ~(-1 << (remain - 32)) : 0;
        vx0 = vload_lm_float16x32_mz(x + roundsize64, mask_0);
        vx1 = vload_lm_float16x32_mz(x + roundsize64 + 32, mask_1);
        vx0 = vabs_float16x32(vx0);
        vx1 = vabs_float16x32(vx1);
        vec_max = vvmax_float16x32(vec_max, vx0);
        vec_max = vvmax_float16x32(vec_max, vx1);
    }
    for (int i = 0; i < roundsize64; i += 64) {
        vx0 = vload_lm_float16x32(x + i);
        vx1 = vload_lm_float16x32(x + i + 32);
        vx0 = vabs_float16x32(vx0);
        vx1 = vabs_float16x32(vx1);
        vec_max = vvmax_float16x32(vec_max, vx0);
        vec_max = vvmax_float16x32(vec_max, vx1);
    }
    thresh = vrmax_float16x32(vec_max);
    *y = float162float(thresh);
    mfence_lm();
}

template <>
__device__ void primitive_reduce_max_lm<float16, false>(const float16* x, float* y, int len) {
    unsigned int mask_0 = -1;
    unsigned int mask_1 = -1;
    unsigned long long mask = -1;
    float16x32_t vx0;
    float16x32_t vx1;
    float16 thresh = float16(0.0f);
    float16x32_t vec_max = vset_zero_fp16();
    int roundsize64 = rounddown64(len);
    int remain = len - roundsize64;
    if (remain) {
        mask_0 = (remain >= 32) ? -1 : ~(-1 << remain);
        mask_1 = (remain > 32) ? ~(-1 << (remain - 32)) : 0;
        vx0 = vload_lm_float16x32_mz(x + roundsize64, mask_0);
        vx1 = vload_lm_float16x32_mz(x + roundsize64 + 32, mask_1);
        vx0 = vabs_float16x32(vx0);
        vx1 = vabs_float16x32(vx1);
        vec_max = vvmax_float16x32(vec_max, vx0);
        vec_max = vvmax_float16x32(vec_max, vx1);
    }
    for (int i = 0; i < roundsize64; i += 64) {
        vx0 = vload_lm_float16x32(x + i);
        vx1 = vload_lm_float16x32(x + i + 32);
        vx0 = vabs_float16x32(vx0);
        vx1 = vabs_float16x32(vx1);
        vec_max = vvmax_float16x32(vec_max, vx0);
        vec_max = vvmax_float16x32(vec_max, vx1);
    }
    thresh = vrmax_float16x32(vec_max);
    *y = float162float(thresh);
}

static inline __device__ void cast_to_int8_lm_rn(
        float32x16_t a_ll,
        float32x16_t a_lh,
        float32x16_t a_hl,
        float32x16_t a_hh,
        int8_t* y) {
    __asm__ __volatile__(
            "vfloat2fix8_ll.rn vr0, %0\t\n"
            "vfloat2fix8_lh.rn vr0, %1\t\n"
            "vfloat2fix8_hl.rn vr0, %2\t\n"
            "vfloat2fix8_hh.rn vr0, %3\t\n"
            "vstore.mz vr0{mr1}, 0(%4)" ::"v"(a_ll),
            "v"(a_lh),
            "v"(a_hl),
            "v"(a_hh),
            "r"(y)
            : "vr0");
}

static inline __device__ void cast_to_int8_lm_rn_double(
        float32x16_t a_ll,
        float32x16_t a_lh,
        float32x16_t a_hl,
        float32x16_t a_hh,
        float32x16_t b_ll,
        float32x16_t b_lh,
        float32x16_t b_hl,
        float32x16_t b_hh,
        int8_t* y) {
    __asm__ __volatile__(
            "vfloat2fix8_ll.rn vr0, %0\t\n"
            "vfloat2fix8_ll.rn vr1, %1\t\n"
            "vfloat2fix8_lh.rn vr0, %2\t\n"
            "vfloat2fix8_lh.rn vr1, %3\t\n"
            "vfloat2fix8_hl.rn vr0, %4\t\n"
            "vfloat2fix8_hl.rn vr1, %5\t\n"
            "vfloat2fix8_hh.rn vr0, %6\t\n"
            "vfloat2fix8_hh.rn vr1, %7\t\n"
            "vstore.mz vr0{mr1}, 0(%8)\t\n"
            "vstore.mz vr1{mr1}, 0(%9)" ::"v"(a_ll),
            "v"(b_ll),
            "v"(a_lh),
            "v"(b_lh),
            "v"(a_hl),
            "v"(b_hl),
            "v"(a_hh),
            "v"(b_hh),
            "r"(y),
            "r"(y + 64)
            : "vr0", "vr1");
}

template <typename T, bool use_fence = true>
static __device__ void primitive_mul_and_round_lm(const T* x, int8_t* y, float scale, int len) {
    int rounddown_size_128 = rounddown128(len);
    int rounddown_size_64 = rounddown64(len);

    int mask_l = -1;
    int mask_h = -1;
    int remain_size = len - rounddown_size_64;
    int remain_size_64 = rounddown_size_64 - rounddown_size_128;
    float32x16_t a_ll, a_lh, a_hl, a_hh;
    float32x16_t b_ll, b_lh, b_hl, b_hh;

    for (int i = 0; i < rounddown_size_128; i += 128) {
        vload2_lm(x + i, a_ll, a_lh);
        vload2_lm(x + i + 32, a_hl, a_hh);
        vload2_lm(x + i + 64, b_ll, b_lh);
        vload2_lm(x + i + 96, b_hl, b_hh);
        a_hh = svmul_float32x16(scale, a_hh);
        a_hl = svmul_float32x16(scale, a_hl);
        a_lh = svmul_float32x16(scale, a_lh);
        a_ll = svmul_float32x16(scale, a_ll);

        b_hh = svmul_float32x16(scale, b_hh);
        b_hl = svmul_float32x16(scale, b_hl);
        b_lh = svmul_float32x16(scale, b_lh);
        b_ll = svmul_float32x16(scale, b_ll);
        // store to int8 local buffer.
        cast_to_int8_lm_rn_double(a_ll, a_lh, a_hl, a_hh, b_ll, b_lh, b_hl, b_hh, y + i);
    }

    if (remain_size_64) {
        vload2_lm_mz(x + rounddown_size_128, a_ll, a_lh, mask_l);
        vload2_lm_mz(x + rounddown_size_128 + 32, a_hl, a_hh, mask_h);

        a_hh = svmul_float32x16(scale, a_hh);
        a_hl = svmul_float32x16(scale, a_hl);
        a_lh = svmul_float32x16(scale, a_lh);
        a_ll = svmul_float32x16(scale, a_ll);
        // store to local buffer.
        cast_to_int8_lm_rn(a_ll, a_lh, a_hl, a_hh, y + rounddown_size_128);
    }

    if (remain_size) {
        //*
        mask_l = (remain_size >= 32) ? -1 : ~(-1 << remain_size);
        mask_h = (remain_size > 32) ? ~(-1 << (remain_size - 32)) : 0;
        vload2_lm_mz(x + rounddown_size_64, a_ll, a_lh, mask_l);
        vload2_lm_mz(x + rounddown_size_64 + 32, a_hl, a_hh, mask_h);
        a_hh = svmul_float32x16(scale, a_hh);
        a_hl = svmul_float32x16(scale, a_hl);
        a_lh = svmul_float32x16(scale, a_lh);
        a_ll = svmul_float32x16(scale, a_ll);
        // store to local buffer.
        cast_to_int8_lm_rn(a_ll, a_lh, a_hl, a_hh, y + rounddown_size_64);
    }
#if use_fence 
        mfence_lm();
#endif    
}

template <typename T>
static __device__ void quant_int8_and_round_lm(const T* x, int8_t* y, float scale, int len) {
    int rounddown_size = rounddown64(len);
    int mask_l = -1;
    int mask_h = -1;
    int remain_size = len - rounddown_size;
    float32x16_t a_ll, a_lh, a_hl, a_hh;
    const float int8_max = 127.0f;
    scale = int8_max / scale;
    const float int8_min = -128.0f;
    if (remain_size) {
        //*
        mask_l = (remain_size >= 32) ? -1 : ~(-1 << remain_size);
        mask_h = (remain_size > 32) ? ~(-1 << (remain_size - 32)) : 0;
        // load fp16 data and convert to fp32
        vload2_lm_mz(x + rounddown_size, a_ll, a_lh, mask_l);
        vload2_lm_mz(x + rounddown_size + 32, a_hl, a_hh, mask_h);
        // QX = clip(X*(qmax/fmax))
        CALC_QUANT_VEC_FP32(scale, a_hh, a_hl, a_lh, a_ll);
        // store to local buffer.
        cast_to_int8_lm_rn(a_ll, a_lh, a_hl, a_hh, y + rounddown_size);
        //*/
    }
    if (rounddown_size >= 64) {
        for (int i = 0; i < rounddown_size; i += 64) {
            // load fp16 data and convert to fp32
            // TODO (syj): To see if we can optimize vload2_lm to vload2_lm_unordered in bf16
            // precision later on.
            vload2_lm(x + i, a_ll, a_lh);
            vload2_lm(x + i + 32, a_hl, a_hh);
            // QX = clip(X*(qmax/fmax))
            CALC_QUANT_VEC_FP32(scale, a_hh, a_hl, a_lh, a_ll);
            // store to int8 local buffer.
            cast_to_int8_lm_rn(a_ll, a_lh, a_hl, a_hh, y + i);
        }
    }
    mfence_lm();
}

template <typename T>
static __device__ void quant_int8_and_round_lm_sm(
        const T* x,
        int8_t* y,
        __shared_ptr__ const float* scale,
        int len) {
    int rounddown_size = rounddown64(len);
    int mask_l = -1;
    int mask_h = -1;
    int remain_size = len - rounddown_size;
    float32x16_t a_ll, a_lh, a_hl, a_hh;
    float32x16_t s_ll, s_lh, s_hl, s_hh;
    const float int8_max = 127.0f;
    const float int8_min = -128.0f;
    if (remain_size) {
        //*
        mask_l = (remain_size >= 32) ? -1 : ~(-1 << remain_size);
        mask_h = (remain_size > 32) ? ~(-1 << (remain_size - 32)) : 0;
        // load fp16 data and convert to fp32
        vload2_lm_mz(x + rounddown_size, a_ll, a_lh, mask_l);
        vload2_lm_mz(x + rounddown_size + 32, a_hl, a_hh, mask_h);
        vload2_sm_mz(scale + rounddown_size, s_ll, s_lh, mask_l);
        vload2_sm_mz(scale + rounddown_size + 32, s_hl, s_hh, mask_h);
        // QX = clip(X*(qmax/fmax))
        CALC_QUANT_VEC_VEC_FP32(a_hh, a_hl, a_lh, a_ll, s_ll, s_lh, s_hl, s_hh);
        // store to local buffer.
        cast_to_int8_lm_rn(a_ll, a_lh, a_hl, a_hh, y + rounddown_size);
        //*/
    }
    if (rounddown_size >= 64) {
        for (int i = 0; i < rounddown_size; i += 64) {
            // load fp16 data and convert to fp32
            vload2_lm(x + i, a_ll, a_lh);
            vload2_lm(x + i + 32, a_hl, a_hh);
            vload2_sm(scale + i, s_ll, s_lh);
            vload2_sm(scale + i + 32, s_hl, s_hh);
            // QX = clip(X*(qmax/fmax))
            CALC_QUANT_VEC_VEC_FP32(a_hh, a_hl, a_lh, a_ll, s_ll, s_lh, s_hl, s_hh);
            // store to int8 local buffer.
            cast_to_int8_lm_rn(a_ll, a_lh, a_hl, a_hh, y + i);
        }
    }
    mfence();
}

KERNEL_NAMESPACE_END