/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
    \file
    \brief Defines a class for using IEEE half-precision floating-point types in host or
      device code.
*/
#pragma once

#include <cuda_fp16.h>

#include "cutlass/cutlass.h"

#if defined(__CUDACC_RTC__)

#include "cutlass/floating_point_nvrtc.h"

#else
//
// Standard Library headers belong here to avoid conflicts with NVRTC.
//
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

#if (__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#ifndef CUDA_PTX_FP8_CVT_ENABLED
#define CUDA_PTX_FP8_CVT_ENABLED 1
#endif
#endif
#endif

#ifdef __GNUC__
// Ignore checks on reinterpret-casts that are being used for bitcasts.
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//  FP8 Has 2 encodings possible : E4M3 and E5M2
//
//  E4M3 : 7  |  6 5 4 3  |  2 1 0
//  E5M2 : 7  |  6 5 4 3 2  |  1 0
//
///////////////////////////////////////////////////////////////////////////////////////////////////

enum class FloatEncoding {
    E4M3,
    E5M2
};

template<FloatEncoding T>
struct alignas(1) float8_base {

    static constexpr bool IS_E4M3 = (T == FloatEncoding::E4M3);
    static constexpr bool IS_E5M2 = (T == FloatEncoding::E5M2);

    // Number of Bits representing mantissa and exponents
    static constexpr int FP32_NUM_BITS = 32;
    static constexpr int FP32_NUM_EXPONENT_BITS = 8;
    static constexpr int FP32_NUM_MANTISSA_BITS = 23;
    static constexpr uint32_t FP32_NAN = 0x7fffffff;
    static constexpr uint32_t FP32_INFINITY_MASK = 0x7f800000;
    static constexpr int FP32_MAX_EXPONENT  =  127;
    static constexpr int FP32_MIN_EXPONENT  = -126;
    static constexpr int FP32_EXPONENT_BIAS =  127;

    static constexpr int FP16_NUM_BITS = 16;
    static constexpr int FP16_NUM_EXPONENT_BITS = 5;
    static constexpr int FP16_NUM_MANTISSA_BITS = 10;
    static constexpr uint16_t FP16_NAN = 0x7fff;
    static constexpr uint16_t FP16_INFINITY_MASK = 0x7c00;
    static constexpr int FP16_MAX_EXPONENT  = 15;
    static constexpr int FP16_MIN_EXPONENT  = -14;
    static constexpr int FP16_EXPONENT_BIAS = 15;

    static constexpr int FP8_NUM_BITS = 8;
    static constexpr int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;
    static constexpr int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;
    static constexpr uint8_t  FP8_NAN = 0x7f; // Also F8_INF 
    static constexpr uint8_t  FP8_INFINITY_MASK = IS_E4M3 ? 0x78 : 0x7c;
    static constexpr int FP8_MAX_EXPONENT  = IS_E4M3 ?  7 :  15;
    static constexpr int FP8_MIN_EXPONENT  = IS_E4M3 ? -6 : -14;
    static constexpr int FP8_EXPONENT_BIAS = IS_E4M3 ?  7 :  15;

    static constexpr uint8_t  FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1;
    static constexpr uint8_t  FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1;

    static constexpr uint8_t FP8_MAX_FLT = (IS_E4M3 ? 0x7e : 0x7b);

    // 256 in float
    static constexpr uint32_t FP8_SAT_VAL_FP32 = 0x43800000;

    //
    // Data members
    //

    /// Data container
    uint8_t storage;

    /// Ctors.
    CUTLASS_HOST_DEVICE
    float8_base() : storage(0) { }

    /// Is finite implementation
    CUTLASS_HOST_DEVICE
    static bool isfinite(float flt) {
        uint32_t s;

        #if defined(__CUDA_ARCH__)
        s = reinterpret_cast<uint32_t const &>(flt);
        #else
        std::memcpy(&s, &flt, sizeof(s));
        #endif

        return (s & 0x7f800000) < 0x7f800000;
    }

    /// Is NaN implementation
    CUTLASS_HOST_DEVICE
    static bool isnan(float flt) {
        uint32_t s;

        #if defined(__CUDA_ARCH__)
        s = reinterpret_cast<uint32_t const &>(flt);
        #else
        std::memcpy(&s, &flt, sizeof(s));
        #endif

        return (s & 0x7fffffff) > 0x7f800000;
    }

    /// Is infinite implementation
    CUTLASS_HOST_DEVICE
    static bool isinf(float flt) {
        uint32_t s;

        #if defined(__CUDA_ARCH__)
        s = reinterpret_cast<uint32_t const &>(flt);
        #else
        std::memcpy(&s, &flt, sizeof(s));
        #endif

        // Sign = 0 for +inf, 1 for -inf
        // Exponent = all ones
        // Mantissa = all zeros
        return (s == 0x7f800000) || (s == 0xff800000);
    }

    /// FP32 -> FP8 conversion - rounds to nearest even
    CUTLASS_HOST_DEVICE
    static uint8_t convert_float_to_fp8(float const& flt) {

        // software implementation rounds toward nearest even
        uint32_t s;

        #if defined(__CUDA_ARCH__)
        s = reinterpret_cast<uint32_t const &>(flt);
        #else
        std::memcpy(&s, &flt, sizeof(s));
        #endif

        // Extract the bits in the FP32 type
        uint8_t sign = uint8_t((s >> 24 & 0x80));
        int8_t exp = uint8_t(((s >> FP32_NUM_MANTISSA_BITS) & 0xff) - FP32_EXPONENT_BIAS);
        int mantissa = s & 0x7fffff;
        uint8_t u = 0;

        uint8_t const kF8_NaN = 0x7f;

        // NaN => NaN
        if (isnan(flt)) {
            return kF8_NaN;
        }

        // Inf => MAX_FLT (satfinite)
        if (isinf(flt)) {
            return sign | FP8_MAX_FLT;
        }

        // Special handling
        if ( exp == -128 ) {
            // int8 range is from -128 to 127
            // So 255(inf) - 127(bias) = 128 - will show up as -128

            // satfinite
            return (sign | FP8_MAX_FLT);
        }

        int sticky_bit = 0;

        bool skip_sign = false;
        bool may_be_nan = false;

        if ( (exp >= FP8_MIN_EXPONENT) && (exp <= FP8_MAX_EXPONENT) ) {
            // normal fp32 to normal fp8
            exp = uint8_t(exp + uint8_t(FP8_EXPONENT_BIAS));
            u = uint8_t(((exp & FP8_EXPONENT_MASK) << FP8_NUM_MANTISSA_BITS));
            u = uint8_t(u | (mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
        } else if(exp < FP8_MIN_EXPONENT) {
            // normal single-precision to subnormal float8-precision representation
            int rshift = (FP8_MIN_EXPONENT - exp);
            if (rshift < FP32_NUM_BITS) {
                mantissa |= (1 << FP32_NUM_MANTISSA_BITS);

                sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

                mantissa = (mantissa >> rshift);
                u = (uint8_t(mantissa >> (FP32_NUM_MANTISSA_BITS- FP8_NUM_MANTISSA_BITS)) & FP8_MANTISSA_MASK);
            } else {
                mantissa = 0;
                u = 0;
            }
        // Exponent > FP8_MAX_EXPONENT - this is a special case done to match HW
        // 0x4380_0000 to 0x43e0_0000 - maps from 256 to 448, and does not saturate / inf.
        } else {
            if( exp == (FP8_MAX_EXPONENT + 1) ) {
                uint8_t mantissa_tmp = uint8_t(mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
                if( mantissa_tmp < FP8_MANTISSA_MASK) {
                    exp = uint8_t(exp + uint8_t(FP8_EXPONENT_BIAS));
                    u = uint8_t(exp << FP8_NUM_MANTISSA_BITS) | mantissa_tmp;
                    may_be_nan =  (mantissa_tmp == (FP8_MANTISSA_MASK-1));
                } else {
                    // satfinite
                    return (sign | FP8_MAX_FLT);
                }
            } else{
                // satfinite
                return (sign | FP8_MAX_FLT);
            }
        }

        // round to nearest even
        int NUM_BITS_SHIFT = FP32_NUM_MANTISSA_BITS - (FP8_NUM_MANTISSA_BITS + 1);
        int round_bit = ((mantissa >> NUM_BITS_SHIFT) & 1);
        sticky_bit |= ((mantissa & ((1 << NUM_BITS_SHIFT) - 1)) != 0);

        if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
            u = uint8_t(u + 1);
            if( may_be_nan ) {
                skip_sign = true;
            }
        }

        if (u > FP8_MAX_FLT) {
            // satfinite
            u = (sign | FP8_MAX_FLT);
        }

        if( ! skip_sign ) {
            u |= sign;
        }

        return u;
    }


    /// Converts a fp8 value stored as a uint8_t to a float
    CUTLASS_HOST_DEVICE
    static float convert_fp8_to_float(uint8_t const& x) {

        uint32_t constexpr kF32_NaN = 0x7fffffff;

        uint8_t const &f8 = x;
        int sign = (f8 >> (FP8_NUM_BITS - 1)) & 1;
        int exp = (f8 >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK;
        int mantissa = f8 & FP8_MANTISSA_MASK;
        unsigned f = (sign << (FP32_NUM_BITS-1));

        if (IS_E4M3 && exp == 15 && mantissa == 0x7) {
            f = kF32_NaN;
        }
        else if (exp > 0 && (IS_E4M3 || exp < (FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1))) {
            // normal
            exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS);
            f = f |
                (exp << FP32_NUM_MANTISSA_BITS) |
                (mantissa << (FP32_NUM_MANTISSA_BITS-FP8_NUM_MANTISSA_BITS));
        } else if (exp == 0) {
            if (mantissa) {
                // subnormal
                exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS) + 1;
                while ((mantissa & (1 << FP8_NUM_MANTISSA_BITS)) == 0) {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= FP8_MANTISSA_MASK;
                f = f |
                    (exp << FP32_NUM_MANTISSA_BITS) |
                    (mantissa << (FP32_NUM_MANTISSA_BITS-FP8_NUM_MANTISSA_BITS));
            } else {
                // sign-preserving zero
            }
        } else {
            if(mantissa == 0){
                // Sign-preserving infinity
                f = (f | 0x7f800000);
            } else {
                // Canonical NaN
                f = kF32_NaN;
            }
        }

        #if defined(__CUDA_ARCH__)
        return reinterpret_cast<float const&>(f);
        #else
        float flt;
        std::memcpy(&flt, &f, sizeof(flt));
        return flt;
        #endif
    }
};


// Forward declaration of float_e5m2_t to define float_e4m3_t <=> float_e5m2_t
// conversions in class float_e4m3_t
struct float_e5m2_t;


///////////////////////////////////////////////////////////////
///
/// floating-point 8 type : E4M3
///
///////////////////////////////////////////////////////////////
struct alignas(1) float_e4m3_t : float8_base<FloatEncoding::E4M3> {

    using Base = float8_base<FloatEncoding::E4M3>;

    static constexpr int MAX_EXPONENT = Base::FP8_MAX_EXPONENT;

    //
    // Static conversion operators
    //

    /// Constructs from an uint8_t
    CUTLASS_HOST_DEVICE
    static float_e4m3_t bitcast(uint8_t x) {
        float_e4m3_t f;
        f.storage = x;
        return f;
    }

    /// FP32 -> FP8 conversion - rounds to nearest even
    CUTLASS_HOST_DEVICE
    static float_e4m3_t from_float(float const& flt) {
    #if defined(CUDA_PTX_FP8_CVT_ENABLED)
        uint16_t tmp;
        float y = float();
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(tmp) : "f"(y), "f"(flt));

        return *reinterpret_cast<float_e4m3_t *>(&tmp);
    #else
        return bitcast(Base::convert_float_to_fp8(flt));
    #endif
    }

    /// FP16 -> E5M2 conversion - rounds to nearest even
    CUTLASS_HOST_DEVICE
    static float_e4m3_t from_half(half const& flt) {
    #if defined(CUDA_PTX_FP8_CVT_ENABLED)
        uint16_t tmp = 0;
        uint32_t bits = reinterpret_cast<uint16_t const &>(flt);
        asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(tmp) : "r"(bits));

        return *reinterpret_cast<float_e4m3_t *>(&tmp);
    #else
        return bitcast(Base::convert_float_to_fp8(__half2float(flt)));
    #endif
    }

    // E4M3 -> half
    CUTLASS_HOST_DEVICE
    static half to_half(float_e4m3_t const& x) {
    #if defined(CUDA_PTX_FP8_CVT_ENABLED)
        uint16_t bits = x.storage;
        uint32_t packed;
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(packed) : "h"(bits));

        return reinterpret_cast<half2 const &>(packed).x;
    #else
        return __float2half(Base::convert_fp8_to_float(x.storage));
    #endif
    }

    // E4M3 -> Float
    CUTLASS_HOST_DEVICE
    static float to_float(float_e4m3_t const& x) {
    #if defined(CUDA_PTX_FP8_CVT_ENABLED)
        uint16_t bits = x.storage;
        uint32_t packed;
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(packed) : "h"(bits));

        return __half2float(reinterpret_cast<half2 const &>(packed).x);
    #else
        return Base::convert_fp8_to_float(x.storage);
    #endif
    }

    //
    // Methods
    //

    /// Default constructor
    CUTLASS_HOST_DEVICE
    float_e4m3_t() : Base() { }

    /// Reinterpret cast from CUDA's FP8 type
    CUTLASS_HOST_DEVICE
    float_e4m3_t(float_e4m3_t const& x) {
    #if defined(__CUDA_ARCH__)
        storage = reinterpret_cast<uint8_t const &>(x);
    #else
        uint8_t raw = x.storage;
        std::memcpy(&storage, &raw, sizeof(storage));
    #endif
    }

    /// Floating point conversion
    CUTLASS_HOST_DEVICE
    explicit float_e4m3_t(float x) {
        storage = from_float(x).storage;
    }

    CUTLASS_HOST_DEVICE
    explicit float_e4m3_t(half x) {
        storage = from_half(x).storage;
    }

    /// Floating point conversion
    CUTLASS_HOST_DEVICE
    explicit float_e4m3_t(double x): float_e4m3_t(float(x)) {
    }

    /// Integer conversion
    CUTLASS_HOST_DEVICE
    explicit float_e4m3_t(int x): float_e4m3_t(float(x)) {
    }

    /// E5M2 conversion. Defined after float_e5m2_t is defined.
    CUTLASS_HOST_DEVICE
    explicit float_e4m3_t(float_e5m2_t x);

    /// Assignment
    CUTLASS_HOST_DEVICE
    float_e4m3_t & operator=(float_e4m3_t const &x) {
    #if defined(__CUDA_ARCH__)
        storage = reinterpret_cast<uint8_t const &>(x);
    #else
        uint8_t raw = x.storage;
        std::memcpy(&storage, &raw, sizeof(storage));
    #endif
        return *this;
    }

    /// Converts to float
    CUTLASS_HOST_DEVICE
    operator float() const {
        return to_float(*this);
    }

    /// Converts to half
    CUTLASS_HOST_DEVICE
    operator half() const {
        return to_half(*this);
    }

    /// Converts to float
    CUTLASS_HOST_DEVICE
    explicit operator double() const {
        return double(to_float(*this));
    }

    /// Converts to int
    CUTLASS_HOST_DEVICE
    explicit operator int() const {
    #if defined(__CUDA_ARCH__)
        return __half2int_rn(to_half(*this));
    #else
        return int(to_float(*this));
    #endif
    }

    /// Casts to bool
    CUTLASS_HOST_DEVICE
    explicit operator bool() const {
    #if defined(__CUDA_ARCH__)
        return bool(__half2int_rn(to_half(*this)));
    #else
        return bool(int(to_float(*this)));
    #endif
    }

    /// Accesses raw internal state
    CUTLASS_HOST_DEVICE
    uint8_t& raw() {
        return storage;
    }

    /// Accesses raw internal state
    CUTLASS_HOST_DEVICE
    uint8_t raw() const {
        return storage;
    }

    /// Returns the sign bit
    CUTLASS_HOST_DEVICE
    bool signbit() const {
        return ((storage & (1 << (Base::FP8_NUM_BITS - 1))) != 0);
    }

    /// Returns the biased exponent
    CUTLASS_HOST_DEVICE
    int exponent_biased() const {
        return int((storage >> FP8_NUM_MANTISSA_BITS) & Base::FP8_EXPONENT_MASK);
    }

    /// Returns the unbiased exponent
    CUTLASS_HOST_DEVICE
    int exponent() const {
        return exponent_biased() - 15;
    }

    /// Returns the mantissa
    CUTLASS_HOST_DEVICE
    int mantissa() const {
        return int(storage & Base::FP8_MANTISSA_MASK);
    }
};

///////////////////////////////////////////////////////////////
///
/// floating-point 8 type : E5M2
///
///////////////////////////////////////////////////////////////
struct alignas(1) float_e5m2_t : float8_base<FloatEncoding::E5M2> {

    using Base = float8_base<FloatEncoding::E5M2>;

    static constexpr int MAX_EXPONENT = Base::FP8_MAX_EXPONENT;

    //
    // Static conversion operators
    //

    /// Constructs from an uint8_t
    CUTLASS_HOST_DEVICE
    static float_e5m2_t bitcast(uint8_t x) {
        float_e5m2_t f;
        f.storage = x;
        return f;
    }

    /// FP32 -> FP8 conversion - rounds to nearest even
    CUTLASS_HOST_DEVICE
    static float_e5m2_t from_float(float const& flt) {
    #if defined(CUDA_PTX_FP8_CVT_ENABLED)
        uint16_t tmp;
        float y = float();
        asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(tmp) : "f"(y), "f"(flt));

        return *reinterpret_cast<float_e5m2_t *>(&tmp);
    #else
        return bitcast(Base::convert_float_to_fp8(flt));
    #endif
    }

    /// FP16 -> E5M2 conversion - rounds to nearest even
    CUTLASS_HOST_DEVICE
    static float_e5m2_t from_half(half const& flt) {
    #if defined(CUDA_PTX_FP8_CVT_ENABLED)
        uint16_t tmp = 0;
        uint32_t bits = reinterpret_cast<uint16_t const &>(flt);
        asm volatile("cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;" : "=h"(tmp) : "r"(bits));

        return *reinterpret_cast<float_e5m2_t *>(&tmp);
    #else
        return bitcast(Base::convert_float_to_fp8(__half2float(flt)));
    #endif
    }

    // E5M2 -> half
    CUTLASS_HOST_DEVICE
    static half to_half(float_e5m2_t const& x) {
    #if defined(CUDA_PTX_FP8_CVT_ENABLED)
        uint16_t bits = x.storage;
        uint32_t packed;
        asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;\n" : "=r"(packed) : "h"(bits));

        return reinterpret_cast<half2 const &>(packed).x;
    #else
        return __float2half(Base::convert_fp8_to_float(x.storage));
    #endif
    }

    // E5M2 -> Float
    CUTLASS_HOST_DEVICE
    static float to_float(float_e5m2_t const& x) {
    #if defined(CUDA_PTX_FP8_CVT_ENABLED)
        uint16_t bits = x.storage;
        uint32_t packed;
        asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;\n" : "=r"(packed) : "h"(bits));

        return __half2float(reinterpret_cast<half2 const &>(packed).x);
    #else
        return Base::convert_fp8_to_float(x.storage);
    #endif
    }

    //
    // Methods
    //

    /// Default constructor
    CUTLASS_HOST_DEVICE
    float_e5m2_t() : Base() { }

    /// Reinterpret cast from CUDA's FP8 type
    CUTLASS_HOST_DEVICE
    float_e5m2_t(float_e5m2_t const& x) {
    #if defined(__CUDA_ARCH__)
        storage = reinterpret_cast<uint8_t const &>(x);
    #else
        uint8_t raw = x.storage;
        std::memcpy(&storage, &raw, sizeof(storage));
    #endif
    }

    /// Floating point conversion
    CUTLASS_HOST_DEVICE
    explicit float_e5m2_t(float x) {
        storage = from_float(x).storage;
    }

    CUTLASS_HOST_DEVICE
    explicit float_e5m2_t(half x) {
      storage = from_half(x).storage;
    }

    /// Floating point conversion
    CUTLASS_HOST_DEVICE
    explicit float_e5m2_t(double x): float_e5m2_t(float(x)) {
    }

    /// Integer conversion
    CUTLASS_HOST_DEVICE
    explicit float_e5m2_t(int x): float_e5m2_t(float(x)) {
    }

    /// E4M3 conversion
    CUTLASS_HOST_DEVICE
    explicit float_e5m2_t(float_e4m3_t x);

    /// Assignment
    CUTLASS_HOST_DEVICE
    float_e5m2_t & operator=(float_e5m2_t const &x) {
    #if defined(__CUDA_ARCH__)
        storage = reinterpret_cast<uint8_t const &>(x);
    #else
        uint8_t raw = x.storage;
        std::memcpy(&storage, &raw, sizeof(storage));
    #endif
        return *this;
    }

    /// Converts to float
    CUTLASS_HOST_DEVICE
    operator float() const {
        return to_float(*this);
    }

    /// Converts to half
    CUTLASS_HOST_DEVICE
    operator half() const {
      return to_half(*this);
    }

    /// Converts to float
    CUTLASS_HOST_DEVICE
    explicit operator double() const {
        return double(to_float(*this));
    }

    /// Converts to int
    CUTLASS_HOST_DEVICE
    explicit operator int() const {
    #if defined(__CUDA_ARCH__)
        return __half2int_rn(to_half(*this));
    #else
        return int(to_float(*this));
    #endif
    }

    /// Casts to bool
    CUTLASS_HOST_DEVICE
    explicit operator bool() const {
    #if defined(__CUDA_ARCH__)
        return bool(__half2int_rn(to_half(*this)));
    #else
        return bool(int(to_float(*this)));
    #endif
    }

    /// Accesses raw internal state
    CUTLASS_HOST_DEVICE
    uint8_t& raw() {
        return storage;
    }

    /// Accesses raw internal state
    CUTLASS_HOST_DEVICE
    uint8_t raw() const {
        return storage;
    }

    /// Returns the sign bit
    CUTLASS_HOST_DEVICE
    bool signbit() const {
        return ((storage & (1 << (Base::FP8_NUM_BITS - 1))) != 0);
    }

    /// Returns the biased exponent
    CUTLASS_HOST_DEVICE
    int exponent_biased() const {
        return int((storage >> FP8_NUM_MANTISSA_BITS) & Base::FP8_EXPONENT_MASK);
    }

    /// Returns the unbiased exponent
    CUTLASS_HOST_DEVICE
    int exponent() const {
        return exponent_biased() - 15;
    }

    /// Returns the mantissa
    CUTLASS_HOST_DEVICE
    int mantissa() const {
        return int(storage & Base::FP8_MANTISSA_MASK);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
bool operator==(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float(lhs) == float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator!=(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float(lhs) != float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator<(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float(lhs) < float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator<=(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float(lhs) <= float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator>(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float(lhs) > float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator>=(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float(lhs) >= float(rhs);
}

CUTLASS_HOST_DEVICE
float_e4m3_t operator+(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float_e4m3_t(float(lhs) + float(rhs));
}

CUTLASS_HOST_DEVICE
float_e4m3_t operator-(float_e4m3_t const& lhs) {
    return float_e4m3_t(-float(lhs));
}

CUTLASS_HOST_DEVICE
float_e4m3_t operator-(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float_e4m3_t(float(lhs) - float(rhs));
}

CUTLASS_HOST_DEVICE
float_e4m3_t operator*(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float_e4m3_t(float(lhs) * float(rhs));
}

CUTLASS_HOST_DEVICE
float_e4m3_t operator/(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float_e4m3_t(float(lhs) / float(rhs));
}

CUTLASS_HOST_DEVICE
float_e4m3_t& operator+=(float_e4m3_t & lhs, float_e4m3_t const& rhs) {
    lhs = float_e4m3_t(float(lhs) + float(rhs));
    return lhs;
}

CUTLASS_HOST_DEVICE
float_e4m3_t& operator-=(float_e4m3_t & lhs, float_e4m3_t const& rhs) {
    lhs = float_e4m3_t(float(lhs) - float(rhs));
    return lhs;
}

CUTLASS_HOST_DEVICE
float_e4m3_t& operator*=(float_e4m3_t & lhs, float_e4m3_t const& rhs) {
    lhs = float_e4m3_t(float(lhs) * float(rhs));
    return lhs;
}

CUTLASS_HOST_DEVICE
float_e4m3_t& operator/=(float_e4m3_t & lhs, float_e4m3_t const& rhs) {
    lhs = float_e4m3_t(float(lhs) / float(rhs));
    return lhs;
}

CUTLASS_HOST_DEVICE
float_e4m3_t& operator++(float_e4m3_t & lhs) {
    float tmp(lhs);
    ++tmp;
    lhs = float_e4m3_t(tmp);
    return lhs;
}

CUTLASS_HOST_DEVICE
float_e4m3_t& operator--(float_e4m3_t & lhs) {
    float tmp(lhs);
    --tmp;
    lhs = float_e4m3_t(tmp);
    return lhs;
}

CUTLASS_HOST_DEVICE
float_e4m3_t operator++(float_e4m3_t & lhs, int) {
    float_e4m3_t ret(lhs);
    float tmp(lhs);
    tmp++;
    lhs = float_e4m3_t(tmp);
    return ret;
}

CUTLASS_HOST_DEVICE
float_e4m3_t operator--(float_e4m3_t & lhs, int) {
    float_e4m3_t ret(lhs);
    float tmp(lhs);
    tmp--;
    lhs = float_e4m3_t(tmp);
    return ret;
}

CUTLASS_HOST_DEVICE
bool operator==(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float(lhs) == float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator!=(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float(lhs) != float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator<(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float(lhs) < float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator<=(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float(lhs) <= float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator>(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float(lhs) > float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator>=(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float(lhs) >= float(rhs);
}

CUTLASS_HOST_DEVICE
float_e5m2_t operator+(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float_e5m2_t(float(lhs) + float(rhs));
}

CUTLASS_HOST_DEVICE
float_e5m2_t operator-(float_e5m2_t const& lhs) {
    return float_e5m2_t(-float(lhs));
}

CUTLASS_HOST_DEVICE
float_e5m2_t operator-(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float_e5m2_t(float(lhs) - float(rhs));
}

CUTLASS_HOST_DEVICE
float_e5m2_t operator*(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float_e5m2_t(float(lhs) * float(rhs));
}

CUTLASS_HOST_DEVICE
float_e5m2_t operator/(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float_e5m2_t(float(lhs) / float(rhs));
}

CUTLASS_HOST_DEVICE
float_e5m2_t& operator+=(float_e5m2_t & lhs, float_e5m2_t const& rhs) {
    lhs = float_e5m2_t(float(lhs) + float(rhs));
    return lhs;
}

CUTLASS_HOST_DEVICE
float_e5m2_t& operator-=(float_e5m2_t & lhs, float_e5m2_t const& rhs) {
    lhs = float_e5m2_t(float(lhs) - float(rhs));
    return lhs;
}

CUTLASS_HOST_DEVICE
float_e5m2_t& operator*=(float_e5m2_t & lhs, float_e5m2_t const& rhs) {
    lhs = float_e5m2_t(float(lhs) * float(rhs));
    return lhs;
}

CUTLASS_HOST_DEVICE
float_e5m2_t& operator/=(float_e5m2_t & lhs, float_e5m2_t const& rhs) {
    lhs = float_e5m2_t(float(lhs) / float(rhs));
    return lhs;
}

CUTLASS_HOST_DEVICE
float_e5m2_t& operator++(float_e5m2_t & lhs) {
    float tmp(lhs);
    ++tmp;
    lhs = float_e5m2_t(tmp);
    return lhs;
}

CUTLASS_HOST_DEVICE
float_e5m2_t& operator--(float_e5m2_t & lhs) {
    float tmp(lhs);
    --tmp;
    lhs = float_e5m2_t(tmp);
    return lhs;
}

CUTLASS_HOST_DEVICE
float_e5m2_t operator++(float_e5m2_t & lhs, int) {
    float_e5m2_t ret(lhs);
    float tmp(lhs);
    tmp++;
    lhs = float_e5m2_t(tmp);
    return ret;
}

CUTLASS_HOST_DEVICE
float_e5m2_t operator--(float_e5m2_t & lhs, int) {
    float_e5m2_t ret(lhs);
    float tmp(lhs);
    tmp--;
    lhs = float_e5m2_t(tmp);
    return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// float_e4m3_t <=> float_e5m2_t conversions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// float_e4m3_t <= float_e5m2_t
CUTLASS_HOST_DEVICE
float_e4m3_t::float_e4m3_t(float_e5m2_t x) {
    storage = from_float(float_e5m2_t::to_float(x)).storage;
}

/// float_e5m2_t <= float_e4m3_t
CUTLASS_HOST_DEVICE
float_e5m2_t::float_e5m2_t(float_e4m3_t x) {
    storage = from_float(float_e4m3_t::to_float(x)).storage;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined(__CUDACC_RTC__)
namespace std {

/// Numeric limits common to all float8 types
template <typename T>
struct float8_base_numeric_limits {
private:
  using F8Type = T;
public:
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
  static std::float_denorm_style const has_denorm = std::denorm_present;
  static bool const has_denorm_loss = true;
  static std::float_round_style const round_style = std::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = F8Type::FP8_NUM_MANTISSA_BITS;

  /// Least positive value
  static F8Type min() { return F8Type::bitcast(0x01); }

  /// Maximum finite value
  static F8Type max() { return F8Type::bitcast(F8Type::FP8_MAX_FLT); }

  /// Returns maximum rounding error
  static F8Type round_error() { return F8Type(0.5f); }

  /// Returns positive infinity value
  static F8Type infinity() { return F8Type::bitcast(F8Type::FP8_INFINITY_MASK); }

  /// Returns quiet NaN value
  static F8Type quiet_NaN() { return F8Type::bitcast(F8Type::FP8_NAN); }

  /// Returns signaling NaN value
  static F8Type signaling_NaN() { return F8Type::bitcast(F8Type::FP8_NAN); }

  /// Returns smallest positive subnormal value
  static F8Type denorm_min() { return F8Type::bitcast(0x01); }
};

/// Numeric limits for float_e4m3_t
template <>
struct numeric_limits<cutlass::float_e4m3_t> :
    public float8_base_numeric_limits<cutlass::float_e4m3_t> {
  static bool const has_infinity = false;

  /// Minimum finite value
  static cutlass::float_e4m3_t lowest() { return cutlass::float_e4m3_t::bitcast(0xfe); }

  /// Returns smallest finite value
  static cutlass::float_e4m3_t epsilon() { return cutlass::float_e4m3_t::bitcast(0x20); }
};

/// Numeric limits for float_e5m2_t
template <>
struct numeric_limits<cutlass::float_e5m2_t>  :
    public float8_base_numeric_limits<cutlass::float_e5m2_t> {
  static bool const has_infinity = true;

  /// Minimum finite value
  static cutlass::float_e5m2_t lowest() { return cutlass::float_e5m2_t::bitcast(0xfb); }

  /// Returns smallest finite value
  static cutlass::float_e5m2_t epsilon() { return cutlass::float_e5m2_t::bitcast(0x34); }
};

}  // namespace std
#endif

namespace platform {

/// Numeric limits common to all float8 types
template <typename T>
struct float8_base_numeric_limits {
private:
  using F8Type = T;
public:
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
#if !defined(__CUDACC_RTC__)
  static std::float_denorm_style const has_denorm = std::denorm_present;
#endif
  static bool const has_denorm_loss = true;
#if !defined(__CUDACC_RTC__)
  static std::float_round_style const round_style = std::round_to_nearest;
#endif
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = F8Type::FP8_NUM_MANTISSA_BITS;

  /// Least positive value
  static F8Type min() { return F8Type::bitcast(0x01); }

  /// Maximum finite value
  static F8Type max() { return F8Type::bitcast(F8Type::FP8_MAX_FLT); }

  /// Returns maximum rounding error
  static F8Type round_error() { return F8Type(0.5f); }

  /// Returns positive infinity value
  static F8Type infinity() { return F8Type::bitcast(F8Type::FP8_INFINITY_MASK); }

  /// Returns quiet NaN value
  static F8Type quiet_NaN() { return F8Type::bitcast(F8Type::FP8_NAN); }

  /// Returns signaling NaN value
  static F8Type signaling_NaN() { return F8Type::bitcast(F8Type::FP8_NAN); }

  /// Returns smallest positive subnormal value
  static F8Type denorm_min() { return F8Type::bitcast(0x01); }
};

/// std::numeric_limits
template <class T>
struct numeric_limits;

/// Numeric limits for float_e4m3_t
template <>
struct numeric_limits<cutlass::float_e4m3_t> :
    public float8_base_numeric_limits<cutlass::float_e4m3_t> {
  static bool const has_infinity = false;

  /// Minimum finite value
  static cutlass::float_e4m3_t lowest() { return cutlass::float_e4m3_t::bitcast(0xfe); }

  /// Returns smallest finite value
  static cutlass::float_e4m3_t epsilon() { return cutlass::float_e4m3_t::bitcast(0x20); }
};

/// Numeric limits for float_e5m2_t
template <>
struct numeric_limits<cutlass::float_e5m2_t>  :
    public float8_base_numeric_limits<cutlass::float_e5m2_t> {
  static bool const has_infinity = true;

  /// Minimum finite value
  static cutlass::float_e5m2_t lowest() { return cutlass::float_e5m2_t::bitcast(0xfb); }

  /// Returns smallest finite value
  static cutlass::float_e5m2_t epsilon() { return cutlass::float_e5m2_t::bitcast(0x34); }
};

}  // namespace platform

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

CUTLASS_HOST_DEVICE
cutlass::float_e4m3_t operator "" _fe4m3(long double x) {
  return cutlass::float_e4m3_t(float(x));
}

CUTLASS_HOST_DEVICE
cutlass::float_e4m3_t operator "" _fe4m3(unsigned long long int x) {
  return cutlass::float_e4m3_t(int(x));
}

CUTLASS_HOST_DEVICE
cutlass::float_e5m2_t operator "" _fe5m2(long double x) {
  return cutlass::float_e5m2_t(float(x));
}

CUTLASS_HOST_DEVICE
cutlass::float_e5m2_t operator "" _fe5m2(unsigned long long int x) {
  return cutlass::float_e5m2_t(int(x));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
