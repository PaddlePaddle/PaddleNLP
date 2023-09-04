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
/*! \file
    \brief Define basic numeric operators

    This is inspired by the Standard Library's <functional> header.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include <mma.h>
#endif // defined(CUTLASS_ARCH_WMMA_ENABLED)

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct absolute_value_op {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return abs(lhs);
  }
};

template <>
struct absolute_value_op<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float lhs) const { return fabs(lhs); }
};

template <typename T>
struct plus {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs += rhs;
    return lhs;
  }
};

template <typename T>
struct minus {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs -= rhs;
    return lhs;
  }
};

template <typename T>
struct multiplies {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs *= rhs;
    return lhs;
  }
};

#if defined(__CUDA_ARCH__)
/// Partial specializations needed when __CUDA_NO_HALF2_OPERATORS__ is set
template<>
struct plus<__half2> {
  CUTLASS_HOST_DEVICE
  __half2 operator()(__half2 lhs, __half2 const &rhs) const {
    return __hadd2(lhs, rhs);
  }
};

template<>
struct minus<__half2> {
  CUTLASS_HOST_DEVICE
  __half2 operator()(__half2 lhs, __half2 const &rhs) const {
    return __hsub2(lhs, rhs);
  }
};

template<>
struct multiplies<__half2> {
  CUTLASS_HOST_DEVICE
  __half2 operator()(__half2 lhs, __half2 const &rhs) const {
    return __hmul2(lhs, rhs);
  }
};

/// Partial specializations needed when __CUDA_NO_HALF_OPERATORS__ is set
template<>
struct plus<__half> {
  CUTLASS_HOST_DEVICE
  __half operator()(__half lhs, __half const &rhs) const {
    return __hadd(lhs, rhs);
  }
};

template<>
struct minus<__half> {
  CUTLASS_HOST_DEVICE
  __half operator()(__half lhs, __half const &rhs) const {
    return __hsub(lhs, rhs);
  }
};

template<>
struct multiplies<__half> {
  CUTLASS_HOST_DEVICE
  __half operator()(__half lhs, __half const &rhs) const {
    return __hmul(lhs, rhs);
  }
};
#endif // defined(__CUDA_ARCH__)


// Maximum with nan propogation
// To propgate the NANs, the "max" of a two element that contains NaNs should also return a NaN 
template <typename T>
struct maximum_with_nan_propogation {
  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    return lhs > rhs or std::isnan(lhs) ? lhs : rhs;
  }
};

template <>
struct maximum_with_nan_propogation<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float const lhs, float const rhs) const {
    float res;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("max.NaN.f32 %0, %1, %2;\n" : "=f"(res) : "f"(lhs), "f"(rhs));
#else
    res = lhs > rhs or std::isnan(lhs) ? lhs : rhs;
#endif
    return res;
  }
};

/// Squares with optional conversion
template <typename T, typename Output = T>
struct square {
  CUTLASS_HOST_DEVICE
  Output operator()(T lhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs);
    return mul_op(y, y);
  }
};

/// Returns the magnitude squared of an element.
template <typename T, typename Output = T>
struct magnitude_squared {
  CUTLASS_HOST_DEVICE
  Output operator()(T lhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs);
    return mul_op(y, y);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output = T>
struct square_difference {
  CUTLASS_HOST_DEVICE
  Output operator()(T lhs, T rhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs) - Output(rhs);
    return mul_op(y, y);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output = T>
struct magnitude_squared_difference {
  CUTLASS_HOST_DEVICE
  Output operator()(T lhs, T rhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs) - Output(rhs);
    return mul_op(y, y);
  }
};

/// Divides
template <typename T>
struct divides {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs /= rhs;
    return lhs;
  }
};

/// Negate
template <typename T>
struct negate {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return -lhs;
  }
};

/// Greater equal 
template <typename T>
struct greater_equal {
  CUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs >= rhs);
  }
};

/// Greater  
template <typename T>
struct greater {
  CUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs > rhs);
  }
};

/// Less equal 
template <typename T>
struct less_equal {
  CUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs <= rhs);
  }
};

/// Less  
template <typename T>
struct less {
  CUTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs < rhs);
  }
};

template <typename T>
struct maximum {

  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    return (lhs < rhs ? rhs : lhs);
  }
};

template <>
struct maximum<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float const &lhs, float const &rhs) const {
    return fmaxf(lhs, rhs);
  }
};

template <typename T>
struct minimum {

  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    return (rhs < lhs ? rhs : lhs);
  }
};

template <>
struct minimum<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float const &lhs, float const &rhs) const {
    return fminf(lhs, rhs);
  }
};

/// Fused multiply-add
template <typename A, typename B = A, typename C = A>
struct multiply_add {
  CUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    return C(a) * C(b) + c;
  }
};

/// Fused multiply-add
template <typename A, typename B = A, typename C = A>
struct multiply_add_relu0 {
  CUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    maximum<C> mx;
    return mx(C(a) * C(b) + c, C(0));
  }
};

/// Fused multiply-add
template <typename T>
struct and_add {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a & b) + c);
  }
};


/// Fused multiply-add
template <typename T>
struct xor_add {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a ^ b) + c);
  }
};

template <typename T>
struct conjugate {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return a;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct logical_and {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return ((a && b) ? T(1) : T());
  }
};

template <typename T>
struct logical_or {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return ((a || b) ? T(1) : T());
  }
};

template <typename T>
struct logical_not {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return T(!(a));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct bit_and {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a & b;
  }
};

template <typename T>
struct bit_or {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a | b;
  }
};

template <typename T>
struct bit_not {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return ~a;
  }
};

template <typename T>
struct bit_xor {
  CUTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a ^ b;
  }
};



//////////////////////////////////////////////////////////////////////////////////////////////////

/// Reduces value into the data pointed to by ptr
template <typename T>
struct red
{
  CUTLASS_DEVICE
  void operator()(T *ptr, const T &data)
  {
    atomicAdd(ptr, data);
  }
};


/// Reduces value into the data pointed to by ptr (double specialization)
template<>
struct red<double>
{
  CUTLASS_DEVICE
  void operator()(double *ptr, const double &data)
  {
#if !defined(__CUDA_ARCH__)
      CUTLASS_UNUSED(ptr);
      CUTLASS_UNUSED(data);
#elif (__CUDA_ARCH__ >= 600)

    atomicAdd(ptr, data);

#else

    // Use CAS loop
    unsigned long long int* ptr_int = reinterpret_cast<unsigned long long int*>(ptr);
    unsigned long long int old_int = *ptr_int;
    unsigned long long int assumed_int;

    do {
      double update = data + __longlong_as_double(old_int);
      assumed_int = old_int;
      old_int = atomicCAS(ptr_int, assumed_int, __double_as_longlong(update));
    } while (assumed_int != old_int);

#endif // (__CUDA_ARCH__ >= 600)
  }
};


/// Reduces value into the data pointed to by ptr (half2 specialization)
template<>
struct red<half2>
{
  CUTLASS_DEVICE
  void operator()(half2 *ptr, const half2 &data)
  {
#if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__)  && (__CUDA_ARCH__ < 600))
      CUTLASS_UNUSED(ptr);
      CUTLASS_UNUSED(data);
#else

    // Vector-2 atomic reduction requires .target sm_60 or higher
    uint32_t word = reinterpret_cast<const uint32_t&>(data);
    asm volatile ("red.gpu.global.add.noftz.f16x2 [%0], %1;\n" : : "l"(ptr), "r"(word));

#endif // (__CUDA_ARCH__ >= 600)
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for nvcuda::wmma::fragment<Use, m, n, k, T, Layout>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_WMMA_ENABLED)

template<typename Use, int m, int n, int k, typename T, typename Layout>
struct plus<nvcuda::wmma::fragment<Use, m, n, k, T, Layout>>
{
  using Fragment = nvcuda::wmma::fragment<Use, m, n, k, T, Layout>;
  using ElementType = typename Fragment::element_type;

  CUTLASS_HOST_DEVICE
  Fragment operator()(Fragment const &lhs, Fragment const &rhs) const
  {
    Fragment result;
    plus<ElementType> scalar_op;

    ElementType *result_elts = reinterpret_cast<ElementType*>(&result);
    const ElementType *lhs_elts = reinterpret_cast<const ElementType*>(&lhs);
    const ElementType *rhs_elts = reinterpret_cast<const ElementType*>(&rhs);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Fragment::num_elements; i++) {
      result_elts[i] = scalar_op(lhs_elts[i], rhs_elts[i]);
    }

    return result;
  }
};

#endif // defined(CUTLASS_ARCH_WMMA_ENABLED)



/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
