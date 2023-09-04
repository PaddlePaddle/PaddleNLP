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
    \brief Reference implementation for GETT in host-side code.
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/complex.h"
#include "cutlass/numeric_conversion.h"

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::reference::host {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ElementAccumulator_,
  class TensorA_,                                                                                         // (M, K, L)
  class TensorB_                                                                                          // (N, K, L)
>
struct GettMainloopParams {
  using ElementAccumulator = ElementAccumulator_;
  using TensorA = TensorA_;
  using TensorB = TensorB_;
  using EngineA = typename TensorA::engine_type;
  using LayoutA = typename TensorA::layout_type;
  using EngineB = typename TensorB::engine_type;
  using LayoutB = typename TensorB::layout_type;

  TensorA A{};
  TensorB B{};

  ComplexTransform transform_A = ComplexTransform::kNone;
  ComplexTransform transform_B = ComplexTransform::kNone;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ElementScalar_,
  class ElementAccumulator_,
  class ElementCompute_,
  class TensorC_,                                                                                          // (M, N, L)
  class TensorD_                                                                                           // (M, N, L)
>
struct GettEpilogueParams {
  using ElementScalar = ElementScalar_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using TensorC = TensorC_;
  using TensorD = TensorD_;
  using EngineC = typename TensorC::engine_type;
  using LayoutC = typename TensorC::layout_type;
  using EngineD =  typename TensorD::engine_type;
  using LayoutD = typename TensorD::layout_type;
  ElementScalar alpha = ElementScalar(1);
  ElementScalar beta = ElementScalar(0);

  TensorC C{};
  TensorD D{};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GETT - General Tensor-Tensor contraction reference kernel
template <
  class MainloopParams,
  class EpilogueParams
>
void Gett(
    MainloopParams const& mainloop_params,
    EpilogueParams const& epilogue_params)
{

  static int constexpr kBlockM = 64;
  static int constexpr kBlockN = 64;

  #pragma omp parallel for collapse(3)
  for (int64_t l = 0; l < cute::size<2>(mainloop_params.A.layout()); ++l) {
    for (int64_t m = 0; m < cute::size<0>(mainloop_params.A.layout()); m += kBlockM) {
      for (int64_t n = 0; n < cute::size<0>(mainloop_params.B.layout()); n += kBlockN) {
        typename MainloopParams::ElementAccumulator acc[kBlockM][kBlockN];
        gett_mainloop(mainloop_params, m, n, l, acc);
        gett_epilogue(epilogue_params, m, n, l, acc);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GETT - Mainloop
template <class MainloopParams, class ElementAccumulator, int kBlockM, int kBlockN>
void gett_mainloop(
    MainloopParams const& mainloop_params,
    int64_t m,
    int64_t n,
    int64_t l,
    ElementAccumulator (&acc)[kBlockM][kBlockN])
{

  static_assert(cute::rank(typename MainloopParams::LayoutA{}) == 3, "M, K, B");
  static_assert(cute::rank(typename MainloopParams::LayoutB{}) == 3, "N, K, B");

  using ElementA = typename MainloopParams::EngineA::value_type;
  using ElementB = typename MainloopParams::EngineB::value_type;

  using RingOp = multiply_add<ElementAccumulator, ElementAccumulator, ElementAccumulator>;
  RingOp fma_op;

  // Zero out accumulators
  for (int m_b = 0; m_b < kBlockM; ++m_b) {
    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      acc[m_b][n_b] = ElementAccumulator(0); // RingOp::AdditionIdentity
    }
  }

  // Compute on this k-block
  for (int64_t k = 0; k < cute::size<1>(mainloop_params.A.layout()); ++k) {
    // Load A
    ElementAccumulator a_frag[kBlockM];
    for (int m_b = 0; m_b < kBlockM; ++m_b) {
      if (m + m_b < cute::size<0>(mainloop_params.A.layout())) {
        a_frag[m_b] = static_cast<ElementAccumulator>(mainloop_params.A(m + m_b, k, l));
        if (mainloop_params.transform_A == ComplexTransform::kConjugate) {
          a_frag[m_b] = conj(a_frag[m_b]);
        }
      } else {
        a_frag[m_b] = ElementAccumulator(0); // RingOp::AdditionIdentity
      }
    }

    // Load B
    ElementAccumulator b_frag[kBlockN];
    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      if (n + n_b < cute::size<0>(mainloop_params.B.layout())) {
        b_frag[n_b] = static_cast<ElementAccumulator>(mainloop_params.B(n + n_b, k, l));
        if (mainloop_params.transform_B == ComplexTransform::kConjugate) {
          b_frag[n_b] = conj(b_frag[n_b]);
        }
      } else {
        b_frag[n_b] = ElementAccumulator(0); // RingOp::AdditionIdentity
      }
    }

    // do compute
    for (int m_b = 0; m_b < kBlockM; ++m_b) {
      for (int n_b = 0; n_b < kBlockN; ++n_b) {
        acc[m_b][n_b] = fma_op(a_frag[m_b], b_frag[n_b], acc[m_b][n_b]);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GETT - Epilogue
template <class EpilogueParams, class ElementAccumulator, int kBlockM, int kBlockN>
void gett_epilogue(
    EpilogueParams const& epilogue_params,
    int64_t m,
    int64_t n,
    int64_t l,
    ElementAccumulator (&acc)[kBlockM][kBlockN])
{
  static_assert(cute::rank(typename EpilogueParams::LayoutC{}) == 3, "M, K, B");
  static_assert(cute::rank(typename EpilogueParams::LayoutD{}) == 3, "N, K, B");

  using ElementCompute = typename EpilogueParams::ElementCompute;
  using ElementC = typename EpilogueParams::EngineC::value_type;

  using ElementD = typename EpilogueParams::EngineD::value_type;
  using ElementScalar = typename EpilogueParams::ElementScalar;
  // Input related converter
  NumericConverter<ElementCompute, ElementAccumulator> accumulator_converter;
  NumericConverter<ElementCompute, ElementC> source_converter;

  // Scale related converter
  NumericConverter<ElementCompute, ElementScalar> scale_converter;
  // Output related converter
  NumericConverter<ElementD, ElementCompute> destination_converter;
  // Epilogue operations
  multiply_add<ElementCompute, ElementCompute, ElementCompute> epilogue_fma;
  multiplies<ElementCompute> mul;

  // Do conversion
  ElementCompute converted_alpha = scale_converter(epilogue_params.alpha);
  ElementCompute converted_beta = scale_converter(epilogue_params.beta);
  for (int n_b = 0; n_b < kBlockN; ++n_b) {
    for (int m_b = 0; m_b < kBlockM; ++m_b) {
      if (m + m_b < cute::size<0>(epilogue_params.D.layout()) && n + n_b < cute::size<1>(epilogue_params.D.layout())) {
        // Convert every type to ElementCompute first, do compute, convert to output type, write it out
        ElementCompute converted_acc = accumulator_converter(acc[m_b][n_b]);
        ElementCompute converted_src = source_converter(epilogue_params.C(m + m_b, n + n_b, l));

        ElementScalar output = epilogue_fma(converted_alpha, converted_acc, ElementCompute(0));
        output = epilogue_fma(converted_beta, converted_src, output);

        epilogue_params.D(m + m_b, n + n_b, l) = destination_converter(output);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GEMM - General Matrix-Matrix contraction without conjugation options
template <
  class MainloopParams,
  class EpilogueParams
>
void Gemm3x(
    MainloopParams const& mainloop_params,
    EpilogueParams const& epilogue_params)
{
  using namespace cute;

  static_assert(rank(typename MainloopParams::LayoutA{}) == rank(typename MainloopParams::LayoutB{}));
  static_assert(rank(typename EpilogueParams::LayoutC{}) == rank(typename EpilogueParams::LayoutD{}));
  static_assert(rank(typename MainloopParams::LayoutA{}) == rank(typename EpilogueParams::LayoutC{}));

  if constexpr (rank(typename MainloopParams::LayoutA{}) == 2) {
    // append a batch mode of size 1 if we do not have tensors that are rank 3
    Layout layout_A = make_layout(
      make_shape(get<0>(mainloop_params.A.shape()), get<1>(mainloop_params.A.shape()), Int<1>{}),
      make_stride(get<0>(mainloop_params.A.stride()), get<1>(mainloop_params.A.stride()), int64_t(cosize(mainloop_params.A.layout()))));

    Layout layout_B = make_layout(
      make_shape(get<0>(mainloop_params.B.shape()), get<1>(mainloop_params.B.shape()), Int<1>{}),
      make_stride(get<0>(mainloop_params.B.stride()), get<1>(mainloop_params.B.stride()), int64_t(cosize(mainloop_params.B.layout()))));

    Layout layout_C = make_layout(
      make_shape(get<0>(epilogue_params.C.shape()), get<1>(epilogue_params.C.shape()), Int<1>{}),
      make_stride(get<0>(epilogue_params.C.stride()), get<1>(epilogue_params.C.stride()), int64_t(cosize(epilogue_params.C.layout()))));

    Layout layout_D = make_layout(
      make_shape(get<0>(epilogue_params.D.shape()), get<1>(epilogue_params.D.shape()), Int<1>{}),
      make_stride(get<0>(epilogue_params.D.stride()), get<1>(epilogue_params.D.stride()), int64_t(cosize(epilogue_params.D.layout()))));
    auto TensorA = make_tensor(mainloop_params.A.data(), layout_A);
    auto TensorB = make_tensor(mainloop_params.B.data(), layout_B);
    auto TensorC = make_tensor(epilogue_params.C.data(), layout_C);
    auto TensorD = make_tensor(epilogue_params.D.data(), layout_D);
    // Reconstruct mainloop params
    GettMainloopParams<typename MainloopParams::ElementAccumulator,
                       decltype(TensorA),
                       decltype(TensorB)>
        mainloop_params_converted{TensorA,
                                  TensorB,
                                  mainloop_params.transform_A,
                                  mainloop_params.transform_B};

    // Reconstruct epilogue params
    GettEpilogueParams<typename EpilogueParams::ElementScalar,
                       typename EpilogueParams::ElementAccumulator,
                       typename EpilogueParams::ElementCompute,
                       decltype(TensorC),
                       decltype(TensorD)
                      >
        epilogue_params_converted{epilogue_params.alpha,
                                  epilogue_params.beta,
                                  TensorC,
                                  TensorD
                                  };

    Gett(mainloop_params_converted, epilogue_params_converted);
  }
  else {
    // if we already have a batch mode, just pass it through
    Gett(mainloop_params, epilogue_params);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // cutlass::reference::host

/////////////////////////////////////////////////////////////////////////////////////////////////
