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
 * this layernormware without specific prior written permission.
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
  
  \brief A file contains the unary ops
*/

#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////


/// Scalar multiplication
template <typename T, int N>
struct Mult {

    struct Arguments {
        T alpha;

        CUTLASS_HOST_DEVICE
        Arguments():alpha(T(1.0)){ }

        CUTLASS_HOST_DEVICE
        Arguments(T alpha): alpha(alpha) { }
    };
    
    struct Params {
        T alpha;   ///< scales accumulators

        CUTLASS_HOST_DEVICE
        Params():alpha(T(1.0)){ }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args): alpha(args.alpha) { }
    };

    T alpha_;

    CUTLASS_HOST_DEVICE
    Mult(
        Params const &params
    ):
        alpha_(params.alpha)
    { }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &source) const {
        cutlass::multiplies<Array<T, N>> multiply_op;
        return multiply_op(source, alpha_);
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return alpha_ != T(0);
    }

};


/// ReLU
template <typename T, int N>
struct ReLUVisitor {
    struct Arguments {
        T threshold;

        CUTLASS_HOST_DEVICE
        Arguments():threshold(T(0.0)) { }

        CUTLASS_HOST_DEVICE
        Arguments(T threshold): threshold(threshold) { }
    };

    struct Params {
        T threshold;

        CUTLASS_HOST_DEVICE
        Params():threshold(T(0.0)) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args): threshold(args.threshold) { }
    };

    T threshold_;

    CUTLASS_HOST_DEVICE
    ReLUVisitor(Params const &params):
        threshold_(params.threshold) { }
    
    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &frag) const {
        maximum<Array<T, N>> mx;
        return mx(frag, threshold_);
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }
};

/// leakyReLU
template <typename T, int N>
struct LeakyReLUVisitor {
    struct Arguments {
        T leaky_alpha;

        CUTLASS_HOST_DEVICE
        Arguments():leaky_alpha(T(0.0)) { }

        CUTLASS_HOST_DEVICE
        Arguments(T leaky_alpha): leaky_alpha(leaky_alpha) { }
    };

    struct Params {
        T leaky_alpha;

        CUTLASS_HOST_DEVICE
        Params():leaky_alpha(T(0.0)) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args): leaky_alpha(args.leaky_alpha) { }
    };

    T leaky_alpha_;

    CUTLASS_HOST_DEVICE
    LeakyReLUVisitor(Params const &params):
        leaky_alpha_(params.leaky_alpha) { }
    
    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &frag) const {
        cutlass::epilogue::thread::LeakyReLU<Array<T, N>> leaky_op;
        return leaky_op(frag, leaky_alpha_);
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }
    
};

/// Tanh
template <typename T, int N>
struct TanhVisitor {
    /// Argument
    struct Arguments {
        // a placeholder argument to ensure correctness of ctypes
        int tmp;

        CUTLASS_HOST_DEVICE
        Arguments(): tmp(0) { };

        CUTLASS_HOST_DEVICE
        Arguments(int tmp): tmp(tmp) { };
    };

    /// Param
    struct Params {
        CUTLASS_HOST_DEVICE
        Params(){ };
        Params(Arguments const &args) { }
    };

    /// Constructor
    CUTLASS_HOST_DEVICE
    TanhVisitor(Params const &params) { }

    // scalar operator
    CUTLASS_HOST_DEVICE
    T tanh_op(T const &scalar) const {
        return fast_tanh(scalar);
    }

    /// vector operator
    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &frag) const {
        Array<T, N> y;

        CUTLASS_PRAGMA_UNROLL
        for (int i=0; i < N; ++i) {
            y[i] = tanh_op(frag[i]);
        }

        return y;
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
