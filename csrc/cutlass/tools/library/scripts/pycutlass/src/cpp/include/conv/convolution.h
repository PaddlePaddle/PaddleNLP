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
/* \file
   \brief Bind convolution related enum types to python
*/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "conv_problem_size.h"
#include "host.h"
#include "cutlass/conv/convolution.h"

namespace py = pybind11;

void bind_convolution(py::module &m) {
    //
    // Enumerate types
    // cutlass/include/cutlass/conv/convolution.h
    //

    /// Convolutional operator
    py::enum_<cutlass::conv::Operator>(m, "Operator", R"pbdoc(Convolutional operator)pbdoc")
        .value("fprop", cutlass::conv::Operator::kFprop, "Forward propagation")
        .value("dgrad", cutlass::conv::Operator::kDgrad, "Activation grad")
        .value("wgrad", cutlass::conv::Operator::kWgrad, "Weight grad");

    /// Distinguishes convolution  from cross correlation
    py::enum_<cutlass::conv::Mode>(m, "Mode")
        .value("cross_correlation", cutlass::conv::Mode::kCrossCorrelation)
        .value("convolution", cutlass::conv::Mode::kConvolution);
    
    /// Selects among several implementation variants trading off performance with simplicity
    py::enum_<cutlass::conv::IteratorAlgorithm>(m, "IteratorAlgorithm",
        R"pbdoc(Selects among several implementation variants trading off performance with simplicity)pbdoc")
        .value("analytic", cutlass::conv::IteratorAlgorithm::kAnalytic, R"pbdoc(functionally correct in all cases but lower performance)pbdoc")
        .value("optimized", cutlass::conv::IteratorAlgorithm::kOptimized, R"pbdoc(optimized for R <= 32, S <= 32 and unity-stride dgrad)pbdoc")
        .value("fixed_channels", cutlass::conv::IteratorAlgorithm::kFixedChannels, R"pbdoc(Analytic algorithm optimized for fixed channel count (C == AccessSize))pbdoc")
        .value("few_channels", cutlass::conv::IteratorAlgorithm::kFewChannels, R"pbdoc(Analytic algorithm optimized for few channels (C divisible by AccessSize))pbdoc");
    
    /// Distinguishes among partial specializations that accelerate certain problems where convolution
    /// stride is unit.
    py::enum_<cutlass::conv::StrideSupport>(m, "StrideSupport",
        R"pbdoc(Distinguishes among partial specializations that accelerate certain problems where convolution
        stride is unit.)pbdoc")
        .value("strided", cutlass::conv::StrideSupport::kStrided, R"pbdoc(arbitrary convolution stride)pbdoc")
        .value("unity", cutlass::conv::StrideSupport::kUnity, R"pbdoc(unit convolution stride)pbdoc");
    
    /// Identifies split-K mode
    py::enum_<cutlass::conv::SplitKMode>(m, "SplitKMode")
        .value("None", cutlass::conv::SplitKMode::kNone)
        .value("Serial", cutlass::conv::SplitKMode::kSerial)
        .value("Parallel", cutlass::conv::SplitKMode::kParallel);
    
    // Conv problem sizes
    bind_conv_problem_size(m);

    //
    // host helper functions
    //
    py::module_ host_submodule = m.def_submodule("host");
    bind_conv_host_helper(host_submodule);
}
