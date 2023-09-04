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
   \brief Bind Convolution problem sizes to python
*/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "cutlass/conv/conv2d_problem_size.h"

namespace py = pybind11;

void bind_conv_problem_size(py::module &m) {
    //
    // Conv2d Problem Size: 
    // include/cutlass/conv/conv2d_problem_sizd.h
    //
    py::class_<cutlass::conv::Conv2dProblemSize>(m, "Conv2dProblemSize")
         // constructors
        .def(py::init<int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cutlass::conv::Mode, int, int>())
        .def(py::init<cutlass::Tensor4DCoord, cutlass::Tensor4DCoord, cutlass::Tensor4DCoord, cutlass::MatrixCoord, cutlass::MatrixCoord, cutlass::conv::Mode, int, int>())
        // attribute accessors
        .def_readwrite("N", &cutlass::conv::Conv2dProblemSize::N)
        .def_readwrite("H", &cutlass::conv::Conv2dProblemSize::H)
        .def_readwrite("W", &cutlass::conv::Conv2dProblemSize::W)
        .def_readwrite("C", &cutlass::conv::Conv2dProblemSize::C)
        .def_readwrite("P", &cutlass::conv::Conv2dProblemSize::P)
        .def_readwrite("Q", &cutlass::conv::Conv2dProblemSize::Q)
        .def_readwrite("K", &cutlass::conv::Conv2dProblemSize::K)
        .def_readwrite("R", &cutlass::conv::Conv2dProblemSize::R)
        .def_readwrite("S", &cutlass::conv::Conv2dProblemSize::S)
        .def_readwrite("pad_h", &cutlass::conv::Conv2dProblemSize::pad_h)
        .def_readwrite("pad_w", &cutlass::conv::Conv2dProblemSize::pad_w)
        .def_readwrite("stride_h", &cutlass::conv::Conv2dProblemSize::stride_h)
        .def_readwrite("stride_w", &cutlass::conv::Conv2dProblemSize::stride_w)
        .def_readwrite("dilation_h", &cutlass::conv::Conv2dProblemSize::dilation_h)
        .def_readwrite("dilation_w", &cutlass::conv::Conv2dProblemSize::dilation_w)
        .def_readwrite("mode", &cutlass::conv::Conv2dProblemSize::mode)
        .def_readwrite("split_k_slices", &cutlass::conv::Conv2dProblemSize::split_k_slices)
        .def_readwrite("groups", &cutlass::conv::Conv2dProblemSize::groups)
        // functions
        .def("reset_split_k_slices", &cutlass::conv::Conv2dProblemSize::reset_split_k_slices)
        .def("activation_extent", &cutlass::conv::Conv2dProblemSize::activation_extent)
        .def("filter_extent", &cutlass::conv::Conv2dProblemSize::filter_extent)
        .def("output_extent", &cutlass::conv::Conv2dProblemSize::output_extent)
        .def("activation_size", &cutlass::conv::Conv2dProblemSize::activation_size)
        .def("filter_size", &cutlass::conv::Conv2dProblemSize::filter_size)
        .def("output_size", &cutlass::conv::Conv2dProblemSize::output_size);
    
    // Get tensor size
    m.def("implicit_gemm_tensor_a_size", py::overload_cast<cutlass::conv::Operator, const cutlass::conv::Conv2dProblemSize&>(&cutlass::conv::implicit_gemm_tensor_a_size));
    m.def("implicit_gemm_tensor_b_size", py::overload_cast<cutlass::conv::Operator, const cutlass::conv::Conv2dProblemSize&>(&cutlass::conv::implicit_gemm_tensor_b_size));
    m.def("implicit_gemm_tensor_c_size", py::overload_cast<cutlass::conv::Operator, const cutlass::conv::Conv2dProblemSize&>(&cutlass::conv::implicit_gemm_tensor_c_size));

    // Get tensor extent
    m.def("implicit_gemm_tensor_a_extent",
        py::overload_cast<
            cutlass::conv::Operator, const cutlass::conv::Conv2dProblemSize&
        >(&cutlass::conv::implicit_gemm_tensor_a_extent));

    m.def("implicit_gemm_tensor_b_extent",
        py::overload_cast<
            cutlass::conv::Operator, const cutlass::conv::Conv2dProblemSize&
        >(&cutlass::conv::implicit_gemm_tensor_b_extent));
    
    m.def("implicit_gemm_tensor_c_extent",
        py::overload_cast<
            cutlass::conv::Operator, const cutlass::conv::Conv2dProblemSize&
        >(&cutlass::conv::implicit_gemm_tensor_c_extent));
    
    m.def("implicit_gemm_problem_size", py::overload_cast<cutlass::conv::Operator, const cutlass::conv::Conv2dProblemSize &>(&cutlass::conv::implicit_gemm_problem_size));
    
}
