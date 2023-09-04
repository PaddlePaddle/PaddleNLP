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
   \brief Bind Matrix layouts to python
*/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "cutlass/layout/matrix.h"

namespace py = pybind11;

void bind_matrix_layout(py::module &m) {
    //
    // Matrix layouts
    // cutlass/layout/matrix.h
    //

    py::class_<cutlass::layout::RowMajor>(m, "RowMajor", R"pbdoc(
        Mapping function for row-major matrices.
    )pbdoc")
        .def_static("packed", &cutlass::layout::RowMajor::packed, 
            py::arg("extent"), 
            R"pbdoc(Helper returns a layout to a tightly packed tensor)pbdoc")
        .def("stride", [](const cutlass::layout::RowMajor & layout){
            return layout.stride().at(0);
        }, R"pbdoc(Returns the stride of the layout)pbdoc");

    py::class_<cutlass::layout::ColumnMajor>(m, "ColumnMajor", R"pbdoc(
        Mapping function for column-major matrices.
    )pbdoc")
        .def_static("packed", &cutlass::layout::ColumnMajor::packed, 
            py::arg("extent"),
            R"pbdoc(Helper returns a layout to a tightly packed tensor)pbdoc" )
        .def("stride", [](const cutlass::layout::ColumnMajor & layout){
            return layout.stride().at(0);
        }, R"pbdoc(Returns the stride of the layout)pbdoc");

    py::class_<cutlass::layout::RowMajorInterleaved<32>>(m, "RowMajorInterleaved32",
        R"pbdoc(Mapping function for interleaved matrices. Matrix is structured 
        as row-major arrangement of fixed-size columns 32)pbdoc")
        .def_static("packed", &cutlass::layout::RowMajorInterleaved<32>::packed,
            py::arg("extent"), 
            R"pbdoc(Helper returns a layout to a tightly packed tensor)pbdoc")
        .def("stride", [](const cutlass::layout::RowMajorInterleaved<32> & layout){
            return layout.stride().at(0);
        }, R"pbdoc(Returns the stride of the layout)pbdoc");

    py::class_<cutlass::layout::ColumnMajorInterleaved<32>>(m, "ColumnMajorInterleaved32",
        R"pbdoc(Mapping function for interleaved matrices. Matrix is structured 
        as column-major arrangement of fixed-size rows 32)pbdoc")
        .def_static("packed", &cutlass::layout::ColumnMajorInterleaved<32>::packed,
            py::arg("extent"), 
            R"pbdoc(Helper returns a layout to a tightly packed tensor)pbdoc")
        .def("stride", [](const cutlass::layout::ColumnMajorInterleaved<32> & layout){
            return layout.stride().at(0);
        }, R"pbdoc(Returns the stride of the layout)pbdoc");
}
