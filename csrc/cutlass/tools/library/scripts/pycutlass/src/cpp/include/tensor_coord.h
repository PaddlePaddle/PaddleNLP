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
   \brief Bind Tensor Coord to python
*/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "cutlass/tensor_coord.h"

namespace py = pybind11;

void bind_tensor_coord(py::module &m) {
    //
    // Tensor Coords
    // cutlass/include/cutlass/tensor_coord.h
    //

    /// Defines a canonical 4D coordinate used by tensor operations.
    py::class_<cutlass::Tensor4DCoord>(m, "Tensor4DCoord",
        R"pbdoc(Defines a canonical 4D coordinate used by tensor operations)pbdoc")
        .def(py::init<int, int, int, int>(),
            py::arg("n"), py::arg("h"), py::arg("w"), py::arg("c"),
            R"pbdoc(Helper to construct from N, H, W, and C)pbdoc")
        .def("at", py::overload_cast<int>(&cutlass::Tensor4DCoord::at),
            py::arg("dim"),
            R"pbdoc(Gets the index of a given Coord element)pbdoc")
        .def("size", [](const cutlass::Tensor4DCoord & coord) {
            return coord.at(0) * coord.at(1) * coord.at(2) * coord.at(3);},
            R"pbdoc(The size of the tensor coord)pbdoc");
    
    py::class_<cutlass::Coord<3>>(m, "Tensor3DCoord",
        R"pbdoc(Defines a canonical 3D coordinate used by tensor operations)pbdoc")
        .def("at", py::overload_cast<int>(&cutlass::Coord<3>::at),
            py::arg("dim"),
            R"pbdoc(Gets the index of a given Coord element)pbdoc");

    // Matrix Size
    py::class_<cutlass::MatrixCoord>(m, "MatrixCoord",
        R"pbdoc(MatrixCoord wraps Coord<2, int> to provide a helper for accessing named dimensions. Classes
        expecting a coordinate in the rank=2 index space of a matrix should use MatrixCoord.)pbdoc")
        .def(py::init<int, int>(),
            py::arg("row"), py::arg("column"), R"pbdoc(Helper to construct from a row and column)pbdoc")
        .def("row", py::overload_cast<>(&cutlass::MatrixCoord::row),
            R"pbdoc(Returns the row of the coordinate)pbdoc")
        .def("column", py::overload_cast<>(&cutlass::MatrixCoord::column),
            R"pbdoc(Returns the column of the coordinate)pbdoc");

}
