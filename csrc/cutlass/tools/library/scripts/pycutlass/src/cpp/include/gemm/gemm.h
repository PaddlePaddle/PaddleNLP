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
   \brief Bind gemm related enum types to python
*/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "cutlass/gemm/gemm.h"
#include "host.h"

namespace py = pybind11;

void bind_gemm(py::module &m) {
    //
    // Enumerate types
    // cutlass/gemm/gemm.h

    py::enum_<cutlass::gemm::GemmUniversalMode>(m, "Mode")
        .value("Gemm", cutlass::gemm::GemmUniversalMode::kGemm, "Ordinary GEMM & GEMM Split-K serial")
        .value("GemmSplitKParallel", cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel, "GEMM Split-K parallel")
        .value("Batched", cutlass::gemm::GemmUniversalMode::kBatched, "Batched GEMM")
        .value("Array", cutlass::gemm::GemmUniversalMode::kArray)
        .value("Invalid", cutlass::gemm::GemmUniversalMode::kInvalid);
    
    /// GemmCoord is a structure that specifies a location within the coordiate space of a GEMM problem
    py::class_<cutlass::gemm::GemmCoord>(m, "GemmCoord")
        .def(py::init<int, int, int>())
        .def("m", py::overload_cast<>(&cutlass::gemm::GemmCoord::m))
        .def("n", py::overload_cast<>(&cutlass::gemm::GemmCoord::n))
        .def("k", py::overload_cast<>(&cutlass::gemm::GemmCoord::k))
        // get tensor coords
        .def("mk", 
            [](const cutlass::gemm::GemmCoord & problem_size) {
                return cutlass::MatrixCoord(problem_size.mk());
            })
        .def("kn", 
            [](const cutlass::gemm::GemmCoord & problem_size) {
                return cutlass::MatrixCoord(problem_size.kn());
            })
        .def("mn", 
            [](const cutlass::gemm::GemmCoord & problem_size) {
                return cutlass::MatrixCoord(problem_size.mn());
            });
    
    py::module_ host_submodule = m.def_submodule("host");
    bind_gemm_host_helper(host_submodule);
}
