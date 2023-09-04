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
   \brief Bind threadblock swizzling to python
*/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/conv/threadblock/threadblock_swizzle.h"

#include <cxxabi.h>
#include <cuda_runtime.h>

namespace py = pybind11;

std::string demangle(const char* mangled_name) {
    std::size_t len = 0;
    int status = 0;
    std::unique_ptr<char> ptr(
                __cxxabiv1::__cxa_demangle(mangled_name, nullptr, &len, &status));
    return ptr.get();
}

template<typename T>
void bind_identity_swizzle(py::module & m, std::string name) {
    py::class_<T>(m, name.c_str(),
        R"pbdoc(Threadblock swizzling function for GEMMs)pbdoc")
        .def(py::init<>())
        .def("get_tiled_shape",
            py::overload_cast<cutlass::gemm::GemmCoord, cutlass::gemm::GemmCoord, int>(
                &T::get_tiled_shape, py::const_
            ), py::arg("problem_size"), py::arg("tile_size"), py::arg("split_k_slices"),
            R"pbdoc(Returns the shape of the problem in units of logical tiles
            
            :param problem_size: gemm(M, N, K)
            :type problem_size: :class:`cutlass.gemm.GemmCoord`
            )pbdoc")
        .def("get_tiled_shape",
            py::overload_cast<cutlass::conv::Operator, const cutlass::conv::Conv2dProblemSize&, cutlass::gemm::GemmCoord, int>(
                &T::get_tiled_shape, py::const_
            ), py::arg("conv_operator"), py::arg("problem_size"), py::arg("tile_size"), py::arg("split_k_slices"),
            R"pbdoc(Returns the shape of the problem in units of logical tiles
            
            :param problem_size: Implicit gemm problem size conv_operator(NPQK, NHWC, KRSC)
            :type problem_size: :class:`cutlass.gemm.GemmCoord`)
            )pbdoc")
        .def("get_tiled_shape",
            py::overload_cast<cutlass::conv::Operator, const cutlass::conv::Conv3dProblemSize&, cutlass::gemm::GemmCoord, int>(
                &T::get_tiled_shape, py::const_
            ), py::arg("conv_operator"), py::arg("problem_size"), py::arg("tile_size"), py::arg("split_k_slices"),
            R"pbdoc(Returns the shape of the problem in units of logical tiles
            
            :param problem_size: Implicit gemm problem size conv_operator(NZPQK, NDHWC, KTRSC)
            :type problem_size: :class:`cutlass.gemm.GemmCoord`)
            )pbdoc")
        .def("get_grid_shape", &T::get_grid_shape,
            py::arg("tiled_shape"), 
            R"pbdoc(Computes CUDA grid dimensions given a size in units of logical tiles)pbdoc")
        .def("tag", [](const T & swizzle){
            return demangle(typeid(T).name());
        }, R"pbdoc(Returns the c++ name of the swizzling for code emittion)pbdoc");
}

template<typename T>
void bind_swizzle(py::module & m, std::string name, std::string doc) {
    py::class_<T>(m, name.c_str(), doc.c_str())
        .def(py::init<>())
        .def("get_tiled_shape",
            py::overload_cast<cutlass::gemm::GemmCoord, cutlass::gemm::GemmCoord, int>(
                &T::get_tiled_shape, py::const_
            ), py::arg("problem_size"), py::arg("tile_size"), py::arg("split_k_slices"),
            R"pbdoc(Returns the shape of the problem in units of logical tiles
            
            :param problem_size: gemm(M, N, K)
            :type problem_size: :class:`cutlass.gemm.GemmCoord`
            )pbdoc")
        .def("get_grid_shape", &T::get_grid_shape,
            py::arg("tiled_shape"), 
            R"pbdoc(Computes CUDA grid dimensions given a size in units of logical tiles)pbdoc")
        .def("tag", [](const T & swizzle){
            return demangle(typeid(T).name());
        }, R"pbdoc(Returns the c++ name of the swizzling for code emittion)pbdoc");
}

template<typename T>
void bind_dgrad_swizzle(py::module & m, std::string name) {
    py::class_<T>(m, name.c_str(),
        R"pbdoc(Threadblock swizzling function for strided dgrad convolution)pbdoc")
        .def(py::init<>())
        .def("get_tiled_shape",
            py::overload_cast<cutlass::conv::Operator, const cutlass::conv::Conv2dProblemSize&, cutlass::gemm::GemmCoord, int>(
                &T::get_tiled_shape, py::const_
            ), py::arg("conv_operator"), py::arg("problem_size"), py::arg("tile_size"), py::arg("split_k_slices"),
            R"pbdoc(Returns the shape of the problem in units of logical tiles
            
            :param problem_size: Implicit gemm problem size conv_operator(NPQK, NHWC, KRSC)
            :type problem_size: :class:`cutlass.gemm.GemmCoord`)
            )pbdoc")
        .def("get_grid_shape", [](const T & swizzle, cutlass::gemm::GemmCoord tiled_shape) {
            return dim3(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
        }, py::arg("tiled_shape"), 
            R"pbdoc(Computes CUDA grid dimensions given a size in units of logical tiles)pbdoc")
        .def("tag", [](const T & swizzle){
            return demangle(typeid(T).name());
        }, R"pbdoc(Returns the c++ name of the swizzling for code emittion)pbdoc");
}

void bind_threadblock_swizzle(py::module &m) {

    py::class_<dim3>(m, "dim3",
        R"pbdoc(A int3 type xyz contains three integers)pbdoc")
        .def(py::init<int, int, int>(),
            py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &dim3::x, R"pbdoc(get value x)pbdoc")
        .def_readwrite("y", &dim3::y, R"pbdoc(get value y)pbdoc")
        .def_readwrite("z", &dim3::z, R"pbdoc(get value z)pbdoc");

    bind_identity_swizzle<cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>>(m, "IdentitySwizzle1");
    bind_identity_swizzle<cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>>(m, "IdentitySwizzle2");
    bind_identity_swizzle<cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>(m, "IdentitySwizzle4");
    bind_identity_swizzle<cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>>(m, "IdentitySwizzle8");

    bind_swizzle<cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle>(m, "HorizontalSwizzle",  R"pbdoc(Threadblock swizzling function for GEMMs)pbdoc");
    bind_swizzle<cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle>(m, "BatchedIdentitySwizzle",  R"pbdoc(Threadblock swizzling function for batched GEMMs)pbdoc");

    bind_dgrad_swizzle<cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>>(m, "StridedDgradIdentitySwizzle1");
    bind_dgrad_swizzle<cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<4>>(m, "StridedDgradIdentitySwizzle4");
    bind_dgrad_swizzle<cutlass::conv::threadblock::StridedDgradHorizontalThreadblockSwizzle>(m, "StridedDgradHorizontalSwizzle");
}
