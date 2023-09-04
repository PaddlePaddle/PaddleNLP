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
   \brief binding CUTLASS C++ APIs to Python
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "builtin_types.h"
#include "device_launch_parameters.h"
#include "stddef.h"
#include "cutlass/cutlass.h"

#include "include/conv/convolution.h"
#include "include/gemm/gemm.h"
#include "include/types.h"
#include "include/layout/layout.h"
#include "include/tensor_coord.h"
#include "include/arch.h"
#include "include/tensor_ref_view.h"
#include "include/swizzling.h"
#include "test/conv/convolution.h"
#include "test/gemm/gemm.h"


// Data Types
#include "library.h"

// compiler
#include "compiler.h"


namespace py = pybind11;


PYBIND11_MODULE(cutlass, m) {

    // module doc
    m.doc() = "cutlass C++ binding";

    //
    // Bind data type
    //
    bind_cutlass_types(m);

    //
    // Bind layout
    //
    bind_layout(m);

    //
    // Bind tensor coord
    //
    bind_tensor_coord(m);

    //
    // Bind tensor ref
    //
    bind_tensor_refs_and_views(m);

    //
    // Bind opcode
    //
    bind_opcode(m);

    //
    // Bind convolution
    //
    py::module_ conv_submodule = m.def_submodule("conv");
    bind_convolution(conv_submodule);

    //
    // Bind gemm
    //
    py::module_ gemm_submodule = m.def_submodule("gemm");
    bind_gemm(gemm_submodule);

    //
    // Bind swizzling
    //
    bind_threadblock_swizzle(m);


    //
    // Bind test units
    //
    py::module_ test = m.def_submodule("test");
    py::module_ test_conv = test.def_submodule("conv");
    bind_convolution_test(test_conv);

    py::module_ test_gemm = test.def_submodule("gemm");
    bind_gemm_test(test_gemm);

    // data types
    py::enum_<cutlass::DataType>(m, "dtype")
        .value("b1", cutlass::DataType::kB1)
        .value("u2", cutlass::DataType::kU2)
        .value("u4", cutlass::DataType::kU4)
        .value("u8", cutlass::DataType::kU8)
        .value("u16", cutlass::DataType::kU16)
        .value("u32", cutlass::DataType::kU32)
        .value("u64", cutlass::DataType::kU64)
        .value("s2", cutlass::DataType::kS2)
        .value("s4", cutlass::DataType::kS4)
        .value("s16", cutlass::DataType::kS16)
        .value("s64", cutlass::DataType::kS64)
        .value("cf16", cutlass::DataType::kCF16)
        .value("cbf16", cutlass::DataType::kCBF16)
        .value("cf32", cutlass::DataType::kCF32)
        .value("ctf32", cutlass::DataType::kCTF32)
        .value("cf64", cutlass::DataType::kCF64)
        .value("cs2", cutlass::DataType::kCS2)
        .value("cs4", cutlass::DataType::kCS4)
        .value("cs8", cutlass::DataType::kCS8)
        .value("cs16", cutlass::DataType::kCS16)
        .value("cs32", cutlass::DataType::kCS32)
        .value("cs64", cutlass::DataType::kCS64)
        .value("cu2", cutlass::DataType::kCU2)
        .value("cu4", cutlass::DataType::kCU4)
        .value("cu8", cutlass::DataType::kCU8)
        .value("cu16", cutlass::DataType::kCU16)
        .value("cu32", cutlass::DataType::kCU32)
        .value("cu64", cutlass::DataType::kCU64)
        .value("invalid", cutlass::DataType::kInvalid);
    
    // layout types
    py::enum_<cutlass::LayoutType>(m, "layout")
        .value("ColumnMajorInterleaved2", cutlass::LayoutType::kColumnMajorInterleaved2)
        .value("RowMajorInterleaved2", cutlass::LayoutType::kRowMajorInterleaved2)
        .value("ColumnMajorInterleaved64", cutlass::LayoutType::kColumnMajorInterleaved64)
        .value("RowMajorInterleaved64", cutlass::LayoutType::kRowMajorInterleaved64)
        .value("TensorNDHWC", cutlass::LayoutType::kTensorNDHWC)
        .value("TensorNCHW", cutlass::LayoutType::kTensorNCHW)
        .value("TensorNGHWC", cutlass::LayoutType::kTensorNGHWC)
        .value("TensorNC64HW64", cutlass::LayoutType::kTensorNC64HW64)
        .value("TensorC64RSK64", cutlass::LayoutType::kTensorC64RSK64);
    
    // transform types
    py::enum_<cutlass::ComplexTransform>(m, "complex_transform")
        .value("none", cutlass::ComplexTransform::kNone)
        .value("conj", cutlass::ComplexTransform::kConjugate);

    //
    // Compiler
    //
    py::class_<cutlass::CompileCache>(m, "CompileCache")
        .def(py::init<>())
        .def("at", &cutlass::CompileCache::at)
        .def("insert", &cutlass::CompileCache::insert)
        .def("size", &cutlass::CompileCache::size)
        .def("clear", &cutlass::CompileCache::clear);

}
