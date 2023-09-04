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
   \brief Bind opcode classes to python
*/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "cutlass/arch/mma.h"

namespace py = pybind11;

namespace cutlass {
enum class OpcodeClass {
    kSimt, kTensorOp, kWmmaTensorOp, kSparseTensorOp
};
}

void bind_opcode(py::module &m) {
    py::enum_<cutlass::OpcodeClass>(m, "OpClass",
        R"pbdoc(classification of math operators)pbdoc")
        .value("Simt", cutlass::OpcodeClass::kSimt, 
            R"pbdoc(Tag classifying math operators as thread-level operations)pbdoc")
        .value("TensorOp", cutlass::OpcodeClass::kTensorOp, 
            R"pbdoc(Tag classifing operators as Tensor Core operations)pbdoc")
        .value("WmmaTensorOp", cutlass::OpcodeClass::kWmmaTensorOp, 
            R"pbdoc(Tag classifing operators as WMMA Tensor Core operations)pbdoc")
        .value("SparseTensorOp", cutlass::OpcodeClass::kSparseTensorOp, 
            R"pbdoc(Tag classifing operators as sparseTensor Core operations)pbdoc");
}
