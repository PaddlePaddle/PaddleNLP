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
   \brief Bind CUTLASS types to python
*/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "cutlass/half.h"


namespace py = pybind11;

namespace cutlass {

/// IEEE 32-bit signed integer
struct alignas(1) int8 {
    int8_t storage;
    explicit int8(int x) {
        storage = int8_t(x);
    }
    explicit int8(float x) {
        storage = int8_t(x);
    }

    int8_t c_value(){return storage;}
};

/// IEEE 32-bit signed integer
struct alignas(4) int32 {
    int storage;
    explicit int32(int x) {
        storage = x;
    }
    explicit int32(float x) {
        storage = int(x);
    }

    int c_value(){return storage;}
};
/// IEEE single-precision floating-point type
struct alignas(4) float32 {
    float storage;
    explicit float32(float x) {
        storage = x;
    }
    explicit float32(int x) {
        storage = float(x);
    }
    float c_value(){return storage;}
};
/// IEEE double-precision floating-point type
struct alignas(4) float64 {
    double storage;
    explicit float64(float x) {
        storage = double(x);
    }
    explicit float64(int x) {
        storage = double(x);
    }
    double c_value(){return storage;}
};
}

void bind_cutlass_types(py::module &m) {

    // s8
    py::class_<cutlass::int8>(m, "int8")
        .def(py::init<float>())
        .def(py::init<int>())
        .def_readwrite("storage", &cutlass::int8::storage)
        .def("value", &cutlass::int8::c_value);

    // s32
    py::class_<cutlass::int32>(m, "int32")
        .def(py::init<float>())
        .def(py::init<int>())
        .def_readwrite("storage", &cutlass::int32::storage)
        .def("value", &cutlass::int32::c_value);

    // f16
    py::class_<cutlass::half_t>(m, "float16")
        .def(py::init<float>())
        .def(py::init<double>())
        .def(py::init<int>())
        .def(py::init<unsigned>())
        .def_readwrite("storage", &cutlass::half_t::storage)
        .def("value", [](const cutlass::half_t& value) {return value;});
    
    // bf16
    py::class_<cutlass::bfloat16_t>(m, "bfloat16")
        .def(py::init<float>())
        .def(py::init<int>())
        .def_readwrite("storage", &cutlass::bfloat16_t::storage)
        .def("value", [](const cutlass::bfloat16_t& value) {return value;});

    // f32
    py::class_<cutlass::float32>(m, "float32")
        .def(py::init<float>())
        .def(py::init<int>())
        .def_readwrite("storage", &cutlass::float32::storage)
        .def("value", &cutlass::float32::c_value);

    // tf32
    py::class_<cutlass::tfloat32_t>(m, "tfloat32")
        .def(py::init<float>())
        .def(py::init<int>())
        .def_readwrite("storage", &cutlass::tfloat32_t::storage)
        .def("value", [](const cutlass::tfloat32_t& value) {return value;});
    
    // f64
    py::class_<cutlass::float64>(m, "float64")
        .def(py::init<float>())
        .def(py::init<int>())
        .def_readwrite("storage", &cutlass::float64::storage)
        .def("value", &cutlass::float64::c_value);
}
