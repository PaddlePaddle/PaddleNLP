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
   \brief Bind Convolution host test helpers to python
*/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include "unit/conv/device/cache_testbed_output.h"


#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/host/tensor_compare.h"

namespace py = pybind11;


template<typename Ta, typename La, typename Tb, typename Lb, typename Tc, typename Lc, typename Tacc, typename Te>
void bind_conv2d_host(py::module &m) {
    m.def("conv2d", \
        &cutlass::reference::host::Conv2d< \
            Ta, La, Tb, Lb, Tc, Lc, Te, Tacc>);
    
    m.def("CreateCachedConv2dTestKey", &test::conv::device::CreateCachedConv2dTestKey<Ta, La, Tb, Lb, Tc, Lc, Tacc, Te>);
}

template<typename Ta, typename La, typename Tb, typename Lb, typename Tc, typename Lc, typename Tacc, typename Te>
void bind_conv2d_host_sat(py::module &m) {
    m.def("conv2d", \
        &cutlass::reference::host::Conv2d< \
            Ta, La, Tb, Lb, Tc, Lc, Te, Tacc, cutlass::NumericConverterClamp<Tc, Te>>);
    
    m.def("CreateCachedConv2dTestKey", &test::conv::device::CreateCachedConv2dTestKey<Ta, La, Tb, Lb, Tc, Lc, Tacc, Te>);
}

template<typename Ta, typename Tb, typename Tc, typename Tacc, typename Te>
void bind_conv2d_host_nhwc(py::module &m) {
    bind_conv2d_host<
        Ta, cutlass::layout::TensorNHWC, 
        Tb, cutlass::layout::TensorNHWC, 
        Tc, cutlass::layout::TensorNHWC, 
        Tacc, Te>(m);
}

template<typename Ta, typename Tb, typename Tc, typename Tacc, typename Te>
void bind_conv2d_host_nc32hw32(py::module &m) {
    bind_conv2d_host_sat<
        Ta, cutlass::layout::TensorNCxHWx<32>,
        Tb, cutlass::layout::TensorCxRSKx<32>,
        Tc, cutlass::layout::TensorNCxHWx<32>,
        Tacc, Te>(m);
}


template<typename T, typename Layout>
void bind_tensor_equals(py::module &m) {
    m.def("equals", py::overload_cast<
        const cutlass::TensorView<T, Layout>&, const cutlass::TensorView<T, Layout>&>(
            &cutlass::reference::host::TensorEquals<T, Layout>
        ));
}

#define BIND_TENSOR_HASH(Element, Layout) { \
    m.def("TensorHash", &test::conv::device::TensorHash<Element, Layout>, py::arg("view"), py::arg("hash") = test::conv::device::CRC32(), py::arg("crc")=uint32_t()); \
}

void bind_conv_host_references(py::module &m) {
    //
    // Conv2d reference on host
    // tools/util/include/cutlass/util/reference/host/convolution.h

    /// double
    bind_conv2d_host_nhwc<double, double, double, double, double>(m);
    /// float
    bind_conv2d_host_nhwc<float, float, float, float, float>(m);
    /// half
    bind_conv2d_host_nhwc<cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t>(m);
    bind_conv2d_host_nhwc<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, cutlass::half_t>(m);
    bind_conv2d_host_nhwc<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, float>(m);
    bind_conv2d_host_nhwc<cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t, float>(m);
    bind_conv2d_host_nhwc<cutlass::half_t, cutlass::half_t, float, cutlass::half_t, cutlass::half_t>(m);
    bind_conv2d_host_nhwc<cutlass::half_t, cutlass::half_t, float, float, cutlass::half_t>(m);
    bind_conv2d_host_nhwc<cutlass::half_t, cutlass::half_t, float, float, float>(m);
    bind_conv2d_host_nhwc<cutlass::half_t, cutlass::half_t, float, cutlass::half_t, float>(m);
    /// bfloat16
    bind_conv2d_host_nhwc<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, cutlass::bfloat16_t>(m);
    bind_conv2d_host_nhwc<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, float>(m);
    bind_conv2d_host_nhwc<cutlass::bfloat16_t, cutlass::bfloat16_t, float, float, cutlass::bfloat16_t>(m);
    bind_conv2d_host_nhwc<cutlass::bfloat16_t, cutlass::bfloat16_t, float, float, float>(m);
    /// s8
    bind_conv2d_host_nhwc<int8_t, int8_t, int8_t, int32_t, int32_t>(m);
    bind_conv2d_host_nhwc<int8_t, int8_t, int8_t, int32_t, int8_t>(m);
    bind_conv2d_host_nhwc<int8_t, int8_t, int32_t, int32_t, int32_t>(m);
    bind_conv2d_host_nhwc<int8_t, int8_t, int32_t, int32_t, int8_t>(m);
    bind_conv2d_host_nhwc<int8_t, int8_t, int8_t, int32_t, float>(m);
    bind_conv2d_host_nhwc<int8_t, int8_t, int8_t, int32_t, float>(m);
    bind_conv2d_host_nhwc<int8_t, int8_t, int32_t, int32_t, float>(m);
    bind_conv2d_host_nhwc<int8_t, int8_t, int32_t, int32_t, float>(m);

    bind_conv2d_host_nc32hw32<int8_t, int8_t, int8_t, int32_t, int32_t>(m);
    bind_conv2d_host_nc32hw32<int8_t, int8_t, int8_t, int32_t, int8_t>(m);
    bind_conv2d_host_nc32hw32<int8_t, int8_t, int32_t, int32_t, int32_t>(m);
    bind_conv2d_host_nc32hw32<int8_t, int8_t, int32_t, int32_t, int8_t>(m);
    bind_conv2d_host_nc32hw32<int8_t, int8_t, int8_t, int32_t, float>(m);
    bind_conv2d_host_nc32hw32<int8_t, int8_t, int8_t, int32_t, float>(m);
    bind_conv2d_host_nc32hw32<int8_t, int8_t, int32_t, int32_t, float>(m);
    bind_conv2d_host_nc32hw32<int8_t, int8_t, int32_t, int32_t, float>(m);

    //
    // Compare whether two tensors are equal
    //
    /// double
    bind_tensor_equals<double, cutlass::layout::TensorNHWC>(m);
    /// float
    bind_tensor_equals<float, cutlass::layout::TensorNHWC>(m);
    /// half
    bind_tensor_equals<cutlass::half_t, cutlass::layout::TensorNHWC>(m);
    /// bfloat16
    bind_tensor_equals<cutlass::bfloat16_t, cutlass::layout::TensorNHWC>(m);
    /// s32
    bind_tensor_equals<int32_t, cutlass::layout::TensorNHWC>(m);
    bind_tensor_equals<int32_t, cutlass::layout::TensorNCxHWx<32>>(m);
    /// s8
    bind_tensor_equals<int8_t, cutlass::layout::TensorNHWC>(m);
    bind_tensor_equals<int8_t, cutlass::layout::TensorNCxHWx<32>>(m);

    /// Cache
    py::class_<test::conv::device::CachedTestKey>(m, "CachedTestKey")
        .def(py::init<>())
        .def(py::init<std::string, std::string, std::string, uint32_t, uint32_t, uint32_t>());
    
    py::class_<test::conv::device::CachedTestResult>(m, "CachedTestResult")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def_readwrite("D", &test::conv::device::CachedTestResult::D);
    
    py::class_<test::conv::device::CachedTestResultListing>(m, "CachedTestResultListing")
        .def(py::init<const std::string &>())
        .def("find", &test::conv::device::CachedTestResultListing::find)
        .def("append", &test::conv::device::CachedTestResultListing::append)
        .def("write", &test::conv::device::CachedTestResultListing::write);
    
    py::class_<test::conv::device::CRC32>(m, "CRC32")
        .def(py::init<>());
    
    BIND_TENSOR_HASH(double, cutlass::layout::TensorNHWC)
    BIND_TENSOR_HASH(float, cutlass::layout::TensorNHWC);
    BIND_TENSOR_HASH(cutlass::half_t, cutlass::layout::TensorNHWC);
    BIND_TENSOR_HASH(cutlass::bfloat16_t, cutlass::layout::TensorNHWC);
    BIND_TENSOR_HASH(int32_t, cutlass::layout::TensorNHWC);
    BIND_TENSOR_HASH(int8_t, cutlass::layout::TensorNCxHWx<32>);
}
