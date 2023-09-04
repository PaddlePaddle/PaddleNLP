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
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSE<cutlass::TensorRef<QUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/* \file
   \brief Bind TensorRef and View to python
*/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "types.h"


template<typename T, typename L, typename TF>
void bind_tensor_ref_view(py::module &m, std::string name) {
    py::class_<cutlass::TensorRef<T, L>>(m, ("TensorRef" + name).c_str())
        .def("__init__", [](cutlass::TensorRef<T, L>& tensor_ref, int64_t address, const L& layout_ ) {
            T* ptr = reinterpret_cast< T*>(address);
            new (&tensor_ref) cutlass::TensorRef<T, L>(ptr, layout_);
        })
        .def("data", [](cutlass::TensorRef<T, L>& tensor_ref) {
            T* ptr = tensor_ref.data();
            return int64_t(ptr);
        })
        .def("layout", py::overload_cast<>(&cutlass::TensorRef<T, L>::layout));
    
    m.def("get_tensor_ref", [](int64_t address, TF data, const L& layout_) {
        T* ptr = reinterpret_cast<T*>(address);
        cutlass::TensorRef<T, L> tensor_ref = cutlass::TensorRef<T, L>(ptr, layout_);
        return tensor_ref;
    });
    
    py::class_<cutlass::TensorView<T, L>>(m, ("TensorView" + name).c_str())
        .def(py::init<const cutlass::TensorRef<T, L>&, const typename L::TensorCoord &>());
}


void bind_tensor_refs_and_views(py::module &m) {

    /// float
    bind_tensor_ref_view<float, cutlass::layout::RowMajor, cutlass::float32>(m, "F32RowMajor");
    bind_tensor_ref_view<float, cutlass::layout::ColumnMajor, cutlass::float32>(m, "F32ColumnMajor");
    bind_tensor_ref_view<float, cutlass::layout::TensorNHWC, cutlass::float32>(m, "F32NHWC");

    /// double
    bind_tensor_ref_view<double, cutlass::layout::RowMajor, cutlass::float64>(m, "F64RowMajor");
    bind_tensor_ref_view<double, cutlass::layout::ColumnMajor, cutlass::float64>(m, "F64ColumnMajor");
    bind_tensor_ref_view<double, cutlass::layout::TensorNHWC, cutlass::float64>(m, "F64NHWC");

    // half_t
    bind_tensor_ref_view<cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t>(m, "F16RowMajor");
    bind_tensor_ref_view<cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::half_t>(m, "F16ColumnMajor");
    bind_tensor_ref_view<cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t>(m, "F16NHWC");

    // bfloat16
    bind_tensor_ref_view<cutlass::bfloat16_t, cutlass::layout::RowMajor, cutlass::bfloat16_t>(m, "BF16RowMajor");
    bind_tensor_ref_view<cutlass::bfloat16_t, cutlass::layout::ColumnMajor, cutlass::bfloat16_t>(m, "BF16ColumnMajor");
    bind_tensor_ref_view<cutlass::bfloat16_t, cutlass::layout::TensorNHWC, cutlass::bfloat16_t>(m, "BF16NHWC");

    // int8_t
    bind_tensor_ref_view<int8_t, cutlass::layout::RowMajorInterleaved<32>, cutlass::int8>(m, "S8RowMajorInterleaved32");
    bind_tensor_ref_view<int8_t, cutlass::layout::ColumnMajorInterleaved<32>, cutlass::int8>(m, "S8ColumnMajorInterleaved32");
    bind_tensor_ref_view<int8_t, cutlass::layout::RowMajor, cutlass::int8>(m, "S8RowMajor");
    bind_tensor_ref_view<int8_t, cutlass::layout::ColumnMajor, cutlass::int8>(m, "S8ColumnMajor");
    bind_tensor_ref_view<int8_t, cutlass::layout::TensorNHWC, cutlass::int8>(m, "S8NHWC");
    bind_tensor_ref_view<int8_t, cutlass::layout::TensorNCxHWx<32>, cutlass::int8>(m, "S8NC32HW32");
    bind_tensor_ref_view<int8_t, cutlass::layout::TensorCxRSKx<32>, cutlass::int8>(m, "S8C32RSK32");

    // int32_t
    bind_tensor_ref_view<int32_t, cutlass::layout::RowMajor, cutlass::int32>(m, "S32RowMajor");
    bind_tensor_ref_view<int32_t, cutlass::layout::ColumnMajor, cutlass::int32>(m, "S32ColumnMajor");
    bind_tensor_ref_view<int32_t, cutlass::layout::TensorNHWC, cutlass::int32>(m, "S32NHWC");
}
