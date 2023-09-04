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
   \brief Bind gemm test host functions to python
*/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "cutlass/cutlass.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/host_reorder.h"

#include "cutlass/functional.h"

namespace py = pybind11;


template<
    typename ElementA, typename LayoutA,
    typename ElementB, typename LayoutB,
    typename ElementC, typename LayoutC,
    typename AccumulatorType, typename ComputeType, 
    typename InnerProductOp>
void bind_host_gemm_saturate(py::module &m) {
    m.def("gemm_saturate", py::overload_cast<
        cutlass::gemm::GemmCoord, ComputeType,
        cutlass::TensorRef<ElementA, LayoutA>,
        cutlass::TensorRef<ElementB, LayoutB>,
        ComputeType,
        cutlass::TensorRef<ElementC, LayoutC>,
        cutlass::TensorRef<ElementC, LayoutC>,
        AccumulatorType>(
            &cutlass::reference::host::compute_gemm<
                        ElementA, LayoutA,
                        ElementB, LayoutB,
                        ElementC, LayoutC,
                        ComputeType,
                        AccumulatorType,
                        InnerProductOp, 
                        cutlass::NumericConverterClamp<ElementC, AccumulatorType>>
                        ));
}

template<
    typename ElementA, typename LayoutA,
    typename ElementB, typename LayoutB,
    typename ElementC, typename LayoutC,
    typename AccumulatorType, typename ComputeType, 
    typename InnerProductOp>
void bind_host_gemm(py::module &m) {
    m.def("gemm", py::overload_cast<
        cutlass::gemm::GemmCoord, ComputeType,
        cutlass::TensorRef<ElementA, LayoutA>,
        cutlass::TensorRef<ElementB, LayoutB>,
        ComputeType,
        cutlass::TensorRef<ElementC, LayoutC>,
        cutlass::TensorRef<ElementC, LayoutC>,
        AccumulatorType>(
            &cutlass::reference::host::compute_gemm<
                        ElementA, LayoutA,
                        ElementB, LayoutB,
                        ElementC, LayoutC,
                        ComputeType,
                        AccumulatorType,
                        InnerProductOp, 
                        cutlass::NumericConverter<ElementC, AccumulatorType>>
                        ));
}


template<
    typename ElementA, typename ElementB, typename ElementC,
    typename AccumulatorType, typename ComputeType>
void bind_host_gemm_multiply_add(py::module &m) {
    bind_host_gemm<
        ElementA, cutlass::layout::RowMajor, 
        ElementB, cutlass::layout::RowMajor, 
        ElementC, cutlass::layout::RowMajor, 
        ComputeType, AccumulatorType,
        cutlass::multiply_add<AccumulatorType>>(m);
    
    bind_host_gemm<
        ElementA, cutlass::layout::ColumnMajor, 
        ElementB, cutlass::layout::RowMajor, 
        ElementC, cutlass::layout::RowMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm<
        ElementA, cutlass::layout::RowMajor, 
        ElementB, cutlass::layout::ColumnMajor, 
        ElementC, cutlass::layout::RowMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm<
        ElementA, cutlass::layout::RowMajor, 
        ElementB, cutlass::layout::RowMajor, 
        ElementC, cutlass::layout::ColumnMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm<
        ElementA, cutlass::layout::RowMajor, 
        ElementB, cutlass::layout::ColumnMajor, 
        ElementC, cutlass::layout::ColumnMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm<
        ElementA, cutlass::layout::ColumnMajor, 
        ElementB, cutlass::layout::RowMajor, 
        ElementC, cutlass::layout::ColumnMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm<
        ElementA, cutlass::layout::ColumnMajor, 
        ElementB, cutlass::layout::ColumnMajor, 
        ElementC, cutlass::layout::RowMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm<
        ElementA, cutlass::layout::ColumnMajor, 
        ElementB, cutlass::layout::ColumnMajor, 
        ElementC, cutlass::layout::ColumnMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);
}

template<
    typename ElementA, typename ElementB, typename ElementC,
    typename AccumulatorType, typename ComputeType>
void bind_host_gemm_multiply_add_saturate(py::module &m) {
    bind_host_gemm_saturate<
        ElementA, cutlass::layout::RowMajor, 
        ElementB, cutlass::layout::RowMajor, 
        ElementC, cutlass::layout::RowMajor, 
        ComputeType, AccumulatorType,
        cutlass::multiply_add<AccumulatorType>>(m);
    
    bind_host_gemm_saturate<
        ElementA, cutlass::layout::ColumnMajor, 
        ElementB, cutlass::layout::RowMajor, 
        ElementC, cutlass::layout::RowMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm_saturate<
        ElementA, cutlass::layout::RowMajor, 
        ElementB, cutlass::layout::ColumnMajor, 
        ElementC, cutlass::layout::RowMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm_saturate<
        ElementA, cutlass::layout::RowMajor, 
        ElementB, cutlass::layout::RowMajor, 
        ElementC, cutlass::layout::ColumnMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm_saturate<
        ElementA, cutlass::layout::RowMajor, 
        ElementB, cutlass::layout::ColumnMajor, 
        ElementC, cutlass::layout::ColumnMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm_saturate<
        ElementA, cutlass::layout::ColumnMajor, 
        ElementB, cutlass::layout::RowMajor, 
        ElementC, cutlass::layout::ColumnMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm_saturate<
        ElementA, cutlass::layout::ColumnMajor, 
        ElementB, cutlass::layout::ColumnMajor, 
        ElementC, cutlass::layout::RowMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm_saturate<
        ElementA, cutlass::layout::ColumnMajor, 
        ElementB, cutlass::layout::ColumnMajor, 
        ElementC, cutlass::layout::ColumnMajor, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);
}


template<
    typename ElementA, typename ElementB, typename ElementC,
    typename AccumulatorType, typename ComputeType>
void bind_host_gemm_multiply_add_interleaved(py::module &m) {
    bind_host_gemm<
        ElementA, cutlass::layout::RowMajorInterleaved<32>, 
        ElementB, cutlass::layout::RowMajorInterleaved<32>, 
        ElementC, cutlass::layout::RowMajorInterleaved<32>, 
        ComputeType, AccumulatorType,
        cutlass::multiply_add<AccumulatorType>>(m);
    
    bind_host_gemm<
        ElementA, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementB, cutlass::layout::RowMajorInterleaved<32>, 
        ElementC, cutlass::layout::RowMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm<
        ElementA, cutlass::layout::RowMajorInterleaved<32>, 
        ElementB, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementC, cutlass::layout::RowMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm<
        ElementA, cutlass::layout::RowMajorInterleaved<32>, 
        ElementB, cutlass::layout::RowMajorInterleaved<32>, 
        ElementC, cutlass::layout::ColumnMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm<
        ElementA, cutlass::layout::RowMajorInterleaved<32>, 
        ElementB, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementC, cutlass::layout::ColumnMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm<
        ElementA, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementB, cutlass::layout::RowMajorInterleaved<32>, 
        ElementC, cutlass::layout::ColumnMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm<
        ElementA, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementB, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementC, cutlass::layout::RowMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm<
        ElementA, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementB, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementC, cutlass::layout::ColumnMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);
}

template<
    typename ElementA, typename ElementB, typename ElementC,
    typename AccumulatorType, typename ComputeType>
void bind_host_gemm_multiply_add_saturate_interleaved(py::module &m) {
    bind_host_gemm_saturate<
        ElementA, cutlass::layout::RowMajorInterleaved<32>, 
        ElementB, cutlass::layout::RowMajorInterleaved<32>, 
        ElementC, cutlass::layout::RowMajorInterleaved<32>, 
        ComputeType, AccumulatorType,
        cutlass::multiply_add<AccumulatorType>>(m);
    
    bind_host_gemm_saturate<
        ElementA, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementB, cutlass::layout::RowMajorInterleaved<32>, 
        ElementC, cutlass::layout::RowMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm_saturate<
        ElementA, cutlass::layout::RowMajorInterleaved<32>, 
        ElementB, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementC, cutlass::layout::RowMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm_saturate<
        ElementA, cutlass::layout::RowMajorInterleaved<32>, 
        ElementB, cutlass::layout::RowMajorInterleaved<32>, 
        ElementC, cutlass::layout::ColumnMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm_saturate<
        ElementA, cutlass::layout::RowMajorInterleaved<32>, 
        ElementB, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementC, cutlass::layout::ColumnMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm_saturate<
        ElementA, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementB, cutlass::layout::RowMajorInterleaved<32>, 
        ElementC, cutlass::layout::ColumnMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm_saturate<
        ElementA, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementB, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementC, cutlass::layout::RowMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);

    bind_host_gemm_saturate<
        ElementA, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementB, cutlass::layout::ColumnMajorInterleaved<32>, 
        ElementC, cutlass::layout::ColumnMajorInterleaved<32>, 
        AccumulatorType, ComputeType, 
        cutlass::multiply_add<AccumulatorType>>(m);
}

#define BIND_TENSOR_EQUAL(Element, Layout) { \
    m.def("equals", py::overload_cast< \
        const cutlass::TensorView<Element, Layout>&, const cutlass::TensorView<Element, Layout>&>( \
        &cutlass::reference::host::TensorEquals<Element, Layout>)); \
}

void bind_gemm_host_reference(py::module &m) {

    /// double
    bind_host_gemm_multiply_add<double, double, double, double, double>(m);
    /// float
    bind_host_gemm_multiply_add<float, float, float, float, float>(m);
    /// half_t
    bind_host_gemm_multiply_add<cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t>(m);
    bind_host_gemm_multiply_add<cutlass::half_t, cutlass::half_t, cutlass::half_t, float, float>(m);
    bind_host_gemm_multiply_add<cutlass::half_t, cutlass::half_t, float, cutlass::half_t, cutlass::half_t>(m);
    bind_host_gemm_multiply_add<cutlass::half_t, cutlass::half_t, float, float, float>(m);
    /// bfloat16
    bind_host_gemm_multiply_add<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float, float>(m);
    bind_host_gemm_multiply_add<cutlass::bfloat16_t, cutlass::bfloat16_t, float, float, float>(m);

    /// s8
    bind_host_gemm_multiply_add<int8_t, int8_t, int8_t, int32_t, int32_t>(m);
    bind_host_gemm_multiply_add<int8_t, int8_t, int8_t, int32_t, int8_t>(m);
    bind_host_gemm_multiply_add<int8_t, int8_t, int32_t, int32_t, int32_t>(m);
    bind_host_gemm_multiply_add<int8_t, int8_t, int32_t, int32_t, int8_t>(m);
    bind_host_gemm_multiply_add<int8_t, int8_t, int8_t, int32_t, float>(m);
    bind_host_gemm_multiply_add<int8_t, int8_t, int8_t, int32_t, float>(m);
    bind_host_gemm_multiply_add<int8_t, int8_t, int32_t, int32_t, float>(m);
    bind_host_gemm_multiply_add<int8_t, int8_t, int32_t, int32_t, float>(m);

    bind_host_gemm_multiply_add_interleaved<int8_t, int8_t, int8_t, int32_t, int32_t>(m);
    bind_host_gemm_multiply_add_interleaved<int8_t, int8_t, int8_t, int32_t, int8_t>(m);
    bind_host_gemm_multiply_add_interleaved<int8_t, int8_t, int32_t, int32_t, int32_t>(m);
    bind_host_gemm_multiply_add_interleaved<int8_t, int8_t, int32_t, int32_t, int8_t>(m);
    bind_host_gemm_multiply_add_interleaved<int8_t, int8_t, int8_t, int32_t, float>(m);
    bind_host_gemm_multiply_add_interleaved<int8_t, int8_t, int8_t, int32_t, float>(m);
    bind_host_gemm_multiply_add_interleaved<int8_t, int8_t, int32_t, int32_t, float>(m);
    bind_host_gemm_multiply_add_interleaved<int8_t, int8_t, int32_t, int32_t, float>(m);

    bind_host_gemm_multiply_add_saturate<int8_t, int8_t, int8_t, int32_t, int32_t>(m);
    bind_host_gemm_multiply_add_saturate<int8_t, int8_t, int8_t, int32_t, int8_t>(m);
    bind_host_gemm_multiply_add_saturate<int8_t, int8_t, int32_t, int32_t, int32_t>(m);
    bind_host_gemm_multiply_add_saturate<int8_t, int8_t, int32_t, int32_t, int8_t>(m);
    bind_host_gemm_multiply_add_saturate<int8_t, int8_t, int8_t, int32_t, float>(m);
    bind_host_gemm_multiply_add_saturate<int8_t, int8_t, int8_t, int32_t, float>(m);
    bind_host_gemm_multiply_add_saturate<int8_t, int8_t, int32_t, int32_t, float>(m);
    bind_host_gemm_multiply_add_saturate<int8_t, int8_t, int32_t, int32_t, float>(m);

    bind_host_gemm_multiply_add_saturate_interleaved<int8_t, int8_t, int8_t, int32_t, int32_t>(m);
    bind_host_gemm_multiply_add_saturate_interleaved<int8_t, int8_t, int8_t, int32_t, int8_t>(m);
    bind_host_gemm_multiply_add_saturate_interleaved<int8_t, int8_t, int32_t, int32_t, int32_t>(m);
    bind_host_gemm_multiply_add_saturate_interleaved<int8_t, int8_t, int32_t, int32_t, int8_t>(m);
    bind_host_gemm_multiply_add_saturate_interleaved<int8_t, int8_t, int8_t, int32_t, float>(m);
    bind_host_gemm_multiply_add_saturate_interleaved<int8_t, int8_t, int8_t, int32_t, float>(m);
    bind_host_gemm_multiply_add_saturate_interleaved<int8_t, int8_t, int32_t, int32_t, float>(m);
    bind_host_gemm_multiply_add_saturate_interleaved<int8_t, int8_t, int32_t, int32_t, float>(m);

    // float
    BIND_TENSOR_EQUAL(float, cutlass::layout::RowMajor);
    BIND_TENSOR_EQUAL(float, cutlass::layout::ColumnMajor);

    // double
    BIND_TENSOR_EQUAL(double, cutlass::layout::RowMajor);
    BIND_TENSOR_EQUAL(double, cutlass::layout::ColumnMajor);

    // half_t
    BIND_TENSOR_EQUAL(cutlass::half_t, cutlass::layout::RowMajor);
    BIND_TENSOR_EQUAL(cutlass::half_t, cutlass::layout::ColumnMajor);

    // bfloat16
    BIND_TENSOR_EQUAL(cutlass::bfloat16_t, cutlass::layout::RowMajor);
    BIND_TENSOR_EQUAL(cutlass::bfloat16_t, cutlass::layout::ColumnMajor);

    // int32_t
    BIND_TENSOR_EQUAL(int32_t, cutlass::layout::RowMajor);
    BIND_TENSOR_EQUAL(int32_t, cutlass::layout::ColumnMajor);

    // int8_t
    BIND_TENSOR_EQUAL(int8_t, cutlass::layout::RowMajor);
    BIND_TENSOR_EQUAL(int8_t, cutlass::layout::ColumnMajor);
    BIND_TENSOR_EQUAL(int8_t, cutlass::layout::RowMajorInterleaved<32>);
    BIND_TENSOR_EQUAL(int8_t, cutlass::layout::ColumnMajorInterleaved<32>);
    

}
