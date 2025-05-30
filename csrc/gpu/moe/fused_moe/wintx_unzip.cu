// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "moe/wintx_unzip_impl_op.h"
#include "helper.h"


template <paddle::DataType T>
void WintxUnzipKernel(const paddle::Tensor& weight,
                      const paddle::Tensor& super_scale,
                      const paddle::Tensor& bs_shift,
                      paddle::Tensor& unzip_deq_weight,
                      const std::string& quant_method) {
    typedef PDTraits<T> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    auto super_scale_data = super_scale.data<data_t>();
    auto bs_shift_data = bs_shift.data<int32_t>();
    auto unzip_deq_weight_data = unzip_deq_weight.data<data_t>();
    using NvType = typename traits_::DataType;
    if(quant_method == "weight_only_int2.5") {
        auto weight_data = weight.data<int16_t>();
        const int num_experts = unzip_deq_weight.shape()[0];
        const int64_t k = unzip_deq_weight.shape()[1];
        const int64_t n  =  unzip_deq_weight.shape()[2];
        const uint16_t* weight_data_ptr = reinterpret_cast<const uint16_t*>(weight_data);
        wintx_unzip_quantization_kernel_Launcher<DataType_>(
            weight_data_ptr, reinterpret_cast<const NvType*>(super_scale_data), 
            bs_shift_data, reinterpret_cast<NvType*>(unzip_deq_weight_data),
            n, k, num_experts,
            64, 7, 0x7, 0x1FFF, 4);
    }
    return;
}

std::vector<paddle::Tensor> WintXUnzip(const paddle::Tensor& weight, 
                                       const paddle::Tensor& super_scale,
                                       const paddle::Tensor& bs_shift,
                                       const std::string& quant_method) {
    std::vector<int64_t> output_shape(weight.shape());
    const int unzip_axis = 1;
    if(quant_method == "weight_only_int2.5") {
        output_shape[unzip_axis] = weight.shape()[unzip_axis] / 10 * 64;
    } else {
        PD_THROW("Unsupported data type for WintxUnzip");
    }
    auto output_tensor = paddle::empty(output_shape, super_scale.dtype());

    switch (super_scale.dtype()) {
        case paddle::DataType::BFLOAT16:
            WintxUnzipKernel<paddle::DataType::BFLOAT16>(weight,
                                                         super_scale,
                                                         bs_shift,
                                                         output_tensor,
                                                         quant_method);
            break;
        case paddle::DataType::FLOAT16:
            WintxUnzipKernel<paddle::DataType::FLOAT16>(weight,
                                                        super_scale,
                                                        bs_shift,
                                                        output_tensor,
                                                        quant_method);
            break;
        default:
            PD_THROW("Unsupported data type for WintxUnzip");
    }
    return {output_tensor};
}

std::vector<std::vector<int64_t>> WintXUnzipInferShape(
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& super_scale_shape,
    const std::vector<int64_t>& bs_shift_shape,
    const std::string& quant_method) {
    std::vector<int64_t> output_shape(weight_shape);
    const int unzip_axis = 1;
    if(quant_method == "weight_only_int2.5") {
        output_shape[unzip_axis] = weight_shape[unzip_axis] / 10 * 64;
        PD_CHECK(output_shape[unzip_axis] % 64 == 0, "unzip_size must be divisible by 64 in wint2_5!");
    } else {
        PD_THROW("Unsupported data type for WintxUnzip");
    }
    return {output_shape};
}

std::vector<paddle::DataType> WintXUnzipInferDtype(
    const paddle::DataType& weight_dtype,
    const paddle::DataType& super_scale_dtype,
    const paddle::DataType& bs_shift_dtype) {
    return {super_scale_dtype};
}


PD_BUILD_OP(winx_unzip)
    .Inputs({"weight",
             "super_scale",
             "bs_shift"})
    .Outputs({"output_tensor"})
    .Attrs({"quant_method:std::string"})
    .SetKernelFn(PD_KERNEL(WintXUnzip))
    .SetInferShapeFn(PD_INFER_SHAPE(WintXUnzipInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WintXUnzipInferDtype));