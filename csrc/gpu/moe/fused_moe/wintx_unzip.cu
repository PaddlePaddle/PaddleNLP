// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "moe/wintx_unzip_impl_op.h"
#include "helper.h"

template <paddle::DataType T>
void WintxUnzipKernel(const paddle::Tensor& zipped_weight,
                      const paddle::Tensor& super_scale,
                      paddle::Tensor& weight,
                      const std::string& quant_method) {
    using DataType_ = typename PDTraits<T>::DataType;
    using data_t = typename PDTraits<T>::data_t;
    using NvType = typename PDTraits<T>::DataType;

    const auto* super_scale_ptr = super_scale.data<data_t>();
    auto* weight_ptr = weight.data<data_t>();

    if (quant_method == "weight_only_int2.5") {
        const auto* zipped_weight_ptr = zipped_weight.data<int16_t>();
        const int64_t batch = weight.shape()[0];
        const int64_t num_rows = weight.shape()[1];
        const int64_t num_columns = weight.shape()[2];
        WintxUnzipKernelLauncher<DataType_>(
            reinterpret_cast<const uint16_t*>(zipped_weight_ptr),
            reinterpret_cast<const NvType*>(super_scale_ptr), 
            reinterpret_cast<NvType*>(weight_ptr),
            batch, num_rows, num_columns);
    } else {
        PD_THROW("Unsupported quant_method for WintxUnzip.");
    }
}

std::vector<paddle::Tensor> WintXUnzip(const paddle::Tensor& zipped_weight, 
                                       const paddle::Tensor& super_scale,
                                       const std::string& quant_method) {
  auto place = zipped_weight.place();
  auto dtype = super_scale.dtype();                              

  auto output_dims = zipped_weight.dims();
  const int unzip_axis = 1;
  if (quant_method == "weight_only_int2.5") {
    output_dims[unzip_axis] = zipped_weight.dims()[unzip_axis] / 10 * 64;
  } else {
    PD_THROW("Unsupported data type for WintxUnzip");
  }
  auto output_tensor = GetEmptyTensor(output_dims, dtype, place);

  switch (super_scale.dtype()) {
    case paddle::DataType::BFLOAT16:
      WintxUnzipKernel<paddle::DataType::BFLOAT16>(zipped_weight,
                                                   super_scale,
                                                   output_tensor,
                                                   quant_method);
      break;
    case paddle::DataType::FLOAT16:
      WintxUnzipKernel<paddle::DataType::FLOAT16>(zipped_weight,
                                                  super_scale,
                                                  output_tensor,
                                                  quant_method);
      break;
    default:
      PD_THROW("Unsupported data type for WintxUnzip");
  }
  return {output_tensor};
}

std::vector<std::vector<int64_t>> WintXUnzipInferShape(
    const std::vector<int64_t>& zipped_weight_shape,
    const std::vector<int64_t>& super_scale_shape,
    const std::string& quant_method) {
    std::vector<int64_t> output_shape(zipped_weight_shape);
    const int unzip_axis = 1;
    if(quant_method == "weight_only_int2.5") {
        output_shape[unzip_axis] = zipped_weight_shape[unzip_axis] / 10 * 64;
        PD_CHECK(output_shape[unzip_axis] % 64 == 0, "unzip_size must be divisible by 64 in wint2_5!");
    } else {
        PD_THROW("Unsupported data type for WintxUnzip");
    }
    return {output_shape};
}

std::vector<paddle::DataType> WintXUnzipInferDtype(
    const paddle::DataType& zipped_weight_dtype,
    const paddle::DataType& super_scale_dtype) {
    return {super_scale_dtype};
}

PD_BUILD_OP(winx_unzip)
    .Inputs({"zipped_weight", "super_scale"})
    .Outputs({"output_tensor"})
    .Attrs({"quant_method:std::string"})
    .SetKernelFn(PD_KERNEL(WintXUnzip))
    .SetInferShapeFn(PD_INFER_SHAPE(WintXUnzipInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WintXUnzipInferDtype));