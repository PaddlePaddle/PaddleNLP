
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

#include "moe/wint2_unzip_impl_op.h"
#include "helper.h"

template <paddle::DataType T>
void Wint2UnzipKernel(const paddle::Tensor& w,
                      const paddle::Tensor& w_scale,
                      const paddle::Tensor& w_code_scale,
                      const paddle::Tensor& w_code_zp,
                      const paddle::Tensor& w_super_scale,
                      paddle::Tensor& output_tensor,
                      const std::string& quant_method) {
    using data_t = typename PDTraits<T>::data_t;
    using NvType = typename PDTraits<T>::DataType;

    if (quant_method == "weight_only_int2") {
        const uint8_t* w_ptr = w.data<uint8_t>();
        const NvType* w_scale_ptr = reinterpret_cast<const NvType*>(w_scale.data<data_t>());
        const float* w_code_scale_ptr = w_code_scale.data<float>();
        const float* w_code_zp_ptr = w_code_zp.data<float>();
        const NvType* w_super_scale_kernel_ptr = w_super_scale.initialized() ? reinterpret_cast<const NvType*>(w_super_scale.data<data_t>()) : nullptr;

        NvType* output_tensor_ptr = reinterpret_cast<NvType*>(output_tensor.data<data_t>());

        const int64_t batch = output_tensor.shape()[0];
        const int64_t num_rows = output_tensor.shape()[1];
        const int64_t num_columns = output_tensor.shape()[2];
        Wint2UnzipKernelLauncher<NvType>(
            w_ptr,
            w_scale_ptr, 
            w_code_scale_ptr, 
            w_code_zp_ptr, 
            w_super_scale_kernel_ptr, 
            output_tensor_ptr,
            batch, num_rows, num_columns);
    } else {
        PD_THROW("Unsupported quant_method for Wint2Unzip.");
    }
}

std::vector<paddle::Tensor> Wint2Unzip(const paddle::Tensor& w, 
                                       const paddle::Tensor& w_scale,
                                       const paddle::Tensor& w_code_scale,
                                       const paddle::Tensor& w_code_zp,
                                       const paddle::Tensor& w_super_scale,
                                       const std::string& quant_method) {
  auto place = w.place();
  auto dtype = w_scale.dtype();                              

  auto output_dims = w.dims();
  const int unzip_axis = 1;

  if (quant_method == "weight_only_int2") {
    output_dims[unzip_axis] = output_dims[unzip_axis] * WeightOnlyTraits::kPackNum;
    // PD_CHECK(output_shape[unzip_axis] % WeightOnlyTraits::kGroupSize == 0, "unzip_size must be divisible by 64 in wint2!");
  } else {
    PD_THROW("Unsupported data type for Wint2Unzip");
  }
  auto output_tensor = GetEmptyTensor(output_dims, dtype, place);

  switch (w_scale.dtype()) {
    case paddle::DataType::BFLOAT16:
      Wint2UnzipKernel<paddle::DataType::BFLOAT16>(w,
                                                   w_scale,
                                                   w_code_scale,
                                                   w_code_zp,
                                                   w_super_scale,
                                                   output_tensor,
                                                   quant_method);
      break;
    case paddle::DataType::FLOAT16:
      Wint2UnzipKernel<paddle::DataType::FLOAT16>(w,
                                                  w_scale,
                                                  w_code_scale,
                                                  w_code_zp,
                                                  w_super_scale,
                                                  output_tensor,
                                                  quant_method);
      break;
    default:
      PD_THROW("Unsupported data type for Wint2Unzip");
  }
  return {output_tensor};
}

std::vector<std::vector<int64_t>> Wint2UnzipInferShape(
    const std::vector<int64_t>& w_shape,
    const std::vector<int64_t>& w_scale_shape,
    const std::vector<int64_t>& w_code_scale_shape,
    const std::vector<int64_t>& w_code_zp_shape,
    const std::vector<int64_t>& w_super_scale_shape,
    const std::string& quant_method) {
    std::vector<int64_t> output_shape(w_shape);
    const int unzip_axis = 1;
    if(quant_method == "weight_only_int2") {
        output_shape[unzip_axis] = w_shape[unzip_axis] * WeightOnlyTraits::kPackNum;
        PD_CHECK(output_shape[unzip_axis] % WeightOnlyTraits::kGroupSize == 0, "unzip_size must be divisible by 64 in wint2!");
    } else {
        PD_THROW("Unsupported data type for Wint2Unzip");
    }
    return {output_shape};
}

std::vector<paddle::DataType> Wint2UnzipInferDtype(
    const paddle::DataType& w_dtype,
    const paddle::DataType& w_scale_dtype,
    const paddle::DataType& w_code_scale_dtype,
    const paddle::DataType& w_code_zp_dtype,
    const paddle::DataType& w_super_scale_dtype) {
    return {w_scale_dtype};
}

PD_BUILD_OP(win2_unzip)
    .Inputs({"w", "w_scale", "w_code_scale", "w_code_zp", "w_super_scale"})
    .Outputs({"output_tensor"})
    .Attrs({"quant_method:std::string"})
    .SetKernelFn(PD_KERNEL(Wint2Unzip))
    .SetInferShapeFn(PD_INFER_SHAPE(Wint2UnzipInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Wint2UnzipInferDtype));
