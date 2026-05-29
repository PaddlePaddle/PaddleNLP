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


#pragma once
#include "helper.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/kernel/gemm_universal_streamk.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/permute.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
  }

void cutlass_scaled_sparse_mm_sm90(paddle::Tensor& c, paddle::Tensor const& a,
                                   paddle::Tensor const& b,
                                   paddle::Tensor const& e,
                                   paddle::Tensor const& a_scales,
                                   paddle::Tensor const& b_scales,
                                   std::optional<paddle::Tensor> const& bias);


std::vector<paddle::Tensor> SparseMm(const paddle::Tensor& input,
                    const paddle::Tensor& a_nzs, 
                    const paddle::Tensor& a_meta,
                    const paddle::Tensor& scalar_scale, 
                    const paddle::Tensor& channel_scale,
                    const paddle::optional<paddle::Tensor>& bias,
                    std::string output_dtype) {
    int64_t m = input.shape()[0];// 非转置后的维度 mxk
    int64_t n = a_nzs.shape()[0];
    
    std::optional<paddle::Tensor> std_bias;
    if (bias) {
        std_bias = *bias; // 或者可以使用 std::optional<paddle::Tensor>(bias.value());
    }

    if (output_dtype == "bfloat16") {
      paddle::Tensor out =
              paddle::empty({m, n}, paddle::DataType::BFLOAT16, a_nzs.place());      
      cutlass_scaled_sparse_mm_sm90(out, input, a_nzs, a_meta, scalar_scale, channel_scale,
                                  std_bias);
      return {out};
    } else if (output_dtype == "float16") {
       paddle::Tensor out =
              paddle::empty({m, n}, paddle::DataType::FLOAT16, a_nzs.place());
      cutlass_scaled_sparse_mm_sm90(out, input, a_nzs, a_meta, scalar_scale, channel_scale,
                                  std_bias);
      return {out};
    }
}

std::vector<std::vector<int64_t>> SparseMmInferShape(const std::vector<int64_t>& input,
      const std::vector<int64_t>& a_nzs,
      const std::vector<int64_t>& a_meta,
      const std::vector<int64_t>& scalar_scale,
      const std::vector<int64_t>& channel_scale,
      const paddle::optional<std::vector<int64_t>>&  bias_shape) {
    int64_t m = input[0];
    int64_t n = a_nzs[0];
    return {{m, n}};
}

std::vector<paddle::DataType> SparseMmInferDtype(const paddle::DataType& input, 
    const paddle::DataType& a_nzs, 
    const paddle::DataType& a_meta,
    const paddle::DataType& scalar_scale_type, 
    const paddle::DataType& channel_scale_type,
    const paddle::optional<paddle::DataType>& bias_type,
    std::string output_dtype) {
  
    paddle::DataType data_type;
    if (output_dtype == "bfloat16")
        data_type = paddle::DataType::BFLOAT16;
    else if (output_dtype ==  "float16")
        data_type = paddle::DataType::FLOAT16;
    else 
        PD_THROW(
                "fp8_fp8_half_gemm_fused only support bfloat16 and float16 output");
    return {data_type};
}

PD_BUILD_OP(sparse_mm)
    .Inputs({"input", "a_nzs", "a_meta",  "scalar_scale", "channel_scale", paddle::Optional("bias")})
    .Attrs({"output_dtype: std::string"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(SparseMm))
    .SetInferShapeFn(PD_INFER_SHAPE(SparseMmInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SparseMmInferDtype));