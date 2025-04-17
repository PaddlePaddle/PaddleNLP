// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>

#include "fp8_gemm_fused/fp8_fp8_dual_gemm_scale_bias_act.h"
#include "fp8_common.h"  // NOLINT

std::vector<paddle::Tensor> cutlass_fp8_fp8_fp8_dual_gemm(
    const paddle::Tensor& x,
    const paddle::Tensor& y0,
    const paddle::Tensor& y1,
    const paddle::optional<paddle::Tensor>& bias0,
    const paddle::optional<paddle::Tensor>& bias1,
    bool trans_x,
    bool trans_y,
    float scale0,     // only support per-tensor quantization
    float scale1,     // only support per-tensor quantization
    float scale_out,  // only support per-tensor quantization
    std::string activation_type) {
  paddle::Tensor out;
  void* out_ptr = nullptr;
  const void* x_ptr = nullptr;
  const void* y0_ptr = nullptr;
  const void* y1_ptr = nullptr;

  auto place = x.place();
  cudaStream_t stream = x.stream();
  int64_t device_id = place.GetDeviceId();
  int sm_version = GetGPUComputeCapability(device_id);

  int rank = x.dims().size();
  int M = 0;
  int K = 0;
  int N = 0;
  int ldd = 0;

  int lda = x.dims()[rank - 1];
  int ldb = y0.dims()[rank - 1];
  if (!trans_x) {
    M = x.dims()[rank - 2];
    K = x.dims()[rank - 1];

  } else {
    M = x.dims()[rank - 1];
    K = x.dims()[rank - 2];
  }

  if (!trans_y) {
    N = y0.dims()[rank - 1];
    ldd = y0.dims()[rank - 1];
  } else {
    N = y0.dims()[rank - 2];
    ldd = y0.dims()[rank - 2];
  }

  int batch_count = 1;
  for (size_t i = 0; i < rank - 2; ++i) {
    batch_count *= x.dims()[i];
  }

  std::string input_dtype = "";
  std::string output_dtype = "";
  std::vector<int64_t> out_shape = x.shape();
  out_shape[rank - 1] = N;
  out_shape[rank - 2] = M;
  
  if (x.dtype() == phi::DataType::FLOAT8_E4M3FN) {
    input_dtype = "float8_e4m3fn";
    output_dtype = "float8_e4m3fn";
    x_ptr = reinterpret_cast<const void*>(x.data<phi::dtype::float8_e4m3fn>());
    y0_ptr = reinterpret_cast<const void*>(y0.data<phi::dtype::float8_e4m3fn>());
    y1_ptr = reinterpret_cast<const void*>(y1.data<phi::dtype::float8_e4m3fn>());
    out = paddle::empty(out_shape, paddle::DataType::FLOAT8_E4M3FN, x.place());
    out_ptr = reinterpret_cast<void*>(out.data<phi::dtype::float8_e4m3fn>());
  } 
  else if (x.dtype() == phi::DataType::FLOAT8_E5M2) {
    input_dtype = "float8_e5m2";
    output_dtype = "float8_e5m2";
    x_ptr = reinterpret_cast<const void*>(x.data<phi::dtype::float8_e5m2>());
    y0_ptr = reinterpret_cast<const void*>(y0.data<phi::dtype::float8_e5m2>());
    y1_ptr = reinterpret_cast<const void*>(y1.data<phi::dtype::float8_e5m2>());
    out = paddle::empty(out_shape, paddle::DataType::FLOAT8_E5M2, x.place());
    out_ptr = reinterpret_cast<void*>(out.data<phi::dtype::float8_e5m2>());
  } else {
    PADDLE_THROW(phi::errors::Fatal(
        "fp8_fp8_fp8_dual_gemm_fused only support e4m3 and e5m2 input"));
  }

  std::string isbias = "false";
  std::string bias_dtype = "float16";
  void* bias_data0 = nullptr;
  void* bias_data1 = nullptr;
  std::vector<int64_t> bias_dims0{};
  std::vector<int64_t> bias_dims1{};
  if (bias0 && bias1) {
    isbias = "true";
    bias_dims0 = common::vectorize(bias0.get().dims());
    bias_dims1 = common::vectorize(bias1.get().dims());
    if (bias0.get().dtype() == phi::DataType::FLOAT16) {
      bias_dtype = "float16";
      bias_data0 = reinterpret_cast<void*>(const_cast<phi::dtype::float16*>(
          bias0.get().data<phi::dtype::float16>()));
      bias_data1 = reinterpret_cast<void*>(const_cast<phi::dtype::float16*>(
          bias1.get().data<phi::dtype::float16>()));
    } else {
      bias_dtype = "bfloat16";
      bias_data0 = reinterpret_cast<void*>(const_cast<phi::dtype::bfloat16*>(
          bias0.get().data<phi::dtype::bfloat16>()));
      bias_data1 = reinterpret_cast<void*>(const_cast<phi::dtype::bfloat16*>(
          bias1.get().data<phi::dtype::bfloat16>()));
    }
  }

  paddle::Tensor out0;
  paddle::Tensor out1;
  void* out0_ptr = nullptr;
  void* out1_ptr = nullptr;
  if (bias0 && bias1) {
    if (bias_dtype == "float16") {
      out0 = paddle::empty(out_shape, phi::DataType::FLOAT16, x.place());
      out0_ptr = reinterpret_cast<void*>(out0.data<phi::dtype::float16>());
      out1 = paddle::empty(out_shape, phi::DataType::FLOAT16, x.place());
      out1_ptr = reinterpret_cast<void*>(out1.data<phi::dtype::float16>());
    } else {
      out0 = paddle::empty(out_shape, phi::DataType::BFLOAT16, x.place());
      out0_ptr = reinterpret_cast<void*>(out0.data<phi::dtype::bfloat16>());
      out1 = paddle::empty(out_shape, phi::DataType::BFLOAT16, x.place());
      out1_ptr = reinterpret_cast<void*>(out1.data<phi::dtype::bfloat16>());
    }
  }

  std::string act = (activation_type == "") ? "swiglu" : activation_type;

  std::string fuse_gemm_config =
      input_dtype + "_" + output_dtype + "_" + bias_dtype + "_" + isbias + "_" + act;

  DualGemmEpilogueAllParams params = {
      x_ptr,
      y0_ptr,
      y1_ptr,
      out0_ptr,
      out1_ptr,
      out_ptr,
      scale0,
      scale1,
      scale_out,
      M,
      N,
      K,
      lda,
      ldb,
      ldd,
      batch_count,
      place,
      stream,
      sm_version,
      bias_data0,
      bias_data1,
      bias_dims0,
      bias_dims1,
      fuse_gemm_config};
  
  fp8_fp8_dual_gemm_scale_bias_act(params);
  
  return {out};
}



std::vector<std::vector<int64_t>> CutlassFp8Fp8Fp8DualGemmFusedInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y0_shape,
    const std::vector<int64_t>& y1_shape,
    const paddle::optional<std::vector<int64_t>>&  bias0_shape,
    const paddle::optional<std::vector<int64_t>>&  bias1_shape,
    bool trans_x,
    bool trans_y){
  if(x_shape.size()!=y0_shape.size()){
    PD_THROW("The rank of input X and Y0 should be equal, but received X's rank is %d, Y0's rank is %d",
                      x_shape.size(),
                      y0_shape.size());
  }

  if(y0_shape.size()!=y1_shape.size()){
    PD_THROW("The rank of input Y0 and Y1 should be equal, but received Y0's rank is %d, Y1's rank is %d.",
                      y0_shape.size(),
                      y1_shape.size());
  }

  int rank = x_shape.size();
  int M = 0;
  int N = 0;

  if (!trans_x) {
    M = x_shape[rank - 2];

  } else {
    M = x_shape[rank - 1];
  }
  if (!trans_y) {
    N = y0_shape[rank - 1];
  } else {
    N = y0_shape[rank - 2];
  }
  std::vector<int64_t> out_shape = x_shape;
  out_shape[rank - 1] = N;
  out_shape[rank - 2] = M;
  return {out_shape};
}

std::vector<paddle::DataType> CutlassFp8Fp8Fp8DualGemmFusedInferDtype(
    const paddle::DataType& x_type,
    const paddle::DataType& y0_type,
    const paddle::DataType& y1_type,
    const paddle::optional<paddle::DataType>& bias0_type,
    const paddle::optional<paddle::DataType>& bias1_type) {


    if(x_type != y0_type){
      PD_THROW("The type of input X and Y0 should be equal, but received X's type is %s, Y0's type is %s.",
                    x_type,
                    y0_type);
    }

    if(y0_type != y1_type){
      PD_THROW("The type of input Y0 and Y1 should be equal, but received Y0's type is %s, Y1's type is %s.",
                    y0_type,
                    y1_type);
    }

    if(bias0_type != bias1_type){
      PD_THROW("The type of bias0 and bias1 should be equal, but received bias0's type is %s, bias1's type is %s.",
                      bias0_type,
                      bias1_type);
    }


    paddle::DataType data_type;
    data_type = paddle::DataType::FLOAT8_E4M3FN;
    return {data_type};
}

PD_BUILD_OP(cutlass_fp8_fp8_fp8_dual_gemm_fused)
    .Inputs({"x", "y0", "y1", paddle::Optional("bias0"), paddle::Optional("bias1")})
    .Attrs({"transpose_x: bool",
            "transpose_y: bool",
            "scale0: float",
            "scale1: float",
            "scale_out: float",
            "act: std::string"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(cutlass_fp8_fp8_fp8_dual_gemm))
    .SetInferShapeFn(PD_INFER_SHAPE(CutlassFp8Fp8Fp8DualGemmFusedInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(CutlassFp8Fp8Fp8DualGemmFusedInferDtype));
