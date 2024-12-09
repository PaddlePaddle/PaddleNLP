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

#include "fp8_gemm_fused/fp8_fp8_gemm_scale_bias_act.h"
#include "fp8_common.h"  // NOLINT
#include "fp8_fp8_half_cuda_core_gemm.h"

std::vector<paddle::Tensor> cutlass_fp8_fp8_half_gemm(
    const paddle::Tensor& x,
    const paddle::Tensor& y,
    const paddle::optional<paddle::Tensor>& bias,
    bool trans_x,
    bool trans_y,
    float scale,  // only support per-tensor quantization
    std::string output_dtype,
    std::string activation_type) {
  paddle::Tensor out;
  void* out_ptr = nullptr;
  const void* x_ptr = nullptr;
  const void* y_ptr = nullptr;

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
  int ldb = y.dims()[rank - 1];

  if (!trans_x) {
    M = x.dims()[rank - 2];
    K = x.dims()[rank - 1];

  } else {
    M = x.dims()[rank - 1];
    K = x.dims()[rank - 2];
  }
  if (!trans_y) {
    N = y.dims()[rank - 1];
    ldd = y.dims()[rank - 1];
  } else {
    N = y.dims()[rank - 2];
    ldd = y.dims()[rank - 2];
  }

  int batch_count = 1;
  for (size_t i = 0; i < rank - 2; ++i) {
    batch_count *= x.dims()[i];
  }

  std::string input_dtype = "";
  if (x.dtype() == phi::DataType::FLOAT8_E4M3FN) {
    input_dtype = "float8_e4m3fn";
    x_ptr = reinterpret_cast<const void*>(x.data<phi::dtype::float8_e4m3fn>());
    y_ptr = reinterpret_cast<const void*>(y.data<phi::dtype::float8_e4m3fn>());
  } else if (x.dtype() == phi::DataType::FLOAT8_E5M2) {
    input_dtype = "float8_e5m2";
    x_ptr = reinterpret_cast<const void*>(x.data<phi::dtype::float8_e5m2>());
    y_ptr = reinterpret_cast<const void*>(y.data<phi::dtype::float8_e5m2>());    
  } else {
    PADDLE_THROW(phi::errors::Fatal(
        "fp8_fp8_half_gemm_fused only support e4m3 and e5m2 input"));
  }

  std::vector<int64_t> out_shape = x.shape();
  out_shape[rank - 1] = N;
  out_shape[rank - 2] = M;

  if (output_dtype == "bfloat16") {
    out = paddle::empty(out_shape, paddle::DataType::BFLOAT16, x.place());
    out_ptr = reinterpret_cast<void*>(out.data<phi::dtype::bfloat16>());
  } else if (output_dtype == "float16") {
    out = paddle::empty(out_shape, paddle::DataType::FLOAT16, x.place());
    out_ptr = reinterpret_cast<void*>(out.data<phi::dtype::float16>());
  } else {
    PADDLE_THROW(phi::errors::Fatal(
        "fp8_fp8_half_gemm_fused only support bfloat16 and float16 output"));
  }

  std::string isbias = bias ? "true" : "false";
  std::string act = (activation_type == "" || activation_type == "identity")
                        ? "noact"
                        : activation_type;

  std::string fuse_gemm_config =
      input_dtype + "_" + output_dtype + "_" + isbias + "_" + act;

  void* bias_data = nullptr;
  std::vector<int64_t> bias_dims{};
  if (bias) {
    bias_dims = common::vectorize(bias.get().dims());
    if (output_dtype == "bfloat16") {
      bias_data = reinterpret_cast<void*>(const_cast<phi::dtype::bfloat16*>(
          bias.get().data<phi::dtype::bfloat16>()));
    } else {
      bias_data = reinterpret_cast<void*>(const_cast<phi::dtype::float16*>(
          bias.get().data<phi::dtype::float16>()));
    }
  }

  if (M <=4 && trans_y && !trans_x && act == "noact" && enable_cuda_core_fp8_gemm()) {
        GemmParams params = {
            x_ptr,
            y_ptr,
            bias_data,
            out_ptr,
            M,
            N,
            K,
            scale,
            stream,
        };

        if (x.dtype() == phi::DataType::FLOAT8_E4M3FN)
        {
            if(output_dtype == "bfloat16") {
                cuda_core_gemm_launcher<__nv_fp8_e4m3, __nv_bfloat16>(params);
                
            } else {
                cuda_core_gemm_launcher<__nv_fp8_e4m3, half>(params);
            }
        } else {
            if(output_dtype == "bfloat16") {
                cuda_core_gemm_launcher<__nv_fp8_e5m2, __nv_bfloat16>(params);
            } else {
                cuda_core_gemm_launcher<__nv_fp8_e5m2, half>(params);
            }
        }
    } else {
        GemmEpilogueAllParams params = {x_ptr,
                                        y_ptr,
                                        out_ptr,
                                        scale,
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
                                        0.01,  // for leaky_relu
                                        bias_data,
                                        bias_dims,
                                        fuse_gemm_config};
        fp8_fp8_gemm_scale_bias_act(params);
    }
  return {out};
}

std::vector<std::vector<int64_t>> CutlassFp8Fp8HalfGemmFusedInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const paddle::optional<std::vector<int64_t>>&  bias_shape,
    bool trans_x,
    bool trans_y){
  PADDLE_ENFORCE_EQ(x_shape.size(),
                    y_shape.size(),
                    phi::errors::InvalidArgument(
                      "The rank of input X and Y should be equal, but received X's rank is %d, Y's rank is %d.",
                      x_shape.size(),
                      y_shape.size()));
      
  int rank = x_shape.size();
  int M = 0;
  int N = 0;

  if (!trans_x) {
    M = x_shape[rank - 2];

  } else {
    M = x_shape[rank - 1];
  }
  if (!trans_y) {
    N = y_shape[rank - 1];
  } else {
    N = y_shape[rank - 2];
  }
  std::vector<int64_t> out_shape = x_shape;
  out_shape[rank - 1] = N;
  out_shape[rank - 2] = M;
  return {out_shape};
}

std::vector<paddle::DataType> CutlassFp8Fp8HalfGemmFusedInferDtype(
    const paddle::DataType& x_type,
    const paddle::DataType& y_type,
    const paddle::optional<paddle::DataType>& bias_type,
    bool trans_x,
    bool trans_y,
    float scale,  // only support per-tensor quantization
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

PD_BUILD_OP(cutlass_fp8_fp8_half_gemm_fused)
    .Inputs({"x", "y", paddle::Optional("bias")})
    .Attrs({"transpose_x: bool",
            "transpose_y: bool",
            "scale: float",
            "output_dtype: std::string",
            "act: std::string"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(cutlass_fp8_fp8_half_gemm))
    .SetInferShapeFn(PD_INFER_SHAPE(CutlassFp8Fp8HalfGemmFusedInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(CutlassFp8Fp8HalfGemmFusedInferDtype));
