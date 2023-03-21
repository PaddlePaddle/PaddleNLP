/*  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
    Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
    This code is copied fron NVIDIA apex: 
    https://github.com/NVIDIA/apex with minor changes.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and 
    limitations under the License. */


#include "paddle/extension.h"
#include "ln.h"

/*

Supported Type combinations:

input    compute   weights   output    
=======================================
fp32     fp32      fp32      fp32      
fp16     fp32      fp16      fp16      
bf16     fp32      bf16      bf16      
fp32     fp32      fp16      fp16      
fp32     fp32      bf16      bf16      

Remarks:
Output type = Weight type
Compute always in FP32

*/

namespace layer_norm {

// Create registries and provide runtime versions of config hash functions.

FwdRegistry FWD_FUNCS;
BwdRegistry BWD_FUNCS;

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t get_type_id(paddle::DataType dtype){
    if( dtype == paddle::DataType::FLOAT16 ) {
        return TypeId<fp16>::Value;
    } else if( dtype == paddle::DataType::BFLOAT16 ) {
        return TypeId<bf16>::Value;
    } else if( dtype == paddle::DataType::FLOAT32 ) {
        return TypeId<fp32>::Value;
    } else {
        PD_CHECK(false, "Type not supported: ", dtype);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t get_key(paddle::DataType wtype, paddle::DataType itype, paddle::DataType otype, paddle::DataType ctype, uint64_t hidden_size) {
    using namespace layer_norm;
    uint64_t type_key = get_type_id(wtype) | (get_type_id(itype) << 2) | (get_type_id(otype) << 4) | (get_type_id(ctype) << 6);
    uint64_t launcher_key = (type_key << 32) | hidden_size;
    return launcher_key;
}

}  // namespace layer_norm

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::FwdFunction & get_fwd_launcher(paddle::DataType wtype, paddle::DataType itype, paddle::DataType otype, paddle::DataType ctype, uint32_t hidden_size) {
    auto iter = layer_norm::FWD_FUNCS.find(layer_norm::get_key(wtype, itype, otype, ctype, hidden_size));
    if( iter != layer_norm::FWD_FUNCS.end() ) {
        return iter->second;
    } else {
        PD_CHECK(false, "FWD: Unsupported hidden_size or types: ", hidden_size, wtype, itype, otype, ctype);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::BwdFunction & get_bwd_launcher(paddle::DataType wtype, paddle::DataType itype, paddle::DataType otype, paddle::DataType ctype, uint32_t hidden_size) {
    auto iter = layer_norm::BWD_FUNCS.find(layer_norm::get_key(wtype, itype, otype, ctype, hidden_size));
    if( iter != layer_norm::BWD_FUNCS.end() ) {
        return iter->second;
    } else {
        PD_CHECK(false, "BWD: Unsupported hidden_size or types: ", hidden_size, wtype, itype, otype, ctype);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static cudaDeviceProp GetDevicePropImpl() {
  int device = -1;
  PD_CHECK(cudaGetDevice(&device) == cudaSuccess); 
  cudaDeviceProp prop;
  PD_CHECK(cudaGetDeviceProperties(&prop, device) == cudaSuccess); 
  return prop;
}

static cudaDeviceProp *GetDeviceProp() {
  static auto prop = GetDevicePropImpl();
  return &prop;
}


std::vector<paddle::Tensor> ln_fwd(const paddle::Tensor &x,      // BxSxhidden_size
                               const paddle::Tensor &gamma,   // hidden_size
                               const paddle::Tensor &beta,   // hidden_size
                               const float epsilon
) {
    auto itype = x.type();
    auto wtype = gamma.type();
    auto otype = wtype;
    auto ctype = paddle::DataType::FLOAT32;

    PD_CHECK(beta.type() == wtype);

    PD_CHECK(!x.is_cpu());
    PD_CHECK(!gamma.is_cpu());
    PD_CHECK(!beta.is_cpu());

    auto sizes = x.shape();
    PD_CHECK(sizes.size() >= 2);

    int rows = 1;  
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
      rows *= sizes[i]; 
    }

    const int cols = sizes[sizes.size() - 1];
    auto hidden_size = gamma.numel();

    PD_CHECK(gamma.shape() == beta.shape());
    PD_CHECK(hidden_size == cols);

    PD_CHECK(epsilon >= 0.f);

    auto place = x.place();

    auto z = paddle::empty(sizes, otype, place);   

    auto mu = paddle::empty({ rows }, ctype, place);
    auto rsigma = paddle::empty({ rows }, ctype, place);

    layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;

    launch_params.props = GetDeviceProp();
    launch_params.stream = x.stream();

    // Request the kernel launcher.
    auto launcher = get_fwd_launcher(wtype, itype, otype, ctype, hidden_size);

    // Query the kernel-specific launch parameters.
    launcher(launch_params, true);

    paddle::Tensor workspace, barrier;

    // Set the kernel runtime parameters.
    layer_norm::FwdParams &params = launch_params.params;
    params.rows = rows;
    params.cols = cols;
    params.x = const_cast<void *>(x.data());
    params.mu = mu.data();
    params.rs = rsigma.data();
    params.gamma = const_cast<void *>(gamma.data());
    params.beta = const_cast<void *>(beta.data());
    params.z = z.data();
    params.epsilon = epsilon;

    if( launch_params.barrier_size > 0 ) {
        barrier = paddle::zeros({launch_params.barrier_size}, paddle::DataType::INT32, place);  
        workspace = paddle::empty({launch_params.workspace_bytes}, paddle::DataType::UINT8, place);
        params.workspace = workspace.data();
        params.barrier = barrier.data<int>();
    }

    // Launch the kernel.
    launcher(launch_params, false);

    return { z, mu, rsigma };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<paddle::Tensor> ln_bwd(const paddle::Tensor &dz,     // BxSxhidden_size
                               const paddle::Tensor &x,      // BxSxhidden_size
                               const paddle::Tensor &mu,     // BxS, FP32!
                               const paddle::Tensor &rsigma, // BxS, FP32!
                               const paddle::Tensor &gamma   // hidden_size
) {

    auto itype = x.type();
    auto wtype = gamma.type();
    auto otype = wtype;
    auto ctype = paddle::DataType::FLOAT32;

    PD_CHECK(dz.dtype() == otype);
    PD_CHECK(mu.dtype() == ctype);
    PD_CHECK(rsigma.dtype() == ctype);

    PD_CHECK(!x.is_cpu());
    PD_CHECK(!dz.is_cpu());
    PD_CHECK(!mu.is_cpu());
    PD_CHECK(!rsigma.is_cpu());
    PD_CHECK(!gamma.is_cpu());


    auto sizes = x.shape();
    PD_CHECK(sizes.size() >= 2);
    PD_CHECK(dz.shape() == sizes);

    int64_t rows = 1; 
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
      rows *= sizes[i];
    }
    auto cols = sizes[sizes.size() - 1];

    auto hidden_size = gamma.numel();

    PD_CHECK(mu.numel() == rows);
    PD_CHECK(mu.shape() == rsigma.shape());

    PD_CHECK(gamma.numel() == cols);


    auto dx = paddle::empty_like(x);
    auto dgamma = paddle::empty_like(gamma);
    auto dbeta = paddle::empty_like(gamma);

    layer_norm::LaunchParams<layer_norm::BwdParams> launch_params;
    launch_params.stream = x.stream();
    launch_params.props = GetDeviceProp();

    auto launcher = get_bwd_launcher(wtype, itype, otype, ctype, hidden_size);

    launcher(launch_params, true);

    auto place = x.place();
    auto dgamma_part = paddle::empty({ launch_params.params.ctas_per_col, hidden_size }, ctype, place); 
    auto dbeta_part = paddle::empty({ launch_params.params.ctas_per_col, hidden_size }, ctype, place);
    paddle::Tensor workspace, barrier;

    layer_norm::BwdParams &params = launch_params.params;
    params.rows = rows;
    params.cols = cols;
    params.x = const_cast<void *>(x.data());
    params.mu = const_cast<void *>(mu.data());
    params.rs = const_cast<void *>(rsigma.data());
    params.gamma = const_cast<void *>(gamma.data());
    params.dz = const_cast<void *>(dz.data());
    params.dx = dx.data();
    params.dbeta = dbeta.data();
    params.dgamma = dgamma.data();
    params.dbeta_part = dbeta_part.data();
    params.dgamma_part = dgamma_part.data();

    if( launch_params.barrier_size > 0 ) {
        barrier = paddle::zeros({launch_params.barrier_size}, paddle::DataType::INT32, place); 
        workspace = paddle::empty({launch_params.workspace_bytes}, paddle::DataType::UINT8, place);
        params.workspace = workspace.data();
        params.barrier = barrier.data<int>();
    }

    launcher(launch_params, false);

    return { dx, dgamma, dbeta, dgamma_part, dbeta_part };
}

////////////////////////////////////////////////////////////////////////////////////////////////////


std::vector<paddle::Tensor> LnFwd(const paddle::Tensor &x, const paddle::Tensor &gamma, const paddle::Tensor &beta, float epsilon) {
  return ln_fwd(x, gamma, beta, epsilon);
} 

std::vector<std::vector<int64_t>> LnFwdInferShape(std::vector<int64_t> x_shape, std::vector<int64_t> gamma_shape, std::vector<int64_t> beta_shape, float epsilon) {
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < x_shape.size(); ++i) {
    rows *= x_shape[i];
  }
  return {x_shape, {rows}, {rows}};  
}  

std::vector<paddle::DataType> LnFwdInferDtype(paddle::DataType x_dtype, paddle::DataType gamma_dtype, paddle::DataType beta_dtype) {
  return {gamma_dtype, paddle::DataType::FLOAT32, paddle::DataType::FLOAT32};  
} 

std::vector<paddle::Tensor> LnBwd(const paddle::Tensor &x, const paddle::Tensor &gamma, const paddle::Tensor &mu, const paddle::Tensor &rsigma, const paddle::Tensor &z_grad, float epsilon) {
  auto result = ln_bwd(z_grad, x, mu, rsigma, gamma);  
  result.resize(3);
  return result;
}

std::vector<std::vector<int64_t>> LnBwdInferShape(std::vector<int64_t> x_shape, std::vector<int64_t> gamma_shape, std::vector<int64_t> mu_shape, std::vector<int64_t> rsigma_shape, std::vector<int64_t> z_grad_shape, float epsilon) {
  return {x_shape, gamma_shape, gamma_shape};  
} 

PD_BUILD_OP(fast_ln)
    .Inputs({"x", "gamma", "beta"})
    .Outputs({"z", "mu", "rsigma"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(LnFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(LnFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(LnFwdInferDtype));

PD_BUILD_GRAD_OP(fast_ln)
    .Inputs({"x", "gamma", "mu", "rsigma", paddle::Grad("z")})
    .Outputs({paddle::Grad("x"), paddle::Grad("gamma"), paddle::Grad("beta")})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(LnBwd))
    .SetInferShapeFn(PD_INFER_SHAPE(LnBwdInferShape));
