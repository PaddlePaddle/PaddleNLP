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

#include <vector>

#include "paddle/extension.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/ddim.h"

using paddle::Tensor;

namespace paddle {
namespace experimental {

PADDLE_API void add_grad(const Tensor& x,
                         const Tensor& y,
                         const Tensor& out_grad,
                         int axis,
                         Tensor* dx,
                         Tensor* dy);

}
} // namespace paddle

namespace phi {



} // namespace phi

std::vector<Tensor> SRAddBwd(const Tensor& x,
                             const Tensor& weight,
                             const Tensor& bias,
                             const Tensor& out_grad,
                             int axis);

std::vector<Tensor> SRAddBwd(const Tensor& x,
                             const Tensor& weight,
                             const Tensor& bias,
                             const Tensor& out_grad,
                             int axis){
    std::vector<Tensor> res(2);

    std::vector<int64_t> dims_x = phi::vectorize(x.dims());
    std::vector<int64_t> dims_w = phi::vectorize(weight.dims());
    auto ndims_x = dims_x.size();
    auto ndims_w = dims_w.size();

    PADDLE_ENFORCE_GT(ndims_x,
                      1UL,
                      phi::errors::InvalidArgument(
                          "The Input(x) dims size must be greater than 1. Other cases are not supported"));
  
    PADDLE_ENFORCE_GT(ndims_w,
                      1UL,
                      phi::errors::InvalidArgument(
                          "The Input(w) dims size must be greater than 1. Other cases are not supported"));

    size_t M, N;
    M = dims_x[ndims_x - 2];
    N = dims_w[ndims_w - 1];

    std::vector<int64_t> new_dims;
    if (ndims_x > ndims_w) {
      new_dims.assign(dims_x.begin(), dims_x.end() - 2);
    } else if (ndims_x < ndims_w) {
      new_dims.assign(dims_w.begin(), dims_w.end() - 2);
    } else {
      new_dims.reserve(ndims_x);
      for (size_t i = 0; i < ndims_x - 2; ++i) {
        new_dims.push_back(std::max(dims_x[i], dims_w[i]));
      }
    }

    new_dims.push_back(M);
    new_dims.push_back(N);  
    auto ddim_out = phi::make_ddim(new_dims);

    phi::DenseTensor* new_x = new phi::DenseTensor();
    new_x->Resize(ddim_out);
    Tensor tensor_x(std::make_shared<phi::DenseTensor>(*new_x));

    paddle::experimental::add_grad(tensor_x, bias, out_grad, axis, &res[0], &res[1]);

    delete new_x;
    return res;
}

std::vector<paddle::DataType> SRAddBwdDtype(paddle::DataType x_dtype,
                                            paddle::DataType y_dtype,
                                            paddle::DataType z_dtype){
    return {x_dtype, y_dtype, z_dtype};
}

std::vector<std::vector<int64_t>> SRAddBwdInferShape(std::vector<int64_t> x_shape,
                                                     std::vector<int64_t> y_shape,
                                                     std::vector<int64_t> z_shape){
    return {x_shape, y_shape, z_shape};
}

PD_BUILD_OP(add_bwd)
    .Inputs({"x", "weight", "bias", "out_grad"})
    .Outputs({"xw_grad", "bias_grad"})
    .Attrs({"axis: int"})
    .SetKernelFn(PD_KERNEL(SRAddBwd))
    .SetInferShapeFn(PD_INFER_SHAPE(SRAddBwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SRAddBwdDtype));