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

#include "paddle/extension.h"
#include <vector>

using paddle::Tensor;

namespace paddle {
namespace experimental {

PADDLE_API void matmul_grad(const Tensor& x,
                            const Tensor& y,
                            const Tensor& out_grad,
                            bool transpose_x,
                            bool transpose_y,
                            Tensor* dx,
                            Tensor* dy);

}
} // namespace paddle

std::vector<Tensor> SRMatmulBwd(const Tensor& x,
                                const Tensor& y,
                                const Tensor& out_grad,
                                bool transpose_x,
                                bool transpose_y);

std::vector<Tensor> SRMatmulBwd(const Tensor& x,
                                const Tensor& y,
                                const Tensor& out_grad,
                                bool transpose_x,
                                bool transpose_y){
    std::vector<Tensor> res(2);
    paddle::experimental::matmul_grad(x, y, out_grad, transpose_x, transpose_y,
                                      &res[0], &res[1]);
    return res;
}

std::vector<paddle::DataType> SRMatmulBwdDtype(paddle::DataType x_dtype,
                                               paddle::DataType y_dtype){
    return {x_dtype, y_dtype};
}

std::vector<std::vector<int64_t>> SRMatmulBwdInferShape(std::vector<int64_t> x_shape,
                                                     std::vector<int64_t> y_shape){
    return {x_shape, y_shape};
}

PD_BUILD_OP(matmul_bwd)
    .Inputs({"x", "y", "out_grad"})
    .Outputs({"x_grad", "y_grad"})
    .Attrs({"transpose_x: bool", "transpose_y: bool"})
    .SetKernelFn(PD_KERNEL(SRMatmulBwd))
    .SetInferShapeFn(PD_INFER_SHAPE(SRMatmulBwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SRMatmulBwdDtype));