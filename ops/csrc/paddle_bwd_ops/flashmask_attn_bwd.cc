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

PADDLE_API void flashmask_attention_grad(const Tensor& q, 
                                const Tensor& k, 
                                const Tensor& v, 
                                const Tensor& startend_row_indices,
                                const Tensor& out, 
                                const Tensor& softmax_lse, 
                                const Tensor& seed_offset, 
                                const Tensor& out_grad, 
                                float dropout, 
                                bool causal, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad);
}
} // namespace paddle



std::vector<Tensor> SRFlashMaskAttnBwd(const Tensor &q, 
                                const Tensor &k, 
                                const Tensor &v, 
                                const Tensor &startend_row_indices,
                                const Tensor &out, 
                                const Tensor &softmax_lse, 
                                const Tensor &seed_offset, 
                                const Tensor &out_grad, 
                                float dropout, 
                                bool causal);


std::vector<Tensor> SRFlashMaskAttnBwd(const Tensor &q, 
                                const Tensor &k, 
                                const Tensor &v, 
                                const Tensor &startend_row_indices,
                                const Tensor &out, 
                                const Tensor &softmax_lse, 
                                const Tensor &seed_offset, 
                                const Tensor &out_grad, 
                                float dropout, 
                                bool causal){
    std::vector<Tensor> res(3);
    paddle::experimental::flashmask_attention_grad(q, k, v, startend_row_indices, out, softmax_lse, seed_offset,
                                        out_grad, dropout, causal, &res[0], &res[1], &res[2]);
    return res;
}



std::vector<paddle::DataType> SRFlashMaskAttnBwdDtype(paddle::DataType q_dtype,
                                            paddle::DataType k_dtype,
                                            paddle::DataType v_dtype,
                                            paddle::DataType startend_row_indices_dtype) {
  return {q_dtype, k_dtype, v_dtype, startend_row_indices_dtype};

}


std::vector<std::vector<int64_t>> SRFlashMaskAttnBwdInferShape(
    std::vector<int64_t> q_shape, std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape, std::vector<int64_t> startend_row_indices_shape) {
    return {q_shape, k_shape, v_shape, startend_row_indices_shape};
}


PD_BUILD_OP(flashmask_attn_bwd)
    .Inputs({"q", "k", "v", "startend_row_indices", "out", "softmax_lse", "seed_offset", "out_grad"})
    .Outputs({"q_grad", "k_grad", "v_grad"})
    .Attrs({"dropout: float", "causal: bool"})
    .SetKernelFn(PD_KERNEL(SRFlashMaskAttnBwd))
    .SetInferShapeFn(PD_INFER_SHAPE(SRFlashMaskAttnBwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SRFlashMaskAttnBwdDtype));
