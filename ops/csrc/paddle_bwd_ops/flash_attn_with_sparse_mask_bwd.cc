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

PADDLE_API void flash_attn_with_sparse_mask_grad(const Tensor& q, 
                                const Tensor& k, 
                                const Tensor& v, 
                                const Tensor& attn_mask_start_row_indices,
                                const Tensor& out, 
                                const Tensor& softmax_lse, 
                                const Tensor& seed_offset, 
                                const Tensor& out_grad, 
                                float dropout, 
                                bool causal, int attn_mask_start_row, Tensor* q_grad, Tensor* k_grad, Tensor* v_grad);
}
} // namespace paddle



std::vector<Tensor> SRFlashAttnWithSparseMaskBwd(const Tensor &q, 
                                const Tensor &k, 
                                const Tensor &v, 
                                const Tensor &attn_mask_start_row_indices,
                                const Tensor &out, 
                                const Tensor &softmax_lse, 
                                const Tensor &seed_offset, 
                                const Tensor &out_grad, 
                                float dropout, 
                                bool causal, int attn_mask_start_row);


std::vector<Tensor> SRFlashAttnWithSparseMaskBwd(const Tensor &q, 
                                const Tensor &k, 
                                const Tensor &v, 
                                const Tensor &attn_mask_start_row_indices,
                                const Tensor &out, 
                                const Tensor &softmax_lse, 
                                const Tensor &seed_offset, 
                                const Tensor &out_grad, 
                                float dropout, 
                                bool causal, int attn_mask_start_row){
    std::vector<Tensor> res(3);
    paddle::experimental::flash_attn_with_sparse_mask_grad(q, k, v, attn_mask_start_row_indices, out, softmax_lse, seed_offset,
                                        out_grad, dropout, causal, attn_mask_start_row, &res[0], &res[1], &res[2]);
    return res;
}



std::vector<paddle::DataType> SRFlashAttnWithSparseMaskBwdDtype(paddle::DataType q_dtype,
                                            paddle::DataType k_dtype,
                                            paddle::DataType v_dtype,
                                            paddle::DataType attn_mask_start_row_indices_dtype) {
  return {q_dtype, k_dtype, v_dtype, attn_mask_start_row_indices_dtype};

}


std::vector<std::vector<int64_t>> SRFlashAttnWithSparseMaskBwdInferShape(
    std::vector<int64_t> q_shape, std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape, std::vector<int64_t> attn_mask_start_row_indices_shape) {
    return {q_shape, k_shape, v_shape, attn_mask_start_row_indices_shape};
}


PD_BUILD_OP(flash_attn_with_sparse_mask_bwd)
    .Inputs({"q", "k", "v", "attn_mask_start_row_indices", "out", "softmax_lse", "seed_offset", "out_grad"})
    .Outputs({"q_grad", "k_grad", "v_grad"})
    .Attrs({"dropout: float", "causal: bool", "attn_mask_start_row: int"})
    .SetKernelFn(PD_KERNEL(SRFlashAttnWithSparseMaskBwd))
    .SetInferShapeFn(PD_INFER_SHAPE(SRFlashAttnWithSparseMaskBwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SRFlashAttnWithSparseMaskBwdDtype));
