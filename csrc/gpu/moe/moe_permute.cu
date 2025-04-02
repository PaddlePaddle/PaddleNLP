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

#include "helper.h"
#include "moe_perm_unperm/moe_perm_unperm_kernel.h"

template <paddle::DataType T>
void MoePermuteKernel(
    paddle::Tensor& input,                            // [n_token, hidden]
    paddle::Tensor& topk_weights,                     //[n_token, topk]
    paddle::Tensor& topk_ids,                         // [n_token, topk]
    paddle::Tensor& token_expert_indicies,            // [n_token, topk]
    paddle::optional<paddle::Tensor>& expert_map,  // [n_expert]
    int n_expert, int n_local_expert, int topk,
    int align_block_size,
    paddle::Tensor& permuted_input,  // [align_expand_m, hidden]
    paddle::Tensor& expert_first_token_offset,  // [n_local_expert + 1]
    paddle::Tensor& src_row_id2dst_row_id_map,  // [n_token, topk]
    paddle::Tensor& m_indices) {                // [align_expand_m]
  PD_CHECK(topk_weights.dtype() == paddle::DataType::FLOAT32,
              "topk_weights must be float32");
  PD_CHECK(expert_first_token_offset.dtype() == paddle::DataType::INT64,
              "expert_first_token_offset must be int64");
  PD_CHECK(topk_ids.dtype() == paddle::DataType::INT32,
              "topk_ids must be int32");
  PD_CHECK(token_expert_indicies.dtype() == paddle::DataType::INT32,
              "token_expert_indicies must be int32");
  PD_CHECK(src_row_id2dst_row_id_map.dtype() == paddle::DataType::INT32,
              "src_row_id2dst_row_id_map must be int32");
  PD_CHECK(expert_first_token_offset.shape()[0] == n_local_expert + 1,
              "expert_first_token_offset shape != n_local_expert+1");
  PD_CHECK(
      src_row_id2dst_row_id_map.shape() == token_expert_indicies.shape(),
      "token_expert_indicies shape must be same as src_row_id2dst_row_id_map");

  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
  auto n_token = input.shape()[0];
  auto n_hidden = input.shape()[1];
  auto stream = input.stream();
  const long sorter_size =
      CubKeyValueSorter::getWorkspaceSize(n_token * topk, n_expert);
  auto sort_workspace = paddle::empty(
      {sorter_size},
      paddle::DataType::INT8,
      input.place());
  auto permuted_experts_id = paddle::empty_like(topk_ids);
  auto dst_row_id2src_row_id_map = paddle::empty_like(src_row_id2dst_row_id_map);
  auto align_expert_first_token_offset = paddle::full(
    expert_first_token_offset.shape(), 0, expert_first_token_offset.dtype(), expert_first_token_offset.place());

  CubKeyValueSorter sorter{};
  int64_t* valid_num_ptr = nullptr;
  // pre-process kernel for expert-parallelism:
  // no local expert id plus "n_expert" offset for priority to local expert
  // map local expert id [n, .., n+n_local_expert-1] to [0, n_local_expert -1]
  // For example, 4 expert with ep_size=2. ep_rank=1 owns global expert id
  // [2,3] with expert_map[-1, -1, 0, 1], preprocess_topk_id  process topk_ids
  // and map global expert id [2, 3] to local_expert id [0, 1] and map global
  // expert id [0, 1] ( not in ep rank=1)  to [4, 5] by plus n_expert. This map
  // operation is to make local expert high priority in following sort topk_ids
  // and scan local expert_first_token_offset for each ep rank for next group
  // gemm.
  if (expert_map) {
    int* expert_map_ptr = reinterpret_cast<int*>(expert_map.get().data<int>());
    valid_num_ptr =
        get_ptr<int64_t>(expert_first_token_offset) + n_local_expert;
    preprocessTopkIdLauncher(get_ptr<int>(topk_ids), n_token * topk,
                             expert_map_ptr, n_expert, stream);
  }
  // expert sort topk expert id and scan expert id get expert_first_token_offset
  sortAndScanExpert(get_ptr<int>(topk_ids), get_ptr<int>(token_expert_indicies),
                    get_ptr<int>(permuted_experts_id),
                    get_ptr<int>(dst_row_id2src_row_id_map),
                    get_ptr<int64_t>(expert_first_token_offset), n_token,
                    n_expert, n_local_expert, topk, sorter,
                    reinterpret_cast<void*>(sort_workspace.data<int8_t>()), stream);

  expandInputRowsKernelLauncher<data_t>(
        get_ptr<data_t>(input), get_ptr<data_t>(permuted_input),
        get_ptr<float>(topk_weights), get_ptr<int>(permuted_experts_id),
        get_ptr<int>(dst_row_id2src_row_id_map),
        get_ptr<int>(src_row_id2dst_row_id_map),
        get_ptr<int64_t>(expert_first_token_offset), n_token, valid_num_ptr,
        n_hidden, topk, n_local_expert, align_block_size, stream);

  // get m_indices and update expert_first_token_offset with align block
  getMIndices(get_ptr<int64_t>(expert_first_token_offset),
              get_ptr<int64_t>(align_expert_first_token_offset),
              get_ptr<int>(m_indices), n_local_expert, align_block_size,
              stream);
  if (align_block_size > 0) {
    // update align_expert_first_token_offset
    expert_first_token_offset.copy_(align_expert_first_token_offset, input.place(), true);
  }
}

std::vector<paddle::Tensor> MoePermute(
    paddle::Tensor& input,
    paddle::Tensor& topk_weights,
    paddle::Tensor& topk_ids,
    paddle::Tensor& token_expert_indicies,
    paddle::optional<paddle::Tensor>& expert_map,
    int n_expert, int n_local_expert, int topk,
    int align_block_size = -1){

    const auto input_type = input.dtype();
    auto place = input.place();
    const int n_token = input.shape()[0];
    const int hidden_size = input.shape()[1];
    int permuted_row_size = n_token * topk;
    if(align_block_size > 0){
      permuted_row_size = ((permuted_row_size + n_expert * (align_block_size - 1)) / align_block_size) * align_block_size;
    }
    auto permute_input =
        GetEmptyTensor({permuted_row_size, hidden_size}, input_type, place);
    auto expert_first_token_offset =
        GetEmptyTensor({n_local_expert + 1}, paddle::DataType::INT64, place);
    auto src_row_id2dst_row_id_map =
        GetEmptyTensor({n_token, topk}, paddle::DataType::INT32, place);
    auto m_indices =
        GetEmptyTensor({permuted_row_size}, paddle::DataType::INT32, place);
    
    switch (input_type) {
      case paddle::DataType::FLOAT32: 
        MoePermuteKernel<paddle::DataType::FLOAT32>(
            input, topk_weights, topk_ids, token_expert_indicies, expert_map,
            n_expert, n_local_expert, topk, align_block_size,
            permute_input, expert_first_token_offset, src_row_id2dst_row_id_map,
            m_indices);
        break;
      case paddle::DataType::FLOAT16:
        MoePermuteKernel<paddle::DataType::FLOAT16>(
          input, topk_weights, topk_ids, token_expert_indicies, expert_map,
          n_expert, n_local_expert, topk, align_block_size,
          permute_input, expert_first_token_offset, src_row_id2dst_row_id_map,
          m_indices);
        break;
      case paddle::DataType::BFLOAT16:
        MoePermuteKernel<paddle::DataType::BFLOAT16>(
          input, topk_weights, topk_ids, token_expert_indicies, expert_map,
          n_expert, n_local_expert, topk, align_block_size,
          permute_input, expert_first_token_offset, src_row_id2dst_row_id_map,
          m_indices);
        break;
      default:
        PD_THROW("Unsupported data type for MoePermuteKernel");
    }
    return {permute_input, expert_first_token_offset, src_row_id2dst_row_id_map, m_indices};
  }

std::vector<std::vector<int64_t>> MoePermuteInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& topk_weights_shape,
    const std::vector<int64_t>& topk_ids_shape,
    const std::vector<int64_t>& token_expert_indicies_shape,
    const paddle::optional<std::vector<int64_t>>& expert_map_shape,
    int n_expert, int n_local_expert, int topk, int align_block_size) {
  
  const int token_num = input_shape[0];
  const int hidden_size = input_shape[1];

  int permuted_row_size = token_num * topk;
  if(align_block_size > 0){
    permuted_row_size = ((permuted_row_size + n_expert * (align_block_size - 1)) / align_block_size) * align_block_size;
  }

  return {{permuted_row_size, hidden_size},
          {n_local_expert + 1},
          {token_num, topk},
          {permuted_row_size}};
}

std::vector<paddle::DataType> MoePermuteInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& topk_weights_dtype,
    const paddle::DataType& topk_ids_dtype,
    const paddle::DataType& token_expert_indicies_dtype,
    const paddle::optional<paddle::DataType>& expert_map_dtype
  ) {
  return {input_dtype,
          paddle::DataType::INT64,
          paddle::DataType::INT32,
          paddle::DataType::INT32};
}

PD_BUILD_OP(moe_permute)
    .Inputs({"input", "topk_weigths", 
             "topk_ids", 
             "token_expert_indicies", 
             paddle::Optional("expert_map"),
             })
    .Outputs({"permute_input",
              "expert_first_token_offset",
              "src_row_id2dst_row_id_map",
              "m_indices"})
    .Attrs({"n_expert:int", "n_local_expert:int", "topk:int", "align_block_size:int"})
    .SetKernelFn(PD_KERNEL(MoePermute))
    .SetInferShapeFn(PD_INFER_SHAPE(MoePermuteInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoePermuteInferDtype));
