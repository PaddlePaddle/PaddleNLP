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

// #include "moe_perm_unperm/moe_perm_unperm_kernel.h"

// void moe_permute(
//     paddle::Tensor& input,                            // [n_token, hidden]
//     paddle::Tensor& topk_weights,                     //[n_token, topk]
//     paddle::Tensor& topk_ids,                         // [n_token, topk]
//     paddle::Tensor& token_expert_indicies,            // [n_token, topk]
//     const std::optional<paddle::Tensor>& expert_map,  // [n_expert]
//     int64_t n_expert, int64_t n_local_expert, int64_t topk,
//     const std::optional<int64_t>& align_block_size,
//     paddle::Tensor&
//         permuted_input,  // [topk * n_token/align_block_size_m, hidden]
//     paddle::Tensor& expert_first_token_offset,  // [n_local_expert + 1]
//     paddle::Tensor& src_row_id2dst_row_id_map,  // [n_token, topk]
//     paddle::Tensor& m_indices) {                // [align_expand_m]
//   PD_CHECK(topk_weights.scalar_type() == at::ScalarType::Float,
//               "topk_weights must be float32");
//   PD_CHECK(expert_first_token_offset.scalar_type() == at::ScalarType::Long,
//               "expert_first_token_offset must be int64");
//   PD_CHECK(topk_ids.scalar_type() == at::ScalarType::Int,
//               "topk_ids must be int32");
//   PD_CHECK(token_expert_indicies.scalar_type() == at::ScalarType::Int,
//               "token_expert_indicies must be int32");
//   PD_CHECK(src_row_id2dst_row_id_map.scalar_type() == at::ScalarType::Int,
//               "src_row_id2dst_row_id_map must be int32");
//   PD_CHECK(expert_first_token_offset.size(0) == n_local_expert + 1,
//               "expert_first_token_offset shape != n_local_expert+1")
//   PD_CHECK(
//       src_row_id2dst_row_id_map.sizes() == token_expert_indicies.sizes(),
//       "token_expert_indicies shape must be same as src_row_id2dst_row_id_map");
//   auto n_token = input.sizes()[0];
//   auto n_hidden = input.sizes()[1];
//   auto align_block_size_value =
//       align_block_size.has_value() ? align_block_size.value() : -1;
//   auto stream = at::cuda::getCurrentCUDAStream().stream();
//   const long sorter_size =
//       CubKeyValueSorter::getWorkspaceSize(n_token * topk, n_expert);
//   auto sort_workspace = paddle::empty(
//       {sorter_size},
//       paddle::dtype(paddle::kInt8).device(paddle::kCUDA).requires_grad(false));
//   auto permuted_experts_id = paddle::empty_like(topk_ids);
//   auto dst_row_id2src_row_id_map = paddle::empty_like(src_row_id2dst_row_id_map);
//   auto align_expert_first_token_offset =
//       paddle::zeros_like(expert_first_token_offset);

//   CubKeyValueSorter sorter{};
//   int64_t* valid_num_ptr = nullptr;
//   // pre-process kernel for expert-parallelism:
//   // no local expert id plus "n_expert" offset for priority to local expert
//   // map local expert id [n, .., n+n_local_expert-1] to [0, n_local_expert -1]
//   // For example, 4 expert with ep_size=2. ep_rank=1 owns global expert id
//   // [2,3] with expert_map[-1, -1, 0, 1], preprocess_topk_id  process topk_ids
//   // and map global expert id [2, 3] to local_expert id [0, 1] and map global
//   // expert id [0, 1] ( not in ep rank=1)  to [4, 5] by plus n_expert. This map
//   // operation is to make local expert high priority in following sort topk_ids
//   // and scan local expert_first_token_offset for each ep rank for next group
//   // gemm.
//   if (expert_map.has_value()) {
//     int* expert_map_ptr = reinterpret_cast<int*>(expert_map.value().data_ptr());
//     valid_num_ptr =
//         get_ptr<int64_t>(expert_first_token_offset) + n_local_expert;
//     preprocessTopkIdLauncher(get_ptr<int>(topk_ids), n_token * topk,
//                              expert_map_ptr, n_expert, stream);
//   }
//   // std::cout << "tops id " << topk_ids << std::endl;
//   // expert sort topk expert id and scan expert id get expert_first_token_offset
//   sortAndScanExpert(get_ptr<int>(topk_ids), get_ptr<int>(token_expert_indicies),
//                     get_ptr<int>(permuted_experts_id),
//                     get_ptr<int>(dst_row_id2src_row_id_map),
//                     get_ptr<int64_t>(expert_first_token_offset), n_token,
//                     n_expert, n_local_expert, topk, sorter,
//                     get_ptr<int>(sort_workspace), stream);
//   // std::cout << "permuted_experts_id" << permuted_experts_id << std::endl;
//   // std::cout << "dst_row_id2src_row_id_map" << dst_row_id2src_row_id_map
//   //           << std::endl;

//   // dispatch expandInputRowsKernelLauncher
//   MOE_DISPATCH(input.scalar_type(), [&] {
//     expandInputRowsKernelLauncher<scalar_t>(
//         get_ptr<scalar_t>(input), get_ptr<scalar_t>(permuted_input),
//         get_ptr<float>(topk_weights), get_ptr<int>(permuted_experts_id),
//         get_ptr<int>(dst_row_id2src_row_id_map),
//         get_ptr<int>(src_row_id2dst_row_id_map),
//         get_ptr<int64_t>(expert_first_token_offset), n_token, valid_num_ptr,
//         n_hidden, topk, n_local_expert, align_block_size_value, stream);
//   });

//   // get m_indices and update expert_first_token_offset with align block
//   getMIndices(get_ptr<int64_t>(expert_first_token_offset),
//               get_ptr<int64_t>(align_expert_first_token_offset),
//               get_ptr<int>(m_indices), n_local_expert, align_block_size_value,
//               stream);
//   if (align_block_size.has_value()) {
//     // update align_expert_first_token_offset
//     expert_first_token_offset.copy_(align_expert_first_token_offset);
//   }
// }

// void moe_unpermute(
//     paddle::Tensor& permuted_hidden_states,     // [n_token * topk, hidden]
//     paddle::Tensor& topk_weights,               //[n_token, topk]
//     paddle::Tensor& topk_ids,                   // [n_token, topk]
//     paddle::Tensor& src_row_id2dst_row_id_map,  // [n_token, topk]
//     paddle::Tensor& expert_first_token_offset,  // [n_local_expert+1]
//     int64_t n_expert, int64_t n_local_expert, int64_t topk,
//     paddle::Tensor& hidden_states  // [n_token, hidden]
// ) {
//   PD_CHECK(src_row_id2dst_row_id_map.sizes() == topk_ids.sizes(),
//               "topk_ids shape must be same as src_row_id2dst_row_id_map");
//   PD_CHECK(topk_ids.scalar_type() == at::ScalarType::Int,
//               "topk_ids must be int32");
//   PD_CHECK(
//       permuted_hidden_states.scalar_type() == hidden_states.scalar_type(),
//       "topk_ids dtype must be same as src_row_id2dst_row_id_map");
//   // PD_CHECK(permuted_hidden_states.size(0) == hidden_states.size(0) * topk,
//   //             "permuted_hidden_states must be [n_token * topk, n_hidden],"
//   //             "hidden_states must be [n_token, n_hidden]");
//   auto n_token = hidden_states.size(0);
//   auto n_hidden = hidden_states.size(1);
//   auto stream = at::cuda::getCurrentCUDAStream().stream();
//   int64_t* valid_ptr =
//       get_ptr<int64_t>(expert_first_token_offset) + n_local_expert;
//   MOE_DISPATCH(hidden_states.scalar_type(), [&] {
//     finalizeMoeRoutingKernelLauncher<scalar_t, scalar_t>(
//         get_ptr<scalar_t>(permuted_hidden_states),
//         get_ptr<scalar_t>(hidden_states), get_ptr<float>(topk_weights),
//         get_ptr<int>(src_row_id2dst_row_id_map), get_ptr<int>(topk_ids),
//         n_token, n_hidden, topk, valid_ptr, stream);
//   });
// }

// // TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
// //   m.impl("moe_permute", &moe_permute);
// //   m.impl("moe_unpermute", &moe_unpermute);
// // }

// PD_BUILD_OP(moe_permute)
//     .Inputs({"input", "gating_output"})
//     .Outputs({"permute_input",
//               "token_nums_per_expert",
//               "permute_indices_per_token",
//               "expert_scales_float",
//               "top_k_indices"})
//     .Attrs({"moe_topk:int", "group_moe:bool", "topk_only_mode:bool"})
//     .SetKernelFn(PD_KERNEL(MoeExpertDispatch))
//     .SetInferShapeFn(PD_INFER_SHAPE(MoeExpertDispatchInferShape))
//     .SetInferDtypeFn(PD_INFER_DTYPE(MoeExpertDispatchInferDtype));
