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


// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma once

#include "moe/fused_moe_helper.h"
#include "moe/fused_moe_op.h"
#pragma GCC diagnostic pop

#include "helper.h"


template <paddle::DataType T>
void MoeDispatchKernel(const paddle::Tensor& input,
                       const paddle::Tensor& gating_output,
                       const int moe_topk,
                       const bool group_moe,
                       const bool topk_only_mode,
                       const int num_rows,
                       const int hidden_size,
                       const int expert_num,
                       paddle::Tensor* permute_input,
                       paddle::Tensor* tokens_expert_prefix_sum,
                       paddle::Tensor* permute_indices_per_token,
                       paddle::Tensor* top_k_weight,
                       paddle::Tensor* top_k_indices) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto stream = input.stream();
  auto place = input.place();

  if (group_moe) {
    // Check if expert_num is divisible by moe_topk, else throw an error
    PADDLE_ENFORCE_EQ(expert_num % moe_topk,
                      0,
                      common::errors::InvalidArgument(
                          "The number of experts (expert_num) "
                          "must be divisible by moe_topk. "
                          "Got expert_num = %d and moe_topk = %d.",
                          expert_num,
                          moe_topk));
  }

  const int num_moe_inputs = AlignTo16(num_rows * moe_topk);
  const int bytes = num_moe_inputs * sizeof(int);

  CubKeyValueSorter sorter_;
  sorter_.update_num_experts(expert_num);

  const int sorter_ws_size_bytes =
      AlignTo16(sorter_.getWorkspaceSize(moe_topk * num_rows));
  const int sort_tmp_in_out_size = num_moe_inputs * 2 * sizeof(int);

  paddle::Tensor ws_ptr_tensor =
      GetEmptyTensor({bytes + sorter_ws_size_bytes + sort_tmp_in_out_size},
                     paddle::DataType::INT8,
                     place);

  int8_t* ws_ptr = ws_ptr_tensor.data<int8_t>();
  int* source_rows_ = reinterpret_cast<int*>(ws_ptr);
  int8_t* sorter_ws_ptr = reinterpret_cast<int8_t*>(ws_ptr + bytes);
  int* permuted_experts_ =
      reinterpret_cast<int*>(sorter_ws_ptr + sorter_ws_size_bytes);
  int* permuted_rows_ = permuted_experts_ + num_moe_inputs;

  int* expert_for_source_row = top_k_indices->data<int>();

  float* softmax_max_prob = nullptr;
  if (group_moe) {
    paddle::Tensor softmax_max_prob_tensor =
        GetEmptyTensor({num_rows, moe_topk}, paddle::DataType::FLOAT32, place);
    // (TODO: check fill success ?)
    paddle::experimental::fill(softmax_max_prob_tensor, 0.f);
    softmax_max_prob = softmax_max_prob_tensor.data<float>();
  }

  float* softmax_out_;

  const bool is_pow_2 =
      (expert_num != 0) && ((expert_num & (expert_num - 1)) == 0);

  paddle::Tensor softmax_buffer;

  if (!is_pow_2 || expert_num > 256 || group_moe) {
    softmax_buffer = GetEmptyTensor(
        {num_rows * expert_num}, paddle::DataType::FLOAT32, place);
    softmax_out_ = softmax_buffer.data<float>();
  } else {
    softmax_out_ = nullptr;
  }

  topk_gating_softmax_kernelLauncher<float>(gating_output.data<float>(),
                                            top_k_weight->data<float>(),
                                            softmax_out_,
                                            expert_for_source_row,
                                            source_rows_,
                                            softmax_max_prob,
                                            num_rows,
                                            expert_num,
                                            moe_topk,
                                            group_moe,
                                            stream,
                                            topk_only_mode);

  sorter_.run(reinterpret_cast<void*>(sorter_ws_ptr),
              sorter_ws_size_bytes,
              expert_for_source_row,
              permuted_experts_,
              source_rows_,
              permuted_rows_,
              moe_topk * num_rows,
              false,
              stream);


  initialize_moe_routing_kernelLauncher(
      input.data<data_t>(),
      permute_input->data<data_t>(),
      permuted_rows_,
      permute_indices_per_token->data<int32_t>(),
      num_rows,
      num_rows,
      hidden_size,
      moe_topk,
      stream);


  compute_total_rows_before_expert(
      permuted_experts_,
      moe_topk * num_rows,
      expert_num,
      tokens_expert_prefix_sum->data<int64_t>(),
      stream);
}


std::vector<paddle::Tensor> MoeExpertDispatch(
    const paddle::Tensor& input,
    const paddle::Tensor& gating_output,
    const int moe_topk,
    const bool group_moe,
    const bool topk_only_mode) {
  const auto input_type = input.dtype();
  auto place = input.place();
  int token_rows = 0;
  auto input_dims = input.dims();
  auto gating_dims = gating_output.dims();
  const int expert_num = gating_dims[gating_dims.size() - 1];

  if (input_dims.size() == 3) {
    token_rows = input_dims[0] * input_dims[1];
  } else {
    token_rows = input_dims[0];
  }
  const int num_rows = token_rows;
  const int hidden_size = input.dims()[input_dims.size() - 1];

  auto permute_input =
      GetEmptyTensor({moe_topk * num_rows, hidden_size}, input_type, place);
  // correspond to the weighted coefficients of the results from each expert.
  auto top_k_weight =
      GetEmptyTensor({num_rows, moe_topk}, paddle::DataType::FLOAT32, place);
  auto top_k_indices =
      GetEmptyTensor({num_rows, moe_topk}, paddle::DataType::INT32, place);

  auto tokens_expert_prefix_sum =
      GetEmptyTensor({expert_num}, paddle::DataType::INT64, place);
  auto permute_indices_per_token =
      GetEmptyTensor({moe_topk, num_rows}, paddle::DataType::INT32, place);


  switch (input_type) {
    case paddle::DataType::BFLOAT16:
      MoeDispatchKernel<paddle::DataType::BFLOAT16>(input,
                                                    gating_output,
                                                    moe_topk,
                                                    group_moe,
                                                    topk_only_mode,
                                                    num_rows,
                                                    hidden_size,
                                                    expert_num,
                                                    &permute_input,
                                                    &tokens_expert_prefix_sum,
                                                    &permute_indices_per_token,
                                                    &top_k_weight,
                                                    &top_k_indices);
      break;
    case paddle::DataType::FLOAT16:
      MoeDispatchKernel<paddle::DataType::FLOAT16>(input,
                                                   gating_output,
                                                   moe_topk,
                                                   group_moe,
                                                   topk_only_mode,
                                                   num_rows,
                                                   hidden_size,
                                                   expert_num,
                                                   &permute_input,
                                                   &tokens_expert_prefix_sum,
                                                   &permute_indices_per_token,
                                                   &top_k_weight,
                                                   &top_k_indices);
      break;
    default:
      PD_THROW("Unsupported data type for MoeDispatchKernel");
  }
  return {permute_input,
          tokens_expert_prefix_sum,
          permute_indices_per_token,
          top_k_weight,
          top_k_indices};
}


std::vector<std::vector<int64_t>> MoeExpertDispatchInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& gating_output_shape,
    const int moe_topk) {
  int token_rows = -1;

  if (input_shape.size() == 3) {
    token_rows = input_shape[0] * input_shape[1];
  } else {
    token_rows = input_shape[0];
  }
  const int expert_num = gating_output_shape[gating_output_shape.size() - 1];
  const int num_rows = token_rows;
  const int hidden_size = input_shape[input_shape.size() - 1];

  return {{moe_topk * num_rows, hidden_size},
          {expert_num},
          {moe_topk, num_rows},
          {num_rows, moe_topk},
          {num_rows, moe_topk}};
}

std::vector<paddle::DataType> MoeExpertDispatchInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& gating_output_dtype,
    const int moe_topk) {
  return {input_dtype,
          paddle::DataType::INT64,
          paddle::DataType::INT32,
          paddle::DataType::FLOAT32,
          paddle::DataType::INT32};
}


PD_BUILD_OP(moe_expert_dispatch)
    .Inputs({"input", "gating_output"})
    .Outputs({"permute_input",
              "tokens_expert_prefix_sum",
              "permute_indices_per_token",
              "top_k_weight",
              "top_k_indices"})
    .Attrs({"moe_topk:int", "group_moe:bool", "topk_only_mode:bool"})
    .SetKernelFn(PD_KERNEL(MoeExpertDispatch))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeExpertDispatchInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeExpertDispatchInferDtype));
