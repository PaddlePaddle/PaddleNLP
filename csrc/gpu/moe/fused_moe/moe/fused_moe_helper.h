
/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "cutlass_kernels/moe_gemm/fused_moe_gemm_kernels.h"
#include "moe/fused_moe_op.h"

using namespace phi;

template <typename T, int VecSize>
__global__ void moe_token_type_ids_kernel(T *gating_output,
                                          const int *moe_token_type_ids_out,
                                          const int num_rows,
                                          const int num_experts,
                                          const int k) {
  const int moe_token_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (moe_token_index >= num_rows) {
    return;
  }

  gating_output[moe_token_index * 2] =
      gating_output[moe_token_index * 2] +
      (moe_token_type_ids_out[moe_token_index]) * -1e10;
  gating_output[moe_token_index * 2 + 1] =
      gating_output[moe_token_index * 2 + 1] +
      (1 - moe_token_type_ids_out[moe_token_index]) * -1e10;
}

template <typename T>
void moe_token_type_ids_kernelLauncher(T *gating_output,
                                       const int *moe_token_type_ids_out,
                                       const int num_rows,
                                       const int num_experts,
                                       const int k,
                                       cudaStream_t stream) {
  const int blocks = num_rows * k / 512 + 1;
  const int threads = 512;
  moe_token_type_ids_kernel<T, 1><<<blocks, 512, 0, stream>>>(
      gating_output, moe_token_type_ids_out, num_rows, num_experts, k);
}


template <typename T, typename NvType>
class MoeHelper {
public:
  MoeHelper(const std::string gemm_method,
            MoeGemmRunner<NvType, NvType> *fp16_moe_gemm_runner,
            MoeGemmRunner<NvType, uint8_t> *int8_moe_gemm_runner,
            MoeGemmRunner<NvType, cutlass::uint4b_t> *int4_moe_gemm_runner,
            int layernum = 0)
      : gemm_method_(gemm_method),
        fp16_moe_gemm_runner_(fp16_moe_gemm_runner),
        int8_moe_gemm_runner_(int8_moe_gemm_runner),
        int4_moe_gemm_runner_(int4_moe_gemm_runner),
        layernum_(layernum) {}

  // --------      getWorkspaceSize      -------- //
  template <typename KeyT>
  size_t getWorkspaceSize(const int64_t num_rows,
                          const int64_t hidden_size,
                          const int64_t inter_size,
                          const int64_t num_experts,
                          const int64_t k) {
    const size_t buf_size = AlignTo16(k * num_rows * hidden_size);
    const size_t interbuf_size = AlignTo16(k * num_rows * inter_size);
    const size_t padded_experts = AlignTo16(num_experts);
    const size_t num_moe_inputs = AlignTo16(k * num_rows);
    // softmax output, permuted_rows and permuted_experts have moved to outside
    // of moe kernel, allocate them in Encoder or Decoder before invoking
    // FfnLayer forward.
    size_t total_ws_bytes =
        5 * num_moe_inputs *
        sizeof(int);  // source_rows_, permuted_rows_, permuted_experts_
    total_ws_bytes += buf_size * sizeof(KeyT);  // permuted_data
    total_ws_bytes +=
        padded_experts * sizeof(int64_t);  // Hold total_rows_before_expert_

    const size_t bytes_for_fc1_result = interbuf_size * sizeof(KeyT);
    const size_t sorter_ws_size_bytes =
        AlignTo16(sorter_.getWorkspaceSize(num_rows));
    sorter_.update_num_experts(num_experts);

    int64_t bytes_for_intermediate_and_sorting = bytes_for_fc1_result;
    if (sorter_ws_size_bytes > bytes_for_fc1_result) {
      int64_t remaining_bytes =
          AlignTo16(sorter_ws_size_bytes - bytes_for_fc1_result);
      bytes_for_intermediate_and_sorting += remaining_bytes;
    }

    total_ws_bytes +=
        bytes_for_intermediate_and_sorting;  // intermediate (fc1) output + cub
                                             // sorting workspace

    int64_t num_softmax_outs = 0;
    const bool is_pow_2 =
        (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256) {
      num_softmax_outs = AlignTo16(num_rows * num_experts);
    }

    total_ws_bytes += num_softmax_outs * sizeof(float);

    return total_ws_bytes;
  }

  void ComputeFFN(const paddle::Tensor *input,
                  const paddle::Tensor *gate_weight,
                  const paddle::Tensor *ffn1_weight,
                  const paddle::Tensor *ffn1_scale,
                  const paddle::Tensor *ffn1_bias,
                  const paddle::Tensor *ffn2_weight,
                  const paddle::Tensor *ffn2_scale,
                  const paddle::Tensor *ffn2_bias,
                  const paddle::Tensor *moe_token_type_ids,
                  const int moe_topk,
                  const bool group_moe,
                  const bool norm_topk_prob,
                  const float routed_scaling_factor,
                  const std::string moe_type,
                  paddle::Tensor *output) {
    auto *input_activations = input->data<T>();
    auto *gating_weights = gate_weight->data<float>();
    const T *fc1_expert_biases = ffn1_bias ? ffn1_bias->data<T>() : nullptr;
    const T *fc2_expert_biases = ffn2_bias ? ffn2_bias->data<T>() : nullptr;

    auto *output_ = output->data<T>();
    auto stream = input->stream();
    auto place = input->place();
    auto input_type = input->dtype();

    auto input_dims = input->dims();
    auto ffn1_dims = ffn1_weight->dims();
    int64_t token_num = 0;
    if (input_dims.size() == 3) {
      token_num = input_dims[0] * input_dims[1];
    } else {
      token_num = input_dims[0];
    }
    const int64_t num_rows = token_num;

    const int64_t hidden_size = ffn1_dims[1];
    int64_t inter_dim = 0;
    if (moe_type == "qkv") {
      inter_dim = ffn1_dims[2] * ffn1_dims[3] * ffn1_dims[4];
    } else {
      inter_dim = ffn1_dims[2];
    }

    if (gemm_method_ == "weight_only_int4") {
      inter_dim = inter_dim * 2;
    }

    const int64_t inter_size = inter_dim;
    const int64_t num_experts = ffn1_dims[0];
    const int64_t k = moe_topk;


    int64_t bytes =
        getWorkspaceSize<T>(num_rows, hidden_size, inter_size, num_experts, k);

    // Pointers
    int *expert_for_source_row;
    int *source_rows_;
    int *permuted_rows_;
    int *permuted_experts_;
    int *expanded_source_row_to_expanded_dest_row;

    T *permuted_data_;
    int64_t *total_rows_before_expert_;
    T *fc1_result_;
    float *softmax_out_;

    paddle::Tensor ws_ptr_tensor =
        GetEmptyTensor({bytes}, paddle::DataType::INT8, place);
    int8_t *ws_ptr = ws_ptr_tensor.data<int8_t>();

    const int64_t buf_size = AlignTo16(k * num_rows * hidden_size);
    const int64_t interbuf_size = AlignTo16(k * num_rows * inter_size);
    const int64_t padded_experts = AlignTo16(num_experts);
    const int64_t num_moe_inputs = AlignTo16(k * num_rows);

    expert_for_source_row = reinterpret_cast<int *>(ws_ptr);
    source_rows_ = expert_for_source_row + num_moe_inputs;
    permuted_rows_ = source_rows_ + num_moe_inputs;
    permuted_experts_ = permuted_rows_ + num_moe_inputs;
    expanded_source_row_to_expanded_dest_row =
        permuted_experts_ + num_moe_inputs;
    permuted_data_ = reinterpret_cast<T *>(
        expanded_source_row_to_expanded_dest_row + num_moe_inputs);
    total_rows_before_expert_ =
        reinterpret_cast<int64_t *>(permuted_data_ + buf_size);
    fc1_result_ =
        reinterpret_cast<T *>(total_rows_before_expert_ + padded_experts);

    const bool is_pow_2 =
        (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256) {
      softmax_out_ = reinterpret_cast<float *>(fc1_result_ + interbuf_size);
    } else {
      softmax_out_ = nullptr;
    }

    paddle::Tensor expert_scales_float_tensor =
        GetEmptyTensor({num_rows, moe_topk}, paddle::DataType::FLOAT32, place);
    float *expert_scales_float = expert_scales_float_tensor.data<float>();

    float *softmax_max_prob = nullptr;
    if (group_moe) {
      paddle::Tensor softmax_max_prob_tensor = GetEmptyTensor(
          {num_rows, moe_topk}, paddle::DataType::FLOAT32, place);
      // (TODO: check fill success ?)
      paddle::experimental::fill(softmax_max_prob_tensor, 0.f);
      softmax_max_prob = softmax_max_prob_tensor.data<float>();
    }

    paddle::Tensor fc1_out_tensor =
        GetEmptyTensor({num_rows * k, inter_size}, input_type, place);
    T *fc1_out = fc1_out_tensor.data<T>();

    auto input_cast_tensor =
        paddle::experimental::cast(*input, paddle::DataType::FLOAT32);
    auto gate_tensor =
        paddle::experimental::matmul(input_cast_tensor, *gate_weight);
    float *gating_output = gate_tensor.data<float>();

    if (moe_token_type_ids) {
      auto *moe_token_type_ids_out = moe_token_type_ids->data<int>();
      moe_token_type_ids_kernelLauncher<float>(gating_output,
                                               moe_token_type_ids_out,
                                               num_rows,
                                               num_experts,
                                               k,
                                               stream);
    }

    topk_gating_softmax_kernelLauncher<float>(gating_output,
                                              expert_scales_float,
                                              softmax_out_,
                                              expert_for_source_row,
                                              source_rows_,
                                              softmax_max_prob,
                                              num_rows,
                                              num_experts,
                                              k,
                                              group_moe,
                                              stream);

    const int64_t sorter_ws_size_bytes =
        AlignTo16(sorter_.getWorkspaceSize(int64_t(k * num_rows)));

    sorter_.run(fc1_result_,
                sorter_ws_size_bytes,
                expert_for_source_row,
                permuted_experts_,
                source_rows_,
                permuted_rows_,
                k * num_rows,
                false,
                stream);

    initialize_moe_routing_kernelLauncher(
        input_activations,
        permuted_data_,
        permuted_rows_,
        expanded_source_row_to_expanded_dest_row,
        num_rows,
        num_rows,
        hidden_size,
        k,
        stream);

    const int64_t expanded_active_expert_rows = k * num_rows;

    compute_total_rows_before_expert(permuted_experts_,
                                     expanded_active_expert_rows,
                                     num_experts,
                                     total_rows_before_expert_,
                                     stream);

    if (gemm_method_ == "weight_only_int8") {
      int8_moe_gemm_runner_->moe_gemm_bias_act(
          reinterpret_cast<NvType *>(permuted_data_),
          reinterpret_cast<const uint8_t *>(ffn1_weight->data<int8_t>()),
          reinterpret_cast<const NvType *>(ffn1_scale->data<T>()),
          reinterpret_cast<const NvType *>(fc1_expert_biases),
          reinterpret_cast<NvType *>(fc1_out),
          total_rows_before_expert_,
          expanded_active_expert_rows,
          inter_size,
          hidden_size,
          num_experts,
          "none",
          stream);
    } else if (gemm_method_ == "weight_only_int4") {
      int4_moe_gemm_runner_->moe_gemm_bias_act(
          reinterpret_cast<NvType *>(permuted_data_),
          reinterpret_cast<const cutlass::uint4b_t *>(
              ffn1_weight->data<int8_t>()),
          reinterpret_cast<const NvType *>(ffn1_scale->data<T>()),
          reinterpret_cast<const NvType *>(fc1_expert_biases),
          reinterpret_cast<NvType *>(fc1_out),
          total_rows_before_expert_,
          expanded_active_expert_rows,
          inter_size,
          hidden_size,
          num_experts,
          "none",
          stream);
    } else {
      fp16_moe_gemm_runner_->moe_gemm_bias_act(
          reinterpret_cast<NvType *>(permuted_data_),
          reinterpret_cast<const NvType *>(ffn1_weight->data<T>()),
          nullptr,
          reinterpret_cast<const NvType *>(fc1_expert_biases),
          reinterpret_cast<NvType *>(fc1_out),
          total_rows_before_expert_,
          expanded_active_expert_rows,
          inter_size,
          hidden_size,
          num_experts,
          "none",
          stream);
    }

    if (moe_type == "ffn") {
      auto act_out_tensor =
          paddle::experimental::swiglu(fc1_out_tensor, nullptr);
      auto act_out = act_out_tensor.data<T>();

      paddle::Tensor fc2_output_tensor =
          GetEmptyTensor({k * num_rows, hidden_size}, input_type, place);
      T *fc2_result = fc2_output_tensor.data<T>();

      if (gemm_method_ == "weight_only_int8") {
        int8_moe_gemm_runner_->moe_gemm(
            reinterpret_cast<NvType *>(act_out),
            reinterpret_cast<const uint8_t *>(ffn2_weight->data<int8_t>()),
            reinterpret_cast<const NvType *>(ffn2_scale->data<T>()),
            reinterpret_cast<NvType *>(fc2_result),
            total_rows_before_expert_,
            expanded_active_expert_rows,
            hidden_size,
            inter_size / 2,
            num_experts,
            stream);
      } else if (gemm_method_ == "weight_only_int4") {
        int4_moe_gemm_runner_->moe_gemm(
            reinterpret_cast<NvType *>(act_out),
            reinterpret_cast<const cutlass::uint4b_t *>(
                ffn2_weight->data<int8_t>()),
            reinterpret_cast<const NvType *>(ffn2_scale->data<T>()),
            reinterpret_cast<NvType *>(fc2_result),
            total_rows_before_expert_,
            expanded_active_expert_rows,
            hidden_size,
            inter_size / 2,
            num_experts,
            stream);
      } else {
        fp16_moe_gemm_runner_->moe_gemm(
            reinterpret_cast<NvType *>(act_out),
            reinterpret_cast<const NvType *>(ffn2_weight->data<T>()),
            nullptr,
            reinterpret_cast<NvType *>(fc2_result),
            total_rows_before_expert_,
            expanded_active_expert_rows,
            hidden_size,
            inter_size / 2,
            num_experts,
            stream);
      }

      finalize_moe_routing_kernelLauncher(
          fc2_result,
          output_,
          fc2_expert_biases,
          reinterpret_cast<float *>(expert_scales_float),
          expanded_source_row_to_expanded_dest_row,
          expert_for_source_row,
          num_rows,
          hidden_size,
          k,
          static_cast<int>(1),
          norm_topk_prob,
          routed_scaling_factor,
          stream);
    } else {
      finalize_moe_routing_kernelLauncher(
          // fc2_result,
          fc1_out,
          output_,
          fc1_expert_biases,  // fc2_expert_biases,
          reinterpret_cast<float *>(expert_scales_float),
          expanded_source_row_to_expanded_dest_row,
          expert_for_source_row,
          num_rows,
          inter_size,
          k,
          static_cast<int>(0),
          norm_topk_prob,
          routed_scaling_factor,
          stream);
    }
  }

private:
  std::string gemm_method_;
  MoeGemmRunner<NvType, NvType> *fp16_moe_gemm_runner_;
  MoeGemmRunner<NvType, uint8_t> *int8_moe_gemm_runner_;
  MoeGemmRunner<NvType, cutlass::uint4b_t> *int4_moe_gemm_runner_;
  int layernum_;
  CubKeyValueSorter sorter_;
};
