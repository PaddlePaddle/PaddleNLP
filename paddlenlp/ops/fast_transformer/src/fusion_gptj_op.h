/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "fastertransformer/gptj.h"
#include "fastertransformer/open_decoder.h"
#include "fastertransformer/utils/common.h"

#ifdef PADDLE_ON_INFERENCE
#include "paddle/include/experimental/ext_all.h"
#else
#include "paddle/extension.h"
#endif


std::vector<paddle::Tensor> GPTJCUDAForward(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& start_length,
    const paddle::Tensor& word_embedding,
    const std::vector<paddle::Tensor>& self_ln_weight,
    const std::vector<paddle::Tensor>& self_ln_bias,
    const std::vector<paddle::Tensor>& self_q_weight,
    const std::vector<paddle::Tensor>& self_out_weight,
    const std::vector<paddle::Tensor>& ffn_inter_weight,
    const std::vector<paddle::Tensor>& ffn_inter_bias,
    const std::vector<paddle::Tensor>& ffn_out_weight,
    const std::vector<paddle::Tensor>& ffn_out_bias,
    const paddle::Tensor& decoder_ln_weight,
    const paddle::Tensor& decoder_ln_bias,
    const paddle::Tensor& emb_weight,
    const paddle::Tensor& emb_bias,
    paddle::Tensor& output_ids,
    const int topk,
    const float topp,
    const int max_len,
    const int n_head,
    const int size_per_head,
    const int num_layer,
    const int bos_id,
    const int eos_id,
    const float temperature,
    const int rotary_embedding_dim,
    const float repetition_penalty,
    const int min_length,
    const bool use_fp16,
    const int tensor_para_size,
    const int layer_para_size,
    const int layer_para_batch_size);
