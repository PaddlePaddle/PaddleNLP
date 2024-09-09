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

#include <paddle/extension.h>
#include <vector>

template <typename T>
void min_length_logits_process(T* logits,
                               const int64_t* cur_len,
                               const int64_t* min_len,
                               const int64_t* eos_token_id,
                               const int64_t bs,
                               const int64_t length,
                               const int64_t end_length) {
  for (int bi = 0; bi < bs; ++bi) {
    if (cur_len[bi] < 0) {
      continue;
    }
    if (cur_len[bi] < min_len[bi]) {
      for (int i = 0; i < end_length; ++i) {
        logits[bi * length + eos_token_id[i]] = -1e10;
      }
    }
  }
}


void update_repeat_times(const int64_t* pre_ids,
                         const int64_t* cur_len,
                         int* repeat_times,
                         const int64_t bs,
                         const int64_t length,
                         const int64_t length_id) {
  for (int bi = 0; bi < bs; ++bi) {
    if (cur_len[bi] < 0) {
      continue;
    }
    const int64_t* pre_ids_now = pre_ids + bi * length_id;
    int* repeat_times_now = repeat_times + bi * length;
    for (int i = 0; i < length_id; i++) {
      int64_t id = pre_ids_now[i];
      if (id < 0) {
        break;
      }
      repeat_times_now[id] += 1;
    }
  }
}

template <typename T>
void update_value_by_repeat_times(const int* repeat_times,
                                  const T* penalty_scores,
                                  const T* frequency_score,
                                  const T* presence_score,
                                  T* logits,
                                  const int64_t bs,
                                  const int64_t length) {
  for (int bi = 0; bi < bs; ++bi) {
    T* logits_now = logits + bi * length;
    const int* repeat_times_now = repeat_times + bi * length;
    float alpha = static_cast<float>(penalty_scores[bi]);
    float beta = static_cast<float>(frequency_score[bi]);
    float gamma = static_cast<float>(presence_score[bi]);
    for (int i = 0; i < length; ++i) {
      int times = repeat_times_now[i];
      if (times == 0) continue;
      float logit_now = static_cast<float>(logits_now[i]);
      logit_now = logit_now < 0 ? logit_now * alpha : logit_now / alpha;
      logits_now[i] = static_cast<T>(logit_now - times * beta - gamma);
    }
  }
}
template <paddle::DataType D>
std::vector<paddle::Tensor> token_penalty_multi_scores_kernel(
    const paddle::Tensor& pre_ids,
    const paddle::Tensor& logits,
    const paddle::Tensor& penalty_scores,
    const paddle::Tensor& frequency_score,
    const paddle::Tensor& presence_score,
    const paddle::Tensor& cur_len,
    const paddle::Tensor& min_len,
    const paddle::Tensor& eos_token_id) {
  std::vector<int64_t> shape = logits.shape();
  auto repeat_times =
      paddle::full(shape, 0, paddle::DataType::INT32, pre_ids.place());
  int64_t bs = shape[0];
  int64_t length = shape[1];
  int64_t length_id = pre_ids.shape()[1];
  auto logits_out = logits.copy_to(logits.place(), false);
  int64_t end_length = eos_token_id.shape()[0];

  min_length_logits_process(const_cast<float*>(logits_out.data<float>()),
                            cur_len.data<int64_t>(),
                            min_len.data<int64_t>(),
                            eos_token_id.data<int64_t>(),
                            bs,
                            length,
                            end_length);
  update_repeat_times(pre_ids.data<int64_t>(),
                      cur_len.data<int64_t>(),
                      repeat_times.data<int>(),
                      bs,
                      length,
                      length_id);
  update_value_by_repeat_times(
      repeat_times.data<int>(),
      const_cast<float*>(penalty_scores.data<float>()),
      const_cast<float*>(frequency_score.data<float>()),
      const_cast<float*>(presence_score.data<float>()),
      const_cast<float*>(logits_out.data<float>()),
      bs,
      length);
  return {logits_out};
}

std::vector<paddle::Tensor> TokenPenaltyMultiScores(
    const paddle::Tensor& pre_ids,
    const paddle::Tensor& logits,
    const paddle::Tensor& penalty_scores,
    const paddle::Tensor& frequency_scores,
    const paddle::Tensor& presence_scores,
    const paddle::Tensor& cur_len,
    const paddle::Tensor& min_len,
    const paddle::Tensor& eos_token_id) {
  switch (logits.type()) {
    case paddle::DataType::FLOAT32: {
      return token_penalty_multi_scores_kernel<paddle::DataType::FLOAT32>(
          pre_ids,
          logits,
          penalty_scores,
          frequency_scores,
          presence_scores,
          cur_len,
          min_len,
          eos_token_id);
    }
    default: {
      PD_THROW(
          "NOT supported data type. "
          "Only float32 are supported. ");
      break;
    }
  }
}

std::vector<std::vector<int64_t>> TokenPenaltyMultiScoresInferShape(
    const std::vector<int64_t>& pre_ids_shape,
    const std::vector<int64_t>& logits_shape,
    const std::vector<int64_t>& penalty_scores_shape,
    const std::vector<int64_t>& frequency_scores_shape,
    const std::vector<int64_t>& presence_scores_shape,
    const std::vector<int64_t>& cur_len_shape,
    const std::vector<int64_t>& min_len_shape,
    const std::vector<int64_t>& eos_token_id_shape) {
  return {logits_shape};
}

std::vector<paddle::DataType> TokenPenaltyMultiScoresInferDtype(
    const paddle::DataType& pre_ids_dtype,
    const paddle::DataType& logits_dtype,
    const paddle::DataType& penalty_scores_dtype,
    const paddle::DataType& frequency_scores_dtype,
    const paddle::DataType& presence_scores_dtype,
    const paddle::DataType& cur_len_dtype,
    const paddle::DataType& min_len_dtype,
    const paddle::DataType& eos_token_id_dtype) {
  return {logits_dtype};
}
PD_BUILD_OP(get_token_penalty_multi_scores)
    .Inputs({"pre_ids",
             "logits",
             "penalty_scores",
             "frequency_scores",
             "presence_scores",
             "cur_len",
             "min_len",
             "eos_token_id"})
    .Outputs({"logits_out"})
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScores))
    .SetInferShapeFn(PD_INFER_SHAPE(TokenPenaltyMultiScoresInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TokenPenaltyMultiScoresInferDtype));