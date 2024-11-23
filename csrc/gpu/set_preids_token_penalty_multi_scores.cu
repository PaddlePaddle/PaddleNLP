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

#include "helper.h"

template<typename T>
__global__ void set_preids_token_penalty_multi_scores_kernel(const bool *stop_flags,
                                 int64_t *pre_ids,
                                 const int64_t *input_ids, 
                                 const int *seq_lens_encoder, 
                                 const int *seq_lens_decoder, 
                                 const int64_t *step_idx, 
                                 const T *penalty_scores,
                                 const T *frequency_score,
                                 const T *presence_score,
                                 const float *temperatures,
                                 const int64_t *cur_len,
                                 const int64_t *min_len,
                                 const int64_t *eos_token_id,
                                 const int64_t *bad_words_list,
                                 int *repeat_times,
                                 T *logits,
                                 const int64_t bs,
                                 const int64_t length,
                                 const int64_t end_length,
                                 const int64_t length_id,
                                 const int64_t bad_words_length,
                                 const int64_t length_input_ids) {
    int bi = blockIdx.x;
    T *logits_now = logits + bi * length;
    int tid = threadIdx.x;

    if (tid < bs && !stop_flags[tid]) {
        int64_t *pre_ids_now = pre_ids + tid * length;
        const int64_t *input_ids_now = input_ids + tid * length_input_ids;
        const int seq_len_dec = seq_lens_decoder[tid];
        const int seq_len_enc = seq_lens_encoder[tid];
        if (seq_len_dec == 0 && seq_len_enc == 0) return; // stoped

        const int step_idx_now = step_idx[bi];
        if (tid == 0 && step_idx_now >= 0) {
            if (seq_len_enc > 0) { // encoder, get last token accord to seq_lens_encoder
                pre_ids_now[step_idx_now] = input_ids_now[seq_len_enc - 1];
            } else { // decoedr, get first token
                pre_ids_now[step_idx_now] = input_ids_now[0];
            }
        }
    }
    __syncthreads();
    // min_length process
    if (bi < bs) {
        if (cur_len[bi] < min_len[bi]) {
            if (tid < end_length) {
                logits_now[eos_token_id[tid]] = -1e10;
            }
        }
    }
    // update repeat_times
    int *repeat_times_now = repeat_times + bi * length;
    const int64_t *pre_ids_now = pre_ids + bi * length_id;
    for (int i = tid; i < length_id; i += blockDim.x) {
        int64_t id = pre_ids_now[i];
        if (id < 0) break;
        atomicAdd(&repeat_times_now[id], 1);
    }
    __syncthreads();
    // penalty_scores process
    float alpha = static_cast<float>(penalty_scores[bi]);
    float beta = static_cast<float>(frequency_score[bi]);
    float gamma = static_cast<float>(presence_score[bi]);
    for (int i = tid; i < length; i += blockDim.x) {
        int times = repeat_times_now[i];
        float logit_now = static_cast<float>(logits_now[i]);
        if (times != 0) {
            logit_now = logit_now < 0 ? logit_now * alpha : logit_now / alpha;
            logit_now = logit_now - times * beta - gamma;
        }
        logits_now[i] = static_cast<T>(logit_now / temperatures[bi]);
    }
    __syncthreads();
    // bad_words process
    for (int i = tid; i < bad_words_length; i += blockDim.x) {
        const int64_t bad_words_token_id = bad_words_list[i];
        if (bad_words_token_id >= length || bad_words_token_id < 0) continue;
        logits_now[bad_words_token_id] = -1e10;
    }
}

template <paddle::DataType D>
void set_preids_token_penalty_multi_scores(const paddle::Tensor& pre_ids,
                                                  const paddle::Tensor& input_ids,
                                                  const paddle::Tensor& seq_lens_encoder,
                                                  const paddle::Tensor& seq_lens_decoder,
                                                  const paddle::Tensor& step_idx,
                                                  const paddle::Tensor& stop_flags,
                                                  const paddle::Tensor& logits,
                                                  const paddle::Tensor& penalty_scores,
                                                  const paddle::Tensor& frequency_score,
                                                  const paddle::Tensor& presence_score,
                                                  const paddle::Tensor& temperatures,
                                                  const paddle::Tensor& bad_tokens,
                                                  const paddle::Tensor& cur_len,
                                                  const paddle::Tensor& min_len,
                                                  const paddle::Tensor& eos_token_id) {

    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    auto cu_stream = logits.stream();
    std::vector<int64_t> shape = logits.shape();
    auto repeat_times = paddle::full(shape, 0, paddle::DataType::INT32, pre_ids.place());
    int64_t bs = shape[0];
    int64_t length = shape[1];
    int64_t length_id = pre_ids.shape()[1];
    int64_t length_bad_words = bad_tokens.shape()[0];
    int64_t length_input_ids = input_ids.shape()[1];

    int64_t end_length = eos_token_id.shape()[0];

    set_preids_token_penalty_multi_scores_kernel<DataType_><<<bs, 1024, 0, cu_stream>>>(
        stop_flags.data<bool>(), 
        const_cast<int64_t*>(pre_ids.data<int64_t>()),
        input_ids.data<int64_t>(), 
        seq_lens_encoder.data<int>(),
        seq_lens_decoder.data<int>(),
        step_idx.data<int64_t>(), 
        reinterpret_cast<DataType_*>(const_cast<data_t*>(penalty_scores.data<data_t>())),
        reinterpret_cast<DataType_*>(const_cast<data_t*>(frequency_score.data<data_t>())),
        reinterpret_cast<DataType_*>(const_cast<data_t*>(presence_score.data<data_t>())),
        temperatures.data<float>(),
        cur_len.data<int64_t>(),
        min_len.data<int64_t>(),
        eos_token_id.data<int64_t>(),
        bad_tokens.data<int64_t>(),
        repeat_times.data<int>(), 
        reinterpret_cast<DataType_*>(const_cast<data_t*>(logits.data<data_t>())),
        bs,
        length,
        end_length,
        length_id,
        length_bad_words,
        length_input_ids
    );
}

void SetPreidsTokenPenaltyMultiScores(const paddle::Tensor& pre_ids,
                                      const paddle::Tensor& input_ids,
                                      const paddle::Tensor& seq_lens_encoder,
                                      const paddle::Tensor& seq_lens_decoder,
                                      const paddle::Tensor& step_idx,
                                      const paddle::Tensor& stop_flags,
                                      const paddle::Tensor& logits,
                                      const paddle::Tensor& penalty_scores,
                                      const paddle::Tensor& frequency_scores,
                                      const paddle::Tensor& presence_scores,
                                      const paddle::Tensor& temperatures,
                                      const paddle::Tensor& bad_tokens,
                                      const paddle::Tensor& cur_len,
                                      const paddle::Tensor& min_len,
                                      const paddle::Tensor& eos_token_id) {

    switch (logits.type()) {
        case paddle::DataType::BFLOAT16: {
            return set_preids_token_penalty_multi_scores<paddle::DataType::BFLOAT16>(
                pre_ids,
                input_ids,
                seq_lens_encoder,
                seq_lens_decoder,
                step_idx,
                stop_flags,
                logits,
                penalty_scores,
                frequency_scores,
                presence_scores,
                temperatures,
                bad_tokens,
                cur_len,
                min_len,
                eos_token_id
            );
        }
        case paddle::DataType::FLOAT16: {
            return set_preids_token_penalty_multi_scores<paddle::DataType::FLOAT16>(
                pre_ids,
                input_ids,
                seq_lens_encoder,
                seq_lens_decoder,
                step_idx,
                stop_flags,
                logits,
                penalty_scores,
                frequency_scores,
                presence_scores,
                temperatures,
                bad_tokens,
                cur_len,
                min_len,
                eos_token_id
            );
        }
        case paddle::DataType::FLOAT32: {
            return set_preids_token_penalty_multi_scores<paddle::DataType::FLOAT32>(
                pre_ids,
                input_ids,
                seq_lens_encoder,
                seq_lens_decoder,
                step_idx,
                stop_flags,
                logits,
                penalty_scores,
                frequency_scores,
                presence_scores,
                temperatures,
                bad_tokens,
                cur_len,
                min_len,
                eos_token_id
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16, bfloat16 and float32 are supported. ");
            break;
        }
    }
}

PD_BUILD_OP(set_preids_token_penalty_multi_scores)
    .Inputs({"pre_ids", 
             "input_ids",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx",
             "stop_flags",
             "logits", 
             "penalty_scores", 
             "frequency_scores", 
             "presence_scores", 
             "temperatures", 
             "bad_tokens", 
             "cur_len", 
             "min_len", 
             "eos_token_id"})
    .Outputs({"logits_out", "pre_ids_out"})
    .SetInplaceMap({{"logits", "logits_out"}, {"pre_ids", "pre_ids_out"}})
    .SetKernelFn(PD_KERNEL(SetPreidsTokenPenaltyMultiScores));