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
#include "paddle/extension.h"

// #define DEBUG_EAGLE_KERNEL

__global__ void ComputeOrderKernel(const int* seq_lens_this_time,
                                   const int* seq_lens_encoder,
                                   const int* base_model_seq_lens_this_time,
                                   const int* base_model_seq_lens_encoder,
                                   const int* accept_nums,
                                   int* positon_map,
                                   int* output_token_num,
                                   const int bsz,
                                   const int actual_draft_token_num,
                                   const int input_token_num) {
  int in_offset = 0;   // input_offset(long)
  int out_offset = 0;  // output_offset(short)
  if (threadIdx.x == 0) {
    for (int i = 0; i < bsz; ++i) {
      int cur_base_model_seq_lens_this_time = base_model_seq_lens_this_time[i];
      int cur_base_model_seq_lens_encoder = base_model_seq_lens_encoder[i];
      int cur_seq_lens_this_time = seq_lens_this_time[i];
      int accept_num = accept_nums[i];
      int cur_seq_lens_encoder = seq_lens_encoder[i];
#ifdef DEBUG_EAGLE_KERNEL
      printf(
          "batch %d: cur_base_model_seq_lens_this_time%d. "
          "cur_seq_lens_this_time%d, accept_num %d\n",
          i,
          cur_base_model_seq_lens_this_time,
          cur_seq_lens_this_time,
          accept_num);
#endif
      // 1. eagle encoder. Base step=1
      if (cur_seq_lens_encoder > 0) {
#ifdef DEBUG_EAGLE_KERNEL
        printf("batch %d: cur_seq_lens_encoder > 0 \n", i);
#endif
        for (int j = 0; j < cur_seq_lens_encoder; j++) {
          positon_map[in_offset++] = out_offset++;
        }
        // 2. base model encoder. Base step=0
      } else if (cur_base_model_seq_lens_encoder != 0) {
        // 3. New end
      } else if (cur_base_model_seq_lens_this_time != 0 &&
                 cur_seq_lens_this_time == 0) {
#ifdef DEBUG_EAGLE_KERNEL
        printf("batch %d: base=0. draft !=0 \n", i);
#endif

        in_offset += cur_base_model_seq_lens_this_time;
        // 4. stopped
      } else if (cur_base_model_seq_lens_this_time == 0 &&
                 cur_seq_lens_this_time == 0) /* end */ {
      } else {
        if (accept_num <=
            actual_draft_token_num) /*Accept partial draft tokens*/ {
#ifdef DEBUG_EAGLE_KERNEL
          printf("batch %d: accept_num <= actual_draft_token_num \n", i);
#endif
          positon_map[in_offset + accept_num - 1] = out_offset++;
          in_offset += cur_base_model_seq_lens_this_time;
        } else /*Accept all draft tokens*/ {
#ifdef DEBUG_EAGLE_KERNEL
          printf("batch %d: accept_num > actual_draft_token_num \n", i);
#endif
          positon_map[in_offset + accept_num - 2] = out_offset++;
          positon_map[in_offset + accept_num - 1] = out_offset++;
          in_offset += cur_base_model_seq_lens_this_time;
        }
      }
    }
    output_token_num[0] = out_offset;
#ifdef DEBUG_EAGLE_KERNEL
    printf("position map output_token_num%d:\n", output_token_num[0]);
    for (int i = 0; i < output_token_num[0]; i++) {
      printf("%d ", positon_map[i]);
    }
    printf("\n");
#endif
  }
}

template <typename T, int VecSize>
__global__ void rebuildHiddenStatesKernel(const T* input,
                                          const int* position_map,
                                          T* out,
                                          const int dim_embed,
                                          const int elem_cnt) {
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;

  int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int elem_idx = global_thread_idx * VecSize; elem_idx < elem_cnt;
       elem_idx += blockDim.x * gridDim.x * VecSize) {
    int ori_token_idx = elem_idx / dim_embed;
    int token_idx = position_map[ori_token_idx];
    if (token_idx >= 0) {
      int offset = elem_idx % dim_embed;
      if (token_idx == 0) {
      }
      Load<T, VecSize>(input + ori_token_idx * dim_embed + offset, &src_vec);
      Store<T, VecSize>(src_vec, out + token_idx * dim_embed + offset);
    }
  }
}


template <paddle::DataType D>
std::vector<paddle::Tensor> DispatchDtype(
    const paddle::Tensor& input,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& accept_nums,
    const paddle::Tensor& base_model_seq_lens_this_time,
    const paddle::Tensor& base_model_seq_lens_encoder,
    const int actual_draft_token_num) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto input_token_num = input.shape()[0];

  // auto output_token_num = padding_offset.shape()[0];
  auto dim_embed = input.shape()[1];

  int bsz = seq_lens_this_time.shape()[0];

  auto position_map = paddle::full(
      {input_token_num}, -1, seq_lens_this_time.dtype(), input.place());
  auto output_token_num = paddle::full(
      {1}, 0, seq_lens_this_time.dtype(), seq_lens_this_time.place());
  ComputeOrderKernel<<<1, 1>>>(seq_lens_this_time.data<int>(),
                               seq_lens_encoder.data<int>(),
                               base_model_seq_lens_this_time.data<int>(),
                               base_model_seq_lens_encoder.data<int>(),
                               accept_nums.data<int>(),
                               position_map.data<int>(),
                               output_token_num.data<int>(),
                               bsz,
                               actual_draft_token_num,
                               input_token_num);

  int output_token_num_cpu =
      output_token_num.copy_to(paddle::CPUPlace(), false).data<int>()[0];

  auto out = paddle::full(
      {output_token_num_cpu, dim_embed}, -1, input.dtype(), input.place());

  constexpr int packSize = VEC_16B / (sizeof(DataType_));
  int elem_cnt = input_token_num * dim_embed;

  assert(elem_cnt % packSize == 0);

  int pack_num = elem_cnt / packSize;

  int grid_size = 1;

  GetNumBlocks(pack_num, &grid_size);

  constexpr int thread_per_block = 128;

  rebuildHiddenStatesKernel<DataType_, packSize>
      <<<grid_size, thread_per_block>>>(
          reinterpret_cast<const DataType_*>(input.data<data_t>()),
          position_map.data<int>(),
          reinterpret_cast<DataType_*>(out.data<data_t>()),
          dim_embed,
          elem_cnt);

  return {out};
}


std::vector<paddle::Tensor> EagleGetHiddenStates(
    const paddle::Tensor& input,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& accept_nums,
    const paddle::Tensor& base_model_seq_lens_this_time,
    const paddle::Tensor& base_model_seq_lens_encoder,
    const int actual_draft_token_num) {
  switch (input.dtype()) {
    case paddle::DataType::FLOAT16: {
      return DispatchDtype<paddle::DataType::FLOAT16>(
          input,
          seq_lens_this_time,
          seq_lens_encoder,
          seq_lens_decoder,
          stop_flags,
          accept_nums,
          base_model_seq_lens_this_time,
          base_model_seq_lens_encoder,
          actual_draft_token_num);
    }
    case paddle::DataType::BFLOAT16: {
      return DispatchDtype<paddle::DataType::BFLOAT16>(
          input,
          seq_lens_this_time,
          seq_lens_encoder,
          seq_lens_decoder,
          stop_flags,
          accept_nums,
          base_model_seq_lens_this_time,
          base_model_seq_lens_encoder,
          actual_draft_token_num);
    }
    default: {
      PD_THROW("Not support this data type");
    }
  }
}


PD_BUILD_OP(eagle_get_base_model_hidden_states)
    .Inputs({"input",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "stop_flags",
             "accept_nums",
             "base_model_seq_lens_this_time",
             "base_model_seq_lens_encoder"})
    .Attrs({"actual_draft_token_num: int"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(EagleGetHiddenStates));
