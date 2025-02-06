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

__global__ void computeOrderKernel(const int* last_seq_lens_this_time,
                                   const int* seq_lens_this_time,
                                   const int64_t* step_idx,
                                   int* src_map,
                                   int* output_token_num,
                                   int bsz) {
  int in_offset = 0;
  int out_offset = 0;
  if (threadIdx.x == 0) {
    for (int i = 0; i < bsz; ++i) {
      int cur_seq_lens_this_time = seq_lens_this_time[i];
      int cur_last_seq_lens_this_time = last_seq_lens_this_time[i];
#ifdef DEBUG_EAGLE_KERNEL
      printf(
          "batch %d: cur_seq_lens_this_time:%d. "
          "cur_last_seq_lens_this_time:%d\n",
          i,
          cur_seq_lens_this_time,
          cur_last_seq_lens_this_time);
#endif
      // 1. encoder
      if (step_idx[i] == 1 && cur_seq_lens_this_time > 0) {
#ifdef DEBUG_EAGLE_KERNEL
        printf("batch %d last_step is encoder \n", i);
#endif
        in_offset += 1;
        src_map[out_offset++] = in_offset - 1;
#ifdef DEBUG_EAGLE_KERNEL
        printf("batch %d finish. src_map[%d]=%d \n",
               i,
               out_offset - 1,
               in_offset - 1);
#endif
        // 2. decoder
      } else if (cur_seq_lens_this_time > 0) /* =1 */ {
#ifdef DEBUG_EAGLE_KERNEL
        printf("batch %d is decoder\n", i);
#endif
        in_offset += cur_last_seq_lens_this_time;
        src_map[out_offset++] = in_offset - 1;
        // 3. stop
      } else {
        // first token end
        if (step_idx[i] == 1) {
#ifdef DEBUG_EAGLE_KERNEL
          printf("batch %d finished in first token \n", i);
#endif
          in_offset += cur_last_seq_lens_this_time > 0 ? 1 : 0;
          // normal end
        } else {
#ifdef DEBUG_EAGLE_KERNEL
          printf("batch %d finished in non-first token \n", i);
#endif
          in_offset += cur_last_seq_lens_this_time;
        }
      }
    }
    output_token_num[0] = out_offset;
#ifdef DEBUG_EAGLE_KERNEL
    printf("position map output_token_num%d:\n", output_token_num[0]);
    for (int i = 0; i < output_token_num[0]; i++) {
      printf("%d ", src_map[i]);
    }
    printf("\n");
#endif
  }
}

template <typename T, int PackSize>
__global__ void rebuildSelfHiddenStatesKernel(
    const T* input, int* src_map, T* output, int dim_embed, int elem_cnt) {
  using LoadT = AlignedVector<T, PackSize>;
  LoadT src_vec;

  int global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int elem_id = global_thread_idx * PackSize; elem_id < elem_cnt;
       elem_id += blockDim.x * gridDim.x * PackSize) {
    int output_token_idx = elem_id / dim_embed;
    int input_token_idx = src_map[output_token_idx];
    int offset = elem_id % dim_embed;
    Load<T, PackSize>(input + input_token_idx * dim_embed + offset, &src_vec);
    Store<T, PackSize>(src_vec, output + output_token_idx * dim_embed + offset);
  }
}


template <paddle::DataType D>
std::vector<paddle::Tensor> DispatchDtype(
    const paddle::Tensor input,
    const paddle::Tensor last_seq_lens_this_time,
    const paddle::Tensor seq_lens_this_time,
    const paddle::Tensor step_idx) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  int input_token_num = input.shape()[0];
  int dim_embed = input.shape()[1];
  int bsz = seq_lens_this_time.shape()[0];
  auto src_map = paddle::full({input_token_num},
                              -1,
                              seq_lens_this_time.dtype(),
                              seq_lens_this_time.place());
  auto output_token_num = paddle::full(
      {1}, 0, seq_lens_this_time.dtype(), seq_lens_this_time.place());

  computeOrderKernel<<<1, 1, 0, seq_lens_this_time.stream()>>>(
      last_seq_lens_this_time.data<int>(),
      seq_lens_this_time.data<int>(),
      step_idx.data<int64_t>(),
      src_map.data<int>(),
      output_token_num.data<int>(),
      bsz);

  int output_token_num_cpu =
      output_token_num.copy_to(paddle::CPUPlace(), false).data<int>()[0];

  auto out = paddle::full(
      {output_token_num_cpu, dim_embed}, -1, input.type(), input.place());

  constexpr int packSize = VEC_16B / (sizeof(DataType_));
  int elem_cnt = output_token_num_cpu * dim_embed;
  // printf("output_token_num: %d, dim_embed: %d, cnt: %d. packSize: %d\n",
  // output_token_num_cpu, dim_embed,elem_cnt, packSize);
  assert(elem_cnt % packSize == 0);

  int pack_num = elem_cnt / packSize;

  int grid_size = 1;

  GetNumBlocks(pack_num, &grid_size);

  constexpr int threadPerBlock = 128;

  rebuildSelfHiddenStatesKernel<DataType_, packSize>
      <<<grid_size, threadPerBlock, 0, input.stream()>>>(
          reinterpret_cast<const DataType_*>(input.data<data_t>()),
          src_map.data<int>(),
          reinterpret_cast<DataType_*>(out.data<data_t>()),
          dim_embed,
          elem_cnt);


  return {out};
}


std::vector<paddle::Tensor> EagleGetSelfHiddenStates(
    const paddle::Tensor& input,
    const paddle::Tensor& last_seq_lens_this_time,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& step_idx) {
  switch (input.dtype()) {
    case paddle::DataType::BFLOAT16:
      return DispatchDtype<paddle::DataType::BFLOAT16>(
          input, last_seq_lens_this_time, seq_lens_this_time, step_idx);
    case paddle::DataType::FLOAT16:
      return DispatchDtype<paddle::DataType::FLOAT16>(
          input, last_seq_lens_this_time, seq_lens_this_time, step_idx);
    default:
      PD_THROW("Not support this data type");
  }
}


PD_BUILD_OP(eagle_get_self_hidden_states)
    .Inputs(
        {"input", "last_seq_lens_this_time", "seq_lens_this_time", "step_idx"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(EagleGetSelfHiddenStates));