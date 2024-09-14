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

template <typename T>
inline __device__ __host__ T div_up(T m, T n) {
  return (m + n - 1) / n;
}

__global__ void split_q_block(const int* __restrict__ seq_lens_q,
                              const int* __restrict__ seq_lens_encoder,
                              int* __restrict__ batch_ids,
                              int* __restrict__ tile_ids_per_batch,
                              int* __restrict__ num_blocks_x,
                              const int bsz,
                              const int num_rows_per_block,
                              const int group_size) {
  if (threadIdx.x == 0) {
    int gridx = 0;
    int index = 0;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      int seq_len = seq_lens_q[bid];
      if (seq_lens_encoder && seq_lens_encoder[bid] > 0) {
        seq_len = 0;
      }
      const int loop_times = div_up(seq_len * group_size, num_rows_per_block);
      for (uint32_t tile_id = 0; tile_id < loop_times; tile_id++) {
        batch_ids[index] = bid;
        tile_ids_per_batch[index++] = tile_id;
      }
      gridx += loop_times;
    }
    *num_blocks_x = gridx;
  }
}

// encoder stage input: seq_len_encoder, none, max_len_this_time, group_size,
// bsz, FLAGS_flag_block_shape_q decoder stage input: seq_len, seq_len_encoder,
// 1, group_size, bsz, FLAGS_flag_dec_block_shape_q
std::vector<paddle::Tensor> GetBlockShape(
    const paddle::Tensor& sequence_lengths_stage,
    const paddle::optional<paddle::Tensor>& sequence_lengths_remove,
    const paddle::Tensor& max_len,
    const paddle::Tensor& com_offsets,
    const int num_qrow_per_block,
    const int group_size) {
  auto stream = sequence_lengths_stage.stream();
  auto max_len_cpu = max_len.copy_to(paddle::CPUPlace(), false);
  int max_len_data = max_len_cpu.data<int>()[0];
  if (max_len_data <= 0) {
    auto batch_ids =
        paddle::full({1}, -1, paddle::DataType::INT32, paddle::GPUPlace());
    auto tile_ids_per_batch =
        paddle::full({1}, -1, paddle::DataType::INT32, paddle::GPUPlace());
    auto num_blocks_x_cpu =
        paddle::full({1}, -1, paddle::DataType::INT32, paddle::CPUPlace());
    return {batch_ids, tile_ids_per_batch, num_blocks_x_cpu};
  }
  int bsz = com_offsets.shape()[0];
  const uint32_t max_tile_size_per_bs_q =
      div_up((max_len_data * group_size), num_qrow_per_block);
  auto batch_ids = paddle::empty({bsz * max_tile_size_per_bs_q},
                                 paddle::DataType::INT32,
                                 paddle::GPUPlace());
  auto tile_ids_per_batch = paddle::empty({bsz * max_tile_size_per_bs_q},
                                          paddle::DataType::INT32,
                                          paddle::GPUPlace());
  auto num_blocks_x =
      paddle::empty({1}, paddle::DataType::INT32, paddle::GPUPlace());
  auto num_blocks_x_cpu =
      paddle::empty({1}, paddle::DataType::INT32, paddle::CPUPlace());

  const int* seq_lens_remove = nullptr;
  if (sequence_lengths_remove) {
    seq_lens_remove = sequence_lengths_remove.get().data<int>();
  }
  split_q_block<<<1, 32, 0, stream>>>(sequence_lengths_stage.data<int>(),
                                      seq_lens_remove,
                                      batch_ids.data<int>(),
                                      tile_ids_per_batch.data<int>(),
                                      num_blocks_x.data<int>(),
                                      bsz,
                                      num_qrow_per_block,
                                      group_size);

  cudaMemcpy(num_blocks_x_cpu.data<int>(),
             num_blocks_x.data<int>(),
             sizeof(int),
             cudaMemcpyDeviceToHost);
  return {batch_ids, tile_ids_per_batch, num_blocks_x_cpu};
}

std::vector<paddle::DataType> GetBlockShapeInferDtype(
    const paddle::DataType& sequence_lengths_stage_dtype,
    const paddle::optional<paddle::DataType>& sequence_lengths_remove_dtype,
    const paddle::DataType& max_len_dtype,
    const paddle::DataType& com_offsets_dtype) {
  return {paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32};
}

std::vector<std::vector<int64_t>> GetBlockShapeInferShape(
    const std::vector<int64_t>& sequence_lengths_stage_shape,
    const paddle::optional<std::vector<int64_t>>& sequence_lengths_remove_shape,
    const std::vector<int64_t>& max_len_shape,
    const std::vector<int64_t>& com_offsets_shape) {
  std::vector<int64_t> dynamic_shape = {-1};

  return {dynamic_shape, dynamic_shape, {1}};
}

PD_BUILD_OP(get_block_shape)
    .Inputs({"sequence_lengths_stage",
             paddle::Optional("sequence_lengths_remove"),
             "max_len",
             "com_offsets"})
    .Outputs({"batch_ids", "tile_ids_per_batch", "num_blocks_x_cpu"})
    .Attrs({"num_qrow_per_block: int", "group_size: int"})
    .SetKernelFn(PD_KERNEL(GetBlockShape))
    .SetInferShapeFn(PD_INFER_SHAPE(GetBlockShapeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetBlockShapeInferDtype));
