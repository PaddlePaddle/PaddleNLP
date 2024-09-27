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

__global__ void split_kv_block(const int * __restrict__ seq_lens_decoder,
                               const int * __restrict__ seq_len_encoder,
                               int * __restrict__ batch_ids,
                               int * __restrict__ tile_ids_per_batch,
                               int * __restrict__ num_blocks_x,
                               const int bsz,
                               const int pad_len,
                               const int num_row_per_block) {
  if (threadIdx.x == 0) {
    int gridx = 0;
    int index = 0;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      const int start_len = seq_lens_decoder[bid];
      int seq_len = seq_len_encoder[bid] + start_len % pad_len;
      if (seq_len_encoder[bid] == 0) {
        seq_len = 0;
      }
      const int loop_times = div_up(seq_len, num_row_per_block);
      for (uint32_t tile_id = 0; tile_id < loop_times; tile_id++) {
        batch_ids[index] = bid;
        tile_ids_per_batch[index++] = tile_id;
      }
      gridx += loop_times;
    }
    *num_blocks_x = gridx;
  }
}

// encoder stage input: seq_len_encoder, none, max_len_this_time, bsz, FLAGS_flag_block_shape_q
std::vector<paddle::Tensor> SplitKVBlock(const paddle::Tensor& sequence_lengths_stage,
                                        const paddle::Tensor& sequence_lengths_remove,
                                        const paddle::Tensor& max_len,
                                        const paddle::Tensor& com_offsets,
                                        const int padding_len,
                                        const int num_row_per_block) {
	auto stream = sequence_lengths_stage.stream();
  int max_len_data = max_len.data<int>()[0];
  int bsz = com_offsets.shape()[0];
  if (max_len_data <= 0) {
    auto batch_ids = paddle::full({1}, -1, paddle::DataType::INT32, paddle::GPUPlace());
    auto tile_ids_per_batch = paddle::full({1}, -1, paddle::DataType::INT32, paddle::GPUPlace());
    auto num_blocks_x_cpu = paddle::full({1}, -1, paddle::DataType::INT32, paddle::CPUPlace());
    return {batch_ids, tile_ids_per_batch, num_blocks_x_cpu};
  }
	const uint32_t max_tile_size_per_bs = div_up(max_len_data, num_row_per_block);
  auto batch_ids = paddle::empty({bsz * max_tile_size_per_bs}, paddle::DataType::INT32, paddle::GPUPlace());
	auto tile_ids_per_batch = paddle::empty({bsz * max_tile_size_per_bs}, paddle::DataType::INT32, paddle::GPUPlace());
	auto num_blocks = paddle::empty({1}, paddle::DataType::INT32, paddle::GPUPlace());
  
  split_kv_block<<<1, 32, 0, stream>>>(
          sequence_lengths_stage.data<int>(),
          sequence_lengths_remove.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          num_blocks.data<int>(),
          bsz,
          padding_len,
          num_row_per_block
        );
  auto num_blocks_cpu = num_blocks.copy_to(paddle::CPUPlace(), false);

	return {batch_ids, tile_ids_per_batch, num_blocks_cpu};
}

std::vector<paddle::DataType> SplitKVBlockInferDtype(const paddle::DataType& sequence_lengths_stage_dtype, const paddle::DataType& sequence_lengths_remove_dtype, const paddle::DataType& max_len_dtype, const paddle::DataType& com_offsets_dtype) {
    return {paddle::DataType::INT32, paddle::DataType::INT32, paddle::DataType::INT32};
}

std::vector<std::vector<int64_t>>SplitKVBlockInferShape(const std::vector<int64_t>& sequence_lengths_stage_shape, const std::vector<int64_t>& sequence_lengths_remove_shape, const std::vector<int64_t>& max_len_shape, const std::vector<int64_t>& com_offsets_shape) {
    std::vector<int64_t> dynamic_shape = {-1};

    return {dynamic_shape, dynamic_shape, {1}};
}

PD_BUILD_OP(split_kv_block)
    .Inputs({"sequence_lengths_stage", "sequence_lengths_remove", "max_len", "com_offsets"})
    .Outputs({"batch_ids", "tile_ids_per_batch", "num_blocks_cpu"})
    .Attrs({"padding_len: int", "num_row_per_block: int"})
    .SetKernelFn(PD_KERNEL(SplitKVBlock))
    .SetInferDtypeFn(PD_INFER_DTYPE(SplitKVBlockInferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(SplitKVBlockInferShape));
