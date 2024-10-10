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
                              const int gqa_group_size) {
  if (threadIdx.x == 0) {
    int gridx = 0;
    int index = 0;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      int seq_len = seq_lens_q[bid];
      if (seq_lens_encoder && seq_lens_encoder[bid] > 0) {
        seq_len = 0;
      }
      const int loop_times =
          div_up(seq_len * gqa_group_size, num_rows_per_block);
      for (uint32_t tile_id = 0; tile_id < loop_times; tile_id++) {
        batch_ids[index] = bid;
        tile_ids_per_batch[index++] = tile_id;
      }
      gridx += loop_times;
    }
    *num_blocks_x = gridx;
  }
}

__global__ void split_kv_block(const int* __restrict__ seq_lens_decoder,
                               const int* __restrict__ seq_lens_encoder,
                               int* __restrict__ batch_ids,
                               int* __restrict__ tile_ids_per_batch,
                               int* __restrict__ num_blocks_x,
                               const int bsz,
                               const int pad_len,
                               const int num_row_per_block) {
  if (threadIdx.x == 0) {
    int gridx = 0;
    int index = 0;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      const int start_len = seq_lens_decoder[bid];
      int seq_len = seq_lens_encoder[bid] + start_len % pad_len;
      if (seq_lens_encoder[bid] == 0) {
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

std::vector<paddle::Tensor> GetBlockShapeAndSplitKVBlock(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& max_enc_len_this_time,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& cum_offsets,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int gqa_group_size,
    const int block_size) {
  auto stream = seq_lens_encoder.stream();
  int bsz = cum_offsets.shape()[0];

  // decoder
  const uint32_t decoder_max_tile_size_per_bs_q =
      div_up((1 * gqa_group_size), decoder_block_shape_q);
  auto decoder_batch_ids =
      GetEmptyTensor({bsz * decoder_max_tile_size_per_bs_q},
                     paddle::DataType::INT32,
                     seq_lens_encoder.place());
  auto decoder_tile_ids_per_batch =
      GetEmptyTensor({bsz * decoder_max_tile_size_per_bs_q},
                     paddle::DataType::INT32,
                     seq_lens_encoder.place());
  auto decoder_num_blocks_x =
      GetEmptyTensor({1}, paddle::DataType::INT32, seq_lens_encoder.place());
  split_q_block<<<1, 32, 0, stream>>>(seq_lens_this_time.data<int>(),
                                      seq_lens_encoder.data<int>(),
                                      decoder_batch_ids.data<int>(),
                                      decoder_tile_ids_per_batch.data<int>(),
                                      decoder_num_blocks_x.data<int>(),
                                      bsz,
                                      decoder_block_shape_q,
                                      gqa_group_size);
  auto decoder_num_blocks_x_cpu =
      decoder_num_blocks_x.copy_to(paddle::CPUPlace(), false);

  int max_enc_len_this_time_data = max_enc_len_this_time.data<int>()[0];
  if (max_enc_len_this_time_data <= 0) {
    auto encoder_batch_ids =
        paddle::full({1}, -1, paddle::DataType::INT32, paddle::GPUPlace());
    auto encoder_tile_ids_per_batch =
        paddle::full({1}, -1, paddle::DataType::INT32, paddle::GPUPlace());
    auto encoder_num_blocks_x_cpu =
        paddle::full({1}, -1, paddle::DataType::INT32, paddle::CPUPlace());
    auto kv_batch_ids =
        paddle::full({1}, -1, paddle::DataType::INT32, paddle::GPUPlace());
    auto kv_tile_ids_per_batch =
        paddle::full({1}, -1, paddle::DataType::INT32, paddle::GPUPlace());
    auto kv_num_blocks_x_cpu =
        paddle::full({1}, -1, paddle::DataType::INT32, paddle::CPUPlace());

    return {encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks_x_cpu, /*cpu*/
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks_x_cpu, /*cpu*/
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks_x_cpu /*cpu*/};
  }

  // encoder
  const uint32_t encoder_max_tile_size_per_bs_q = div_up(
      (max_enc_len_this_time_data * gqa_group_size), encoder_block_shape_q);
  auto encoder_batch_ids =
      GetEmptyTensor({bsz * encoder_max_tile_size_per_bs_q},
                     paddle::DataType::INT32,
                     seq_lens_encoder.place());
  auto encoder_tile_ids_per_batch =
      GetEmptyTensor({bsz * encoder_max_tile_size_per_bs_q},
                     paddle::DataType::INT32,
                     seq_lens_encoder.place());
  auto encoder_num_blocks_x =
      GetEmptyTensor({1}, paddle::DataType::INT32, seq_lens_encoder.place());
  split_q_block<<<1, 32, 0, stream>>>(seq_lens_encoder.data<int>(),
                                      nullptr,
                                      encoder_batch_ids.data<int>(),
                                      encoder_tile_ids_per_batch.data<int>(),
                                      encoder_num_blocks_x.data<int>(),
                                      bsz,
                                      encoder_block_shape_q,
                                      gqa_group_size);
  auto encoder_num_blocks_x_cpu =
      encoder_num_blocks_x.copy_to(paddle::CPUPlace(), false);

  // kv
  const uint32_t max_tile_size_per_bs_kv =
      div_up(max_enc_len_this_time_data, block_size);
  auto kv_batch_ids = GetEmptyTensor({bsz * max_tile_size_per_bs_kv},
                                     paddle::DataType::INT32,
                                     seq_lens_encoder.place());
  auto kv_tile_ids_per_batch = GetEmptyTensor({bsz * max_tile_size_per_bs_kv},
                                              paddle::DataType::INT32,
                                              seq_lens_encoder.place());
  auto kv_num_blocks_x =
      GetEmptyTensor({1}, paddle::DataType::INT32, seq_lens_encoder.place());
  split_kv_block<<<1, 32, 0, stream>>>(seq_lens_decoder.data<int>(),
                                       seq_lens_encoder.data<int>(),
                                       kv_batch_ids.data<int>(),
                                       kv_tile_ids_per_batch.data<int>(),
                                       kv_num_blocks_x.data<int>(),
                                       bsz,
                                       block_size,
                                       block_size);
  auto kv_num_blocks_x_cpu = kv_num_blocks_x.copy_to(paddle::CPUPlace(), false);
  return {encoder_batch_ids,
          encoder_tile_ids_per_batch,
          encoder_num_blocks_x_cpu, /*cpu*/
          kv_batch_ids,
          kv_tile_ids_per_batch,
          kv_num_blocks_x_cpu, /*cpu*/
          decoder_batch_ids,
          decoder_tile_ids_per_batch,
          decoder_num_blocks_x_cpu /*cpu*/};
}

std::vector<paddle::DataType> GetBlockShapeAndSplitKVBlockInferDtype(
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& max_enc_len_this_time_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& cum_offsets_dtype) {
  return {paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32};
}

std::vector<std::vector<int64_t>> GetBlockShapeAndSplitKVBlockInferShape(
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& max_enc_len_this_time_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& cum_offsets_shape) {
  std::vector<int64_t> dynamic_shape = {-1};

  return {dynamic_shape,
          dynamic_shape,
          {1},
          dynamic_shape,
          dynamic_shape,
          {1},
          dynamic_shape,
          dynamic_shape,
          {1}};
}

PD_BUILD_OP(get_block_shape_and_split_kv_block)
    .Inputs({"seq_lens_encoder",
             "seq_lens_decoder",
             "max_enc_len_this_time",
             "seq_lens_this_time",
             "cum_offsets"})
    .Outputs({"encoder_batch_ids",
              "encoder_tile_ids_per_batch",
              "encoder_num_blocks",
              "kv_batch_ids",
              "kv_tile_ids_per_batch",
              "kv_num_blocks",
              "decoder_batch_ids",
              "decoder_tile_ids_per_batch",
              "decoder_num_blocks"})
    .Attrs({"encoder_block_shape_q: int",
            "decoder_block_shape_q: int",
            "gqa_group_size: int",
            "block_size: int"})
    .SetKernelFn(PD_KERNEL(GetBlockShapeAndSplitKVBlock))
    .SetInferShapeFn(PD_INFER_SHAPE(GetBlockShapeAndSplitKVBlockInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetBlockShapeAndSplitKVBlockInferDtype));
