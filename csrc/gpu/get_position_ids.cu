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

__global__ void GetPositionIdsKernel(
    const int* seq_lens_encoder,  // [bsz] 每个批次的 encoder 长度
    const int* seq_lens_decoder,  // [bsz] 每个批次的 decoder 长度
    int* position_ids,            // 输出的一维 position_ids
    const int bsz) {              // 批次大小
  // 当前线程索引（每个线程对应一个批次）
  int tid = threadIdx.x;
  if (tid >= bsz) return;

  // 动态计算当前批次的偏移量
  int offset = 0;
  for (int i = 0; i < tid; i++) {
    offset += seq_lens_encoder[i];
    if (seq_lens_decoder[i] > 0) {
      offset += 1;
    }
  }

  // 当前批次的 encoder 和 decoder 长度
  int encoder_len = seq_lens_encoder[tid];
  int decoder_len = seq_lens_decoder[tid];

  // 写入 encoder 的 position_ids
  for (int i = 0; i < encoder_len; i++) {
    position_ids[offset + i] = i;
  }
  offset += encoder_len;

  // 写入 decoder 的 position_ids
  if (decoder_len > 0) {
    position_ids[offset] = decoder_len;  // 使用 decoder 长度本身
  }
}


void GetPositionIds(const paddle::Tensor& seq_lens_encoder,
                    const paddle::Tensor& seq_lens_decoder,
                    const paddle::Tensor& position_ids) {
  const int bsz = seq_lens_encoder.shape()[0];

  GetPositionIdsKernel<<<1, bsz, 0, position_ids.stream()>>>(
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      const_cast<int*>(position_ids.data<int>()),
      bsz);
}

PD_BUILD_OP(get_position_ids)
    .Inputs({"seq_lens_encoder", "seq_lens_decoder", "position_ids"})
    .Outputs({"position_ids_out"})
    .SetInplaceMap({{"position_ids", "position_ids_out"}})
    .SetKernelFn(PD_KERNEL(GetPositionIds));