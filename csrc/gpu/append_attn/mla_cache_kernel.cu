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
#pragma once

#include "mla_cache_kernel.cuh"

template <paddle::DataType T>
std::vector<paddle::Tensor> PrefillMLAWriteCache(
                    const AppendAttnMetaData& meta_data,
                    const paddle::Tensor& kv_nope,
                    const paddle::Tensor& kv_pe,
                    const paddle::Tensor& seq_lens,
                    const paddle::Tensor& seq_lens_decoder,
                    const paddle::Tensor& padding_offsets,
                    const paddle::Tensor& cum_offsets,
                    const paddle::Tensor& block_tables,
                    const int max_seq_len,
                    cudaStream_t& stream,
                    paddle::Tensor* kv_cache) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;
  auto num_tokens = meta_data.token_nums;
  auto block_size = meta_data.block_size;
  auto nope_size = meta_data.head_dims_v;
  auto all_size = meta_data.head_dims;
  int pe_size = all_size - nope_size;
  auto kv_num_heads = meta_data.kv_num_heads;
  const uint32_t elem_nums = num_tokens * kv_num_heads * (nope_size + pe_size);

  constexpr int PackSize = 16 / sizeof(DataType_);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);

  prefill_absorb_cache_kernel<DataType_, PackSize>
      <<<grid_size, blocksize, 0, stream>>>(
          reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_nope.data<data_t>())),
          reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_pe.data<data_t>())),
          reinterpret_cast<DataType_*>(kv_cache->data<data_t>()),
          block_tables.data<int>(),
          padding_offsets.data<int>(),
          cum_offsets.data<int>(),
          seq_lens.data<int>(),
          seq_lens_decoder.data<int>(),
          max_seq_len,
          max_blocks_per_seq,
          kv_num_heads,
          nope_size,
          pe_size,
          block_size,
          elem_nums);
  return {};
}

std::vector<paddle::Tensor> PrefillMLAWriteCacheKernel(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const std::string& cache_quant_type_str,
    const int max_seq_len) {
  cudaStream_t stream = kv_pe.stream();
  AppendAttnMetaData meta_data;
  const auto& kv_nope_dims = kv_nope.dims();
  const auto& kv_pe_dims = kv_pe.dims();
  const auto& kv_cache_dims = kv_cache.dims();
  meta_data.kv_num_heads = kv_cache_dims[1];
  const auto nope_size = kv_nope_dims[kv_nope_dims.size() - 1];
  meta_data.head_dims = kv_cache_dims[3];
  meta_data.head_dims_v = nope_size;

  meta_data.max_blocks_per_seq = block_tables.dims()[1];
  meta_data.block_size = kv_cache_dims[2];
  meta_data.batch_size = cum_offsets.dims()[0];
  switch (kv_pe.dtype()) {
    case paddle::DataType::BFLOAT16: {
      return PrefillMLAWriteCache<paddle::DataType::BFLOAT16>(meta_data,
                              kv_nope,
                              kv_pe,
                              seq_lens,
                              seq_lens_decoder,
                              padding_offsets,
                              cum_offsets,
                              block_tables,
                              max_seq_len,
                              stream,
                              const_cast<paddle::Tensor*>(&kv_cache));
    }
    case paddle::DataType::FLOAT16: {
      return PrefillMLAWriteCache<paddle::DataType::FLOAT16>(meta_data,
                              kv_nope,
                              kv_pe,
                              seq_lens,
                              seq_lens_decoder,
                              padding_offsets,
                              cum_offsets,
                              block_tables,
                              max_seq_len,
                              stream,
                              const_cast<paddle::Tensor*>(&kv_cache));
    }
  }
  return {};
}

template <paddle::DataType T>
std::vector<paddle::Tensor> DecodeMLAWriteCache(
                    const AppendAttnMetaData& meta_data,
                    const paddle::Tensor& kv_nope,
                    const paddle::Tensor& kv_pe,
                    const paddle::Tensor& seq_lens,
                    const paddle::Tensor& seq_lens_encoder,
                    const paddle::Tensor& padding_offsets,
                    const paddle::Tensor& cum_offsets,
                    const paddle::Tensor& block_tables,
                    const int max_seq_len,
                    cudaStream_t& stream,
                    paddle::Tensor* kv_cache) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
  
  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;
  auto bsz = meta_data.batch_size;
  auto block_size = meta_data.block_size;
  auto nope_size = meta_data.head_dims_v;
  auto all_size = meta_data.head_dims;
  int pe_size = all_size - nope_size;
  auto kv_num_heads = meta_data.kv_num_heads;
  const uint32_t elem_nums = bsz * kv_num_heads * (nope_size + pe_size);

  constexpr int PackSize = 16 / sizeof(DataType_);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);

  decode_absorb_cache_kernel<DataType_, PackSize>
      <<<grid_size, blocksize, 0, stream>>>(
          reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_nope.data<data_t>())),
          reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_pe.data<data_t>())),
          reinterpret_cast<DataType_*>(kv_cache->data<data_t>()),
          block_tables.data<int>(),
          cum_offsets.data<int>(),
          seq_lens.data<int>(),
          seq_lens_encoder.data<int>(),
          max_seq_len,
          max_blocks_per_seq,
          kv_num_heads,
          nope_size,
          pe_size,
          block_size,
          elem_nums);
  return {};
}

std::vector<paddle::Tensor> DecodeMLAWriteCacheKernel(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const std::string& cache_quant_type_str,
    const int max_seq_len) {
  cudaStream_t stream = kv_pe.stream();
  AppendAttnMetaData meta_data;
  const auto& kv_nope_dims = kv_nope.dims();
  const auto& kv_pe_dims = kv_pe.dims();
  const auto& kv_cache_dims = kv_cache.dims();
  meta_data.kv_num_heads = kv_cache_dims[1];
  const auto nope_size = kv_nope_dims[kv_nope_dims.size() - 1];
  meta_data.head_dims = kv_cache_dims[3];
  meta_data.head_dims_v = nope_size;

  meta_data.max_blocks_per_seq = block_tables.dims()[1];
  meta_data.block_size = kv_cache_dims[2];
  meta_data.batch_size = cum_offsets.dims()[0];
  switch (kv_pe.dtype()) {
    case paddle::DataType::BFLOAT16: {
      return DecodeMLAWriteCache<paddle::DataType::BFLOAT16>(meta_data,
                              kv_nope,
                              kv_pe,
                              seq_lens,
                              seq_lens_encoder,
                              padding_offsets,
                              cum_offsets,
                              block_tables,
                              max_seq_len,
                              stream,
                              const_cast<paddle::Tensor*>(&kv_cache));
    }
    case paddle::DataType::FLOAT16: {
      return DecodeMLAWriteCache<paddle::DataType::FLOAT16>(meta_data,
                              kv_nope,
                              kv_pe,
                              seq_lens,
                              seq_lens_encoder,
                              padding_offsets,
                              cum_offsets,
                              block_tables,
                              max_seq_len,
                              stream,
                              const_cast<paddle::Tensor*>(&kv_cache));
    }
  }
  return {};
}


PD_BUILD_OP(prefill_mla_write_cache)
    .Inputs({"kv_nope",
             "kv_pe",
             "kv_cache",
             "seq_lens",
             "seq_lens_decoder",
             "padding_offsets",
             "cum_offsets",
             "block_tables"})
    .Outputs({"kv_cache_out"})
    .SetInplaceMap({{"kv_cache", "kv_cache_out"}})
    .Attrs({"cache_quant_type_str: std::string",
            "max_seq_len: int"})
    .SetKernelFn(PD_KERNEL(PrefillMLAWriteCacheKernel));

PD_BUILD_OP(decode_mla_write_cache)
    .Inputs({"kv_nope",
             "kv_pe",
             "kv_cache",
             "seq_lens",
             "seq_lens_encoder",
             "padding_offsets",
             "cum_offsets",
             "block_tables"})
    .Outputs({"kv_cache_out"})
    .SetInplaceMap({{"kv_cache", "kv_cache_out"}})
    .Attrs({"cache_quant_type_str: std::string",
            "max_seq_len: int"})
    .SetKernelFn(PD_KERNEL(DecodeMLAWriteCacheKernel));