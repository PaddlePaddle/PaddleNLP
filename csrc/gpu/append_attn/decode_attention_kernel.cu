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

#include "decode_attention_func.cuh"

#define CHECK(call)                                                             \
do                                                                              \
{                                                                               \
    const cudaError_t error_code = call;                                        \
    if (error_code != cudaSuccess)                                              \
    {                                                                           \
        printf("CUDA Error:\n");                                                \
        printf("     File:      %s\n", __FILE__);                               \
        printf("     Line       %d:\n", __LINE__);                              \
        printf("     Error code:%d\n", error_code);                             \
        printf("     Error text:%s\n", cudaGetErrorString(error_code));         \
        exit(1);                                                                \
    }                                                                           \
}while(0)

template <typename T, typename OutT, int vec_size, uint32_t bdy, uint32_t HEAD_DIM>
__global__ void merge_varlen_multi_chunks_v2_kernel(const T * __restrict__ multi_out, // [bsz, num_chunks, num_heads, head_dim]
                                                    const T * __restrict__ multi_m, // [bsz, num_chunks, num_heads]
                                                    const T * __restrict__ multi_d, // [bsz, num_chunks, num_heads]
                                                    const int * __restrict__ seq_lens_q,
                                                    const int * __restrict__ seq_lens_kv,
                                                    const int * __restrict__ cum_offsets,
                                                    const T * __restrict__ shift_bias, // [q_num_heads * HEAD_DIM]
                                                    const T * __restrict__ smooth_weight, // [q_num_heads * HEAD_DIM]
                                                    OutT * __restrict__ out, // [token_num, num_heads, head_dim]
                                                    const float in_scale,
                                                    const int num_chunks,
                                                    const int chunk_size,
                                                    const int max_seq_len,
                                                    const int num_heads,
                                                    const int head_dim) {
  const int vid = threadIdx.x, ty = threadIdx.y;
  const int qid = blockIdx.x, hid = blockIdx.y;
  const int seq_len_q = seq_lens_q[qid];
  if (seq_len_q == 0) return;
  int seq_len_kv = seq_lens_kv[qid];
  if (seq_len_kv == 0) return;
  seq_len_kv += seq_len_q;
  const int num_chunks_this_seq = div_up(seq_len_kv, chunk_size);
  if (num_chunks_this_seq == 1 || ty >= num_chunks_this_seq) {
    return;
  }
  __shared__ T smem[bdy * HEAD_DIM];
  __shared__ T md_smem[bdy * 2];

  const int start_token_ids = qid * max_seq_len - __ldg(&cum_offsets[qid]);
  using LoadT = AlignedVector<T, vec_size>;
  LoadT load_vec;
  LoadT res_vec;
  if constexpr (std::is_same<T, half>::value) {
#pragma unroll
    for (int i = 0; i < vec_size / 2; ++i) {
      *((half2*)(&res_vec) + i) = make_half2(0, 0);
    }
  } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
#pragma unroll
    for (int i = 0; i < vec_size / 2; ++i) {
      *((nv_bfloat162*)(&res_vec) + i) = make_bfloat162(0, 0);
    }
  }
  T m;
  T d = 1.f;
  if constexpr (std::is_same<T, half>::value) {
    m = __float2half(-5e4f);
  } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
    m = __float2bfloat16(-3.38953e38f);
  }
  // merge per ty
#pragma unroll 2
  for (int i = ty; i < num_chunks_this_seq; i += bdy) {
    uint32_t offset = (qid * num_chunks + i) * num_heads + hid;
    T m_prev = m;
    T d_prev = d;
    const T m_now = multi_m[offset];
    const T d_now = multi_d[offset];
    m = m_prev > m_now ? m_prev : m_now;
    offset = (qid * num_chunks * num_heads + i * num_heads + hid) * head_dim + vid * vec_size;
    Load<T, vec_size>(&multi_out[offset], &load_vec);
    const T scale1 = hexp(m_prev - m), scale2 = hexp(m_now - m);
    d = d * scale1 + d_now * scale2;
#pragma once
    for (int j = 0; j < vec_size; j++) {
      res_vec[j] = res_vec[j] * scale1 + load_vec[j] * scale2;
    }
  }
  // store ty res
  Store<T, vec_size>(res_vec, &smem[ty * head_dim + vid * vec_size]);
  md_smem[2 * ty] = m;
  md_smem[2 * ty + 1] = d;
  __syncthreads();

  // merge bdy
  softmax_state_t<vec_size, T> st{};
  const uint32_t iter_num = min(num_chunks_this_seq, bdy);
#pragma once
  for (int i = 0; i < iter_num; i++) {
    Load<T, vec_size>(&smem[i * head_dim + vid * vec_size], &load_vec);
    const T m_tmp = md_smem[2 * i], d_tmp = md_smem[2 * i + 1];
    st.merge(load_vec, m_tmp, d_tmp);
  }
  st.normalize();

  AlignedVector<OutT, vec_size> out_vec;

#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    out_vec[i] = static_cast<OutT>(st.o[i]);
  }
  Store<OutT, vec_size>(out_vec, &out[(start_token_ids * num_heads + hid) * head_dim + vid * vec_size]);
}

template <bool partition_kv, typename T, typename OutT, typename CacheT, uint32_t NUM_STAGES, uint32_t DEAL_EACH_TIME, uint32_t GROUP_SIZE, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_V, 
          uint32_t BLOCK_SIZE, uint32_t VEC_SIZE, uint32_t CACHE_VEC_SIZE, uint32_t bdx, uint32_t bdy>
__global__ void multi_query_decode_attention_kernel(T * __restrict__ q, // [token_num, num_heads, head_dim]
                                                    CacheT * __restrict__ cache_k, // [max_block_num, num_heads, block_size, head_dim]
                                                    CacheT * __restrict__ cache_v,
                                                    const T * __restrict__ shift_bias, // [q_num_heads * HEAD_DIM]
                                                    const T * __restrict__ smooth_weight, // [q_num_heads * HEAD_DIM]
                                                    const int * __restrict__ seq_lens_q,
                                                    const int * __restrict__ seq_lens_kv,
                                                    const int * __restrict__ cum_offsets,
                                                    const int * __restrict__ block_table, // [bsz, block_num_per_seq]
                                                    const int max_seq_len,
                                                    const int max_dec_len,
                                                    const int max_block_num_per_seq,
                                                    const float scale,
                                                    const float in_scale,
                                                    const uint32_t chunk_size,
                                                    T * __restrict__ tmp_workspace, // [batch_size, num_chunks, num_heads, head_dim]
                                                    T * __restrict__ tmp_m, // [batch_size, num_chunks, num_heads]
                                                    T * __restrict__ tmp_d, // [batch_size, num_chunks, num_heads]
                                                    OutT * __restrict__ out) {
  constexpr uint32_t q_block_num = GROUP_SIZE / bdy;
  const uint32_t bidx = blockIdx.x, kv_head_idx = blockIdx.z;
  const uint32_t bid = bidx / q_block_num, q_block_idx = bidx % q_block_num;
  const uint32_t bidy = threadIdx.y;
  const uint32_t gid = q_block_idx * bdy + threadIdx.y;
  const uint32_t tidx = threadIdx.x;
  constexpr uint32_t num_vec_per_head_qk = HEAD_DIM_QK / VEC_SIZE;
  constexpr uint32_t num_vec_per_head_v = HEAD_DIM_V / VEC_SIZE;
  constexpr uint32_t num_tile_v = (num_vec_per_head_v + bdx - 1) / bdx;

  const uint32_t q_head_idx = kv_head_idx * GROUP_SIZE + gid;
  const uint32_t kv_num_heads = gridDim.z;
  const uint32_t q_num_heads = kv_num_heads * GROUP_SIZE;
  
  const int *block_table_now = block_table + bid * max_block_num_per_seq;
  
  const uint32_t num_chunks = gridDim.y;
  const uint32_t chunk_id = blockIdx.y;
  const uint32_t q_len = seq_lens_q[bid];
  if (q_len <= 0) {
    return;
  }
  uint32_t kv_len = seq_lens_kv[bid]; // !!!!!!!!
  if (kv_len <= 0) {
     return;
  }
  kv_len += q_len;
  const uint32_t num_chunk_this_seq = div_up(kv_len, chunk_size);
  const uint32_t q_start_idx = bid * max_seq_len - __ldg(&cum_offsets[bid]);
  const uint32_t q_write_idx = bid * max_seq_len - __ldg(&cum_offsets[bid]);
  if (chunk_id >= num_chunk_this_seq) {
    return;
  }

  const uint32_t chunk_start = partition_kv ? chunk_id * chunk_size : 0;
  const uint32_t chunk_end = partition_kv ? min(kv_len, chunk_start + chunk_size) : kv_len;
  const uint32_t chunk_len = chunk_end - chunk_start;

  extern __shared__ uint8_t smem[];
  const T *q_now = q + (q_start_idx * q_num_heads + q_head_idx) * HEAD_DIM_QK;
  T *q_smem = reinterpret_cast<T*>(smem); // [HEAD_DIM_QK * sizeof(T)]
  T *cu_q_smem = q_smem + bidy * HEAD_DIM_QK;
#pragma unroll
  for(uint32_t vid = tidx; vid < num_vec_per_head_qk; vid += bdx) {
    ((float4*)(&cu_q_smem[vid * VEC_SIZE]))[0] = ((float4*)(&q_now[vid * VEC_SIZE]))[0];
  }
  __syncthreads();
  using VecT = AlignedVector<T, VEC_SIZE>;
  VecT q_vec;
#pragma unroll
  for(uint32_t vid = tidx; vid < num_vec_per_head_qk; vid += bdx) {
    Load<T, VEC_SIZE>(cu_q_smem + vid * VEC_SIZE, &q_vec);
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      q_vec[i] *= scale;
    }
    Store<T, VEC_SIZE>(q_vec, cu_q_smem + vid * VEC_SIZE);
  }


  CacheT *kv_smem = reinterpret_cast<CacheT*>(smem + bdy * HEAD_DIM_QK * sizeof(CacheT)); 
  uint32_t stage_idx = 0;  
  constexpr int loop_times = DEAL_EACH_TIME / bdy;
#pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
#pragma unroll
    for (int j = 0; j < loop_times; ++j) {
      const uint32_t k_seq_offset = i * DEAL_EACH_TIME + j * bdy + bidy;
      const uint32_t k_seq_id = chunk_start + k_seq_offset;
      produce_kv<SharedMemFillMode::kNoFill, HEAD_DIM_QK, VEC_SIZE, num_vec_per_head_qk, bdx, BLOCK_SIZE, CACHE_VEC_SIZE>(
        kv_smem,
        cache_k,
        block_table_now,
        k_seq_id,
        k_seq_offset,
        kv_head_idx,
        kv_num_heads,
        tidx,
        chunk_start,
        chunk_end
      );
    }
    commit_group();
    stage_idx = (stage_idx + 1) % NUM_STAGES;
  }


  softmax_state_ts<VEC_SIZE, T, num_tile_v> st;
  float s[DEAL_EACH_TIME];
  
  const uint32_t num_iters = div_up(chunk_len, DEAL_EACH_TIME);
  for (int iter = 0; iter < num_iters; ++iter) {
    wait_group<NUM_STAGES - 1>();
    __syncthreads();
    // compute qk
    compute_qk<VEC_SIZE, num_vec_per_head_qk, bdx, bdy, HEAD_DIM_QK, DEAL_EACH_TIME, num_tile_v>(
      cu_q_smem,
      kv_smem,
      chunk_start + iter * DEAL_EACH_TIME,
      stage_idx,
      iter * DEAL_EACH_TIME,
      chunk_len,
      tidx,
      gid,
      scale,
      s,
      st
    );
    __syncthreads();

    // compute sv
    compute_sv<VEC_SIZE, num_vec_per_head_v, bdx, DEAL_EACH_TIME, HEAD_DIM_QK, num_tile_v>(
      s,
      kv_smem,
      stage_idx,
      iter * DEAL_EACH_TIME,
      chunk_len,
      tidx,
      st
    );
    __syncthreads();

#pragma unroll
    for (int j = 0; j < loop_times; ++j) {
      const uint32_t k_seq_offset = j * bdy + bidy;
      produce_kv<SharedMemFillMode::kNoFill, HEAD_DIM_QK, VEC_SIZE, num_vec_per_head_qk, bdx, BLOCK_SIZE, CACHE_VEC_SIZE>(
        kv_smem,
        cache_k,
        block_table_now,
        chunk_start + k_seq_offset + (iter + NUM_STAGES) * DEAL_EACH_TIME,
        stage_idx * DEAL_EACH_TIME + k_seq_offset,
        kv_head_idx,
        kv_num_heads,
        tidx,
        chunk_start,
        chunk_end
      );
    }
    commit_group();
    stage_idx = (stage_idx + 1) % NUM_STAGES;
  }
  wait_group<0>();
  __syncthreads();

  // normize if not partition_kv
  for(uint32_t vid = tidx; vid < num_vec_per_head_v; vid += bdx) {
    const uint32_t tile_id = vid / bdx;
    if (!partition_kv || num_chunk_this_seq == 1) {
      st.normalize(tile_id);
    }
    if (partition_kv && num_chunk_this_seq > 1) {
      const uint32_t head_idx = (bid * num_chunks + chunk_id) * q_num_heads + q_head_idx;
      Store<T, VEC_SIZE>(st.o[tile_id], tmp_workspace + head_idx * HEAD_DIM_V + vid * VEC_SIZE);
      tmp_m[head_idx] = st.m;
      tmp_d[head_idx] = st.d;
    } else {
      Store<OutT, VEC_SIZE>(st.o[tile_id], out + (q_write_idx * q_num_heads + q_head_idx) * HEAD_DIM_V + vid * VEC_SIZE);
    }
  }
}


template <typename T, uint32_t GROUP_SIZE, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_V, uint32_t BLOCK_SIZE, bool CAUSAL, uint32_t NUM_STAGE, uint32_t cache_bytes, uint32_t DEAL_EACH_TIME>
void MultiQueryDecoderAttention(
  const AppendAttnMetaData& meta_data,
  cudaStream_t &stream,
  const paddle::Tensor &q,
  const paddle::Tensor &cache_k, // [max_block_num, num_kv_heads, block_size, head_dim]
  const paddle::Tensor &cache_v, // [num_kv_heads, head_dim]
  const paddle::optional<paddle::Tensor>& attn_mask,
  const paddle::optional<paddle::Tensor>& shift_bias,
  const paddle::optional<paddle::Tensor>& smooth_weight,
  const paddle::Tensor &seq_lens_q,
  const paddle::Tensor &seq_lens_kv,
  const paddle::Tensor &padding_offsets,
  const paddle::Tensor &cum_offsets,
  const paddle::Tensor &block_table,
  const int max_seq_len,
  const int max_dec_len,
  const float rope_scale,
  const float rope_theta,
  const float softmax_scale,
  const float in_scale,
  paddle::Tensor *out) {
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;

  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto token_num = meta_data.token_nums;
  auto bsz = meta_data.batch_size;
  auto max_block_num_per_seq = meta_data.max_blocks_per_seq;
  constexpr int num_stages = NUM_STAGE;

  constexpr int vec_size = 16 / sizeof(T); // 8 16 32
  constexpr int cache_vec_size = 128 / cache_bytes; // 8 16 32
  constexpr int blockxc = HEAD_DIM_QK / cache_vec_size;
  constexpr int num_vec_per_head = HEAD_DIM_QK / vec_size;
  constexpr int blockx = num_vec_per_head < 32 ? num_vec_per_head : 32;

  constexpr int blocky = GROUP_SIZE > 16 ? 16 : GROUP_SIZE;
  const int gridx = bsz * (GROUP_SIZE / blocky);
  
  constexpr int num_threads = blockx * blocky; 

  auto splitkv_kernel = multi_query_decode_attention_kernel<true, NV_TYPE, NV_TYPE, NV_TYPE, num_stages, DEAL_EACH_TIME, GROUP_SIZE, HEAD_DIM_QK, HEAD_DIM_V,
                                                                        BLOCK_SIZE, vec_size, cache_vec_size, blockx, blocky>;
  uint32_t cache_smem_bytes = 0;
  
  const T *shift_bias_ptr = shift_bias ? shift_bias.get().data<T>() : nullptr;
  const T *smooth_weight_ptr = smooth_weight ? smooth_weight.get().data<T>() : nullptr;
  cache_smem_bytes = num_stages * DEAL_EACH_TIME * HEAD_DIM_QK * sizeof(T);
  
  const uint32_t chunk_size = get_max_partition_size(bsz);
  const int num_chunks = div_up(max_dec_len, chunk_size);
  size_t smem_size = cache_smem_bytes + blocky * HEAD_DIM_QK * sizeof(T);

  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
      splitkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }
  const int dev_id = 0;
  int sm_count;
  int act_blocks_per_sm;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &act_blocks_per_sm, splitkv_kernel, num_threads, smem_size);
  assert(act_blocks_per_sm > 1);
  
  const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
  const int num_blocks_need = gridx * num_chunks * kv_num_heads;
  const int max_num_chunks = div_up(num_blocks_per_wave, num_blocks_need);
  const float ratio = static_cast<float>(num_blocks_need) / static_cast<float>(num_blocks_per_wave);

  dim3 grids(gridx, num_chunks, kv_num_heads);
  dim3 blocks(blockx, blocky);
  if (num_chunks <= 1) {
    auto no_splitkv_kernel = multi_query_decode_attention_kernel<false, NV_TYPE, NV_TYPE, NV_TYPE, num_stages, DEAL_EACH_TIME, GROUP_SIZE, HEAD_DIM_QK, HEAD_DIM_V, BLOCK_SIZE, vec_size, 
                                                                             cache_vec_size, blockx, blocky>;
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(
        no_splitkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    no_splitkv_kernel<<<grids, blocks, smem_size, stream>>>(
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(q.data<T>())),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k.data<T>())),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v.data<T>())),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(shift_bias_ptr)),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(smooth_weight_ptr)),
      seq_lens_q.data<int>(),
      seq_lens_kv.data<int>(),
      cum_offsets.data<int>(),
      block_table.data<int>(),
      max_seq_len,
      max_dec_len,
      max_block_num_per_seq,
      softmax_scale,
      in_scale,
      chunk_size,
      nullptr,
      nullptr,
      nullptr,
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(out->data<T>()))
    );

    // CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
  } else {
    auto *allocator = paddle::GetAllocator(q.place());
    phi::Allocator::AllocationPtr tmp_workspace, tmp_m, tmp_d;
    tmp_workspace = allocator->Allocate(
        phi::SizeOf(q.dtype()) *
        static_cast<size_t>(bsz * num_chunks * num_heads * HEAD_DIM_V));
    tmp_m = allocator->Allocate(
        phi::SizeOf(q.dtype()) *
        static_cast<size_t>(bsz * num_chunks * num_heads));
    tmp_d = allocator->Allocate(
        phi::SizeOf(q.dtype()) *
        static_cast<size_t>(bsz * num_chunks * num_heads));

    splitkv_kernel<<<grids, blocks, smem_size, stream>>>(
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(q.data<T>())),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k.data<T>())),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v.data<T>())),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(shift_bias_ptr)),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(smooth_weight_ptr)),
      seq_lens_q.data<int>(),
      seq_lens_kv.data<int>(),
      cum_offsets.data<int>(),
      block_table.data<int>(),
      max_seq_len,
      max_dec_len,
      max_block_num_per_seq,
      softmax_scale,
      in_scale,
      chunk_size,
      reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
      reinterpret_cast<NV_TYPE*>(tmp_m->ptr()),
      reinterpret_cast<NV_TYPE*>(tmp_d->ptr()),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(out->data<T>()))
    );
    // CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
    
    constexpr int mblockx = HEAD_DIM_V / vec_size;
    constexpr int bdy = 256 / mblockx;
    dim3 grids_merge(bsz, num_heads);
    dim3 blocks_merge(mblockx, bdy);
    merge_varlen_multi_chunks_v2_kernel<NV_TYPE, NV_TYPE, vec_size, bdy, HEAD_DIM_V><<<grids_merge, blocks_merge, 0, stream>>>(
      reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
      reinterpret_cast<NV_TYPE*>(tmp_m->ptr()),
      reinterpret_cast<NV_TYPE*>(tmp_d->ptr()),
      seq_lens_q.data<int>(),
      seq_lens_kv.data<int>(),
      cum_offsets.data<int>(),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(shift_bias_ptr)),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(smooth_weight_ptr)),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(out->data<T>())),
      in_scale,
      num_chunks,
      chunk_size,
      max_seq_len,
      num_heads,
      HEAD_DIM_V
    );
  }
  // CHECK(cudaGetLastError());
  // CHECK(cudaDeviceSynchronize());
}

template <typename T>
void DecodeMLAAttentionKernel(
  const AppendAttnMetaData& meta_data,
  const paddle::Tensor &q, // [token_num, num_heads, head_dim]
  const paddle::Tensor &cache_k,
  const paddle::Tensor &cache_v,
  const paddle::optional<paddle::Tensor>& attn_mask,
  const paddle::optional<paddle::Tensor>& shift_bias,
  const paddle::optional<paddle::Tensor>& smooth_weight,
  const paddle::Tensor &seq_lens_q, // q_seq_len is 1
  const paddle::Tensor &seq_lens_kv,
  const paddle::Tensor &padding_offsets,
  const paddle::Tensor &cum_offsets,
  const paddle::Tensor &block_table,
  int max_seq_len,
  int max_dec_len,
  float softmax_scale,
  float in_scale,
  bool causal,
  cudaStream_t &stream,
  paddle::Tensor *out) { 
  const auto token_num = meta_data.token_nums;
  const auto block_size = meta_data.block_size;
  const auto bsz = meta_data.batch_size;
  const auto num_heads = meta_data.q_num_heads;
  const auto group_size = meta_data.q_num_heads / meta_data.kv_num_heads;
  const auto head_dim_qk = meta_data.head_dims;
  const auto head_dim_v = meta_data.head_dims_v;
  const float rope_scale = 0.0;
  const float rope_theta = 0.0;
  const uint32_t deal_each_time = get_cascade_attention_deal_each_time();
  const uint32_t num_stage = get_cascade_attention_num_stages();
  const uint32_t num_threads = get_cascade_attention_num_threads();

  DISPATCH_CAUSAL(causal, CAUSAL,
    {DISPATCH_MLA_GROUP_SIZE(group_size, GROUP_SIZE,
      {DISPATCH_MLA_HEAD_DIM(head_dim_qk, HEAD_DIM_QK,  
        {DISPATCH_MLA_HEAD_DIM(head_dim_v, HEAD_DIM_V, 
          {DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, 
              {DISPATCH_DEAL_EACH_TIME(deal_each_time, DEAL_EACH_TIME,
                  {MultiQueryDecoderAttention<T, GROUP_SIZE, HEAD_DIM_QK, HEAD_DIM_V, BLOCK_SIZE, CAUSAL, 2, 16, DEAL_EACH_TIME>(
                  meta_data, stream, q, cache_k, cache_v, attn_mask, shift_bias, smooth_weight, seq_lens_q, seq_lens_kv, padding_offsets, cum_offsets, 
                  block_table, max_seq_len, max_dec_len, rope_scale, rope_theta, softmax_scale, in_scale, out);})})})})})});
}

template void DecodeMLAAttentionKernel<paddle::bfloat16>(
  const AppendAttnMetaData& meta_data,
  const paddle::Tensor &q, // [token_num, num_heads, head_dim]
  const paddle::Tensor &cache_k,
  const paddle::Tensor &cache_v,
  const paddle::optional<paddle::Tensor>& attn_mask,
  const paddle::optional<paddle::Tensor>& shift_bias,
  const paddle::optional<paddle::Tensor>& smooth_weight,
  const paddle::Tensor &seq_lens_q, // q_seq_len is 1
  const paddle::Tensor &seq_lens_kv,
  const paddle::Tensor &padding_offsets,
  const paddle::Tensor &cum_offsets,
  const paddle::Tensor &block_table,
  int max_seq_len,
  int max_dec_len,
  float softmax_scale,
  float in_scale,
  bool causal,
  cudaStream_t &stream,
  paddle::Tensor *out);

template void DecodeMLAAttentionKernel<paddle::float16>(
  const AppendAttnMetaData& meta_data,
  const paddle::Tensor &q, // [token_num, num_heads, head_dim]
  const paddle::Tensor &cache_k,
  const paddle::Tensor &cache_v,
  const paddle::optional<paddle::Tensor>& attn_mask,
  const paddle::optional<paddle::Tensor>& shift_bias,
  const paddle::optional<paddle::Tensor>& smooth_weight,
  const paddle::Tensor &seq_lens_q, // q_seq_len is 1
  const paddle::Tensor &seq_lens_kv,
  const paddle::Tensor &padding_offsets,
  const paddle::Tensor &cum_offsets,
  const paddle::Tensor &block_table,
  int max_seq_len,
  int max_dec_len,
 float softmax_scale,
  float in_scale,
  bool causal,
  cudaStream_t &stream,
  paddle::Tensor *out);