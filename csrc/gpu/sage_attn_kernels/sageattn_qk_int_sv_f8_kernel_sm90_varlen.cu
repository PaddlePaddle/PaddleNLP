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

#include "sageattn_utils.cuh"
#include "sageattn_fused_varlen.cuh"

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t NUM_THREADS, uint32_t head_dim, QuantGranularity Q_GRAN, QuantGranularity K_GRAN, typename DTypeOut, MaskMode mask_mode = MaskMode::kNone, bool fuse_v_scale=false>
__global__ void qk_int8_sv_f8_attn_varlen_kernel(const __grid_constant__ CUtensorMap tensorMapQ, 
                                                const __grid_constant__ CUtensorMap tensorMapK,
                                                const __grid_constant__ CUtensorMap tensorMapV,
                                                float *__restrict__ Q_scale, float *__restrict__ K_scale, float *__restrict__ V_scale,
                                                DTypeOut* O, 
                                                uint32_t *__restrict__ cu_seqlen,
                                                uint32_t *__restrict__ cu_seqlen_v_padded,
                                                uint32_t stride_h_o, uint32_t stride_seq_o,
                                                const uint32_t qo_len, // max_seqlen_q
                                                const uint32_t kv_len, 
                                                const uint32_t num_kv_groups,
                                                float sm_scale)
{
  static_assert(NUM_THREADS == 128);
  static_assert(CTA_Q <= CTA_K);
  
  const uint32_t warp_idx = (threadIdx.x % 128) / 32;
  const uint32_t lane_id = threadIdx.x % 32;

  constexpr uint32_t num_tiles_q = CTA_Q / 64;
  constexpr uint32_t num_tiles_k = CTA_K / 16;
  constexpr uint32_t num_tiles_qk_inner = head_dim / 32;
  constexpr uint32_t num_tiles_v = head_dim / 16;
  constexpr uint32_t num_tiles_pv_inner = CTA_K / 32;

  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t kv_head_id = head_id / num_kv_groups;

  // return this kernel, in block-level
  const uint32_t bz_seqlen = cu_seqlen[batch_id + 1] - cu_seqlen[batch_id];
  const uint32_t thread_base_token = bx * CTA_Q;
  if (thread_base_token >= bz_seqlen) return;

  sm_scale *= math::log2e;

  extern __shared__ __align__(128) int8_t smem_[];

  int8_t *sQ = (int8_t*)smem_;
  int8_t *sK = (int8_t*)(smem_ + CTA_Q * head_dim * sizeof(int8_t));
  int8_t *sV = (int8_t*)(smem_ + CTA_Q * head_dim * sizeof(int8_t) + CTA_K * head_dim * sizeof(int8_t));
  half *sO = (half*)smem_;

  int32_t RS[num_tiles_q][num_tiles_k][8];
  float RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2];
  float d[num_tiles_q][2];

  uint32_t q_scale_idx, k_scale_idx;

  if constexpr (Q_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_q = gridDim.x;
    q_scale_idx = batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx;
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_q = gridDim.x * 4;
    q_scale_idx = batch_id * num_qo_heads * num_warp_block_q + head_id * num_warp_block_q + bx * 4 + warp_idx;
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_q = gridDim.x * 4;
    q_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) + head_id * (num_warp_block_q * 8) + bx * (4 * 8) + warp_idx * 8 + lane_id / 4;
  }

  if constexpr (K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_block_k + (head_id / num_kv_groups) * num_block_k;
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * (num_block_k * 4) + (head_id / num_kv_groups) * (num_block_k * 4) + lane_id % 4;
  }

  constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp) ? 1 : 4;

  uint32_t Q_idx_lane_base = bx * CTA_Q + warp_idx * 16 + lane_id / 4;

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
    m[fq][0] = -5000000.0f;
    m[fq][1] = -5000000.0f;
    d[fq][0] = 1.0f;
    d[fq][1] = 1.0f;
  }

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
      {
        RO[fq][fv][k] = 0.0f;
      }
    }
  }

  __shared__ __align__(8) uint64_t barrier_Q;
  __shared__ __align__(8) uint64_t barrier_K;
  __shared__ __align__(8) uint64_t barrier_V;

  if (threadIdx.x == 0) {
    init_barrier(&barrier_Q, 1);
    init_barrier(&barrier_K, 1);
    init_barrier(&barrier_V, 1);
  }

  __syncthreads();

  // load Q, K, V
  if (threadIdx.x == 0)
  {
    expect_bytes<(CTA_Q * head_dim) * sizeof(int8_t)>(&barrier_Q);
    expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);
    expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
    //                                     head_dim  seqlen     num_head    bsz
    //                                            |  |             |         |
    // load_async_4D(sQ, &tensorMapQ, &barrier_Q, 0, bx * CTA_Q, head_id, batch_id); // original input tensor map: [bsz, num_head, seqlen. head_dim]
    // load_async_4D(sK, &tensorMapK, &barrier_K, 0, 0, kv_head_id, batch_id);
    load_async_3D(sQ, &tensorMapQ, &barrier_Q, 0, bx * CTA_Q + cu_seqlen[batch_id], head_id);   // now shape: [num_head, total_seqlen, head_dim]
    load_async_3D(sK, &tensorMapK, &barrier_K, 0, cu_seqlen[batch_id], kv_head_id);

    //                                       seqlen  head_dim num_head    bsz
    //                                            |  |         |         |
    // load_async_4D(sV, &tensorMapV, &barrier_V, 0, 0, kv_head_id, batch_id);
    load_async_3D(sV, &tensorMapV, &barrier_V, cu_seqlen_v_padded[batch_id], 0, kv_head_id);
  }

  float q_scale = Q_scale[q_scale_idx];
  float original_sm_scale = sm_scale;

  // wait for Q
  wait(&barrier_Q, 0);

  const uint32_t num_iterations = div_ceil(
      mask_mode == MaskMode::kCausal
          ? min(kv_len, (bx + 1) * CTA_Q)
          : kv_len,
      CTA_K);

  int p = 1;
  for (uint32_t iter = 1; iter < num_iterations; iter++)
  { 
    p ^= 1;

    float dequant_scale = q_scale * K_scale[k_scale_idx + (iter - 1) * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // wait for K
    wait(&barrier_K, p);

    // compute QK^T
    wgmma::warpgroup_arrive();
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      int8_t *sQ_local = sQ + fq * 64 * head_dim;
      wgmma::wgmma_s8s8s32<CTA_K, 0, head_dim>(RS[fq], sQ_local, sK);
#pragma unroll
      for (int k_it = 1; k_it < num_tiles_qk_inner; k_it++)
      {
        wgmma::wgmma_s8s8s32<CTA_K, 1, head_dim>(RS[fq], &sQ_local[k_it*32], &sK[k_it*32]);
      }
    }
    wgmma::warpgroup_commit_batch();
    wgmma::warpgroup_wait<0>();

    // load K
    if (threadIdx.x == 0)
    {
      expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_K);
      // load_async_4D(sK, &tensorMapK, &barrier_K, 0, iter * CTA_K, kv_head_id, batch_id);
      load_async_3D(sK, &tensorMapK, &barrier_K, 0, iter * CTA_K + cu_seqlen[batch_id], kv_head_id);
    }

    // convert RS to float
    float RS_f32[num_tiles_q][num_tiles_k][8];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
        }
      }
    }

    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, m, d, sm_scale);

    // accumulate d on thread basis
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unrol
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
        d[fq][0] += (RS_f32[fq][fk][0] + RS_f32[fq][fk][1] + RS_f32[fq][fk][4] + RS_f32[fq][fk][5]);
        d[fq][1] += (RS_f32[fq][fk][2] + RS_f32[fq][fk][3] + RS_f32[fq][fk][6] + RS_f32[fq][fk][7]);
      }
    }

    uint32_t RS_f8[num_tiles_q][num_tiles_pv_inner][4];
    RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);

    // wait for V
    wait(&barrier_V, p);

    float RO_temp[num_tiles_q][num_tiles_v][8];
    wgmma::warpgroup_arrive();
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      wgmma::wgmma_f8f8f32<head_dim, 0, CTA_K>(RO_temp[fq], RS_f8[fq][0], &sV[0]);
#pragma unroll
      for (uint32_t v_it = 1; v_it < num_tiles_pv_inner; v_it++)
      {
        wgmma::wgmma_f8f8f32<head_dim, 1, CTA_K>(RO_temp[fq], RS_f8[fq][v_it], &sV[v_it * 32]);
      }
    }

    wgmma::warpgroup_commit_batch();
    wgmma::warpgroup_wait<0>();

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RO[fq][fv][k] += RO_temp[fq][fv][k];
        }
      }
    }

    // load V
    if (threadIdx.x == 0)
    {
      expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_V);
      // load_async_4D(sV, &tensorMapV, &barrier_V, iter * CTA_K, 0, kv_head_id, batch_id);  // original v shape: [b, num_head, headdim, seqlen, ]
      load_async_3D(sV, &tensorMapV, &barrier_V, iter * CTA_K + cu_seqlen_v_padded[batch_id], 0, kv_head_id);
    }
  }

  { 
    p ^= 1;

    float dequant_scale = q_scale * K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset];
    sm_scale = original_sm_scale;

    // wait for K
    wait(&barrier_K, p);

    // compute QK^T
    wgmma::warpgroup_arrive();
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      int8_t *sQ_local = sQ + fq * 64 * head_dim;
      wgmma::wgmma_s8s8s32<CTA_K, 0, head_dim>(RS[fq], sQ_local, sK);
#pragma unroll
      for (int k_it = 1; k_it < num_tiles_qk_inner; k_it++)
      {
        wgmma::wgmma_s8s8s32<CTA_K, 1, head_dim>(RS[fq], &sQ_local[k_it*32], &sK[k_it*32]);
      }
    }
    wgmma::warpgroup_commit_batch();
    wgmma::warpgroup_wait<0>();

    // convert RS to float
    float RS_f32[num_tiles_q][num_tiles_k][8];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    // masking
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          const uint32_t q_idx = Q_idx_lane_base + fq * 64 + 8 * ((k % 4) / 2);
          const uint32_t k_idx = (num_iterations - 1) * CTA_K + fk * 16 + 2 * (lane_id % 4) + 8 * (k / 4) + k % 2;

          bool is_out_of_bounds;

          if constexpr (mask_mode == MaskMode::kCausal)
          {
            is_out_of_bounds = (k_idx > q_idx) || (k_idx >= kv_len);
          }
          else
          {
            is_out_of_bounds = (k_idx >= kv_len);
          }

          if (is_out_of_bounds)
          {
            RS_f32[fq][fk][k] = -5000000.0f;
          }
        }
      }
    }

    // inside this function, the primritive `__shfl_xor` was used, so the kernel cannot be returned unless finished the warp-level executing
    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, m, d, sm_scale);

    // accumulate d on thread basis
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unrol
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
        d[fq][0] += (RS_f32[fq][fk][0] + RS_f32[fq][fk][1] + RS_f32[fq][fk][4] + RS_f32[fq][fk][5]);
        d[fq][1] += (RS_f32[fq][fk][2] + RS_f32[fq][fk][3] + RS_f32[fq][fk][6] + RS_f32[fq][fk][7]);
      }
    }

    uint32_t RS_f8[num_tiles_q][num_tiles_pv_inner][4];
    RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);

    // wait for V
    wait(&barrier_V, p);

    float RO_temp[num_tiles_q][num_tiles_v][8];
    wgmma::warpgroup_arrive();
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      wgmma::wgmma_f8f8f32<head_dim, 0, CTA_K>(RO_temp[fq], RS_f8[fq][0], &sV[0]);
#pragma unroll
      for (uint32_t v_it = 1; v_it < num_tiles_pv_inner; v_it++)
      {
        wgmma::wgmma_f8f8f32<head_dim, 1, CTA_K>(RO_temp[fq], RS_f8[fq][v_it], &sV[v_it * 32]);
      }
    }

    wgmma::warpgroup_commit_batch();
    wgmma::warpgroup_wait<0>();

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RO[fq][fv][k] += RO_temp[fq][fv][k];
        }
      }
    }
  }

  normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, m, d);

  if constexpr (fuse_v_scale)
  {
    float v_scale[4];
    float *V_scale_base_ptr = V_scale +  batch_id * (num_qo_heads / num_kv_groups) * head_dim + (head_id / num_kv_groups) * head_dim + (lane_id % 4 ) * 2;
  #pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      ((float2*)v_scale)[0] = *((float2*)(V_scale_base_ptr + fv * 16));
      ((float2*)v_scale)[1] = *((float2*)(V_scale_base_ptr + fv * 16 + 8));

  #pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        RO[fq][fv][0] *= v_scale[0];
        RO[fq][fv][1] *= v_scale[1];
        RO[fq][fv][2] *= v_scale[0];
        RO[fq][fv][3] *= v_scale[1];
        RO[fq][fv][4] *= v_scale[2];
        RO[fq][fv][5] *= v_scale[3];
        RO[fq][fv][6] *= v_scale[2];
        RO[fq][fv][7] *= v_scale[3];
      }
    }
  }

  // re-write the output idx
  DTypeOut *O_lane_ptr = O + cu_seqlen[batch_id] * stride_seq_o + head_id * stride_h_o + (bx * CTA_Q + warp_idx * 16 + (lane_id / 4)) * stride_seq_o + (lane_id % 4) * 2 ;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < head_dim/16; fv++)
    { 
      // if (Q_idx_lane_base + fq * 64 < qo_len) -> qo_len is the original max_seq_len
      if (Q_idx_lane_base + fq * 64 < bz_seqlen)  // -> shift to this seqlen
      {
        if constexpr (std::is_same<DTypeOut, half>::value)
        {
          ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16))[0] = __float22half2_rn(((float2*)(RO[fq][fv]))[0]);
          ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8))[0] = __float22half2_rn(((float2*)(RO[fq][fv]))[2]);
        }
        else
        {
          ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16))[0] = __float22bfloat162_rn(((float2*)(RO[fq][fv]))[0]);
          ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8))[0] = __float22bfloat162_rn(((float2*)(RO[fq][fv]))[2]);  
        }
      }
      
      if (Q_idx_lane_base + fq * 64 + 8 < bz_seqlen)  // -> shift to this seqlen
      {
        if constexpr (std::is_same<DTypeOut, half>::value)
        {
          ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 * stride_seq_o))[0] = __float22half2_rn(((float2*)(RO[fq][fv]))[1]);
          ((half2*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 + 8 * stride_seq_o))[0] = __float22half2_rn(((float2*)(RO[fq][fv]))[3]);
        }
        else
        {
          ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 * stride_seq_o))[0] = __float22bfloat162_rn(((float2*)(RO[fq][fv]))[1]);
          ((nv_bfloat162*)(O_lane_ptr + fq * 64 * stride_seq_o + fv * 16 + 8 + 8 * stride_seq_o))[0] = __float22bfloat162_rn(((float2*)(RO[fq][fv]))[3]);      
        }
      }
    }
  }
}

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90_varlen_fwd(
                    paddle::Tensor& query,      // total_seqlen x num_head x head_dim
                    paddle::Tensor& key,        // total_seqlen x num_head x head_dim
                    paddle::Tensor& value,      // head_dim x num_head x total_seqlen_padded
                    paddle::Tensor& output,     // total_seqlen x num_head x head_dim
                    paddle::Tensor& query_scale,  // b, h_qk, seqlen // 64
                    paddle::Tensor& key_scale,    // b, h_qk, seqlen // 64
                    paddle::Tensor& value_scale,  // b, h_kv, head_dim
                    paddle::Tensor& cu_seqlen_q,
                    paddle::Tensor& cu_seqlen_v_padded,
                    int max_seqlen_q,
                    int max_seqlen_k,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(value_scale);

  CHECK_LASTDIM_CONTIGUOUS(query);
  CHECK_LASTDIM_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(value_scale);

  CHECK_DTYPE(query, paddle::DataType::INT8);
  CHECK_DTYPE(key, paddle::DataType::INT8);
  CHECK_DTYPE(value, paddle::DataType::FLOAT8_E4M3FN);
  CHECK_DTYPE(query_scale, paddle::DataType::FLOAT32);
  CHECK_DTYPE(key_scale, paddle::DataType::FLOAT32);
  CHECK_DTYPE(value_scale, paddle::DataType::FLOAT32);

  CHECK_DIMS(query, 3);
  CHECK_DIMS(key, 3);
  CHECK_DIMS(value, 3);
  CHECK_DIMS(output, 3);

  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(value_scale, 3);

  const int batch_size = cu_seqlen_q.shape()[0] - 1;
  const int head_dim = query.shape()[2];

  int qo_len, kv_len, padded_kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_h_q, stride_seq_k, stride_h_k;
  int stride_h_v, stride_d_v;
  int stride_seq_o, stride_h_o;

  qo_len = max_seqlen_q;
  kv_len = max_seqlen_k;

  num_qo_heads = query.shape()[1];
  num_kv_heads = key.shape()[1];

  stride_seq_q = query.strides()[0];
  stride_seq_k = key.strides()[0];

  stride_seq_o = output.strides()[0];
  stride_h_o = output.strides()[1];

  stride_h_q = query.strides()[1];
  stride_h_k = key.strides()[1];

  stride_h_v = value.strides()[1];
  stride_d_v = value.strides()[0];
  

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());  
  }

  paddle::Tensor lse = paddle::empty({1}, paddle::DataType::FLOAT32);
  if (return_lse)
  {
    lse = paddle::empty({batch_size, num_qo_heads, qo_len}, paddle::DataType::FLOAT32);
  }

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  auto output_dtype = output.dtype();

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
          constexpr int CTA_Q = 64;
          constexpr int CTA_K = 128;
          constexpr int NUM_THREADS = 128;

          constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

          assert(value.shape()[2] >= div_ceil(kv_len, CTA_K) * CTA_K);  // check if 128-padding is applied

          if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp))
          {
            CHECK_SHAPE(query_scale, batch_size, num_qo_heads, static_cast<long>(div_ceil(qo_len, CTA_Q) * (NUM_THREADS / 32)));
            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, static_cast<long>(div_ceil(kv_len, CTA_K)));
          }
          else if constexpr (QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread))
          {
            CHECK_SHAPE(query_scale, batch_size, num_qo_heads, static_cast<long>(div_ceil(qo_len, CTA_Q) * (NUM_THREADS / 32) * 8));
            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, static_cast<long>(div_ceil(kv_len, CTA_K) * 4));    
          }
          else
          {
            static_assert(QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerWarp) || QK_QUANT_GRAN == static_cast<int>(QuantGranularity::kPerThread), "Unsupported quantization granularity");
          }

          CHECK_SHAPE(value_scale, batch_size, num_kv_heads, head_dim);

          // TODO: we need to change here
          CUtensorMap tma_map_Q = create_tensor_map_3D<CTA_Q, HEAD_DIM>(reinterpret_cast<int8_t*>(query.data()), num_qo_heads, query.shape()[0], HEAD_DIM, 
            stride_h_q, stride_seq_q);
          CUtensorMap tma_map_K = create_tensor_map_3D<CTA_K, HEAD_DIM>(reinterpret_cast<int8_t*>(key.data()), num_kv_heads, key.shape()[0], HEAD_DIM, 
            stride_h_k, stride_seq_k);
          CUtensorMap tma_map_V = create_tensor_map_3D<HEAD_DIM, CTA_K>(reinterpret_cast<int8_t*>(value.data()), num_kv_heads, HEAD_DIM, value.shape()[2], 
            stride_h_v, stride_d_v);

          auto* kernel = qk_int8_sv_f8_attn_varlen_kernel<CTA_Q, CTA_K, NUM_THREADS, HEAD_DIM,  static_cast<QuantGranularity>(QK_QUANT_GRAN), static_cast<QuantGranularity>(QK_QUANT_GRAN), DTypeOut, mask_mode, true>;
          size_t sMemSize = CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t);
          cudaFuncSetAttribute(
              kernel,
              cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize);
          
          dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);   // [max_seqlen / CTA_Q, num_heads, batch]
          kernel<<<grid, NUM_THREADS, sMemSize>>>(
            tma_map_Q,
            tma_map_K,
            tma_map_V,
            reinterpret_cast<float*>(query_scale.data()),
            reinterpret_cast<float*>(key_scale.data()),
            reinterpret_cast<float*>(value_scale.data()),
            reinterpret_cast<DTypeOut*>(output.data()),
            reinterpret_cast<uint32_t*>(cu_seqlen_q.data()),
            reinterpret_cast<uint32_t*>(cu_seqlen_v_padded.data()),
            stride_h_o, stride_seq_o,
            qo_len, kv_len, num_kv_groups, sm_scale);
        });
      });
    });
  });

  return {lse};
}

std::vector<paddle::Tensor> sage_attention_varlen_fwd(paddle::Tensor& q,          // total_seqlen x num_head x head_dim
                                                      paddle::Tensor& k,          // total_seqlen x num_head x head_dim
                                                      paddle::Tensor& v,          // total_seqlen x num_head x head_dim
                                                      paddle::Tensor& cu_seqlen_q,
                                                      paddle::Tensor& cu_seqlen_v_padded,
                                                      paddle::Tensor& km,
                                                      paddle::optional<paddle::Tensor>& vm,
                                                      const std::vector<int64_t>& split_vec,
                                                      int max_seqlen_q,
                                                      int max_seqlen_k,
                                                      int total_seqlen_v_padded,
                                                      float sm_scale,
                                                      std::string qk_quant_gran,
                                                      std::string pv_accum_dtype,
                                                      int tensor_layout,
                                                      bool is_causal,
                                                      bool smooth_k,
                                                      bool smooth_v,
                                                      bool return_lse)
{
  int _is_causal = int(is_causal);
  int _qk_quant_gran = (qk_quant_gran == std::string("per_thread")) ? 3 : 2;
  int _return_lse = int(return_lse);

  PD_CHECK(q.shape()[2] == 64 || q.shape()[2] == 128, "head_dim must be either 64 or 128");
  PD_CHECK(q.strides()[2] == 1 && k.strides()[2] == 1 && v.strides()[2] == 1, "Last dim of qkv must be contiguous.");

  // split, padding to 128-align, and concat
  std::vector<paddle::Tensor> v_splited = paddle::split(v, split_vec, {0}); // split along the total_seqlen axis.
  for (auto& vi : v_splited) {
    int v_pad_len = (vi.shape()[0] % 128 != 0) ? (128 - vi.shape()[0] % 128) : 0;
    if (v_pad_len > 0) {
      vi = paddle::concat({vi, paddle::zeros({v_pad_len, vi.shape()[1], vi.shape()[2]}, vi.dtype(), paddle::GPUPlace())}, {0}); // along the total_seqlen axis
    }
  }
  paddle::Tensor v_padded = paddle::concat(v_splited, {0}); // final concat along the total_seqlen axis

  constexpr int BLKQ = 64;
  int WARPQ = 16;
  constexpr int BLKK = 128;
  std::vector<paddle::Tensor>&& quant_qk_results = per_warp_int8_varlen_cuda_fwd(q, k, cu_seqlen_q, km, max_seqlen_q, max_seqlen_k, BLKQ, WARPQ, BLKK); // q_int8, q_scale, k_int8, k_scale

  // v was padded, so we cannot use v for output shape
  paddle::Tensor o = paddle::empty(q.shape(), q.dtype(), paddle::GPUPlace()); // so far, the shape of v is not permutted and transposed. Still [total_seqlen, num_head, head_dim]

  std::vector<paddle::Tensor>&& quant_vfp8_results = per_channel_varlen_fp8(v_padded, 
      cu_seqlen_q, 
      cu_seqlen_v_padded, 
      max_seqlen_k, 
      total_seqlen_v_padded, 
      tensor_layout, 448.0, smooth_v);

  qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90_varlen_fwd(quant_qk_results[0], // q
    quant_qk_results[2],    // k
    quant_vfp8_results[0],  // v_padded, after quant fp8
    o,                      // o
    quant_qk_results[1],    // q_scale
    quant_qk_results[3],    // k_scale
    quant_vfp8_results[1],  // v_scale
    cu_seqlen_q, 
    cu_seqlen_v_padded,
    max_seqlen_q,
    max_seqlen_k,
    tensor_layout, 
    _is_causal, 
    _qk_quant_gran, 
    sm_scale, 
    _return_lse);

  return {o};  // debug: return qkv
}