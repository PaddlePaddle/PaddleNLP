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

#include <type_traits>

#include "paddle/extension.h"

#include "sageattn_utils.cuh"
#include "sageattn_fused_varlen.cuh"

#define PACK_SIZE_QK 16 // as if it is int8
#define PACK_SIZE_V 16  // fp8
#define PACK_SIZE_O 8   // fp16

// treat as if int8 tensor core
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

// fp8 tensor core
#define MMA_SV_M 16
#define MMA_SV_N 16
#define MMA_SV_K 32

// kernel impl
// qk_int_sv_f8 impl
// inner impl
template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, SADataType DTypeQK, QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
        typename DTypeSVAccum = float, bool use_inst_buffer = false, typename DTypeOut = half, ComputeUnit DenominatorAccumUnit, MaskMode mask_mode = MaskMode::kNone, bool return_lse = false, bool fuse_v_scale=false, bool fuse_v_mean=false>
__global__ void qk_int_sv_f8_attn_varlen_kernel(int8_t *__restrict__ Q, int8_t *__restrict__ K, int8_t *__restrict__ V, DTypeOut *__restrict__ O, float *__restrict__ Lse,
                      float *__restrict__ Q_scale, float *__restrict__ K_scale, float *__restrict__ V_scale, float *__restrict__ V_mean,
                      uint32_t *__restrict__ cu_seqlen,
                      const uint32_t qo_len, const uint32_t kv_len, // notice: this is max_seqlen
                      const uint32_t num_kv_groups,
                      const uint32_t stride_seq_q, const uint32_t stride_h_q, // no stride bz
                      const uint32_t stride_seq_k, const uint32_t stride_h_k,
                      const uint32_t stride_h_v, const uint32_t stride_d_v,
                      const uint32_t stride_seq_o, const uint32_t stride_h_o,
                      float sm_scale)
{
  // compile time check
  static_assert(DTypeQK == SADataType::kInt8 || DTypeQK == SADataType::kInt4, "DTypeQK must be int8 or int4");
  static_assert(Q_GRAN == QuantGranularity::kPerBlock || Q_GRAN == QuantGranularity::kPerWarp || Q_GRAN == QuantGranularity::kPerThread, "Q_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp || K_GRAN == QuantGranularity::kPerThread, "K_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
  static_assert(std::is_same<DTypeSVAccum, float>::value, "DTypeSVAccum must be float, half is WIP");
  static_assert(std::is_same<DTypeOut, half>::value || std::is_same<DTypeOut, nv_bfloat16>::value, "DTypeOut must be half or nv_bfloat16");
  static_assert(CTA_K % 64 == 0);
  static_assert(CTA_Q / CTA_K <= 2); // for efficient causal implementation

  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = CTA_K / WARP_K;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = WARP_K / MMA_QK_N;
  constexpr uint32_t num_tiles_qk_inner = (DTypeQK == SADataType::kInt8) ? (head_dim / MMA_QK_K) : (head_dim / 2 / MMA_QK_K);
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;

  constexpr uint32_t QK_SMEM_STRIDE = (DTypeQK == SADataType::kInt8) ? (head_dim) : (head_dim / 2);
  constexpr uint32_t O_SMEM_STRIDE = head_dim;
  //                       for fp16: head_dim
  constexpr uint32_t V_SMEM_STRIDE = CTA_K;

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();

  // maximize L2 hit rate
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;

  // transfer to base 2 instead of base e with better numerical efficiency
  sm_scale *= math::log2e;

  // RS holds the fragment of S
  int32_t RS[num_tiles_q][num_tiles_k][8];
  DTypeSVAccum RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2]; // max
  float d[num_tiles_q][2]; // denominator

  uint32_t q_scale_idx, k_scale_idx;

  if constexpr (Q_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_q = gridDim.x;
    q_scale_idx = batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx;
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * num_warp_block_q + head_id * num_warp_block_q + bx * num_warps_q + get_warp_idx_q<num_warps_q, num_warps_k>();
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) + head_id * (num_warp_block_q * 8) + bx * (num_warps_q * 8) + get_warp_idx_q<num_warps_q, num_warps_k>() * 8 + lane_id / 4;
  }

  if constexpr (K_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_block_k + (head_id / num_kv_groups) * num_block_k;
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_warp_block_k + (head_id / num_kv_groups) * num_warp_block_k + get_warp_idx_k<num_warps_q, num_warps_k>();
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * (num_warp_block_k * 4) + (head_id / num_kv_groups) * (num_warp_block_k * 4) + get_warp_idx_k<num_warps_q, num_warps_k>() * 4 + lane_id % 4;
  }

  constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerBlock) ? 1 : (K_GRAN == QuantGranularity::kPerWarp) ? (CTA_K / WARP_K) : (CTA_K / WARP_K) * 4;

  // initialize o, m, d
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      if constexpr (std::is_same<DTypeSVAccum, float>::value)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {        
          RO[fq][fv][k] = 0.0f;
        }
      }
      else if constexpr (std::is_same<DTypeSVAccum, half>::value)
      {
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        {
          ((int32_t*)RO[fq][fv])[k] = 0;
        }
      }
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      m[fq][k] = -5000000.0f;
      d[fq][k] = 1.0f;
    }
  }

  constexpr uint32_t K_smem_idx_offset = CTA_Q;
  constexpr uint32_t V_smem_idx_offset = CTA_Q + CTA_K;

  constexpr SwizzleMode swizzle_mode_QK = (QK_SMEM_STRIDE == 32) ? SwizzleMode::k32B : (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(smem + K_smem_idx_offset * QK_SMEM_STRIDE);
  //                                             for fp16: 32
  constexpr SwizzleMode swizzle_mode_V = (V_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(smem + V_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_O = (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

  constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2 : (QK_SMEM_STRIDE == 64) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_QK = (QK_SMEM_STRIDE == 32) ? 16 : (QK_SMEM_STRIDE == 64) ? 8 : 4;
  //                                                         for fp16: 32
  constexpr uint32_t global_to_shared_line_lanes_V = (V_SMEM_STRIDE == 64) ? 4 : 8;
  //                                                                  for fp16: 32
  constexpr uint32_t global_to_shared_copy_lines_per_warp_V = (V_SMEM_STRIDE == 64) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_O = (O_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_O = (O_SMEM_STRIDE == 32) ? 8 : 4;

  constexpr uint32_t QK_smem_iters_row = QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
  constexpr uint32_t Q_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t V_smem_iters_row = V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
  //                          for fp16: CTA_K
  constexpr uint32_t V_smem_iters_col = head_dim / (num_warps * global_to_shared_copy_lines_per_warp_V);
  constexpr uint32_t O_smem_iters_row = O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

  int8_t *Q_lane_base_ptr = Q + batch_id * stride_bz_q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_q + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  int8_t *K_lane_base_ptr = K + batch_id * stride_bz_k + (head_id / num_kv_groups) * stride_h_k + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_k + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  //                                                                for fp16: CTA_K / num_warps * warp_id * stride_seq_v + lane_id / global_to_shared_line_lanes_V * stride_seq_v
  int8_t *V_lane_base_ptr = V + batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v + head_dim / num_warps * warp_id * stride_d_v + lane_id / global_to_shared_line_lanes_V * stride_d_v + (lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V;
  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t K_smem_offset_load = smem_K.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * K_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t V_smem_offset_load = smem_V.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_V * V_smem_iters_col + lane_id / global_to_shared_line_lanes_V, lane_id % global_to_shared_line_lanes_V);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16, lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  // for fp 16:
  // uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 16, lane_id / 16);
  uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K / PACK_SIZE_V + (lane_id / 8) % 2);

  // for causal masking
  uint32_t Q_idx_lane_base = bx * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
  uint32_t K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);

  // for loading
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t K_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;

  const uint32_t num_iterations = div_ceil(
      mask_mode == MaskMode::kCausal
          ? min(kv_len, (bx + 1) * CTA_Q)
          : kv_len,
      CTA_K);

  // load Q with predicate
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, Q_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
    &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q, Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  // for num_tiles_qk_inner = 1, we load all Qs in register
  uint32_t RQ[num_tiles_q][4];
  if constexpr (num_tiles_qk_inner == 1)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      smem_Q.ldmatrix_m8n8x4(Q_smem_offset_mma, RQ[fq]);
      Q_smem_offset_mma = smem_Q.advance_offset_by_row<16>(Q_smem_offset_mma);
    }
  }

  // load K with predicate
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
    &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  float q_scale = Q_scale[q_scale_idx];

  float original_sm_scale = sm_scale;
  float dequant_scale = q_scale * K_scale[k_scale_idx + 0 * k_scale_advance_offset];

  sm_scale = original_sm_scale * dequant_scale;

  // load V
  // ! we assume that V is padded. If not, there might be illegal memory access or nan issue.
  // for fp16: 
  // load_global_to_share                stride_seq_v
  load_fp8_V_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
    &V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
  cp_async::commit_group();

  K_load_idx_lane_base += CTA_K;

#pragma unroll
  for (uint32_t iter = 1; iter < num_iterations - 1; iter++)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }
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

    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, m, d, sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, true, false>(RS_f32, RO, m, d, sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
    RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, d);
    }

    __syncthreads();

    // load K without predicate
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
    cp_async::commit_group();

    dequant_scale = q_scale * K_scale[k_scale_idx + iter * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;
    
    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // for fp16:
    // compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
    //   smem_V, RS_f16, RO, d, V_smem_offset_mma);
    if constexpr (!use_inst_buffer)
    {
      compute_fp8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);
    }
    else
    {
      compute_fp8_sv_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);   
    }
    __syncthreads();
    // load V
    // for fp16: 
    // load_global_to_share                stride_seq_v
    load_fp8_V_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
    cp_async::commit_group();
  
    K_load_idx_lane_base += CTA_K;
  }

  // second last iter, apply causal mask
  if (num_iterations > 1)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

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

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    // apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, m, d, original_sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, true, false>(RS_f32, RO, m, d, original_sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
    RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, d);
    }

    __syncthreads();

    // load K with predicate
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    dequant_scale = q_scale * K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // for fp16:
    // compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
    //   smem_V, RS_f16, RO, d, V_smem_offset_mma);
    if constexpr (!use_inst_buffer)
    {
      compute_fp8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);
    }
    else
    {
      compute_fp8_sv_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);
    }

    __syncthreads();
    // load V
    // for fp16: 
    // load_global_to_share                stride_seq_v
    load_fp8_V_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
    cp_async::commit_group();
    K_load_idx_lane_base += CTA_K;
  }

  // last iter, apply causal mask and out of bound mask
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

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

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS_f32, RO, m, d, original_sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, true, false>(RS_f32, RO, m, d, original_sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
    RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, d);
    }

    // ensure V is ready
    cp_async::wait_group<0>();
    __syncthreads();

    // for fp16:
    // compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
    //   smem_V, RS_f16, RO, d, V_smem_offset_mma);
    if constexpr (!use_inst_buffer)
    {
      compute_fp8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);
    }
    else
    {
      compute_fp8_sv_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);
    }

    __syncthreads();

  }

  // TODO: thread block sync mdo state for num_warps_k > 0. Then only one thread block needs to do the final saving.

  normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, m, d);

  // ! here we just implement the case for fp32 acumulation
  if constexpr (fuse_v_scale)
  {
    float v_scale[4];
    float *V_scale_base_ptr = V_scale + batch_id * (num_qo_heads / num_kv_groups) * head_dim + (head_id / num_kv_groups) * head_dim + (lane_id % 4 ) * 2;
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

  if constexpr (fuse_v_mean)
  {
    float v_mean[4];
    float *V_mean_base_ptr = V_mean + batch_id * (num_qo_heads / num_kv_groups) * head_dim + (head_id / num_kv_groups) * head_dim + (lane_id % 4 ) * 2;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      ((float2*)v_mean)[0] = *((float2*)(V_mean_base_ptr + fv * 16));
      ((float2*)v_mean)[1] = *((float2*)(V_mean_base_ptr + fv * 16 + 8));
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        RO[fq][fv][0] += v_mean[0];
        RO[fq][fv][1] += v_mean[1];
        RO[fq][fv][2] += v_mean[0];
        RO[fq][fv][3] += v_mean[1];
        RO[fq][fv][4] += v_mean[2];
        RO[fq][fv][5] += v_mean[3];
        RO[fq][fv][6] += v_mean[2];
        RO[fq][fv][7] += v_mean[3];
      }
    }
  }

  // save the result to shared memory
  uint32_t smem_O_row_base = get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      uint32_t offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));

      if constexpr (std::is_same<DTypeSVAccum, float>::value)
      {
        // convert RO to half
        uint32_t RO_f16[4];
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        {
          if constexpr (std::is_same<DTypeOut, half>::value)
          {
            ((half2*)RO_f16)[k] = __float22half2_rn(((float2*)RO[fq][fv])[k]);
          }
          else
          {
            ((nv_bfloat162*)RO_f16)[k] = __float22bfloat162_rn(((float2*)RO[fq][fv])[k]);
          }
        }

        ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
        ((uint32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];

        offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O) + 1);
        ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[2];
        ((uint32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[3];
      }
      else if constexpr (std::is_same<DTypeSVAccum, half>::value)
      { 
        // ! need to convert to bf16 if necessary
        // ((int32_t*)(smem_O.base + offset_O))[lane_id % 4] = ((int32_t*)RO[fq][fv])[0];
        // ((int32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = ((int32_t*)RO[fq][fv])[1];

        // offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O) + 1);
        // ((int32_t*)(smem_O.base + offset_O))[lane_id % 4] = ((int32_t*)RO[fq][fv])[2];
        // ((int32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = ((int32_t*)RO[fq][fv])[3];
      }
    }
  }

  // ! do we need to sync here?
  __syncwarp();

  // shared memory to global memory
  DTypeOut *O_lane_ptr = O + batch_id * stride_bz_o + head_id * stride_h_o + (bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() + lane_id / global_to_shared_line_lanes_O) * stride_seq_o + lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O;
  uint32_t offset_O = smem_O.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / global_to_shared_line_lanes_O, lane_id % global_to_shared_line_lanes_O);
  uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_O;

#pragma unroll
  for (uint32_t i = 0; i < O_smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < O_smem_iters_row; j++)
    {
      if (O_load_idx_lane_base < qo_len)
      {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += (global_to_shared_line_lanes_O * PACK_SIZE_O);
      offset_O = smem_O.advance_offset_by_column<global_to_shared_line_lanes_O>(offset_O);
    }

    offset_O = smem_O.advance_offset_by_row<global_to_shared_copy_lines_per_warp_O>(offset_O - (O_smem_iters_row * global_to_shared_line_lanes_O));
    O_lane_ptr += ((global_to_shared_copy_lines_per_warp_O * stride_seq_o) - (O_smem_iters_row * global_to_shared_line_lanes_O * PACK_SIZE_O));
    O_load_idx_lane_base += global_to_shared_copy_lines_per_warp_O;
  }

  if constexpr (return_lse)
  { 
    // ! this only works for num_tiles_q = 2
    uint32_t lse_idx = bx * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>();
    float *lse_lane_ptr = Lse + batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_idx;
    uint32_t fq = (lane_id % 4) / 2;
    uint32_t k = (lane_id % 4) % 2;

    if (lse_idx < qo_len)
    {
      lse_lane_ptr[0] = (math::ptx_log2(d[fq][k]) + m[fq][k] - S_FP8_OFFSET);
    }
  }
} // kernel impl end

//
//  =========== Exposed to Outside API - ARCH: SM89 ===========
//

std::vector<paddle::Tensor> sage_attention_varlen_fwd(paddle::Tensor& q,        // total_seqlen x num_head x head_dim
                                                    paddle::Tensor& k,          // total_seqlen x num_head x head_dim
                                                    paddle::Tensor& v,          // total_seqlen x num_head x head_dim
                                                    paddle::Tensor& v_padded,   // total_seqlen_padded x num_head x head_dim
                                                    paddle::Tensor& cu_seqlen_q,
                                                    paddle::Tensor& cu_seqlen_v,
                                                    paddle::Tensor& cu_seqlen_v_padded,
                                                    paddle::Tensor& km,
                                                    paddle::optional<paddle::Tensor>& vm,
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

  PD_CHECK(pv_accum_dtype == std::string("fp32+fp32") || pv_accum_dtype == std::string("fp32"), "pv_accum_dtype must be either fp32 or fp32+fp32");
  auto pv_accum_dtype_const = (pv_accum_dtype == std::string("fp32+fp32")) ? paddle::DataType::UNDEFINED : paddle::DataType::FLOAT32;

  PD_CHECK(q.shape()[2] == 64 || q.shape()[2] == 128, "head_dim must be either 64 or 128");
  PD_CHECK(q.strides()[2] == 1 && k.strides()[2] == 1 && v.strides()[2] == 1, "Last dim of qkv must be contiguous.");

  int seq_dim = (tensor_layout == 0) ? 1 : 2;

  // quant q, k -> q_int8, k_int8
  constexpr int BLKQ = 128;
  int WARPQ = 32;
  constexpr int BLKK = 64;
  std::vector<paddle::Tensor>&& quant_qk_results = per_warp_int8_varlen_cuda_fwd(q, k, cu_seqlen_q, km, max_seqlen_q, max_seqlen_k, BLKQ, WARPQ, BLKK); // q_int8, q_scale, k_int8, k_scale

  paddle::Tensor o = paddle::empty(v.shape(), v.dtype(), paddle::GPUPlace());

  if (pv_accum_dtype_const == paddle::DataType::UNDEFINED) {
    if (smooth_v) smooth_v = false;
  }

  std::vector<paddle::Tensor>&& quant_vfp8_results = per_channel_varlen_fp8(v, tensor_layout, 448.0, smooth_v);

  switch (pv_accum_dtype_const) {
    case paddle::DataType::FLOAT32: {
      if (smooth_v) {
        qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_fwd(quant_qk_results[0], quant_qk_results[2], quant_vfp8_results[0], o, quant_qk_results[1], quant_qk_results[3], quant_vfp8_results[1], quant_vfp8_results[2], tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse);
      } else {
        qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_fwd(quant_qk_results[0], quant_qk_results[2], quant_vfp8_results[0], o, quant_qk_results[1], quant_qk_results[3], quant_vfp8_results[1], tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse);
      }
      break;
    }
    case paddle::DataType::UNDEFINED: {
      qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm89_fwd(quant_qk_results[0], quant_qk_results[2], quant_vfp8_results[0], o, quant_qk_results[1], quant_qk_results[3], quant_vfp8_results[1], tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse);
      break;
    }
    default: {
      throw std::runtime_error("pv_accum_dtype must be fp32 or fp32+fp32");
      break;
    }
  }

  return {o};
}