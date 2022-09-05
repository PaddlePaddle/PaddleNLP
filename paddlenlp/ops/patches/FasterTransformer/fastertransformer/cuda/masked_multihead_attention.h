/*
 * Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call) do { \
  cudaError_t status_ = call; \
  if( status_ != cudaSuccess ) { \
    fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
    exit(1); \
  } \
} while(0)

////////////////////////////////////////////////////////////////////////////////////////////////////

// The structure of parameters for the masked multihead attention kernel.
//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.

template< typename T >
struct Masked_multihead_attention_params {

  // The output buffer. Dimensions B x D.
  T *out;

  // The input Qs and the associated bias. Dimensions B x D and D, resp.
  const T *q, *q_bias;
  // The input Ks and the associated bias. Dimensions B x D and D, resp.
  const T *k, *k_bias;
  // The input Vs and the associated bias. Dimensions B x D and D, resp.
  const T *v, *v_bias;

  // The cache for the Ks. The size must be at least B x L x D.
  T *k_cache;
  // The cache for the Vs. The size must be at least B x L x D.
  T *v_cache;

  // The indirections to use for cache when beam sampling.
  const int* cache_indir = nullptr;

  // allows to exist attention eary
  bool *finished;

  // Stride to handle the case when KQV is a single buffer
  int stride;

  // The batch size.
  int batch_size;
  // The sequence length.
  int seq_length;
  // The number of heads (H).
  int num_heads;
  // The hidden dimension per head (Dh).
  int hidden_size_per_head;
  // The current timestep.
  int timestep;

  // The per-head latent space reserved for rotary embeddings.
  int rotary_embedding_dim = 0;

  // The 1.f / sqrt(Dh). Computed on the host.
  float inv_sqrt_dh;

  // params for masking.
  bool is_mask;
  const int *input_lengths = input_lengths;
  int max_input_len = max_input_len;

  const float* relative_attention_bias_float = nullptr;
  const half* relative_attention_bias_half = nullptr;
  int relative_attention_bias_stride;
  // The beam width
  int beam_width = 1;
  // required in case of cross attention
  int* memory_length_per_sample = nullptr;
  // required in case of masked attention with different length
  const int* length_per_sample = nullptr;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention    (const Masked_multihead_attention_params<float>    &params, const cudaStream_t &stream);
void masked_multihead_attention    (const Masked_multihead_attention_params<uint16_t> &params, const cudaStream_t &stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

