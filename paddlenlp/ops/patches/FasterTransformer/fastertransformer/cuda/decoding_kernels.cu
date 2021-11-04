/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

namespace fastertransformer {

template <typename T, bool ALIVE = false>
__global__ void init_kernel_v2(bool* finished,
                               int* sequence_length,
                               int* word_ids,
                               T* cum_log_probs,
                               const int sentence_id,
                               const int beam_width,
                               const int batch_size) {
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : 1e20f;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < batch_size * beam_width;
       index += blockDim.x * gridDim.x) {
    finished[index] = false;
    sequence_length[index] = 0;
    if (ALIVE) {
      if (index < batch_size * beam_width / 2) word_ids[index] = sentence_id;
      cum_log_probs[index] =
          (index % beam_width == beam_width / 2) ? (T)0.0f : -MAX_T_VAL;
    } else {
      word_ids[index] = sentence_id;
      cum_log_probs[index] = (index % beam_width == 0) ? (T)0.0f : -MAX_T_VAL;
    }
  }
}

template <typename T>
void init_kernelLauncher_v2(bool* finished,
                            int* sequence_length,
                            int* word_ids,
                            T* cum_log_probs,
                            const int sentence_id,
                            const int batch_size,
                            const int beam_width,
                            cudaStream_t stream) {
  dim3 grid((int)ceil(batch_size * beam_width * 1.0 / 256));
  dim3 block(256);

  init_kernel_v2<T, true><<<grid, block, 0, stream>>>(finished,
                                                      sequence_length,
                                                      word_ids,
                                                      cum_log_probs,
                                                      sentence_id,
                                                      beam_width,
                                                      batch_size);
}

template <typename T>
__global__ void embedding_position_lookups_bart_kernel(
    T* from_tensor,
    const T* embedding_table,
    const T* position_encoding,
    const int* word_ids,
    const int batch_size,
    const int hidden_units) {
  // 1. lookup from embedding table
  // 2. add the position encoding
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < batch_size * hidden_units;
       index += blockDim.x * gridDim.x) {
    const int row_index = index / hidden_units;
    const int col_index = index % hidden_units;
    from_tensor[index] =
        embedding_table[word_ids[row_index] * hidden_units + col_index] +
        position_encoding[col_index];
  }
}

template <typename T>
void embedding_position_lookups_bart_kernel_launcher(T* from_tensor,
                                                     const T* embedding_table,
                                                     const T* position_encoding,
                                                     const int* word_ids,
                                                     const int batch_size,
                                                     const int hidden_units,
                                                     cudaStream_t stream) {
  dim3 grid(min(batch_size, 65536));
  dim3 block(min(hidden_units, 1024));
  embedding_position_lookups_bart_kernel<T><<<grid, block, 0, stream>>>(
      from_tensor,
      embedding_table,
      position_encoding,
      word_ids,
      batch_size,
      hidden_units);
}

template <typename T>
__global__ void update_with_force_decoding_kernel(const int* trg_word,
                                                  const int* trg_length,
                                                  bool* finished,
                                                  int* word_ids,
                                                  int* sequence_length,
                                                  int* parent_ids_buf,
                                                  int* parent_ids,
                                                  int* output_ids,
                                                  T* scores,
                                                  bool keep_alive_beam,
                                                  const int batch_size,
                                                  const int beam_width,
                                                  const int max_trg_len,
                                                  const int step) {
  int bid = blockIdx.x;   // batch_size
  int tid = threadIdx.x;  // beam_width

  const T MAX_T_VAL = (sizeof(T) == 2) ? HALF_FLT_MAX : 1e20f;
  if (step <= trg_length[bid]) {
    finished[bid * beam_width + tid] = false;

    int word_id = trg_word[bid * max_trg_len + step - 1];

    if (keep_alive_beam) {
      if (tid >= beam_width / 2) {
        word_ids[bid * beam_width / 2 + tid - beam_width / 2] = word_id;
      }
    } else {
      word_ids[bid * beam_width + tid] = word_id;
    }

    output_ids[bid * beam_width + tid] = word_id;
    if (sequence_length) {
      sequence_length[bid * beam_width + tid]++;
    }

    if (parent_ids && scores) {
      if (keep_alive_beam) {
        parent_ids[bid * beam_width + tid] = bid * beam_width + beam_width / 2;
        if (tid >= beam_width / 2) {
          parent_ids_buf[bid * beam_width / 2 + tid - beam_width / 2] =
              bid * beam_width / 2;
        }

        if (tid == beam_width / 2) {
          scores[bid * beam_width + tid] = 0;
        } else {
          scores[bid * beam_width + tid] = -MAX_T_VAL;
        }
      } else {
        parent_ids[bid * beam_width + tid] = bid * beam_width;

        if (tid == 0) {
          scores[bid * beam_width + tid] = 0;
        } else {
          scores[bid * beam_width + tid] = -MAX_T_VAL;
        }
      }
    }
  }
}

template <typename T>
void update_with_force_decodingLauncher(const int* trg_word,
                                        const int* trg_length,
                                        bool* finished,
                                        int* word_ids,
                                        int* sequence_length,
                                        int* parent_ids_buf,
                                        int* parent_ids,
                                        int* output_ids,
                                        T* scores,
                                        bool keep_alive_beam,
                                        const int batch_size,
                                        const int beam_width,
                                        const int max_trg_len,
                                        const int step,
                                        cudaStream_t stream) {
  if (trg_word == nullptr) {
    return;
  }

  update_with_force_decoding_kernel<<<batch_size, beam_width, 0, stream>>>(
      trg_word,
      trg_length,
      finished,
      word_ids,
      sequence_length,
      parent_ids_buf,
      parent_ids,
      output_ids,
      scores,
      keep_alive_beam,
      batch_size,
      beam_width,
      max_trg_len,
      step);
}

template void init_kernelLauncher_v2(bool* finished,
                                     int* sequence_length,
                                     int* word_ids,
                                     float* cum_log_probs,
                                     const int sentence_id,
                                     const int batch_size,
                                     const int beam_width,
                                     cudaStream_t stream);

template void init_kernelLauncher_v2(bool* finished,
                                     int* sequence_length,
                                     int* word_ids,
                                     half* cum_log_probs,
                                     const int sentence_id,
                                     const int batch_size,
                                     const int beam_width,
                                     cudaStream_t stream);

template void embedding_position_lookups_bart_kernel_launcher(
    float* from_tensor,
    const float* embedding_table,
    const float* position_encoding,
    const int* word_ids,
    const int batch_size,
    const int hidden_units,
    cudaStream_t stream);

template void embedding_position_lookups_bart_kernel_launcher(
    half* from_tensor,
    const half* embedding_table,
    const half* position_encoding,
    const int* word_ids,
    const int batch_size,
    const int hidden_units,
    cudaStream_t stream);

template void update_with_force_decodingLauncher(const int* trg_word,
                                                 const int* trg_length,
                                                 bool* finished,
                                                 int* word_ids,
                                                 int* sequence_length,
                                                 int* parent_ids_buf,
                                                 int* parent_ids,
                                                 int* output_ids,
                                                 float* scores,
                                                 bool keep_alive_beam,
                                                 const int batch_size,
                                                 const int beam_width,
                                                 const int max_trg_len,
                                                 const int step,
                                                 cudaStream_t stream);

template void update_with_force_decodingLauncher(const int* trg_word,
                                                 const int* trg_length,
                                                 bool* finished,
                                                 int* word_ids,
                                                 int* sequence_length,
                                                 int* parent_ids_buf,
                                                 int* parent_ids,
                                                 int* output_ids,
                                                 half* scores,
                                                 bool keep_alive_beam,
                                                 const int batch_size,
                                                 const int beam_width,
                                                 const int max_trg_len,
                                                 const int step,
                                                 cudaStream_t stream);

}  // end of name space fastertransformer
