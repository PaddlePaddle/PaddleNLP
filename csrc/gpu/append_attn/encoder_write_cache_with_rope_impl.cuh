#include "helper.h"
#include "mem_util.cuh"
#include "mma_tensor_op.cuh"
#include "utils.cuh"

template <typename T, int VecSize = 1>
__global__ void VariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_biases,          // [3, num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = AlignedVector<int, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadScaleT = AlignedVector<float, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadBiasT bias_vec;
  LoadScaleT out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * last_dim;
  const int offset = 3 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int bias_idx = qkv_id * hidden_size + hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * 3 * hidden_size + bias_idx;
    Load<int, VecSize>(&qkv[base_idx], &src_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    }
    Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
    if (qkv_id < 2) {
      Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);
      // dequant + bias_add
      input_left = qkv_biases ? input_left * out_scale_vec[2 * i] +
                   static_cast<float>(bias_vec[2 * i]) : input_left * out_scale_vec[2 * i];
      input_right = qkv_biases ? input_right * out_scale_vec[2 * i + 1] +
                    static_cast<float>(bias_vec[2 * i + 1]) : input_right * out_scale_vec[2 * i + 1];
      if (qkv_id < 2) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        bias_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        bias_vec[2 * i] = static_cast<T>(input_left);
        bias_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    Store<T, VecSize>(bias_vec, &qkv_out[base_idx]);
  }
}

template <typename T, int VecSize = 1>
__global__ void NeoxVariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_biases,          // [3, num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = AlignedVector<int, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadScaleT = AlignedVector<float, VecSize>;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadBiasT left_bias_vec;
  LoadBiasT right_bias_vec;
  LoadScaleT left_out_scale_vec;
  LoadScaleT right_out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * half_lastdim;
  const int full_hidden_size = num_head * last_dim;
  const int offset = 3 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / half_lastdim;
    const int h_bias = qkv_bias % half_lastdim;

    const int ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    const int bias_idx_left =
        qkv_id * full_hidden_size + hi * last_dim + h_bias;
    const int bias_idx_right = bias_idx_left + half_lastdim;
    const int base_idx_left = token_idx * 3 * full_hidden_size + bias_idx_left;
    const int base_idx_right = base_idx_left + half_lastdim;
    Load<int, VecSize>(&qkv[base_idx_left], &left_vec);
    Load<int, VecSize>(&qkv[base_idx_right], &right_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx_left], &left_bias_vec);
      Load<T, VecSize>(&qkv_biases[bias_idx_right], &right_bias_vec);
    }
    Load<float, VecSize>(&qkv_out_scales[bias_idx_left],
                              &left_out_scale_vec);
    Load<float, VecSize>(&qkv_out_scales[bias_idx_right],
                              &right_out_scale_vec);
    if (qkv_id < 2) {
      Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      float input_left = static_cast<float>(left_vec[i]);
      float input_right = static_cast<float>(right_vec[i]);
      // dequant + bias_add
      input_left = qkv_biases ? input_left * left_out_scale_vec[i] +
                   static_cast<float>(left_bias_vec[i]) : input_left * left_out_scale_vec[i];
      input_right = qkv_biases ? input_right * right_out_scale_vec[i] +
                    static_cast<float>(right_bias_vec[i]) : input_right * right_out_scale_vec[i];
      if (qkv_id < 2) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_bias_vec[i] = static_cast<T>(input_left);
        right_bias_vec[i] = static_cast<T>(input_right);
      }
    }
    Store<T, VecSize>(left_bias_vec, &qkv_out[base_idx_left]);
    Store<T, VecSize>(right_bias_vec, &qkv_out[base_idx_right]);
  }
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,  // [3, q_num_head, dim_head]
    const T *qkv_biases,          // [3, q_num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = AlignedVector<int, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadScaleT = AlignedVector<float, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadBiasT bias_vec;
  LoadScaleT out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + 2 * kv_num_head) * last_dim;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / last_dim;
    const int h_bias = bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t bias_idx = hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * offset + bias_idx;
    Load<int, VecSize>(&qkv[base_idx], &src_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    }
    Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
    if (hi < q_num_head + kv_num_head) {
      Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);
      // dequant + bias_add
      input_left = qkv_biases ? input_left * out_scale_vec[2 * i] +
                   static_cast<float>(bias_vec[2 * i]) : input_left * out_scale_vec[2 * i];
      input_right = qkv_biases ? input_right * out_scale_vec[2 * i + 1] +
                    static_cast<float>(bias_vec[2 * i + 1]) : input_right * out_scale_vec[2 * i + 1];
      if (hi < q_num_head + kv_num_head) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        bias_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        bias_vec[2 * i] = static_cast<T>(input_left);
        bias_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    Store<T, VecSize>(bias_vec, &qkv_out[base_idx]);
  }
}

template <typename T, int VecSize = 1>
__global__ void GQANeoxVariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,  // [3, q_num_head, dim_head]
    const T *qkv_biases,          // [3, q_num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = AlignedVector<int, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadScaleT = AlignedVector<float, VecSize>;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadBiasT left_bias_vec;
  LoadBiasT right_bias_vec;
  LoadScaleT left_out_scale_vec;
  LoadScaleT right_out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + 2 * kv_num_head) * half_lastdim;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / half_lastdim;
    const int h_bias = bias % half_lastdim;

    const int ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    const int bias_idx_left = hi * last_dim + h_bias;
    const int bias_idx_right = bias_idx_left + half_lastdim;
    const int base_idx_left = token_idx * offset + bias_idx_left;
    const int base_idx_right = base_idx_left + half_lastdim;
    Load<int, VecSize>(&qkv[base_idx_left], &left_vec);
    Load<int, VecSize>(&qkv[base_idx_right], &right_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx_left], &left_bias_vec);
      Load<T, VecSize>(&qkv_biases[bias_idx_right], &right_bias_vec);
    }
    Load<float, VecSize>(&qkv_out_scales[bias_idx_left],
                              &left_out_scale_vec);
    Load<float, VecSize>(&qkv_out_scales[bias_idx_right],
                              &right_out_scale_vec);
    if (hi < (q_num_head + kv_num_head)) {
      Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      float input_left = static_cast<float>(left_vec[i]);
      float input_right = static_cast<float>(right_vec[i]);
      // dequant + bias_add
      input_left = qkv_biases ? input_left * left_out_scale_vec[i] +
                   static_cast<float>(left_bias_vec[i]) : input_left * left_out_scale_vec[i];
      input_right = qkv_biases ? input_right * right_out_scale_vec[i] +
                    static_cast<float>(right_bias_vec[i]) : input_right * right_out_scale_vec[i];
      if (hi < (q_num_head + kv_num_head)) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_bias_vec[i] = static_cast<T>(input_left);
        right_bias_vec[i] = static_cast<T>(input_right);
      }
    }
    Store<T, VecSize>(left_bias_vec, &qkv_out[base_idx_left]);
    Store<T, VecSize>(right_bias_vec, &qkv_out[base_idx_right]);
  }
}

template <typename T, int VecSize = 1>
__global__ void VariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,
    const T *qkv_biases,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * last_dim;
  const int offset = 3 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int bias_idx = qkv_id * hidden_size + hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * 3 * hidden_size + bias_idx;
    Load<T, VecSize>(&qkv[base_idx], &src_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    }
    Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left = qkv_biases ?
          static_cast<float>(src_vec[2 * i] + bias_vec[2 * i]) : static_cast<float>(src_vec[2 * i]);
      const float input_right = qkv_biases ?
          static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1]) : static_cast<float>(src_vec[2 * i + 1]);

      if (qkv_id < 2) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        src_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        src_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        src_vec[2 * i] = static_cast<T>(input_left);
        src_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

template <typename T, int VecSize = 1>
__global__ void NeoxVariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,
    const T *qkv_biases,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadT left_bias_vec;
  LoadT right_bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * half_lastdim;
  const int full_hidden_size = num_head * last_dim;
  const int offset = 3 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / half_lastdim;
    const int h_bias = qkv_bias % half_lastdim;

    const int ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    const int bias_idx_left =
        qkv_id * full_hidden_size + hi * last_dim + h_bias;
    const int bias_idx_right = bias_idx_left + half_lastdim;
    const int base_idx_left = token_idx * 3 * full_hidden_size + bias_idx_left;
    const int base_idx_right = base_idx_left + half_lastdim;
    Load<T, VecSize>(&qkv[base_idx_left], &left_vec);
    Load<T, VecSize>(&qkv[base_idx_right], &right_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx_left], &left_bias_vec);
      Load<T, VecSize>(&qkv_biases[bias_idx_right], &right_bias_vec);
    }
    Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      const float input_left = qkv_biases ?
          static_cast<float>(left_vec[i] + left_bias_vec[i]) : static_cast<float>(left_vec[i]);
      const float input_right = qkv_biases ?
          static_cast<float>(right_vec[i] + right_bias_vec[i]) : static_cast<float>(right_vec[i]);

      if (qkv_id < 2) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_vec[i] = static_cast<T>(input_left);
        right_vec[i] = static_cast<T>(input_right);
      }
    }
    Store<T, VecSize>(left_vec, &qkv_out[base_idx_left]);
    Store<T, VecSize>(right_vec, &qkv_out[base_idx_right]);
  }
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,
    const T *qkv_biases,
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + 2 * kv_num_head) * last_dim;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / last_dim;
    const int h_bias = bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t bias_idx = hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * offset + bias_idx;
    Load<T, VecSize>(&qkv[base_idx], &src_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    }
    Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left = qkv_biases ?
          static_cast<float>(src_vec[2 * i] + bias_vec[2 * i]) : static_cast<float>(src_vec[2 * i]);
      const float input_right = qkv_biases ?
          static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1]) : static_cast<float>(src_vec[2 * i + 1]);

      if (hi < q_num_head + kv_num_head) {  // qk rope
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      src_vec[2 * i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      src_vec[2 * i + 1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        src_vec[2 * i] = static_cast<T>(input_left);
        src_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

template <typename T, int VecSize = 1>
__global__ void GQANeoxVariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const float *qkv_out_scales,
    const T *qkv_biases,
    T *qkv_out,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (q_num_head + kv_num_head) * half_lastdim;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / half_lastdim;
    const int h_bias = bias % half_lastdim;

    const int ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

    const int emb_idx = ori_seq_id * last_dim + h_bias;
    const int base_idx_left =
        token_idx * (q_num_head + 2 * kv_num_head) * last_dim + hi * last_dim +
        h_bias;
    const int base_idx_right = base_idx_left + half_lastdim;

    Load<T, VecSize>(&qkv[base_idx_left], &left_vec);
    Load<T, VecSize>(&qkv[base_idx_right], &right_vec);
    Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      const float input_left = static_cast<float>(left_vec[i]);
      const float input_right = static_cast<float>(right_vec[i]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      left_vec[i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      right_vec[i] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    Store<T, VecSize>(left_vec, &qkv_out[base_idx_left]);
    Store<T, VecSize>(right_vec, &qkv_out[base_idx_right]);
  }
}


//================
// template <typename T, int VecSize = 1>
// __global__ void VariableLengthRotaryKernel(
//     const T *qkv,
//     const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
//     const float *sin_emb,
//     const int *padding_offsets,
//     const int *seq_lens,
//     const int *seq_lens_decoder,
//     const float *qkv_out_scales,
//     const T *qkv_biases,
//     T *qkv_out,
//     const int64_t elem_cnt,
//     const int num_head,
//     const int seq_len,
//     const int last_dim) {
//   using LoadT = AlignedVector<T, VecSize>;
//   constexpr int HalfVecSize = VecSize / 2;
//   using LoadEmbT = AlignedVector<float, HalfVecSize>;
//   LoadT src_vec;
//   LoadT bias_vec;
//   LoadEmbT cos_emb_vec;
//   LoadEmbT sin_emb_vec;
//   int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
//   const int half_lastdim = last_dim / 2;
//   const int hidden_size = num_head * last_dim;
//   const int offset = 3 * hidden_size;
//   for (int64_t linear_index = global_thread_idx * VecSize,
//                step = gridDim.x * blockDim.x * VecSize;
//        linear_index < elem_cnt;
//        linear_index += step) {
//     const int token_idx = linear_index / offset;
//     const int ori_token_idx = token_idx + padding_offsets[token_idx];
//     const int ori_bi = ori_token_idx / seq_len;
//     if (seq_lens[ori_bi] == 0) continue;
//     const int bias = linear_index % offset;
//     const int qkv_id = bias / hidden_size;
//     const int qkv_bias = bias % hidden_size;
//     const int hi = qkv_bias / last_dim;
//     const int h_bias = qkv_bias % last_dim;

//     int ori_seq_id;
//     ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

//     const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
//     const int64_t bias_idx = qkv_id * hidden_size + hi * last_dim + h_bias;
//     const int64_t base_idx = token_idx * 3 * hidden_size + bias_idx;
//     Load<T, VecSize>(&qkv[base_idx], &src_vec);
//     if (qkv_biases) {
//       Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
//     }
//     Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
//     Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
// #pragma unroll
//     for (int i = 0; i < HalfVecSize; i++) {
//       const float input_left = qkv_biases ?
//           static_cast<float>(src_vec[2 * i] + bias_vec[2 * i]) : static_cast<float>(src_vec[2 * i]);
//       const float input_right = qkv_biases ?
//           static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1]) : static_cast<float>(src_vec[2 * i + 1]);
//       // const float cos_tmp = cos_emb_vec[i];
//       // const float sin_tmp = sin_emb_vec[i];
//       // src_vec[2 * i] = static_cast<T>(input_left * cos_tmp - input_right *
//       // sin_tmp); src_vec[2 * i + 1] = static_cast<T>(input_right * cos_tmp +
//       // input_left * sin_tmp);

//       if (qkv_id < 2) {  // qk rope
//         const float cos_tmp = cos_emb_vec[i];
//         const float sin_tmp = sin_emb_vec[i];
//         src_vec[2 * i] =
//             static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
//         src_vec[2 * i + 1] =
//             static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
//       } else {
//         src_vec[2 * i] = static_cast<T>(input_left);
//         src_vec[2 * i + 1] = static_cast<T>(input_right);
//       }
//     }
//     Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
//   }
// }

// template <typename T, int VecSize = 1>
// __global__ void VariableLengthRotaryKernel(
//     const int *qkv,
//     const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
//     const float *sin_emb,
//     const int *padding_offsets,
//     const int *seq_lens,
//     const int *seq_lens_decoder,
//     const float *qkv_out_scales,  // [3, num_head, dim_head]
//     const T *qkv_biases,          // [3, num_head, dim_head]
//     T *qkv_out,
//     const int64_t elem_cnt,
//     const int num_head,
//     const int seq_len,
//     const int last_dim) {
//   using LoadT = AlignedVector<int, VecSize>;
//   using LoadBiasT = AlignedVector<T, VecSize>;
//   using LoadScaleT = AlignedVector<float, VecSize>;
//   constexpr int HalfVecSize = VecSize / 2;
//   using LoadEmbT = AlignedVector<float, HalfVecSize>;
//   LoadT src_vec;
//   LoadBiasT bias_vec;
//   LoadScaleT out_scale_vec;
//   LoadEmbT cos_emb_vec;
//   LoadEmbT sin_emb_vec;
//   int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
//   const int half_lastdim = last_dim / 2;
//   const int hidden_size = num_head * last_dim;
//   const int offset = 3 * hidden_size;
//   for (int64_t linear_index = global_thread_idx * VecSize,
//                step = gridDim.x * blockDim.x * VecSize;
//        linear_index < elem_cnt;
//        linear_index += step) {
//     const int token_idx = linear_index / offset;
//     const int ori_token_idx = token_idx + padding_offsets[token_idx];
//     const int ori_bi = ori_token_idx / seq_len;
//     if (seq_lens[ori_bi] == 0) continue;
//     const int bias = linear_index % offset;
//     const int qkv_id = bias / hidden_size;
//     const int qkv_bias = bias % hidden_size;
//     const int hi = qkv_bias / last_dim;
//     const int h_bias = qkv_bias % last_dim;

//     int ori_seq_id;  // = seq_lens_decoder[ori_bi];
//     // if (seq_lens_decoder[ori_bi] == 0) {
//     //   ori_seq_id = ori_token_idx % seq_len;
//     // } else {
//     //   ori_seq_id = seq_lens_decoder[ori_bi];
//     // }
//     ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

//     const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
//     const int64_t bias_idx = qkv_id * hidden_size + hi * last_dim + h_bias;
//     const int64_t base_idx = token_idx * 3 * hidden_size + bias_idx;
//     Load<int, VecSize>(&qkv[base_idx], &src_vec);
//     if (qkv_biases) {
//       Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
//     }
//     Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
//     if (qkv_id < 2) {
//       Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
//       Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
//     }
// #pragma unroll
//     for (int i = 0; i < HalfVecSize; i++) {
//       float input_left = static_cast<float>(src_vec[2 * i]);
//       float input_right = static_cast<float>(src_vec[2 * i + 1]);
//       // dequant + bias_add
//       input_left = qkv_biases ? input_left * out_scale_vec[2 * i] +
//                    static_cast<float>(bias_vec[2 * i]) : input_left * out_scale_vec[2 * i];
//       input_right = qkv_biases ? input_right * out_scale_vec[2 * i + 1] +
//                     static_cast<float>(bias_vec[2 * i + 1]) : input_right * out_scale_vec[2 * i + 1];
//       if (qkv_id < 2) {  // qk rope
//         const float cos_tmp = cos_emb_vec[i];
//         const float sin_tmp = sin_emb_vec[i];
//         bias_vec[2 * i] =
//             static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
//         bias_vec[2 * i + 1] =
//             static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
//       } else {
//         bias_vec[2 * i] = static_cast<T>(input_left);
//         bias_vec[2 * i + 1] = static_cast<T>(input_right);
//       }
//     }
//     Store<T, VecSize>(bias_vec, &qkv_out[base_idx]);
//   }
// }

// template <typename T, int VecSize = 1>
// __global__ void GQAVariableLengthRotaryKernel(
//     const T *qkv,
//     const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
//     const float *sin_emb,
//     const int *padding_offsets,
//     const int *seq_lens,
//     const int *seq_lens_decoder,
//     const float *qkv_out_scales,
//     const T *qkv_biases,
//     T *qkv_out,
//     const int64_t elem_cnt,
//     const int num_head,
//     const int seq_len,
//     const int last_dim,
//     const int gqa_group_size) {
//   using LoadT = AlignedVector<T, VecSize>;
//   constexpr int HalfVecSize = VecSize / 2;
//   using LoadEmbT = AlignedVector<float, HalfVecSize>;
//   LoadT src_vec;
//   LoadT bias_vec;
//   LoadEmbT cos_emb_vec;
//   LoadEmbT sin_emb_vec;
//   int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
//   const int half_lastdim = last_dim / 2;
//   // const int hidden_size = num_head * last_dim;
//   const int offset = (num_head + 2 * gqa_group_size) * last_dim;
//   for (int64_t linear_index = global_thread_idx * VecSize,
//                step = gridDim.x * blockDim.x * VecSize;
//        linear_index < elem_cnt;
//        linear_index += step) {
//     const int token_idx = linear_index / offset;
//     const int ori_token_idx = token_idx + padding_offsets[token_idx];
//     const int ori_bi = ori_token_idx / seq_len;
//     if (seq_lens[ori_bi] == 0) continue;
//     const int bias = linear_index % offset;
//     const int hi = bias / last_dim;
//     const int h_bias = bias % last_dim;

//     int ori_seq_id;
//     ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

//     const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
//     const int64_t bias_idx = hi * last_dim + h_bias;
//     const int64_t base_idx = token_idx * offset + bias_idx;
//     Load<T, VecSize>(&qkv[base_idx], &src_vec);
//     if (qkv_biases) {
//       Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
//     }
//     Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
//     Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
// #ifdef DEBUG_SENGMENTED_LORA_GEMM
//     __syncthreads();

//     printf("block.x:%d, thread.x:%d --212\n", blockIdx.x, threadIdx.x);
// #endif
// #pragma unroll
//     for (int i = 0; i < HalfVecSize; i++) {
//       const float input_left =
//           qkv_biases ? static_cast<float>(src_vec[2 * i] + bias_vec[2 * i])
//                      : static_cast<float>(src_vec[2 * i]);
//       const float input_right =
//           qkv_biases
//               ? static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1])
//               : static_cast<float>(src_vec[2 * i + 1]);

//       if (hi < num_head + gqa_group_size) {  // qk rope
//         const float cos_tmp = cos_emb_vec[i];
//         const float sin_tmp = sin_emb_vec[i];
//         src_vec[2 * i] =
//             static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
//         src_vec[2 * i + 1] =
//             static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
//       } else {
//         src_vec[2 * i] = static_cast<T>(input_left);
//         src_vec[2 * i + 1] = static_cast<T>(input_right);
//       }
//     }
//     Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
//   }
// #ifdef DEBUG_SENGMENTED_LORA_GEMM
//   __syncthreads();

//   printf("block.x:%d, thread.x:%d --212\n", blockIdx.x, threadIdx.x);
// #endif
// }

// template <typename T, int VecSize = 1>
// __global__ void GQAVariableLengthRotaryKernel(
//     const int *qkv,
//     const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
//     const float *sin_emb,
//     const int *padding_offsets,
//     const int *seq_lens,
//     const int *seq_lens_decoder,
//     const float *qkv_out_scales,  // [3, num_head, dim_head]
//     const T *qkv_biases,          // [3, num_head, dim_head]
//     T *qkv_out,
//     const int64_t elem_cnt,
//     const int num_head,
//     const int seq_len,
//     const int last_dim,
//     const int gqa_group_size) {
//   using LoadT = AlignedVector<int, VecSize>;
//   using LoadBiasT = AlignedVector<T, VecSize>;
//   using LoadScaleT = AlignedVector<float, VecSize>;
//   constexpr int HalfVecSize = VecSize / 2;
//   using LoadEmbT = AlignedVector<float, HalfVecSize>;
//   LoadT src_vec;
//   LoadBiasT bias_vec;
//   LoadScaleT out_scale_vec;
//   LoadEmbT cos_emb_vec;
//   LoadEmbT sin_emb_vec;
//   int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
//   const int half_lastdim = last_dim / 2;
//   const int offset = (num_head + 2 * gqa_group_size) * last_dim;
//   for (int64_t linear_index = global_thread_idx * VecSize,
//                step = gridDim.x * blockDim.x * VecSize;
//        linear_index < elem_cnt;
//        linear_index += step) {
//     const int token_idx = linear_index / offset;
//     const int ori_token_idx = token_idx + padding_offsets[token_idx];
//     const int ori_bi = ori_token_idx / seq_len;
//     if (seq_lens[ori_bi] == 0) continue;
//     const int bias = linear_index % offset;
//     const int hi = bias / last_dim;
//     const int h_bias = bias % last_dim;

//     int ori_seq_id;  
//     ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];
//     // const int ori_seq_id = ori_token_idx % seq_len;

//     const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
//     const int64_t bias_idx = hi * last_dim + h_bias;
//     const int64_t base_idx = token_idx * offset + bias_idx;
//     Load<int, VecSize>(&qkv[base_idx], &src_vec);
//     if (qkv_biases) {
//       Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
//     }
//     Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
//     if (hi < num_head + gqa_group_size) {
//       Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
//       Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
//     }
// #pragma unroll
//     for (int i = 0; i < HalfVecSize; i++) {
//       float input_left = static_cast<float>(src_vec[2 * i]);
//       float input_right = static_cast<float>(src_vec[2 * i + 1]);
//       // dequant + bias_add
//       input_left = qkv_biases ? input_left * out_scale_vec[2 * i] +
//                                     static_cast<float>(bias_vec[2 * i])
//                               : input_left * out_scale_vec[2 * i];
//       input_right = qkv_biases ? input_right * out_scale_vec[2 * i + 1] +
//                                      static_cast<float>(bias_vec[2 * i + 1])
//                                : input_right * out_scale_vec[2 * i + 1];
//       if (hi < num_head + gqa_group_size) {  // qk rope
//         const float cos_tmp = cos_emb_vec[i];
//         const float sin_tmp = sin_emb_vec[i];
//         bias_vec[2 * i] =
//             static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
//         bias_vec[2 * i + 1] =
//             static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
//       } else {
//         bias_vec[2 * i] = static_cast<T>(input_left);
//         bias_vec[2 * i + 1] = static_cast<T>(input_right);
//       }
//     }
//     Store<T, VecSize>(bias_vec, &qkv_out[base_idx]);
//   }
// }

template <typename T, int VecSize = 1>
__global__ void cache_kernel(
    const T *__restrict__ qkv,  // [num_tokens, num_heads + 2 * gqa_group_size,
                                // head_size]
    T *__restrict__ key_cache,  // [num_blocks, gqa_group_size, block_size,
                                // head_size]
    T *__restrict__ value_cache,  // [num_blocks, gqa_group_size, block_size,
                                  // head_size]
    const int *__restrict__ block_tables,      // [bsz, max_blocks_per_seq]
    const int *__restrict__ padding_offsets,   // [num_tokens]
    const int *__restrict__ seq_lens,          // [bsz]
    const int *__restrict__ seq_lens_decoder,  // [bsz]
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int block_size,
    const uint32_t elem_cnt,
    const int gqa_group_size) {
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;

  uint32_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t hidden_size = gqa_group_size * head_size;
  const uint32_t offset = 2 * hidden_size;
  for (uint32_t linear_index = global_thread_idx * VecSize,
                step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const uint32_t token_idx = linear_index / offset;
    const uint32_t bias = linear_index % offset;
    const uint32_t qkv_id = bias / hidden_size;  // skip q
    const uint32_t qkv_bias = bias % hidden_size;
    const uint32_t hi = qkv_bias / head_size;
    const uint32_t h_bias = qkv_bias % head_size;
    const uint32_t ori_token_idx = token_idx + padding_offsets[token_idx];
    const uint32_t ori_bi = ori_token_idx / max_seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const uint32_t ori_seq_id =
        ori_token_idx % max_seq_len + seq_lens_decoder[ori_bi];

    const int32_t *block_table_now = nullptr;

    block_table_now = block_tables + ori_bi * max_blocks_per_seq;

    const uint32_t block_idx = block_table_now[ori_seq_id / block_size];
    const uint32_t block_offset = ori_seq_id % block_size;

    const uint32_t tgt_idx =
        block_idx * gqa_group_size * block_size * head_size +
        hi * block_size * head_size + block_offset * head_size + h_bias;
    const uint32_t ori_idx =
        token_idx * (num_heads + 2 * gqa_group_size) * head_size +
        num_heads * head_size + qkv_id * hidden_size + hi * head_size + h_bias;
    Load<T, VecSize>(&qkv[ori_idx], &src_vec);
    if (qkv_id == 0) {
      Store<T, VecSize>(src_vec, &key_cache[tgt_idx]);
    } else {
      Store<T, VecSize>(src_vec, &value_cache[tgt_idx]);
    }
  }
}


template <typename T,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t NUM_WARPS>
__global__ void append_write_cache_kv_c8_qkv(
    uint8_t *__restrict__ cache_k,  // [max_block_id, kv_num_heads, block_size,
                                    // head_dim]
    uint8_t *__restrict__ cache_v,  // [max_block_id, kv_num_heads, head_dim,
                                    // block_size]
    const T *__restrict__ qkv_input,  // [token_num, num_heads + 2 *
                                      // kv_num_heads, head_dim]
    const T *__restrict__ cache_k_scales,
    const T *__restrict__ cache_v_scales,
    const int *__restrict__ batch_ids,
    const int *__restrict__ tile_ids,
    const int *__restrict__ seq_lens_this_time,
    const int *__restrict__ seq_lens_decoder,
    const int *__restrict__ padding_offsets,
    const int *__restrict__ cum_offsets,
    const int *__restrict__ block_tables,
    const int max_seq_len,
    const int max_block_num_per_seq,
    const int q_num_heads,
    const int kv_num_heads) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
  constexpr uint32_t pad_len = BLOCK_SIZE;
  const uint32_t btid = blockIdx.x, kv_head_idx = blockIdx.z;  // !!!
  const T cache_k_scale = cache_k_scales[kv_head_idx];
  const T cache_v_scale = cache_v_scales[kv_head_idx];
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;
  const uint32_t batch_id = batch_ids[btid];
  const uint32_t tile_id = tile_ids[btid];
  const uint32_t seq_len_this_time = seq_lens_this_time[batch_id];
  if (seq_len_this_time <= 0) {
    return;
  }
  const int *block_table_now = nullptr;

  block_table_now = block_tables + batch_id * max_block_num_per_seq;

  const uint32_t num_rows_per_block =
      NUM_WARPS * num_frags_z * 16;  // BLOCK_SIZE
  const uint32_t start_len = seq_lens_decoder[batch_id];
  const uint32_t bf_pad_len = start_len % pad_len;
  const uint32_t start_len_pad = start_len - bf_pad_len;
  const uint32_t end_len = start_len + seq_len_this_time;
  // const uint32_t end_len_pad_16 = div_up(end_len, num_rows_per_block) *
  // num_rows_per_block;
  const uint32_t tile_start = start_len_pad + tile_id * num_rows_per_block;
  uint32_t chunk_start = tile_start + wid * num_frags_z * 16 + tid / 8;

  // 前 start_len % pad_len 部分不搬，
  const uint32_t start_token_idx =
      batch_id * max_seq_len - cum_offsets[batch_id];
  const uint32_t kv_batch_stride = (q_num_heads + 2 * kv_num_heads) * HEAD_DIM;
  const uint32_t kv_h_stride = HEAD_DIM;
  __shared__ T k_smem_ori[num_rows_per_block * HEAD_DIM];
  __shared__ T v_smem_ori[num_rows_per_block * HEAD_DIM];

  smem_t k_smem(k_smem_ori);
  smem_t v_smem(v_smem_ori);

  uint32_t kv_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + tid / 8, tid % 8);  // 4 * 8 per warp
  /*
    1 ｜ 2
    ——————
    3 ｜ 4
  */
  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  /*
    1 ｜ 3
    ——————
    2 ｜ 4   transpose
  */
  constexpr uint32_t num_frags_v = num_frags_y / NUM_WARPS;
  uint32_t v_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      tid % 16, wid * num_frags_v * 2 + tid / 16);

  // load kv gmem to smem
  const uint32_t real_start_token_idx = start_token_idx - bf_pad_len +
                                        tile_id * num_rows_per_block +
                                        wid * num_frags_z * 16 + tid / 8;
  uint32_t k_read_idx = real_start_token_idx * kv_batch_stride +
                        (q_num_heads + kv_head_idx) * kv_h_stride +
                        tid % 8 * num_elems_per_128b<T>();
  uint32_t v_read_idx =
      real_start_token_idx * kv_batch_stride +
      (q_num_heads + kv_num_heads + kv_head_idx) * kv_h_stride +
      tid % 8 * num_elems_per_128b<T>();
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y / 4;
           ++fy) {  // (num_frags_y * 16) / (8 *  num_elems_per_128b<T>())
        if (chunk_start >= start_len && chunk_start < end_len) {
          k_smem.load_128b_async<SharedMemFillMode::kNoFill>(
              kv_smem_offset_w, qkv_input + k_read_idx, chunk_start < end_len);
          v_smem.load_128b_async<SharedMemFillMode::kNoFill>(
              kv_smem_offset_w, qkv_input + v_read_idx, chunk_start < end_len);
        }
        kv_smem_offset_w =
            k_smem.advance_offset_by_column<8>(kv_smem_offset_w, fy);
        k_read_idx += 8 * num_elems_per_128b<T>();
        v_read_idx += 8 * num_elems_per_128b<T>();
      }
      kv_smem_offset_w =
          k_smem.advance_offset_by_row<4, num_vecs_per_head>(kv_smem_offset_w) -
          2 * num_frags_y;
      chunk_start += 4;
      k_read_idx +=
          4 * kv_batch_stride - 2 * num_frags_y * num_elems_per_128b<T>();
      v_read_idx +=
          4 * kv_batch_stride - 2 * num_frags_y * num_elems_per_128b<T>();
    }
  }
  commit_group();
  wait_group<0>();
  __syncthreads();

  // mask, quant, store
  using LoadKVT = AlignedVector<uint8_t, 4>;
  LoadKVT cache_vec1;
  LoadKVT cache_vec2;

  uint32_t chunk_start_k = tile_start + wid * num_frags_z * 16 + tid / 4;
  uint32_t kv_frag[4];
  int block_id = __ldg(&block_table_now[tile_start / BLOCK_SIZE]);
  const uint32_t write_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM;
  const uint32_t write_h_stride = BLOCK_SIZE * HEAD_DIM;
  const uint32_t write_b_stride = HEAD_DIM;
  const uint32_t write_d_stride = BLOCK_SIZE;
  uint32_t k_write_idx = block_id * write_n_stride +
                         kv_head_idx * write_h_stride +
                         (wid * num_frags_z * 16 + tid / 4) * write_b_stride +
                         tid % 4 * 4;  // 4 * int8 = 8 * int4 = 32bit
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
    uint32_t k_write_idx_now_z = k_write_idx + fz * 16 * write_b_stride;
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      uint32_t k_write_idx_now = k_write_idx_now_z +
                                 fy % 2 * 8 * write_b_stride +
                                 fy / 2 * 32;  // + fy % 2 * 16;
      // load
      k_smem.ldmatrix_m8n8x4(k_smem_offset_r, kv_frag);
      // quant
      T *k_frag_T = reinterpret_cast<T *>(kv_frag);
      if (bf_pad_len != 0) {
        Load<uint8_t, 4>(cache_k + k_write_idx_now, &cache_vec1);
        Load<uint8_t, 4>(cache_k + k_write_idx_now + 16, &cache_vec2);
      }
#pragma unroll
      for (uint32_t v_id = 0; v_id < 8; ++v_id) {
        uint8_t uint_quant_value;
        if (chunk_start_k + (v_id / 4) * 8 >= start_len &&
            chunk_start_k + (v_id / 4) * 8 < end_len) {
          float quant_value =
              static_cast<float>(cache_k_scale * k_frag_T[v_id]);
          quant_value = roundWithTiesToEven(quant_value);
          quant_value = quant_value > 127.0f ? 127.0f : quant_value;
          quant_value = quant_value < -127.0f ? -127.0f : quant_value;
          uint_quant_value = static_cast<uint8_t>(quant_value + 127.0f);
        } else {
          uint_quant_value = 0;
        }
        if (bf_pad_len != 0) {
          if (v_id < 4) {
            cache_vec1[v_id] |= uint_quant_value;
          } else {
            cache_vec2[v_id % 4] |= uint_quant_value;
          }
        } else {
          if (v_id < 4) {
            cache_vec1[v_id] = uint_quant_value;
          } else {
            cache_vec2[v_id - 4] = uint_quant_value;
          }
        }
      }
      // store
      Store<uint8_t, 4>(cache_vec1, cache_k + k_write_idx_now);
      Store<uint8_t, 4>(cache_vec2, cache_k + k_write_idx_now + 16);
      k_smem_offset_r = k_smem.advance_offset_by_column<2>(k_smem_offset_r, fy);
    }
    k_smem_offset_r =
        k_smem.advance_offset_by_row<16, num_vecs_per_head>(k_smem_offset_r) -
        2 * num_frags_y;
    chunk_start_k += 16;
  }

  uint32_t chunk_start_v = tile_start + tid % 4 * 2;
  uint32_t v_write_idx = block_id * write_n_stride +
                         kv_head_idx * write_h_stride +
                         (wid * num_frags_v * 16 + tid / 4) * write_d_stride +
                         tid % 4 * 4;  // 4 * int8 = 8 * int4 = 32bit
  const uint32_t num_frags_z_v = num_frags_z * NUM_WARPS;
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_v; ++fy) {  // !!!
    uint32_t v_write_idx_now_v = v_write_idx + fy * 16 * write_d_stride;
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z_v; ++fz) {
      uint32_t v_write_idx_now = v_write_idx_now_v +
                                 fz % 2 * 8 * write_d_stride +
                                 fz / 2 * 32;  // + fz % 2 * 16;
      // load
      v_smem.ldmatrix_m8n8x4_trans(v_smem_offset_r, kv_frag);
      // quant
      T *v_frag_T = reinterpret_cast<T *>(kv_frag);
      if (bf_pad_len != 0) {
        Load<uint8_t, 4>(cache_v + v_write_idx_now, &cache_vec1);
        Load<uint8_t, 4>(cache_v + v_write_idx_now + 16, &cache_vec2);
      }
#pragma unroll
      for (uint32_t v_id = 0; v_id < 8; ++v_id) {
        uint8_t uint_quant_value;
        if (chunk_start_v + v_id % 2 + (v_id % 4) / 2 * 8 >= start_len &&
            chunk_start_v + v_id % 2 + (v_id % 4) / 2 * 8 < end_len) {
          float quant_value =
              static_cast<float>(cache_v_scale * v_frag_T[v_id]);
          quant_value = roundWithTiesToEven(quant_value);
          quant_value = quant_value > 127.0f ? 127.0f : quant_value;
          quant_value = quant_value < -127.0f ? -127.0f : quant_value;
          uint_quant_value = static_cast<uint8_t>(quant_value + 127.0f);
          // store now
        } else {
          uint_quant_value = 0;
        }
        if (bf_pad_len != 0) {
          if (v_id < 4) {
            cache_vec1[v_id] |= uint_quant_value;
          } else {
            cache_vec2[v_id % 4] |= uint_quant_value;
          }
        } else {
          if (v_id < 4) {
            cache_vec1[v_id] = uint_quant_value;
          } else {
            cache_vec2[v_id - 4] = uint_quant_value;
          }
        }
      }
      // store
      Store<uint8_t, 4>(cache_vec1, cache_v + v_write_idx_now);
      Store<uint8_t, 4>(cache_vec2, cache_v + v_write_idx_now + 16);
      chunk_start_v += 16;
      v_smem_offset_r =
          k_smem.advance_offset_by_row<16, num_vecs_per_head>(v_smem_offset_r);
    }
    v_smem_offset_r = k_smem.advance_offset_by_column<2>(
                          v_smem_offset_r, wid * num_frags_v + fy) -
                      16 * num_frags_z_v * num_vecs_per_head;
    chunk_start_v -= 16 * num_frags_z_v;
  }
}

// Write Cache KV in Append
template <typename T,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t NUM_WARPS>
__global__ void append_write_cache_kv_c4_qkv(
    uint8_t *__restrict__ cache_k,  // [max_block_id, kv_num_heads, block_size,
                                    // head_dim / 2]
    uint8_t *__restrict__ cache_v,  // [max_block_id, kv_num_heads, head_dim,
                                    // block_size / 2]
    const T *__restrict__ qkv_input,       // [token_num, num_heads + 2 *
                                           // kv_num_heads, head_dim]
    const T *__restrict__ cache_k_scales,  // [kv_num_heads, head_dim]
    const T *__restrict__ cache_v_scales,
    const T *__restrict__ cache_k_zero_points,
    const T *__restrict__ cache_v_zero_points,
    const int *__restrict__ batch_ids,
    const int *__restrict__ tile_ids,
    const int *__restrict__ seq_lens_this_time,
    const int *__restrict__ seq_lens_decoder,
    const int *__restrict__ padding_offsets,
    const int *__restrict__ cum_offsets,
    const int *__restrict__ block_tables,
    const int max_seq_len,
    const int max_block_num_per_seq,
    const int q_num_heads,
    const int kv_num_heads) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
  constexpr uint32_t pad_len = BLOCK_SIZE;
  const uint32_t btid = blockIdx.x, kv_head_idx = blockIdx.z;  // !!!
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;
  const uint32_t batch_id = batch_ids[btid];
  const uint32_t tile_id = tile_ids[btid];
  const uint32_t seq_len_this_time = seq_lens_this_time[batch_id];
  if (seq_len_this_time <= 0) {
    return;
  }
  const int *block_table_now = nullptr;

  block_table_now = block_tables + batch_id * max_block_num_per_seq;

  const uint32_t num_rows_per_block =
      NUM_WARPS * num_frags_z * 16;  // BLOCK_SIZE
  const uint32_t start_len = seq_lens_decoder[batch_id];
  const uint32_t bf_pad_len = start_len % pad_len;
  const uint32_t start_len_pad = start_len - bf_pad_len;
  const uint32_t end_len = start_len + seq_len_this_time;
  // const uint32_t end_len_pad_16 = div_up(end_len, num_rows_per_block) *
  // num_rows_per_block;
  const uint32_t tile_start = start_len_pad + tile_id * num_rows_per_block;
  uint32_t chunk_start = tile_start + wid * num_frags_z * 16 + tid / 8;

  // 前 start_len % pad_len 部分不搬，
  const uint32_t start_token_idx =
      batch_id * max_seq_len - cum_offsets[batch_id];
  const uint32_t kv_batch_stride = (q_num_heads + 2 * kv_num_heads) * HEAD_DIM;
  const uint32_t kv_h_stride = HEAD_DIM;
  __shared__ T k_smem_ori[num_rows_per_block * HEAD_DIM];
  __shared__ T v_smem_ori[num_rows_per_block * HEAD_DIM];
  __shared__ T k_scale_smem[HEAD_DIM];
  __shared__ T v_scale_smem[HEAD_DIM];
  __shared__ T k_zero_point_smem[HEAD_DIM];
  __shared__ T v_zero_point_smem[HEAD_DIM];
  const T *cache_k_scale_now = cache_k_scales + kv_head_idx * HEAD_DIM;
  const T *cache_k_zp_now = cache_k_zero_points + kv_head_idx * HEAD_DIM;
  const T *cache_v_scale_now = cache_v_scales + kv_head_idx * HEAD_DIM;
  const T *cache_v_zp_now = cache_v_zero_points + kv_head_idx * HEAD_DIM;
#pragma unroll
  for (uint32_t i = wid * 32 + tid; i < HEAD_DIM; i += 128) {
    k_scale_smem[i] = cache_k_scale_now[i];
    k_zero_point_smem[i] = cache_k_zp_now[i];
    v_scale_smem[i] = cache_v_scale_now[i];
    v_zero_point_smem[i] = cache_v_zp_now[i];
  }

  smem_t k_smem(k_smem_ori);
  smem_t v_smem(v_smem_ori);

  uint32_t kv_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + tid / 8, tid % 8);  // 4 * 8 per warp
  /*
    1 ｜ 2
    ——————
    3 ｜ 4
  */
  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  /*
    1 ｜ 3
    ——————
    2 ｜ 4   transpose
  */
  constexpr uint32_t num_frags_v = num_frags_y / NUM_WARPS;
  uint32_t v_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      tid % 16,
      wid * num_frags_v * 2 + tid / 16);  // wid * num_frags_v * 16 / 8

  // load kv gmem to smem
  const uint32_t real_start_token_idx = start_token_idx - bf_pad_len +
                                        tile_id * num_rows_per_block +
                                        wid * num_frags_z * 16 + tid / 8;
  uint32_t k_read_idx = real_start_token_idx * kv_batch_stride +
                        (q_num_heads + kv_head_idx) * kv_h_stride +
                        tid % 8 * num_elems_per_128b<T>();
  uint32_t v_read_idx =
      real_start_token_idx * kv_batch_stride +
      (q_num_heads + kv_num_heads + kv_head_idx) * kv_h_stride +
      tid % 8 * num_elems_per_128b<T>();
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y / 4;
           ++fy) {  // (num_frags_y * 16) / (8 *  num_elems_per_128b<T>())
        if (chunk_start >= start_len && chunk_start < end_len) {
          k_smem
              .load_128b_async<SharedMemFillMode::kNoFill>(  // can be kNoFill?
                  kv_smem_offset_w,
                  qkv_input + k_read_idx,
                  chunk_start < end_len);
          v_smem
              .load_128b_async<SharedMemFillMode::kNoFill>(  // can be kNoFill?
                  kv_smem_offset_w,
                  qkv_input + v_read_idx,
                  chunk_start < end_len);
        }
        kv_smem_offset_w =
            k_smem.advance_offset_by_column<8>(kv_smem_offset_w, fy);
        k_read_idx += 8 * num_elems_per_128b<T>();
        v_read_idx += 8 * num_elems_per_128b<T>();
      }
      kv_smem_offset_w =
          k_smem.advance_offset_by_row<4, num_vecs_per_head>(kv_smem_offset_w) -
          2 * num_frags_y;
      k_read_idx +=
          4 * kv_batch_stride - 2 * num_frags_y * num_elems_per_128b<T>();
      v_read_idx +=
          4 * kv_batch_stride - 2 * num_frags_y * num_elems_per_128b<T>();
      chunk_start += 4;
    }
  }
  commit_group();
  wait_group<0>();
  __syncthreads();
#ifdef DEBUG_WRITE_C4
  if (tid == 28 && wid == 1 && kv_head_idx == 0) {
    printf("k\n");
    for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
      for (uint32_t j = 0; j < HEAD_DIM; ++j) {
        printf("%f  ", (float)k_smem_ori[i * HEAD_DIM + j]);
      }
      printf("\n");
    }
    printf("k end\n");
    printf("v\n");
    for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
      for (uint32_t j = 0; j < HEAD_DIM; ++j) {
        printf("%f  ", (float)v_smem_ori[i * HEAD_DIM + j]);
      }
      printf("\n");
    }
    printf("v end\n");
  }
  __syncthreads();
#endif

  // mask, quant, store
  T cache_k_scale_frag[num_frags_y][4];
  T cache_k_zp_frag[num_frags_y][4];
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
    *(reinterpret_cast<uint32_t *>(&cache_k_scale_frag[fy][0])) =
        *(reinterpret_cast<uint32_t *>(&k_scale_smem[fy * 16]) + tid % 4);
    *(reinterpret_cast<uint32_t *>(&cache_k_scale_frag[fy][2])) =
        *(reinterpret_cast<uint32_t *>(&k_scale_smem[fy * 16]) + tid % 4 + 4);
    *(reinterpret_cast<uint32_t *>(&cache_k_zp_frag[fy][0])) =
        *(reinterpret_cast<uint32_t *>(&k_zero_point_smem[fy * 16]) + tid % 4);
    *(reinterpret_cast<uint32_t *>(&cache_k_zp_frag[fy][2])) =
        *(reinterpret_cast<uint32_t *>(&k_zero_point_smem[fy * 16]) + tid % 4 +
          4);
  }
  T cache_v_scale_frag[num_frags_y][2];
  T cache_v_zp_frag[num_frags_y][2];
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
    cache_v_scale_frag[fy][0] = v_scale_smem[fy * 16 + tid / 4];
    cache_v_scale_frag[fy][1] = v_scale_smem[fy * 16 + tid / 4 + 8];
    cache_v_zp_frag[fy][0] = v_zero_point_smem[fy * 16 + tid / 4];
    cache_v_zp_frag[fy][1] = v_zero_point_smem[fy * 16 + tid / 4 + 8];
  }

  using LoadKVT = AlignedVector<uint8_t, 4>;
  LoadKVT cache_vec;

  uint32_t chunk_start_k = tile_start + wid * num_frags_z * 16 + tid / 4;
  uint32_t kv_frag[4];
  int block_id = __ldg(&block_table_now[tile_start / BLOCK_SIZE]);
  const uint32_t write_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM / 2;
  const uint32_t write_h_stride = BLOCK_SIZE * HEAD_DIM / 2;
  const uint32_t write_b_stride = HEAD_DIM / 2;
  const uint32_t write_d_stride = BLOCK_SIZE / 2;
  uint32_t k_write_idx = block_id * write_n_stride +
                         kv_head_idx * write_h_stride +
                         (wid * num_frags_z * 16 + tid / 4) * write_b_stride +
                         tid % 4 * 4;  // 4 * int8 = 8 * int4 = 32bit
#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
    uint32_t k_write_idx_now_z = k_write_idx + fz * 16 * write_b_stride;
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      uint32_t k_write_idx_now = k_write_idx_now_z +
                                 (fy % 4) / 2 * 8 * write_b_stride +
                                 fy / 4 * 32 + fy % 2 * 16;
      // load
      k_smem.ldmatrix_m8n8x4(k_smem_offset_r, kv_frag);
      // quant
      T *k_frag_T = reinterpret_cast<T *>(kv_frag);
      // bf_pad_len为0表示新块写入，每个块第一次写入的时候不或，前后补0
      if (bf_pad_len != 0) {
        Load<uint8_t, 4>(cache_k + k_write_idx_now, &cache_vec);
      }

#ifdef DEBUG_WRITE_C4
      if (tid == 28 && wid == 1 && kv_head_idx == 0) {
#pragma unroll
        for (uint32_t t_id = 0; t_id < 8; ++t_id) {
          printf(
              "fy: %d, k_smem_offset_r: %d, k_write_idx_now: %d, load "
              "k_frag_T[%d] = %f, cache_vec[%d] = %f, "
              "cache_k_scale_frag[%d][%d] = %f, cache_k_zp_frag[%d][%d] = %f\n",
              (int)fy,
              (int)k_smem_offset_r,
              (int)k_write_idx_now,
              (int)t_id,
              (float)k_frag_T[t_id],
              (int)t_id % 4,
              (float)cache_vec[tid % 4],
              (int)fy,
              (int)tid % 4,
              (float)cache_k_scale_frag[fy][tid % 4],
              (int)fy,
              (int)tid % 4,
              (float)cache_k_zp_frag[fy][tid % 4]);
        }
      }
      __syncthreads();
#endif

#pragma unroll
      for (uint32_t v_id = 0; v_id < 4; ++v_id) {
        float quant_value1, quant_value2;
        uint8_t uint_quant_value1, uint_quant_value2;
        if (chunk_start_k >= start_len && chunk_start_k < end_len) {
          quant_value1 =
              static_cast<float>(cache_k_scale_frag[fy][v_id] * k_frag_T[v_id] +
                                 cache_k_zp_frag[fy][v_id]);
          quant_value1 = roundWithTiesToEven(quant_value1);
          quant_value1 = quant_value1 > 7.0f ? 7.0f : quant_value1;
          quant_value1 = quant_value1 < -8.0f ? -8.0f : quant_value1;
          uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
        } else {
          uint_quant_value1 = 0;
        }
        if (chunk_start_k + 8 >= start_len && chunk_start_k + 8 < end_len) {
          quant_value2 = static_cast<float>(cache_k_scale_frag[fy][v_id] *
                                                k_frag_T[v_id + 4] +
                                            cache_k_zp_frag[fy][v_id]);
          quant_value2 = roundWithTiesToEven(quant_value2);
          quant_value2 = quant_value2 > 7.0f ? 7.0f : quant_value2;
          quant_value2 = quant_value2 < -8.0f ? -8.0f : quant_value2;
          uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
        } else {
          uint_quant_value2 = 0;
        }
        if (bf_pad_len != 0) {
          cache_vec[v_id] |=
              (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);
        } else {
          cache_vec[v_id] =
              (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);
        }
      }
#ifdef DEBUG_WRITE_C4
      if (tid == 0 && wid == 0) {
#pragma unroll
        for (uint32_t t_id = 0; t_id < 4; ++t_id) {
          printf("fy: %d, k_write_idx_now: %d, cache_vec[%d] = %f\n",
                 (int)fy,
                 (int)k_write_idx_now,
                 (int)t_id,
                 (float)cache_vec[t_id]);
        }
      }
      __syncthreads();
#endif
      // store
      Store<uint8_t, 4>(cache_vec, cache_k + k_write_idx_now);
      k_smem_offset_r = k_smem.advance_offset_by_column<2>(k_smem_offset_r, fy);
    }
    k_smem_offset_r =
        k_smem.advance_offset_by_row<16, num_vecs_per_head>(k_smem_offset_r) -
        2 * num_frags_y;
    chunk_start_k += 16;
  }

  uint32_t chunk_start_v = tile_start + tid % 4 * 2;
  uint32_t v_write_idx = block_id * write_n_stride +
                         kv_head_idx * write_h_stride +
                         (wid * num_frags_v * 16 + tid / 4) * write_d_stride +
                         tid % 4 * 4;  // 4 * int8 = 8 * int4 = 32bit
  const uint32_t num_frags_z_v = num_frags_z * NUM_WARPS;
#ifdef DEBUG_WRITE_C4
  if (tid == 28 && wid == 1 && kv_head_idx == 0) {
    printf(
        "chunk_start_v: %d, v_write_idx: %d, num_frags_v: %d, num_frags_z_v: "
        "%d, v_smem_offset_r: %d\n",
        (int)chunk_start_v,
        (int)v_write_idx,
        int(num_frags_v),
        (int)num_frags_z_v,
        (int)v_smem_offset_r);
  }
  __syncthreads();
#endif
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_v; ++fy) {  // !!!
    uint32_t v_write_idx_now_v = v_write_idx + fy * 16 * write_d_stride;
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z_v; ++fz) {
      uint32_t v_write_idx_now = v_write_idx_now_v +
                                 (fz % 4) / 2 * 8 * write_d_stride +
                                 fz / 4 * 32 + fz % 2 * 16;
      // load
#ifdef DEBUG_WRITE_C4
      if (tid == 28 && wid == 1 && kv_head_idx == 0) {
        printf("fy: %d, fz: %d, v_smem_offset_r: %d\n",
               (int)fy,
               (int)fz,
               (int)v_smem_offset_r);
      }
      __syncthreads();
#endif
      v_smem.ldmatrix_m8n8x4_trans(v_smem_offset_r, kv_frag);
      // quant
      T *v_frag_T = reinterpret_cast<T *>(kv_frag);
#ifdef DEBUG_WRITE_C4
      if (tid == 28 && wid == 1 && kv_head_idx == 0) {
        printf("fy: %d, fz: %d, v_write_idx_now: %d\n",
               (int)fy,
               (int)fz,
               (int)v_write_idx_now);
        for (int tii = 0; tii < 8; ++tii) {
          printf("kv_frag[%d]: %f ", (int)tii, (float)kv_frag[tii]);
        }
        printf("\n");
      }
      __syncthreads();
#endif
      if (bf_pad_len != 0) {
        Load<uint8_t, 4>(cache_v + v_write_idx_now, &cache_vec);
      }
#pragma unroll
      for (uint32_t v_id = 0; v_id < 4; ++v_id) {
        float quant_value1, quant_value2;
        uint8_t uint_quant_value1, uint_quant_value2;
        if (chunk_start_v + v_id % 2 + v_id / 2 * 8 >= start_len &&
            chunk_start_v + v_id % 2 + v_id / 2 * 8 < end_len) {
          quant_value1 = static_cast<float>(
              cache_v_scale_frag[wid * num_frags_v + fy][0] * v_frag_T[v_id] +
              cache_v_zp_frag[wid * num_frags_v + fy][0]);
          quant_value1 = roundWithTiesToEven(quant_value1);
          quant_value1 = quant_value1 > 7.0f ? 7.0f : quant_value1;
          quant_value1 = quant_value1 < -8.0f ? -8.0f : quant_value1;
          uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
          quant_value2 =
              static_cast<float>(cache_v_scale_frag[wid * num_frags_v + fy][1] *
                                     v_frag_T[v_id + 4] +
                                 cache_v_zp_frag[wid * num_frags_v + fy][1]);
          quant_value2 = roundWithTiesToEven(quant_value2);
          quant_value2 = quant_value2 > 7.0f ? 7.0f : quant_value2;
          quant_value2 = quant_value2 < -8.0f ? -8.0f : quant_value2;
          uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
        } else {
          uint_quant_value1 = 0;
          uint_quant_value2 = 0;
        }
#ifdef DEBUG_WRITE_C4
        if (tid == 28 && wid == 1 && kv_head_idx == 0) {
          printf(
              "v_frag_T[%d]: %f, v_frag_T[%d]: %f, cache_v_scale_frag[%d][0]: "
              "%f, cache_v_scale_frag[%d][1]: %f, uint_quant_value1: %d, "
              "uint_quant_value2: %d\n",
              (int)v_id,
              (float)v_frag_T[v_id],
              (int)(v_id + 4),
              (float)v_frag_T[v_id + 4],
              (int)(wid * num_frags_v + fy),
              (float)cache_v_scale_frag[wid * num_frags_v + fy][0],
              (int)(wid * num_frags_v + fy),
              (float)cache_v_scale_frag[wid * num_frags_v + fy][1],
              (int)uint_quant_value1,
              (int)uint_quant_value1);
        }
        __syncthreads();
#endif
        if (bf_pad_len != 0) {
          cache_vec[v_id] |=
              (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);
        } else {
          cache_vec[v_id] =
              (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);
        }
      }
      // store
      Store<uint8_t, 4>(cache_vec, cache_v + v_write_idx_now);
      chunk_start_v += 16;
      v_smem_offset_r =
          v_smem.advance_offset_by_row<16, num_vecs_per_head>(v_smem_offset_r);
    }
    v_smem_offset_r = v_smem.advance_offset_by_column<2>(
                          v_smem_offset_r, wid * num_frags_v + fy) -
                      16 * num_frags_z_v * num_vecs_per_head;
    chunk_start_v -= 16 * num_frags_z_v;
  }
}