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

#define WARP_SIZE 32

template <typename T>
__forceinline__ __device__ T
CudaShuffleDownSync(unsigned mask, T val, int delta, int width = warpSize) {
  return __shfl_down_sync(mask, val, static_cast<unsigned>(delta), width);
}

template <>
__forceinline__ __device__ phi::dtype::float16 CudaShuffleDownSync(
    unsigned mask, phi::dtype::float16 val, int delta, int width) {
  return paddle::float16(__shfl_down_sync(
      mask, val.to_half(), static_cast<unsigned>(delta), width));
}

template <>
__forceinline__ __device__ phi::dtype::bfloat16 CudaShuffleDownSync(
    unsigned mask, phi::dtype::bfloat16 val, int delta, int width) {
  return paddle::bfloat16(__shfl_down_sync(
      mask, val.to_nv_bfloat16(), static_cast<unsigned>(delta), width));
}

struct BlockPrefixCallbackOp {
  // Running prefix
  float running_total;
  // Constructor
  __device__ BlockPrefixCallbackOp(float running_total)
      : running_total(running_total) {}
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide
  // scan.
  __device__ float operator()(float block_aggregate) {
    float old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

#define FINAL_MASK 0xFFFFFFFF

#define FIXED_BLOCK_DIM_BASE(dim, ...) \
  case (dim): {                        \
    constexpr auto kBlockDim = (dim);  \
    __VA_ARGS__;                       \
  } break

#define FIXED_BLOCK_DIM(...)                 \
  FIXED_BLOCK_DIM_BASE(1024, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(512, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(256, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(128, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(64, ##__VA_ARGS__);   \
  FIXED_BLOCK_DIM_BASE(32, ##__VA_ARGS__)


  #define FIXED_TOPK_BASE(topk, ...) \
    case (topk): {                   \
      constexpr auto kTopK = topk;   \
      __VA_ARGS__;                   \
    } break
  
  #define FIXED_TOPK(...) \
    FIXED_TOPK_BASE(2, ##__VA_ARGS__);  \
    FIXED_TOPK_BASE(3, ##__VA_ARGS__);  \
    FIXED_TOPK_BASE(4, ##__VA_ARGS__);  \
    FIXED_TOPK_BASE(5, ##__VA_ARGS__);  \
    FIXED_TOPK_BASE(8, ##__VA_ARGS__);  \
    FIXED_TOPK_BASE(10, ##__VA_ARGS__)


struct SegmentOffsetIter {
  explicit SegmentOffsetIter(int num_cols) : num_cols_(num_cols) {}

  __host__ __device__ __forceinline__ int operator()(int idx) const {
    return idx * num_cols_;
  }

  int num_cols_;
};

inline int div_up(int a, int n) { return (a + n - 1) / n; }

int GetBlockSize(int vocab_size) {
  if (vocab_size > 512) {
    return 1024;
  } else if (vocab_size > 256) {
    return 512;
  } else if (vocab_size > 128) {
    return 256;
  } else if (vocab_size > 64) {
    return 128;
  } else {
    return 64;
  }
}

template <typename T>
__global__ void FillIndex(T* indices, T num_rows, T num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (T j = row_id; j < num_rows; j += gridDim.x) {
    for (T i = col_id; i < num_cols; i += blockDim.x) {
      indices[j * num_cols + i] = i;
    }
  }
}

__global__ void SetCountIter(int* count_iter, int num) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + tid;
  for (int i = idx; i < num; i += gridDim.x * blockDim.x) {
    count_iter[i] = i;
  }
}


template <typename T, int BLOCK_SIZE>
__global__ void top_p_candidates_kernel(T* sorted_probs,
                                 int64_t* sorted_id,
                                 T* out_val,
                                 int64_t* out_id,
                                 int* actual_candidates_lens,
                                 const int vocab_size,
                                 const float topp,
                                 const int candidates_len) {
  __shared__ int stop_shared;
  __shared__ float rand_p;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;

  typedef cub::BlockScan<float, BLOCK_SIZE> BlockScan;
  typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ typename BlockReduce::TempStorage temp_storage_reduce;
  __shared__ uint32_t selected_shared[NUM_WARPS];

  if (lane_id == 0) {
    selected_shared[warp_id] = 0;
  }


  // Initialize running total
  BlockPrefixCallbackOp prefix_op(0);

  __syncthreads();

  int offset = bid * vocab_size;
  int end = ((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int i_activate = 0;
  float thread_offset = 0;
  for (int i = tid; i < end; i += BLOCK_SIZE) {
    float thread_count =
        (i < vocab_size) ? static_cast<float>(sorted_probs[offset + i]) : 0.f;

    BlockScan(temp_storage)
        .InclusiveSum(thread_count, thread_offset, prefix_op);

    if (i < candidates_len) {
        out_id[bid * candidates_len + i] = sorted_id[offset + i];
        out_val[bid * candidates_len + i] = sorted_probs[offset + i];
    }

    uint32_t activate_mask = __ballot_sync(FINAL_MASK, topp <= thread_offset);
    i_activate = i;
    if (activate_mask != 0 || i >= candidates_len) {
      if (lane_id == 0) {
        atomicAdd(&stop_shared, 1);
        selected_shared[warp_id] = activate_mask;
      }
    }
    __syncthreads();
    if (stop_shared > 0) {
      break;
    }
  }
  __syncthreads();
  bool skip = (selected_shared[warp_id] > 0) ? false : true;
  for (int i = 0; i < warp_id; i++) {
    if (selected_shared[i] != 0) {
      // If the previous has stopped, skip the current warp
      skip = true;
    }
  }
  if (!skip) {
    int active_lane_id =
        WARP_SIZE - __popc(selected_shared[warp_id]);  // first not 0
    if (lane_id == active_lane_id) {
        actual_candidates_lens[bid] = i_activate + 1;
    }
  }
  __syncthreads();
  if (tid == 0) {
    // printf("actual_candidates_lens[%d] %d\n", bid, actual_candidates_lens[bid]);
    if (actual_candidates_lens[bid] == 0) {
        actual_candidates_lens[bid] = candidates_len;
    }
  }
}


template <typename T>
struct Pair {
  __device__ __forceinline__ Pair() {}
  __device__ __forceinline__ Pair(T value, int id) : v(value), id(id) {}

  __device__ __forceinline__ void set(T value, int id) {
    this->v = value;
    this->id = id;
  }

  __device__ __forceinline__ void operator=(const Pair<T>& in) {
    v = in.v;
    id = in.id;
  }

  __device__ __forceinline__ bool operator<(const T value) const {
    return (static_cast<float>(v) < static_cast<float>(value));
  }

  __device__ __forceinline__ bool operator>(const T value) const {
    return (static_cast<float>(v) > static_cast<float>(value));
  }
  __device__ __forceinline__ bool operator<(const Pair<T>& in) const {
    return (static_cast<float>(v) < static_cast<float>(in.v)) ||
           ((static_cast<float>(v) == static_cast<float>(in.v)) &&
            (id > in.id));
  }

  __device__ __forceinline__ bool operator>(const Pair<T>& in) const {
    return (static_cast<float>(v) > static_cast<float>(in.v)) ||
           ((static_cast<float>(v) == static_cast<float>(in.v)) &&
            (id < in.id));
  }

  T v;
  int id;
};

template <typename T>
__device__ __forceinline__ void AddTo(Pair<T> topk[],
                                      const Pair<T>& p,
                                      int beam_size) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (topk[k] < p) {
      topk[k + 1] = topk[k];
    } else {
      topk[k + 1] = p;
      return;
    }
  }
  topk[0] = p;
}



template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(
    Pair<T> topk[], const T* src, int idx, int dim, int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      AddTo<T>(topk, tmp, beam_size);
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[],
                                        const T* src,
                                        int idx,
                                        int dim,
                                        const Pair<T>& max,
                                        int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      if (tmp < max) {
        AddTo<T>(topk, tmp, beam_size);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetTopK(Pair<T> topk[],
                                              int* beam,
                                              int beam_size,
                                              const T* src,
                                              bool* firstStep,
                                              bool* is_empty,
                                              Pair<T>* max,
                                              int dim,
                                              const int tid) {
  if (*beam > 0) {
    int length = (*beam) < beam_size ? *beam : beam_size;
    if (*firstStep) {
      *firstStep = false;
      GetTopK<T, BlockSize>(topk, src, tid, dim, length);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - (*beam)) {
          topk[k] = topk[k + *beam];
        } else {
          topk[k].set(std::numeric_limits<T>::min(), -1);
        }
      }
      if (!(*is_empty)) {
        GetTopK<T, BlockSize>(
            topk + MaxLength - *beam, src, tid, dim, *max, length);
      }
    }

    *max = topk[MaxLength - 1];
    if ((*max).id == -1) *is_empty = true;
    *beam = 0;
  }
}


template <typename T>
__forceinline__ __device__ Pair<T> WarpReduce(Pair<T> input) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    T tmp_val =
        CudaShuffleDownSync(FINAL_MASK, input.v, offset);
    int tmp_id =
        CudaShuffleDownSync(FINAL_MASK, input.id, offset);
    if (static_cast<float>(input.v) < static_cast<float>(tmp_val)) {
      input.v = tmp_val;
      input.id = tmp_id;
    }
  }
  return input;
}



template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void BlockReduce(Pair<T> shared_max[],
                                            Pair<T> topk[],
                                            Pair<T> beam_max[],
                                            int* beam,
                                            int* k,
                                            int* count,
                                            const int tid,
                                            const int wid,
                                            const int lane) {
  while (true) {
    __syncthreads();
    Pair<T> input_now = topk[0];
    input_now = WarpReduce(input_now);

    if (lane == 0) {
      shared_max[wid] = input_now;
    }
    __syncthreads();
    input_now = (tid < BlockSize / 32)
                    ? shared_max[lane]
                    : Pair<T>(std::numeric_limits<T>::min(), -1);
    if (wid == 0) {
      input_now = WarpReduce(input_now);
      if (lane == 0) shared_max[0] = input_now;
    }
    __syncthreads();
    if (tid == 0) {
      beam_max[*count] = shared_max[0];
      (*count)++;
    }
    int tid_max = shared_max[0].id % BlockSize;
    if (tid == tid_max) {
      (*beam)++;
    }
    if (--(*k) == 0) break;
    __syncthreads();

    if (tid == tid_max) {
      if (*beam < MaxLength) {
        topk[0] = topk[*beam];
      }
    }

    if (MaxLength < 5) {
      if (*beam >= MaxLength) break;
    } else {
      unsigned mask = 0u;
      mask = __ballot_sync(FINAL_MASK, true);
      if (tid_max / 32 == wid) {
        if (__shfl_down_sync(FINAL_MASK, *beam, tid_max % 32, 32) == MaxLength)
          break;
      }
    }
  }
}

template <typename T, int MaxLength, int TopPBeamTopK, int BlockSize>
__global__ void KeMatrixTopPBeamTopKFt(const T* src,
                                       const T* top_ps,
                                       const int* output_padding_offset,
                                       int64_t* out_id,  // [max_cadidate_len, 1]
                                       T* out_val,       // [max_cadidate_len, 1]
                                       int* actual_candidates_lens,
                                       int vocab_size,
                                       const int max_cadidate_len,
                                       const int max_seq_len) {
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lane = tid % 32;
  const int token_id = blockIdx.x;
  const int ori_token_id = token_id + output_padding_offset[token_id];
  const int bid = ori_token_id / max_seq_len;

  int top_num = TopPBeamTopK;
  float top_p_value = static_cast<float>(top_ps[bid]);

  __shared__ Pair<T> shared_max[BlockSize / 32];
  __shared__ Pair<T> beam_max[TopPBeamTopK];

  Pair<T> topk[MaxLength];
  int beam = MaxLength;
  Pair<T> max;
  bool is_empty = false;
  bool firststep = true;
  __shared__ int count;

  if (tid == 0) {
    count = 0;
  }

  for (int j = 0; j < MaxLength; j++) {
    topk[j].set(std::numeric_limits<T>::min(), -1);
  }

  while (top_num) {
    ThreadGetTopK<T, MaxLength, BlockSize>(topk,
                                           &beam,
                                           TopPBeamTopK,
                                           src + token_id * vocab_size,
                                           &firststep,
                                           &is_empty,
                                           &max,
                                           vocab_size,
                                           tid);
    BlockReduce<T, MaxLength, BlockSize>(
        shared_max, topk, beam_max, &beam, &top_num, &count, tid, wid, lane);
  }
  if (tid == 0) {
    float sum_prob = 0.0f;
    bool flag = false;
    for(int i = 0; i < TopPBeamTopK; i++) {
      out_id[token_id * max_cadidate_len + i] = static_cast<int64_t>(beam_max[i].id);
      out_val[token_id * max_cadidate_len + i] = beam_max[i].v;
      float val = static_cast<float>(beam_max[i].v);
      sum_prob += val;

      if(sum_prob >= top_p_value) {
        actual_candidates_lens[token_id] = i + 1;
        break;
      }
    }
  }
}


template <typename T, int TopKMaxLength>
void DispatchTopK(const T* src,
                  const T* top_ps,
                  const int* output_padding_offset,
                  int64_t* out_id,  // topk id
                  T* out_val,       // topk val
                  int* actual_candidates_lens_data,
                  const int vocab_size,
                  const int token_num,
                  const int cadidate_len,
                  const int max_seq_len,
                  cudaStream_t& stream) {
  int BlockSize = GetBlockSize(vocab_size);
  switch (cadidate_len) {
    FIXED_TOPK(
      switch (BlockSize) {
        FIXED_BLOCK_DIM(
            KeMatrixTopPBeamTopKFt<T, TopKMaxLength, kTopK, kBlockDim>
            <<<token_num, kBlockDim, 0, stream>>>(
                src,
                top_ps,
                output_padding_offset,
                out_id,
                out_val,
                actual_candidates_lens_data,
                vocab_size,
                cadidate_len,
                max_seq_len)
        );
        default:
          PD_THROW("the input data shape has error in the topp_beam_topk kernel.");
      }
    );
     default:
        PD_THROW("the input topk is not implemented.");
  }
}



template <paddle::DataType D>
std::vector<paddle::Tensor> LaunchTopPCandidates(const paddle::Tensor& probs, // [token_num, vocab_size]
                                                 const paddle::Tensor& top_p, // [token_num]
                                                 const paddle::Tensor& output_padding_offset,
                                                 const int candidates_len,
                                                 const int max_seq_len) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    std::vector<int64_t> input_shape = probs.shape();
    const int token_num = input_shape[0];
    const int vocab_size = input_shape[1];

    auto verify_scores = paddle::full({token_num, candidates_len}, 0, D, probs.place());
    auto verify_tokens = paddle::full({token_num, candidates_len}, 0, paddle::DataType::INT64, probs.place());
    auto actual_candidate_lens = paddle::full({token_num}, 0, paddle::DataType::INT32, probs.place());

    auto stream = probs.stream();

    constexpr int TopKMaxLength = 2;
    DispatchTopK<DataType_, TopKMaxLength>(
        reinterpret_cast<const DataType_*>(probs.data<data_t>()),
        reinterpret_cast<const DataType_*>(top_p.data<data_t>()),
        output_padding_offset.data<int>(),
        verify_tokens.data<int64_t>(),
        reinterpret_cast<DataType_*>(verify_scores.data<data_t>()),
        actual_candidate_lens.data<int>(),
        vocab_size,
        token_num,
        candidates_len,
        max_seq_len,
        stream
    );

    return {verify_scores, verify_tokens, actual_candidate_lens};
    
}


std::vector<paddle::Tensor> DispatchTopPCandidatesWithDtype(const paddle::Tensor& probs,
                                              const paddle::Tensor& top_p,
                                              const paddle::Tensor& output_padding_offset,
                                              int candidates_len,
                                              int max_seq_len) {
    switch (probs.type()) {
        case paddle::DataType::BFLOAT16: 
            return LaunchTopPCandidates<paddle::DataType::BFLOAT16>(probs, top_p, output_padding_offset, candidates_len, max_seq_len);
            break;
        case paddle::DataType::FLOAT16: 
            return LaunchTopPCandidates<paddle::DataType::FLOAT16>(probs, top_p, output_padding_offset, candidates_len, max_seq_len);
            break;
        case paddle::DataType::FLOAT32: 
            return LaunchTopPCandidates<paddle::DataType::FLOAT32>(probs, top_p, output_padding_offset, candidates_len, max_seq_len);
            break;
        default:
            PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");
            break;
    }
}


std::vector<paddle::Tensor> TopPCandidates(const paddle::Tensor& probs,
                                           const paddle::Tensor& top_p,
                                           const paddle::Tensor& output_padding_offset,
                                           int candidates_len,
                                           int max_seq_len) {
    return DispatchTopPCandidatesWithDtype(probs, top_p, output_padding_offset, candidates_len, max_seq_len);
}

std::vector<std::vector<int64_t>> TopPCandidatesInferShape(const std::vector<int64_t>& probs_shape, 
                                                           const std::vector<int64_t>& top_p_shape, 
                                                           const std::vector<int64_t>& output_padding_offset_shape, 
                                                           int max_candidates_len) {
    int token_num = probs_shape[0];
    return {{token_num, max_candidates_len}, {token_num, max_candidates_len}, {token_num}};
}

std::vector<paddle::DataType> TopPCandidatesInferDtype(const paddle::DataType& probs_dtype, 
                                                       const paddle::DataType& top_p_dtype,
                                                       const paddle::DataType& output_padding_offset_dtype) {
    return {probs_dtype, paddle::DataType::INT64, paddle::DataType::INT32};
}

PD_BUILD_OP(top_p_candidates)
    .Inputs({"probs","top_p", "output_padding_offset"})
    .Outputs({"verify_scores", "verify_tokens", "actual_candidate_lens"})
    .Attrs({"candidates_len: int", "max_seq_len: int"})
    .SetKernelFn(PD_KERNEL(TopPCandidates))
    .SetInferShapeFn(PD_INFER_SHAPE(TopPCandidatesInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TopPCandidatesInferDtype));