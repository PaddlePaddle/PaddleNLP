// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <curand_kernel.h>
#include <cuda_fp16.h>
#include "cub/cub.cuh"
#include "paddle/extension.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

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

template <paddle::DataType D>
class PDTraits;

template <>
class PDTraits<paddle::DataType::FLOAT32> {
public:
  typedef float DataType;
  typedef float data_t;
};

template <>
class PDTraits<paddle::DataType::FLOAT16> {
public:
  typedef half DataType;
  typedef paddle::float16 data_t;
};

struct SegmentOffsetIter {
    explicit SegmentOffsetIter(int num_cols) : num_cols_(num_cols) {}

    __host__ __device__ __forceinline__ int operator()(int idx) const {
        return idx * num_cols_;
    }

    int num_cols_;
};

template <typename T>
struct Pair {
  __device__ __forceinline__ Pair() {}
  __device__ __forceinline__ Pair(T value, int id) : v(value), id(id) {}

  __device__ __forceinline__ void set(T value, int id) {
    v = value;
    id = id;
  }

  __device__ __forceinline__ void operator=(const Pair<T>& in) {
    v = in.v;
    id = in.id;
  }

  __device__ __forceinline__ bool operator<(const T value) const {
    return ((float)v < (float)value);
  }

  __device__ __forceinline__ bool operator>(const T value) const {
    return ((float)v > (float)value);
  }
  __device__ __forceinline__ bool operator<(const Pair<T>& in) const {
    return ((float)v < (float)in.v) || (((float)v == (float)in.v) && (id > in.id));
  }

  __device__ __forceinline__ bool operator>(const Pair<T>& in) const {
    return ((float)v > (float)in.v) || (((float)v == (float)in.v) && (id < in.id));
  }

  T v;
  int id;
};

inline int div_up(int a, int n)
{
    return (a + n - 1) / n;
}

__global__ void setup_kernel(curandState_t *state, const uint64_t seed, const int bs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < bs; i += gridDim.x * blockDim.x) {
    curand_init(seed, 0, i, &state[i]);
  }
}

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
__device__ __forceinline__ void GetTopK(Pair<T> topk[],
                                        const T* src,
                                        int idx,
                                        int dim,
                                        int beam_size) {
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
        T tmp_val = __shfl_down_sync(FINAL_MASK, input.v, static_cast<unsigned>(offset), 32);
        int tmp_id = __shfl_down_sync(FINAL_MASK, input.id, static_cast<unsigned>(offset), 32);
        if ((float)input.v < (float)tmp_val) {
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
                                            int *count,
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
        if (__shfl_down_sync(FINAL_MASK, *beam, tid_max % 32, 32) ==
            MaxLength)
          break;
      }
    }
  }
}

template <typename T, int MaxLength, int TopPBeamTopK, int BlockSize>
__global__ void KeMatrixTopPBeamTopK(const T* src,
                                     T *top_ps,
                                     int64_t *out_id, // topk id
                                     T *out_val, // topk val
                                     int vocab_size,
                                     curandState_t *state,
                                     int *count_iter,
                                     int *count_iter_begin) {
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    const int bid = blockIdx.x;

    int top_num = TopPBeamTopK;
    float top_p_num = (float)top_ps[bid];

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
                                               src + bid * vocab_size,
                                               &firststep,
                                               &is_empty,
                                               &max,
                                               vocab_size,
                                               tid);
        BlockReduce<T, MaxLength, BlockSize>(shared_max,
                                             topk,
                                             beam_max,
                                             &beam,
                                             &top_num,
                                             &count,
                                             tid,
                                             wid,
                                             lane);
    }
    if (tid == 0) {
        count_iter_begin[bid] = count_iter[bid];
        float rand_top_p = curand_uniform(state + bid) * top_p_num;
        top_ps[bid] = (T)rand_top_p;
        float sum_prob = 0.0f;
#pragma unroll
        for(int i = 0; i < TopPBeamTopK; i++) {
            sum_prob += (float)(beam_max[i].v);
            if(sum_prob >= rand_top_p) {
                count_iter_begin[bid] += 1;
                out_id[bid] = (int64_t)beam_max[i].id;
                out_val[bid] = beam_max[i].v;
                break;
            }
        }
    }
}

__global__ void SetCountIter(int *count_iter, int num) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    for (int i = idx; i < num; i += gridDim.x * blockDim.x) {
        count_iter[i] = i;
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

struct BlockPrefixCallbackOp {
    // Running prefix
    float running_total;
    // Constructor
    __device__ BlockPrefixCallbackOp(float running_total): running_total(running_total) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ float operator()(float block_aggregate)
    {
        float old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};

template <typename T, int BLOCK_SIZE>
__global__ void topp_sampling(T *sorted_probs,
                              int64_t *sorted_id,
                              T *out_val,
                              int64_t *out_id,
                              const T *top_ps,
                              int p_num,
                              int vocab_size,
                              int *count_iter,
                              int *count_iter_begin) {
    __shared__ int stop_shared;
    __shared__ float rand_p;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const float p_t = (float)top_ps[bid];
    if (tid == 0) {
        stop_shared = 0;
        rand_p = p_t;
    }
    if (count_iter_begin[bid] == count_iter[bid + 1]) {
        // topk
        return;
    }

    typedef cub::BlockScan<float, BLOCK_SIZE>  BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ uint32_t selected_shared[NUM_WARPS];

    // Initialize running total
    BlockPrefixCallbackOp prefix_op(0);

    if (lane_id == 0) {
        selected_shared[warp_id] = 0;
    }
    __syncthreads();

    int offset = bid * vocab_size;
    int end = ((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    int i_activate = 0;
    float thread_offset = 0;
    for (int i = tid; i < end; i += BLOCK_SIZE) {
        float thread_count = (i < vocab_size) ? (float)sorted_probs[offset + i] : 0.f;
        BlockScan(temp_storage).InclusiveSum(thread_count, thread_offset, prefix_op);
    
        uint32_t activate_mask = __ballot_sync(FINAL_MASK, rand_p <= thread_offset);
        
        i_activate = i;
        if (activate_mask != 0) {
            if (lane_id == 0) {
                atomicAdd(&stop_shared, 1);
                selected_shared[warp_id] = activate_mask;
            }
        }
        __syncthreads();
        if(stop_shared > 0) {
            break;
        }
    }

    bool skip = (selected_shared[warp_id] > 0) ? false : true;
    for (int i=0; i < warp_id; i++) {
        if(selected_shared[i] != 0) {
            skip = true;
        }
    }
    if (!skip) {
        int active_lane_id = WARP_SIZE - __popc(selected_shared[warp_id]); // first not 0
        if (lane_id == active_lane_id) {
            // printf("active_lane_id: %d, i_activate: %d.\n", active_lane_id, i_activate);
            // for (int i=0; i < active_lane_id; i++) {
            //   printf("p %d, value: %f\n", i, (float)(sorted_probs[offset + i]));
            // }
            out_id[bid] = sorted_id[offset + i_activate];
            out_val[bid] = sorted_probs[offset + i_activate];
        }
    }
}

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
__global__ void print_kernel(T *input, int size) {
  printf("[");
  for (int i=0; i < size; i++) {
    if (i != size-1) {
      printf("%f, ", (float)input[i]);
    } else {
      printf("%f]\n", (float)input[i]);
    }
  }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> top_p_sampling_kernel(const paddle::Tensor& x, const paddle::Tensor& top_ps, int random_seed) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    std::vector<int64_t> shape = x.shape();
    auto cu_stream = x.stream();

    int bs = shape[0];
    int p_num = top_ps.numel();
    PD_CHECK(bs == p_num, "PD_CHECK returns ", false, ", expected bs == p_num.");
    int vocab_size = shape[1];
    auto topp_ids = paddle::full({bs, 1}, 1, paddle::DataType::INT64, x.place());
    auto topp_probs = paddle::full({bs, 1}, 1, x.dtype(), x.place());
    auto inds_input = paddle::full({bs, vocab_size}, 1, paddle::DataType::INT64, x.place());
    auto sorted_out = paddle::full({bs, vocab_size}, 1, x.dtype(), x.place());
    auto sorted_id = paddle::full({bs, vocab_size}, 1, paddle::DataType::INT64, x.place());
    

    int BlockSize = GetBlockSize(vocab_size);
    switch (BlockSize) {
        FIXED_BLOCK_DIM(FillIndex<int64_t><<<bs, kBlockDim, 0, cu_stream>>>(inds_input.data<int64_t>(), bs, vocab_size));
        default:
            PD_THROW("the input data shape has error in the FillIndex kernel.");
    }

    
    static int count = 0;
    static curandState_t* dev_curand_states;
    if (count == 0) {
#if CUDA_VERSION >= 11020
      cudaMallocAsync(&dev_curand_states, bs * sizeof(curandState_t), cu_stream);
#else
      cudaMalloc(&dev_curand_states, bs * sizeof(curandState_t));
#endif
    }
    srand((unsigned int)(time(NULL)));
    setup_kernel<<<1, 256, 0, cu_stream>>>(dev_curand_states, rand() % random_seed, bs);
    PD_CHECK(bs == p_num, "PD_CHECK returns ", false, ", expected bs == p_num.");

    auto count_iter = paddle::empty({bs + 1}, paddle::DataType::INT32, x.place());
    auto count_iter_begin = paddle::empty({bs}, paddle::DataType::INT32, x.place());
    SetCountIter<<<1, 256, 0, cu_stream>>>(count_iter.data<int>(), bs + 1);

    constexpr int TopKMaxLength = 1;
    constexpr int TopPBeamTopK = 1;
    switch (BlockSize) {
        FIXED_BLOCK_DIM(
            KeMatrixTopPBeamTopK<DataType_, TopKMaxLength, TopPBeamTopK, kBlockDim><<<bs, kBlockDim, 0, cu_stream>>>(
                reinterpret_cast<DataType_*>(const_cast<data_t*>(x.data<data_t>())),
                reinterpret_cast<DataType_*>(const_cast<data_t*>(top_ps.data<data_t>())),
                topp_ids.data<int64_t>(),
                reinterpret_cast<DataType_*>(topp_probs.data<data_t>()),
                vocab_size,
                dev_curand_states,
                count_iter.data<int>(),
                count_iter_begin.data<int>()));
        default:
            PD_THROW("the input data shape has error in the topp_beam_topk kernel.");
    }
//     if (count % random_seed == random_seed - 1) {
// #if CUDA_VERSION >= 11020
//       cudaFreeAsync(dev_curand_states, cu_stream);
// #else
//       cudaFree(dev_curand_states);
// #endif
//     }
    count++;

    size_t temp_storage_bytes = 0;

    cub::TransformInputIterator<int, SegmentOffsetIter, int*>
        segment_offsets_t_begin(count_iter_begin.data<int>(),
                                SegmentOffsetIter(vocab_size));

    cub::TransformInputIterator<int, SegmentOffsetIter, int*>
        segment_offsets_t_end(count_iter.data<int>(),
                              SegmentOffsetIter(vocab_size));
    
    DataType_ *x_ptr = reinterpret_cast<DataType_*>(const_cast<data_t*>(x.data<data_t>()));
    DataType_ *sorted_out_ptr = reinterpret_cast<DataType_*>(const_cast<data_t*>(sorted_out.data<data_t>()));
    int64_t *in_id_ptr = inds_input.data<int64_t>();
    int64_t *out_id_ptr = sorted_id.data<int64_t>();

    cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,
                                                       temp_storage_bytes,
                                                       x_ptr,
                                                       sorted_out_ptr,
                                                       in_id_ptr,
                                                       out_id_ptr,
                                                       vocab_size * bs,
                                                       bs,
                                                       segment_offsets_t_begin,
                                                       segment_offsets_t_end + 1,
                                                       0,
                                                       sizeof(data_t) * 8,
                                                       cu_stream);

    temp_storage_bytes = div_up(temp_storage_bytes, 256) * 256;
    int64_t temp_size = temp_storage_bytes;
    auto temp_storage = paddle::empty({temp_size}, paddle::DataType::UINT8, x.place());

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data<uint8_t>(),
        temp_storage_bytes,
        x_ptr,
        sorted_out_ptr,
        in_id_ptr,
        out_id_ptr,
        vocab_size * bs,
        bs,
        segment_offsets_t_begin,
        segment_offsets_t_end + 1,
        0,
        sizeof(data_t) * 8,
        cu_stream);

    switch (BlockSize) {
      FIXED_BLOCK_DIM(
          topp_sampling<DataType_, kBlockDim><<<bs, kBlockDim, 0, cu_stream>>>(
              sorted_out_ptr,
              out_id_ptr,
              reinterpret_cast<DataType_*>(topp_probs.data<data_t>()),
              topp_ids.data<int64_t>(),
              reinterpret_cast<DataType_*>(const_cast<data_t*>(top_ps.data<data_t>())),
              p_num,
              vocab_size,
              count_iter.data<int>(),
              count_iter_begin.data<int>()));
      default:
          PD_THROW("the input data shape has error in the topp_sampling kernel.");
    }
    return {topp_probs, topp_ids};
}


std::vector<paddle::Tensor> TopPSampling(const paddle::Tensor& x, const paddle::Tensor& top_ps, int random_seed) {
    switch (x.type()) {
        case paddle::DataType::FLOAT16: {
            return top_p_sampling_kernel<paddle::DataType::FLOAT16>(
                x,
                top_ps,
                random_seed
            );
        }
        case paddle::DataType::FLOAT32: {
            return top_p_sampling_kernel<paddle::DataType::FLOAT32>(
                x,
                top_ps,
                random_seed
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16 and float32 are supported. ");
            break;
        }
    }
}

std::vector<std::vector<int64_t>> TopPSamplingInferShape(const std::vector<int64_t>& x_shape,
                                                         const std::vector<int64_t>& top_ps_shape) {
    std::vector<int64_t> out_probs_shape = {x_shape[0], 1};                                                          
    std::vector<int64_t> out_ids_shape = {x_shape[0], 1};
    return {out_probs_shape, out_ids_shape};
}

std::vector<paddle::DataType> TopPSamplingInferDtype(const paddle::DataType& x_dtype,
                                                     const paddle::DataType& top_ps_dtype) {
    return {x_dtype, paddle::DataType::INT64};
}

PD_BUILD_OP(topp_sampling)
    .Inputs({"x", "top_ps"})
    .Outputs({"topp_probs", "topp_ids"})
    .Attrs({"random_seed: int"})
    .SetKernelFn(PD_KERNEL(TopPSampling))
    .SetInferShapeFn(PD_INFER_SHAPE(TopPSamplingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TopPSamplingInferDtype));