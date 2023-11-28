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

#include "helper.h"

template <typename T, int VecSize>
__global__ void fusedQKV_transpose_split_kernel(
    T *q_buf,
    T *k_buf,
    T *v_buf,
    const T *qkv,
    const int *padding_offset,
    const int *seq_lens,
    const int32_t elem_cnt,
    const int batch_size,
    const int max_len_this_time,
    const int seq_len,
    const int token_num,
    const int head_num,
    const int kv_head_num,
    const int size_per_head) {
  // const int32_t offset = batch_size * max_len_this_time * head_num * size_per_head;
  const int32_t hidden_size = head_num * size_per_head;
  const int32_t fused_hidden_size = hidden_size + kv_head_num * size_per_head + kv_head_num * size_per_head;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    Load<T, VecSize>(&qkv[linear_index], &src_vec);
    int32_t bias_idx = linear_index % fused_hidden_size;
    const int32_t token_idx = linear_index / fused_hidden_size;
    const int32_t ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int32_t target_batch_id = ori_token_idx / seq_len;
    if (seq_lens[target_batch_id] == 0) continue;
    const int32_t seq_id = ori_token_idx % seq_len;

    // equal to:
    // const int qkv_id  = (linear_index % fused_hidden_size) / hidden_size;
    const int32_t qkv_id = bias_idx < hidden_size ? 0 : (bias_idx -  hidden_size) / ( kv_head_num * size_per_head) + 1;
    const int32_t head_id = qkv_id == 0 ? bias_idx / size_per_head : (bias_idx -  hidden_size) / size_per_head % kv_head_num;
    const int32_t size_id = bias_idx % size_per_head;

    if (qkv_id == 0) {
      Store<T, VecSize>(
          src_vec,
          &q_buf[target_batch_id * head_num * max_len_this_time * size_per_head +
                 head_id * max_len_this_time * size_per_head + seq_id * size_per_head +
                 size_id]);
    } else if (qkv_id == 1) {
      Store<T, VecSize>(
          src_vec,
          &k_buf[target_batch_id * kv_head_num * max_len_this_time * size_per_head +
                 head_id * max_len_this_time * size_per_head + seq_id * size_per_head +
                 size_id]);
    } else {
      Store<T, VecSize>(
          src_vec,
          &v_buf[target_batch_id * kv_head_num * max_len_this_time * size_per_head +
                 head_id * max_len_this_time * size_per_head + seq_id * size_per_head +
                 size_id]);
    }
  }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> qkv_transpose_split(const paddle::Tensor& qkv, // [token_num, dim_embed]
                                                const paddle::Tensor& padding_offset, // [bsz, 1]
                                                const paddle::Tensor& seq_lens,
                                                const paddle::Tensor& input_ids,
                                                int num_head,
                                                int head_size) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    auto cu_stream = qkv.stream();
    std::vector<int64_t> qkv_shape = qkv.shape();
    const int token_num = qkv_shape[0];
    const int bsz = seq_lens.shape()[0];
    const int max_seq_len = input_ids.shape()[1]; //max_seq_len_tensor.copy_to(paddle::CPUPlace(), false).data<int>()[0];
    
    int64_t fused_hidden_size = qkv.shape()[1];
    int kv_num_head = (fused_hidden_size - num_head * head_size) / head_size / 2;

    auto q_out = paddle::full({bsz, num_head, max_seq_len, head_size}, 0, qkv.dtype(), qkv.place());
    auto k_out = paddle::full({bsz, kv_num_head, max_seq_len, head_size}, 0, qkv.dtype(), qkv.place());
    auto v_out = paddle::full({bsz, kv_num_head, max_seq_len, head_size}, 0, qkv.dtype(), qkv.place());
    constexpr int PackSize = VEC_16B / sizeof(DataType_);
    const int elem_cnt = qkv_shape[0] * qkv_shape[1];

    const int pack_num = elem_cnt / PackSize;
    const int blocksize = 128;
    const int grid_size = (pack_num + blocksize - 1) / blocksize;
    fusedQKV_transpose_split_kernel<DataType_, PackSize>
      <<<grid_size, blocksize, 0, qkv.stream()>>>(
        reinterpret_cast<DataType_*>(q_out.data<data_t>()),
        reinterpret_cast<DataType_*>(k_out.data<data_t>()),
        reinterpret_cast<DataType_*>(v_out.data<data_t>()),
        reinterpret_cast<DataType_*>(const_cast<data_t*>(qkv.data<data_t>())),
        padding_offset.data<int>(),
        seq_lens.data<int>(),
        elem_cnt,
        bsz,
        max_seq_len,
        max_seq_len,
        token_num,
        num_head,
        kv_num_head,
        head_size);
    return {q_out, k_out, v_out};
}

std::vector<paddle::Tensor> QKVTransposeSplit(const paddle::Tensor& qkv, 
                                              const paddle::Tensor& padding_offset, 
                                              const paddle::Tensor& seq_lens,
                                              const paddle::Tensor& input_ids,
                                              int num_head,
                                              int head_size) {
    switch (qkv.type()) {
        case paddle::DataType::BFLOAT16: {
            return qkv_transpose_split<paddle::DataType::BFLOAT16>(
                qkv,
                padding_offset,
                seq_lens,
                input_ids,
                num_head,
                head_size
            );
        }
        case paddle::DataType::FLOAT16: {
            return qkv_transpose_split<paddle::DataType::FLOAT16>(
                qkv,
                padding_offset,
                seq_lens,
                input_ids,
                num_head,
                head_size
            );
        }
        case paddle::DataType::FLOAT32: {
            return qkv_transpose_split<paddle::DataType::FLOAT32>(
                qkv,
                padding_offset,
                seq_lens,
                input_ids,
                num_head,
                head_size
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16, bfloat16 and float32 are supported. ");
            break;
        }
    }
}

std::vector<std::vector<int64_t>> QKVTransposeSplitInferShape(const std::vector<int64_t>& qkv_shape,
                                                              const std::vector<int64_t>& padding_offset_shape,
                                                              const std::vector<int64_t>& seq_lens_shape,
                                                              const std::vector<int64_t>& input_ids_shape,
                                                              int num_head,
                                                              int head_size) {
    int64_t bsz = seq_lens_shape[0];
    int64_t fused_hidden_size = qkv_shape[1];
    int kv_num_head = (fused_hidden_size - num_head * head_size) / head_size / 2;
    return {{bsz, num_head, -1, head_size}, {bsz, kv_num_head, -1, head_size}, {bsz, kv_num_head, -1, head_size}};
}

std::vector<paddle::DataType> QKVTransposeSplitInferDtype(const paddle::DataType& qkv_dtype,
                                                        const paddle::DataType& padding_offset_dtype,
                                                        const paddle::DataType& seq_lens_dtype,
                                                        const paddle::DataType& input_ids_dtype) {
    return {qkv_dtype, qkv_dtype, qkv_dtype};
}

PD_BUILD_OP(qkv_transpose_split)
    .Inputs({"qkv", "padding_offset", "seq_lens", "input_ids"})
    .Outputs({"q_out", "k_out", "v_out"})
    .Attrs({"num_head: int",
            "head_size: int"})
    .SetKernelFn(PD_KERNEL(QKVTransposeSplit))
    .SetInferShapeFn(PD_INFER_SHAPE(QKVTransposeSplitInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(QKVTransposeSplitInferDtype));