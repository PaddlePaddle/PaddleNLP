#include "paddle/extension.h"

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

template <>
class PDTraits<paddle::DataType::BFLOAT16> {
public:
  typedef __nv_bfloat16 DataType;
  typedef paddle::bfloat16 data_t;
};

template <typename T>
__global__ void NeoXRotaryKernel(const T *input,
                                 const float *cos_emb,
                                 const float *sin_emb,
                                 const int *sequence_lengths,
                                 T *output,
                                 const int rotary_emb_dims,
                                 const int batch_size,
                                 const int num_head,
                                 const int seq_len,
                                 const int last_dim) {
  int bi = blockIdx.x;
  int hi = blockIdx.y;
  int si = blockIdx.z;
  if (sequence_lengths && si >= sequence_lengths[bi] * rotary_emb_dims) return;
  int half_lastdim = last_dim / 2;
  for (int ti = threadIdx.x; ti < half_lastdim; ti += blockDim.x) {
    int base_idx = bi * num_head * seq_len * last_dim +
                   hi * seq_len * last_dim + si * last_dim;
    int left_idx = base_idx + ti;
    const int right_idx = base_idx + ti + half_lastdim;
    int emb_idx_left = bi * seq_len * last_dim + si * last_dim + ti;
    int emb_idx_right =
        bi * seq_len * last_dim + si * last_dim + ti + half_lastdim;
    float input_left = static_cast<float>(input[left_idx]);
    float input_right = static_cast<float>(input[right_idx]);

    float cos_tmp_left = cos_emb[emb_idx_left];
    float sin_tmp_left = sin_emb[emb_idx_left];
    float cos_tmp_right = cos_emb[emb_idx_right];
    float sin_tmp_right = sin_emb[emb_idx_right];

    T res1 =
        static_cast<T>(input_left * cos_tmp_left - input_right * sin_tmp_left);
    T res2 = static_cast<T>(input_right * cos_tmp_right +
                            input_left * sin_tmp_right);
    output[left_idx] = res1;
    output[right_idx] = res2;
  }
}


template <paddle::DataType D>
std::vector<paddle::Tensor> LaunchRotaryQK(const paddle::Tensor& q, 
                                           const paddle::Tensor& k, 
                                           const paddle::Tensor& rotary_emb) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    const int64_t batch_size = q.shape()[0]; 
    const int64_t num_head = q.shape()[1]; 
    const int64_t seq_len = q.shape()[2]; 
    const int64_t dim_head = q.shape()[3]; 

    auto q_out = paddle::full({batch_size, num_head, seq_len, dim_head}, -1, q.dtype(), q.place());
    auto k_out = paddle::full({batch_size, num_head, seq_len, dim_head}, -1, k.dtype(), k.place());

    auto cu_stream = q.stream();

    assert(dim_head % 2 == 0); 
    
    dim3 grid(batch_size, num_head, seq_len);
    const int rotary_emb_dims = 1; // Since LLAMA Neox RotaryEmbedding no need rotary_emb_dims. 
    const int last_dim = dim_head;
    auto getBlockSize = [](int dim) {
        if (dim > 256) {
        return 512;
        } else if (dim > 128) {
        return 256;
        } else if (dim > 64) {
        return 128;
        } else if (dim > 32) {
        return 64;
        } else {
        return 32;
        }
    };
    int BlockSize = getBlockSize(last_dim / 2);
    const float *cos_emb = rotary_emb.data<float>();
    const float *sin_emb = rotary_emb.data<float>() + batch_size * seq_len * dim_head;

    NeoXRotaryKernel<<<grid, BlockSize, 0, cu_stream>>>(
        reinterpret_cast<const DataType_*>(q.data<data_t>()), 
        cos_emb,
        sin_emb,
        nullptr, /*sequence_lengths*/ // TODO(Zhengzekang): Support Variable length. 
        reinterpret_cast<DataType_*>(const_cast<data_t*>(q_out.data<data_t>())),
        rotary_emb_dims,
        batch_size,
        num_head,
        seq_len * rotary_emb_dims,
        last_dim);
    NeoXRotaryKernel<<<grid, BlockSize, 0, cu_stream>>>(
        reinterpret_cast<const DataType_*>(k.data<data_t>()),
        cos_emb,
        sin_emb,
        nullptr, /*sequence_lengths*/ // TODO(Zhengzekang): Support Variable length. 
        reinterpret_cast<DataType_*>(const_cast<data_t*>(k_out.data<data_t>())),
        rotary_emb_dims,
        batch_size,
        num_head,
        seq_len * rotary_emb_dims,
        last_dim);
    return {q_out, k_out};
}

std::vector<paddle::Tensor> RotaryQK(const paddle::Tensor& q, 
                                     const paddle::Tensor& k, 
                                     const paddle::Tensor& rotary_emb) {
    switch (q.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchRotaryQK<paddle::DataType::BFLOAT16>(
                q, k, rotary_emb
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchRotaryQK<paddle::DataType::FLOAT16>(
                q, k, rotary_emb
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchRotaryQK<paddle::DataType::FLOAT32>(
                q, k, rotary_emb
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");
            break;
        }
    }
}


std::vector<std::vector<int64_t>> RotaryQKInferShape(const std::vector<int64_t>& q_shape, 
                                                     const std::vector<int64_t>& k_shape, 
                                                     const std::vector<int64_t>& rotary_emb_shape) {
    const int64_t batch_size = q_shape[0]; 
    const int64_t num_head = q_shape[0]; 
    const int64_t seq_len = q_shape[0]; 
    const int64_t dim_head = q_shape[0]; 

    std::vector<int64_t> q_out_shape = {batch_size, num_head, seq_len, dim_head};                                                          
    return {q_out_shape, q_out_shape};
}

std::vector<paddle::DataType> RotaryQKInferDtype(const paddle::DataType& q_dtype, 
                                                 const paddle::DataType& k_dtype, 
                                                 const paddle::DataType& rotary_emb_dtype) {
    return {q_dtype, k_dtype};
}

PD_BUILD_OP(neox_rope)
    .Inputs({"q", "k", "rotary_emb"})
    .Outputs({"q_out", "k_out"})
    .SetKernelFn(PD_KERNEL(RotaryQK))
    .SetInferShapeFn(PD_INFER_SHAPE(RotaryQKInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RotaryQKInferDtype));