#include "paddle/extension.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "flash_attention/flash.h"
#include "cutlass/numeric_types.h"
#include "flash_attention/static_switch.h"

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

template <>
class PDTraits<paddle::DataType::BFLOAT16> {
 public:
  typedef __nv_bfloat16 DataType;
  typedef paddle::bfloat16 data_t;
};

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      paddle::Tensor& q,
                      paddle::Tensor& k,
                      paddle::Tensor& v,
                      paddle::Tensor* out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *p_d,
                      void *softmax_lse_d,
                      const float softmax_scale,
                      const bool is_causal) {
  // Reset the parameters
  memset(&params, 0, sizeof(params));
  params.is_bf16 = q.type() == paddle::DataType::BFLOAT16;
//   params.is_bf16 = q.dtype() == torch::kBFloat16;

  // Set the pointers and strides.
  params.q_ptr = q.data();
  params.k_ptr = k.data();
  params.v_ptr = v.data();
  params.o_ptr = out->data();
  // All stride are in elements, not bytes.
  auto q_dims = q.shape();
  auto k_dims = k.shape();
  auto v_dims = v.shape();
  auto out_dims = out->shape();
  params.q_head_stride = q_dims[q_dims.size() - 1];
  params.k_head_stride = k_dims[k_dims.size() - 1];
  params.v_head_stride = v_dims[v_dims.size() - 1];
  params.o_head_stride = out_dims[out_dims.size() - 1];
  params.q_row_stride = q_dims[q_dims.size() - 2] * params.q_head_stride;
  params.k_row_stride = k_dims[k_dims.size() - 2] * params.k_head_stride;
  params.v_row_stride = v_dims[v_dims.size() - 2] * params.v_head_stride;
  params.o_row_stride = out_dims[out_dims.size() - 2] * params.o_head_stride;

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = q_dims[1] * params.q_row_stride;
    params.k_batch_stride = k_dims[1] * params.k_row_stride;
    params.v_batch_stride = v_dims[1] * params.v_row_stride;
    params.o_batch_stride = out_dims[1] * params.o_row_stride;
  }

  params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

  // P = softmax(QK^T)
  params.p_ptr = p_d;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;
  
  params.p_dropout = 1.f;
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

  params.is_causal = is_causal;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
  FP16_SWITCH(!params.is_bf16, [&] {
    FWD_HEADDIM_SWITCH(
        params.d, [&] { run_mha_fwd_<elem_type, kHeadDim>(params, stream); });
  });
}

template <paddle::DataType T>
std::vector<paddle::Tensor> FlashAttnFwdKernel(paddle::Tensor& q,    // batch_size x seqlen_q x num_heads x head_size
                                               paddle::Tensor& k,    // batch_size x seqlen_k x num_heads_k x head_size
                                               paddle::Tensor& v,    // batch_size x seqlen_k x num_heads_k x head_size
                                               const float softmax_scale,
                                               const bool is_causal) {
    typedef PDTraits<T> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    
    const auto sizes = q.shape();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k.shape()[1];
    const int num_heads_k = k.shape()[2];

    // out = torch::empty_like(q_padded)
    auto out = paddle::empty(q.shape(), q.dtype(), paddle::GPUPlace());
    auto softmax_lse = paddle::empty({batch_size, num_heads, seqlen_q}, paddle::DataType::FLOAT32, paddle::GPUPlace());

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
    Flash_fwd_params params;
    set_params_fprop(params,
                    batch_size,
                    seqlen_q,
                    seqlen_k,
                    seqlen_q_rounded,
                    seqlen_k_rounded,
                    num_heads,
                    num_heads_k,
                    head_size,
                    head_size_rounded,
                    q,
                    k,
                    v,
                    &out,
                    /*cu_seqlens_q_d=*/nullptr,
                    /*cu_seqlens_k_d=*/nullptr,
                    nullptr,
                    softmax_lse.data(),
                    softmax_scale,
                    is_causal);
    auto stream = q.stream();
    run_mha_fwd(params, stream);

    return {out};
}

std::vector<paddle::Tensor> FlashAttnFwd(paddle::Tensor& q,
                  paddle::Tensor& k,
                  paddle::Tensor& v,
                  const float softmax_scale,
                  const bool is_causal) {
    {
        switch (q.type()) {
            case paddle::DataType::FLOAT16: {
                return FlashAttnFwdKernel<paddle::DataType::FLOAT16>(
                    q,
                    k,
                    v,
                    softmax_scale,
                    is_causal);
            }
            case paddle::DataType::BFLOAT16: {
                return FlashAttnFwdKernel<paddle::DataType::BFLOAT16>(
                    q,
                    k,
                    v,
                    softmax_scale,
                    is_causal);
            }
            default: {
                PD_THROW(
                    "NOT supported data type. "
                    "Only float16 and bfloat16 are supported. ");
                break;
            }
        }
    }

}
std::vector<std::vector<int64_t>> FlashAttnFwdInferShape(const std::vector<int64_t>& q_shape, const std::vector<int64_t>& k_shape, const std::vector<int64_t>& v_shape) {
    return {q_shape};
}

std::vector<paddle::DataType> FlashAttnFwdInferDtype(const paddle::DataType& q_dtype, const paddle::DataType& k_dtype, const paddle::DataType& v_dtype) {
    return {q_dtype};
}

PD_BUILD_OP(flash_attn_fwd)
    .Inputs({"q", "k", "v"})
    .Outputs({"out"})
    .Attrs({"softmax_scale: float", "is_causal: bool"})
    .SetKernelFn(PD_KERNEL(FlashAttnFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(FlashAttnFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FlashAttnFwdInferDtype));


template <paddle::DataType T>
std::vector<paddle::Tensor> FlashAttnVarlenFwdKernel(
    paddle::Tensor& q,    // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    paddle::Tensor& k,    // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    paddle::Tensor& v,    // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    paddle::Tensor& cu_seqlens_q,
    paddle::Tensor& cu_seqlens_k,
    paddle::Tensor& max_seqlen_q,
    paddle::Tensor& max_seqlen_k,
    const float softmax_scale,
    const bool zero_tensors,
    const bool is_causal) {
    typedef PDTraits<T> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    int max_seqlen_q_p[1], max_seqlen_k_p[1];
    cudaMemcpy((void*)max_seqlen_q_p, max_seqlen_q.data(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)max_seqlen_k_p, max_seqlen_k.data(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    const int max_seqlength_q = max_seqlen_q_p[0];
    const int max_seqlength_k = max_seqlen_k_p[0];
    const auto sizes = q.shape();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int total_k = k.shape()[0];
    const int num_heads_k = k.shape()[1];

    // out = torch::empty_like(q_padded)

    paddle::Tensor out, softmax_lse;
    if (zero_tensors) {
        out = paddle::zeros(q.shape(), q.dtype(), paddle::GPUPlace());
        softmax_lse = paddle::full({batch_size, num_heads, max_seqlength_q}, -std::numeric_limits<float>::infinity(), paddle::DataType::FLOAT32, paddle::GPUPlace());
    } else {
        out = paddle::empty(q.shape(), q.dtype(), paddle::GPUPlace());
        softmax_lse = paddle::empty({batch_size, num_heads, max_seqlength_q}, paddle::DataType::FLOAT32, paddle::GPUPlace());
    }
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlength_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlength_k, 128);

    Flash_fwd_params params;
    set_params_fprop(params,
                    batch_size,
                    max_seqlength_q,
                    max_seqlength_k,
                    seqlen_q_rounded,
                    seqlen_k_rounded,
                    num_heads,
                    num_heads_k,
                    head_size,
                    head_size_rounded,
                    q,
                    k,
                    v,
                    &out,
                    cu_seqlens_q.data(),
                    cu_seqlens_k.data(),
                    nullptr,
                    softmax_lse.data(),
                    softmax_scale,
                    is_causal);
    auto stream = k.stream();
    run_mha_fwd(params, stream);
    return {out};
}

std::vector<paddle::Tensor> FlashAttnVarlenFwd(
    paddle::Tensor& q,
    paddle::Tensor& k,
    paddle::Tensor& v,
    paddle::Tensor& cu_seqlens_q,
    paddle::Tensor& cu_seqlens_k,
    paddle::Tensor& max_seqlen_q,
    paddle::Tensor& max_seqlen_k,
    const float softmax_scale,
    const bool zero_tensors,
    const bool is_causal) {
    {
    switch (q.type()) {
        case paddle::DataType::FLOAT16: {
            return FlashAttnVarlenFwdKernel<paddle::DataType::FLOAT16>(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                zero_tensors,
                is_causal);
        }
        case paddle::DataType::BFLOAT16: {
            return FlashAttnVarlenFwdKernel<paddle::DataType::BFLOAT16>(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                zero_tensors,
                is_causal);
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16 and bfloat16 are supported. ");
            break;
        }
    }
}

}
std::vector<std::vector<int64_t>> FlashAttnVarlenFwdInferShape(const std::vector<int64_t>& q_shape, const std::vector<int64_t>& k_shape, const std::vector<int64_t>& v_shape, const std::vector<int64_t>& cu_seqlens_q_shape, const std::vector<int64_t>& cu_seqlens_k_shape, const std::vector<int64_t>& max_seqlen_q_shape, const std::vector<int64_t>& max_seqlen_k_shape) {
    return {q_shape};
}

std::vector<paddle::DataType> FlashAttnVarlenFwdInferDtype(const paddle::DataType& q_dtype, const paddle::DataType& k_dtype, const paddle::DataType& v_dtype, const paddle::DataType& cu_seqlens_q_dtype, const paddle::DataType& cu_seqlens_k_dtype, const paddle::DataType& max_seqlen_q_dtype, const paddle::DataType& max_seqlen_k_dtype) {
    return {q_dtype};
}

PD_BUILD_OP(flash_attn_varlen_fwd)
    .Inputs({"q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"})
    .Outputs({"out"})
    .Attrs({"softmax_scale: float", "zero_tensors: bool", "is_causal: bool"})
    .SetKernelFn(PD_KERNEL(FlashAttnVarlenFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(FlashAttnVarlenFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FlashAttnVarlenFwdInferDtype));
