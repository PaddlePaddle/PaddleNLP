/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <paddle/extension.h>
#include <paddle/phi/common/data_type.h>
#include <vector>

#include "causal_conv1d.h"

#define CHECK_SHAPE(x, ...) PD_CHECK(x.dims() == common::make_ddim({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                     \
    if (ITYPE == paddle::DataType::FLOAT16) {                                        \
        using input_t = phi::dtype::float16;                                         \
        __VA_ARGS__();                                                               \
    } else if (ITYPE == paddle::DataType::BFLOAT16) {                                \
        using input_t = phi::dtype::bfloat16;                                        \
        __VA_ARGS__();                                                               \
    } else if (ITYPE == paddle::DataType::FLOAT32)  {                                \
        using input_t = float;                                                       \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        PADDLE_THROW(#NAME, " not implemented for input type '", ITYPE, "'");        \
    }

#define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
    if (WTYPE == paddle::DataType::FLOAT16) {                                        \
        using weight_t = phi::dtype::float16;                                        \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == paddle::DataType::BFLOAT16) {                                \
        using weight_t = phi::dtype::bfloat16;                                       \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == paddle::DataType::FLOAT32)  {                                \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        PADDLE_THROW(#NAME, " not implemented for weight type '", WTYPE, "'");       \
    }

template<typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);
template <typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void causal_conv1d_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream);
template<typename input_t, typename weight_t>
void causal_conv1d_channellast_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase &params, cudaStream_t stream);

void set_conv_params_fwd(ConvParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t seqlen,
                         const size_t width,
                         // device pointers
                         const paddle::Tensor x,
                         const paddle::Tensor weight,
                         const paddle::Tensor out,
                         void* bias_ptr,
                         bool silu_activation) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;

    params.silu_activation = silu_activation;

    // Set the pointers and strides.
    params.x_ptr = const_cast<void*>(x.data());
    params.weight_ptr = const_cast<void*>(weight.data());
    params.bias_ptr = const_cast<void*>(bias_ptr);
    params.out_ptr = const_cast<void*>(out.data());
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.strides()[0];
    params.x_c_stride = x.strides()[1];
    params.x_l_stride = x.strides()[x.strides().size() - 1];
    params.weight_c_stride = weight.strides()[0];
    params.weight_width_stride = weight.strides()[1];
    params.out_batch_stride = out.strides()[0];
    params.out_c_stride = out.strides()[1];
    params.out_l_stride = out.strides()[out.strides().size() - 1];
}


void set_conv_params_bwd(ConvParamsBwd &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t seqlen,
                         const size_t width,
                         // device pointers
                         const paddle::Tensor x,
                         const paddle::Tensor weight,
                         void* bias_ptr,
                         const paddle::Tensor dout,
                         const paddle::Tensor dx,
                         const paddle::Tensor dweight,
                         void* dbias_ptr,
                         bool silu_activation) {
    // Pass in "dout" instead of "out", we're not gonna use "out" at all.
    set_conv_params_fwd(params, batch, dim, seqlen, width,
                        x, weight, dout, bias_ptr, silu_activation);

    // Set the pointers and strides.
    params.dout_ptr = const_cast<void*>(dout.data());
    params.dx_ptr = const_cast<void*>(dx.data());
    params.dweight_ptr = const_cast<void*>(dweight.data());
    params.dbias_ptr = const_cast<void*>(dbias_ptr);
    // All stride are in elements, not bytes.
    params.dout_batch_stride = dout.strides()[0];
    params.dout_c_stride = dout.strides()[1];
    params.dout_l_stride = dout.strides()[2];
    params.dweight_c_stride = dweight.strides()[0];
    params.dweight_width_stride = dweight.strides()[1];
    params.dx_batch_stride = dx.strides()[0];
    params.dx_c_stride = dx.strides()[1];
    params.dx_l_stride = dx.strides()[2];
}

paddle::Tensor
causal_conv1d_fwd(const paddle::Tensor &x, const paddle::Tensor &weight,
                  const std::optional<paddle::Tensor> &bias_,
                  const std::optional<paddle::Tensor> &seq_idx_,
                  const std::optional<paddle::Tensor> &initial_states_,
                  std::optional<paddle::Tensor> &final_states_out_,
                  bool silu_activation) {
    auto input_type = x.dtype();
    auto weight_type = weight.dtype();
    PD_CHECK(input_type == paddle::DataType::FLOAT32 || input_type == paddle::DataType::FLOAT16 || input_type == paddle::DataType::BFLOAT16);
    PD_CHECK(weight_type == paddle::DataType::FLOAT32 || weight_type == paddle::DataType::FLOAT16 || weight_type == paddle::DataType::BFLOAT16);

    PD_CHECK(x.is_gpu());
    PD_CHECK(weight.is_gpu());

    const auto sizes = x.dims();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.dims()[weight.dims().size() - 1];

    CHECK_SHAPE(x, batch_size, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);

    PD_CHECK(x.strides()[2] == 1 || x.strides()[1] == 1);
    const bool is_channel_last = x.strides()[1] == 1 && x.strides()[2] > 1;

    if (is_channel_last) {
        PD_CHECK(dim % 8 == 0, "causal_conv1d only supports channel dimension divisible by 8 for now");
        PD_CHECK(x.strides()[2] % 8 == 0 and x.strides()[0] % 8 == 0, "causal_conv1d with channel last layout requires strides (x.strides()[0] and x.strides()[2]) to be multiples of 8");
    }
    PD_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        PD_CHECK(bias.dtype() == weight_type);
        PD_CHECK(bias.is_gpu());
        PD_CHECK(bias.strides()[bias.strides().size() - 1] == 1);
        CHECK_SHAPE(bias, dim);
    }

    if (seq_idx_.has_value()) {
        PD_CHECK(is_channel_last, "seq_idx is only supported for channel last layout");
        auto seq_idx = seq_idx_.value();
        PD_CHECK(seq_idx.dtype() == paddle::DataType::INT32 || seq_idx.dtype() == paddle::DataType::INT64);
        PD_CHECK(seq_idx.is_gpu());
        // PD_CHECK(seq_idx.is_contiguous());
        CHECK_SHAPE(seq_idx, batch_size, seqlen);
    }

    paddle::Tensor out = paddle::empty_like(x);
    // NOTE: new added
    if (is_channel_last) {
        out = paddle::experimental::as_strided(out, {batch_size, dim, seqlen}, {dim * seqlen, 1, dim});  
    }

    ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                        bias_.has_value() ? const_cast<void*>(bias_.value().data()) : nullptr,
                        silu_activation);

    if (seq_idx_.has_value()) {
        params.seq_idx_ptr = const_cast<void*>(seq_idx_.value().data());
    } else {
        params.seq_idx_ptr = nullptr;
    }

    if (initial_states_.has_value()) {
        PD_CHECK(is_channel_last, "initial_states is only supported for channel last layout");
        auto initial_states = initial_states_.value();
        PD_CHECK(initial_states.dtype() == input_type);
        PD_CHECK(initial_states.is_gpu());
        CHECK_SHAPE(initial_states, batch_size, dim, width - 1);
        PD_CHECK(initial_states.strides()[1] == 1);
        params.initial_states_ptr = const_cast<void*>(initial_states.data());
        params.initial_states_batch_stride = initial_states.strides()[0];
        params.initial_states_c_stride = initial_states.strides()[1];
        params.initial_states_l_stride = initial_states.strides()[2];
    } else {
        params.initial_states_ptr = nullptr;
    }

    if (final_states_out_.has_value()) {
        PD_CHECK(is_channel_last, "final_states is only supported for channel last layout");
        auto final_states = final_states_out_.value();
        PD_CHECK(final_states.dtype() == input_type);
        PD_CHECK(final_states.is_gpu());
        CHECK_SHAPE(final_states, batch_size, dim, width - 1);
        PD_CHECK(final_states.strides()[1] == 1);
        params.final_states_ptr = const_cast<void*>(final_states.data());
        params.final_states_batch_stride = final_states.strides()[0];
        params.final_states_c_stride = final_states.strides()[1];
        params.final_states_l_stride = final_states.strides()[2];
    } else {
        params.final_states_ptr = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    auto stream = x.stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.dtype(), "causal_conv1d_fwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(weight.dtype(), "causal_conv1d_fwd", [&] {
            if (!is_channel_last) {
                causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream);
            } else {
                causal_conv1d_channellast_fwd_cuda<input_t, weight_t>(params, stream);
            }
        });
    });
    return out;
}

std::vector<paddle::Tensor>
causal_conv1d_bwd(const paddle::Tensor &x, const paddle::Tensor &weight,
                  const std::optional<paddle::Tensor> &bias_,
                  paddle::Tensor &dout,
                  const std::optional<paddle::Tensor> &seq_idx_,
                  const std::optional<paddle::Tensor> &initial_states_,
                  const std::optional<paddle::Tensor> &dfinal_states_,
                  std::optional<paddle::Tensor> &dx_,
                  bool return_dinitial_states,
                  bool silu_activation) {
    auto input_type = x.dtype();
    auto weight_type = weight.dtype();
    PD_CHECK(input_type == paddle::DataType::FLOAT32 || input_type == paddle::DataType::FLOAT16 || input_type == paddle::DataType::BFLOAT16);
    PD_CHECK(weight_type == paddle::DataType::FLOAT32 || weight_type == paddle::DataType::FLOAT16 || weight_type == paddle::DataType::BFLOAT16);

    PD_CHECK(x.is_gpu());
    PD_CHECK(weight.is_gpu());
    PD_CHECK(dout.is_gpu());

    const auto sizes = x.dims();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.dims()[weight.dims().size() - 1];

    PD_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    CHECK_SHAPE(x, batch_size, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);
    CHECK_SHAPE(dout, batch_size, dim, seqlen);

    PD_CHECK(x.strides()[2] == 1 || x.strides()[1] == 1);
    const bool is_channel_last = x.strides()[1] == 1 && x.strides()[2] > 1;
    // NOTE: 由于缺少contiguous算子，所以在外面做。
    // if (!is_channel_last && dout.stride(2) != 1) { dout = dout.contiguous(); }
    // if (is_channel_last && dout.stride(1) != 1) { dout = dout.transpose({0, 2, 1}).contiguous().transpose({0, 2, 1}); }

    if (is_channel_last) {
        PD_CHECK(dim % 8 == 0, "causal_conv1d only supports channel dimension divisible by 8 for now");
        PD_CHECK(x.strides()[2] % 8 == 0 and x.strides()[0] % 8 == 0, "causal_conv1d with channel last layout requires strides (x.strides()[0] and x.strides()[2]) to be multiples of 8");
        PD_CHECK(dout.strides()[2] % 8 == 0 and dout.strides()[0] % 8 == 0, "causal_conv1d with channel last layout requires strides (dout.strides()[0] and dout.strides()[2]) to be multiples of 8");
    }

    if (bias_.has_value()) {
        auto bias = bias_.value();
        PD_CHECK(bias.dtype() == weight_type);
        PD_CHECK(bias.is_gpu());
        PD_CHECK(bias.strides()[bias.strides().size() - 1] == 1);
        CHECK_SHAPE(bias, dim);
    }

    if (seq_idx_.has_value()) {
        PD_CHECK(is_channel_last, "seq_idx only supported for channel last layout");
        auto seq_idx = seq_idx_.value();
        PD_CHECK(seq_idx.dtype() == paddle::DataType::INT32 || seq_idx.dtype() == paddle::DataType::INT64);
        PD_CHECK(seq_idx.is_gpu());
        // PD_CHECK(seq_idx.is_contiguous());
        CHECK_SHAPE(seq_idx, batch_size, seqlen);
    }

    paddle::Tensor dx;
    if (dx_.has_value()) {
        dx = dx_.value();
        PD_CHECK(dx.dtype() == input_type);
        PD_CHECK(dx.is_gpu());
        CHECK_SHAPE(dx, batch_size, dim, seqlen);
        if (!is_channel_last) { PD_CHECK(dx.strides()[2] == 1); }
        if (is_channel_last) { PD_CHECK(dx.strides()[1] == 1); }
    } else {
        dx = paddle::empty_like(x);
        if (is_channel_last) {
            dx = paddle::experimental::as_strided(dx, {batch_size, dim, seqlen}, {dim * seqlen, 1, dim});  
        }
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    // make sure dweight and dbias dtype paddle::DataType::FLOAT32
    paddle::Tensor dweight = paddle::experimental::zeros_like(weight, paddle::DataType::FLOAT32);
    paddle::Tensor dbias;
    if (bias_.has_value()) { dbias = paddle::experimental::zeros_like(bias_.value(), paddle::DataType::FLOAT32); }

    ConvParamsBwd params;
    set_conv_params_bwd(params, batch_size, dim, seqlen, width,
                        x, weight, bias_.has_value() ? const_cast<void*>(bias_.value().data()) : nullptr,
                        dout, dx, dweight, bias_.has_value() ? const_cast<void*>(dbias.data()) : nullptr,
                        silu_activation);

    if (seq_idx_.has_value()) {
        params.seq_idx_ptr = const_cast<void*>(seq_idx_.value().data());
    } else {
        params.seq_idx_ptr = nullptr;
    }

    if (initial_states_.has_value()) {
        PD_CHECK(is_channel_last, "initial_states is only supported for channel last layout");
        auto initial_states = initial_states_.value();
        PD_CHECK(initial_states.dtype() == input_type);
        PD_CHECK(initial_states.is_gpu());
        CHECK_SHAPE(initial_states, batch_size, dim, width - 1);
        PD_CHECK(initial_states.strides()[1] == 1);
        params.initial_states_ptr = const_cast<void*>(initial_states.data());
        params.initial_states_batch_stride = initial_states.strides()[0];
        params.initial_states_c_stride = initial_states.strides()[1];
        params.initial_states_l_stride = initial_states.strides()[2];
    } else {
        params.initial_states_ptr = nullptr;
    }

    if (dfinal_states_.has_value()) {
        PD_CHECK(is_channel_last, "dfinal_states is only supported for channel last layout");
        auto dfinal_states = dfinal_states_.value();
        PD_CHECK(dfinal_states.dtype() == input_type);
        PD_CHECK(dfinal_states.is_gpu());
        CHECK_SHAPE(dfinal_states, batch_size, dim, width - 1);
        params.dfinal_states_ptr = const_cast<void*>(dfinal_states.data());
        params.dfinal_states_batch_stride = dfinal_states.strides()[0];
        params.dfinal_states_c_stride = dfinal_states.strides()[1];
        params.dfinal_states_l_stride = dfinal_states.strides()[2];
    } else {
        params.dfinal_states_ptr = nullptr;
    }

    paddle::Tensor dinitial_states;
    if (return_dinitial_states) {
        dinitial_states = paddle::experimental::transpose(paddle::empty({batch_size, width - 1, dim}, x.dtype(), x.place()), {0, 2, 1});
        PD_CHECK(dinitial_states.strides()[1] == 1);
        params.dinitial_states_ptr = const_cast<void*>(dinitial_states.data());
        params.dinitial_states_batch_stride = dinitial_states.strides()[0];
        params.dinitial_states_c_stride = dinitial_states.strides()[1];
        params.dinitial_states_l_stride = dinitial_states.strides()[2];
    } else {
        params.dinitial_states_ptr = nullptr;
    }

    auto stream = dx.stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.dtype(), "causal_conv1d_bwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(weight.dtype(), "causal_conv1d_bwd", [&] {
            if (!is_channel_last) {
                causal_conv1d_bwd_cuda<input_t, weight_t>(params, stream);
            } else {
                causal_conv1d_channellast_bwd_cuda<input_t, weight_t>(params, stream);
            }
        });
    });
    return {dx, dweight.cast(weight.dtype()), bias_.has_value() ? dbias.cast(bias_.value().dtype()) : dbias, dinitial_states};
}

paddle::Tensor
causal_conv1d_update(const paddle::Tensor &x,
                     const paddle::Tensor &conv_state,
                     const paddle::Tensor &weight,
                     const std::optional<paddle::Tensor> &bias_,
                     bool silu_activation,
                     const std::optional<paddle::Tensor> &cache_seqlens_
                     ) {
    auto input_type = x.dtype();
    auto weight_type = weight.dtype();
    PD_CHECK(input_type == paddle::DataType::FLOAT32 || input_type == paddle::DataType::FLOAT16 || input_type == paddle::DataType::BFLOAT16);
    PD_CHECK(weight_type == paddle::DataType::FLOAT32 || weight_type == paddle::DataType::FLOAT16 || weight_type == paddle::DataType::BFLOAT16);
    PD_CHECK(conv_state.dtype() == input_type);

    PD_CHECK(x.is_gpu());
    PD_CHECK(conv_state.is_gpu());
    PD_CHECK(weight.is_gpu());

    const auto sizes = x.dims();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.dims()[weight.dims().size() - 1];
    const int conv_state_len = conv_state.dims()[2];
    PD_CHECK(conv_state_len >= width - 1);

    CHECK_SHAPE(x, batch_size, dim, seqlen);
    CHECK_SHAPE(conv_state, batch_size, dim, conv_state_len);
    CHECK_SHAPE(weight, dim, width);

    PD_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        PD_CHECK(bias.dtype() == weight_type);
        PD_CHECK(bias.is_gpu());
        PD_CHECK(bias.strides()[bias.strides().size() - 1] == 1);
        CHECK_SHAPE(bias, dim);
    }

    paddle::Tensor out = paddle::empty_like(x);

    ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                        bias_.has_value() ? const_cast<void*>(bias_.value().data()) : nullptr,
                        silu_activation);
    params.conv_state_ptr = const_cast<void*>(conv_state.data());
    params.conv_state_len = conv_state_len;
    // All stride are in elements, not bytes.
    params.conv_state_batch_stride = conv_state.strides()[0];
    params.conv_state_c_stride = conv_state.strides()[1];
    params.conv_state_l_stride = conv_state.strides()[2];

    if (cache_seqlens_.has_value()) {
        auto cache_seqlens = cache_seqlens_.value();
        PD_CHECK(cache_seqlens.dtype() == paddle::DataType::INT32 || cache_seqlens.dtype() == paddle::DataType::INT64);
        PD_CHECK(cache_seqlens.is_gpu());
        PD_CHECK(cache_seqlens.strides()[cache_seqlens.dims().size() - 1] == 1);
        CHECK_SHAPE(cache_seqlens, batch_size);
        params.cache_seqlens = cache_seqlens.data<int32_t>();
    } else {
        params.cache_seqlens = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    auto stream = x.stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.dtype(), "causal_conv1d_update", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(weight.dtype(), "causal_conv1d_update", [&] {
            causal_conv1d_update_cuda<input_t, weight_t>(params, stream);
        });
    });
    return out;
}

PYBIND11_MODULE(causal_conv1d_cuda_pd, m) {
    m.def("causal_conv1d_fwd", &causal_conv1d_fwd, "Causal conv1d forward");
    m.def("causal_conv1d_bwd", &causal_conv1d_bwd, "Causal conv1d backward");
    m.def("causal_conv1d_update", &causal_conv1d_update, "Causal conv1d update");
}
