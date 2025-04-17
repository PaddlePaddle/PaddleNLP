/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <paddle/extension.h>
#include <paddle/phi/common/data_type.h>
#include <vector>

#include "selective_scan.h"

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
        PADDLE_THROW(#NAME, " not implemented for weight type '", WTYPE, "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_COMPLEX(WTYPE, NAME, ...)                           \
    if (WTYPE == paddle::DataType::FLOAT32) {                                        \
       using weight_t = float;                                                       \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == paddle::DataType::COMPLEX64) {                               \
        using weight_t = phi::dtype::complex<float>;                                 \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        PADDLE_THROW(#NAME, " not implemented for weight type '", WTYPE, "'");       \
    }

template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);

template <typename input_t, typename weight_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream);

void set_ssm_params_fwd(SSMParamsBase &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const paddle::Tensor u,
                        const paddle::Tensor delta,
                        const paddle::Tensor A,
                        const paddle::Tensor B,
                        const paddle::Tensor C,
                        const paddle::Tensor out,
                        const paddle::Tensor z,
                        const paddle::Tensor out_z,
                        void* D_ptr,
                        void* delta_bias_ptr,
                        void* x_ptr,
                        bool has_z,
                        bool delta_softplus) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.dstate = dstate;
    params.n_groups = n_groups;
    params.n_chunks = n_chunks;
    params.dim_ngroups_ratio = dim / n_groups;

    params.delta_softplus = delta_softplus;

    params.is_variable_B = is_variable_B;
    params.is_variable_C = is_variable_C;

    // Set the pointers and strides.
    params.u_ptr = const_cast<void*>(u.data());
    params.delta_ptr = const_cast<void*>(delta.data());
    params.A_ptr = const_cast<void*>(A.data());
    params.B_ptr = const_cast<void*>(B.data());
    params.C_ptr = const_cast<void*>(C.data());
    params.D_ptr = const_cast<void*>(D_ptr);
    params.delta_bias_ptr = const_cast<void*>(delta_bias_ptr);
    params.out_ptr = const_cast<void*>(out.data());
    params.x_ptr = const_cast<void*>(x_ptr);
    params.z_ptr = has_z ? const_cast<void*>(z.data()) : nullptr;
    params.out_z_ptr = has_z ? const_cast<void*>(out_z.data()) : nullptr;
    // All stride are in elements, not bytes.
    params.A_d_stride = A.strides()[0];
    params.A_dstate_stride = A.strides()[1];
    if (!is_variable_B) {
        params.B_d_stride = B.strides()[0];
    } else {
        params.B_batch_stride = B.strides()[0];
        params.B_group_stride = B.strides()[1];
    }
    params.B_dstate_stride = !is_variable_B ? B.strides()[1] : B.strides()[2];
    if (!is_variable_C) {
        params.C_d_stride = C.strides()[0];
    } else {
        params.C_batch_stride = C.strides()[0];
        params.C_group_stride = C.strides()[1];
    }
    params.C_dstate_stride = !is_variable_C ? C.strides()[1] : C.strides()[2];
    params.u_batch_stride = u.strides()[0];
    params.u_d_stride = u.strides()[1];
    params.delta_batch_stride = delta.strides()[0];
    params.delta_d_stride = delta.strides()[1];
    if (has_z) {
        params.z_batch_stride = z.strides()[0];
        params.z_d_stride = z.strides()[1];
        params.out_z_batch_stride = out_z.strides()[0];
        params.out_z_d_stride = out_z.strides()[1];
    }
    params.out_batch_stride = out.strides()[0];
    params.out_d_stride = out.strides()[1];
}

void set_ssm_params_bwd(SSMParamsBwd &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const paddle::Tensor u,
                        const paddle::Tensor delta,
                        const paddle::Tensor A,
                        const paddle::Tensor B,
                        const paddle::Tensor C,
                        const paddle::Tensor z,
                        const paddle::Tensor out,
                        const paddle::Tensor out_z,
                        void* D_ptr,
                        void* delta_bias_ptr,
                        void* x_ptr,
                        const paddle::Tensor dout,
                        const paddle::Tensor du,
                        const paddle::Tensor ddelta,
                        const paddle::Tensor dA,
                        const paddle::Tensor dB,
                        const paddle::Tensor dC,
                        const paddle::Tensor dz,
                        void* dD_ptr,
                        void* ddelta_bias_ptr,
                        bool has_z,
                        bool delta_softplus,
                        bool recompute_out_z) {
    // Pass in "dout" instead of "out", we're not gonna use "out" unless we have z
    set_ssm_params_fwd(params, batch, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, has_z ? out : dout,
                       has_z ? z : dout,
                       // If not recompute_out_z, pass dout instead of out_z.
                       // This won't be used by the bwd kernel
                       recompute_out_z ? out_z : dout,
                       D_ptr, delta_bias_ptr, x_ptr, has_z, delta_softplus);
    if (!recompute_out_z) { params.out_z_ptr = nullptr; }

    // Set the pointers and strides.
    params.dout_ptr = const_cast<void*>(dout.data());
    params.du_ptr = const_cast<void*>(du.data());
    params.dA_ptr = const_cast<void*>(dA.data());
    params.dB_ptr = const_cast<void*>(dB.data());
    params.dC_ptr = const_cast<void*>(dC.data());
    params.dD_ptr = const_cast<void*>(dD_ptr);
    params.ddelta_ptr = const_cast<void*>(ddelta.data());
    params.ddelta_bias_ptr = const_cast<void*>(ddelta_bias_ptr);
    params.dz_ptr = has_z ? const_cast<void*>(dz.data()) : nullptr;
    // All stride are in elements, not bytes.
    params.dout_batch_stride = dout.strides()[0];
    params.dout_d_stride = dout.strides()[1];
    params.dA_d_stride = dA.strides()[0];
    params.dA_dstate_stride = dA.strides()[1];
    if (!is_variable_B) {
        params.dB_d_stride = dB.strides()[0];
    } else {
        params.dB_batch_stride = dB.strides()[0];
        params.dB_group_stride = dB.strides()[1];
    }
    params.dB_dstate_stride = !is_variable_B ? dB.strides()[1] : dB.strides()[2];
    if (!is_variable_C) {
        params.dC_d_stride = dC.strides()[0];
    } else {
        params.dC_batch_stride = dC.strides()[0];
        params.dC_group_stride = dC.strides()[1];
    }
    params.dC_dstate_stride = !is_variable_C ? dC.strides()[1] : dC.strides()[2];
    params.du_batch_stride = du.strides()[0];
    params.du_d_stride = du.strides()[1];
    params.ddelta_batch_stride = ddelta.strides()[0];
    params.ddelta_d_stride = ddelta.strides()[1];
    if (has_z) {
        params.dz_batch_stride = dz.strides()[0];
        params.dz_d_stride = dz.strides()[1];
    }
}

std::vector<paddle::Tensor>
selective_scan_fwd(const paddle::Tensor &u, const paddle::Tensor &delta,
                  const paddle::Tensor &A, const paddle::Tensor &B, const paddle::Tensor &C,
                  const std::optional<paddle::Tensor> &D_,
                  const std::optional<paddle::Tensor> &z_,
                  const std::optional<paddle::Tensor> &delta_bias_,
                  bool delta_softplus) {
    auto input_type = u.dtype();
    auto weight_type = A.dtype();
    PD_CHECK(input_type == paddle::DataType::FLOAT32 || input_type == paddle::DataType::FLOAT16 || input_type == paddle::DataType::BFLOAT16);
    PD_CHECK(weight_type == paddle::DataType::FLOAT32 || weight_type == paddle::DataType::COMPLEX64);

    const bool is_variable_B = B.dims().size() >= 3;
    const bool is_variable_C = C.dims().size() >= 3;
    const bool is_complex = weight_type == paddle::DataType::COMPLEX64;

    PD_CHECK(delta.dtype() == input_type);
    PD_CHECK(B.dtype() == (!is_variable_B ? weight_type : input_type));
    PD_CHECK(C.dtype() == (!is_variable_C ? weight_type : input_type));

    PD_CHECK(u.is_gpu());
    PD_CHECK(delta.is_gpu());
    PD_CHECK(A.is_gpu());
    PD_CHECK(B.is_gpu());
    PD_CHECK(C.is_gpu());

    PD_CHECK(u.strides()[u.strides().size() - 1] == 1 || u.dims()[u.dims().size() - 1] == 1);
    PD_CHECK(delta.strides()[delta.strides().size() - 1] == 1 || delta.dims()[delta.dims().size() - 1] == 1);

    const auto sizes = u.dims();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.dims()[1];
    const int n_groups = is_variable_B ? B.dims()[1] : 1;

    PD_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    if (!is_variable_B) {
        CHECK_SHAPE(B, dim, dstate);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, !is_complex ? seqlen : seqlen * 2);
        PD_CHECK(B.strides()[B.strides().size() - 1] == 1 || B.dims()[B.dims().size() - 1] == 1);
    }
    if (!is_variable_C) {
        CHECK_SHAPE(C, dim, dstate);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, !is_complex ? seqlen: seqlen * 2);
        PD_CHECK(C.strides()[C.strides().size() - 1] == 1 || C.dims()[C.dims().size() - 1] == 1);
    }

    if (D_.has_value()) {
        auto D = D_.value();
        PD_CHECK(D.dtype() == paddle::DataType::FLOAT32);
        PD_CHECK(D.is_gpu());
        PD_CHECK(D.strides()[D.strides().size() - 1] == 1 || D.dims()[D.dims().size() - 1] == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        PD_CHECK(delta_bias.dtype() == paddle::DataType::FLOAT32);
        PD_CHECK(delta_bias.is_gpu());
        PD_CHECK(delta_bias.strides()[delta_bias.strides().size() - 1] == 1 || delta_bias.dims()[delta_bias.dims().size() - 1] == 1);
        CHECK_SHAPE(delta_bias, dim);
    }

    paddle::Tensor z, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        PD_CHECK(z.dtype() == input_type);
        PD_CHECK(z.is_gpu());
        PD_CHECK(z.strides()[z.strides().size() - 1] == 1 || z.dims()[z.dims().size() - 1] == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);
        out_z = paddle::empty_like(z);
    }

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    // paddle::Tensor out = paddle::empty_like(u);
    // Right now u has BHL layout and delta has HBL layout, and we want out to have HBL layout
    paddle::Tensor out = paddle::empty_like(delta);
    paddle::Tensor x;
    x = paddle::empty({batch_size, dim, n_chunks, dstate * 2}, weight_type, delta.place());

    SSMParamsBase params;
    set_ssm_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, out, z, out_z,
                       D_.has_value() ? const_cast<void*>(D_.value().data()) : nullptr,
                       delta_bias_.has_value() ? const_cast<void*>(delta_bias_.value().data()) : nullptr,
                       x.data(),
                       has_z,
                       delta_softplus);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    auto stream = x.stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.dtype(), "selective_scan_fwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_COMPLEX(A.dtype(), "selective_scan_fwd", [&] {
            selective_scan_fwd_cuda<input_t, weight_t>(params, stream);
        });
    });
    std::vector<paddle::Tensor> result = {out, x};
    if (has_z) { result.push_back(out_z); }
    return result;
}

std::vector<paddle::Tensor>
selective_scan_bwd(const paddle::Tensor &u, const paddle::Tensor &delta,
                  const paddle::Tensor &A, const paddle::Tensor &B, const paddle::Tensor &C,
                  const std::optional<paddle::Tensor> &D_,
                  const std::optional<paddle::Tensor> &z_,
                  const std::optional<paddle::Tensor> &delta_bias_,
                  const paddle::Tensor &dout,
                  const std::optional<paddle::Tensor> &x_,
                  const std::optional<paddle::Tensor> &out_,
                  std::optional<paddle::Tensor> &dz_,
                  bool delta_softplus,
                  bool recompute_out_z) {
    auto input_type = u.dtype();
    auto weight_type = A.dtype();
    PD_CHECK(input_type == paddle::DataType::FLOAT32 || input_type == paddle::DataType::FLOAT16 || input_type == paddle::DataType::BFLOAT16);
    PD_CHECK(weight_type == paddle::DataType::FLOAT32 || weight_type == paddle::DataType::COMPLEX64);

    const bool is_variable_B = B.dims().size() >= 3;
    const bool is_variable_C = C.dims().size() >= 3;
    const bool is_complex = weight_type == paddle::DataType::COMPLEX64;

    PD_CHECK(delta.dtype() == input_type);
    PD_CHECK(B.dtype() == (!is_variable_B ? weight_type : input_type));
    PD_CHECK(C.dtype() == (!is_variable_C ? weight_type : input_type));
    PD_CHECK(dout.dtype() == input_type);

    PD_CHECK(u.is_gpu());
    PD_CHECK(delta.is_gpu());
    PD_CHECK(A.is_gpu());
    PD_CHECK(B.is_gpu());
    PD_CHECK(C.is_gpu());
    PD_CHECK(dout.is_gpu());

    PD_CHECK(u.strides()[u.strides().size() - 1] == 1 || u.dims()[u.dims().size() - 1] == 1);
    PD_CHECK(delta.strides()[delta.strides().size() - 1] == 1 || delta.dims()[delta.dims().size() - 1] == 1);
    PD_CHECK(dout.strides()[dout.strides().size() - 1] == 1 || dout.dims()[dout.dims().size() - 1] == 1);

    const auto sizes = u.dims();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.dims()[1];
    const int n_groups = is_variable_B ? B.dims()[1] : 1;

    PD_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    if (!is_variable_B) {
        CHECK_SHAPE(B, dim, dstate);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, !is_complex ? seqlen : seqlen * 2);
        PD_CHECK(B.strides()[B.strides().size() - 1] == 1 || B.dims()[B.dims().size() - 1] == 1);
    }
    if (!is_variable_C) {
        CHECK_SHAPE(C, dim, dstate);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, !is_complex ? seqlen: seqlen * 2);
        PD_CHECK(C.strides()[C.strides().size() - 1] == 1 || C.dims()[C.dims().size() - 1] == 1);
    }
    CHECK_SHAPE(dout, batch_size, dim, seqlen);

    if (D_.has_value()) {
        auto D = D_.value();
        PD_CHECK(D.dtype() == paddle::DataType::FLOAT32);
        PD_CHECK(D.is_gpu());
        PD_CHECK(D.strides()[D.strides().size() - 1] == 1 || D.dims()[D.dims().size() - 1] == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        PD_CHECK(delta_bias.dtype() == paddle::DataType::FLOAT32);
        PD_CHECK(delta_bias.is_gpu());
        PD_CHECK(delta_bias.strides()[delta_bias.strides().size() - 1] == 1 || delta_bias.dims()[delta_bias.dims().size() - 1] == 1);
        CHECK_SHAPE(delta_bias, dim);
    }

    paddle::Tensor z, out, dz, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        PD_CHECK(z.dtype() == input_type);
        PD_CHECK(z.is_gpu());
        PD_CHECK(z.strides()[z.strides().size() - 1] == 1 || z.dims()[z.dims().size() - 1] == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);

        PD_CHECK(out_.has_value());
        out = out_.value();
        PD_CHECK(out.dtype() == input_type);
        PD_CHECK(out.is_gpu());
        PD_CHECK(out.strides()[out.strides().size() - 1] == 1 || out.dims()[out.dims().size() - 1] == 1);
        CHECK_SHAPE(out, batch_size, dim, seqlen);

        if (dz_.has_value()) {
            dz = dz_.value();
            PD_CHECK(dz.dtype() == input_type);
            PD_CHECK(dz.is_gpu());
            PD_CHECK(dz.strides()[dz.strides().size() - 1] == 1 || dz.dims()[dz.dims().size() - 1] == 1);
            CHECK_SHAPE(dz, batch_size, dim, seqlen);
        } else {
            dz = paddle::empty_like(z);
        }
        if (recompute_out_z) {
            out_z = paddle::empty_like(out);
        }
    }

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    if (n_chunks > 1) { PD_CHECK(x_.has_value()); }
    if (x_.has_value()) {
        auto x = x_.value();
        PD_CHECK(x.dtype() == weight_type);
        PD_CHECK(x.is_gpu());
        // PD_CHECK(x.is_contiguous());
        CHECK_SHAPE(x, batch_size, dim, n_chunks, 2 * dstate);
    }

    paddle::Tensor du = paddle::empty_like(u);
    paddle::Tensor ddelta = paddle::empty_like(delta);
    paddle::Tensor dA = paddle::experimental::zeros_like(A);
    paddle::Tensor dB = !is_variable_B ? paddle::experimental::zeros_like(B) : paddle::experimental::zeros_like(B, paddle::DataType::FLOAT32);
    paddle::Tensor dC = !is_variable_C ? paddle::experimental::zeros_like(C) : paddle::experimental::zeros_like(C, paddle::DataType::FLOAT32);
    paddle::Tensor dD;
    if (D_.has_value()) { dD = paddle::experimental::zeros_like(D_.value()); }
    paddle::Tensor ddelta_bias;
    if (delta_bias_.has_value()) { ddelta_bias = paddle::experimental::zeros_like(delta_bias_.value()); }

    SSMParamsBwd params;
    set_ssm_params_bwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, z, out, out_z,
                       D_.has_value() ? const_cast<void*>(D_.value().data()) : nullptr,
                       delta_bias_.has_value() ? const_cast<void*>(delta_bias_.value().data()) : nullptr,
                       x_.has_value() ? const_cast<void*>(x_.value().data()) : nullptr,
                       dout, du, ddelta, dA, dB, dC, dz,
                       D_.has_value() ? const_cast<void*>(dD.data()) : nullptr,
                       delta_bias_.has_value() ? const_cast<void*>(ddelta_bias.data()) : nullptr,
                       has_z, delta_softplus, recompute_out_z);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    auto stream = u.stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.dtype(), "selective_scan_bwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_COMPLEX(A.dtype(), "selective_scan_bwd", [&] {
            selective_scan_bwd_cuda<input_t, weight_t>(params, stream);
        });
    });
    std::vector<paddle::Tensor> result = {du, ddelta, dA, dB.cast(B.dtype()), dC.cast(C.dtype()), dD, ddelta_bias};
    if (has_z) { result.push_back(dz); }
    if (recompute_out_z) { result.push_back(out_z); }
    return result;
}

PYBIND11_MODULE(selective_scan_cuda_pd, m) {
    m.def("fwd", &selective_scan_fwd, "Selective scan forward");
    m.def("bwd", &selective_scan_bwd, "Selective scan backward");
}
