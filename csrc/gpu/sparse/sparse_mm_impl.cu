// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// clang-format will break include orders
// clang-format off
#include <cudaTypedefs.h>

#if defined CUDA_VERSION && CUDA_VERSION >= 12020
#include "sparse_mm_impl.cuh"
// clang-format on

using namespace cute;
using namespace paddle;

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_gemm_sm90_fp8_dispatch(paddle::Tensor& out, paddle::Tensor const& a,
                                    paddle::Tensor const& bt_nzs,
                                    paddle::Tensor const& bt_meta,
                                    EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());

  using Cutlass3xGemmDefault =
      typename sm90_config_default<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM64 =
      typename sm90_fp8_config_M64<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM128 =
      typename sm90_fp8_config_M128<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM256 =
      typename sm90_fp8_config_M256<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM512 =
      typename sm90_fp8_config_M512<InType, OutType, Epilogue>::Cutlass3xGemm;

  using Cutlass3xGemm1 =
      typename sm90_fp8_config_1<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm2 =
      typename sm90_fp8_config_2<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm3 =
      typename sm90_fp8_config_3<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm4 =
      typename sm90_fp8_config_4<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm5 =
      typename sm90_fp8_config_5<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm6 =
      typename sm90_fp8_config_6<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm7 =
      typename sm90_fp8_config_7<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemm8 =
      typename sm90_fp8_config_8<InType, OutType, Epilogue>::Cutlass3xGemm;

  uint32_t const n = a.dims()[1]; //bt_nzs.dims()[1];
  uint32_t const m = a.dims()[0]; // Batch size
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(64), next_pow_2(m));  // next power of 2
  if (mp2 <= 64) {
    if (n == 28672) {
      return cutlass_sparse_gemm_caller<Cutlass3xGemm2>(
          out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
    } else if (n == 4096 || n == 6144) {
      return cutlass_sparse_gemm_caller<Cutlass3xGemm1>(
          out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
    }
  } else if (mp2 <= 128) {
    if (n == 4096) {
      return cutlass_sparse_gemm_caller<Cutlass3xGemm3>(
          out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
    } else if (n == 28672) {
      return cutlass_sparse_gemm_caller<Cutlass3xGemm5>(
          out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
    } else if (n == 6144) {
      return cutlass_sparse_gemm_caller<Cutlass3xGemm4>(
          out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
    }
  } else if (mp2 <= 256) {
    if (n == 4096) {
      return cutlass_sparse_gemm_caller<Cutlass3xGemm6>(
          out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
    } else if (n == 28672) {
      return cutlass_sparse_gemm_caller<Cutlass3xGemm8>(
          out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
    } else if (n == 6144) {
      return cutlass_sparse_gemm_caller<Cutlass3xGemm7>(
          out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
    }
  } else {
    if (n == 6144 || n == 28672) {
      return cutlass_sparse_gemm_caller<Cutlass3xGemm8>(
          out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
    } else if (n == 4096) {
      return cutlass_sparse_gemm_caller<Cutlass3xGemm7>(
          out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
    }
  }

  // Otherwise the default heuristic
  if (mp2 <= 64) {
    // n in [1, 64]
    return cutlass_sparse_gemm_caller<Cutlass3xGemmM64>(
        out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // n in (64, 128]
    return cutlass_sparse_gemm_caller<Cutlass3xGemmM128>(
        out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 256) {
    // n in (128, 256]
    return cutlass_sparse_gemm_caller<Cutlass3xGemmM256>(
        out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
  } else {
    // n in (256, inf)
    return cutlass_sparse_gemm_caller<Cutlass3xGemmM512>(
        out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
  }
}

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_gemm_sm90_fp16_dispatch(paddle::Tensor& out, paddle::Tensor const& a,
                                     paddle::Tensor const& bt_nzs,
                                     paddle::Tensor const& bt_meta,
                                     EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::half_t>());

  using Cutlass3xGemmDefault =
      typename sm90_config_default<InType, OutType, Epilogue>::Cutlass3xGemm;

  // m in (128, inf)
  return cutlass_sparse_gemm_caller<Cutlass3xGemmDefault>(
      out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
}

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_gemm_sm90_bf16_dispatch(paddle::Tensor& out, paddle::Tensor const& a,
                                     paddle::Tensor const& bt_nzs,
                                     paddle::Tensor const& bt_meta,
                                     EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::bfloat16_t>());

  using Cutlass3xGemmDefault =
      typename sm90_config_default<InType, OutType, Epilogue>::Cutlass3xGemm;

  // m in (128, inf)
  return cutlass_sparse_gemm_caller<Cutlass3xGemmDefault>(
      out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
}

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_gemm_sm90_int8_dispatch(paddle::Tensor& out, paddle::Tensor const& a,
                                     paddle::Tensor const& bt_nzs,
                                     paddle::Tensor const& bt_meta,
                                     EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, int8_t>());

  using Cutlass3xGemmDefault =
      typename sm90_config_default<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM128 =
      typename sm90_int8_config_M128<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM64 =
      typename sm90_int8_config_M64<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM32NBig =
      typename sm90_int8_config_M32_NBig<InType, OutType,
                                         Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM32NSmall =
      typename sm90_int8_config_M32_NSmall<InType, OutType,
                                           Epilogue>::Cutlass3xGemm;

  uint32_t const n = out.dims()[1];
  bool const is_small_n = n < 8192;

  uint32_t const m = a.dims()[0];
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(32), next_pow_2(m));  // next power of 2

  if (mp2 <= 32) {
    // m in [1, 32]
    if (is_small_n) {
      return cutlass_sparse_gemm_caller<Cutlass3xGemmM32NSmall>(
          out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
    } else {
      return cutlass_sparse_gemm_caller<Cutlass3xGemmM32NBig>(
          out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
    }
  } else if (mp2 <= 64) {
    // m in (32, 64]
    return cutlass_sparse_gemm_caller<Cutlass3xGemmM64>(
        out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // m in (64, 128]
    return cutlass_sparse_gemm_caller<Cutlass3xGemmM128>(
        out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
  } else {
    // m in (128, inf)
    return cutlass_sparse_gemm_caller<Cutlass3xGemmDefault>(
        out, a, bt_nzs, bt_meta, std::forward<EpilogueArgs>(args)...);
  }
}

template <template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_sparse_mm_sm90_epilogue(paddle::Tensor& out,
                                            paddle::Tensor const& a,
                                            paddle::Tensor const& bt_nzs,
                                            paddle::Tensor const& bt_meta,
                                            EpilogueArgs&&... epilogue_args) {
  if (a.dtype() == phi::DataType::INT8) {
    if (out.dtype() == phi::DataType::BFLOAT16) {
      return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::bfloat16_t,
                                             Epilogue>(
          out, a, bt_nzs, bt_meta,
          std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::half_t, Epilogue>(
          out, a, bt_nzs, bt_meta,
          std::forward<EpilogueArgs>(epilogue_args)...);
    }
  } else if (a.dtype() == phi::DataType::FLOAT8_E4M3FN) {
    if (out.dtype() == phi::DataType::BFLOAT16) {
      return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::bfloat16_t, Epilogue>(
          out, a, bt_nzs, bt_meta,
          std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::half_t, Epilogue>(
          out, a, bt_nzs, bt_meta,
          std::forward<EpilogueArgs>(epilogue_args)...);
    }
  } else if (a.dtype() == phi::DataType::BFLOAT16) {
    if (out.dtype() == phi::DataType::BFLOAT16) {
      return cutlass_gemm_sm90_bf16_dispatch<cutlass::bfloat16_t,
                                            cutlass::bfloat16_t, Epilogue>(
          out, a, bt_nzs, bt_meta,
          std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      return cutlass_gemm_sm90_bf16_dispatch<cutlass::bfloat16_t,
                                            cutlass::half_t, Epilogue>(
          out, a, bt_nzs, bt_meta,
          std::forward<EpilogueArgs>(epilogue_args)...);
    }
  } else if (a.dtype() == phi::DataType::FLOAT16) {
    if (out.dtype() == phi::DataType::BFLOAT16) {
      return cutlass_gemm_sm90_fp16_dispatch<cutlass::half_t,
                                            cutlass::bfloat16_t, Epilogue>(
          out, a, bt_nzs, bt_meta,
          std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      return cutlass_gemm_sm90_fp16_dispatch<cutlass::half_t,
                                            cutlass::half_t, Epilogue>(
          out, a, bt_nzs, bt_meta,
          std::forward<EpilogueArgs>(epilogue_args)...);
    }
  }
}

void cutlass_scaled_sparse_mm_sm90(paddle::Tensor& out, paddle::Tensor const& a,
                                   paddle::Tensor const& bt_nzs,
                                   paddle::Tensor const& bt_meta,
                                   paddle::Tensor const& a_scales,
                                   paddle::Tensor const& b_scales,
                                   std::optional<paddle::Tensor> const& bias) {
  if (bias) {
    return cutlass_scaled_sparse_mm_sm90_epilogue<c3x::ScaledEpilogueColumnBias>(
        out, a, bt_nzs, bt_meta, b_scales,  a_scales, *bias);
  } else {
    return cutlass_scaled_sparse_mm_sm90_epilogue<c3x::ScaledEpilogue>(
        out, a, bt_nzs, bt_meta, b_scales , a_scales);
  }
}

#endif
