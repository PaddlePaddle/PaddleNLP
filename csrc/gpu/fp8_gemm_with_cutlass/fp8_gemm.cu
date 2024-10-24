#include "fuse_gemm_gelu_template_fp8.h"

template<>
bool dispatch_fuse_gemm_gelu_fp8_noact<phi::dtype::float8_e4m3fn, phi::dtype::float16, phi::dtype::float8_e4m3fn,
                                        cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>,
                                        cutlass::gemm::GemmShape<16, 8, 32>, 2, true, cutlass::arch::Sm89>(GemmEpilogueAllParamsFP8);

template<>
bool dispatch_fuse_gemm_gelu_fp8_noact<phi::dtype::float8_e4m3fn, phi::dtype::float16, phi::dtype::float8_e4m3fn,
                                        cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>,
                                        cutlass::gemm::GemmShape<16, 8, 32>, 2, false, cutlass::arch::Sm89>(GemmEpilogueAllParamsFP8);


// template<>
// bool dispatch_fuse_gemm_gelu_fp8_relu<phi::dtype::float8_e4m3fn, phi::dtype::float16, phi::dtype::float8_e4m3fn,
//                                         cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>,
//                                         cutlass::gemm::GemmShape<16, 8, 32>, 2, true, cutlass::arch::Sm89>(GemmEpilogueAllParamsFP8);

// template<>
// bool dispatch_fuse_gemm_gelu_fp8_relu<phi::dtype::float8_e4m3fn, phi::dtype::float16, phi::dtype::float8_e4m3fn,
//                                         cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>,
//                                         cutlass::gemm::GemmShape<16, 8, 32>, 2, false, cutlass::arch::Sm89>(GemmEpilogueAllParamsFP8);


// template<>
// bool dispatch_fuse_gemm_gelu_fp8_gelu<phi::dtype::float8_e4m3fn, phi::dtype::float16, phi::dtype::float8_e4m3fn,
//                                         cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>,
//                                         cutlass::gemm::GemmShape<16, 8, 32>, 2, true, cutlass::arch::Sm89>(GemmEpilogueAllParamsFP8);

// template<>
// bool dispatch_fuse_gemm_gelu_fp8_gelu<phi::dtype::float8_e4m3fn, phi::dtype::float16, phi::dtype::float8_e4m3fn,
//                                         cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>,
//                                         cutlass::gemm::GemmShape<16, 8, 32>, 2, false, cutlass::arch::Sm89>(GemmEpilogueAllParamsFP8);