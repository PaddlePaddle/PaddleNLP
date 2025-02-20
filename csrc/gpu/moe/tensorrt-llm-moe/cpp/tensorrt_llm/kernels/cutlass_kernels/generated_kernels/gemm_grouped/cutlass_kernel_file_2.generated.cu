#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_launcher_sm90.inl"
namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{


        template void sm90_generic_moe_gemm_kernelLauncher<half, half, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<half, half, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<half, half, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<half, half, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<half, half, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<half, half, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<half, half, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<half, half, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<float, float, float,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<32>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<float, float, float,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<32>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<float, float, float,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<32>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<float, float, float,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<32>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<float, float, float,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<32>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<float, float, float,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<32>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<float, float, float,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<32>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<float, float, float,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<32>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::NONE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, half,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


        template void sm90_generic_moe_gemm_kernelLauncher<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                tensorrt_llm::cutlass_extensions::EpilogueOpDefault, tensorrt_llm::HopperGroupedGemmInput::EpilogueFusion::FINALIZE, cute::Shape<cute::Int<256>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, false>
                (HopperGroupedGemmInput, int, int, cudaStream_t, int*, size_t*);


} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
