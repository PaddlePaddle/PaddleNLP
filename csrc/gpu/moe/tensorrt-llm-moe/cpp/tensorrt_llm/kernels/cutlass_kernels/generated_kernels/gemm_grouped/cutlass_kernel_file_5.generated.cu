#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl"
namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{


            template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 64, 128, 64, 2, tensorrt_llm::cutlass_extensions::EpilogueOpDefaultSilu>
                    (cutlass::half_t const* A, cutlass::half_t const* B, cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
    

            template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 64, 128, 64, 3, tensorrt_llm::cutlass_extensions::EpilogueOpDefaultSilu>
                    (cutlass::half_t const* A, cutlass::half_t const* B, cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
    

            template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 64, 128, 64, 4, tensorrt_llm::cutlass_extensions::EpilogueOpDefaultSilu>
                    (cutlass::half_t const* A, cutlass::half_t const* B, cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
    

            template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 64, 128, 64, 2, tensorrt_llm::cutlass_extensions::EpilogueOpDefaultFtGelu>
                    (cutlass::half_t const* A, cutlass::half_t const* B, cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
    

            template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 64, 128, 64, 3, tensorrt_llm::cutlass_extensions::EpilogueOpDefaultFtGelu>
                    (cutlass::half_t const* A, cutlass::half_t const* B, cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
    

            template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 64, 128, 64, 4, tensorrt_llm::cutlass_extensions::EpilogueOpDefaultFtGelu>
                    (cutlass::half_t const* A, cutlass::half_t const* B, cutlass::half_t const* biases, bool bias_is_broadcast, cutlass::half_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
    

            template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::bfloat16_t, cutlass::bfloat16_t, 64, 128, 64, 2, tensorrt_llm::cutlass_extensions::EpilogueOpDefaultSilu>
                    (cutlass::bfloat16_t const* A, cutlass::bfloat16_t const* B, cutlass::bfloat16_t const* biases, bool bias_is_broadcast, cutlass::bfloat16_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
    

            template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::bfloat16_t, cutlass::bfloat16_t, 64, 128, 64, 3, tensorrt_llm::cutlass_extensions::EpilogueOpDefaultSilu>
                    (cutlass::bfloat16_t const* A, cutlass::bfloat16_t const* B, cutlass::bfloat16_t const* biases, bool bias_is_broadcast, cutlass::bfloat16_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
    

            template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::bfloat16_t, cutlass::bfloat16_t, 64, 128, 64, 4, tensorrt_llm::cutlass_extensions::EpilogueOpDefaultSilu>
                    (cutlass::bfloat16_t const* A, cutlass::bfloat16_t const* B, cutlass::bfloat16_t const* biases, bool bias_is_broadcast, cutlass::bfloat16_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
    

            template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::bfloat16_t, cutlass::bfloat16_t, 64, 128, 64, 2, tensorrt_llm::cutlass_extensions::EpilogueOpDefaultFtGelu>
                    (cutlass::bfloat16_t const* A, cutlass::bfloat16_t const* B, cutlass::bfloat16_t const* biases, bool bias_is_broadcast, cutlass::bfloat16_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
    

            template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::bfloat16_t, cutlass::bfloat16_t, 64, 128, 64, 3, tensorrt_llm::cutlass_extensions::EpilogueOpDefaultFtGelu>
                    (cutlass::bfloat16_t const* A, cutlass::bfloat16_t const* B, cutlass::bfloat16_t const* biases, bool bias_is_broadcast, cutlass::bfloat16_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
    

            template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::bfloat16_t, cutlass::bfloat16_t, 64, 128, 64, 4, tensorrt_llm::cutlass_extensions::EpilogueOpDefaultFtGelu>
                    (cutlass::bfloat16_t const* A, cutlass::bfloat16_t const* B, cutlass::bfloat16_t const* biases, bool bias_is_broadcast, cutlass::bfloat16_t* C, int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream, int* kernel_occupancy);
    

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
