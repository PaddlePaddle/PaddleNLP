from paddle.utils.cpp_extension import CUDAExtension, setup
extra_compile_args={
        "nvcc": [
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-std=c++17"
        ]
}
setup(
    name='flash_atten2',
    ext_modules=CUDAExtension(
        sources = [
            'flash_attn_fwd.cu',
            "flash_attention/flash_fwd_hdim32_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim32_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim64_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim64_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim96_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim96_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim128_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim128_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim160_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim160_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim192_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim192_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim224_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim224_bf16_sm80.cu",
            "flash_attention/flash_fwd_hdim256_fp16_sm80.cu",
            "flash_attention/flash_fwd_hdim256_bf16_sm80.cu",
        ],
        include_dirs=['cutlass/include'],
        extra_compile_args= extra_compile_args
    ),
)