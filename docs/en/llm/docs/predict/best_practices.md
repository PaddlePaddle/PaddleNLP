# Best Practices for High-Performance Inference

PaddleNLP provides multiple environment variables to optimize inference performance and resource utilization. Below are some best practices for tuning PaddleNLP inference performance.

**GEMM Optimization**

- `FLAGS_enable_blaslt_global_search`: Whether to enable global search for int8 gemm. Default value is 0 (disabled). When set to 1, PaddleNLP will dynamically search for the optimal gemm algorithm during inference. This FLAG improves performance when inferring A8W8 models.

- `FLAGS_cublaslt_device_best_config`: Specifies the pre-tuned int8 gemm configuration file when `FLAGS_enable_blaslt_global_search` is set to 1. Default value is "". The configuration file can be generated using `PaddleNLP/csrc/utils/tune_cublaslt_int8_gemm.py`, which automatically searches for optimal gemm configurations under current input sizes. Note that different CUDA versions require separate tuning. This FLAG provides better performance when inferring A8W8 models with `FLAGS_enable_blaslt_global_search=1`.

- `FLAGS_CUTLASS_FP8_GEMM`: Whether to use CUTLASS for fp8 gemm. Default is False. When set to True, Tensor Core will be used for A8W8 models, resulting in better performance.

- `FLAGS_use_cutlass_device_best_config_path`: Specifies the pre-tuned fp8 gemm configuration file when `FLAGS_CUTLASS_FP8_GEMM` is True. The configuration file can be generated using `csrc/utils/tune_cutlass_fp8_*.py`, which automatically searches for optimal gemm configurations and outputs `fp8_fuse_gemm_config.json` by default. Separate tuning is required for different NVIDIA GPUs and CUDA versions. SM89 and SM90 GPUs require dual_gemm tuning (refer to `dual_gemm.py`). Supported values: `tune` (enable tuning), empty or `default` (use default settings), or any other value (use specified config file). Also supports DeepSeek series models' Block-wise FP8 gemm tuning (refer to `csrc/tools/tune_fp8_gemm.sh`).

- `FLAGS_cuda_core_int8_gemm`: Whether to enable small-batch Int8 Gemm optimization. Default is disabled. Setting to 1 accelerates A8W8 model inference by ~40-55% on GPUs with SM>=70.

- `FLAGS_cuda_core_fp8_gemm`: Whether to enable small-batch FP8 Gemm optimization. Default is disabled. Setting to 1 accelerates FP8 model inference by ~30% on GPUs with SM>=89.

**GQA Optimization**

- `FLAGS_use_xqa_optim`: Whether to enable XQA optimization for GQA. Default is 0 (disabled). Setting to 1 improves performance for GQA models (e.g., llama3/3.1, qwen2).

**Memory Optimization**

- (Note: The original text appears incomplete here. Please provide full content for accurate translation.)
`FLAGS_fraction_of_gpu_memory_to_use`: GPU memory usage ratio, default is 0.9. Set to 0.9.

- `FLAGS_gemm_use_half_precision_compute_type`: Whether to use half-precision floating point computation, default is 0. Set to 0.

**Append Attention Optimization**

- `FLAGS_cascade_attention_max_partition_size`: Chunk size for cache_kv partitioning during Attention decoder computation. Default varies by batch_size: 128 when batch_size=1, 512 when batch_size>1. Explicit setting will override batch_size distinction.
- `FLAGS_dec_block_shape_q`: Block size for q partitioning in Append Attention decoder computation, default is 16.
- `FLAGS_enc_block_shape_q`: Block size for q partitioning in Append Attention encoder computation, default is 64.

**MLA-related Optimizations**
- `FLAGS_mla_use_tensorcore`: Whether to use tensor core implementation for MLA computation, default is True (only supported on Hopper architecture GPUs). When False, uses CUDA core implementation (supports both Ampere and Hopper architectures).

**allreduce Optimizations**
- `FLAGS_custom_allreduce`：Whether to use high-performance custom allreduce implementation for multi-card inference calculations. The default value is False.
