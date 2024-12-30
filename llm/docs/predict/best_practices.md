# 最佳实践

PaddleNLP 提供了多种环境变量，用于优化推理性能和资源使用。下面提供一些调整 PaddleNLP 推理性能的最佳实践。

**GEMM 优化**

- `FLAGS_enable_blaslt_global_search`：int8 gemm 是否开启全局调优，默认值为0，表示不开启。设置为1，PaddleNLP 会在推理过程中动态搜索最优的 gemm 算法。推理 A8W8模型时使用此 FLAG 会获得更优的性能。


- `FLAGS_cublaslt_device_best_config`：在 FLAGS_enable_blaslt_global_search 设为1的前提下，使用`FLAGS_cublaslt_device_best_config`来指定离线调优出的 int8 gemm 配置文件，默认值为""。配置文件可以通过`PaddleNLP/csrc/utils/tune_cublaslt_int8_gemm.py`产出，该脚本会自动搜索当前输入大小下 cuBLASLt 提供的最优 gemm 配置并将结果记录下来，需要注意的是不同的 CUDA 版本需要分别 tune。推理 A8W8模型并且 FLAGS_enable_blaslt_global_search 设为1时使用此 FLAG 会获得更优的性能。

- `FLAGS_CUTLASS_FP8_GEMM`: fp8 gemm 是否使用 cutlass 版本，默认值为 False, 表示不开启。设置为 True，则在 A8W8模型上使用 Tensor Core 进行计算，性能会更好。

- `FLAGS_use_cutlass_device_best_config_path`: 在 `FLAGS_CUTLASS_FP8_GEMM` 设为 True 的前提下，使用该环境变量来指定离线调优出的 fp8 gemm 配置文件。配置文件可以通过`PaddleNLP/csrc/utils/tune_cutlass_fp8_*.py`产出，该脚本会自动搜索当前输入大小下提供的最优 gemm 配置并将结果记录下来，默认产出文件为`fp8_fuse_gemm_config.json`。不同 NVIDIA GPU 和 CUDA 版本需要分别调优，SM89架构 GPU 增加 dual_gemm 调优，具体可参考`dual_gemm.py`。可选值：`tune`，开启调优；空值或`default`，使用默认配置；任意值，优先使用配置文件中的参数，若无则使用默认配置。

- `FLAGS_cuda_core_int8_gemm`：是否开启小 Batch Int8 Gemm 优化，默认值不开启。设为1可开启，推理 A8W8模型时，平均性能会加速约40%-55%，适用于 SM>=70的显卡。

- `FLAGS_cuda_core_fp8_gemm`：是否开启小 Batch FP8 Gemm 优化，默认值不开启。设为1可开启，推理 FP8模型时，平均性能会加速约30%左右，适用于 SM>=89的显卡。

**GQA 优化**

- `FLAGS_use_xqa_optim`：gpa 是否开启 xqa 优化，默认值为0，表示不开启。gqa 模型（如 llama3/3.1、qwen2）设为1性能会更好。

**显存优化**

- `FLAGS_fraction_of_gpu_memory_to_use`：GPU 显存使用率，默认值为0.9。设置为0.9即可。

- `FLAGS_gemm_use_half_precision_compute_type`：是否使用半精度浮点数计算，默认值为0。设置为0即可。
