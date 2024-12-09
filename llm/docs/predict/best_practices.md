# 最佳实践

PaddleNLP 提供了多种环境变量，用于优化推理性能和资源使用。下面提供一些调整 PaddleNLP 推理性能的最佳实践。

**GEMM 优化**

- `FLAGS_enable_blaslt_global_search`：int8 gemm 是否开启全局调优，默认值为0，表示不开启。设置为1，PaddleNLP 会在推理过程中动态搜索最优的 gemm 算法。推理 A8W8模型时使用此 FLAG 会获得更优的性能。


- `FLAGS_cublaslt_device_best_config`：在 FLAGS_enable_blaslt_global_search 设为1的前提下，使用`FLAGS_cublaslt_device_best_config`来指定离线调优出的 int8 gemm 配置文件，默认值为""。配置文件可以通过`PaddleNLP/csrc/utils/tune_cublaslt_int8_gemm.py`产出，该脚本会自动搜索当前输入大小下 cuBLASLt 提供的最优 gemm 配置并将结果记录下来，需要注意的是不同的 CUDA 版本需要分别 tune。推理 A8W8模型并且 FLAGS_enable_blaslt_global_search 设为1时使用此 FLAG 会获得更优的性能。

- `FLAGS_cuda_core_int8_gemm`：是否开启小 Batch Int8 Gemm优化，默认值不开启。设为1可开启，推理A8W8模型时，平均性能会加速约40%-55%，适用于SM>=70的显卡。

- `FLAGS_cuda_core_fp8_gemm`：是否开启小 Batch FP8 Gemm优化，默认值不开启。设为1可开启，推理 FP8模型时，平均性能会加速约30%左右，适用于SM>=89的显卡。

**GQA 优化**

- `FLAGS_use_xqa_optim`：gpa 是否开启 xqa 优化，默认值为0，表示不开启。gqa 模型（如 llama3/3.1、qwen2）设为1性能会更好。

**显存优化**

- `FLAGS_fraction_of_gpu_memory_to_use`：GPU 显存使用率，默认值为0.9。设置为0.9即可。

- `FLAGS_gemm_use_half_precision_compute_type`：是否使用半精度浮点数计算，默认值为0。设置为0即可。
