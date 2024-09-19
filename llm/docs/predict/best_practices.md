# 最佳实践

PaddleNLP 提供了多种环境变量，用于优化推理性能和资源使用。下面提供一些调整 PaddleNLP 推理性能的最佳实践。

**GEMM 优化**

- `FLAGS_enable_blaslt_global_search`：int8 gemm是否开启全局调优，默认值为0，表示不开启。设置为1，PaddleNLP 会在推理过程中使用`FLAGS_cublaslt_device_best_config`中记录的最优GEMM配置。

- `FLAGS_cublaslt_device_best_config`：指向性能最优的int8 gemm配置文件，默认值为""。配置文件可以通过`PaddleNLP/csrc/generation/test_tune_cublaslt_gemm.py`产出，该脚本会自动搜索当前输入大小下cuBLASLt提供的最优gemm配置并将结果记录下来。

**GQA 优化**

- `FLAGS_use_xqa_optim`：gpa是否开启xqa优化，默认值为0，表示不开启。gqa模型（如llama3/3.1、qwen2）设为1性能会更好。

**显存优化**

- `FLAGS_allocator_strategy`：显存管理策略，默认值为 `auto_growth`。可优先设为`naive_best_fit`，若显存oom可配置为`auto_growth`。

- `FLAGS_fraction_of_gpu_memory_to_use`：GPU显存使用率，默认值为0.9。设置为0.9即可。

- `FLAGS_gemm_use_half_precision_compute_type`：是否使用半精度浮点数计算，默认值为0。设置为0即可。
