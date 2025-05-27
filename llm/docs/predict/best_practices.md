# 高性能推理最佳实践

PaddleNLP 提供了多种环境变量，用于优化推理性能和资源使用。下面提供一些调整 PaddleNLP 推理性能的最佳实践。

**GEMM 优化**

- `FLAGS_enable_blaslt_global_search`：int8 gemm 是否开启全局调优，默认值为0，表示不开启。设置为1，PaddleNLP 会在推理过程中动态搜索最优的 gemm 算法。推理 A8W8模型时使用此 FLAG 会获得更优的性能。


- `FLAGS_cublaslt_device_best_config`：在 FLAGS_enable_blaslt_global_search 设为1的前提下，使用`FLAGS_cublaslt_device_best_config`来指定离线调优出的 int8 gemm 配置文件，默认值为""。配置文件可以通过`PaddleNLP/csrc/utils/tune_cublaslt_int8_gemm.py`产出，该脚本会自动搜索当前输入大小下 cuBLASLt 提供的最优 gemm 配置并将结果记录下来，需要注意的是不同的 CUDA 版本需要分别 tune。推理 A8W8模型并且 FLAGS_enable_blaslt_global_search 设为1时使用此 FLAG 会获得更优的性能。

- `FLAGS_CUTLASS_FP8_GEMM`: fp8 gemm 是否使用 cutlass 版本，默认值为 False, 表示不开启。设置为 True，则在 A8W8模型上使用 Tensor Core 进行计算，性能会更好。

- `FLAGS_use_cutlass_device_best_config_path`: 在 `FLAGS_CUTLASS_FP8_GEMM` 设为 True 的前提下，使用该环境变量来指定离线调优出的 fp8 gemm 配置文件。配置文件可以通过`csrc/utils/tune_cutlass_fp8_*.py`产出，该脚本会自动搜索当前输入大小下提供的最优 gemm 配置并将结果记录下来，默认产出文件为`fp8_fuse_gemm_config.json`。不同 NVIDIA GPU 和 CUDA 版本需要分别调优，SM89及 SM90架构 GPU 增加 dual_gemm 调优，具体可参考`dual_gemm.py`。可选值：`tune`，开启调优；空值或`default`，使用默认配置；任意值，优先使用配置文件中的参数，若无则使用默认配置。此外，也支持 DeepSeek 系列模型 Block wise FP8 gemm 调优， 具体参考`csrc/tools/tune_fp8_gemm.sh`。

- `FLAGS_cuda_core_int8_gemm`：是否开启小 Batch Int8 Gemm 优化，默认值不开启。设为1可开启，推理 A8W8模型时，平均性能会加速约40%-55%，适用于 SM>=70的显卡。

- `FLAGS_cuda_core_fp8_gemm`：是否开启小 Batch FP8 Gemm 优化，默认值不开启。设为1可开启，推理 FP8模型时，平均性能会加速约30%左右，适用于 SM>=89的显卡。

- `FLAGS_blha_use_fp32_qk_sum`：在 fp16精度下，blha 计算 qkGemm 时是否用 fp32进行累加，默认为0，表示不开启，如果出现乱码可以开启防止精度溢出。

**GQA 优化**

- `FLAGS_use_xqa_optim`：gpa 是否开启 xqa 优化，默认值为0，表示不开启。gqa 模型（如 llama3/3.1、qwen2）设为1性能会更好。

**显存优化**

- `FLAGS_fraction_of_gpu_memory_to_use`：GPU 显存使用率，默认值为0.9。设置为0.9即可。

- `FLAGS_gemm_use_half_precision_compute_type`：是否使用半精度浮点数计算，默认值为0。设置为0即可。

**Append Attention 优化**

- `FLAGS_cascade_attention_max_partition_size`：Attention decoder 计算时对 cache_kv 进行分 chunk 的 chunk 大小，默认值根据 batch_size 设置，batch_size = 1 时设置为 128，batchsize > 1 时设置为 512。显式设置时不再区分 batch_size。
- `FLAGS_dec_block_shape_q`：Append Attention decoder 计算时对 q 进行分块的分块大小，默认值为16。
- `FLAGS_enc_block_shape_q`：Append Attention encoder 计算时对 q 进行分块的分块大小，默认值为64。

**MLA 相关优化**
- `FLAGS_mla_use_tensorcore`：MLA 计算时是否使用 tensor core 实现，默认为 True，仅支持 Hoppers 架构显卡。为 False 时则采用 cuda core 实现，同时支持 Ampper 和
 Hopper 架构。

**allreduce 优化**
- `FLAGS_custom_allreduce`：多卡推理计算时是否使用高性能自定义 allreduce 实现，默认为 False。

**自定义算子通用 优化**
- `DYNAMIC_INFERENCE_MODE`：动态图推理时自定义算子是否采用 pybind 调用方式，默认为 True。
