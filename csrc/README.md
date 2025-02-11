# PaddleNLP 大模型高性能自定义推理算子

此文档介绍如何编译安装 PaddleNLP 大模型高性能自定义推理算子的安装教程。

使用这些高性能算子，可以大幅提升大模型推理速度。
大模型推理相关教程详见[此处](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/README.md#6-%E6%8E%A8%E7%90%86)。

## 安装 C++ 依赖

```shell
pip install -r requirements.txt
```

## 编译 Cuda 算子

```shell
python setup_cuda.py install
```

### FP8 GEMM 自动调优

确保 `cutlass` 库已经安装，然后执行以下命令进行自动调优。
- 对于89架构的 GPU，CUDA 版本至少为12.4
- 对于90架构的 GPU，CUDA 版本至少为12.0
```shell
sh tune_fp8_gemm.sh
```
