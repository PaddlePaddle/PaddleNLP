# 基于 ONNXRuntime 推理部署指南

**目录**
   * [环境准备](#环境准备)
   * [基于 GPU 部署推理样例](#基于 GPU 部署推理样例)
   * [基于 CPU 部署推理样例](#基于 CPU 部署推理样例)
   * [性能与精度测试](#性能与精度测试)
## 环境准备

模型转换与 ONNXRuntime 预测部署依赖 Paddle2ONNX 和 ONNXRuntime，Paddle2ONNX 支持将 Paddle 静态图模型转化为 ONNX 模型格式，算子目前稳定支持导出 ONNX Opset 7~15，更多细节可参考：[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)。如何使用[静态图导出脚本](../../export_model.py)将训练后的模型转为静态图模型详见[静态图导出](../../README.md)，模型使用裁剪 API 进行裁剪之后会自动生成静态图模型。

如果基于 GPU 部署，请先确保机器已正确安装 NVIDIA 相关驱动和基础软件，确保 CUDA >= 11.2，CuDNN >= 8.2，并使用以下命令安装所需依赖:
```shell
python -m pip install onnxruntime-gpu onnx onnxconverter-common==1.9.0 psutil paddle2onnx==1.0.5
```

如果基于 CPU 部署，请使用如下命令安装所需依赖:
```shell
python -m pip install onnxruntime psutil
```

## 基于 GPU 部署推理样例
请使用如下命令进行部署
```
python infer.py \
    --device "gpu" \
    --model_path_prefix "../../export/float32" \
    --model_name_or_path "ernie-3.0-medium-zh" \
    --max_seq_length 128 \
    --batch_size 32 \
    --dataset_dir "../../data"
```
多语言模型加上`--multilingual`,裁剪后的模型前缀为`--model_path_prefix ../../prune/width_mult_XXXX/pruned_model`。
可支持配置的参数：

* `model_path_prefix`：必须，待推理模型路径前缀。
* `model_name_or_path`：选择预训练模型,可选"ernie-1.0-large-zh-cw","ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en","ernie-m-base","ernie-m-large"；默认为"ernie-3.0-medium-zh",根据实际使用的预训练模型选择。
* `max_seq_length`：ERNIE/BERT 模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `use_fp16`：选择是否开启 FP16进行加速；默认为 False。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `device`: 选用什么设备进行训练，可选 cpu、gpu。
* `device_id`: 选择 GPU 卡号；默认为0。
* `perf`：选择进行模型性能和精度评估；默认为 False。
* `dataset_dir`：本地数据集地址，需包含 data.txt, label.txt, test.txt/dev.txt(可选，如果启动模型性能和精度评估)；默认为 None。
* `perf_dataset`：评估数据集，可选'dev'、'test'，选择在开发集或测试集评估模型；默认为"dev"。
型）；默认为 False。

在 GPU 设备的 CUDA 计算能力 (CUDA Compute Capability) 大于7.0，在包括 V100、T4、A10、A100、GTX 20系列和30系列显卡等设备上可以开启 FP16进行加速，在 CPU 或者 CUDA 计算能力 (CUDA Compute Capability) 小于7.0时开启不会带来加速效果。可以使用如下命令开启 ONNXRuntime 的 FP16进行推理加速：

```
python infer.py \
    --use_fp16 \
    --device "gpu" \
    --model_path_prefix "../../export/float32" \
    --model_name_or_path "ernie-3.0-medium-zh" \
    --max_seq_length 128 \
    --batch_size 32 \
    --dataset_dir "../../data"
```

可以使用如下命令开启 ONNXRuntime 推理评估模型的性能和精度：

```
python infer.py \
    --perf \
    --perf_dataset 'dev' \
    --device "gpu" \
    --model_path_prefix "../../export/float32" \
    --model_name_or_path "ernie-3.0-medium-zh" \
    --max_seq_length 128 \
    --batch_size 32 \
    --dataset_dir "../../data"
```

## 基于 CPU 部署推理样例

请使用如下命令进行部署
```
python infer.py \
    --device "cpu" \
    --model_path_prefix "../../export/float32" \
    --model_name_or_path "ernie-3.0-medium-zh" \
    --max_seq_length 128 \
    --batch_size 32 \
    --dataset_dir "../../data"
```

可支持配置的参数：

* `model_path_prefix`：必须，待推理模型路径前缀。
* `model_name_or_path`：选择预训练模型；默认为"ernie-3.0-medium-zh"，中文数据集推荐使用"ernie-3.0-medium-zh"。
* `max_seq_length`：ERNIE/BERT 模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `use_quantize`：选择是否开启 INT8动态量化进行加速；默认为 False。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为200。
* `num_threads`：cpu 线程数；默认为 cpu 的物理核心数量。
* `device`: 选用什么设备进行训练，可选 cpu、gpu。
* `perf`：选择进行模型性能和精度评估；默认为 False。
* `dataset_dir`：本地数据集地址，需包含 data.txt, label.txt, dev.txt/test.txt(可选，如果启动模型性能和精度评估)；默认为 None。
* `perf_dataset`：评估数据集，选择在开发集或测试集评估模型；默认为"dev"。

可以使用如下命令开启 ONNXRuntime 的 INT8动态量化进行推理加速：

```
python infer.py \
    --use_quantize \
    --device "cpu" \
    --model_path_prefix "../../export/float32" \
    --model_name_or_path "ernie-3.0-medium-zh" \
    --max_seq_length 128 \
    --batch_size 32 \
    --dataset_dir "../../data"
```

**Note**：INT8动态量化与 FP16相比精度损失较大，GPU 部署建议使用 FP16加速。

可以使用如下命令开启 ONNXRuntime 推理评估模型的性能和精度：

```
python infer.py \
    --perf \
    --perf_dataset 'dev' \
    --device "cpu" \
    --model_path_prefix "../../export/float32" \
    --model_name_or_path "ernie-3.0-medium-zh" \
    --max_seq_length 128 \
    --batch_size 32 \
    --dataset_dir "../../data"
```

## 性能与精度测试


测试配置如下：

1. [2020语言与智能技术竞赛：事件抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/32/0/introduction)抽取的多标签数据集

2. 物理机环境

    系统: CentOS Linux release 7.7.1908 (Core)

    GPU: Tesla V100-SXM2-32GB

    CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz

    CUDA: 11.2

    cuDNN: 8.1.0

    Driver Version: 460.27.04

    内存: 630 GB

3. PaddlePaddle 版本：2.3.0

4. PaddleNLP 版本：2.3.1

5. 性能数据指标：latency。latency 测试方法：固定 batch size 为 32，GPU 部署运行时间 total_time，计算 latency = total_time / total_samples

6. 精度评价指标：Micro F1分数、Macro F1分数

|                            | Micro F1(%)   | Macro F1(%) | latency(ms) |
| -------------------------- | ------------ | ------------- |------------- |
| ERNIE 3.0 Medium+FP32+GPU  | 95.26|93.22| 1.01|
| ERNIE 3.0 Medium+FP16+GPU  | 95.26|93.22| 0.38|
| ERNIE 3.0 Medium+FP32+CPU  | 95.26|93.22|  18.93 |
| ERNIE 3.0 Medium+INT8+CPU  | 95.03 | 92.87| 12.14  |


经过 FP16转化加速比达到3~4倍左右，精度变化较小，与 FP16相比,INT8在线量化精度下降较大，加速比在1.5~2倍左右。
