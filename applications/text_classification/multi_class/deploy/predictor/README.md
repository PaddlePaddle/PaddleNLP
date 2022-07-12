# 基于ONNXRuntime推理部署指南

**目录**
   * [环境准备](#环境准备)
   * [基于GPU部署推理样例](#基于GPU部署推理样例)
   * [基于CPU部署推理样例](#基于CPU部署推理样例)
   * [性能与精度测试](#性能与精度测试)
       * [GPU精度与性能](#GPU精度与性能)
       * [CPU精度与性能](#CPU精度与性能)
## 环境准备

模型转换与ONNXRuntime预测部署依赖Paddle2ONNX和ONNXRuntime，Paddle2ONNX支持将Paddle静态图模型转化为ONNX模型格式，算子目前稳定支持导出ONNX Opset 7~15，更多细节可参考：[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)。如何使用[静态图导出脚本](../../export_model.py)将训练后的模型转为静态图模型详见[模型静态图导出](../../README.md)，模型使用裁剪API进行裁剪之后会自动生成静态图模型。

如果基于GPU部署，请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保CUDA >= 11.2，CuDNN >= 8.2，并使用以下命令安装所需依赖:
```
python -m pip install onnxruntime-gpu onnx onnxconverter-common
```

如果基于CPU部署，请使用如下命令安装所需依赖:
```
python -m pip install onnxruntime
```


## 基于GPU部署推理样例

请使用如下命令进行部署
```
python infer.py --model_path_prefix ../../export/float32
```

可支持配置的参数：

* `model_path_prefix`：必须，待推理模型路径前缀。
* `model_name_or_path`：选择预训练模型；默认为"ernie-3.0-base-zh"。
* `dataset`：训练数据集;默认为cblue数据集。
* `task_name`：训练数据集任务;默认为KUAKE-QIC数据集。
* `max_seq_length`：ERNIE/BERT模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `use_fp16`：选择是否开启FP16进行加速；默认为False。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为200。
* `device`: 选用什么设备进行训练，可选cpu、gpu。
* `device_id`: 选择GPU卡号；默认为0。
* `perf`：选择进行模型性能和精度评估；默认为False。
* `dataset_dir`：本地数据集地址，需包含data.txt, label.txt, test.txt/dev.txt(可选，如果启动模型性能和精度评估)；默认为None。
* `perf_dataset`：评估数据集，选择在开发集或测试集评估模型；默认为"test"。

在GPU设备的CUDA计算能力 (CUDA Compute Capability) 大于7.0，在包括V100、T4、A10、A100、GTX 20系列和30系列显卡等设备上可以开启FP16进行加速，在CPU或者CUDA计算能力 (CUDA Compute Capability) 小于7.0时开启不会带来加速效果。可以使用如下命令开启ONNXRuntime的FP16进行推理加速：

```
python infer.py --model_path_prefix ../../export/float32 --use_fp16
```

可以使用如下命令开启ONNXRuntime推理评估模型的性能和精度：

```
python infer.py --model_path_prefix ../../export/float32 --perf --perf_dataset dev
```

如需使用本地数据集,请指定本地数据集路径参数 `dataset_dir` ，更多本地数据集使用方式详见[以内置数据集格式读取本地数据集](../../README.md)

## 基于CPU部署推理样例

请使用如下命令进行部署
```
python infer.py --device cpu --model_path_prefix ../../export/float32
```

可支持配置的参数：

* `model_path_prefix`：必须，待推理模型路径前缀。
* `model_name_or_path`：选择预训练模型；默认为"ernie-3.0-base-zh"，中文数据集推荐使用"ernie-3.0-base-zh"。
* `dataset`：训练数据集;默认为cblue数据集。
* `task_name`：训练数据集任务;默认为KUAKE-QIC数据集。
* `max_seq_length`：ERNIE/BERT模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `use_quantize`：选择是否开启INT8动态量化进行加速；默认为False。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为200。
* `num_threads`：cpu线程数；默认为cpu的物理核心数量。
* `device`: 选用什么设备进行训练，可选cpu、gpu。
* `perf`：选择进行模型性能和精度评估；默认为False。
* `dataset_dir`：本地数据集地址，需包含data.txt, label.txt, dev.txt/test.txt(可选，如果启动模型性能和精度评估)；默认为None。
* `perf_dataset`：评估数据集，选择在开发集或测试集评估模型；默认为"test"。

可以使用如下命令开启ONNXRuntime的INT8动态量化进行推理加速：

```
python infer.py --device cpu --model_path_prefix ../../export/float32 --use_quantize
```

**Note**：INT8动态量化与FP16相比精度损失较大，GPU部署建议使用FP16加速。

可以使用如下命令开启ONNXRuntime推理评估模型的性能和精度：

```
python infer.py --device cpu --model_path_prefix ../../export/float32 --perf --perf_dataset dev
```


如需使用本地数据集,请指定本地数据集路径参数 `dataset_dir` ，更多本地数据集使用方式详见[以内置数据集格式读取本地数据集](../../README.md)

## 性能与精度测试


测试配置如下：

1. CBLUE数据集中医疗搜索检索词意图分类(KUAKE-QIC)任务开发集

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

5. 性能数据指标：latency。latency 测试方法：固定 batch size 为 200，GPU部署运行时间 total_time，计算 latency = total_time / total_samples

6. 精度评价指标：Accuracy

### GPU精度与性能

|                            | Accuracy(%) | latency(ms) |
| -------------------------- | ------------ | ------------- |
| ERNIE 3.0+FP32              | 81.43 | 1.95  |
| ERNIE 3.0+FP16             | 81.48(+0.05) | 0.47   |
| ERNIE 3.0+裁剪(保留比例3/4)+FP32     | 82.53(+1.10) | 1.55   |
| ERNIE 3.0+裁剪(保留比例3/4)+FP16    | 82.53(+1.10) | 0.38  |
| ERNIE 3.0+裁剪(保留比例2/3)+FP32     | 82.20(+0.77) | 1.34  |
| ERNIE 3.0+裁剪(保留比例2/3)+FP16    | 82.20(+0.77) | 0.34   |
| ERNIE 3.0+裁剪(保留比例1/2)+FP32     | 81.13(-0.30) | 1.06  |
| ERNIE 3.0+裁剪(保留比例1/2)+FP16    | 81.07(-0.36) | 0.28   |


经过FP16转化加速比达到3~4倍左右，精度变化在0.06内

### CPU精度与性能

测试环境及说明如上，测试 CPU 性能时，线程数设置为40。

|                            | Accuracy(%) | latency(ms) |
| -------------------------- | ------------ | ------------- |
| ERNIE 3.0+FP32             | 81.43  | 96.50   |
| ERNIE 3.0+INT8             | 79.54(-1.91) | 68.31    |
| ERNIE 3.0+裁剪(保留比例3/4)+FP32    | 82.53(+1.10) | 70.49    |
| ERNIE 3.0+裁剪(保留比例3/4)+INT8    | 81.18(-0.25) | 55.31   |
| ERNIE 3.0+裁剪(保留比例2/3)+FP32    | 82.20(+0.77) | 60.53  |
| ERNIE 3.0+裁剪(保留比例2/3)+INT8    | 81.13(-0.30) | 41.56    |
| ERNIE 3.0+裁剪(保留比例1/2)+FP32    | 81.13(-0.30) | 57.08    |
| ERNIE 3.0+裁剪(保留比例1/2)+INT8    | 80.72(-0.71) | 27.24   |

与FP16相比，INT8在线量化精度下降较大，加速比在1.5倍左右
