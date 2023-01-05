# 离线推理部署指南

**目录**
   * [环境准备](#环境准备)
   * [基于GPU部署推理样例](#基于GPU部署推理样例)
   * [基于CPU部署推理样例](#基于CPU部署推理样例)
   * [性能与精度测试](#性能与精度测试)

## 环境准备

模型转换与ONNXRuntime预测部署依赖Paddle2ONNX和ONNXRuntime，Paddle2ONNX支持将Paddle静态图模型转化为ONNX模型格式，算子目前稳定支持导出ONNX Opset 7~15，更多细节可参考：[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)。

如果基于GPU部署，请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保CUDA >= 11.2，CuDNN >= 8.2，并使用以下命令安装所需依赖:
```shell
python -m pip install onnxruntime-gpu onnx onnxconverter-common==1.9.0 paddle2onnx==1.0.5
```

如果基于CPU部署，请使用如下命令安装所需依赖:
```shell
python -m pip install onnxruntime
```

## 基于GPU部署推理样例

请使用如下命令进行部署
```
python infer.py \
    --device "gpu" \
    --model_path_prefix "../../checkpoint/model_best/" \
    --model_name_or_path "utc-large" \
    --max_length 128 \
    --batch_size 32 \
    --data_dir "../../data"
```

可支持配置的参数：

* `model_path_prefix`：必须，待推理模型路径前缀。
* `model_name_or_path`：选择预训练模型,可选`utc-large`。
* `max_length`：模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `use_fp16`：选择是否开启FP16进行加速；默认为False。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `device`: 选用什么设备进行训练，可选cpu、gpu。
* `device_id`: 选择GPU卡号；默认为0。
* `data_dir`：本地数据集地址，需包含data.txt, label.txt, test.txt/dev.txt(可选，如果启动模型性能和精度评估)；默认为None。

在GPU设备的CUDA计算能力 (CUDA Compute Capability) 大于7.0，在包括V100、T4、A10、A100、GTX 20系列和30系列显卡等设备上可以开启FP16进行加速，在CPU或者CUDA计算能力 (CUDA Compute Capability) 小于7.0时开启不会带来加速效果。可以使用如下命令开启ONNXRuntime的FP16进行推理加速：

```
python infer.py \
    --use_fp16 \
    --device "gpu" \
    --model_path_prefix "../../checkpoint/model_best/" \
    --model_name_or_path "utc-large" \
    --max_length 128 \
    --batch_size 32 \
    --data_dir "../../data"

## 基于CPU部署推理样例

请使用如下命令进行部署
```
python infer.py \
    --device "cpu" \
    --model_path_prefix "../../checkpoint/model_best/" \
    --model_name_or_path "utc-large" \
    --max_length 128 \
    --batch_size 32 \
    --data_dir "../../data"

* `model_path_prefix`：必须，待推理模型路径前缀。
* `model_name_or_path`：选择预训练模型,可选`utc-large`。
* `max_length`：模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `device`: 选用什么设备进行训练，可选cpu、gpu。
* `device_id`: 选择GPU卡号；默认为0。
* `data_dir`：本地数据集地址，需包含data.txt, label.txt, test.txt/dev.txt(可选，如果启动模型性能和精度评估)；默认为None。

## 精度测试


测试配置如下：

1.

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

6. 精度评价指标：Micro F1分数、Macro F1分数

|               | Micro F1(%)  | Macro F1(%) |
| ------------- | ------------ | ------------- |
| UTC+FP32+GPU  | 90.57        |79.36|
| UTC+FP16+GPU  | 90.57        | 79.36|
| UTC+FP32+CPU  | 90.57        |79.36|
