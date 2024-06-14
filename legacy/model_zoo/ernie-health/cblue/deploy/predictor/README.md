# 基于ONNXRuntime推理部署指南

本示例以[CBLUE数据集微调](../../README.md)得到的ERNIE-Health模型为例，分别提供了文本分类任务、实体识别任务和关系抽取任务的部署代码，自定义数据集可参考实现。
在推理部署前需将微调后的动态图模型转换导出为静态图，详细步骤见[静态图模型导出](../../README.md)。

**目录**
   * [环境安装](#环境安装)
   * [GPU部署推理样例](#gpu部署推理样例)
   * [CPU部署推理样例](#cpu部署推理样例)
   * [性能与精度测试](#性能与精度测试)
       * [GPU精度与性能](#gpu精度与性能)
       * [CPU精度与性能](#cpu精度与性能)

## 环境安装

ONNX模型转换和推理部署依赖于Paddle2ONNX和ONNXRuntime。其中Paddle2ONNX支持将Paddle静态图模型转化为ONNX模型格式，算子目前稳定支持导出ONNX Opset 7~15，更多细节可参考：[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)。

#### GPU端
请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保CUDA >= 11.2，CuDNN >= 8.2，并使用以下命令安装所需依赖:
```
python -m pip install -r requirements_gpu.tx
```
\* 如需使用半精度（FP16）部署，请确保GPU设备的CUDA计算能力 (CUDA Compute Capability) 大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。 更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)
#### CPU端
请使用如下命令安装所需依赖:
```
python -m pip install -r requirements_cpu.txt
```
## GPU部署推理样例

请使用如下命令进行GPU上的部署，可用`use_fp16`开启**半精度部署推理加速**，可用`device_id`**指定GPU卡号**。

- 文本分类任务

```
python infer_classification.py --device gpu --device_id 0 --dataset KUAKE-QIC --model_path_prefix ../../export/inference
```

- 实体识别任务

```
python infer_ner.py --device gpu --device_id 0 --dataset CMeEE --model_path_prefix ../../export/inference
```

- 关系抽取任务

```
python infer_spo.py --device gpu --device_id 0 --dataset CMeIE --model_path_prefix ../../export/inference
```

可支持配置的参数：

* `model_path_prefix`：必须，待推理模型路径前缀。
* `model_name_or_path`：选择预训练模型；默认为"ernie-health-chinese"。
* `dataset`：CBLUE中的训练数据集。
   * `文本分类任务`：包括KUAKE-QIC, KUAKE-QQR, KUAKE-QTR, CHIP-CTC, CHIP-STS, CHIP-CDN-2C；默认为KUAKE-QIC。
   * `实体抽取任务`：默认为CMeEE。
   * `关系抽取任务`：默认为CMeIE。
* `max_seq_length`：模型使用的最大序列长度，最大不能超过512；`关系抽取任务`默认为300，其余默认为128。
* `use_fp16`：选择是否开启FP16进行加速，仅在`devive=gpu`时生效；默认关闭。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为200。
* `device`: 选用什么设备进行训练，可选cpu、gpu；默认为gpu。
* `device_id`: 选择GPU卡号；默认为0。
* `data_file`：本地待预测数据文件；默认为None。

#### 本地数据集加载
如需使用本地数据集，请指定本地待预测数据文件 `data_file`，每行一条样例，单文本输入每句一行，双文本输入以`\t`分隔符隔开。例如

**ctc-data.txt**
```
在过去的6个月曾服用偏头痛预防性药物或长期服用镇痛药物者，以及有酒精依赖或药物滥用习惯者；
患有严重的冠心病、脑卒中，以及传染性疾病、精神疾病者；
活动性乙肝（包括大三阳或小三阳）或血清学指标（HBsAg或/和HBeAg或/和HBcAb）阳性者，丙肝、肺结核、巨细胞病毒、严重真菌感染或HIV感染；
...
```

**sts-data.txt**
```
糖尿病能吃减肥药吗？能治愈吗？\t糖尿病为什么不能吃减肥药？
为什么慢性乙肝会急性发作\t引起隐匿性慢性乙肝的原因是什么
标准血压是多少高血压指？低血压又指？\t半月前检查血压100／130，正常吗？
...
```

## CPU部署推理样例

请使用如下命令进行CPU上的部署，可用`num_threads`**调整预测线程数量**。

- 文本分类任务

```
python infer_classification.py --device cpu --dataset KUAKE-QIC --model_path_prefix ../../export/inference
```

- 实体识别任务

```
python infer_ner.py --device cpu --dataset CMeEE --model_path_prefix ../../export/inference
```

- 关系抽取任务

```
python infer_spo.py --device cpu --dataset CMeIE --model_path_prefix ../../export/inference
```

可支持配置的参数：

* `model_path_prefix`：必须，待推理模型路径前缀。
* `model_name_or_path`：选择预训练模型；默认为"ernie-health-chinese"。
* `dataset`：CBLUE中的训练数据集。
   * `文本分类任务`：包括KUAKE-QIC, KUAKE-QQR, KUAKE-QTR, CHIP-CTC, CHIP-STS, CHIP-CDN-2C；默认为KUAKE-QIC。
   * `实体抽取任务`：默认为CMeEE。
   * `关系抽取任务`：默认为CMeIE。
* `max_seq_length`：模型使用的最大序列长度，最大不能超过512；`关系抽取任务`默认为300，其余默认为128。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为200。
* `device`: 选用什么设备进行训练，可选cpu、gpu；默认为gpu。
* `num_threads`：cpu线程数，在`device=gpu`时影响较小；默认为cpu的物理核心数量。
* `data_file`：本地待预测数据文件，格式见[GPU部署推理样例](#本地数据集加载)中的介绍；默认为None。

## 性能与精度测试

本节提供了在CBLUE数据集上预测的性能和精度数据，以供参考。

测试配置如下：

1. 数据集

    使用CBLUE数据集中的开发集用于ERNIE-Health微调模型部署推理的性能与精度测试，包括

  - 医疗搜索检索词意图分类（KUAKE-QIC）任务
  - 医疗搜索查询词-页面标题相关性（KUAKE-QTR）任务
  - 医疗搜索查询词-查询词相关性（KUAKE-QQR）任务
  - 临床试验筛选标准短文本分类(CHIP-CTC)任务
  - 平安医疗科技疾病问答迁移学习（CHIP-STS）任务
  - 临床术语标准化匹配（CHIP-CDN-2C）任务
  - 中文医学命名实体识别（CMeEE）任务
  - 中文医学文本实体关系抽取（CMeIE）任务

2. 物理机环境

    系统: CentOS Linux release 7.7.1908 (Core)

    GPU: Tesla V100-SXM2-32GB

    CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz

    CUDA: 11.2

    cuDNN: 8.1.0

    Driver Version: 460.27.04

    内存: 630 GB

3. PaddlePaddle 版本：2.3.0

4. PaddleNLP 版本：2.3.4

5. 性能数据指标：latency。latency 测试方法：固定 batch size 为 200（CHIP-CDN-2C 和 CMeIE 数据集为 20），部署运行时间 total_time，计算 latency = total_time / total_samples


### GPU精度与性能

| 数据集       | 最大文本长度 | 精度评估指标 | FP32 指标值 | FP16 指标值 | FP32 latency(ms) | FP16 latency(ms) |
| ----------  | ---------- | ---------- | ---------- | ---------- | ---------------- | ---------------- |
| KUAKE-QIC   | 128        | Accuracy   | 0.8046     | 0.8046     | 1.92             | 0.46             |
| KUAKE-QTR   | 64         | Accuracy   | 0.6886     | 0.6876 (-) | 0.92             | 0.23             |
| KUAKE-QQR   | 64         | Accuracy   | 0.7755     | 0.7755     | 0.61             | 0.16             |
| CHIP-CTC    | 160        | Macro F1   | 0.8445     | 0.8446 (+) | 2.34             | 0.60             |
| CHIP-STS    | 96         | Macro F1   | 0.8892     | 0.8892     | 1.39             | 0.35             |
| CHIP-CDN-2C | 256        | Macro F1   | 0.8921     | 0.8920 (-) | 1.58             | 0.48             |
| CMeEE       | 128        | Micro F1   | 0.6469     | 0.6468 (-) | 1.90             | 0.48             |
| CMeIE       | 300        | Micro F1   | 0.5903     | 0.5902 (-) | 50.32            | 41.50            |

经过FP16转化加速比达到 1.2 ~ 4 倍左右，精度变化在 1e-4 ~ 1e-3 内。

### CPU精度与性能

测试环境及说明如上，测试 CPU 性能时，线程数设置为40。

| 数据集      | 最大文本长度 | 精度评估指标 | FP32 指标值 | FP32 latency(ms) |
| ----------  | ------------ | ------------ | ---------- | ---------------- |
| KUAKE-QIC   | 128          | Accuracy     | 0.8046     | 37.72            |
| KUAKE-QTR   | 64           | Accuracy     | 0.6886     | 18.40            |
| KUAKE-QQR   | 64           | Accuracy     | 0.7755     | 10.34            |
| CHIP-CTC    | 160          | Macro F1     | 0.8445     | 47.43            |
| CHIP-STS    | 96           | Macro F1     | 0.8892     | 27.67            |
| CHIP-CDN-2C | 256          | Micro F1     | 0.8921     | 26.86            |
| CMeEE       | 128          | Micro F1     | 0.6469     | 37.59            |
| CMeIE       | 300          | Micro F1     | 0.5902     | 213.04           |
