# 基于PaddleNLP SimpleServing 的服务化部署

## 目录
- [环境准备](#环境准备)
- [Server启动服务](#Server服务启动)
- [Client请求启动](#Client请求启动)
- [服务化自定义参数](#服务化自定义参数)

## 环境准备

使用有SimpleServing功能的PaddleNLP版本(或者最新的develop版本)

```shell
pip install paddlenlp >= 2.5.0
```

## Server服务启动

```bash
paddlenlp server server:app --workers 1 --host 0.0.0.0 --port 8190
```

## Client请求启动

```bash
python client.py
```

## 服务化自定义参数

### Server 自定义参数

#### schema替换

```python
# Default schema
schema = ["病情诊断", "治疗方案", "病因分析", "指标解读", "就医建议", "疾病表述", "后果表述", "注意事项", "功效作用", "医疗费用", "其他"]
```

#### 设置模型路径

```python
# Default task_path
utc = Taskflow("zero_shot_text_classification", model="utc-base", task_path="../../checkpoint/model_best/plm", schema=schema)
```

#### 多卡服务化预测
PaddleNLP SimpleServing 支持多卡负载均衡预测，主要在服务化注册的时候，注册两个Taskflow的task即可，下面是示例代码

```python
utc1 = Taskflow("zero_shot_text_classification", model="utc-base", task_path="../../checkpoint/model_best/plm", schema=schema)
utc2 = Taskflow("zero_shot_text_classification", model="utc-base", task_path="../../checkpoint/model_best/plm", schema=schema)
service.register_taskflow("taskflow/utc", [utc1, utc2])
```

#### 更多配置

```python
>>> from paddlenlp import Taskflow
>>> schema = ["病情诊断", "治疗方案", "病因分析", "指标解读", "就医建议", "疾病表述", "后果表述", "注意事项", "功效作用", "医疗费用", "其他"]
>>> utc = Taskflow("zero_shot_text_classification",
                   schema=schema,
                   model="utc-base",
                   max_seq_len=512,
                   batch_size=1,
                   pred_threshold=0.5,
                   precision="fp32")
```

* `schema`：定义任务标签候选集合。
* `model`：选择任务使用的模型，默认为`utc-base`, 可选有`utc-xbase`, `utc-base`, `utc-medium`, `utc-micro`, `utc-mini`, `utc-nano`, `utc-pico`。
* `max_seq_len`：最长输入长度，包括所有标签的长度，默认为512。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `pred_threshold`：模型对标签预测的概率在0～1之间，返回结果去掉小于这个阈值的结果，默认为0.5。
* `precision`：选择模型精度，默认为`fp32`，可选有`fp16`和`fp32`。`fp16`推理速度更快。如果选择`fp16`，请先确保机器正确安装NVIDIA相关驱动和基础软件，**确保CUDA>=11.2，cuDNN>=8.1.1**，初次使用需按照提示安装相关依赖。其次，需要确保GPU设备的CUDA计算能力（CUDA Compute Capability）大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)。

### Client 自定义参数

```python
# Changed to input texts you wanted
texts = ["中性粒细胞比率偏低"]
```
