# FastDeploy ERNIE 3.0 模型高性能部署

**⚡️FastDeploy**是一款**全场景**、**易用灵活**、**极致高效**的AI推理部署工具，满足开发者**多硬件、多平台**的产业部署需求。开发者可以基于FastDeploy将训练好的预测模型在不同的硬件、不同的推理引擎后端上进行部署。目前FastDeploy提供多种编程语言的SDK，包括C++、Python以及Java SDK。

在部署ERNIE模型前，需要安装FastDeploy SDK，可参考[FastDeploy SDK安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)确认部署环境是否满足FastDeploy环境要求，并按照介绍安装相应的SDK。

## 模型详细说明
- [ERNIE 3.0模型说明](../README.md)

## 支持的模型列表

| 模型 |  结构  | 语言 |
| :---: | :--------: | :--------: |
| `ERNIE 3.0-Base`| 12-layers, 768-hidden, 12-heads | 中文 |
| `ERNIE 3.0-Medium`| 6-layers, 768-hidden, 12-heads | 中文 |
| `ERNIE 3.0-Mini`| 6-layers, 384-hidden, 12-heads | 中文 |
| `ERNIE 3.0-Micro`| 4-layers, 384-hidden, 12-heads | 中文 |
| `ERNIE 3.0-Nano `| 4-layers, 312-hidden, 12-heads | 中文 |

## 支持的NLP任务列表

| 任务 Task  |  是否支持   |
| :--------------- | ------- |
| 文本分类 | ✅ |
| 序列标注 | ✅  |

## 导出部署模型

在部署前，需要先将训练好的ERNIE模型导出成部署模型，导出步骤可参考文档[导出模型](../README.md#模型导出).

## 下载微调模型

为了方便开发者的测试，下面分别提供了在文本分类[TNEWS数据集](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)以及序列标注任务[MSRA_NER](https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra)上微调的ERNIE 3.0-Medium模型，开发者可直接下载体验。

- [ERNIE 3.0 Medium TNEWS](https://bj.bcebos.com/fastdeploy/models/ernie-3.0/ernie-3.0-medium-zh-tnews.tgz)
- [ERNIE 3.0 Medium MSRA_NER](https://bj.bcebos.com/fastdeploy/models/ernie-3.0/ernie-3.0-medium-zh-msra.tgz)

## 详细部署文档

- [Python部署](python/README.md)
- [C++部署](cpp/README.md)
