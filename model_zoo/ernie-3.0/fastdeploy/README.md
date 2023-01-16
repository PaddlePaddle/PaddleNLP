# FastDeploy ERNIE 3.0 模型高性能部署

**⚡️FastDeploy**是一款**全场景**、**易用灵活**、**极致高效**的 AI 推理部署工具，满足开发者**多硬件、多平台**的产业部署需求。开发者可以基于 FastDeploy 将训练好的预测模型在不同的硬件、不同的推理引擎后端上进行部署。目前 FastDeploy 提供多种编程语言的 SDK，包括 C++、Python 以及 Java SDK。

在部署 ERNIE 3.0 模型前，需要安装 FastDeploy SDK，可参考 [FastDeploy SDK安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)确认部署环境是否满足 FastDeploy 环境要求，并按照介绍安装相应的 SDK。

## 模型详细说明
- [ERNIE 3.0模型说明](../README.md)

## 支持的模型列表

| 模型 |  结构  | 语言 |
| :---: | :--------: | :--------: |
| **ERNIE 3.0-_Base_** | 12-layers, 768-hidden, 12-heads | 中文 |
| **ERNIE 3.0-_Medium_** | 6-layers, 768-hidden, 12-heads | 中文 |
| **ERNIE 3.0-_Mini_** | 6-layers, 384-hidden, 12-heads | 中文 |
| **ERNIE 3.0-_Micro_** | 4-layers, 384-hidden, 12-heads | 中文 |
| **ERNIE 3.0-_Nano_** | 4-layers, 312-hidden, 12-heads | 中文 |

## 支持的NLP任务列表

| 任务 Task  |  是否支持   |
| :--------------- | ------- |
| 文本分类 | ✅ |
| 序列标注 | ✅  |

## 详细部署文档

- [Python部署](python/README.md)
- [C++部署](cpp/README.md)
