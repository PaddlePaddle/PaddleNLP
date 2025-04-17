# FastDeploy ERNIE 3.0 Tiny 模型高性能部署

**目录**
   * [FastDeploy 部署介绍](#FastDeploy 部署介绍)
   * [代码结构](#代码结构)
   * [环境要求](#环境要求)
   * [详细部署文档](#详细部署文档)

<a name="FastDeploy 部署介绍"></a>

## FastDeploy 部署介绍

**⚡️FastDeploy**是一款**全场景**、**易用灵活**、**极致高效**的 AI 推理部署工具，满足开发者**多硬件、多平台**的产业部署需求。开发者可以基于 FastDeploy 将训练好的预测模型在不同的硬件、不同的操作系统以及不同的推理引擎后端上进行部署。目前 FastDeploy 提供多种编程语言的 SDK，包括 C++、Python 以及 Java SDK。

目前 ERNIE 3.0 Tiny 模型已提供基于 FastDeploy 的云边端的部署示例，在服务端上的 GPU 硬件上，支持`Paddle Inference`、`ONNX Runtime`、`Paddle TensorRT`以及`TensorRT`后端，在 CPU 上支持`Paddle Inference`、`ONNX Runtime`以及`OpenVINO`后端；在移动端上支持`Paddle Lite`后端。多硬件、多推理引擎后端的支持可以满足开发者不同的部署需求。

本部署示例是车载语音场景下的口语理解（Spoken Language Understanding，SLU）任务，详细可看[ERNIE 3.0 Tiny 介绍](../README.md)。


<a name="代码结构"></a>

## 代码结构

```text

├── python
│   ├── infer_demo.py     # Python 部署示例代码
│   └── README.md         # Python 部署示例文档
└── README.md             # 文档

```

<a name="环境要求"></a>

## 环境要求

在部署 ERNIE 3.0 Tiny 模型前，需要安装 FastDeploy SDK，可参考[FastDeploy SDK 安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)确认部署环境是否满足 FastDeploy 环境要求，并按照介绍安装相应的 SDK。

<a name="详细部署文档"></a>

## 详细部署文档

- [Python 部署](python/README.md)
