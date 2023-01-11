# FastDeploy ERNIE 3.0 Tiny模型高性能部署

**目录**
   * [FastDeploy部署介绍](#FastDeploy部署介绍)
   * [代码结构](#代码结构)
   * [环境要求](#环境要求)
   * [详细部署文档](#详细部署文档)

<a name="FastDeploy部署介绍"></a>

## FastDeploy部署介绍

**⚡️FastDeploy**是一款**全场景**、**易用灵活**、**极致高效**的AI推理部署工具，满足开发者**多硬件、多平台**的产业部署需求。开发者可以基于FastDeploy将训练好的预测模型在不同的硬件、不同的操作系统以及不同的推理引擎后端上进行部署。目前FastDeploy提供多种编程语言的SDK，包括C++、Python以及Java SDK。

目前ERNIE 3.0 Tiny模型已提供基于FastDeploy的云边端的部署示例，在服务端上的GPU硬件上，支持`PaddleInference`、`ONNX Runtime`、`Paddle TensorRT`以及`TensorRT`后端，在CPU上支持`PaddleInference`、`ONNX Runtime`以及`OpenVINO`后端；在移动端上支持`PaddleLite`后端。多硬件、多推理引擎后端的支持可以满足开发者不同的部署需求。

本部署示例是车载语音场景下的口语理解（Spoken Language Understanding，SLU）任务，详细可看[ERNIE 3.0 Tiny介绍](../README.md)。


<a name="代码结构"></a>

## 代码结构

```text

├── cpp
│   ├── CMakeLists.txt    # CMake编译脚本
│   ├── infer_demo.cc     # C++ 部署示例代码
│   └── README.md         # C++ 部署示例文档
├── python
│   ├── infer_demo.py     # Python 部署示例代码
│   └── README.md         # Python 部署示例文档
├── android
│   ├── README.md         # Android部署文档
│   ├── app               # App示例代码
│   ├── build.gradle
│   ├── ernie_tiny        # ERNIE 3.0 Tiny JNI & Java封装代码
│   ├── ......            # Android相关的工程文件及目录
│   ├── local.properties
│   └── ui                # 一些辅助用的UI代码
└── README.md             # 文档

```

<a name="环境要求"></a>

## 环境要求

在部署ERNIE 3.0 Tiny模型前，需要安装FastDeploy SDK，可参考[FastDeploy SDK安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)确认部署环境是否满足FastDeploy环境要求，并按照介绍安装相应的SDK。

<a name="详细部署文档"></a>

## 详细部署文档

- [Python部署](python/README.md)
- [C++部署](cpp/README.md)
- [Android部署](android/README.md)
