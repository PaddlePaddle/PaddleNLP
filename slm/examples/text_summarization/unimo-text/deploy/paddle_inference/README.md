# Paddle Inference 部署
本文档将介绍如何使用[Paddle Inference](https://paddle-inference.readthedocs.io/en/latest/guides/introduction/index_intro.html#paddle-inference)工具进行自动文本摘要应用高性能推理推理部署。

**目录**
   * [背景介绍](#背景介绍)
   * [导出预测部署模型](#导出预测部署模型)
   * [基于 Python 预测](#基于 Python 预测)


## 背景介绍
Paddle inference 和主框架的 Model.predict 均可实现推理预测，Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力，主框架的 Model 对象是一个具备训练、测试、推理的神经网络。相比于 Model.predict，inference 可使用 MKLDNN、CUDNN、TensorRT 进行预测加速。Model.predict 适用于训练好的模型直接进行预测，paddle inference 适用于对推理性能、通用性有要求的用户，针对不同平台不同的应用场景进行了深度的适配优化，保证模型在服务器端即训即用，快速部署。由于 Paddle Inference 能力直接基于飞桨的训练算子，因此它支持飞桨训练出的所有模型的推理。


Paddle Inference Python 端预测部署主要包含两个步骤：
- 导出预测部署模型
- 基于 Python 预测


## 导出预测部署模型
部署时需要使用预测格式的模型（即动态图转静态图操作）。预测格式模型相对训练格式模型而言，在拓扑上裁剪掉了预测不需要的算子，并且会做特定部署优化。具体操作详见[FastGeneration 加速及模型静态图导出](../../README.md)。

## 基于 Python 预测
<!-- 同上，高性能预测的默认输入和输出形式也为文件，可分别通过 test_path 和 save_path 进行指定，通过如下命令便可以基于 Paddle Inference 进行高性能预测： -->

在终端输入以下命令可在 GPU 上进行预测：
```shell
python inference_unimo_text.py --inference_model_dir ../../inference_model
```

关键参数释义如下：
* `inference_model_dir`：用于高性能推理的静态图模型参数路径；默认为"../../inference_model"。
