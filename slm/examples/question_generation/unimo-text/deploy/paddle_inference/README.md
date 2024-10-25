# Paddle Inference 部署
本文档将介绍如何使用[Paddle Inference](https://paddle-inference.readthedocs.io/en/latest/guides/introduction/index_intro.html#paddle-inference)工具进行问题生成应用高性能推理推理部署。

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
部署时需要使用预测格式的模型（即动态图转静态图操作）。预测格式模型相对训练格式模型而言，在拓扑上裁剪掉了预测不需要的算子，并且会做特定部署优化。具体操作详见[FasterTransformer 加速及模型静态图导出](../../README.md)。

## 基于 Python 预测
<!-- 同上，高性能预测的默认输入和输出形式也为文件，可分别通过 test_path 和 save_path 进行指定，通过如下命令便可以基于 Paddle Inference 进行高性能预测： -->

在终端输入以下命令可在 GPU 上进行预测：
```shell
python deploy/paddle_inference/inference.py \
               --inference_model_dir ./export_checkpoint \
               --model_name_or_path "unimo-text-1.0" \
               --predict_file predict_file_name \
               --output_path output_path_name \
               --device gpu \
```

<!-- 在终端输入以下命令可在 CPU 上进行预测：
```shell
python deploy/paddle_inference/inference_unimo_text.py --inference_model_dir ./export_checkpoint --device cpu
``` -->
经静态图转换，FastTransformer 性能优化，Paddle Inference 加速后的部署模型在 dureader_qg devset 的预测时间为27.74秒，相较于未优化前169.24秒，耗时缩减为原来的16.39%。
关键参数释义如下：
* `inference_model_dir`：用于高性能推理的静态图模型参数路径，默认为"./export_checkpoint"。
* `model_name_or_path`：tokenizer 对应模型或路径，默认为"unimo-text-1.0"。
* `dataset_name`：数据集名称，默认为`dureader_qg`。
* `predict_file`：本地预测数据地址，数据格式必须与`dataset_name`所指数据集格式相同，默认为 None，当为 None 时默认加载`dataset_name`的 dev 集。
* `output_path`：表示预测结果的保存路径。
* `device`：推理时使用的设备，可选项["gpu"]，默认为"gpu"。
* `batch_size`：进行推理时的批大小，默认为16。
* `precision`：当使用 TensorRT 进行加速推理时，所使用的 TensorRT 精度，可选项["fp32", "fp16"]，默认为"fp32"。
<!-- * `precision`：当使用 TensorRT 进行加速推理时，所使用的 TensorRT 精度，可选项["fp32", "fp16", "int8"]，默认为"fp32"。 -->
<!-- * `device`：推理时使用的设备，可选项["gpu", "cpu", "xpu"]，默认为"gpu"。 -->
<!-- * `enable_mkldnn`：当使用 cpu 时，选择是否使用 MKL-DNN(oneDNN)进行加速推理，默认为 False。 -->
<!-- * `cpu_threads`：当使用 cpu 时，推理所用的进程数，默认为10。 -->
<!-- * `use_tensorrt`：当使用 gpu 时，选择是否使用 TensorRT 进行加速推理，默认为 False。 -->
