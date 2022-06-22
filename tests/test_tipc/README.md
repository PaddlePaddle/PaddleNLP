# 飞桨训推一体认证（TIPC）

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了PaddleNLP中部分模型的飞桨训推一体认证 (Training and Inference Pipeline Certification(TIPC)) 信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。


## 2. 汇总信息

打通情况汇总如下，已填写的部分表示可以使用本工具进行一键测试，未填写的表示正在支持中。

**字段说明：**
- 基础训练预测：包括模型单机单卡训练、单机多卡训练以及Paddle Inference Python预测。
- 更多训练方式：包括多机多卡、混合精度。

更详细的MKLDNN、TensorRT等预测加速相关功能的支持情况可以查看各测试工具的[更多教程](#more)。

| 模型名称 | 模型类型 | 基础<br>训练预测 | 更多<br>训练方式 | 模型压缩 |
| :--- |  :----:  | :--------: |  :----  |   :----  |
| bigru_crf | 序列标注  | 支持 | - | - |
| Transformer | 机器翻译 | 支持 | - | - |



## 3. 测试工具简介
### 目录介绍

```shell
test_tipc/
├── bigru_crf                      # bigru_crf模型实现
│   ├── data.py
│   ├── deploy
│   │   └── predict.py             # python预测部署脚本
│   ├── export_model.py            # 模型导出脚本
│   ├── model.py                   # 模型实现脚本
│   └── train.py                   # 训练脚本
├── transformer                    # Transformer 双精度模型实现
│   ├── modeling.py                # Transformer 双精度模型组网脚本
│   └── train.py                   # Transformer 双精度训练脚本
├── compare_results.py             # 用于对比log中的预测结果与results中的预存结果精度误差是否在限定范围内
├── configs                        # 配置文件目录
│   ├── bigru_crf                  # bigru_crf模型的测试配置文件目录
│       └── train_infer_python.txt # 测试Linux上python训练预测（基础训练预测）的配置文件
│   └── Transformer                # Transformer 模型的测试配置文件目录
│       └── train_infer_python.txt # 测试 Linux 上 python 训练预测（基础训练预测）的配置文件
├── prepare.sh                     # 完成test_*.sh运行所需要的数据和模型下载
├── readme.md                      # 使用文档
├── results                        # 预先保存的预测结果，用于和实际预测结果进行精读比对
│   ├── python_bigru_crf_results_fp16.txt # 预存的bigru_cr模型python预测fp16精度的结果
│   └── python_bigru_crf_results_fp32.txt # 预存的bigru_cr模型python预测fp32精度的结果
└── test_train_inference_python.sh # 测试python训练预测的主程序
```

### 测试流程概述

使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐，测试流程概括如下：

1. 运行prepare.sh准备测试所需数据和模型；
2. 运行要测试的功能对应的测试脚本`test_train_inference_python.sh`，产出log，由log可以看到不同配置是否运行成功；
3. 用`compare_results.py`对比log中的预测结果和预存在results目录下的结果，判断预测精度是否符合预期（在误差范围内）。

测试单项功能仅需两行命令，**如需测试不同模型/功能，替换配置文件即可**，命令格式如下：
```shell
# 功能：准备数据
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/prepare.sh  configs/[model_name]/[params_file_name]  [Mode]

# 功能：运行测试
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/test_train_inference_python.sh configs/[model_name]/[params_file_name]  [Mode]
```

例如，测试基本训练预测功能的`lite_train_lite_infer`模式，运行：
```shell
# 准备数据
bash test_tipc/prepare.sh ./test_tipc/configs/bigru_crf/train_infer_python.txt 'lite_train_lite_infer'
# 运行测试
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/bigru_crf/train_infer_python.txt 'lite_train_lite_infer'
```
关于本示例命令的更多信息可查看[基础训练预测使用文档](docs/test_train_inference_python.md)。


<a name="more"></a>
## 4. 开始测试
各功能测试中涉及MKLDNN、TensorRT等多种预测相关参数配置，请点击下方相应链接了解更多细节和使用教程：
- [test_train_inference_python 使用](docs/test_train_inference_python.md) ：测试基于Python的模型训练、推理等基本功能。
