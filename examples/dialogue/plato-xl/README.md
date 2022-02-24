# PLATO-XL

## 模型简介

构建高质量的开放领域（Open-Domain）的对话机器人，使得它能用自然语言与人自由地交流，这一直是自然语言处理领域终极目标之一。

为了能够简易地构建一个高质量的开放域聊天机器人，本项目在 Paddle 上实现了 PLATO-XL 的预测模型，并实现了高性能的预测加速，而整套 float16 的方案可以确保在 32G V100 单卡上就能 load 并执行 11B 的 PLATO-XL 模型，无需再涉及 float32 相关计算。用户可以通过下载预训练模型快速构建一个开放域聊天机器人。

PLATO-XL的训练过程及其他细节详见 [Knover](https://github.com/PaddlePaddle/Knover)

## 快速开始

### 环境依赖

- python 3.7+
- sentencepiece
- termcolor  

安装方式：
``` python
pip install sentencepiece termcolor
```

### 数据准备

您可以从以下位置下载预训练模型文件：

# todo: modify this url.
* PLATO-XL, 72-layers, 32-heads, 3072-hidden, EN: [预训练模型](https://dialogue.bj.bcebos.com/Knover/projects/PLATO-XL/11B.tar)

```shell
wget https://dialogue.bj.bcebos.com/Knover/projects/PLATO-XL/11B.tar
```

**NOTE:** PLATO-XL 网络参数量较大，即使是在使用 float16 的情况下，72 层网络至少需要显存约 24G，并且需要保证当前使用的 GPU 支持 float16 的计算。
支持 float16 的 GPU 信息可以在 NVIDIA [官网](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix)上查询；
您当前使用的 GPU 的 compute capability 同样可以在 NVIDIA [官网](https://developer.nvidia.com/zh-cn/cuda-gpus#compute)上找到，与上面链接中是否可使用 GPU 相对应。

sentencepiece分词预训练模型和词表文件下载：

```shell
wget https://bj.bcebos.com/paddlenlp/models/transformers/plato2/data.tar.gz
tar -zxf data.tar.gz
```

### 高性能生成

运行如下命令即可开始与聊天机器人用英语进行简单的对话

```shell
export CUDA_VISIBLE_DEVICES=0
python infer.py --vocab_path ./data/vocab.txt --spm_model_file ./data/spm.model --use_role --position_style relative
```

以上参数表示：

* `--vocab_path`: 词表文件路径。
* `--spm_model_file`: sentencepiece 分词预训练模型路径。
* `--use_role`: 是否使用 role embedding。
* `--position_style`: 位置编码方式，这里可以选择是 "relative" 或是 "continuous"。
