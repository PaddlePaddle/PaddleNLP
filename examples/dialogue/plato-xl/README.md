# PLATO-XL

## 模型简介

构建高质量的开放领域（Open-Domain）的对话机器人，使得它能用自然语言与人自由地交流，这一直是自然语言处理领域终极目标之一。

为了能够简易地构建一个高质量的开放域聊天机器人，本项目在 Paddle 上实现了 PLATO-XL 的预测模型，并实现了高性能的预测加速，而整套 float16 的方案可以确保在 32G V100 单卡上就能 load 并执行 11B 的 PLATO-XL 模型，无需再涉及 float32 相关计算。

此外，PLATO-XL 72-layers, 32-heads, 3072-hidden，网络参数量较大，即使是在使用 float16 的情况下，72 层网络至少需要显存约 24G，并且需要保证当前使用的 GPU 支持 float16 的计算。

其中：
* 支持 float16 的 GPU 信息可以在 NVIDIA [官网](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix) 上查询；
* 您当前使用的 GPU 的 compute capability 同样可以在 NVIDIA [官网](https://developer.nvidia.com/zh-cn/cuda-gpus#compute) 上找到，与上面链接中表格对应。

PLATO-XL 的训练过程及其他细节详见 [Knover](https://github.com/PaddlePaddle/Knover/tree/develop/projects/PLATO-XL)

## 快速开始

### 环境依赖

- python 3.7+
- sentencepiece

安装方式：
``` python
pip install sentencepiece
```

### 高性能生成

使用 `infer.py` 脚本进行测试，无需单独下载预训练模型，脚本将自行下载。运行如下命令即可进行高性能预测，脚本使用了一条对话生成的语句作为性能测试的例子，forward 将自动循环 200 次前向以供性能测试需要。

```shell
export CUDA_VISIBLE_DEVICES=0
python infer.py --use_role --position_style relative --max_out_len 64 --min_out_len 1 --topk 4
```

该脚本各个参数含义如下：

* `--use_role`: 是否使用 role embedding。
* `--position_style`: 位置编码方式，这里可以选择是 "relative" 或是 "continuous"。
* `--max_out_len`: 最长的输出的长度。
* `--min_out_len`: 最短的输出长度。
* `--topk`: 用于 top_k sampling 的 k 值的设定。
* `--topp`: 用于 top_p sampling 的 p 值的设定。
