# PLATO-XL

## 模型简介

构建高质量的开放领域（Open-Domain）的对话机器人，使得它能用自然语言与人自由地交流，这一直是自然语言处理领域终极目标之一。

PLATO-XL 是业界首个开源的百亿超大规模开放域对话预训练模型，其使用了参数高效(encoder-decoder共享参数)的 UnifiedTransformer（prefix LM）模型架构，将模型参数量提升到了11B量级，经过了十亿级样本对话数据的预训练，并引入role embedding区分多方对话中的对话角色提升预训练效果，最终模型闲聊测试效果超过了众多代表性的对话模型。可以直接使用 PLATO-XL 构建高质量的开放领域对话机器人。

PaddleNLP 内置了 PLATO-XL 英文预训练模型以供使用。由于 PLATO-XL 模型规模较大，这使得其在预测时生成对话回复的时间较长，并且 11B 的参数量也可能超出部分型号 GPU 显存容量，这是大模型推理与落地存在的普遍和关键问题。PaddleNLP FasterGeneration 为 PLATO-XL 提供了 GPU 上的高性能生成加速能力，并且支持模型并行（张量并行）推理允许通过多张小显存容量的 GPU 使用百亿大模型，此外模型并行能进一步提升预测速度。

本项目提供了 PLATO-XL 英文模型使用 PaddleNLP FasterGeneration 进行高性能预测的使用示例。PLATO-XL 的训练及更多内容请参考 [PaddlePaddle/Knover](https://github.com/PaddlePaddle/Knover/tree/develop/projects/PLATO-XL)。




为了能够简易地构建一个高质量的开放域聊天机器人，本项目在 Paddle 上实现了 PLATO-XL 的预测模型，并实现了高性能的预测加速，而整套 float16 的方案可以确保在 32G V100 单卡上就能 load 并执行 11B 的 PLATO-XL 模型，无需再涉及 float32 相关计算。

此外，PLATO-XL 72-layers, 32-heads, 3072-hidden，网络参数量较大，即使是在使用 float16 的情况下，72 层网络至少需要显存约 24G，并且需要保证当前使用的 GPU 支持 float16 的计算。

其中：
* 支持 float16 的 GPU 信息可以在 NVIDIA [官网](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix) 上查询；
* 您当前使用的 GPU 的 compute capability 同样可以在 NVIDIA [官网](https://developer.nvidia.com/zh-cn/cuda-gpus#compute) 上找到，与上面链接中表格对应。

PLATO-XL 的训练过程及其他细节详见 [PaddlePaddle/Knover](https://github.com/PaddlePaddle/Knover/tree/develop/projects/PLATO-XL)

## 开始运行

### 环境依赖

- mpi (多卡)
- nccl

### 高性能生成

#### 单卡高性能推理

使用 `infer.py` 脚本进行测试，无需单独下载预训练模型，脚本将自行下载。运行如下命令即可进行高性能预测，脚本使用了一条对话生成的语句作为性能测试的例子。

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

**注意** 单卡预测至少需要22G显存。如果去掉

#### 多卡并行推理

```shell
mpirun -n 4 python infer_mp.py --use_role --position_style relative --batch_size 8 --min_out_len 20 --max_out_len 20 --topk 1 --use_faster --use_fp16 --profile
```

### 性能测试

`infer.py` 中同时提供了性能测试的支持，在上面预测命令的基础上加上 `--profile` 即可，将。

```shell
mpirun -n 4 python infer_mp.py --use_role --position_style relative --batch_size 8 --min_out_len 20 --max_out_len 20 --topk 1 --use_faster --use_fp16 --profile
```



### 预测部署
