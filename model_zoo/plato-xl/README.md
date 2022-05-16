# PLATO-XL: Exploring the Large-scale Pre-training of Dialogue Generation

## 模型简介

构建高质量的开放领域（Open-Domain）的对话机器人，使得它能用自然语言与人自由地交流，这一直是自然语言处理领域终极目标之一。

PLATO-XL 是业界首个开源的百亿超大规模开放域对话预训练模型，其使用了参数高效(encoder-decoder共享参数)的 UnifiedTransformer（prefix LM）模型架构，将模型参数量提升到了11B量级，经过了十亿级样本对话数据的预训练，并引入role embedding区分多方对话中的对话角色提升预训练效果，最终模型闲聊测试效果超过了众多代表性的对话模型。可以直接使用 PLATO-XL 构建高质量的开放领域对话机器人。

PaddleNLP 内置了 PLATO-XL 英文预训练模型以供使用。由于 PLATO-XL 模型规模较大，这使得其在预测时生成对话回复的时间较长，并且 11B 的参数量也可能超出部分型号 GPU 显存容量，这是大模型推理与落地存在的普遍和关键问题。PaddleNLP FasterGeneration 为 PLATO-XL 提供了 GPU 上的高性能生成加速能力，并且支持模型并行（张量并行）推理允许通过多张小显存容量的 GPU 使用百亿大模型，相比单卡代码中也只增加了`enable_ft_para()`一行，此外模型并行能进一步提升预测速度。

本项目提供了 PLATO-XL 英文模型使用 PaddleNLP FasterGeneration 进行高性能预测的使用示例。PLATO-XL 的训练及更多内容请参考 [PaddlePaddle/Knover](https://github.com/PaddlePaddle/Knover/tree/develop/projects/PLATO-XL)。

## 开始运行
### 单卡高性能推理

`infer.py` 是 PLATO-XL 高性能预测使用示例脚本，可以使用如下命令运行：

```shell
python infer.py --topk 4 --max_out_len 64 --use_faster --use_fp16
```

该脚本各个参数含义如下：

- `topk` 用于Top-K采样策略，采样时将只从概率最高K个token中采样，默认为1，即greedy search。
- `topp` 用于Top-P采样策略，采样时将只从概率最高且累加概率不超过该值的token中采样，默认为1.0。
- `max_out_len` 指定生成的最大长度，默认为64。
- `min_out_len` 指定生成的最小长度，默认为1。
- `temperature` 用于调整预测概率分布，默认为1.0，即保持模型原有的预测概率。
- `use_faster` 使用 FasterGeneration
- `use_fp16` 使用FP16，只在使用FasterGeneration时生效

脚本中使用了一条如下的多轮对话的样本数据， 由`List[str]`表示，其中每个`str`表示一句话，将根据历史对话内容生成回复。

```python
    history = [
        "hi , Mary ! What do you usually like to do in your spare time ?",
        "well , I spend a lot of time watching movies .",
        "what a confidence ! I always watch a lot of movies , too ."
        "oh really , Frank ? What kind of movies do you like ?"
    ]
```

**注意** 由于 PLATO-XL 模型较大，单卡预测至少需要22G显存（使用FP16时），且模型下载需要一定时间（FP32的权重文件约41G）。

### 多卡并行推理

多卡并行推理当前依赖 MPI（[MPICH](https://www.mpich.org)、[OpenMPI](https://www.open-mpi.org)均可）和[NCCL](https://developer.nvidia.com/nccl)，如需使用还请先安装依赖。安装完成后仍然使用 `infer.py` 来进行预测，相比单卡时不同的只是通过mpi来启动运行，如下：

```shell
mpirun -n 4 python infer.py --topk 4 --max_out_len 64 --use_faster --use_fp16
```

其中`-n 4`指明使用的进程和GPU数，在`n`设置为1时仍将进行单卡推理。由于多卡并行推理使用和单卡使用不同的依赖库，第一次运行时将重新进行JIT编译。

### 性能测试

`infer.py` 中同时提供了性能测试的支持，在上面预测命令的基础上加上 `--profile` 即可，如下：

```shell
mpirun -n 4 python infer.py --batch_size 8 --min_out_len 20 --max_out_len 20 --topk 1 --use_faster --use_fp16 --profile
```

此外还指定了`batch_size`和`min_out_len`来得到特定输入输出大小下的性能，性能测试将给出循环运行多次的平均时延。以下为单卡高性能推理和4卡张量并行推理性能数据（V100，CUDA 10.2，输入长度60、输出长度20），可以看出4卡并行速度为单卡的2倍左右。

<table>
<caption>PLATO-XL 高性能推理速度&nbsp;&nbsp;(in ms/batch)</caption>
    <tr style="text-align:center;">
        <td align=center>batch size</td>
        <td align=center>K</td>
        <td align=center>FasterGeneration</br>1卡</br>FP16</td>
        <td align=center>FasterGeneration</br>4卡</br>FP16</td>
        <td align=center>多卡并行</br>SpeedUp</td>
    </tr>
    <tr style="text-align:center;">
        <td align=center>1</td>
        <td align=center>1</td>
        <td align=center>706.937</td>
        <td align=center>348.653</td>
        <td align=center>2.027</td>
    </tr>
    <tr style="text-align:center;">
        <td align=center>1</td>
        <td align=center>10</td>
        <td align=center>707.514</td>
        <td align=center>348.699</td>
        <td align=center>2.029</td>
    </tr>
    <tr style="text-align:center;">
        <td align=center>4</td>
        <td align=center>1</td>
        <td align=center>768.597</td>
        <td align=center>384.730</td>
        <td align=center>1.997</td>
    </tr>
    <tr style="text-align:center;">
        <td align=center>4</td>
        <td align=center>10</td>
        <td align=center>770.008</td>
        <td align=center>385.244</td>
        <td align=center>1.998</td>
    </tr>
    <tr style="text-align:center;">
        <td align=center>8</td>
        <td align=center>1</td>
        <td align=center>862.017</td>
        <td align=center>418.313</td>
        <td align=center>2.060</td>
    </tr>
    <tr style="text-align:center;">
        <td align=center>8</td>
        <td align=center>10</td>
        <td align=center>866.490</td>
        <td align=center>418.965</td>
        <td align=center>2.068</td>
    </tr>
    <tr style="text-align:center;">
        <td align=center>16</td>
        <td align=center>1</td>
        <td align=center>1016.362</td>
        <td align=center>486.974</td>
        <td align=center>2.087</td>
    </tr>
    <tr style="text-align:center;">
        <td align=center>16</td>
        <td align=center>10</td>
        <td align=center>1060.472</td>
        <td align=center>488.156</td>
        <td align=center>2.172</td>
    </tr>
    <tr style="text-align:center;">
        <td align=center>32</td>
        <td align=center>1</td>
        <td align=center>1325.700</td>
        <td align=center>606.770</td>
        <td align=center>2.184</td>
    </tr>
    <tr style="text-align:center;">
        <td align=center>32</td>
        <td align=center>10</td>
        <td align=center>1326.222</td>
        <td align=center>608.479</td>
        <td align=center>2.179</td>
    </tr>
</table>

## Reference

1. Bao S, He H, Wang F, et al. PLATO-XL: Exploring the Large-scale Pre-training of Dialogue Generation[J]. arXiv preprint arXiv:2109.09519, 2021.
