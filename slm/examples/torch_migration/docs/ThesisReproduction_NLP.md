# 论文复现指南

## 目录

- [1. 总览](#1)
    - [1.1 背景](#1.1)
    - [1.2 前序工作](#1.2)
- [2. 整体框图](#2)
    - [2.1 流程概览](#2.1)
    - [2.2 reprod_log whl 包](#2.2)
- [3. 论文复现理论知识及实战](#3)
    - [3.1 模型结构对齐](#3.1)
    - [3.2 验证/测试集数据读取对齐](#3.2)
    - [3.3 评估指标对齐](#3.3)
    - [3.4 损失函数对齐](#3.4)
    - [3.5 优化器对齐](#3.5)
    - [3.6 学习率对齐](#3.6)
    - [3.7 正则化策略对齐](#3.7)
    - [3.8 反向对齐](#3.8)
    - [3.9 训练集数据读取对齐](#3.9)
    - [3.10 网络初始化对齐](#3.10)
    - [3.11 模型训练对齐](#3.11)
    - [3.12 单机多卡训练](#3.12)
- [4. 论文复现注意事项与 FAQ](#4)
    - [4.0 通用注意事项](#4.0)
    - [4.1 模型结构对齐](#4.1)
    - [4.2 验证/测试集数据读取对齐](#4.2)
    - [4.3 评估指标对齐](#4.3)
    - [4.4 损失函数对齐](#4.4)
    - [4.5 优化器对齐](#4.5)
    - [4.6 学习率对齐](#4.6)
    - [4.7 正则化策略对齐](#4.7)
    - [4.8 反向对齐](#4.8)
    - [4.9 训练集数据读取对齐](#4.9)
    - [4.10 网络初始化对齐](#4.10)
    - [4.11 模型训练对齐](#4.11)

<a name="1"></a>
## 1. 总览

<a name="1.1"></a>
### 1.1 背景

* 以深度学习为核心的人工智能技术仍在高速发展，通过论文复现，开发者可以获得
    * 学习成长：自我能力提升
    * 技术积累：对科研或工作有所帮助和启发
    * 社区荣誉：成果被开发者广泛使用

<a name="1.2"></a>
### 1.2 前序工作

基于本指南复现论文过程中，建议开发者准备以下内容。

* 了解该模型输入输出格式。以 BERT 的情感分类任务为例，通过阅读论文与参考代码，了解到模型输入为`[batch_size, sequence_length]`的 tensor，类型为`int64`，label 为`[batch, ]`的 label，类型为`int64`。
* 准备好训练/验证数据集，用于模型训练与评估
* 准备好 fake input data 以及 label，与模型输入 shape、type 等保持一致，用于后续模型前向对齐。
    * 在对齐模型前向过程中，我们不需要考虑数据集模块等其他模块，此时使用 fake data 是将模型结构和数据部分解耦非常合适的一种方式。
    * 将 fake data 以文件的形式存储下来，也可以保证 PaddlePaddle 与参考代码的模型结构输入是完全一致的，更便于排查问题。
    * 在该步骤中，以 BERT 为例，生成 fake data 的脚本可以参考：[gen_fake_data.py](https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/fake_data/gen_fake_data.py)。
* 在特定设备(CPU/GPU)上，跑通参考代码的预测过程(前向)以及至少2轮(iteration)迭代过程，保证后续基于 PaddlePaddle 复现论文过程中可对比。
* 本文档基于 `BERT-SST2-Prod` 代码以及`reprod_log` whl 包进行说明与测试。如果希望体验，建议参考[BERT-SST2-Prod 文档](https://github.com/JunnYu/BERT-SST2-Prod/blob/main/README.md)进行安装与测试。
* 在复现的过程中，只需要将 PaddlePaddle 的复现代码以及打卡日志上传至 github，不能在其中添加参考代码的实现，在验收通过之后，需要删除打卡日志。建议在初期复现的时候，就将复现代码与参考代码分成2个文件夹进行管理。

<a name="2"></a>
## 2. 整体框图

<a name="2.1"></a>
### 2.1 流程概览

面对一篇自然语言处理的论文，复现该论文的整体流程如下图所示。

![图片](https://user-images.githubusercontent.com/16911935/199389647-b000a7b1-28d1-485e-8ec0-3e7e2c05884a.png)

总共包含11个步骤。为了高效复现论文，设置了5个验收节点。如上图中黄色框所示。后续章节会详细介绍上述步骤和验收节点，具体内容安排如下：

* 第3章：介绍11个复现步骤的理论知识、实战以及验收流程。
* 第4章：针对复现流程过程中每个步骤可能出现的问题，本章会进行详细介绍。如果还是不能解决问题，可以提 ISSUE 进行讨论，提 ISSUE 地址：[https://github.com/PaddlePaddle/Paddle/issues/new/choose](https://github.com/PaddlePaddle/Paddle/issues/new/choose)

<a name="2.2"></a>
### 2.2 reprod_log whl 包

#### 2.2.1 reprod_log 工具简介
`reprod_log`是用于论文复现赛中辅助自查和验收工具。该工具源代码地址在：[https://github.com/WenmuZhou/reprod_log](https://github.com/WenmuZhou/reprod_log)。主要功能如下：

* 存取指定节点的输入输出 tensor
* 基于文件的 tensor 读写
* 2个字典的对比验证
* 对比结果的输出与记录

更多 API 与使用方法可以参考：[reprod_log API 使用说明](https://github.com/WenmuZhou/reprod_log/blob/master/README.md)。

#### 2.2.2 reprod_log 使用 demo

下面基于代码：[https://github.com/JunnYu/BERT-SST2-Prod/tree/main/pipeline/reprod_log_demo](https://github.com/JunnYu/BERT-SST2-Prod/tree/main/pipeline/reprod_log_demo)，给出如何使用该工具。

文件夹中包含`write_log.py`和`check_log_diff.py`文件，其中`write_log.py`中给出了`ReprodLogger`类的使用方法，`check_log_diff.py`给出了`ReprodDiffHelper`类的使用方法，依次运行两个 python 文件，使用下面的方式运行代码。

```shell
# 进入文件夹
cd pipeline/reprod_log_demo
# 随机生成矩阵，写入文件中
python write_log.py
# 进行文件对比，输出日志
python check_log_diff.py
```

最终会输出以下内容

```
[2021/11/18 09:29:31] root INFO: demo_test_1:
[2021/11/18 09:29:31] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/18 09:29:31] root INFO: demo_test_2:
[2021/11/18 09:29:31] root INFO:     mean diff: check passed: False, value: 0.33387675881385803
[2021/11/18 09:29:31] root INFO: diff check failed
```

可以看出：对于 key 为`demo_test_1`的矩阵，由于 diff 为0，小于设置的阈值`1e-6`，核验成功；对于 key 为`demo_test_2`的矩阵，由于 diff 为0.33，大于设置的阈值`1e-6`，核验失败。

#### 2.2.3 reprod_log 在论文复现中应用

在论文复现中，基于 reprod_log 的结果记录模块，产出下面若干文件
```
log_reprod
├── forward_paddle.npy
├── forward_torch.npy    # 与forward_paddle.npy作为一并核查的文件对
├── metric_paddle.npy
├── metric_torch.npy     # 与metric_paddle.npy作为一并核查的文件对
├── loss_paddle.npy
├── loss_torch.npy       # 与loss_paddle.npy作为一并核查的文件对
├── bp_align_paddle.npy
├── bp_align_torch.npy   # 与bp_align_paddle.npy作为一并核查的文件对
├── train_align_paddle.npy
├── train_align_torch.npy # pytorch运行得到的参考评估指标
```

基于 reprod_log 的`ReprodDiffHelper`模块，产出下面5个日志文件。

```
├── forward_diff.log     # forward_paddle.npy与forward_torch.npy生成的diff结果文件
├── metric_diff.log      # metric_paddle.npy与metric_torch.npy生成的diff结果文件
├── loss_diff.log          # loss_paddle.npy与loss_torch.npy生成的diff结果文件
├── bp_align_diff.log    # bp_align_paddle.npy与bp_align_torch.npy生成的diff结果文件
├── train_align_diff.log # train_align_paddle.train_align_torch.npy生成的diff结果文件
```

上述文件的生成代码都需要开发者进行开发，验收时需要提供上面罗列的所有文件（不需要提供产生这些文件的可运行程序）以及完整的模型训练评估程序和日志。
BERT-SST2-Prod 项目提供了基于 reprod_log 的5个验收点对齐验收示例，具体代码地址为：[https://github.com/JunnYu/BERT-SST2-Prod/tree/main/pipeline](https://github.com/JunnYu/BERT-SST2-Prod/tree/main/pipeline)，
每个文件夹中的 README.md 文档提供了使用说明。

<a name="3"></a>
## 3. 论文复现理论知识及实战

<a name="3.1"></a>
### 3.1 模型结构对齐

对齐模型结构时，一般有3个主要步骤：

* 网络结构代码转换
* 权重转换
* 模型组网正确性验证

下面详细介绍这3个部分。

#### 3.1.1 网络结构代码转换

**【基本流程】**

由于 PyTorch 的 API 和 PaddlePaddle 的 API 非常相似，可以参考[PyTorch-PaddlePaddle API 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/08_api_mapping/pytorch_api_mapping_cn.html)
，组网部分代码直接进行手动转换即可。

**【注意事项】**

如果遇到 PaddlePaddle 没有的 API，可以尝试用多种 API 来组合，也可以给 PaddlePaddle 团队提[ISSUE](https://github.com/PaddlePaddle/Paddle/issues)，获得支持。

**【实战】**

BERT 网络结构的 PyTorch 实现: [transformers-bert](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py)

对应转换后的 PaddlePaddle 实现: [paddlenlp-bert](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/bert/modeling.py)


#### 3.1.2 权重转换

**【基本流程】**

组网代码转换完成之后，需要对模型权重进行转换，如果 PyTorch repo 中已经提供权重，那么可以直接下载并进行后续的转换；如果没有提供，则可以基于 PyTorch 代码，随机生成一个初始化权重(定义完 model 以后，使用`torch.save()` API 保存模型权重)，然后进行权重转换。

**【注意事项】**

在权重转换的时候，需要注意`paddle.nn.Linear`等 API 的权重保存格式和名称等与 PyTorch 稍有 diff，具体内容可以参考`4.1章节`。

**【实战】**

BERT 的代码转换脚本可以在这里查看：[https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/weights/torch2paddle.py](https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/weights/torch2paddle.py)，

注意：运行该代码需要首先下载 Huggingface 的 BERT 预训练模型到该目录下，下载地址为：[https://huggingface.co/bert-base-uncased/blob/main/pytorch_model.bin](https://huggingface.co/bert-base-uncased/blob/main/pytorch_model.bin)

```python
# https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/weights/torch2paddle.py

from collections import OrderedDict

import numpy as np
import paddle
import torch
from paddlenlp.transformers import BertForPretraining as PDBertForMaskedLM
from transformers import BertForMaskedLM as PTBertForMaskedLM


def convert_pytorch_checkpoint_to_paddle(
        pytorch_checkpoint_path="pytorch_model.bin",
        paddle_dump_path="model_state.pdparams",
        version="old", ):
    hf_to_paddle = {
        "embeddings.LayerNorm": "embeddings.layer_norm",
        "encoder.layer": "encoder.layers",
        "attention.self.query": "self_attn.q_proj",
        "attention.self.key": "self_attn.k_proj",
        "attention.self.value": "self_attn.v_proj",
        "attention.output.dense": "self_attn.out_proj",
        "intermediate.dense": "linear1",
        "output.dense": "linear2",
        "attention.output.LayerNorm": "norm1",
        "output.LayerNorm": "norm2",
        "predictions.decoder.": "predictions.decoder_",
        "predictions.transform.dense": "predictions.transform",
        "predictions.transform.LayerNorm": "predictions.layer_norm",
    }
    do_not_transpose = []
    if version == "old":
        hf_to_paddle.update({
            "predictions.bias": "predictions.decoder_bias",
            ".gamma": ".weight",
            ".beta": ".bias",
        })
        do_not_transpose = do_not_transpose + ["predictions.decoder.weight"]

    pytorch_state_dict = torch.load(
        pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        is_transpose = False
        if k[-7:] == ".weight":
            # embeddings.weight and LayerNorm.weight do not transpose
            if all(d not in k for d in do_not_transpose):
                if ".embeddings." not in k and ".LayerNorm." not in k:
                    if v.ndim == 2:
                        v = v.transpose(0, 1)
                        is_transpose = True
        oldk = k
        for hf_name, pd_name in hf_to_paddle.items():
            k = k.replace(hf_name, pd_name)

        # add prefix `bert.`
        if "bert." not in k and "cls." not in k and "classifier" not in k:
            k = "bert." + k

        print(f"Converting: {oldk} => {k} | is_transpose {is_transpose}")
        paddle_state_dict[k] = v.data.numpy()

    paddle.save(paddle_state_dict, paddle_dump_path)


def compare(out_torch, out_paddle):
    out_torch = out_torch.detach().numpy()
    out_paddle = out_paddle.detach().numpy()
    assert out_torch.shape == out_paddle.shape
    abs_dif = np.abs(out_torch - out_paddle)
    mean_dif = np.mean(abs_dif)
    max_dif = np.max(abs_dif)
    min_dif = np.min(abs_dif)
    print("mean_dif:{}".format(mean_dif))
    print("max_dif:{}".format(max_dif))
    print("min_dif:{}".format(min_dif))


def test_forward():
    paddle.set_device("cpu")
    model_torch = PTBertForMaskedLM.from_pretrained("./bert-base-uncased")
    model_paddle = PDBertForMaskedLM.from_pretrained("./bert-base-uncased")
    model_torch.eval()
    model_paddle.eval()
    np.random.seed(42)
    x = np.random.randint(
        1, model_paddle.bert.config["vocab_size"], size=(4, 64))
    input_torch = torch.tensor(x, dtype=torch.int64)
    out_torch = model_torch(input_torch)[0]

    input_paddle = paddle.to_tensor(x, dtype=paddle.int64)
    out_paddle = model_paddle(input_paddle)[0]

    print("torch result shape:{}".format(out_torch.shape))
    print("paddle result shape:{}".format(out_paddle.shape))
    compare(out_torch, out_paddle)


if __name__ == "__main__":
    convert_pytorch_checkpoint_to_paddle(
        "./bert-base-uncased/pytorch_model.bin",
        "./bert-base-uncased/model_state.pdparams")
    test_forward()
    # torch result shape:torch.Size([4, 64, 30522])
    # paddle result shape:[4, 64, 30522]
    # mean_dif:1.666686512180604e-05
    # max_dif:0.00015211105346679688
    # min_dif:0.0
```

运行完成之后，会在当前目录生成`model_state.pdparams`文件，即为转换后的 PaddlePaddle 预训练模型。
**Tips**: 由于 paddlenlp 中已有转换后的 bert-base-uncased 模型，因此可以一键加载，程序会自动下载对应权重！


#### 3.1.3 模型组网正确性验证

**【基本流程】**

1. 定义 PyTorch 模型，加载权重，固定 seed，基于 numpy 生成随机数，转换为 PyTorch 可以处理的 tensor，送入网络，获取输出，使用 reprod_log 保存结果。
2. 定义 PaddlePaddle 模型，加载权重，固定 seed，基于 numpy 生成随机数，转换为 PaddlePaddle 可以处理的 tensor，送入网络，获取输出，使用 reprod_log 保存结果。
3.  使用 reprod_log 排查 diff，小于阈值，即可完成自测。

**【注意事项】**

* 模型在前向对齐验证时，需要调用`model.eval()`方法，保证组网中的随机量被关闭，比如 BatchNorm、Dropout 等。
* 给定相同的输入数据，为保证可复现性，如果有随机数生成，固定相关的随机种子。
* 输出 diff 可以使用`np.mean(np.abs(o1 - o2))`进行计算，一般小于1e-6的话，可以认为前向没有问题。如果最终输出结果 diff 较大，可以使用二分的方法进行排查，比如说 BERT，包含1个 embdding 层、12个 transformer-block 以及最后的 MLM head 层，那么完成模型组网和权重转换之后，如果模型输出没有对齐，可以尝试输出中间某一个 transformer-block 的 tensor 进行对比，如果相同，则向后进行排查；如果不同，则继续向前进行排查，以此类推，直到找到导致没有对齐的操作。

**【实战】**

BERT 模型组网正确性验证可以参考如下示例代码：
[https://github.com/JunnYu/BERT-SST2-Prod/tree/main/pipeline/Step1](https://github.com/JunnYu/BERT-SST2-Prod/tree/main/pipeline/Step1

**【验收】**

对于待复现的项目，前向对齐验收流程如下。

1. 准备输入：fake data
    * 使用参考代码的 dataloader，生成一个 batch 的数据，保存下来，在前向对齐时，直接从文件中读入。
    * 固定随机数种子，生成 numpy 随机矩阵，转化 tensor
2. 保存输出：
    * PaddlePaddle/PyTorch：dict，key 为 tensor 的 name（自定义），value 为 tensor 的值。最后将 dict 保存到文件中。建议命名为`forward_paddle.npy`和`forward_torch.npy`。
3. 自测：使用 reprod_log 加载2个文件，使用 report 功能，记录结果到日志文件中，建议命名为`forward_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。
4. 提交内容：新建文件夹，将`forward_paddle.npy`、`forward_torch.npy`与`forward_diff_log.txt`文件放在文件夹中，后续的输出结果和自查日志也放在该文件夹中，一并打包上传即可。
5. 注意：
    * PaddlePaddle 与 PyTorch 保存的 dict 的 key 需要保持相同，否则 report 过程可能会提示 key 无法对应，从而导致 report 失败，之后的`【验收】`环节也是如此。
    * 如果是固定随机数种子，建议将 fake data 保存到 dict 中，方便 check 参考代码和 PaddlePaddle 的输入是否一致。

<a name="3.2"></a>
### 3.2 验证/测试集数据读取对齐

**【基本流程】**

对于一个数据集，一般有以下一些信息需要重点关注

* 数据集名称、下载地址
* 训练集/验证集/测试集

PaddlePaddle 中数据集相关的 API 为`paddle.io.Dataset`，PyTorch 中对应为`torch.utils.data.Dataset`，二者功能一致，在绝大多数情况下，可以使用该类构建数据集。它是描述 Dataset 方法和行为的抽象类，在具体实现的时候，需要继承这个基类，实现其中的`__getitem__`和`__len__`方法。除了参考代码中相关实现，也可以参考待复现论文中的说明。

复现完 Dataset 之后，可以构建 Dataloader，对数据进行组 batch、批处理，送进网络进行计算。

`paddle.io.DataLoader`可以进行数据加载，将数据分成批数据，并提供加载过程中的采样。PyTorch 对应的实现为`torch.utils.data.DataLoader`，二者在功能上一致，只是在参数方面稍有 diff：（1）PaddlePaddle 缺少对`pin_memory`等参数的支持；（2）PaddlePaddle 增加了`use_shared_memory`参数来选择是否使用共享内存加速数据加载过程。

**【注意事项】**

论文中一般会提供数据集的名称以及基本信息。复现过程中，我们在下载完数据之后，建议先检查下是否和论文中描述一致，否则可能存在的问题有：

* 数据集版本不同，比如论文中使用了 cnn_dailymail 的 v3.0.0版本数据集，但是我们下载的是 cnn_dailymail 的 v1.0.0版本数据集，如果不对其进行检查，可能会导致我们最终训练的数据量等与论文中有 diff
* 数据集使用方式不同，有些论文中，可能只是抽取了该数据集的子集进行方法验证，此时需要注意抽取方法，需要保证抽取出的子集完全相同。
* 在评估指标对齐时，我们可以固定 batch size，关闭 Dataloader 的 shuffle 操作。

构建数据集时，可以使用 paddlenlp 中的数据集加载方式，具体可以参考：[如何自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)。对应地，PyTorch 中的数据处理 api 可以参考：[huggingface 的 datasets 自定义数据集](https://huggingface.co/docs/datasets/about_dataset_load.html#building-a-dataset)。对于其中之一，可以找到另一个平台的实现。

此外，
* 有些自定义的数据处理方法，如果不涉及到深度学习框架的部分，可以直接复用。
* 对于特定任务中的数据预处理方法，比如说 Tokenizer，如果没有现成的 API 可以调用，可以参考官方模型套件中的一些实现方法，比如 PaddleClas、PaddleDetection、PaddleSeg 等。

**【实战】**

BERT 模型复现过程中，数据预处理和 Dataset、Dataloader 的检查可以参考该文件：
[https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/Step2/test_data.py](https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/Step2/test_data.py)


使用方法可以参考[数据检查文档](https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/Step2/README.md)。

<a name="3.3"></a>
### 3.3 评估指标对齐

**【基本流程】**

PaddlePaddle 提供了一系列 Metric 计算类，比如说`Accuracy`, `Auc`, `Precision`, `Recall`等，而 PyTorch 中，目前可以通过组合的方式实现 metric 计算，或者调用[huggingface-datasets](https://huggingface.co/docs/datasets/about_metrics.html?highlight=metric)，在论文复现的过程中，需要注意保证对于该模块，给定相同的输入，二者输出完全一致。具体流程如下。

1. 构建 fake 数据
1. 使用 PyTorch 的指标获取评估结果，使用 reprod_log 保存结果。
2. 使用 PaddlePaddle 的指标获取评估结果，使用 reprod_log 保存结果。
3. 使用 reprod_log 排查 diff，小于阈值，即可完成自测。

**【注意事项】**

在评估指标对齐之前，需要注意保证对于该模块，给定相同的输入，二者输出完全一致。


**【实战】**

评估指标对齐检查方法可以参考文档：[评估指标对齐检查方法文档](https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/Step2/README.md#%E6%95%B0%E6%8D%AE%E8%AF%84%E4%BC%B0%E5%AF%B9%E9%BD%90%E6%B5%81%E7%A8%8B)


**【验收】**

对于待复现的项目，评估指标对齐验收流程如下。

1. 输入：dataloader, model
2. 输出：
    * PaddlePaddle/PyTorch：dict，key 为 tensor 的 name（自定义），value 为具体评估指标的值。最后将 dict 使用 reprod_log 保存到各自的文件中，建议命名为`metric_paddle.npy`和`metric_torch.npy`。
    * 自测：使用 reprod_log 加载2个文件，使用 report 功能，记录结果到日志文件中，建议命名为`metric_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。
3. 提交内容：将`metric_paddle.npy`、`metric_torch.npy`与`metric_diff_log.txt`文件备份到`3.1节验收环节`新建的文件夹中，后续的输出结果和自查日志也放在该文件夹中，一并打包上传即可。
4. 注意：
    * 数据需要是真实数据
    * 需要检查论文是否只是抽取了验证集/测试集中的部分文件，如果是的话，则需要保证 PaddlePaddle 和参考代码中 dataset 使用的数据集一致。


<a name="3.4"></a>
### 3.4 损失函数对齐

**【基本流程】**

PaddlePaddle 与 PyTorch 均提供了很多 loss function，用于模型训练，具体的 API 映射表可以参考：[Loss 类 API 映射列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/08_api_mapping/pytorch_api_mapping_cn.html#lossapi)。以 CrossEntropyLoss 为例，主要区别为：
* PaddlePaddle 提供了对软标签、指定 softmax 计算纬度的支持。

如果论文中使用的 loss function 没有指定的 API，则可以尝试通过组合 API 的方式，实现自定义的 loss function。

具体流程如下。

1. 定义 PyTorch 模型，加载权重，加载 fake data 和 fake label（或者固定 seed，基于 numpy 生成随机数），转换为 PyTorch 可以处理的 tensor，送入网络，获取 loss 结果，使用 reprod_log 保存结果。
2. 定义 PaddlePaddle 模型，加载 fake data 和 fake label（或者固定 seed，基于 numpy 生成随机数），转换为 PaddlePaddle 可以处理的 tensor，送入网络，获取 loss 结果，使用 reprod_log 保存结果。
3. 使用 reprod_log 排查 diff，小于阈值，即可完成自测。

**【注意事项】**

* 计算 loss 的时候，建议设置`model.eval()`，避免模型中随机量的问题。

**【实战】**

本部分可以参考文档：[https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/Step3/README.md](https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/Step3/README.md)。

**【验收】**

对于待复现的项目，损失函数对齐验收流程如下。

1. 输入：fake data & label
2. 输出：
    * PaddlePaddle/PyTorch：dict，key 为 tensor 的 name（自定义），value 为具体评估指标的值。最后将 dict 使用 reprod_log 保存到各自的文件中，建议命名为`loss_paddle.npy`和`loss_torch.npy`。
3. 自测：使用 reprod_log 加载2个文件，使用 report 功能，记录结果到日志文件中，建议命名为`loss_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。
4. 提交内容：将`loss_paddle.npy`、`loss_torch.npy`与`loss_diff_log.txt`文件备份到`3.1节验收环节`新建的文件夹中，后续的输出结果和自查日志也放在该文件夹中，一并打包上传即可。

<a name="3.5"></a>
### 3.5 优化器对齐

**【基本流程】**

PaddlePaddle 中的 optimizer 有`paddle.optimizer`等一系列实现，PyTorch 中则有`torch.Optim`等一系列实现。

**【注意事项】**

以 SGD 等优化器为例，PaddlePaddle 与 Pytorch 的优化器区别主要如下。

* PaddlePaddle 在优化器中增加了对梯度裁剪的支持，在训练 GAN 或者一些 NLP、多模态任务中，这个用到的比较多。
* PaddlePaddle 的 SGD 不支持动量更新、动量衰减和 Nesterov 动量，这里需要使用`paddle.optimizer.Momentum` API 实现这些功能。

**【实战】**

本部分对齐建议对照[PaddlePaddle 优化器 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html)与参考代码的优化器实现进行对齐，用之后的反向对齐统一验证该模块的正确性。


<a name="3.6"></a>
### 3.6 学习率对齐

**【基本流程】**

* 学习率策略主要用于指定训练过程中的学习率变化曲线，这里可以将定义好的学习率策略，不断 step，即可得到对应的学习率值，可以将学习率值保存在列表或者矩阵中，使用`reprod_log`工具判断二者是否对齐。

**【注意事项】**

PaddlePaddle 中，需要首先构建学习率策略，再传入优化器对象中；对于 PyTorch，如果希望使用更丰富的学习率策略，需要先构建优化器，再传入学习率策略类 API。

**【实战】**

学习率复现对齐，可以参考代码：[学习率对齐验证文档](https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/Step4/README.md#%E5%AD%A6%E4%B9%A0%E7%8E%87%E5%AF%B9%E9%BD%90%E9%AA%8C%E8%AF%81)。

<a name="3.7"></a>
### 3.7 正则化策略对齐

**【基本流程】**

L2正则化策略用于模型训练，可以防止模型对训练数据过拟合，L1正则化可以用于得到稀疏化的权重矩阵，PaddlePaddle 中有`paddle.regularizer.L1Decay`与`paddle.regularizer.L2Decay` API。PyTorch 中，torch.optim 集成的优化器只有 L2正则化方法，直接在构建 optimizer 的时候，传入`weight_decay`参数即可。

**【注意事项】**

* PaddlePaddle 的 optimizer 中支持 L1Decat/L2Decay。
* PyTorch 的 optimizer 支持不同参数列表的学习率分别设置，params 传入字典即可，而 PaddlePaddle 的2.1.0版本目前尚未支持这种行为，可以通过设置`ParamAttr`的`learning_rate`参数，来确定相对学习率倍数。PaddlePaddle 的2.2.0版本中虽然实现该功能，但是模型收敛速度较慢，不建议使用。[优化器收敛速度慢](https://github.com/PaddlePaddle/Paddle/issues/36915)

**【实战】**

本部分对齐建议对照[PaddlePaddle 正则化 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/regularizer/L2Decay_cn.html)与参考代码的优化器实现进行对齐，用之后的反向对齐统一验证该模块的正确性。

<a name="3.8"></a>
### 3.8 反向对齐

**【基本流程】**

此处可以通过 numpy 生成假的数据和 label（推荐），也可以准备固定的真实数据。具体流程如下。

1. 检查两个代码的训练超参数全部一致，如优化器及其超参数、学习率、LayerNorm 中的 eps 等。
2. 将 PaddlePaddle 与 PyTorch 网络中涉及的所有随机操作全部关闭，如 dropout、drop_path 等，推荐将模型设置为 eval 模式（`model.eval()`）
3. 加载相同的 weight dict（可以通过 PyTorch 来存储随机的权重），将准备好的数据分别传入网络并迭代，观察二者 loss 是否一致（此处 batch-size 要一致，如果使用多个真实数据，要保证传入网络的顺序一致）
4. 如果经过2轮以上，loss 均可以对齐，则基本可以认为反向对齐。


**【注意事项】**

* 如果第一轮 loss 就没有对齐，则需要仔细排查一下模型前向部分。
* 如果第二轮开始，loss 开始无法对齐，则首先需要排查下超参数的差异，没问题的话，在`loss.backward()`方法之后，使用`tensor.grad`获取梯度值，二分的方法查找 diff，定位出 PaddlePaddle 与 PyTorch 梯度无法对齐的 API 或者操作，然后进一步验证并反馈。

梯度的打印方法示例代码如下所示，注释掉的内容即为打印网络中所有参数的梯度 shape。

```python
    # 代码地址：https://github.com/JunnYu/BERT-SST2-Prod/blob/2c372656bb1b077b0073c50161771d9fa9d8de5a/pipeline/Step4/test_bp.py#L12
    def pd_train_some_iters(model,
                        criterion,
                        optimizer,
                        fake_data,
                        fake_label,
                        max_iter=2):
        model = PDBertForSequenceClassification.from_pretrained("bert-base-uncased", num_classes=2)
        classifier_weights = paddle.load("../classifier_weights/paddle_classifier_weights.bin")
        model.load_dict(classifier_weights)
        model.eval()
        criterion = paddle.nn.CrossEntropy()
        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        optimizer = paddle.optimizer.AdamW(learning_rate=3e-5, parameters=model.parameters(),
            weight_decay=1e-2,
            epsilon=1e-6,
            apply_decay_param_fun=lambda x: x in decay_params)
        loss_list = []
        for idx in range(max_iter):
            input_ids = paddle.to_tensor(fake_data)
            labels = paddle.to_tensor(fake_label)

            output = model(input_ids)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss_list.append(loss)
        return loss_list
```




**【实战】**

本部分可以参考文档：[反向对齐操作文档](https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/Step4/README.md#%E5%8F%8D%E5%90%91%E5%AF%B9%E9%BD%90%E6%93%8D%E4%BD%9C%E6%96%B9%E6%B3%95)。

**【验收】**

对于待复现的项目，反向对齐验收流程如下。

1. 输入：fake data & label
2. 输出：
    * PaddlePaddle/PyTorch：dict，key 为 tensor 的 name（自定义），value 为具体 loss 的值。最后将 dict 使用 reprod_log 保存到各自的文件中，建议命名为`bp_align_paddle.npy`和`bp_align_torch.npy`。
3. 自测：使用 reprod_log 加载2个文件，使用 report 功能，记录结果到日志文件中，建议命名为`bp_align_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。
4. 提交内容：将`bp_align_paddle.npy`、`bp_align_torch.npy`与`bp_align_diff_log.txt`文件备份到`3.1节验收环节`新建的文件夹中，后续的输出结果和自查日志也放在该文件夹中，一并打包上传即可。
5. 注意：
    * loss 需要保存至少2轮以上。
    * 在迭代的过程中，需要保证模型的 batch size 等超参数完全相同
    * 在迭代的过程中，需要设置`model.eval()`，使用固定的假数据，同时加载相同权重的预训练模型。

<a name="3.9"></a>
### 3.9 训练集数据读取对齐

**【基本流程】**

该部分内容与3.2节内容基本一致，参考 PyTorch 的代码，实现训练集数据读取与预处理模块即可。

**【注意事项】**

该部分内容，可以参考3.8节的自测方法，将输入的`fake data & label`替换为训练的 dataloader，但是需要注意的是：
* 在使用 train dataloader 的时候，建议设置 random seed，对于 PyTorch 来说

```python
#initialize random seed
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)
```

对于 PaddlePaddle 来说

```python
paddle.seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)
```


<a name="3.10"></a>
### 3.10 网络初始化对齐

**【基本流程】**

* 下面给出了部分初始化 API 的映射表。

|PaddlePaddle API | PyTorch API |
|---|---|
| paddle.nn.initializer.KaimingNormal | torch.nn.init.kaiming_normal_ |
| paddle.nn.initializer.KaimingUniform | torch.nn.init.kaiming_uniform_ |
| paddle.nn.initializer.XavierNormal | torch.nn.init.xavier_normal_ |
| paddle.nn.initializer.XavierUniform | torch.nn.init.xavier_uniform_ |

**【注意事项】**

* 更多初始化 API 可以参考[PyTorch 初始化 API 文档](https://pytorch.org/docs/stable/nn.init.html)以及[PaddlePaddle 初始化 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Overview_cn.html#chushihuaxiangguan)。

**【实战】**

本部分对齐建议对照[PaddlePaddle 初始化 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Overview_cn.html#chushihuaxiangguan)与参考代码的初始化实现对齐。

<a name="3.11"></a>
### 3.11 模型训练对齐

**【基本流程】**

完成前面的步骤之后，就可以开始全量数据的训练对齐任务了。按照下面的步骤进行训练对齐。

1. 准备 train/eval data, loader, model
2. 对 model 按照论文所述进行初始化(如果论文中提到加载了预训练模型，则按需加载 pretrained model)
3. 加载配置，开始训练，迭代得到最终模型与评估指标，将评估指标使用 reprod_log 保存到文件中。
4. 将 PaddlePaddle 提供的参考指标使用 reprod_log 提交到另一个文件中。
5. 使用 reprod_log 排查 diff，小于阈值，即可完成自测。

**【注意事项】**

* 【强烈】建议先做完反向对齐之后再进行模型训练对齐，二者之间的不确定量包括：数据集、PaddlePaddle 与参考代码在模型 training mode 下的区别，初始化参数。
* 在训练对齐过程中，受到较多随机量的影响，精度有少量 diff 是正常的，以 SST-2数据集的分类为例，diff 在0.15%以内可以认为是正常的，这里可以根据不同的任务，适当调整对齐检查的阈值(`ReprodDiffHelper.report`函数中的`diff_threshold`参数)。
* 训练过程中的波动是正常的，如果最终收敛结果不一致，可以
    * 仔细排查 Dropout、BatchNorm 以及其他组网模块及超参是否无误。
    * 基于参考代码随机生成一份预训练模型，转化为 PaddlePaddle 的模型，并使用 PaddlePaddle 加载训练，对比二者的收敛曲线与最终结果，排查初始化影响。
    * 使用参考代码的 Dataloader 生成的数据，进行模型训练，排查 train dataloader 的影响。

**【实战】**

本部分可以参考文档：[训练对齐操作文档](https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/Step5/README.md)。

**【验收】**

对于待复现的项目，训练对齐验收流程如下。

1. 输入：train/eval dataloader, model
2. 输出：
    * PaddlePaddle：dict，key 为保存值的 name（自定义），value 为具体评估指标的值。最后将 dict 使用 reprod_log 保存到文件中，建议命名为`train_align_paddle.npy`。
    * benchmark：dict，key 为保存值的 name（自定义），value 为论文复现赛的评估指标要求的值。最后将 dict 使用 reprod_log 保存到文件中，建议命名为`train_align_benchmark.npy`。
3. 自测：使用 reprod_log 加载2个文件，使用 report 功能，记录结果到日志文件中，建议命名为`train_align_diff_log.txt`，观察 diff，二者 diff 小于特定的阈值即可。
4. 提交内容：将`train_align_paddle.npy`、`train_align_benchmark.npy`与`train_align_diff_log.txt`文件备份到`3.1节验收环节`新建的文件夹中，最终一并打包上传即可。

<a name="3.12"></a>
### 3.12 单机多卡训练

如果希望使用单机多卡提升训练效率，可以从以下几个过程对代码进行修改。

#### 3.12.1 数据读取

对于 PaddlePaddle 来说，多卡数据读取这块主要的变化在 sampler

对于单机单卡，sampler 实现方式如下所示。

```python
train_sampler = paddle.io.RandomSampler(dataset)
train_batch_sampler = paddle.io.BatchSampler(
    sampler=train_sampler, batch_size=args.batch_size)
```

对于单机多卡任务，sampler 实现方式如下所示。

```python
train_batch_sampler = paddle.io.DistributedBatchSampler(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
```

注意：在这种情况下，单机多卡的代码仍然能够以单机单卡的方式运行，因此建议以这种 sampler 方式进行论文复现。


#### 3.12.2 多卡模型初始化

如果以多卡的方式运行，需要初始化并行训练环境，代码如下所示。

```python
if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
```

在模型组网并初始化参数之后，需要使用`paddle.DataParallel()`对模型进行封装，使得模型可以通过数据并行的模式被执行。代码如下所示。

```python
if paddle.distributed.get_world_size() > 1:
    model = paddle.DataParallel(model)
```


#### 3.12.3 模型保存、日志保存等其他模块

以模型保存为例，我们只需要在0号卡上保存即可，否则多个 trainer 同时保存的话，可能会造成写冲突，导致最终保存的模型不可用。


#### 3.12.4 程序启动方式

对于单机单卡，启动脚本如下所示。[单机单卡](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/examples/benchmark/glue)

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/ \
    --device gpu \
    --use_amp False
```


对于单机多卡（示例中为4卡训练），启动脚本如下所示。

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0,1,2,3" run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/ \
    --device gpu \
    --use_amp False
```

注意：这里8卡训练时，虽然单卡的 batch size 没有变化(32)，但是总卡的 batch size 相当于是单卡的8倍，因此学习率也设置为了单卡时的8倍。


**【实战】**

本部分可以参考 paddlenlp 库中的例子：[单机多卡训练](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/bert)。

<a name="4"></a>
## 4. 论文复现注意事项与 FAQ

本部分主要总结大家在论文复现赛过程中遇到的问题，如果本章内容没有能够解决你的问题，欢迎给该文档提出优化建议或者给 Paddle 提[ISSUE](https://github.com/PaddlePaddle/Paddle/issues/new/choose)。

<a name="4.0"></a>
### 4.0 通用注意事项

* 需要仔细对照 PaddlePaddle 与参考代码的优化器参数实现，确保优化器参数严格对齐。
* 如果遇到一些 Paddle 不支持的 API 操作，可以尝试使用替代实现进行复现。如下面的 PyTorch 代码，PaddlePaddle 中可以通过 slice + concat API 的组合形式进行功能实现。同时，对于这个问题，建议优先给 Paddle 提[ISSUE](https://github.com/PaddlePaddle/Paddle/issues/new/choose)，列出 Paddle 不支持的实现，开发人员会根据优先级进行开发。

```python
torch.stack([
    per_locations[:, 0] - per_box_regression[:, 0],
    per_locations[:, 1] - per_box_regression[:, 1],
    per_locations[:, 0] + per_box_regression[:, 2],
    per_locations[:, 1] + per_box_regression[:, 3],
], dim=1)
```
* 如果遇到 Paddle 不包含的 OP 或者 API，比如(1) 如果是某些算法实现存在调用了外部 OP，而且 Paddle 也不包含该 OP 实现；(2) 其他框架存在的 API 或者 OP，但是 Paddle 中没有这些 OP。此时：
    * 对于 Paddle 资深用户来说，可以尝试使用 Paddle 的自定义算子功能，存在一定的代码开发量。
    * 对于初学者来说，可以给 Paddle 提[ISSUE](https://github.com/PaddlePaddle/Paddle/issues/new/choose)，列出 Paddle 不支持的实现，Paddle 开发人员会根据优先级进行实现。
* PaddlePaddle 与 PyTorch 对于不同名称的 API，实现的功能可能是相同的，复现的时候注意，比如[paddle.optimizer.lr.StepDecay](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/StepDecay_cn.html#stepdecay)与[torch.optim.lr_scheduler.StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR) 。
* 对于 PaddlePaddle 来说，通过`paddle.set_device`函数（全局）来确定模型结构是运行在什么设备上，对于 torch 来说，是通过`model.to("device")` （局部）来确定模型结构的运行设备，这块在复现的时候需要注意。


<a name="4.1"></a>
### 4.1 模型结构对齐

#### 4.1.1 API
* 对于 `paddle.nn.Linear` 层的 weight 参数，PaddlePaddle 与 PyTorch 的保存方式不同，在转换时需要进行转置，示例代码可以参考[BERT 权重转换脚本](https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/weights/torch2paddle.py)。
* `torch.masked_fill`函数的功能目前可以使用`paddle.where`进行实现，可以参考：[链接](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/faq/train_cn.html#paddletorch-masked-fillapi)。
* `pack_padded_sequence`和`pad_packed_sequence`这两个 API 目前 PaddlePaddle 中没有实现，可以直接在 RNN 或者 LSTM 的输入中传入`sequence_length`来实现等价的功能。


#### 4.1.2 权重转换

* 在权重转换的时候，不能只关注参数的名称，比如说有些`paddle.nn.Linear`层，但是定义的变量名称为`conv`，这种也是需要进行权重转置的。
* 权重转换时，建议同时打印 Paddle 和 PyTorch 对应权重的 shape，以防止名称相似但是 shape 不同的参数权重转换报错。

#### 4.1.3 模型组网正确性验证

* 在论文复现的过程中，可能会遇到一些经典的模型结构，比如 Transformer 等，Paddle 官方也提供了 Transformer 的实现，但是这里建议自己根据 PyTorch 代码重新实现一遍，一方面是对整体的模型结构更加熟悉，另一方面也保证模型结构和权重完全对齐。
* 在复杂的网络结构中，如果前向结果对不齐，可以按照模块排查问题，比如依次获取 embedding、transformer-block、mlm-head 输出等，看下问题具体出现在哪个子模块，再进到子模块详细排查。
* 网络结构对齐后，尽量使用训练好的预训练模型和真实的数据进行前向 diff 计算，这样更准确。

<a name="4.2"></a>
### 4.2 验证/测试集数据读取对齐

* 需要仔细排查数据预处理，不仅包含的预处理方法相同，也需要保证预处理的流程相同，比如先 padding 策略不同和截断策略的不同会导致得到最终的结果是不同的。

<a name="4.3"></a>
### 4.3 评估指标对齐

* 真实数据评估时，需要注意评估时 `paddle.io.DataLoader` 的 ``drop_last`` 参数是否打开(文档[链接](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader))，复现代码需要与参考代码保持一致，否则最后不够 batch-size 的数据的评估会有 diff。
* 在识别或者检索过程中，为了加速评估过程，往往会将评估函数由 CPU 实现改为 GPU 实现，由此会带来评估函数输出的不一致。这是由于 sort 函数对于相同值的排序结果不同带来的。在复现的过程中，如果可以接受轻微的指标不稳定，可以使用 PaddlePaddle 的 sort 函数，如果对于指标非常敏感，同时对速度性能要求很高，可以给 PaddlePaddle 提[ISSUE](https://github.com/PaddlePaddle/Paddle/issues/new/choose)，由研发人员高优开发。


<a name="4.4"></a>
### 4.4 损失函数对齐

* 部分算法的损失函数中会用到 bool 索引，这时候可以使用[paddle.where](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/where_cn.html#where) 代替。
* `paddle.nn.CrossEntropyLoss` 默认是在最后一维(axis=-1)计算损失函数，而 `torch.nn.CrossEntropyLoss` 是在 axis=1的地方计算损失函数，因此如果输入的维度大于2，这里需要保证计算的维(axis)相同，否则可能会出错。
* 在生成模型中会遇到梯度损失，需要对模型中的算子求二次梯度，目前`MaxPooling`暂时不支持二次梯度，如果复现的过程中遇到了需要对`MaxPooling`求二次梯度的情况，可以和 Paddle 官方开发同学反馈，进一步确认解决方案。
* 在保存损失函数值的时候，注意要使用`paddle.no_grad`，或者仅仅保存转换成 numpy 的数组，避免损失没有析构导致内存泄漏问题。

```python
# 错误示范
loss = celoss(pred, label)
avg_loss += loss
# 正确示范1
loss = celoss(pred, label)
avg_loss += loss.numpy()
# 正确示范2
loss = celoss(pred, label)
with paddle.no_grad()
    avg_loss += loss
```

<a name="4.5"></a>
### 4.5 优化器对齐

* Paddle 目前支持在 ``optimizer`` 中通过设置 ``params_groups`` 的方式设置不同参数的更新方式，可以参考[代码示例](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/optimizer/optimizer.py#L107) 。
* 有些模型训练时，会使用梯度累加策略，即累加到一定 step 数量之后才进行参数更新，这时在实现上需要注意对齐。
* 在某些任务中，比如说深度学习可视化、可解释性等任务中，一般只要求模型前向过程，不需要训练，此时优化器、学习率等用于模型训练的模块对于该类论文复现是不需要的。
* 在文本分类领域，大多数 Transformer 模型都采用了 AdamW 优化器，并且会设置 weigh decay，同时部分参数设置为 no weight decay，例如位置编码的参数通常设置为 no weight decay，no weight decay 参数设置不正确，最终会有明显的精度损失，需要特别注意。一般可以通过分析模型权重来发现该问题，分别计算官方模型和复现模型每层参数权重的平均值、方差，对每一层依次对比，有显著差异的层可能存在问题，因为在 weight decay 的作用下，参数权重数值会相对较小，而未正确设置 no weight decay，则会造成该层参数权重数值异常偏小。


<a name="4.6"></a>
### 4.6 学习率对齐

* PaddlePaddle 中参数的学习率受到优化器学习率和`ParamAttr`中设置的学习率影响，因此跟踪学习率需要将二者结合进行跟踪。
* 对于复现代码和参考代码，学习率在整个训练过程中在相同的轮数相同的 iter 下应该保持一致，可以通过`reprod_log`工具、打印学习率值或者可视化二者学习率的 log 来查看 diff。
* 有些网络的学习率策略比较细致，比如带 warmup 的学习率策略，这里需要保证起始学习率等参数都完全一致。


<a name="4.7"></a>
### 4.7 正则化策略对齐

* 在如 Transformer 或者少部分 CNN 模型中，存在一些参数不做正则化(正则化系数为0)的情况。这里需要找到这些参数并对齐取消实施正则化策略，可以参考[这里](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.3/ppcls/arch/backbone/model_zoo/resnest.py#L72)，对特定参数进行修改。

<a name="4.8"></a>
### 4.8 反向对齐

* 反向对齐时，如果第二轮开始，loss 开始无法对齐，则首先需要排查下超参数的差异，没问题的话，在`loss.backward()`方法之后，使用`tensor.grad`获取梯度值，二分的方法查找 diff，定位出 PaddlePaddle 与 PyTorch 梯度无法对齐的 API 或者操作，然后进一步验证。第3章中给出了获取所有参数的梯度方法，如果只希望打印特定参数的梯度，可以用下面的方式。


```python
import paddle

def print_hook_fn(grad):
    print(grad)

x = paddle.to_tensor([0., 1., 2., 3.], stop_gradient=False)
h = x.register_hook(print_hook_fn)
w = x * 4
w.backward()
# backward之后会输出下面的内容
#     Tensor(shape=[4], dtype=float32, place=CPUPlace, stop_gradient=False,
#            [4., 4., 4., 4.])
```


<a name="4.9"></a>
### 4.9 训练集数据读取对齐

#### 4.9.1 API

* 在前向过程中，如果数据预处理过程运行出错，请先将 ``paddle.io.DataLoader`` 的 ``num_workers`` 参数设为0，然后根据单个进程下的报错日志定位出具体的 bug。

#### 4.9.2 数据预处理


* 如果数据处理过程中涉及到随机数生成，建议固定 seed (`np.random.seed(0)`, `random.seed(0)`)，查看复现代码和参考代码处理后的数据是否有 diff。
* 对文本进行 tokenizer 处理时，需要确定文本的截断策略，padding 策略。

<a name="4.10"></a>
### 4.10 网络初始化对齐

* 对于不同的深度学习框架，网络初始化在大多情况下，即使值的分布完全一致，也无法保证值完全一致，这里也是论文复现中不确定性比较大的地方。如果十分怀疑初始化导致的问题，建议将参考的初始化权重转成 paddle 模型，加载该初始化模型训练，看下收敛精度。
* CNN 对于模型初始化相对来说没有那么敏感，在迭代轮数与数据集足够的情况下，最终精度指标基本接近；而 transformer 系列模型对于初始化比较敏感，在 transformer 系列模型训练对齐过程中，建议对这一块进行重点检查。


<a name="4.11"></a>
### 4.11 模型训练对齐

#### 4.11.1 训练对齐通用问题

* 有条件的话，复现工作之前最好先基于官方代码完成训练，保证与官方指标能够对齐，并且将训练策略和训练过程中的关键指标记录保存下来，比如每个 epoch 的学习率、Train Loss、Eval Loss、Eval Acc 等，在复现网络的训练过程中，将关键指标保存下来，这样可以将两次训练中关键指标的变化曲线绘制出来，能够很方便的进行对比。
* 训练过程中可以对 loss 或者 acc 进行可视化，和竞品 loss 或者 acc 进行直观的对比；如果训练较大的数据集，1次完整训练的成本比较高，此时可以隔一段时间查看一下，如果精度差异比较大，建议先停掉实验，排查原因。
* 如果训练的过程中出 nan，一般是因为除0或者 log0的情况， 可以着重看下几个部分：
    * 如果有预训练模型的话，可以确认下是否加载正确
    * 模型结构中计算 loss 的部分是否有考虑到正样本为0的情况
    * 也可能是某个 API 的数值越界导致的，可以测试较小的输入是否还会出现 nan。
* 如果训练过程中如果出现不收敛的情况，可以
    * 简化网络和数据，实验是否收敛；
    * 如果是基于原有实现进行改动，可以尝试控制变量法，每次做一个改动，逐个排查；
    * 检查学习率是否过大、优化器设置是否合理，排查下 weight decay 是否设置正确；
    * 保存不同 step 之间的模型参数，观察模型参数是否更新。
