# 二分类/多分类任务指南

**目录**

- [1. 二分类/多分类简介](#二分类/多分类简介)
- [2. 快速开始](#快速开始)
    - [2.1 运行环境](#运行环境)
    - [2.2 代码结构](#代码结构)
    - [2.3 数据准备](#数据准备)
    - [2.4 模型训练](#模型训练)
    - [2.5 模型预测](#模型预测)
    - [2.6 模型效果](#模型效果)


<a name="二分类/多分类简介"></a>

## 1. 二分类/多分类简介

本项目提供通用场景下**基于预训练模型微调的二分类/多分类端到端应用方案**，打通数据标注-模型训练-模型调优-模型压缩-预测部署全流程，有效缩短开发周期，降低AI开发落地门槛。

二分类/多分类数据集的标签集含有两个或两个以上的类别，所有输入句子/文本有且只有一个标签。在文本多分类场景中，我们需要预测**输入句子/文本最可能来自 `n` 个标签类别中的哪一个类别**。在本项目中二分类任务被视为多分类任务中标签集包含两个类别的情况，以下统一称为多分类任务。以下图为例，该新闻文本的最可能的标签为 `娱乐`。多分类任务在商品分类、网页标签、新闻分类、医疗文本分类等各种现实场景中具有广泛的适用性。

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/187588832-daa4294e-248e-4c69-9a4b-1cf8f25a2fcc.png width="550"/>
</div>
<br>

**方案亮点：**

- **效果领先🏃：** 使用在中文领域内模型效果和模型计算效率有突出效果的ERNIE 3.0 轻量级系列模型作为训练基座，ERNIE 3.0 轻量级系列提供多种尺寸的预训练模型满足不同需求，具有广泛成熟的实践应用性。
- **高效调优✊：** 文本分类应用依托[TrustAI](https://github.com/PaddlePaddle/TrustAI)可信增强能力和[数据增强API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)，提供模型分析模块助力开发者实现模型分析，并提供稀疏数据筛选、脏数据清洗、数据增强等多种解决方案。
- **简单易用👶：** 开发者**无需机器学习背景知识**，仅需提供指定格式的标注分类数据，一行命令即可开启文本分类训练，轻松完成上线部署，不再让技术成为文本分类的门槛。

**更多选择：**

对于大多数多分类任务，我们推荐使用预训练模型微调作为首选的文本分类方案，多分类项目中还提供通用文本分类(UTC)和语义索引的两种方案满足不同开发者需求，更多技术细节请参见[文本分类技术特色介绍](../README.md)。

- 【零样本、小样本场景】 👉 [通用文本分类(UTC)方案](../../zero_shot_text_classification)

- 【标签类别不固定、标签类别众多】 👉 [语义索引分类方案](./retrieval_based)


<a name="快速开始"></a>

## 2. 快速开始

接下来我们将以CBLUE公开数据集KUAKE-QIC任务为示例，演示多分类全流程方案使用。下载数据集：

```shell
wget https://paddlenlp.bj.bcebos.com/datasets/KUAKE_QIC.tar.gz
tar -zxvf KUAKE_QIC.tar.gz
mv KUAKE_QIC data
rm -rf KUAKE_QIC.tar.gz
```

<div align="center">
    <img width="900" alt="image" src="https://user-images.githubusercontent.com/63761690/187828356-e2f4f627-f5fe-4c83-8879-ed6951f7511e.png">
</div>
<div align="center">
    <font size ="2">
    多分类数据标注-模型训练-模型分析-模型压缩-预测部署流程图
     </font>
</div>

<a name="运行环境"></a>

### 2.1 运行环境

- python >= 3.6
- paddlepaddle >= 2.3
- paddlenlp >= 2.5.1
- scikit-learn >= 1.0.2

**安装PaddlePaddle：**

 环境中paddlepaddle-gpu或paddlepaddle版本应大于或等于2.3, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的PaddlePaddle下载命令。


**安装PaddleNLP：**

```shell
pip install --upgrade paddlenlp
```


**安装sklearn：**
```shell
pip install scikit-learn
```

<a name="代码结构"></a>

### 2.2 代码结构

```text
multi_class/
├── few-shot # 小样本学习方案
├── retrieval_based # 语义索引方案
├── analysis # 分析模块
├── deploy # 部署
│   ├── simple_serving # SimpleServing服务化部署
│   └── triton_serving # Triton服务化部署
├── train.py # 训练、评估、裁剪脚本
├── utils.py # 工具函数脚本
└── README.md # 多分类使用说明
```

<a name="数据准备"></a>

### 2.3 数据准备

训练需要准备指定格式的本地数据集,如果没有已标注的数据集，可以参考[文本分类任务doccano数据标注使用指南](../doccano.md)进行文本分类数据标注。指定格式本地数据集目录结构：

```text
data/
├── train.txt # 训练数据集文件
├── dev.txt # 开发数据集文件
└── label.txt # 分类标签文件
```

**训练、开发、测试数据集** 文件中文本与标签类别名用tab符`'\t'`分隔开，文本中避免出现tab符`'\t'`。

- train.txt/dev.txt/test.txt 文件格式：
```text
<文本>'\t'<标签>
<文本>'\t'<标签>
...
```

- train.txt/dev.txt/test.txt 文件样例：
```text
25岁已经感觉脸部松弛了怎么办	治疗方案
小孩的眉毛剪了会长吗？	其他
172的身高还能长高吗？	其他
冻疮用三金冻疮酊有效果么？	功效作用
...
```

**分类标签**

label.txt(分类标签文件)记录数据集中所有标签集合，每一行为一个标签名。
- label.txt 文件格式：
```text
<标签>
<标签>
...
```

- label.txt 文件样例：
```text
病情诊断
治疗方案
病因分析
指标解读
就医建议
...
```

<a name="模型训练"></a>

### 2.4 模型训练

我们推荐使用 Trainer API 对模型进行微调。只需输入模型、数据集等就可以使用 Trainer API 高效快速地进行预训练、微调和模型压缩等任务，可以一键启动多卡训练、混合精度训练、梯度累积、断点重启、日志显示等功能，Trainer API 还针对训练过程的通用训练配置做了封装，比如：优化器、学习率调度等。

#### 2.4.1 预训练模型微调


使用CPU/GPU训练，默认为GPU训练。使用CPU训练只需将设备参数配置改为`--device cpu`，可以使用`--device gpu:0`指定GPU卡号：
```shell
python train.py \
    --do_train \
    --do_eval \
    --do_export \
    --model_name_or_path ernie-3.0-tiny-medium-v2-zh \
    --output_dir checkpoint \
    --device gpu \
    --num_train_epochs 100 \
    --early_stopping True \
    --early_stopping_patience 5 \
    --learning_rate 3e-5 \
    --max_length 128 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32 \
    --metric_for_best_model accuracy \
    --load_best_model_at_end \
    --logging_steps 5 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1
```

如果在GPU环境中使用，可以指定`gpus`参数进行多卡分布式训练。使用多卡训练可以指定多个GPU卡号，例如 --gpus 0,1。如果设备只有一个GPU卡号默认为0，可使用`nvidia-smi`命令查看GPU使用情况:

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus 0,1 train.py \
    --do_train \
    --do_eval \
    --do_export \
    --model_name_or_path ernie-3.0-tiny-medium-v2-zh \
    --output_dir checkpoint \
    --device gpu \
    --num_train_epochs 100 \
    --early_stopping True \
    --early_stopping_patience 5 \
    --learning_rate 3e-5 \
    --max_length 128 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32 \
    --metric_for_best_model accuracy \
    --load_best_model_at_end \
    --logging_steps 5 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1
```

主要的配置的参数为：
- `do_train`: 是否进行训练。
- `do_eval`: 是否进行评估。
- `debug`: 与`do_eval`配合使用，是否开启debug模型，对每一个类别进行评估。
- `do_export`: 训练结束后是否导出静态图。
- `do_compress`: 训练结束后是否进行模型裁剪。
- `model_name_or_path`: 内置模型名，或者模型参数配置目录路径。默认为`ernie-3.0-tiny-medium-v2-zh`。
- `output_dir`: 模型参数、训练日志和静态图导出的保存目录。
- `device`: 使用的设备，默认为`gpu`。
- `num_train_epochs`: 训练轮次，使用早停法时可以选择100。
- `early_stopping`: 是否使用早停法，也即一定轮次后评估指标不再增长则停止训练。
- `early_stopping_patience`: 在设定的早停训练轮次内，模型在开发集上表现不再上升，训练终止；默认为4。
- `learning_rate`: 预训练语言模型参数基础学习率大小，将与learning rate scheduler产生的值相乘作为当前学习率。
- `max_length`: 最大句子长度，超过该长度的文本将被截断，不足的以Pad补全。提示文本不会被截断。
- `per_device_train_batch_size`: 每次训练每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。
- `per_device_eval_batch_size`: 每次评估每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。
- `data_dir`: 训练数据集路径，数据格式要求详见[数据准备](#数据准备)。
- `max_length`: 最大句子长度，超过该长度的文本将被截断，不足的以Pad补全。提示文本不会被截断。
- `train_path`: 训练集路径，默认为"./data/train.txt"。
- `dev_path`: 开发集集路径，默认为"./data/dev.txt"。
- `test_path`: 测试集路径，默认为"./data/dev.txt"。
- `label_path`: 标签路径，默认为"./data/label.txt"。
- `bad_case_path`: 错误样本保存路径，默认为"./data/bad_case.txt"。
- `width_mult_list`：裁剪宽度（multi head）保留的比例列表，表示对self_attention中的 `q`、`k`、`v` 以及 `ffn` 权重宽度的保留比例，保留比例乘以宽度（multi haed数量）应为整数；默认是None。
训练脚本支持所有`TraingArguments`的参数，更多参数介绍可参考[TrainingArguments 参数介绍](https://paddlenlp.readthedocs.io/zh/latest/trainer.html#trainingarguments)。

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存开发集上最佳模型在指定的 `output_dir` 中，保存模型文件结构如下所示：

```text
checkpoint/
├── export # 静态图模型
├── config.json # 模型配置文件
├── model_state.pdparams # 模型参数文件
├── tokenizer_config.json # 分词器配置文件
├── vocab.txt
└── special_tokens_map.json
```

**NOTE:**

* 英文训练任务推荐使用"ernie-3.0-tiny-mini-v2-en", "ernie-2.0-base-en"、"ernie-2.0-large-en"。
* 英文和中文以外语言的文本分类任务，推荐使用基于96种语言（涵盖法语、日语、韩语、德语、西班牙语等几乎所有常见语言）进行预训练的多语言预训练模型"ernie-m-base"、"ernie-m-large"，详情请参见[ERNIE-M论文](https://arxiv.org/pdf/2012.15674.pdf)。

#### 2.4.2 训练评估
训练后的模型我们可以开启`debug`模式，对每个类别分别进行评估，并打印错误预测样本保存在`bad_case.txt`。默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`:

```shell
python train.py \
    --do_eval \
    --debug True \
    --device gpu \
    --model_name_or_path checkpoint \
    --output_dir checkpoint \
    --per_device_eval_batch_size 32 \
    --max_length 128 \
    --test_path './data/dev.txt'
```

输出打印示例：

```text
[2023-02-14 12:35:03,470] [    INFO] - -----Evaluate model-------
[2023-02-14 12:35:03,471] [    INFO] - Dev dataset size: 1955
[2023-02-14 12:35:03,471] [    INFO] - Accuracy in dev dataset: 81.74%
[2023-02-14 12:35:03,471] [    INFO] - Macro average | precision: 77.39 | recall: 79.89 | F1 score 78.32
[2023-02-14 12:35:03,471] [    INFO] - Class name: 病情诊断
[2023-02-14 12:35:03,471] [    INFO] - Evaluation examples in dev dataset: 288(14.7%) | precision: 85.22 | recall: 86.11 | F1 score 85.66
[2023-02-14 12:35:03,471] [    INFO] - ----------------------------
[2023-02-14 12:35:03,471] [    INFO] - Class name: 治疗方案
[2023-02-14 12:35:03,471] [    INFO] - Evaluation examples in dev dataset: 676(34.6%) | precision: 91.72 | recall: 90.09 | F1 score 90.90
...
```

文本分类预测过程中常会遇到诸如"模型为什么会预测出错误的结果"，"如何提升模型的表现"等问题。[Analysis模块](./analysis) 提供了**可解释性分析、数据优化**等功能，旨在帮助开发者更好地分析文本分类模型预测结果和对模型效果进行优化。

#### 2.4.3 模型裁剪(可选)

如果有模型部署上线的需求，需要进一步压缩模型体积，可以使用 PaddleNLP 的 [压缩API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/compression.md), 一行命令即可启动模型裁剪。

使用裁剪功能需要安装 paddleslim：

```shell
pip install paddleslim >= 2.4.1
```

开始模型裁剪训练，默认为GPU训练，使用CPU训练只需将设备参数配置改为`--device "cpu"`：
```shell
python train.py \
    --do_compress \
    --device gpu \
    --data_dir data \
    --model_name_or_path checkpoint \
    --output_dir checkpoint/prune \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 1 \
    --max_length 128 \
    --logging_steps 5 \
    --save_steps 100 \
    --width_mult_list '3/4' '2/3' '1/2'
```
保存模型文件结构如下所示：

```text
checkpoint/prune/
├── width_mult_0.75
│   ├── pruned_model.pdiparams
│   ├── pruned_model.pdiparams.info
│   ├── pruned_model.pdmodel
│   ├── model_state.pdparams
│   └── config.json
└── ...
```
**NOTE:**

1. 目前支持的裁剪策略需要训练，训练时间视下游任务数据量而定，且和微调的训练时间是一个量级。 裁剪类似蒸馏过程，方便起见，可以直接使用微调时的超参。为了进一步提升精度，可以对 `per_device_train_batch_size`、`learning_rate`、`max_length` 等超参进行网格搜索（grid search）。

2. 模型裁剪主要用于推理部署，因此裁剪后的模型都是静态图模型，只可用于推理部署。

3. ERNIE Base、Medium、Mini、Micro、Nano的模型宽度（multi head数量）为12，ERNIE Xbase、Large 模型宽度（multi head数量）为16，保留比例`width_mult`乘以宽度（multi haed数量）应为整数。

<a name="模型预测"></a>

### 2.5 模型预测
我们推荐使用taskflow进行模型预测。
```
from paddlenlp import Taskflow

# 模型预测
cls = Taskflow("text_classification", task_path='checkpoint/export', is_static_model=True)
cls(["黑苦荞茶的功效与作用及食用方法","幼儿挑食的生理原因是"])
# [{'predictions': [{'label': '功效作用', 'score': 0.9683999621710758}], 'text': '黑苦荞茶的功效与作用及食用方法'}, {'predictions': [{'label': '病因分析', 'score': 0.5204789523701855}], 'text': '幼儿挑食的生理原因是'}]

# 裁剪模型预测
cls = Taskflow("text_classification", task_path='checkpoint/prune/width_mult_0.67', is_static_model=True)
cls(["黑苦荞茶的功效与作用及食用方法","幼儿挑食的生理原因是"])
# [{'predictions': [{'label': '功效作用', 'score': 0.964693000149321}], 'text': '黑苦荞茶的功效与作用及食用方法'}, {'predictions': [{'label': '病因分析', 'score': 0.4915921440237312}], 'text': '幼儿挑食的生理原因是'}]
```

#### 可配置参数说明
* `task_path`：自定义任务路径，默认为None。
* `is_static_model`：task_path中是否为静态图模型参数，默认为False。
* `max_length`：最长输入长度，包括所有标签的长度，默认为512。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
* `id2label`：标签映射字典，如果`task_path`中包含id2label.json或加载动态图参数无需定义。
* `precision`：选择模型精度，默认为`fp32`，可选有`fp16`和`fp32`。`fp16`推理速度更快。如果选择`fp16`，请先确保机器正确安装NVIDIA相关驱动和基础软件，**确保CUDA>=11.2，cuDNN>=8.1.1**，初次使用需按照提示安装相关依赖。其次，需要确保GPU设备的CUDA计算能力（CUDA Compute Capability）大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)。

在线服务化部署搭建请参考：
- 【简单易用】 👉 [Simple Serving部署指南](deploy/simple_serving)
- 【低时延】 👉 [Triton部署指南](deploy/triton_serving)。


<a name="模型效果"></a>

### 2.6 模型效果

PaddleNLP提供ERNIE 3.0 全系列轻量化模型，对于中文训练任务可以根据需求选择不同的预训练模型参数进行训练，我们评测了不同预训练模型在KUAKE-QIC任务的表现，测试配置如下：

1. 数据集：CBLUE数据集中医疗搜索检索词意图分类(KUAKE-QIC)任务开发集

2. 物理机环境

    系统: CentOS Linux release 7.7.1908 (Core)

    GPU: Tesla V100-SXM2-32GB

    CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz

    CUDA: 11.2

    cuDNN: 8.1.0

    Driver Version: 460.27.04

    内存: 630 GB

3. PaddlePaddle 版本：2.3.0

4. PaddleNLP 版本：2.3.1

5. 性能数据指标：latency。latency 测试方法：固定 batch size 为 32，GPU部署运行时间 total_time，计算 latency = total_time / total_samples

6. 精度评价指标：Accuracy

|  model_name  | 模型结构  |Accuracy(%)   | latency(ms) |
| -------------------------- | ------------ | ------------ | ------------ |
|ERNIE 1.0 Large Cw |24-layer, 1024-hidden, 20-heads|82.30| 5.62 |
|ERNIE 3.0 Base  |12-layer, 768-hidden, 12-heads|82.25| 2.07 |
|ERNIE 3.0 Medium| 6-layer, 768-hidden, 12-heads|81.79| 1.07|
|ERNIE 3.0 Mini |6-layer, 384-hidden, 12-heads|79.80| 0.38|
|ERNIE 3.0 Micro | 4-layer, 384-hidden, 12-heads|79.80| 0.26|
|ERNIE 3.0 Nano |4-layer, 312-hidden, 12-heads|78.57|0.22|
| ERNIE 3.0 Medium + 裁剪(保留比例3/4)|6-layer, 768-hidden, 9-heads| 81.79| 0.83   |
| ERNIE 3.0 Medium + 裁剪(保留比例2/3)|6-layer, 768-hidden, 8-heads| 81.07  | 0.79  |
| ERNIE 3.0 Medium + 裁剪(保留比例1/2)|6-layer, 768-hidden, 6-heads| 81.07 | 0.64  |
