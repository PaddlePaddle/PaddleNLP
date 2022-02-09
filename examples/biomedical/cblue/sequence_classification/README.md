# 使用医疗领域预训练模型Fine-tune完成中文医疗文本分类任务

近年来，预训练语言模型（Pre-trained Language Model，PLM）逐渐成为自然语言处理（Natural Language Processing，NLP）的主流方法。这类模型可以利用大规模的未标注语料进行训练，得到的模型在下游NLP任务上效果明显提升，在通用领域和特定领域均有广泛应用。在医疗领域，早期的做法是在预先训练好的通用语言模型上进行Fine-tune。后来的研究发现直接使用医疗相关语料学习到的预训练语言模型在医疗文本任务上的效果更好，采用的模型结构也从早期的BERT演变为更新的RoBERTa、ALBERT和ELECTRA。

本示例展示了中文医疗预训练模型eHealth（[Building Chinese Biomedical Language Models via Multi-Level Text Discrimination](https://arxiv.org/abs/2110.07244)）如何Fine-tune完成中文医疗文本分类任务。

## 模型介绍

本项目针对中文医疗文本分类任务，开源了中文医疗预训练模型eHealth（简写`chinese-ehealth`）。eHealth（[Building Chinese Biomedical Language Models via Multi-Level Text Discrimination](https://arxiv.org/abs/2110.07244)）使用了医患对话、科普文章、病历档案、临床病理学教材等脱敏中文语料进行预训练，通过预训练任务设计来学习词级别和句级别的文本信息。该模型的整体结构与ELECTRA相似，包括生成器和判别器两部分。 而Fine-tune过程只用到了判别器模块，由12层Transformer网络组成。

## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
sequence_classification/
├── README.md # 使用说明
└── train.py # 训练评估脚本
```

### 模型训练

我们以中文医疗文本数据集CBLUE中的文本分类数据集为示例数据集，包括：

* CHIP-CDN：给定病历档案，预测其中包含的规范化诊断实体。本项目使用了检索后重新构建的二分类数据集，给定病历档案和规范化诊断实体，预测前者是否包含后者（简写`CHIP-CDN-2C`）。
* CHIP-CTC：给定医疗文本描述，按照中国临床筛选标准进行分类。
* CHIP-STS：给定两个涉及5种不同疾病的句子，预测二者语义是否相似。
* KUAKE-QIC：给定医疗问句，对患者咨询目的进行分类。
* KUAKE-QTR：给定医疗问句和文章标题，预测二者内容是否一致。
* KUAKE-QQR：给定两个医疗问句，预测二者描述内容是否一致。

可以运行下边的命令，在训练集上进行训练，并在开发集上进行验证。
```shell
$ unset CUDA_VISIBLE_DEVICES
$ python -m paddle.distributed.launch --gpus "0" train.py --dataset CHIP-CDN-2C --batch_size 256 --max_seq_length 32 --weight_decay 0.01 --warmup_proportion 0.1
```

可支持配置的参数：

* `save_dir`：可选，保存训练模型的目录；默认保存在当前目录checkpoints文件夹下。
* `dataset`：可选，CHIP-CDN-2C CHIP-CTC CHIP-STS KUAKE-QIC KUAKE-QTR KUAKE-QQR，默认为KUAKE-QIC数据集。
* `max_seq_length`：可选，ELECTRA模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune的最大学习率；默认为6e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.01。
* `epochs`: 训练轮次，默认为3。
* `valid_steps`: evaluate的间隔steps数，默认100。
* `save_steps`: 保存checkpoints的间隔steps数，默认100。
* `logging_steps`: 日志打印的间隔steps数，默认10。
* `warmup_proption`：可选，学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.1。
* `init_from_ckpt`：可选，模型参数路径，热启动模型训练；默认为None。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。
* `use_amp`: 是否使用混合精度训练，默认为False。
* `use_ema`: 是否使用Exponential Moving Average预测，默认为False。

### 依赖安装

```shell
pip install xlrd==1.2.0
```
