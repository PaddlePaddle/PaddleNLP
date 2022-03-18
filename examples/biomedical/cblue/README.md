# 使用医疗领域预训练模型Fine-tune完成中文医疗语言理解任务

近年来，预训练语言模型（Pre-trained Language Model，PLM）逐渐成为自然语言处理（Natural Language Processing，NLP）的主流方法。这类模型可以利用大规模的未标注语料进行训练，得到的模型在下游NLP任务上效果明显提升，在通用领域和特定领域均有广泛应用。在医疗领域，早期的做法是在预先训练好的通用语言模型上进行Fine-tune。后来的研究发现直接使用医疗相关语料学习到的预训练语言模型在医疗文本任务上的效果更好，采用的模型结构也从早期的BERT演变为更新的RoBERTa、ALBERT和ELECTRA。

本示例展示了中文医疗预训练模型eHealth（[Building Chinese Biomedical Language Models via Multi-Level Text Discrimination](https://arxiv.org/abs/2110.07244)）如何Fine-tune完成中文医疗语言理解任务。

## 模型介绍

本项目针对中文医疗语言理解任务，开源了中文医疗预训练模型eHealth（简写`chinese-ehealth`）。eHealth使用了医患对话、科普文章、病历档案、临床病理学教材等脱敏中文语料进行预训练，通过预训练任务设计来学习词级别和句级别的文本信息。该模型的整体结构与ELECTRA相似，包括生成器和判别器两部分。 而Fine-tune过程只用到了判别器模块，由12层Transformer网络组成。

## 数据集介绍

本项目使用了中文医学语言理解测评（[Chinese Biomedical Language Understanding Evaluation，CBLUE](https://github.com/CBLUEbenchmark/CBLUE)）数据集，其包括医学文本信息抽取（实体识别、关系抽取）、医学术语归一化、医学文本分类、医学句子关系判定和医学问答共5大类任务8个子任务。

* CMeEE：中文医学命名实体识别
* CMeIE：中文医学文本实体关系抽取
* CHIP-CDN：临床术语标准化任务
* CHIP-CTC：临床试验筛选标准短文本分类
* CHIP-STS：平安医疗科技疾病问答迁移学习
* KUAKE-QIC：医疗搜索检索词意图分类
* KUAKE-QTR：医疗搜索查询词-页面标题相关性
* KUAKE-QQR：医疗搜索查询词-查询词相关性

更多信息可参考CBLUE的[github](https://github.com/CBLUEbenchmark/CBLUE/blob/main/README_ZH.md)。其中对于临床术语标准化任务（CHIP-CDN），我们按照eHealth中的方法通过检索将原多分类任务转换为了二分类任务，即给定一诊断原词和一诊断标准词，要求判定后者是否是前者对应的诊断标准词。本项目提供了检索处理后的CHIP-CDN数据集（简写`CHIP-CDN-2C`），且构建了基于该数据集的example代码。

## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
cblue/
├── README.md # 使用说明
├── train_classification.py # 分类任务训练评估脚本
├── train_ner.py # 实体识别任务训练评估脚本
└── train_spo.py # 关系抽取任务训练评估脚本
```

### 模型训练

我们按照任务类别划分，同时提供了8个任务的样例代码。可以运行下边的命令，在训练集上进行训练，并在开发集上进行验证。

```shell
$ unset CUDA_VISIBLE_DEVICES
$ python -m paddle.distributed.launch --gpus "0,1,2,3" train.py --dataset CHIP-CDN-2C --batch_size 256 --max_seq_length 32 --learning_rate 3e-5 --epochs 16
```

### 训练参数设置（Training setup）及结果

| Task      | epochs | batch_size | learning_rate | max_seq_length | results |
| --------- | :----: | :--------: | :-----------: | :------------: | :-----: |
| CHIP-STS  |   16   |     32     |      1e-4     |       96       | 0.88550 |
| CHIP-CTC  |   16   |     32     |      3e-5     |      160       | 0.82790 |
| CHIP-CDN  |   16   |    256     |      3e-5     |       32       | 0.76979 |
| KUAKE-QQR |   16   |     32     |      6e-5     |       64       | 0.82364 |
| KUAKE-QTR |   12   |     32     |      6e-5     |       64       | 0.69653 |
| KUAKE-QIC |    4   |     32     |      6e-5     |      128       | 0.81176 |


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

[1] CBLUE: A Chinese Biomedical Language Understanding Evaluation Benchmark [pdf](https://arxiv.org/abs/2106.08087) [git](https://github.com/CBLUEbenchmark/CBLUE) [web](https://tianchi.aliyun.com/specials/promotion/2021chinesemedicalnlpleaderboardchallenge)
