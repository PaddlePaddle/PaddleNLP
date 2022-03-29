# 使用医疗领域预训练模型Fine-tune完成中文医疗语言理解任务

医疗领域存在大量的专业知识和医学术语，人类经过长时间的学习才能成为一名优秀的医生。那机器如何才能“读懂”医疗文献呢？尤其是面对电子病历、生物医疗文献中存在的大量非结构化、非标准化文本，计算机是无法直接使用、处理的。这就需要自然语言处理（Natural Language Processing，NLP）技术大展身手了。

近年来，预训练语言模型（Pre-trained Language Model，PLM）逐渐成为自然语言处理的主流方法。这类模型利用大规模的未标注语料进行训练，得到的模型在下游NLP任务上的效果有着明显提升，在通用领域和特定领域均有广泛应用。在医疗领域，早期的做法是在预先训练好的通用语言模型上进行 Fine-tune。后来的研究发现直接使用医疗相关语料学习到的预训练语言模型在医疗文本任务上的效果更好，采用的模型结构也从早期的BERT演变为更新的 RoBERTa、ALBERT和ELECTRA。

本示例展示了中文医疗预训练模型 ERNIE-Health（[Building Chinese Biomedical Language Models via Multi-Level Text Discrimination](https://arxiv.org/abs/2110.07244)）如何 Fine-tune完成中文医疗语言理解任务。

## 模型介绍

本项目针对中文医疗语言理解任务，开源了中文医疗预训练模型 ERNIE-Health（模型名称`ernie-health-chinese`）。ERNIE-Health模型依托于百度知识增强语义理解框架 ERNIE，以超越人类医学专家水平的成绩登顶中文医疗信息处理权威榜单 CBLUE 冠军, 验证了 ERNIE 在医疗行业应用的重要价值。

![CBLUERank](https://user-images.githubusercontent.com/25607475/160394225-04f75498-ce1a-4665-85f7-d495815eed51.png)

ERNIE-Health 依托百度文心 ERNIE 先进的知识增强预训练语言模型打造, 通过医疗知识增强技术进一步学习海量的医疗数据, 精准地掌握了专业的医学知识。ERNIE-Health 利用医疗实体掩码策略对专业术语等实体级知识学习, 学会了海量的医疗实体知识。同时，通过医疗问答匹配任务学习病患病状描述与医生专业治疗方案的对应关系，获得了医疗实体知识之间的内在联系。ERNIE-Health 共学习了 60 多万的医疗专业术语和 4000 多万的医疗专业问答数据，大幅提升了对医疗专业知识的理解和建模能力。此外，ERNIE-Health 还探索了多级语义判别预训练任务，提升了模型对医疗知识的学习效率。该模型的整体结构与 ELECTRA 相似，包括生成器和判别器两部分。 而 Fine-tune 过程只用到了判别器模块，由 12 层 Transformer 网络组成。

## 数据集介绍

本项目使用了中文医学语言理解测评（[Chinese Biomedical Language Understanding Evaluation，CBLUE](https://github.com/CBLUEbenchmark/CBLUE)）1.0 版本数据集，这是国内首个面向中文医疗文本处理的多任务榜单，涵盖了医学文本信息抽取（实体识别、关系抽取）、医学术语归一化、医学文本分类、医学句子关系判定和医学问答共5大类任务8个子任务。其数据来源分布广泛，包括医学教材、电子病历、临床试验公示以及互联网用户真实查询等。该榜单一经推出便受到了学界和业界的广泛关注，已逐渐发展成为检验AI系统中文医疗信息处理能力的“金标准”。

* CMeEE：中文医学命名实体识别
* CMeIE：中文医学文本实体关系抽取
* CHIP-CDN：临床术语标准化任务
* CHIP-CTC：临床试验筛选标准短文本分类
* CHIP-STS：平安医疗科技疾病问答迁移学习
* KUAKE-QIC：医疗搜索检索词意图分类
* KUAKE-QTR：医疗搜索查询词-页面标题相关性
* KUAKE-QQR：医疗搜索查询词-查询词相关性

更多信息可参考CBLUE的[github](https://github.com/CBLUEbenchmark/CBLUE/blob/main/README_ZH.md)。其中对于临床术语标准化任务（CHIP-CDN），我们按照 ERNIE-Health 中的方法通过检索将原多分类任务转换为了二分类任务，即给定一诊断原词和一诊断标准词，要求判定后者是否是前者对应的诊断标准词。本项目提供了检索处理后的 CHIP-CDN 数据集（简写`CHIP-CDN-2C`），且构建了基于该数据集的example代码。

## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
cblue/
├── README.md # 使用说明
├── train_classification.py # 分类任务训练评估脚本
├── train_ner.py # 实体识别任务训练评估脚本
├── train_spo.py # 关系抽取任务训练评估脚本
└── export_model.py #动态图参数导出静态图参数脚本
```

### 模型训练

我们按照任务类别划分，同时提供了8个任务的样例代码。可以运行下边的命令，在训练集上进行训练，并在开发集上进行验证。

**训练参数设置（Training setup）及结果**

| Task      | epochs | batch_size | learning_rate | max_seq_length |  metric  | results | results (fp16) |
| --------- | :----: | :--------: | :-----------: | :------------: | :------: | :-----: | :------------: |
| CHIP-STS  |    4   |     16     |      1e-4     |       96       | Macro-F1 | 0.88550 |    0.85649     |
| CHIP-CTC  |    4   |     32     |      6e-5     |      160       | Macro-F1 | 0.84136 |    0.83514     |
| CHIP-CDN  |   16   |    256     |      3e-5     |       32       |    F1    | 0.76979 |    0.76489     |
| KUAKE-QQR |    2   |     32     |      6e-5     |       64       | Accuracy | 0.83865 |    0.84053     |
| KUAKE-QTR |    4   |     32     |      6e-5     |       64       | Accuracy | 0.69722 |    0.69722     |
| KUAKE-QIC |    4   |     32     |      6e-5     |      128       | Accuracy | 0.81483 |    0.82046     |
| CMeEE     |    2   |     32     |      6e-5     |      128       | Micro-F1 | 0.66120 |    0.66026     |
| CMeIE     |  100   |     12     |      6e-5     |      300       | Micro-F1 | 0.61385 |    0.60076     |

可支持配置的参数：

* `save_dir`：可选，保存训练模型的目录；默认保存在当前目录checkpoints文件夹下。
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

**NOTE:**
* 如需恢复模型训练，则可以设置`init_from_ckpt`， 如`init_from_ckpt=checkpoints/model_100/model_state.pdparams`。
* 使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，具体代码见export_model.py。静态图参数保存在`output_path`指定路径中。
  运行方式：

```shell
python export_model.py --train_dataset CMeIE --params_path=./checkpoint/model_900/model_state.pdparams --output_path=./export
```

#### 医疗文本分类任务

```shell
$ unset CUDA_VISIBLE_DEVICES
$ python -m paddle.distributed.launch --gpus "0,1,2,3" train_classification.py --dataset CHIP-CDN-2C --batch_size 256 --max_seq_length 32 --learning_rate 3e-5 --epochs 16
```

其他可支持配置的参数：

* `dataset`：可选，CHIP-CDN-2C CHIP-CTC CHIP-STS KUAKE-QIC KUAKE-QTR KUAKE-QQR，默认为KUAKE-QIC数据集。

#### 医疗命名实体识别任务

```shell
$ export CUDA_VISIBLE_DEVICES=0
$ python train_ner.py --batch_size 32 --max_seq_length 128 --learning_rate 6e-5 --epochs 12
```

#### 医疗关系抽取任务

```shell
$ export CUDA_VISIBLE_DEVICES=0
$ python train_spo.py --batch_size 12 --max_seq_length 300 --learning_rate 6e-5 --epochs 100
```

### 依赖安装

```shell
pip install xlrd==1.2.0
```

[1] CBLUE: A Chinese Biomedical Language Understanding Evaluation Benchmark [pdf](https://arxiv.org/abs/2106.08087) [git](https://github.com/CBLUEbenchmark/CBLUE) [web](https://tianchi.aliyun.com/specials/promotion/2021chinesemedicalnlpleaderboardchallenge)
