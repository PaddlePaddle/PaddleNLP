# 千言：面向事实一致性的生成评测比赛基线

## 比赛简介

自然语言生成旨在让机器能够像人一样使用自然语言进行表达和交互，它是人工智能领域重要的前沿课题，近年来受到学术界和工业界广泛关注。

随着神经网络生成模型特别是预训练语言模型的迅速发展，机器生成文本的可读性和流畅性不断提升。然而，自动生成的文本中依然经常出现不符合原文或背景的错误事实描述，这种生成的事实一致性问题是自然语言生成进行落地应用的主要障碍之一，并逐渐受到研究学者的关注。鉴于当前国内外关于事实一致性的生成评测比赛十分匮乏，为了促进自然语言生成的技术发展和实际应用，我们计划组织面向事实一致性的生成评测比赛。

在[此比赛](https://aistudio.baidu.com/aistudio/competition/detail/105)中，我们将提供三个对事实一致性有较高要求的生成任务，包括文案生成、摘要生成和问题生成。同时，在系统评价中，我们将结合文本流畅性和事实一致性两项指标综合评估参赛生成系统的水平。通过这样的任务设定和评价方式，此评测将有助于研究者和开发者更多关注自然语言生成的事实一致性难题，并为大家提供学术交流平台，从而进一步提升自然语言生成的研究水平，推动相关技术的应用发展。

本比赛得到中国中文信息学会自然语言生成专业委员会（筹）支持，将在2021年11月7日首届中国自然语言生成大会（CCNLG-2021）召开评测研讨会，并在大会上对获奖团队颁奖。

## 模型简介
本次比赛提供的基线系统，基于百度提出的ERNIE-UNIMO统一模态预训练框架。在本次比赛的三个文本生成任务中，我们基于本基线使用的模型是UNIMO-text,是基于[ERNIE-UNIMO](https://arxiv.org/pdf/2012.15409.pdf)框架在文本数据上预训练得到模型。

## 快速开始

本基线基于 **PaddleNLP 2.0.8** 版本，该版本包含了基线使用的最新版UNIMO-text模型以及升级后的生成API。更多详细升级信息请查看[Release Note](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.0.8)。请选手们**升级PaddleNLP后使用**。

### 数据准备

比赛使用三个任务数据集测试参赛系统的生成能力，包括文案生成(AdvertiseGen)、摘要生成(LCSTS_new)和问题生成(DuReaderQG)：

- 文案生成根据结构化的商品信息生成合适的广告文案；
- 摘要生成是为输入文档生成简洁且包含关键信息的简洁文本；
- 问题生成则是根据给定段落以及答案生成适合的问题。

为了方便用户快速使用基线，PaddleNLP Dataset API内置了数据集，一键即可完成数据集加载，示例代码如下：

```python
from paddlenlp.datasets import load_dataset
train_ds, dev_ds = load_dataset('dureader_qg', splits=('train', 'dev'))
```

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
.
├── run_gen.py # 模型finetune主程序入口
├── gen_utils.py # 定义参数及一些工具函数
├── scripts # 三个任务的基线训练脚本
└── README.md # 文档说明
```

### 模型训练

运行如下命令即可在样例训练集上进行finetune，并在样例验证集上进行验证。也可以使用./scripts目录下面的训练脚本分别启动三个任务的训练。

```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" --log_dir ./log run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=unimo-text-1.0 \
    --save_dir=./unimo/checkpoints \
    --logging_steps=100 \
    --save_steps=100000 \
    --epochs=6 \
    --batch_size=16 \
    --learning_rate=5e-5 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_train \
    --do_predict \
    --device=gpu
```

关键参数释义如下：
- `gpus` 指示了训练所用的GPU卡号。
- `dataset_name` 数据集名称，`dureader_qg`、`advertisegen`和`lcsts_new`分别对应问题生成、文案生成和摘要生成三个任务。
- `train_file` 本地训练数据地址，数据格式必须与`dataset_name`所指数据集格式相同。
- `predict_file` 本地测试数据地址，数据格式必须与`dataset_name`所指数据集格式相同。
- `model_name_or_path` 指示了finetune使用的具体预训练模型，可以是PaddleNLP提供的预训练模型，或者是本地的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle预训练模型model_state.pdparams。如果使用PaddleNLP提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP提供的预训练模型        |
   |---------------------------------|
   | unimo-text-1.0      |
   | unimo-text-1.0-large |

- `save_dir` 表示模型的保存路径。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `seed` 表示随机数生成器的种子。
- `epochs` 表示训练轮数。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示AdamW优化器中使用的weight_decay的系数。
- `warmup_propotion` 表示学习率逐渐升高到基础学习率（即上面配置的learning_rate）所需要的迭代数占总步数的比例，最早的使用可以参考[这篇论文](https://arxiv.org/pdf/1706.02677.pdf)。
- `max_seq_len` 模型输入序列的最大长度。
- `max_target_len` 模型训练时标签的最大长度。
- `min_dec_len` 模型生成序列的最小长度。
- `max_dec_len` 模型生成序列的最大长度。
- `do_train` 是否进行训练。
- `do_predict` 是否进行预测，在验证集上会自动评估。
- `device` 表示使用的设备，从gpu和cpu中选择。

更多参数详情和参数的默认值请参考`args.py`。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
./checkpoints/
├── model_8000
│   ├── model_config.json
│   ├── model_state.pdparams
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:** 如需恢复模型训练，`model_name_or_path`配置本地模型的目录地址即可。

### 模型预测

运行下方脚本可以使用训练好的模型进行预测。

```shell
export CUDA_VISIBLE_DEVICES=0
python run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=your_model_path \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --device=gpu
```

程序运行结束后会将预测结果保存在`output_path`中。将预测结果准备成比赛官网要求的格式，提交评估即可得评估结果。

Finetuned baseline的模型在各任务验证集上有如下结果(指标为BLEU-4)：

|       model_name        | LCSTS_new | DuReaderQG |    AdvertiseGen    |
| :-----------------------------: | :---: | :-----------: | :-------------------: |
|   finetuned unimo-text-1.0    | 18.82 | 39.78 |     10.03     |
