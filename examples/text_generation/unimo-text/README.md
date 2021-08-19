# 千言：面向事实一致性的生成评测比赛baseline

## 比赛简介

自然语言生成旨在让机器能够像人一样使用自然语言进行表达和交互，它是人工智能领域重要的前沿课题，近年来受到学术界和工业界广泛关注。

随着神经网络生成模型特别是预训练语言模型的迅速发展，机器生成文本的可读性和流畅性不断提升。然而，自动生成的文本中依然经常出现不符合原文或背景的错误事实描述，这种生成的事实一致性问题是自然语言生成进行落地应用的主要障碍之一，并逐渐受到研究学者的关注。鉴于当前国内外关于事实一致性的生成评测比赛十分匮乏，为了促进自然语言生成的技术发展和实际应用，我们计划组织面向事实一致性的生成评测比赛。

在此比赛中，我们将提供三个对事实一致性有较高要求的生成任务，包括文案生成、摘要生成和问题生成。同时，在系统评价中，我们将结合文本流畅性和事实一致性两项指标综合评估参赛生成系统的水平。通过这样的任务设定和评价方式，此评测将有助于研究者和开发者更多关注自然语言生成的事实一致性难题，并为大家提供学术交流平台，从而进一步提升自然语言生成的研究水平，推动相关技术的应用发展。

本比赛得到中国中文信息学会自然语言生成专业委员会（筹）支持，将在2021年11月7日首届中国自然语言生成大会（CCNLG-2021）召开评测研讨会，并在大会上对获奖团队颁奖。


## 快速开始

### 数据准备

比赛使用三个任务数据集测试参赛系统的生成能力，包括文案生成、摘要生成和问题生成：

- 文案生成根据结构化的商品信息生成合适的广告文案；
- 摘要生成是为输入文档生成简洁且包含关键信息的简洁文本；
- 问题生成则是根据给定段落以及答案生成适合的问题。


### 模型训练

运行如下命令即可在样例训练集上进行finetune，并在样例验证集上进行验证

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

其中参数释义如下：
- `gpus` 指示了训练所用的GPU卡号。
- `dataset_name` 数据集名称，dureader_qg、advertisegen和lcsts_new分别对应问题生成、文案生成和摘要生成三个任务。
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
│   ├── spm.model
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:** 如需恢复模型训练，`model_name_or_path`配置本地模型的目录地址即可。

### 模型预测

运行如下命令即可在样例测试集上进行测试

```shell
export CUDA_VISIBLE_DEVICES=0
# GPU启动，预测仅支持单卡
python infer.py \
    --model_name_or_path=./checkpoints/model_80000 \
    --test_data_path=./datasets/test.txt \
    --output_path=./predict.txt \
    --logging_steps=500 \
    --seed=2021 \
    --batch_size=4 \
    --min_dec_len=1 \
    --max_dec_len=64 \
    --num_samples=20 \
    --decode_strategy=sampling \
    --top_k=5 \
    --device=gpu
```

其中参数释义如下：
- `model_name_or_path` 指示了finetune使用的具体预训练模型，可以是PaddleNLP提供的预训练模型，或者是本地的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle预训练模型model_state.pdparams。如果使用PaddleNLP提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP提供的预训练模型        |
   |---------------------------------|
   | unified_transformer-12L-cn      |
   | unified_transformer-12L-cn-luge |

- `test_data_path` 表示预测集文件路径。
- `output_path` 表示预测结果的保存路径。
- `logging_steps` 表示日志打印间隔。
- `seed` 表示随机数生成器的种子。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `min_dec_len` 表示预测生成的句子的最小长度。
- `max_dec_len` 表示预测生成的句子的最大长度。
- `num_samples` 表示每条样本生成的句子的数量。对于每条样本，模型会生成`num_samples`个句子，根据每个句子的概率得分进行排序，得分最高的句子作为最终的生成结果。
- `decode_strategy` 表示预测解码时采取的策略，可选"sampling"、"greedy_search"和"beam_search"之一。
- `top_k` 表示采用"sampling"解码策略时，token的概率按从大到小排序，生成的token只从前`top_k`个中进行采样。
- `device` 表示训练使用的设备。

参数详情和参数的默认值请参考`args.py`。

程序运行结束后会将预测结果保存在`output_path`中。将预测结果准备成比赛官网要求的格式，提交评估即可得评估结果。

采用不同的模型在样例测试集上有如下结果：

|       model_name_or_path        |  F1   | BLEU1 / BLEU2 | DISTINCT1 / DISTINCT2 |
| :-----------------------------: | :---: | :-----------: | :-------------------: |
|   unified_transformer-12L-cn    | 10.62 | 0.070 / 0.022 |     0.065 / 0.304     |
| unified_transformer-12L-cn-luge | 33.11 | 0.245 / 0.157 |     0.074 / 0.238     |
|    ./checkpoints/model_80000    | 32.38 | 0.239 / 0.150 |     0.070 / 0.219     |
