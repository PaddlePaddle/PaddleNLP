# DeBERTa: Decoding-enhanced BERT with Disentangled Attention

## 目录
* [模型简介](#模型简介)
* [快速开始](#快速开始)
  * [通用参数释义](#通用参数释义)
  * [自然语言推断任务](#自然语言推断任务)
* [参考资料](#参考资料)

## 模型简介

[DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)是微软团队提出的一个语言模型。它通过使用相对位置编码和增强的掩码解码器来改进 BERT 模型。

DeBERTa 模型使用了两种新技术改进了 BERT 和 RoBERTa 模型，同时还引入了一种新的微调方法以提高模型的泛化能力。两种新技术的改进包括注意力解耦机制和增强的掩码解码器。新的微调方法为虚拟对抗训练方法。结果表明，这些技术显著提高了模型预训练的效率以及自然语言理解（NLU）和自然语言生成（NLG）下游任务的性能。

DeBERTaV2 是在 DeBERTaV1 的基础上进行了一些改进的版本。它的词表从 V1 中的 50K 扩大到了 128K。在第一个转换器层之外，V2 模型使用了一个额外的卷积层，以更好地学习输入标记的依赖性。此外，V2 模型通过共享位置和内容的变换矩阵来保存参数，同时不影响性能。V2 模型还使用对数桶对相对位置进行编码。

本项目是DeBERTa在PaddleNLP的开源实现，包含了DeBERTa在MNLI数据集上的微调代码。


本文件夹内包含了`Deberta模型`以及`Deberta-v2模型`在`MNLI任务`上的训练和验证内容。以下是本例的简要目录结构及说明：

```text
.
├── README.md                   # README文档
├── run_glue.py               # 自然语言推断训练代码
```

## 快速开始

### 依赖安装

```shell
# 安装最新版本的paddlenlp
pip install paddlenlp
```

### 通用参数释义
- `model_name_or_path` 指示了 Fine-tuning 使用的具体预训练模型以及预训练时使用的tokenizer。若模型相关内容保存在本地，可以提供相应目录地址，例如："./checkpoint/model_xx/"。
- `output_dir` 表示模型保存路径。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断，不足该长度的将会进行 padding。
- `learning_rate` 表示基础学习率大小，本代码并未使用学习率warmup和衰减。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔步数。
- `save_steps` 表示模型保存及评估间隔步数。
- `batch_size` 表示每次迭代**每张**卡上的样本数目。
- `adam_epsilon` 表示Adam优化器的epsilon。
- `max_steps` 表示最大训练步数。若训练`num_train_epochs`轮包含的训练步数大于该值，则达到`max_steps`后就提前结束。
- `seed` 表示随机数种子。
- `device` 表示训练使用的设备, `'gpu'`表示使用 GPU, `'xpu'`表示使用百度昆仑卡, `'cpu'`表示使用 CPU。

### 自然语言推断任务

#### 数据集介绍
MNLI任务是一个自然语言推断任务，给定一个前提和假设，判断假设是否可以从前提中推断出来。MNLI任务的数据集包含了三种类型的数据：训练集、开发集和测试集。其中训练集包含了 392,702 个样本，开发集包含了 9,815 个样本，测试集包含了 9,842 个样本。每个样本包含了前提、假设和标签。标签包含了三种类型：entailment、neutral 和 contradiction。

#### 单卡训练

```shell
python  ./run_glue.py \
    --model_name_or_path {model_name_or_path} \
    --tokenizer_name_or_path {tokenizer_name_or_path}\
    --task_name MNLI \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --logging_steps 1 \
    --save_steps 200 \
    --output_dir MNLI \
    --device gpu
```

在MNLI数据集上微调 MNLI 类型的自然语言推断任务后，在测试集上有如下结果：
| Model | MNLI |
| --- | --- |
| Deberta-large | 90.3 |
| Deberta-v3-large | 90.9 |
注意，以上结果是在单卡 Tesla V100 上训练得到的，不同设备的结果可能会有所差异。其中batch_size设置为32，eval_steps为200.


## 参考资料
- https://github.com/microsoft/DeBERTa
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/deberta

## 引用

Bibtex:
```tex
@inproceedings{
he2021deberta,
title={DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION},
author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=XPZIaotutsD}
}
```
