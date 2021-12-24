# ERNIE-M

* [模型简介](#模型简介)
* [快速开始](#快速开始)
  * [通用参数释义](#通用参数释义)
  * [文本蕴含任务](#文本蕴含任务)
* [参考论文](#参考论文)

## 模型简介

[ERNIE-M](https://arxiv.org/abs/2012.15674) 是百度 NLP 提出的基于回译机制，从单语语料中学习语言间的语义对齐关系的预训练模型，显著提升包括跨语言自然语言推断、语义检索、语义相似度、命名实体识别、阅读理解在内的5种典型跨语言理解任务效果，并登顶权威跨语言理解评测 XTREME 榜首。

本项目是 ERNIE-M 的 PaddlePaddle 动态图实现， 包含模型训练，模型验证等内容。以下是本例的简要目录结构及说明：

```text
.
├── README.md                   # 文档
├── run_classifier.py           # 文本蕴含任务
```

## 快速开始

### 通用参数释义

- `task_type` 表示了文本蕴含任务的类型，目前支持的类型为："cross-lingual-transfer", "translate-train-all"
  ，分别表示在英文数据集上训练并在所有15种语言数据集上测试、在所有15种语言数据集上训练和测试。
- `model_name_or_path` 指示了 Fine-tuning 使用的具体预训练模型以及预训练时使用的tokenizer，目前支持的预训练模型有："ernie-m-base"， "ernie-m-large"
  。若模型相关内容保存在本地，这里也可以提供相应目录地址，例如："./checkpoint/model_xx/"。
- `output_dir` 表示模型保存路径。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断，不足该长度的将会进行 padding。
- `memory_length` 表示当前的句子被截取作为下一个样本的特征的长度。
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔步数。
- `save_steps` 表示模型保存及评估间隔步数。
- `batch_size` 表示每次迭代**每张**卡上的样本数目。
- `weight_decay` 表示AdamW的权重衰减系数。
- `layerwise_decay` 表示 AdamW with Layerwise decay 的逐层衰减系数。
- `adam_epsilon` 表示AdamW优化器的 epsilon。
- `warmup_proportion` 表示学习率warmup系数。
- `max_steps` 表示最大训练步数。若训练`num_train_epochs`轮包含的训练步数大于该值，则达到`max_steps`后就提前结束。
- `seed` 表示随机数种子。
- `device` 表示训练使用的设备, 'gpu'表示使用 GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用 CPU。
- `use_amp` 表示是否启用自动混合精度训练。
- `scale_loss` 表示自动混合精度训练的参数。

### 文本蕴含任务

#### 数据集介绍
XNLI 是 MNLI 的子集，并且已被翻译成14种不同的语言（包含一些较低资源语言）。与 MNLI 一样，目标是预测文本蕴含（句子 A 是否暗示/矛盾/都不是句子 B ）。

#### 单卡训练

```shell
python run_classifier.py \
    --task_type cross-lingual-transfer \
    --batch_size 64 \
    --model_name_or_path ernie-m-base \
    --save_steps 3068 \
    --output_dir output
```

#### 多卡训练

```shell
python -m paddle.distributed.launch --gpus 0,1 --log_dir output run_classifier.py \
    --task_type cross-lingual-transfer \
    --batch_size 64 \
    --model_name_or_path ernie-m-base \
    --save_steps 3068 \
    --output_dir output
```

在XNLI数据集上Finetune cross-lingual-transfer 类型的文本蕴含任务后，在验证集上有如下结果
| 模型 | ar | bg | de | el | en | es | fr | hi | ru | sw | th | tr | ur | vi | zh | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cross-lingual Transfer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 官方ERNIE-M | 76.3 | 80.4 | 79.2 | 79.1 | 85.5 | 81.2 | 80.1 | 72.9 | 78.1 | 69.5 | 75.8 | 76.8 | 68.8 | 78.3 | 77.4 | 77.3 |
| 我们的ERNIE-M | 75.1 | 79.4 | 79.0 | 77.6 | 85.2 | 80.0 | 79.2 | 71.8 | 77.3 | 69.5 | 74.3 | 76.1 | 68.2 | 77.0 | 76.5 | 76.4 |

## 参考论文

 [Ouyang X ,  Wang S ,  Pang C , et al. ERNIE-M: Enhanced Multilingual Representation by Aligning Cross-lingual Semantics with Monolingual Corpora[J].  2020.](https://arxiv.org/abs/2012.15674)
