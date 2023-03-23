# 生成式文本摘要预训练
**目录**

- [生成式文本摘要预训练](#生成式文本摘要预训练)
  - [简介](#简介)
  - [预训练任务介绍](#预训练任务)
  - [预训练定制](#预训练定制)
    - [文本摘要预训练全流程介绍](#文本摘要预训练全流程介绍)
    - [环境依赖](#环境依赖)
    - [代码结构说明](#代码结构说明)
    - [数据准备](#数据准备)
      - [数据加载](#数据加载)
      - [从本地文件创建数据集](#从本地文件创建数据集)
    - [模型预训练](#模型预训练)
  - [模型微调](#模型微调)
  - [References](#references)

## 简介

文本摘要的目标是自动地将输入文本转换成简短摘要,为用户提供简明扼要的内容描述，是缓解文本信息过载的一个重要手段。
文本摘要也是自然语言生成领域中的一个重要任务，有很多应用场景，如新闻摘要、论文摘要、财报摘要、传记摘要、专利摘要、对话摘要、评论摘要、观点摘要、电影摘要、文章标题生成、商品名生成、自动报告生成、搜索结果预览等。

本项目预训练了一个专门为中文文本摘要任务设计的语言模型：PEGASUS。其预训练目标为间隙句子生成（Gap Sentences Generation, GSG），是专门为文本摘要任务设计的上游任务。

## 预训练任务
Gap Sentences Generation（GSG）是一种专门为文本摘要提出的自监督预训练任务，其首先找出输入文本中较为核心的数个句子，然后将它们直接拼接到一起得到伪摘要输出，这些句子在输入中的位置则被替换成mask token，预训练的目标就是生成这些被mask掉的核心句子，即间隙句子。

对于GSG任务如何选择核心句子以及超参数的设置，请参考[原论文](https://arxiv.org/pdf/1912.08777.pdf)

另外，原论文中也用到了Masked Language Model (MLM) 作为预训练任务，但实际效果增幅不大，所以不做使用。


## 预训练定制

### 文本摘要预训练全流程介绍

接下来，我们将按数据准备、预训练、预测的全流程进行介绍。

1. **数据准备**

- 如果没有已标注的数据集，我们推荐[doccano](https://github.com/doccano/doccano)数据标注工具。
  如果已有标注好的本地数据集，我们需要根据将数据集整理为文档要求的格式，请参考[从本地文件创建数据集](#从本地文件创建数据集)
  。
- 此外，还需要准备中文停用词表，存放到stopwords.txt中，建议参考[哈工大停用词表](https://github.com/goto456/stopwords)

2. **模型预训练**

- 数据准备完成后，可以开始使用我们的数据集完成模型的预训练任务。我们可以根据任务需求，调整可配置参数，选择使用GPU或CPU进行模型训练，脚本默认保存在开发集最佳表现模型。预训练的Tokenizer默认使用base版本"IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese"的分词器，还支持large版本的分词器: "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese"


3. **模型预测**

- 预训练结束后，我们可以加载保存的最佳模型进行模型测试，打印模型在文本摘要任务上的预测结果。


### 环境依赖

rouge==1.0.1

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
pretrain/
├── data # 数据
│   ├── train.json # 预训练数据集文件
│   └── test.json # 可选，待预测数据文件
├── stopwords.txt # 停用词表
├── train.py # 训练评估脚本
├── utils.py # 工具函数脚本
├── requirements.txt # 依赖包
└── README.md # 说明文档
```

### 数据准备

#### 数据加载

#### 从本地文件创建数据集

如果您想使用自己的数据来预训练PEGASUS模型，本项目支持使用固定格式本地数据集文件进行预训练。

本地数据集目录结构如下：

```text
data/
├── train.json # 训练数据集文件
└── test.json # 可选，待预测数据文件
```

本地数据集文件格式如下：

- train.json/test.json 文件每行格式：

```text
{
"title": "任志强抨击政府把土地作为投机品地产业被人为破坏",
"content": "“北京的保障房市场就像一个巨大的赌场，每个人都在期待中奖。”面对中国目前现行的保障性住房政策，华远地产董事长任志强再次语出惊人。（分享自@第一财经-中国房地产金融）"
}
```

更多数据集读取格式详见[数据集加载](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_load.html#)
和[自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)。

### 模型预训练

运行如下命令即可在样例训练集上开始pretrain，并在样例验证集上进行验证。

```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "2,3,4,5,6,7" train.py \
    --model_name_or_path=IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese \
    --train_file data/train.json \
    --eval_file data/test.json \
    --output_dir pegasus_out \
    --max_source_length 128 \
    --max_target_length 64 \
    --num_train_epochs 20 \
    --logging_steps 1 \
    --save_steps 10000 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.02 \
    --weight_decay=0.001 \
    --do_train \
    --do_eval \
    --device=gpu
```

关键参数释义如下：

- `gpus` 指示了训练所用的GPU卡号。
- `train_file` 本地训练数据地址。
- `eval_file` 本地测试数据地址。
- `model_name_or_path`
  指示了pretrain所使用的分词器，可以是PaddleNLP提供的分词器，或者是本地的分词器。如果使用本地的分词器，可以配置本地分词器的目录地址，例如:
  ./checkpoints/model_xx/。如果使用PaddleNLP提供的分词器，可以选择下面其中之一。

  | PaddleNLP提供的分词器        |
     |---------------------------------|
  | IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese      |
  | IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese      |

- `output_dir` 表示模型的保存路径。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `seed` 表示随机数生成器的种子。
- `num_train_epochs` 表示训练轮数。
- `per_device_train_batch_size` 表示每次训练**每张卡**上的样本数目。
- `per_device_eval_batch_size` 表示每次验证**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示AdamW优化器中使用的weight_decay的系数。
- `warmup_ratio`
  表示学习率逐渐升高到基础学习率（即上面配置的learning_rate）所需要的迭代数占总步数的比例，最早的使用可以参考[这篇论文](https://arxiv.org/pdf/1706.02677.pdf)。
- `max_source_length` 模型输入序列的最大长度。
- `max_target_length` 模型训练时标签的最大长度。
- `do_train` 是否进行训练。
- `do_eval` 是否进行预测。
- `device` 表示使用的设备，从gpu和cpu中选择。

更多参数详情和参数的默认值请参考`train.py`。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`output_dir`中。
如：

```text
./pegasus_out/
├── model_config.json
├── model_state.pdparams
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.txt
```

**NOTE:** 如需恢复模型训练，`model_name_or_path`配置本地模型的目录地址即可。


## 模型微调
微调代码及效果请参考[PEGASUS微调](../finetune/)


## References

- Zhang J, Zhao Y, Saleh M, et al. Pegasus: Pre-training with extracted gap-sentences for abstractive summarization[C]
  //International Conference on Machine Learning. PMLR, 2020: 11328-11339.
- Wang J, Zhang Y, Zhang L, et al. Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence[J]. arXiv
  preprint arXiv:2209.02970, 2022.
