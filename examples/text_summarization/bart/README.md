# BART

## 模型简介

BART是一种Seq2Seq结构的降噪自编码器，通过增加噪声来破环文本然后重建原文本来训练模型。它使用一个标准的Transformer结构，可以被看作泛化的BERT（由于是双向编码器），GPT（由于是从左到右解码器），和一些其他的预训练模型结构。

本项目是BART在 PaddlePaddle 2.2上开源实现的文本摘要的例子，包含了在[CNN/DailyMail](https://arxiv.org/pdf/1704.04368.pdf)数据集上微调和生成的代码。

## 快速开始

### 环境依赖

- nltk
- rouge_score

安装方式：`pip install -r requirements.txt`

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
.
├── run_summarization.py # 模型finetune主程序入口
├── generate.py # 模型生成主程序入口
├── utils.py # 定义参数及一些工具函数
├── requirements.txt # 环境依赖文件
└── README.md # 文档说明
```

### 数据准备

**CNN/DailyMail**数据集是一个英文数据集，包含CNN和《每日邮报》记者撰写的30多万篇独特新闻文章，常用来做文本摘要。

为了方便用户快速测试，PaddleNLP Dataset API内置了CNN/DailyMail数据集，一键即可完成数据集加载，示例代码如下：

```python
from paddlenlp.datasets import load_dataset
train_set, dev_set, test_set = load_dataset("cnn_dailymail",  splits=["train", "dev", "test"])
```

### 模型训练

运行如下命令即可在训练集上进行finetune，并在验证集上进行验证

```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
# 例如使用1号和2号卡，则：`--gpu 1,2`
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus 1,2 run_summarization.py \
    --model_name_or_path=bart-base \
    --dataset_name=cnn_dailymail \
    --output_dir=output \
    --max_source_length=1024 \
    --max_target_length=142 \
    --learning_rate=1e-4 \
    --num_train_epochs=6 \
    --logging_steps=100 \
    --save_steps=1000 \
    --seed=42 \
    --train_batch_size=20 \
    --eval_batch_size=64 \
    --warmup_proportion=0.1 \
    --ignore_pad_token_for_loss=True \
    --device=gpu
```

其中参数释义如下：
- `gpus` 指示了训练所用的GPU

- `model_name_or_path` 指示了finetune使用的预训练模型，可以是PaddleNLP提供的预训练模型，或者是本地的模型。如果使用本地的模型，则配置为本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle模型参数model_state.pdparams。如果使用PaddleNLP提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP提供的预训练模型        |
   |---------------------------------|
   | bart-base |
   | bart-large |

- `dataset_name` 表示训练的数据集。

- `output_dir` 表示模型的保存路径。

- `max_source_length` 表示输入article的最大长度。

- `max_target_length` 表示输入highlights的最大长度。

- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。

- `num_train_epochs` 表示训练轮数。

- `logging_steps` 表示日志打印间隔。

- `save_steps` 表示模型保存及评估间隔。

- `seed` 表示随机数生成器的种子。

- `epochs` 表示训练轮数。

- `train_batch_size` 表示训练**每张卡**上的样本数目。

- `eval_batch_size` 表示预测**单卡**上的样本数目。

- `warmup_proportion` 表示warmup_steps所占总步数的比例。学习率逐渐升高到基础学习率（即上面配置的learning_rate）所需要的迭代数。

- `ignore_pad_token_for_loss` 表示计算loss时忽略padding。

- `device` 表示使用的设备。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`output_dir`中。如：

```text
./output/
├── bart_model_1000.pdparams
│   ├── model_config.json
│   ├── model_state.pdparams
│   ├── merges.txt
│   ├── tokenizer_config.json
│   └── vocab.json
└── ...
```

**NOTE:** 如需恢复模型训练，只需指定`model_name_or_path`为本地微调模型的路径即可。

### 模型预测

运行如下命令即可在验证集上进行测试

```shell
# GPU启动，预测仅支持单卡
export CUDA_VISIBLE_DEVICES=0
python generate.py \
    --model_name_or_path=bart-base-cnndm-model \
    --dataset_name=cnn_dailymail \
    --output_path=generate.txt \
    --max_source_length=1024 \
    --max_target_length=142 \
    --decode_strategy=greedy_search \
    --top_k=2 \
    --top_p=1.0 \
    --num_beams=1 \
    --length_penalty=0.0 \
    --batch_size=64 \
    --seed=42 \
    --ignore_pad_token_for_loss=True \
    --logging_steps=100 \
    --device=gpu
```

其中参数释义如下：
- `model_name_or_path` 指示了预测使用的模型，可以是PaddleNLP提供的预训练模型，或者是本地的模型。如果使用本地的模型，则配置为本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle模型参数model_state.pdparams。如果使用PaddleNLP提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP提供的预训练模型        |
   |---------------------------------|
   | bart-base |
   | bart-large |

- `dataset_name` 表示预测的数据集。

- `output_path` 表示预测结果的保存路径。

- `max_source_length` 表示输入article的最大长度。

- `max_target_length` 表示输入highlights的最大长度。

- `decode_strategy` 表示预测解码时采取的策略，可选"sampling"、"greedy_search"和"beam_search"之一。

- `top_k` 表示采用"sampling"解码策略时，token的概率按从大到小排序，生成的token只从前`top_k`个中进行采样。

- `top_p` 表示采用"sampling"解码策略时，从词表中采样并选择概率之和大于给定阈值`top_p`的token。

- `num_beams` 表示besm search的beam size。

- `length_penalty` 表示besm search生成长度的指数惩罚。

- `batch_size` 表示每次迭代**单卡**上的样本数目。

- `seed` 表示随机数生成器的种子。

- `ignore_pad_token_for_loss` 表示训练时计算loss时忽略padding。如果训练时设置为True，那么预测时的label需要还原来计算评估指标。

- `logging_steps` 表示日志打印间隔。

- `device` 表示使用的设备。

程序运行结束后会将预测生成的摘要保存在`output_path`中。同时终端中会输出评估结果。

采用预训练模型及微调模型在验证集上有如下结果：

|   model_name_or_path    |     Rouge-1     |     Rouge-2     |    Rouge-L    |
| :----------------------: | :-------------: | :-------------: |:-------------: |
|        [bart-base-cnndm-model](https://bj.bcebos.com/paddlenlp/models/transformers/bart/bart-base-cnndm-model.tar.gz )      | 43.6446 | 20.1447 | 41.0132 |

**NOTE:** `bart-base-cnndm-model`是按本项目中的超参finetune得到的结果。

### 模型高性能预测

在模型预测阶段，我们提供了基于 FasterTransformer 的高性能预测的选项，可以选择性开启是否需要采用高性能预测。只需在上述模型预测上添加两个参数即可：分别是`faster`，`use_fp16_decoding`。

```shell
# GPU启动，预测仅支持单卡
export CUDA_VISIBLE_DEVICES=0
python generate.py \
    --model_name_or_path=bart-base-cnndm-model \
    --dataset_name=cnn_dailymail \
    --output_path=generate.txt \
    --max_source_length=1024 \
    --max_target_length=142 \
    --decode_strategy=greedy_search \
    --top_k=2 \
    --top_p=1.0 \
    --num_beams=1 \
    --length_penalty=0.0 \
    --batch_size=64 \
    --seed=42 \
    --ignore_pad_token_for_loss=True \
    --logging_steps=100 \
    --faster \
    --use_fp16_decoding \
    --device=gpu
```
其中新增参数释义如下：
- `faster` 表示是否开启高性能预测。设置 `--faster` 即表示开启。
- `use_fp16_decoding` 表示在开启高性能预测的时候，是否使用 fp16 来完成预测过程。设置 `--use_fp16_decoding` 即表示使用 fp16 进行预测，否则使用 fp32。

## 参考文献
1. Lewis M , Liu Y , Goyal N , et al. [BART: Denoising Sequence-to-Sequence Pre-training for Natural
Language Generation, Translation, and Comprehension](https://aclanthology.org/2020.acl-main.703.pdf)[C]//Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020: 7871-7880.
2. See A , Liu P J , CD  Manning. [Get To The Point: Summarization with Pointer-Generator Networks](https://aclanthology.org/P17-1099.pdf)[C]// Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics. 2017: 1073–1083.
