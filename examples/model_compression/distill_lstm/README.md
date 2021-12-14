# Distilling Knowledge From Fine-tuned BERT into Bi-LSTM

以下是本例的简要目录结构及说明：
```
.
├── small.py              # 小模型结构以及对小模型单独训练的脚本
├── bert_distill.py       # 用教师模型BERT蒸馏学生模型的蒸馏脚本
├── data.py               # 定义了dataloader等数据读取接口
├── utils.py              # 定义了将样本转成id的转换接口
├── args.py               # 参数配置脚本
└── README.md             # 文档，本文件
```

## 简介
本目录下的实验是将特定任务下BERT模型的知识蒸馏到基于Bi-LSTM的小模型中，主要参考论文 [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)实现。

在模型蒸馏中，较大的模型（在本例中是BERT）通常被称为教师模型，较小的模型（在本例中是Bi-LSTM）通常被称为学生模型。知识的蒸馏通常是通过模型学习蒸馏相关的损失函数实现，在本实验中，损失函数是均方误差损失函数，传入函数的两个参数分别是学生模型的输出和教师模型的输出。

在[论文](https://arxiv.org/abs/1903.12136)的模型蒸馏阶段，作者为了能让教师模型表达出更多的知识供学生模型学习，对训练数据进行了数据增强。作者使用了三种数据增强方式，分别是：

1. Masking，即以一定的概率将原数据中的word token替换成`[MASK]`；

2. POS—guided word replacement，即以一定的概率将原数据中的词用与其有相同POS tag的词替换；

3. n-gram sampling，即以一定的概率，从每条数据中采样n-gram，其中n的范围可通过人工设置。

通过数据增强，可以产生更多无标签的训练数据，在训练过程中，学生模型可借助教师模型的“暗知识”，在更大的数据集上进行训练，产生更好的蒸馏效果。需要指出的是，实验只使用了第1和第3种数据增强方式。
在英文数据集任务上，本文使用了Google News语料[预训练的Word Embedding](https://code.google.com/archive/p/word2vec/)初始化小模型的Embedding层。

本实验分为三个训练过程：在特定任务上对BERT的fine-tuning、在特定任务上对基于Bi-LSTM的小模型的训练（用于评价蒸馏效果）、将BERT模型的知识蒸馏到基于Bi-LSTM的小模型上。

## 数据、预训练模型介绍及获取

本实验使用GLUE中的SST-2、QQP以及中文情感分类数据集ChnSentiCorp中的训练集作为训练语料，用数据集中的验证集评估模型的效果。运行本目录下的实验，数据集会被自动下载到`paddlenlp.utils.env.DATA_HOME` 路径下，例如在linux系统下，例如对于GLUE中的QQP数据集，默认存储路径是`~/.paddlenlp/datasets/glue/QQP`，对于ChnSentiCorp数据集，则会下载到 `~/.paddlenlp/datasets/chnsenticorp`。

对于BERT的fine-tuning任务，本实验中使用了预训练模型`bert-bas-uncased`、`bert-wwm-ext-chinese`、`bert-base-chinese`。同样，这几个模型在训练时会被自动下载到`paddlenlp.utils.env.MODEL_HOME`路径下。例如，对于`bert-base-uncased`模型，在linux系统下，会被下载到`~/.paddlenlp/models/bert-base-uncased`下。

在中文数据集上的小模型训练的输入利用jieba分词，其中词表同本repo下[文本分类项目](../../text_classification/rnn)的词表，可通过运行以下命令进行下载：

```shell
wget https://bj.bcebos.com/paddlenlp/data/senta_word_dict.txt
```

为了节省显存和运行时间，可以对ChnSentiCorp中未出现的词先进行过滤，并将最后的词表文件名和词表大小配置在下面的参数中。


## 蒸馏实验过程
### 训练BERT fine-tuning模型
训练BERT的fine-tuning模型，可以去本repo下example中的[glue目录](../../benchmark/glue)下。关于glue的更多详细说明，可见glue目录下的README文档。

以GLUE的SST-2任务为例，调用BERT fine-tune的训练脚本，配置如下的参数，训练SST-2任务：

```shell
cd ../../benchmark/glue
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=SST-2
python -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 128   \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 10 \
    --output_dir ../model_compression/distill_lstm/pretrained_models/$TASK_NAME/ \
    --device gpu \

```

如果需要训练基于ChnSentiCorp数据集的BERT finetuning模型，可以进入[文本分类目录](../../text_classification/pretrained_models)下，将预训练模型改成BERT，并基于bert-base-chinese和bert-wwm-ext-chinese模型进行fine-tuning训练。

训练完成之后，可将训练效果最好的模型保存在本项目下的`pretrained_models/$TASK_NAME/`下。模型目录下有`model_config.json`, `model_state.pdparams`, `tokenizer_config.json`及`vocab.txt`这几个文件。


### 训练小模型

尝试运行下面的脚本可以分别基于ChnSentiCorp、SST-2、QQP数据集对基于BiLSTM的小模型进行训练。


```shell
CUDA_VISIBLE_DEVICES=0 python small.py \
    --task_name chnsenticorp \
    --max_epoch 20 \
    --vocab_size 1256608 \
    --batch_size 64 \
    --model_name bert-wwm-ext-chinese \
    --optimizer adam \
    --lr 3e-4 \
    --dropout_prob 0.2 \
    --vocab_path senta_word_dict.txt \
    --save_steps 10000 \
    --output_dir small_models/chnsenticorp/

```

```shell
CUDA_VISIBLE_DEVICES=0 python small.py \
    --task_name sst-2 \
    --vocab_size 30522 \
    --max_epoch 10 \
    --batch_size 64 \
    --lr 1.0 \
    --dropout_prob 0.4 \
    --output_dir small_models/SST-2 \
    --save_steps 10000 \
    --embedding_name w2v.google_news.target.word-word.dim300.en

```

```shell
CUDA_VISIBLE_DEVICES=0 python small.py \
    --task_name qqp \
    --vocab_size 30522 \
    --max_epoch 35 \
    --batch_size 256 \
    --lr 2.0 \
    --dropout_prob 0.4 \
    --output_dir small_models/QQP \
    --save_steps 10000 \
    --embedding_name w2v.google_news.target.word-word.dim300.en

```

### 蒸馏模型
这一步是将教师模型BERT的知识蒸馏到基于BiLSTM的学生模型中，可以运行下面的命令分别基于ChnSentiCorp、SST-2、QQP数据集对基于BiLSTM的学生模型进行蒸馏。

```shell
CUDA_VISIBLE_DEVICES=0 python bert_distill.py \
    --task_name chnsenticorp \
    --vocab_size 1256608 \
    --max_epoch 6 \
    --lr 1.0 \
    --dropout_prob 0.1 \
    --batch_size 64 \
    --model_name bert-wwm-ext-chinese \
    --teacher_dir pretrained_models/chnsenticorp/best_bert_wwm_ext_model_880 \
    --vocab_path senta_word_dict.txt \
    --output_dir distilled_models/chnsenticorp \
    --save_steps 10000 \

```

```shell
CUDA_VISIBLE_DEVICES=0 python bert_distill.py \
    --task_name sst-2 \
    --vocab_size 30522 \
    --max_epoch 6 \
    --lr 1.0 \
    --task_name sst-2 \
    --dropout_prob 0.2 \
    --batch_size 128 \
    --model_name bert-base-uncased \
    --output_dir distilled_models/SST-2 \
    --teacher_dir pretrained_models/SST-2/best_model_610 \
    --save_steps 10000 \
    --embedding_name w2v.google_news.target.word-word.dim300.en \

```

```shell
CUDA_VISIBLE_DEVICES=0 python bert_distill.py \
    --task_name qqp \
    --vocab_size 30522 \
    --max_epoch 6 \
    --lr 1.0 \
    --dropout_prob 0.2 \
    --batch_size 256 \
    --model_name bert-base-uncased \
    --n_iter 10 \
    --output_dir distilled_models/QQP \
    --teacher_dir pretrained_models/QQP/best_model_17000 \
    --save_steps 10000 \
    --embedding_name w2v.google_news.target.word-word.dim300.en \

```

各参数的具体说明请参阅 `args.py` ，注意在训练不同任务时，需要调整对应的超参数。


## 蒸馏实验结果
本蒸馏实验基于GLUE的SST-2、QQP、中文情感分类ChnSentiCorp数据集。实验效果均使用每个数据集的验证集（dev）进行评价，评价指标是准确率（acc），其中QQP中包含f1值。利用基于BERT的教师模型去蒸馏基于Bi-LSTM的学生模型，对比Bi-LSTM小模型单独训练，在SST-2、QQP、ChnSentiCorp(中文情感分类)任务上分别有3.3%、1.9%、1.4%的提升。

| Model             | SST-2(dev acc)    | QQP(dev acc/f1)            | ChnSentiCorp(dev acc) | ChnSentiCorp(dev acc) |
| ----------------- | ----------------- | -------------------------- | --------------------- | --------------------- |
| Teacher  model    | bert-base-uncased | bert-base-uncased          | bert-base-chinese     | bert-wwm-ext-chinese  |
| BERT-base         | 0.930046          | 0.905813(acc)/0.873472(f1) | 0.951667              | 0.955000              |
| Bi-LSTM           | 0.854358          | 0.856616(acc)/0.799682(f1) | 0.920000              | 0.920000              |
| Distilled Bi-LSTM | 0.887615          | 0.875216(acc)/0.831254(f1) | 0.932500              | 0.934167              |

## 参考文献

Tang R, Lu Y, Liu L, Mou L, Vechtomova O, Lin J. [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)[J]. arXiv preprint arXiv:1903.12136, 2019.
