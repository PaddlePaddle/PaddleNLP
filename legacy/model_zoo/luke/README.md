# LUKE with PaddleNLP

[LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention](https://arxiv.org/abs/2010.01057)

**模型简介：**
许多NLP任务都涉及实体，例如：关系分类、实体类型、命名实体识别（NER）和问答（QA）。解决此类实体相关任务的关键是学习实体有效表示。传统的实体表示为每个实体分配一个固定的Embedding向量，该向量将有关实体的信息存储在知识库（KB）中。它们需要实体链接(entity linking)来表示文本中的实体，而不能表示KB中不存在的实体。

相比之下，基于contextualized word representations(CWRs) transformer的大型预训练模型，如BERT和RoBERTa，提供了基于语言建模的有效通用词语表征。然而，由于以下两个原因，CWRs的体系结构不适合表示实体：

- 由于CWR不输出实体的跨级(span-level)表示，因此它们通常需要学习如何基于通常较小的下游数据集计算此类表征。

- 许多与实体相关的任务，如关系分类和问答（QA）涉及实体之间关系的推理。尽管transformer可以通过使用self-attention机制将单词相互关联来捕捉单词之间的复杂关系。在实体之间执行关系推理是困难的，因为许多实体在模型中被分割成多个词。此外，基于单词的CWRs预训练任务不适合学习实体的表征，因为在实体中预测一个被MASK的单词，例如预测“Rings”, 给予句子“The Lord of the [MASK]”，一个完整的实体就这样被拆分。

LUKE和现有CWRs之间的一个重要区别在于，它不仅将单词视为独立的token，还将实体视为独立的token，并使用transformer计算所有token的中间表征和输出表征。由于实体被视为token，LUKE可以直接建模实体之间的关系。
本项目是 LUKE 在 Paddle 2.x上的开源实现。

## 快速开始

### 下游任务微调

数据集
下载Open Entity数据集
[下载地址](https://cloud.tsinghua.edu.cn/f/6ec98dbd931b4da9a7f0/)
把下载好的文件解压,并把解压后的Open Entity目录下的`train.json`、`test.json`和`dev.json`分别为训练集、验证集和测试集

下载SQuAD1.1数据集，主流机器阅读理解数据集
[下载地址](https://data.deepai.org/squad1.1.zip)

#### 1、SQuAD1.1
以SQuAD1.1数据集为例

运行以下两个命令即可训练并评估LUKE在SQuAD1.1数据集的精度

```shell
python -m paddle.distributed.launch examples/language_model/luke/run_squad.py
    --model_type luke \
    --device gpu \
    --learning_rate 15e-6 \
    --num_train_epochs 2 \
    --batch_size 8 \
    --do_predict \
    --do_train \
    --model_name_or_path luke-large
```
其中参数释义如下：
- `model_type` 指示了模型类型，当前支持`luke`
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用多GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。
- `num_train_epochs` 表示需要训练的epoch数量
- `do_train` 表示是否开启训练
- `do_predict` 表示是否开启评估
- `model_name_or_path` 模型的名称和路径,支持`luke-base` 和 `luke-large`

训练结束后模型会对模型进行评估，其评估在验证集上完成, 训练完成后你将看到如下结果:
```text
{"exact_match": 89.75691579943235, "f1": 94.95702001984502}
```

#### 2、Open Entity

```shell
python -m paddle.distributed.launch examples/language_model/luke/run_open_entity.py \
    --model_type luke-large \
    --data_dir data/ \
    --output_dir output/ \
    --device gpu \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --train_batch_size 2
```
训练结束后模型会对模型进行评估，其评估在测试集上完成, 训练完成后你将看到如下结果:
```text
Results: {
  "test_f1": 0.7815726767275616,
  "test_precision": 0.7880405766150561,
  "test_recall": 0.7752100840336135
}
```


# Reference

```bibtex
@inproceedings{yamada2020luke,
  title={LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention},
  author={Ikuya Yamada and Akari Asai and Hiroyuki Shindo and Hideaki Takeda and Yuji Matsumoto},
  booktitle={EMNLP},
  year={2020}
}
```
