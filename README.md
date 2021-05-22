简体中文 | [English](./README_en.md)

<p align="center">
  <img src="./docs/imgs/paddlenlp.png" width="720" height ="100" />
</p>

------------------------------------------------------------------------------------------

[![PyPI - PaddleNLP Version](https://img.shields.io/pypi/v/paddlenlp.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/paddlenlp/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paddlenlp)](https://pypi.org/project/paddlenlp/)
[![PyPI Status](https://pepy.tech/badge/paddlenlp/month)](https://pepy.tech/project/paddlenlp)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
![GitHub](https://img.shields.io/github/license/paddlepaddle/paddlenlp)

## 简介

PaddleNLP 2.0是飞桨生态的文本领域核心库，具备**易用的文本领域API**，**多场景的应用示例**、和**高性能分布式训练**三大特点，旨在提升飞桨开发者文本领域建模效率，并提供基于飞桨框架2.0的NLP领域最佳实践。

### 特性

- **易用的文本领域API**
  - 提供从数据集加载、文本预处理、组网建模、评估、到推的领域API：如一键加载丰富中文数据集的[Dataset API](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html)，可灵活高效的进行数据与处理的[Data API](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.data.html)，预置60+预训练词向量的[Embedding API](./docs/embeddings.md), 内置50+预训练模型，提供预训练模型生态基础设施的[Transformer API](./docs/transformers.md)等，可大幅提升NLP任务建模和迭代的效率。更多API详细说明请查看[PaddleNLP官方文档](https://paddlenlp.readthedocs.io/)


- **多场景的应用示例**
  - PaddleNLP 2.0提供多粒度多场景的应用示例，涵盖从NLP基础技术、NLP核心技术、NLP系统应用以及文本相关的拓展应用等。全面基于飞桨2.0全新API体系开发，为开发提供飞桨2.0框架在文本领域的最佳实践。更多详细应用介绍请查看[PaddleNLP应用示例](./examples/)。


- **高性能分布式训练**
  - 基于飞桨核心框架『**动静统一**』的特性与领先的自动混合精度优化策略，通过分布式Fleet API，支持超大规模参数的4D混合并行策略，并且可根据硬件情况灵活可配，高效地完成超大规模参数的模型训练。


## 安装

### 环境依赖

- python >= 3.6
- paddlepaddle >= 2.1.0

### pip安装

```
pip install --upgrade paddlenlp -i https://pypi.org/simple
```

更多关于PaddlePaddle的安装和PaddleNLP安装详细教程请查看[Installation](./docs/get_started/installation.rst)

## 快速开始

### 数据集快速加载

```python
from paddlenlp.datasets import load_dataset

train_ds, dev_ds, test_ds = load_dataset("chnsenticorp", splits=["train", "dev", "test"])
```

可参考[Dataset文档](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html) 查看更多数据集。

### 一键加载预训练中文词向量

```python
from paddlenlp.embeddings import TokenEmbedding

wordemb = TokenEmbedding("w2v.baidu_encyclopedia.target.word-word.dim300")
print(wordemb.cosine_sim("国王", "王后"))
>>> 0.63395125
wordemb.cosine_sim("艺术", "火车")
>>> 0.14792643
```

内置50+中文词向量，更多使用方法请参考[Embedding文档](./examples/word_embedding/README.md)。


### 一键加载预训练模型

```python
from paddlenlp.transformers import ErnieModel, BertModel, RobertaModel, ElectraModel, GPTForPretraining

ernie = ErnieModel.from_pretrained('ernie-1.0')
bert = BertModel.from_pretrained('bert-wwm-chinese')
roberta = RobertaModel.from_pretrained('roberta-wwm-ext')
electra = ElectraModel.from_pretrained('chinese-electra-small')
gpt = GPTForPretraining.from_pretrained('gpt-cpm-large-cn')
```

请参考[Transformer API文档](./docs/transformers.md)查看目前支持的预训练模型。

### 便捷获取文本特征

```python
import paddle
from paddlenlp.transformers import ErnieTokenizer, ErnieModel

tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
model = ErnieModel.from_pretrained('ernie-1.0')

text = tokenizer('自然语言处理')
pooled_output, sequence_output = model(input_ids=paddle.to_tensor([text['input_ids']]))
```

更多的API示例及其使用说明请查阅[PaddleNLP官方文档](https://paddlenlp.readthedocs.io/)

## 丰富的应用示例

PaddleNLP基于PaddlePaddle 2.0全新API体系，提供了丰富的应用场景示例，帮助开发者按图索骥找到所需，更快上手飞桨2.0框架。
更多模型应用场景介绍请参考[PaddleNLP Examples](./examples/)。

### NLP 基础技术

| 任务   | 简介     |
| -------  | ---- |
| [词法分析 (Lexical Analysis)](./examples/lexical_analysis/) | 基于BiGRU-CRF模型实现了分词、词性标注和命名实体识的联合训练任务。输入是一个字符串，而输出是句子中的词边界和词性、实体类别。 |
| [词向量 (Word Embedding)](./exmaples/word_embedding/) | 提供60+预训练词向量，通过`paddlenlp.TokenEmbedding` API实现快速加载，并提供基于VisualDL的降维可视化示例。 |
| [语言模型 (Language Model)](./examples/language_model/rnnlm)  | 给定一个输入词序列计算其生成概率。 语言模型的评价指标 PPL(困惑度)，用于表示模型生成句子的流利程度。 |
| [语义解析 (Text-to-SQL)](./examples/text_to_sql) | 语义解析是自然语言处理技术的核心任务之一，Text-to-SQL是语义解析的一个子方向，让机器自动将用户输入的自然语言问题转成数据库可操作的SQL查询语句，实现基于数据库的自动问答能力。|

### NLP 核心技术

- [文本分类](./examples/text_classification/)
- [文本生成](./examples/text_generation/)
- [语义匹配](./examples/text_matching/)
- [语义索引](./examples/semantic_indexing/)
- [信息抽取](./examples/information_extraction/)

### NLP 系统应用

#### 情感分析 (Sentiment Analysis)

| 模型      | 简介       |
| -------- | ---------- |
| [SKEP](./examples/sentiment_analysis/skep/)  | 百度研究团队提出的基于情感知识增强的情感预训练算法，此算法采用无监督方法自动挖掘情感知识，然后利用情感知识构建预训练目标，从而让机器学会理解情感语义。SKEP为各类情感分析任务提供统一且强大的情感语义表示。 |

#### 阅读理解 (Machine Reading Comprehension)

| 任务   | 简介     |
| -------  | ---- |
| [SQuAD](./examples/machine_reading_comprehension/SQuAD/) | 提供通过预训练模型在SQuAD 2.0数据集上微调的应用示例。 |
| [DuReader-yesno](./examples/machine_reading_comprehension/DuReader-yesno/) | 提供通过预训练模型在**千言数据集DuReader-yesno**上微调的应用示例。 |
| [DuReader-robust](./examples/machine_reading_comprehension/DuReader-robust/) | 提供通过预训练模型在**千言数据集DuReader-robust**上微调的应用示例。 |

#### 机器翻译 (Machine Translation)

| 模型    | 简介     |
| ------ | ------- |
| [Seq2Seq-Attn](./machine_translation/seq2seq) | 使用编码器-解码器（Encoder-Decoder）结构, 同时使用了Attention机制来加强Decoder和Encoder之间的信息交互，Seq2Seq 广泛应用于机器翻译，自动对话机器人，文档摘要自动生成，图片描述自动生成等任务中。|
| [Transformer](./machine_translation/transformer) | 基于PaddlePaddle框架的Transformer结构搭建的机器翻译模型，Transformer 计算并行度高，能解决学习长程依赖问题。并且模型框架集成了训练，验证，预测任务，功能完备，效果突出。|

#### 同声传译（Simultaneous Translation）

| 模型    | 简介     |
| ------ | ------- |
| [STACL](./simultaneous_translation/stacl) | [STACL](https://www.aclweb.org/anthology/P19-1289/)是基于Prefix-to-Prefix框架设计的同传翻译模型，具备一定的隐式预测能力；结合Wait-k策略可以在保持较高的翻译质量的同时实现任意字级别的翻译延迟。|

#### 对话系统 (Dialogue System)

| 模型   | 简介      |
| ----- | ------|
| [PLATO-2](./dialogue/plato-2) | 百度自研领先的开放域对话预训练模型。[PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning](https://arxiv.org/abs/2006.16779) |

### 拓展应用

#### 文本知识关联 (Text to Knowledge)

[**解语**](./examples/text_to_knowledge/)是由百度知识图谱部开发的文本知识关联框架，覆盖中文全词类的知识库和知识标注工具能够帮助开发者面对更加多元的应用场景，方便地融合自有知识体系，显著提升中文文本解析和挖掘效果，还可以便捷地利用知识增强机器学习模型效果。

- [TermTree: 中文全词类的知识库](./examples/text_to_knowledge/termtree)
- [WordTag: 中文词类知识标注工具](./examples/text_to_knowledge/wordtag)

#### 文本图学习 (Text Graph Learning)

| 模型   | 简介     |
| ------- | ------- |
| [ERNIESage](./text_graph/erniesage)| 通过Graph(图)来构建自身节点和邻居节点的连接关系，将自身节点和邻居节点的关系构建成一个关联样本输入到ERNIE中，ERNIE作为聚合函数（Aggregators）来表征自身节点和邻居节点的语义关系，最终强化图中节点的语义表示。|

### 进阶应用

### 模型压缩 (Model Compression)

| 模型     | 简介    |
| -------- | ------- |
| [Distill-LSTM](./model_compression/distill_lstm/) | 基于[Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)论文策略的实现，将BERT中英文分类的下游模型知识通过蒸馏的方式迁移至LSTM的小模型结构中，取得比LSTM单独训练更好的效果。|
| [OFA-BERT](./model_compression/ofa/) | 基于PaddleSlim Once-For-ALL(OFA)策略对BERT在GLUE任务的下游模型进行压缩，在精度无损的情况下可减少33%参数量，达到模型小型化的提速的效果。 |

## API 使用文档

- [Transformer API](./docs/transformers.md)
  * 基于Transformer结构相关的预训练模型API，包含ERNIE, BERT, RoBERTa, Electra等主流经典结构和下游任务。
- [Data API](./docs/data.md)
  * 文本数据处理Pipeline的相关API说明。
- [Dataset API](./docs/datasets.md)
  * 数据集相关API，包含自定义数据集，数据集贡献与数据集快速加载等功能说明。
- [Embedding API](./docs/embeddings.md)
  * 词向量相关API，支持一键快速加载包预训练的中文词向量，VisulDL高维可视化等功能说明。
- [Metrics API](./docs/metrics.md)
  * 针对NLP场景的评估指标说明，与飞桨2.0框架高层API兼容。


## 交互式Notebook教程

- [使用Seq2Vec模块进行句子情感分类](https://aistudio.baidu.com/aistudio/projectdetail/1283423)
- [如何通过预训练模型Fine-tune下游任务](https://aistudio.baidu.com/aistudio/projectdetail/1294333)
- [使用BiGRU-CRF模型完成快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1317771)
- [使用预训练模型ERNIE优化快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1329361)
- [使用Seq2Seq模型完成自动对联](https://aistudio.baidu.com/aistudio/projectdetail/1321118)
- [使用预训练模型ERNIE-GEN实现智能写诗](https://aistudio.baidu.com/aistudio/projectdetail/1339888)
- [使用TCN网络完成新冠疫情病例数预测](https://aistudio.baidu.com/aistudio/projectdetail/1290873)

更多教程参见[PaddleNLP on AI Studio](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)。


## 版本更新

更多版本更新的详细说明请查看[ChangeLog](./docs/change_log.md)

## 社区贡献与技术交流

### 特殊兴趣小组
- 欢迎您加入PaddleNLP的SIG社区，贡献优秀的模型实现、公开数据集、教程与案例等。

### QQ
- 现在就加入PaddleNLP的QQ技术交流群，一起交流NLP技术吧！⬇️

<div align="center">
  <img src="./docs/imgs/qq.png" width="200" height="200" />
</div>  

### Slack
- 欢迎加入[PaddleNLP Slack channel](https://paddlenlp.slack.com/)与我们的开发者进行技术交流。

## License

PaddleNLP遵循[Apache-2.0开源协议](./LICENSE)。
