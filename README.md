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

<img src="./docs/imgs/news_icon.png" width="50"/> *[2021-05-20] PaddleNLP 2.0正式版已发布！:tada:更多详细升级信息请查看[Release Note](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.0.0).*

PaddleNLP 2.0是飞桨生态的文本领域核心库，具备**易用的文本领域API**，**多场景的应用示例**、和**高性能分布式训练**三大特点，旨在提升飞桨开发者文本领域建模效率，并提供基于飞桨框架2.0的NLP领域最佳实践。

## 特性

- **易用的文本领域API**
  - 提供从数据集加载、文本预处理、模型组网、模型评估、和推理的领域API：一键加载丰富中文数据集的[Dataset API](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html)，可灵活高效地完成数据预处理的[Data API](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.data.html)，预置60+预训练词向量的[Embedding API](./docs/embeddings.md); 提供50+预训练模型的生态基础能力的[Transformer API](./docs/transformers.md)等，可大幅提升NLP任务建模和迭代的效率。更多API详细说明请查看[PaddleNLP官方文档](https://paddlenlp.readthedocs.io/)。


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

更多关于PaddlePaddle和PaddleNLP安装的详细教程请查看[Installation](./docs/get_started/installation.rst)。

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
from paddlenlp.transformers import *

ernie = ErnieModel.from_pretrained('ernie-1.0')
bert = BertModel.from_pretrained('bert-wwm-chinese')
albert = AlbertModel.from_pretrained('albert-chinese-tiny')
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
### 更多API使用文档

- [Transformer API](./docs/transformers.md)
  * 基于Transformer结构相关的预训练模型API，包含50+预训练模型，涵盖主流BERT/ERNIE/ALBERT/RoBERTa/Electra等经典结构和下游任务。
- [Data API](./docs/data.md)
  * 文本数据处理Pipeline的相关API说明。
- [Dataset API](./docs/datasets.md)
  * 数据集相关API，包含自定义数据集，数据集贡献与数据集快速加载等功能说明。
- [Embedding API](./docs/embeddings.md)
  * 词向量相关API，支持一键快速加载包预训练的中文词向量，VisulDL高维可视化等功能说明。
- [Metrics API](./docs/metrics.md)
  * 针对NLP场景的评估指标说明，与飞桨2.0框架高层API兼容。

更多的API示例及其使用说明请查阅[PaddleNLP官方文档](https://paddlenlp.readthedocs.io/)

## 丰富的应用示例

PaddleNLP基于PaddlePaddle 2.0全新API体系，提供了丰富的应用场景示例，帮助开发者按图索骥找到所需，更快上手飞桨2.0框架。
更多模型应用场景介绍请参考[PaddleNLP Examples](./examples/)。

### NLP 基础技术

| 任务   | 简介     |
| :------------  | ---- |
| [词法分析](./examples/lexical_analysis/) | 基于BiGRU-CRF模型实现了分词、词性标注和命名实体识的联合训练任务。输入是中文句子，而输出是句子中的词边界、词性与实体类别。 |
| [词向量](./exmaples/word_embedding/) | 提供60+预训练词向量，通过`paddlenlp.TokenEmbedding` API实现快速加载，可用于模型热启动或计算词之间的空间距离，支持通过VisualDL实现降维可视化。 |
| [语言模型](./examples/language_model/)  | 提供了基于RNN和Transformer-XL两种结构的语言模型，支持输入词序列计算其生成概率，并得到Perplexity(困惑度)，用于表示模型生成句子的流利程度。 |
| [语义解析](./examples/text_to_sql):star: | 语义解析Text-to-SQL是自然语言处理技术的核心任务之一，Text-to-SQL是语义解析的一个子方向，让机器自动将用户输入的自然语言问题转成数据库可操作的SQL查询语句，是实现基于数据库自动问答的核心模块。|

### NLP 核心技术

#### 文本分类 (Text Classification)
| 模型  | 简介    |
| :----- | ------ |
| [RNN/CNN/GRU/LSTM](./examples/text_classification/rnn) | 实现了经典的RNN, CNN, GRU, LSTM等经典文本分类结构。|
| [BiLSTM-Attention](./examples/text_classification/rnn) | 基于BiLSTM的网络结构通过引入注意力机制用于提升文本分类效果。 |
| [BERT/ERNIE](./examples/text_classification/pretrained_models) | 基于预训练模型的文本分类的模型，结合ChnSentiCorp数据提供了使用不同的预训练模型进行文本分类的Fine-tuning的示例。 |

#### 文本生成 (Text Generation)
| 模型        | 简介      |
| :------------ | ---------- |
| [Seq2Seq](./examples/text_generation/couplet) | 实现了经典的Seq2Seq with Attention的网络结构，在中文对联数据集上完成文本生成的示例。 |
| [VAE-Seq2Seq](./examples/text_generation/vae-seq2seq) | 在传统的Seq2Seq框架基础上，加入VAE结构以实现更加多样化的文本生成。|
| [ERNIE-GEN](./examples/text_generation/ernie-gen) | [ERNIE-GEN](https://arxiv.org/abs/2001.11314)是百度自研的生成式预训练模型，通过Global-Attention的方式解决训练和预测曝光偏差的问题，同时使用Multi-Flow Attention机制来分别进行Global和Context信息的交互，同时通过片段生成的方式来增加语义相关性。提供了使用中文|

#### 文本匹配 (Text Maching)
| 模型    | 简介       |
| :--------------- | ---------- |
| [SimNet](./examples/text_matching/simnet/)  | 百度提出的*基于表示(Representation-based)*的双塔语义匹配框架，主要使用BOW、CNN、GRNN等模块作为表示层。 |
| [ERNIE](./examples/text_matching/ernie_matching/) | 基于ERNIE使用LCQMC数据完成中文句对匹配任务，提供了Pointwise和Pairwise两种类型学习方式。 |
| [Sentence-BERT](./examples/text_matching/sentence_transformer/) | 基于Siamese双塔结构的[Sentence-BERT](https://arxiv.org/abs/1908.1008)文本匹配模型，可用于获取基于Transformer预训练模型的句子向量化表示。

#### 语义索引 (Semantic Indexing)

开放了完整的语义索引建设流程，并提供了In-Batch Negative和Hardest Negatives两种策略，开发者可基于该示例实现一个轻量级的语义索引系统，更多信息请查看[语义索引](./examples/semantic_indexing/)。

#### 信息抽取 (Information Extraction)
| 任务   | 简介     |
| :---------------  | ---- |
| [DuEE](./examples/information_extraction/DuEE/) | 基于**DuEE**数据集，使用预训练模型的方式提供句子级和篇章级的事件抽取示例。 |
| [DuIE](./examples/information_extraction/DuIE/) | 基于**DuIE**数据集，使用预训练模型的方式提供关系抽取示例。 |
| [快递单信息抽取](./examples/information_extraction/waybill_ie/) | 提供BiLSTM+CRF和预训练模型两种方式完成真实的快递单信息抽取案例。 |

### NLP 系统应用

#### 情感分析 (Sentiment Analysis)

| 模型      | 简介       |
| :--------- | ---------- |
| [SKEP](./examples/sentiment_analysis/skep/):star2: | 百度研究团队提出的基于情感知识增强的情感预训练算法，此算法采用无监督方法自动挖掘情感知识，然后利用情感知识构建预训练目标，从而让机器学会理解情感语义。SKEP为各类情感分析任务提供统一且强大的情感语义表示。 |

#### 阅读理解 (Machine Reading Comprehension)

| 任务   | 简介     |
| :-------------------  | ---- |
| [SQuAD](./examples/machine_reading_comprehension/SQuAD/) | 提供通过预训练模型在SQuAD 2.0数据集上微调的应用示例。 |
| [DuReader-yesno](./examples/machine_reading_comprehension/DuReader-yesno/) | 提供通过预训练模型在**千言数据集DuReader-yesno**上微调的应用示例。 |
| [DuReader-robust](./examples/machine_reading_comprehension/DuReader-robust/) | 提供通过预训练模型在**千言数据集DuReader-robust**上微调的应用示例。 |

#### 机器翻译 (Machine Translation)

| 模型    | 简介     |
| :--------------- | ------- |
| [Seq2Seq-Attn](./examples/machine_translation/seq2seq) | 使用编码器-解码器（Encoder-Decoder）结构, 同时使用了Attention机制来加强Decoder和Encoder之间的信息交互，Seq2Seq 广泛应用于机器翻译，自动对话机器人，文档摘要自动生成，图片描述自动生成等任务中。|
| [Transformer](./examples/machine_translation/transformer) | 基于PaddlePaddle框架的Transformer结构搭建的机器翻译模型，Transformer 计算并行度高，能解决学习长程依赖问题。并且模型框架集成了训练，验证，预测任务，功能完备，效果突出。|

#### 同传翻译 (Simultaneous Translation)

| 模型    | 简介     |
| :---------- | ------- |
| [STACL](./examples/simultaneous_translation/stacl) :star:| [STACL](https://www.aclweb.org/anthology/P19-1289/)是基于Prefix-to-Prefix框架的同传翻译模型，具备一定的隐式预测能力；结合Wait-k策略在保持较高的翻译质量的同时实现任意字级别的翻译延迟，并提供了可视化的Demo。|

#### 对话系统 (Dialogue System)

| 模型   | 简介      |
| :---------------- | ------|
| [PLATO-2](./examples/dialogue/plato-2) | [PLATO-2](https://arxiv.org/abs/2006.16779)是百度自研领先的基于课程学习两阶段方式训练的开放域对话预训练模型。|
| [PLATO-mini](./examples/dialogue/unified_transformer):star2: | 基于海量中文对话语料数据预训练的轻量级**中文**闲聊对话模型。|

### 拓展应用

#### 文本知识关联 (Text to Knowledge)

:star2:[**解语**](./examples/text_to_knowledge/)是由百度知识图谱部开发的文本知识关联框架，覆盖中文全词类的知识库和知识标注工具能够帮助开发者面对更加多元的应用场景，方便地融合自有知识体系，显著提升中文文本解析和挖掘效果，还可以便捷地利用知识增强机器学习模型效果。

- [TermTree: 中文全词类的知识库](./examples/text_to_knowledge/termtree):star2:
- [WordTag: 中文词类知识标注工具](./examples/text_to_knowledge/wordtag):star2:

#### 文本图学习 (Text Graph Learning)

| 模型   | 简介     |
| :------------ | ------- |
| [ERNIESage](./examples/text_graph/erniesage)| 基于[飞桨PGL图学习框架](https://github.com/PaddlePaddle/PGL)结合PaddleNLP Transformer API实现的文本图学习模型。|

#### 模型压缩 (Model Compression)

| 模型     | 简介    |
| :--------------- | ------- |
| [Distill-LSTM](./examples/model_compression/distill_lstm/) | 基于[Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)论文策略的实现，将BERT中英文分类的下游模型知识通过蒸馏的方式迁移至LSTM的小模型结构中，取得比LSTM单独训练更好的效果。|
| [OFA-BERT](./examples/model_compression/ofa/) :star2:| 基于PaddleSlim Once-For-ALL(OFA)策略对BERT在GLUE任务的下游模型进行压缩，在精度无损的情况下可减少33%参数量，达到模型小型化的提速的效果。 |


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
