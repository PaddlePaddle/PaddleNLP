简体中文 | [English](./README_en.md)

<p align="center">
  <img src="./docs/imgs/paddlenlp.png" width="718" height ="100" />
</p>

------------------------------------------------------------------------------------------
[![PyPI - PaddleNLP Version](https://img.shields.io/pypi/v/paddlenlp.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/paddlenlp/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paddlenlp)](https://pypi.org/project/paddlenlp/)
[![PyPI Status](https://pepy.tech/badge/paddlenlp/month)](https://pepy.tech/project/paddlenlp)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
![GitHub](https://img.shields.io/github/license/paddlepaddle/paddlenlp)

## News  <img src="./docs/imgs/news_icon.png" width="40"/>
* [2021-08-22][《千言：面向事实一致性的生成评测比赛》](https://aistudio.baidu.com/aistudio/competition/detail/105)正式开赛啦🔥🔥🔥，欢迎大家踊跃报名!! [PaddleNLP比赛基线地址](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_generation/unimo-text)
* [2021-08-22] PaddleNLP 2.0.8版本已发布！:tada:更多详细升级信息请查看[Release Note](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.0.8).
* [2021-06-07]《基于深度学习的自然语言处理》直播打卡课正在进行中🔥🔥🔥，快来打卡吧：[https://aistudio.baidu.com/aistudio/course/introduce/24177](https://aistudio.baidu.com/aistudio/course/introduce/24177)
* [2021-06-04] 新增多粒度语言知识预训练模型[ERNIE-Gram](https://arxiv.org/abs/2010.12148)，多项中文NLP任务取得SOTA成绩，获取2.0.2版本快速体验吧！

## 简介

PaddleNLP 2.0是飞桨生态的文本领域核心库，具备**易用的文本领域API**，**多场景的应用示例**、和**高性能分布式训练**三大特点，旨在提升开发者文本领域的开发效率，并提供基于飞桨2.0核心框架的NLP任务最佳实践。

- **易用的文本领域API**
  - 提供从数据加载、文本预处理、模型组网评估、到推理加速的领域API：支持丰富中文数据集加载的[Dataset API](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html)；灵活高效地完成数据预处理的[Data API](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.data.html)；提供60+预训练模型的[Transformer API](./docs/model_zoo/transformers.rst)等，可大幅提升NLP任务建模与迭代的效率。


- **多场景的应用示例**
  - 覆盖从学术到工业级的NLP[应用示例](#多场景的应用示例)，涵盖从NLP基础技术、NLP核心技术、NLP系统应用以及相关拓展应用。全面基于飞桨核心框架2.0全新API体系开发，为开发提供飞桨2.0框架在文本领域的最佳实践。


- **高性能分布式训练**
  - 基于飞桨核心框架领先的自动混合精度优化策略，结合分布式Fleet API，支持4D混合并行策略，可高效地完成超大规模参数的模型训练。

## 安装

### 环境依赖

- python >= 3.6
- paddlepaddle >= 2.1.0

### pip安装

```
pip install --upgrade paddlenlp
```

更多关于PaddlePaddle和PaddleNLP安装的详细教程请查看[Installation](./docs/get_started/installation.rst)。

## 易用的文本领域API

### Transformer API: 强大的预训练模型生态底座

覆盖**15**个网络结构和**67**个预训练模型参数，既包括百度自研的预训练模型如ERNIE系列, PLATO, SKEP等，也涵盖业界主流的中文预训练模型。也欢迎开发者进预训练模贡献！🤗

```python
from paddlenlp.transformers import *

ernie = ErnieModel.from_pretrained('ernie-1.0')
ernie_gram = ErnieGramModel.from_pretrained('ernie-gram-zh')
bert = BertModel.from_pretrained('bert-wwm-chinese')
albert = AlbertModel.from_pretrained('albert-chinese-tiny')
roberta = RobertaModel.from_pretrained('roberta-wwm-ext')
electra = ElectraModel.from_pretrained('chinese-electra-small')
gpt = GPTForPretraining.from_pretrained('gpt-cpm-large-cn')
```

对预训练模型应用范式如语义表示、文本分类、句对匹配、序列标注、问答等，提供统一的API体验。

```python
import paddle
from paddlenlp.transformers import ErnieTokenizer, ErnieModel

tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
text = tokenizer('自然语言处理')

# 语义表示
model = ErnieModel.from_pretrained('ernie-1.0')
sequence_output, pooled_output = model(input_ids=paddle.to_tensor([text['input_ids']]))
# 文本分类 & 句对匹配
model = ErnieForSequenceClassification.from_pretrained('ernie-1.0')
# 序列标注
model = ErnieForTokenClassification.from_pretrained('ernie-1.0')
# 问答
model = ErnieForQuestionAnswering.from_pretrained('ernie-1.0')
```

请参考[Transformer API文档](./docs/model_zoo/transformers.rst)查看目前支持的预训练模型结构、参数和详细用法。

### Dataset API: 丰富的中文数据集

Dataset API提供便捷、高效的数据集加载功能；内置[千言数据集](https://www.luge.ai/)，提供丰富的面向自然语言理解与生成场景的中文数据集，为NLP研究人员提供一站式的科研体验。

```python
from paddlenlp.datasets import load_dataset

train_ds, dev_ds, test_ds = load_dataset("chnsenticorp", splits=["train", "dev", "test"])

train_ds, dev_ds = load_dataset("lcqmc", splits=["train", "dev"])
```

可参考[Dataset文档](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html) 查看更多数据集。

### Embedding API: 一键加载预训练词向量

```python
from paddlenlp.embeddings import TokenEmbedding

wordemb = TokenEmbedding("w2v.baidu_encyclopedia.target.word-word.dim300")
print(wordemb.cosine_sim("国王", "王后"))
>>> 0.63395125
wordemb.cosine_sim("艺术", "火车")
>>> 0.14792643
```

内置50+中文词向量，覆盖多种领域语料、如百科、新闻、微博等。更多使用方法请参考[Embedding文档](./docs/model_zoo/embeddings.md)。

### 更多API使用文档

- [Data API](./docs/data.md): 提供便捷高效的文本数据处理功能
- [Metrics API](./docs/metrics.md): 提供NLP任务的评估指标，与飞桨高层API兼容。

更多的API示例与使用说明请查阅[PaddleNLP官方文档](https://paddlenlp.readthedocs.io/)

## 多场景的应用示例

PaddleNLP提供了多粒度、多场景的NLP应用示例，面向动态图模式和全新的API体系开发，更加简单易懂。
涵盖了[NLP基础技术](#nlp-基础技术)、[NLP核心技术](#nlp-核心技术)、[NLP系统应用](#nlp-系统应用)以及文本相关的拓展应用如[模型压缩](./examples/model_compression/)、与知识库结合的[文本知识关联](./examples/text_to_knowledge)、与图结合的[文本图学习](./examples/text_graph/)等。

### NLP 基础技术

| 任务   | 简介     |
| :------------  | ---- |
| [词向量](./examples/word_embedding/) | 利用`TokenEmbedding API`展示如何快速计算词之间语义距离和词的特征提取。 |
| [词法分析](./examples/lexical_analysis/) | 基于BiGRU-CRF模型实现了分词、词性标注和命名实体识的联合训练任务。 |
| [语言模型](./examples/language_model/)  | 提供了基于[RNNLM](./examples/language_model/rnnlm)和[Transformer-XL](./examples/language_model/transformer-xl)两种结构的语言模型，支持输入词序列计算其生成概率，可用于表示模型生成句子的流利程度。 |
| [语义解析](./examples/text_to_sql):star: | 语义解析Text-to-SQL任务是让机器自动让自然语言问题转换数据库可操作的SQL查询语句，是实现基于数据库自动问答的核心模块。|

### NLP 核心技术

#### 文本分类 (Text Classification)
| 模型  | 简介    |
| :----- | ------ |
| [RNN/CNN/GRU/LSTM](./examples/text_classification/rnn) | 实现了经典的RNN, CNN, GRU, LSTM等经典文本分类结构。|
| [BiLSTM-Attention](./examples/text_classification/rnn) | 基于BiLSTM网络结构引入注意力机制提升文本分类效果。 |
| [BERT/ERNIE](./examples/text_classification/pretrained_models) | 提供基于预训练模型的文本分类任务实现，包含训练、预测和推理部署的全流程应用。 |

#### 文本匹配 (Text Matching)
| 模型    | 简介       |
| :--------------- | ---------- |
| [SimNet](./examples/text_matching/simnet/)  | 百度自研的语义匹配框架，使用BOW、CNN、GRNN等核心网络作为表示层，在百度内搜索、推荐等多个应用场景得到广泛易用。|
| [ERNIE](./examples/text_matching/ernie_matching/) | 基于ERNIE使用LCQMC数据完成中文句对匹配任务，提供了Pointwise和Pairwise两种类型学习方式。 |
| [Sentence-BERT](./examples/text_matching/sentence_transformers/) | 提供基于Siamese双塔结构的文本匹配模型[Sentence-BERT](https://arxiv.org/abs/1908.1008)实现，可用于获取文本的向量化表示。

#### 文本生成 (Text Generation)
| 模型        | 简介      |
| :------------ | ---------- |
| [Seq2Seq](./examples/text_generation/couplet) | 实现了经典的Seq2Seq with Attention的网络结构，并提供在自动对联的文本生成应用示例。 |
| [VAE-Seq2Seq](./examples/text_generation/vae-seq2seq) | 在Seq2Seq框架基础上，加入VAE结构以实现更加多样化的文本生成。|
| [ERNIE-GEN](./examples/text_generation/ernie-gen) | [ERNIE-GEN](https://arxiv.org/abs/2001.11314)是百度NLP提出的基于多流(multi-flow)机制生成完整语义片段的预训练模型，基于该模型实现了提供了智能写诗的应用示例。|

#### 语义索引 (Semantic Indexing)

提供一套完整的语义索引开发流程，并提供了In-Batch Negative和Hardest Negatives两种策略，开发者可基于该示例实现一个轻量级的语义索引系统，更多信息请查看[语义索引应用示例](./examples/semantic_indexing/)。

#### 信息抽取 (Information Extraction)
| 任务   | 简介     |
| :---------------  | ---- |
| [DuEE](./examples/information_extraction/DuEE/) | 基于[DuEE](https://link.springer.com/chapter/10.1007/978-3-030-60457-8_44)数据集，使用预训练模型的方式提供句子级和篇章级的事件抽取示例。 |
| [DuIE](./examples/information_extraction/DuIE/) | 基于[DuIE](http://tcci.ccf.org.cn/conference/2019/papers/EV10.pdf)数据集，使用预训练模型的方式提供关系抽取示例。 |
| [快递单信息抽取](./examples/information_extraction/waybill_ie/) | 提供BiLSTM+CRF和预训练模型两种方式完成真实的快递单信息抽取案例。 |

### NLP 系统应用

#### 情感分析 (Sentiment Analysis)

| 模型      | 简介       |
| :--------- | ---------- |
| [SKEP](./examples/sentiment_analysis/skep/):star2: | [SKEP](https://arxiv.org/abs/2005.05635)是百度提出的基于情感知识增强的预训练算法，利用无监督挖掘的海量情感知识构建预训练目标，让模型更好理解情感语义，可为各类情感分析任务提供统一且强大的情感语义表示。 |

#### 阅读理解 (Machine Reading Comprehension)

| 任务   | 简介     |
| :-------------------  | ---- |
| [SQuAD](./examples/machine_reading_comprehension/SQuAD/) | 提供预训练模型在SQuAD 2.0数据集上微调的应用示例。 |
| [DuReader-yesno](./examples/machine_reading_comprehension/DuReader-yesno/) | 提供预训练模型在**千言数据集DuReader-yesno**上微调的应用示例。 |
| [DuReader-robust](./examples/machine_reading_comprehension/DuReader-robust/) | 提供预训练模型在**千言数据集DuReader-robust**上微调的应用示例。 |

#### 文本翻译 (Text Translation)

| 模型    | 简介     |
| :--------------- | ------- |
| [Seq2Seq-Attn](./examples/machine_translation/seq2seq) | 提供了[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025v5)基于注意力机制改进的Seq2Seq经典神经网络机器翻译模型实现。|
| [Transformer](./examples/machine_translation/transformer) | 提供了基于[Attention Is All You Need](https://arxiv.org/abs/1706.03762)论文的Transformer机器翻译实现，包含了完整的训练到推理部署的全流程实现。|

#### 同传翻译 (Simultaneous Translation)

| 模型    | 简介     |
| :---------- | ------- |
| [STACL](./examples/simultaneous_translation/stacl) :star:| [STACL](https://www.aclweb.org/anthology/P19-1289/)是百度自研的基于Prefix-to-Prefix框架的同传翻译模型，结合Wait-k策略可以在保持较高的翻译质量的同时实现任意字级别的翻译延迟，并提供了轻量级同声传译系统搭建教程。|

#### 对话系统 (Dialogue System)

| 模型   | 简介      |
| :---------------- | ------|
| [PLATO-2](./examples/dialogue/plato-2) | [PLATO-2](https://arxiv.org/abs/2006.16779)是百度自研领先的基于课程学习两阶段方式训练的开放域对话预训练模型。|
| [PLATO-mini](./examples/dialogue/unified_transformer):star2: | 基于6层UnifiedTransformer预训练结构，结合海量中文对话语料数据预训练的轻量级**中文**闲聊对话模型。|

### 拓展应用

#### 文本知识关联 (Text to Knowledge)

:star2:[**解语**](./examples/text_to_knowledge/)是由百度知识图谱部开发的文本知识关联框架，覆盖中文全词类的知识库和知识标注工具，能够帮助开发者面对更加多元的应用场景，方便地融合自有知识体系，显著提升中文文本解析和挖掘效果，还可以便捷地利用知识增强机器学习模型效果。

- [TermTree: 中文全词类的知识库](./examples/text_to_knowledge/termtree):star2:
- [WordTag: 中文词类知识标注工具](./examples/text_to_knowledge/wordtag):star2:

#### 文本图学习 (Text Graph Learning)

| 模型   | 简介     |
| :------------ | ------- |
| [ERNIESage](./examples/text_graph/erniesage)| 基于[飞桨PGL](https://github.com/PaddlePaddle/PGL)图学习框架结合PaddleNLP Transformer API实现的文本图学习模型。|

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


## 社区贡献与技术交流

### 特殊兴趣小组

- 欢迎您加入PaddleNLP的SIG社区，贡献优秀的模型实现、公开数据集、教程与案例等。

### QQ

- 现在就加入PaddleNLP的QQ技术交流群，一起交流NLP技术吧！⬇️

<div align="center">
  <img src="./docs/imgs/qq.png" width="200" height="200" />
</div>  



## 版本更新

更多版本更新说明请查看[ChangeLog](./docs/changelog.md)

## License

PaddleNLP遵循[Apache-2.0开源协议](./LICENSE)。
