English | [简体中文](./README.md)

<p align="center">
  <img src="./docs/imgs/paddlenlp.png" width="718" height ="100" />
</p>

------------------------------------------------------------------------------------------
[![PyPI - PaddleNLP Version](https://img.shields.io/pypi/v/paddlenlp.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/paddlenlp/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paddlenlp)](https://pypi.org/project/paddlenlp/)
[![PyPI Status](https://pepy.tech/badge/paddlenlp/month)](https://pepy.tech/project/paddlenlp)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
![GitHub](https://img.shields.io/github/license/paddlepaddle/paddlenlp)

## News  <img src="./docs/imgs/news_icon.png" width="40"/>

* [2021-12-12] PaddleNLP 2.2 has been officially relealsed! :tada: For more information please refer to [Release Note](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.2.0).
* [2021-10-12] PaddleNLP 2.1 has been officially relealsed! :tada: For more information please refer to [Release Note](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.1.0).

## Introduction

**PaddleNLP** is a powerful NLP library with **Awesome** pre-trained Transformer models and easy-to-use interface, supporting wide-range of NLP tasks from research to industrial applications.


* **Easy-to-Use API**
  - The API is fully integrated with PaddlePaddle 2.0 high-level API system. It minimizes the number of user actions required for common use cases like data loading, text pre-processing, awesome transfomer models, and fast inference, which enables developer to deal with text problems more productively.

* **Wide-range NLP Task Support**
  - PaddleNLP support NLP task from research to industrial applications, including Lexical Analysis, Text Classification, Text Matching, Text Generation, Information Extraction, Machine Translation, General Dialogue and Question Answering etc.

* **High Performance Distributed Training**
  -  We provide an industrial level training pipeline for super large-scale Transformer model based on **Auto Mixed Precision** and Fleet distributed training API by PaddlePaddle, which can support customized model pre-training efficiently.

## Installation

### Prerequisites

* python >= 3.6
* paddlepaddle >= 2.2

More information about PaddlePaddle installation please refer to [PaddlePaddle's Website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html).

### Python pip Installation

```
pip install --upgrade paddlenlp
```

## Easy-to-use API

### Taskflow：Off-the-shelf Industial NLP Pre-built Task

Taskflow aims to provide **off-the-shelf** NLP pre-built task covering NLU and NLG scenario, in the meanwhile with extreamly fast infernece satisfying industrial applications.

```python
from paddlenlp import Taskflow

# Chinese Word Segmentation
seg = Taskflow("word_segmentation")
seg("第十四届全运会在西安举办")
>>> ['第十四届', '全运会', '在', '西安', '举办']

# POS Tagging
tag = Taskflow("pos_tagging")
tag("第十四届全运会在西安举办")
>>> [('第十四届', 'm'), ('全运会', 'nz'), ('在', 'p'), ('西安', 'LOC'), ('举办', 'v')]

# Named Entity Recognition
ner = Taskflow("ner")
ner("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")
>>> [('《', 'w'), ('孤女', '作品类_实体'), ('》', 'w'), ('是', '肯定词'), ('2010年', '时间类'), ('九州出版社', '组织机构类'), ('出版', '场景事件'), ('的', '助词'), ('小说', '作品类_概念'), ('，', 'w'), ('作者', '人物类_概念'), ('是', '肯定词'), ('余兼羽', '人物类_实体')]

# Dependency Parsing
ddp = Taskflow("dependency_parsing")
ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫")
>>> [{'word': ['9月9日', '上午', '纳达尔', '在', '亚瑟·阿什球场', '击败', '俄罗斯', '球员', '梅德韦杰夫'], 'head': [2, 6, 6, 5, 6, 0, 8, 9, 6], 'deprel': ['ATT', 'ADV', 'SBV', 'MT', 'ADV', 'HED', 'ATT', 'ATT', 'VOB']}]

# Sentiment Analysis
senta = Taskflow("sentiment_analysis")
senta("这个产品用起来真的很流畅，我非常喜欢")
>>> [{'text': '这个产品用起来真的很流畅，我非常喜欢', 'label': 'positive', 'score': 0.9938690066337585}]
```

For more usage please refer to [Taskflow Docs](./docs/model_zoo/taskflow.md)

### Transformer API: Awesome Pre-trained Model Ecosystem

We provide **30** network architectures and over **100** pretrained models. Not only includes all the SOTA model like ERNIE, PLATO and SKEP released by Baidu, but also integrates most of the high quality Chinese pretrained model developed by other organizations. Use AutoModel to download pretrained mdoels of different architecture. We welcome all developers to contribute your Transformer models to PaddleNLP! 🤗

```python
from paddlenlp.transformers import *

ernie = AutoModel.from_pretrained('ernie-1.0')
ernie_gram = AutoModel.from_pretrained('ernie-gram-zh')
bert = AutoModel.from_pretrained('bert-wwm-chinese')
albert = AutoModel.from_pretrained('albert-chinese-tiny')
roberta = AutoModel.from_pretrained('roberta-wwm-ext')
electra = AutoModel.from_pretrained('chinese-electra-small')
gpt = AutoModelForPretraining.from_pretrained('gpt-cpm-large-cn')
```

PaddleNLP also provides unified API experience for NLP task like semantic representation, text classification, sentence matching, sequence labeling, question answering, etc.

```python
import paddle
from paddlenlp.transformers import *

tokenizer = AutoTokenizer.from_pretrained('ernie-1.0')
text = tokenizer('natural language understanding')

# Semantic Representation
model = AutoModel.from_pretrained('ernie-1.0')
sequence_output, pooled_output = model(input_ids=paddle.to_tensor([text['input_ids']]))
# Text Classificaiton and Matching
model = AutoModelForSequenceClassification.from_pretrained('ernie-1.0')
# Sequence Labeling
model = AutoModelForTokenClassification.from_pretrained('ernie-1.0')
# Question Answering
model = AutoModelForQuestionAnswering.from_pretrained('ernie-1.0')
```

For more pretrained model usage, please refer to [Transformer API](./docs/model_zoo/transformers.rst)

### Dataset API: Abundant Dataset Integration and Quick Loading

```python
from paddlenlp.datasets import load_dataset

train_ds, dev_ds, test_ds = load_dataset("chnsenticorp", splits=["train", "dev", "test"])
```

For more dataset API usage please refer to [Dataset API](./docs/datasets.md).

### Embedding API: Quick Loading for Word Embedding

```python
from paddlenlp.embeddings import TokenEmbedding

wordemb = TokenEmbedding("fasttext.wiki-news.target.word-word.dim300.en")
wordemb.cosine_sim("king", "queen")
>>> 0.77053076
wordemb.cosine_sim("apple", "rail")
>>> 0.29207364
```

For more `TokenEmbedding` usage, please refer to [Embedding API](./docs/model_zoo/embeddings.md)

### More API Usage

- [Transformer API](./docs/model_zoo/transformers.rst)
- [Data API](./docs/data.md)
- [Dataset API](./docs/datasets.md)
- [Embedding API](./docs/model_zoo/embeddings.md)
- [Metrics API](./docs/metrics.md)

Please find more API Reference from our [readthedocs](https://paddlenlp.readthedocs.io/).

## Wide-range NLP Task Support

PaddleNLP provides rich application examples covering mainstream NLP task to help developers accelerate problem solving.

### NLP Basic Technique

- [Word Embedding](./examples/word_embedding/)
- [Lexical Analysis](./examples/lexical_analysis/)
- [Dependency Parsing](./examples/dependency_parsing/)
- [Language Model](./examples/language_model/)
- [Semantic Parsing (Text to SQL)](./examples/text_to_sql):star:
- [Text Classification](./examples/text_classification/)
- [Text Matching](./examples/text_matching/)
- [Text Generation](./examples/text_generation/)
- [Text Correction](./examples/text_correction/):star:
- [Semantic Indexing](./examples/semantic_indexing/)
- [Information Extraction](./examples/information_extraction/)

### NLP System

- [Sentiment Analysis](./examples/sentiment_analysis/):star2:
- [General Dialogue System](./examples/dialogue/)
- [Machine Translation](./examples/machine_translation/)
- [Simultaneous Translation](././examples/simultaneous_translation/)
- [Machine Reading Comprehension](./examples/machine_reading_comprehension/)

### NLP Extented Applications

- [Few-shot Learning](./examples/few_shot/):star2:
- [Text Knowledge Mining](./examples/text_to_knowledge/):star2:
- [Model Compression](./examples/model_compression/)
- [Text Graph Learning](./examples/text_graph/erniesage/)
- [Time Series Prediction](./examples/time_series/)

## Tutorials

Please refer to our official AI Studio account for more interactive tutorials: [PaddleNLP on AI Studio](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)

* [What's Seq2Vec?](https://aistudio.baidu.com/aistudio/projectdetail/1283423) shows how to use simple API to finish LSTM model and solve sentiment analysis task.

* [Sentiment Analysis with ERNIE](https://aistudio.baidu.com/aistudio/projectdetail/1294333) shows how to exploit the pretrained ERNIE to solve sentiment analysis problem.

* [Waybill Information Extraction with BiGRU-CRF Model](https://aistudio.baidu.com/aistudio/projectdetail/1317771) shows how to make use of Bi-GRU plus CRF to finish information extraction task.

* [Waybill Information Extraction with ERNIE](https://aistudio.baidu.com/aistudio/projectdetail/1329361) shows how to use ERNIE, the Chinese pre-trained model improve information extraction performance.

* [Use TCN Model to predict COVID-19 confirmed cases](https://aistudio.baidu.com/aistudio/projectdetail/1290873)

## Community

### Special Interest Group (SIG)

Welcome to join [PaddleNLP SIG](https://iwenjuan.baidu.com/?code=bkypg8) for contribution, eg. Dataset, Models and Toolkit.

### Slack
To connect with other users and contributors, welcome to join our [Slack channel](https://paddlenlp.slack.com/).

### QQ
Join our QQ Technical Group for technical exchange right now! ⬇️

<div align="center">
  <img src="./docs/imgs/qq.png" width="200" height="200" />
</div>

## ChangeLog

For more details about our release, please refer to [ChangeLog](./docs/changelog.md)

## Acknowledge

We have borrowed from Hugging Face's [Transformer](https://github.com/huggingface/transformers)🤗 excellent design on pretrained models usage, and we would like to express our gratitude to the authors of Hugging Face and its open source community.

## License

PaddleNLP is provided under the [Apache-2.0 License](./LICENSE).
