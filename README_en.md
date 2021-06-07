English | [简体中文](./README.md)

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

* [2021-06-04] [ERNIE-Gram](https://arxiv.org/abs/2010.12148) pretrained model has been released! Install v2.0.2 to try it.
* [2021-05-20] PaddleNLP 2.0 has been officially relealsed! :tada: For more information please refer to [Release Note](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.0.0).

## Introduction

PaddleNLP is a powerful text domain library, which aims to accelerate NLP applications through easy-to-use API, rich application examples, and high performance distributed training. We also provide the NLP best practice based on PaddlePaddle 2.0 API system.


* **Easy-to-Use and End-to-End API**
  - The API is fully integrated with PaddlePaddle 2.0 high-level API system. It minimizes the number of user actions required for common use cases like data loading, text pre-processing, transformer model loading, training and deployment, which enables you to deal with text problems more productively.

* **Rich Application Examples**
  - Our model zoo covers mainstream NLP applications, including Lexical Analysis, Text Classification, Text Generation, Text Matching, Text Graph, Information Extraction, Machine Translation, General Dialogue and Question Answering etc.

* **High Performance Distributed Training**
  -  We provide a highly optimized ditributed training implementation for BERT with Fleet API, and mixed precision training strategy based on PaddlePaddle 2.0, it can fully utilize GPU clusters for large-scale model pre-training.


## Installation

### Prerequisites

* python >= 3.6
* paddlepaddle >= 2.1

More information about PaddlePaddle installation please refer to [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html)

### PIP Installation

```
pip install --upgrade paddlenlp -i https://pypi.org/simple
```

## Easy-to-use API


### Transformer API: Powerful Pre-trained Model Ecosystem

We provide 15+ network architecture and 67 pretrained model parameters, not only including all the SOTA pretrained model like ERNIE, PLATO and SKEP released by Baidu, but also most of useful Chinese pretrained model developed by other organizations.

```python
from paddlenlp.transformers import *

ernie = ErnieModel.from_pretrained('ernie-1.0')
ernie_gram = ErnieGramModel.from_pretrained('ernie-gram')
bert = BertModel.from_pretrained('bert-wwm-chinese')
albert = AlbertModel.from_pretrained('albert-chinese-tiny')
roberta = RobertaModel.from_pretrained('roberta-wwm-ext')
electra = ElectraModel.from_pretrained('chinese-electra-small')
gpt = GPTForPretraining.from_pretrained('gpt-cpm-large-cn')
```

PaddleNLP also provides unified API experience for NLP task like semantic representation, text classification, sentence matching, sequence labeling, question answering, etc.

```python
import paddle
from paddlenlp.transformers import ErnieTokenizer, ErnieModel

tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
text = tokenizer('natural language understanding')

# Semantic Representation
model = ErnieModel.from_pretrained('ernie-1.0')
pooled_output, sequence_output = model(input_ids=paddle.to_tensor([text['input_ids']]))
# Text Classificaiton and Matching
model = ErnieForSequenceClassifiation.from_pretrained('ernie-1.0')
# Sequence Labeling
model = ErnieForTokenClassifiation.from_pretrained('ernie-1.0')
# Question Answering
model = ErnieForQuestionAnswering.from_pretrained('ernie-1.0')
```


For more pretrained model usage, please refer to [Transformer API](./docs/model_zoo/transformers.rst)



### Dataset API: Rich Dataset Integration and Quick Loading

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

For more `TokenEmbedding` usage, please refer to [Embedding API](./docs/embeddings.md)

### More API Usage

- [Transformer API](./docs/model_zoo/transformers.rst)
- [Data API](./docs/data.md)
- [Dataset API](./docs/datasets.md)
- [Embedding API](./docs/model_zoo/embeddings.md)
- [Metrics API](./docs/metrics.md)

Please find more API Reference from our [readthedocs](https://paddlenlp.readthedocs.io/).

## Rich Application Examples

PaddleNLP provide rich application examples covers mainstream NLP task to help developer accelerate problem solving.

### NLP Basic Technique

- [Word Embedding](./examples/word_embedding/)
- [Lexical Analysis](./examples/lexical_analysis/)
- [Language Model](./examples/language_model/)
- [Semantic Parsing (Text to SQL)](./examples/text_to_sql):star:


### NLP Core Technique

- [Text Classification](./examples/text_classification/)
- [Text Matching](./examples/text_matching/)
- [Text Generation](./examples/text_generation/)
- [Semantic Indexing](./examples/semantic_indexing/)
- [Information Extraction](./examples/information_extraction/)

### NLP Application in Real System

- [Sentiment Analysis](./examples/sentiment_analysis/skep/):star2:
- [General Dialogue System](./examples/dialogue/)
- [Machine Translation](./examples/machine_translation/)
- [Simultaneous Translation](././examples/simultaneous_translation/)
- [Machine Reading Comprehension](./examples/machine_reading_comprehension/)

### Extention Application

- [Text Knowledge Linking](./examples/text_to_knowledge/):star2:
- [Machine Reading Comprehension](./examples/machine_reading_comprehension)
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

For more information about our release, please refer to [ChangeLog](./docs/changelog.md)

## License

PaddleNLP is provided under the [Apache-2.0 License](./LICENSE).
