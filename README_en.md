English | [简体中文](./README.md)

<p align="center">
  <img src="./docs/imgs/paddlenlp.png" width="520" height ="100" />
</p>

---------------------------------------------------------------------------------

[![PyPI - PaddleNLP Version](https://img.shields.io/pypi/v/paddlenlp.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/paddlenlp/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paddlenlp)](https://pypi.org/project/paddlenlp/)
[![PyPI Status](https://pepy.tech/badge/paddlenlp/month)](https://pepy.tech/project/paddlenlp)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
![GitHub](https://img.shields.io/github/license/paddlepaddle/paddlenlp)

## Introduction

PaddleNLP aims to accelerate NLP applications through powerful model zoo, easy-to-use API and high performance distributed training. It's also the NLP best practice for PaddlePaddle 2.0 API system.

## Features

* **Powerful Model Zoo for Rich Senario**
  - Our Model Zoo covers mainstream NLP applications, including Lexical Analysis, Text Classification, Text Generation, Text Matching, Text Graph, Information Extraction, Machine Translation, General Dialogue and Question Answering etc.

* **Easy-to-Use and End-to-End API**
  - The API is fully integrated with PaddlePaddle 2.0 high-level API system. It minimizes the number of user actions required for common use cases like data loading, text pre-processing, training and evaluation, which enables you to deal with text problems more productively.

* **High Performance and Distributed Training**
-  We provide a highly optimized ditributed training implementation for BERT with Fleet API, and mixed precision training strategy based on PaddlePaddle 2.0, it can fully utilize GPU clusters for large-scale model pre-training.


## Installation

### Prerequisites

* python >= 3.6
* paddlepaddle >= 2.0.1

More information about PaddlePaddle installation please refer to [PaddlePaddle Install](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html)

### PIP Installation

```
pip install --upgrade paddlenlp -i https://pypi.org/simple
```

## Quick Start

### Quick Dataset Loading

```python
from paddlenlp.datasets import load_dataset

train_ds, dev_ds, test_ds = load_dataset("chnsenticorp", splits=["train", "dev", "test"])
```

For more dataset API usage please refer to [Dataset API](./docs/datasets.md).

### Pre-trained Text Embedding Loading

```python

from paddlenlp.embeddings import TokenEmbedding

wordemb = TokenEmbedding("fasttext.wiki-news.target.word-word.dim300.en")
wordemb.cosine_sim("king", "queen")
>>> 0.77053076
wordemb.cosine_sim("apple", "rail")
>>> 0.29207364
```

For more TokenEmbedding usage, please refer to [Embedding API](./docs/embeddings.md)

### Rich Chinese Pre-trained Models

```python
from paddlenlp.transformers import ErnieModel, BertModel, RobertaModel, ElectraModel, GPT2ForPretraining

ernie = ErnieModel.from_pretrained('ernie-1.0')
bert = BertModel.from_pretrained('bert-wwm-chinese')
roberta = RobertaModel.from_pretrained('roberta-wwm-ext')
electra = ElectraModel.from_pretrained('chinese-electra-small')
gpt2 = GPT2ForPretraining.from_pretrained('gpt2-base-cn')
```

For more pretrained model selection, please refer to [Transformer API](./docs/transformers.md)

### Extract Feature Through Pre-trained Model

```python
import paddle
from paddlenlp.transformers import ErnieTokenizer, ErnieModel

tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
model = ErnieModel.from_pretrained('ernie-1.0')

text = tokenizer('自然语言处理')
pooled_output, sequence_output = model.forward(input_ids=paddle.to_tensor([text['input_ids']]))
```

## Model Zoo and Applications

For model zoo introduction please refer to[PaddleNLP Model Zoo](./docs/model_zoo.md). As for applicaiton senario please refer to [PaddleNLP Examples](./examples/)。

- [Word Embedding](./examples/word_embedding/)
- [Lexical Analysis](./examples/lexical_analysis/)
- [Named Entity Recognition](./examples/information_extraction/msra_ner/)
- [Language Model](./examples/language_model/)
- [Text Classification](./examples/text_classification/)
- [Text Gneeration](./examples/text_generation/)
- [Semantic Maching](./examples/text_matching/)
- [Text Graph](./examples/text_graph/erniesage/)
- [Information Extraction](./examples/information_extraction/)
- [General Dialogue](./examples/dialogue/)
- [Machine Translation](./examples/machine_translation/)
- [Machine Readeng Comprehension](./examples/machine_reading_comprehension/)

## Advanced Application

- [Model Compression](./examples/model_compression/)

## API Usage

- [Transformer API](./docs/transformers.md)
- [Data API](./docs/data.md)
- [Dataset API](./docs/datasets.md)
- [Embedding API](./docs/embeddings.md)
- [Metrics API](./docs/metrics.md)


## Tutorials

Please refer to our official AI Studio account for more interactive tutorials: [PaddleNLP on AI Studio](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)

* [What's Seq2Vec?](https://aistudio.baidu.com/aistudio/projectdetail/1283423) shows how to use simple API to finish LSTM model and solve sentiment analysis task.

* [Sentiment Analysis with ERNIE](https://aistudio.baidu.com/aistudio/projectdetail/1294333) shows how to exploit the pretrained ERNIE to solve sentiment analysis problem.

* [Waybill Information Extraction with BiGRU-CRF Model](https://aistudio.baidu.com/aistudio/projectdetail/1317771) shows how to make use of Bi-GRU plus CRF to finish information extraction task.

* [Waybill Information Extraction with ERNIE](https://aistudio.baidu.com/aistudio/projectdetail/1329361) shows how to use ERNIE, the Chinese pre-trained model improve information extraction performance.

* [Use TCN Model to predict COVID-19 confirmed cases](https://aistudio.baidu.com/aistudio/projectdetail/1290873)


## Community

### Special Interest Group(SIG)

Welcome to join [PaddleNLP SIG](https://iwenjuan.baidu.com/?code=bkypg8) for contribution, eg. Dataset, Models and Toolkit.

### Slack
To connect with other users and contributors, welcome to join our [Slack channel](https://paddlenlp.slack.com/).

### QQ
Join our QQ Technical Group for technical exchange right now! ⬇️

<div align="center">
  <img src="./docs/imgs/qq.png" width="200" height="200" />
</div>

## License

PaddleNLP is provided under the [Apache-2.0 License](./LICENSE).
