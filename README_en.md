

English | [简体中文](./README.md)

<p align="center">
  <img src="./docs/imgs/paddlenlp.png" align="middle"  width="500" />
</p>

------------------------------------------------------------------------------------------
[![PyPI - PaddleNLP Version](https://img.shields.io/pypi/v/paddlenlp.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/paddlenlp/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paddlenlp)](https://pypi.org/project/paddlenlp/)
[![PyPI Status](https://pepy.tech/badge/paddlenlp/month)](https://pepy.tech/project/paddlenlp)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
![GitHub](https://img.shields.io/github/license/paddlepaddle/paddlenlp)

## News  <img src="./docs/imgs/news_icon.png" width="40"/>

* [2021-12-12] PaddleNLP v2.2 has been officially relealsed! :tada: For more information please refer to [Release Note](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.2.0).
* [2021-12-12] *End-to-end Question Answering Toolkit**🚀[RocketQA](https://github.com/PaddlePaddle/RocketQA) has been released！:tada:

## Introduction

**PaddleNLP** is an **easy-to-use** and **high performance** NLP library with **awesome** pre-trained Transformer models, supporting wide-range of NLP tasks from research to industrial applications.


* **Easy-to-Use API**
  - The API is fully integrated with PaddlePaddle 2.0 high-level API system. It minimizes the number of user actions required for common use cases like data loading, text pre-processing, awesome transfomer models, and fast inference, which enables developer to deal with text problems more productively.

* **Wide-range NLP Task Support**
  - PaddleNLP support NLP task from research to industrial applications, including Lexical Analysis, Text Classification, Text Matching, Text Generation, Information Extraction, Machine Translation, General Dialogue and Question Answering etc.

* **High Performance Distributed Training**
  -  We provide an industrial level training pipeline for super large-scale Transformer model based on **Auto Mixed Precision** and Fleet distributed training API by PaddlePaddle, which can support customized model pre-training efficiently.

## Community

### Special Interest Group (SIG)

Welcome to join [PaddleNLP SIG](https://iwenjuan.baidu.com/?code=bkypg8) for contribution, eg. Dataset, Models and Toolkit.

### Slack

To connect with other users and contributors, welcome to join our [Slack channel](https://paddlenlp.slack.com/).

### WeChat

Scan the QR code below with your Wechat⬇️. You can access to official technical exchange group. Look forward to your participation.

<div align="center">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/release/2.2/docs/imgs/wechat.png" width="188" height="188" />
</div>


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

![taskflow1](https://user-images.githubusercontent.com/11793384/159693816-fda35221-9751-43bb-b05c-7fc77571dd76.gif)

For more usage please refer to [Taskflow Docs](./docs/model_zoo/taskflow.md)

### Transformer API: Awesome Pre-trained Model Ecosystem

We provide **45+** network architectures and over **500+** pretrained models. Not only includes all the SOTA model like ERNIE, PLATO and SKEP released by Baidu, but also integrates most of the high quality Chinese pretrained model developed by other organizations. Use AutoModel to download pretrained mdoels of different architecture. We welcome all developers to contribute your Transformer models to PaddleNLP! 🤗

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

## ChangeLog

For more details about our release, please refer to [ChangeLog](./docs/changelog.md)

## Citation

If you find PaddleNLP useful in your research, please consider cite
```
@misc{=paddlenlp,
    title={PaddleNLP: An Easy-to-use and High Performance NLP Library},
    author={PaddleNLP Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleNLP}},
    year={2021}
}
```

## Acknowledge

We have borrowed from Hugging Face's [Transformer](https://github.com/huggingface/transformers)🤗 excellent design on pretrained models usage, and we would like to express our gratitude to the authors of Hugging Face and its open source community.

## License

PaddleNLP is provided under the [Apache-2.0 License](./LICENSE).
