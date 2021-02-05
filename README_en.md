English | [简体中文](./README.md)

<p align="center">
  <img src="./docs/imgs/paddlenlp.png" width="520" height ="100" />
</p>

---------------------------------------------------------------------------------

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

## Introduction

PaddleNLP aims to accelerate NLP applications through powerful model zoo, easy-to-use API with detailed tutorials. It's also the NLP best practice for PaddlePaddle 2.0 API system.

## Features

* **Rich and Powerful Model Zoo**
  - Our Model Zoo covers mainstream NLP applications, including Lexical Analysis, Syntactic Parsing, Machine Translation, Text Classification, Text Generation, Text Matching, General Dialogue and Question Answering etc.

* **Easy-to-use API**
  - The API is fully integrated with PaddlePaddle high-level API system. It minimizes the number of user actions required for common use cases like data loading, text pre-processing, training and evaluation. which enables you to deal with text problems more productively.

* **High Performance and Large-scale Training**
  - We provide a highly optimized ditributed training implementation for BERT with Fleet API, it can fully utilize GPU clusters for large-scale model pre-training. Please refer to our [benchmark](./benchmark/bert) for more information.

* **Detailed Tutorials and Industrial Practices**
  - We offers detailed and interactable notebook tutorials to show you the best practices of PaddlePaddle 2.0.

## Installation

### Prerequisites

* python >= 3.6
* paddlepaddle >= 2.0.0

```
pip install paddlenlp>=2.0.0rc
```

## Quick Start

### Quick Dataset Loading

```python
from paddlenlp.datasets import ChnSentiCorp

train_ds, test_ds = ChnSentiCorp.get_datasets(['train','test'])
```

### Chinese Text Emebdding Loading

```python

from paddlenlp.embeddings import TokenEmbedding

wordemb = TokenEmbedding("w2v.baidu_encyclopedia.target.word-word.dim300")
print(wordemb.cosine_sim("国王", "王后"))
>>> 0.63395125
wordemb.cosine_sim("艺术", "火车")
>>> 0.14792643
```

### Rich Chinsese Pre-trained Models

```python
from paddlenlp.transformers import ErnieModel, BertModel, RobertaModel, ElectraModel

ernie = ErnieModel.from_pretrained('ernie-1.0')
bert = BertModel.from_pretrained('bert-wwm-chinese')
roberta = RobertaModel.from_pretrained('roberta-wwm-ext')
electra = ElectraModel.from_pretrained('chinese-electra-small')
```

For more pretrained model selection, please refer to [Pretrained-Models](./paddlenlp/transformers/README.md)

## Model Zoo and Applications

- [Word Embedding](./examples/word_embedding/README.md)
- [Lexical Analysis](./examples/lexical_analysis/README.md)
- [Language Model](./examples/language_model)
- [Text Classification](./examples/text_classification/README.md)
- [Text Generation](./examples/text_generation/README.md)
- [Semantic Matching](./examples/text_matching/README.md)
- [Named Entity Recognition](./examples/named_entity_recognition/README.md)
- [Text Graph](./examples/text_graph/README.md)
- [General Dialogue](./examples/dialogue)
- [Machine Translation](./exmaples/machine_translation)
- [Question Answering](./exmaples/machine_reading_comprehension)

## API Usage

- [Transformer API](./docs/transformers.md)
- [Data API](./docs/data.md)
- [Dataset API](./docs/datasets.md)
- [Embedding API](./docs/embeddings.md)
- [Metrics API](./docs/metrics.md)


## Tutorials

Please refer to our official AI Studio account for more interactive tutorials: [PaddleNLP on AI Studio](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)

* [What's Seq2Vec?](https://aistudio.baidu.com/aistudio/projectdetail/1283423) shows how to use LSTM to do sentiment analysis.

* [Sentiment Analysis with ERNIE](https://aistudio.baidu.com/aistudio/projectdetail/1294333) shows how to exploit the pretrained ERNIE to make sentiment analysis better.

* [Waybill Information Extraction with BiGRU-CRF Model](https://aistudio.baidu.com/aistudio/projectdetail/1317771) shows how to make use of bigru and crf to do information extraction.

* [Waybill Information Extraction with ERNIE](https://aistudio.baidu.com/aistudio/projectdetail/1329361) shows how to exploit the pretrained ERNIE to do information extraction better.


## Community

Join our QQ Technical Group for technical exchange right now! ⬇️

<div align="center">
  <img src="./docs/imgs/qq.png" width="200" height="200" />
</div>

## License

PaddleNLP is provided under the [Apache-2.0 License](./LICENSE).
