
[简体中文🀄](./README.md) | **English🌎**

<p align="center"> <img src="https://user-images.githubusercontent.com/1371212/175816733-8ec25eb0-9af3-4380-9218-27c154518258.png" align="middle"  width="500" /> </p>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleNLP?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleNLP?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/PaddleNLP?color=3af"></a>
    <a href="https://pypi.org/project/paddlenlp/"><img src="https://img.shields.io/pypi/dm/paddlenlp?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleNLP?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleNLP?color=ccf"></a>
</p>

<h4 align="center"> <a href=#features> Features </a> | <a href=#installation> Installation </a> | <a href=#quick-start> Quick Start </a> | <a href=#api-reference> API Reference </a> | <a href=#community> Community </a>

**PaddleNLP** is a NLP library that is both **easy to use** and **powerful**. It aggregates high-quality pretrained models in the industry and provides a **plug-and-play** development experience, covering a model library for various NLP scenarios. With practical examples from industry practices, PaddleNLP can meet the needs of developers who require **flexible customization**.

## News 📢

* **2023.6.12: [Release of PaddleNLP v2.6rc](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.6.0rc)**
  * 🔨 LLM Tools：Introduces comprehensive examples of open-source LLM training and inference, including [Bloom](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bloom), [ChatGLM](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/chatglm), [GLM](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/glm), [Llama](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/llama) and [OPT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/opt). Added Tensor Parallel capability to [Trainer API](./docs/trainer.md) for distributed LLM trainin. Also released [Parameter-Efficient Finetuning](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/peft),which enables training LLMs on consumer hardware.

* **2023.1.12: [Release of PaddleNLP v2.5](<https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.5.0>)**

    * 🔨 NLP Tools: [PPDiffusers](./ppdiffusers), our cross-modal diffusion model toolbox based on PaddlePaddle, has been released! It provides a complete training process for diffusion models, and supports FastDeploy inference acceleration and multi-hardware deployment (supports Ascend chips and Kunlun core deployment).
    * 💎 Industrial Applications: Information extraction, text classification, sentiment analysis, and intelligent question answering have all been newly upgraded. New releases include document information extraction [UIE-X](./applications/information_extraction/document), unified text classification [UTC](./applications/zero_shot_text_classification), unified sentiment analysis [UIE-Senta](./applications/sentiment_analysis/unified_sentiment_extraction) , and [unsupervised QA application](./applications/question_answering/unsupervised_qa). At the same time, the [ERNIE 3.0 Tiny v2](./model_zoo/ernie-tiny) series of pretrained small models have been released, which are more effective with low-resource and foreign data. They provide open-source end-to-end deployment solutions such as model pruning, model quantization, FastDeploy inference acceleration, and edge-side deployment to reduce the difficulty of pretrained model deployment.
    * 💪 Framework Upgrade: Pretrained model [parameter configuration unification](./paddlenlp/transformers/configuration_utils.py), saving and loading custom parameter configurations no longer requires additional development; [Trainer API](./docs/trainer.md) has added BF16 training, recompute recalculations, sharding, and other distributed capabilities. Large-scale pre-training model training can easily be accomplished through simple configuration. [Model Compression API](./docs/compression.md) supports quantization training, vocabulary compression, and other functions. The compressed model has smaller accuracy loss, and the memory consumption of model deployment is greatly reduced. [Data Augmentation API](./docs/dataaug.md) has been comprehensively upgraded to support three granularities of data augmentation strategy: character, word, and sentence, making it easy to customize data augmentation strategies.
    * 🤝 Community: 🤗Huggingface hub officially supports PaddleNLP pretrained models, supporting PaddleNLP Model and Tokenizer downloads and uploads directly from the 🤗Huggingface hub. Everyone is welcome to try out PaddleNLP pretrained models on the 🤗Huggingface hub [here](https://huggingface.co/PaddlePaddle).

* **September 6, 2022: [Release of PaddleNLP v2.4](<https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.4.0>)**

    * 🔨 NLP Tools: [NLP Pipeline System Pipelines](./pipelines) has been released, supporting the rapid construction of search engines and question-answering systems, and can be extended to support various NLP systems, making it easy, flexible, and efficient to solve NLP tasks like building blocks!
    * 💎 Industrial Applications: A new [text classification full-process application solution](./applications/text_classification) has been added, covering various scenarios such as multi-classification, multi-label, and hierarchical classification, supporting small-sample learning and TrustAI trustworthy computing model training and tuning.
    * 🍭 AIGC: The SOTA model [CodeGen](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/code_generation/codegen) for code generation in various programming languages has been added.
    * 💪 Framework Upgrade: [Automatic Model Compression API](./docs/compression.md) has been released, which automatically cuts and quantizes models, greatly reducing the threshold for using model compression technology. [Few-shot Prompt](./applications/text_classification/multi_class/few-shot) capability has been released, integrating classic algorithms such as PET, P-Tuning, and RGL.






## Features

#### <a href=#out-of-box-nlp-toolset> 📦 Out-of-Box NLP Toolset </a>

#### <a href=#awesome-chinese-model-zoo> 🤗 Awesome Chinese Model Zoo </a>

#### <a href=#industrial-end-to-end-system> 🎛️ Industrial End-to-end System </a>

#### <a href=#high-performance-distributed-training-and-inference> 🚀 High Performance Distributed Training and Inference </a>


### Out-of-Box NLP Toolset

Taskflow aims to provide off-the-shelf NLP pre-built task covering NLU and NLG technique, in the meanwhile with extremely fast inference satisfying industrial scenario.

![taskflow1](https://user-images.githubusercontent.com/11793384/159693816-fda35221-9751-43bb-b05c-7fc77571dd76.gif)

For more usage please refer to [Taskflow Docs](./docs/model_zoo/taskflow.md).

### Awesome Chinese Model Zoo

#### 🀄 Comprehensive Chinese Transformer Models

We provide **45+** network architectures and over **500+** pretrained models. Not only includes all the SOTA model like ERNIE, PLATO and SKEP released by Baidu, but also integrates most of the high-quality Chinese pretrained model developed by other organizations. Use `AutoModel` API to **⚡SUPER FAST⚡** download pretrained models of different architecture. We welcome all developers to contribute your Transformer models to PaddleNLP!

```python
from paddlenlp.transformers import *

ernie = AutoModel.from_pretrained('ernie-3.0-medium-zh')
bert = AutoModel.from_pretrained('bert-wwm-chinese')
albert = AutoModel.from_pretrained('albert-chinese-tiny')
roberta = AutoModel.from_pretrained('roberta-wwm-ext')
electra = AutoModel.from_pretrained('chinese-electra-small')
gpt = AutoModelForPretraining.from_pretrained('gpt-cpm-large-cn')
```

Due to the computation limitation, you can use the ERNIE-Tiny light models to accelerate the deployment of pretrained models.
```python
# 6L768H
ernie = AutoModel.from_pretrained('ernie-3.0-medium-zh')
# 6L384H
ernie = AutoModel.from_pretrained('ernie-3.0-mini-zh')
# 4L384H
ernie = AutoModel.from_pretrained('ernie-3.0-micro-zh')
# 4L312H
ernie = AutoModel.from_pretrained('ernie-3.0-nano-zh')
```
Unified API experience for NLP task like semantic representation, text classification, sentence matching, sequence labeling, question answering, etc.

```python
import paddle
from paddlenlp.transformers import *

tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')
text = tokenizer('natural language processing')

# Semantic Representation
model = AutoModel.from_pretrained('ernie-3.0-medium-zh')
sequence_output, pooled_output = model(input_ids=paddle.to_tensor([text['input_ids']]))
# Text Classificaiton and Matching
model = AutoModelForSequenceClassification.from_pretrained('ernie-3.0-medium-zh')
# Sequence Labeling
model = AutoModelForTokenClassification.from_pretrained('ernie-3.0-medium-zh')
# Question Answering
model = AutoModelForQuestionAnswering.from_pretrained('ernie-3.0-medium-zh')
```

#### Wide-range NLP Task Support

PaddleNLP provides rich examples covering mainstream NLP task to help developers accelerate problem solving. You can find our powerful transformer [Model Zoo](./model_zoo), and wide-range NLP application [examples](./examples) with detailed instructions.

Also you can run our interactive [Notebook tutorial](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995) on AI Studio, a powerful platform with **FREE** computing resource.

<details><summary> PaddleNLP Transformer model summary (<b>click to show details</b>) </summary><div>

| Model              | Sequence Classification | Token Classification | Question Answering | Text Generation | Multiple Choice |
| :----------------- | ----------------------- | -------------------- | ------------------ | --------------- | --------------- |
| ALBERT             | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| BART               | ✅                       | ✅                    | ✅                  | ✅               | ❌               |
| BERT               | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| BigBird            | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| BlenderBot         | ❌                       | ❌                    | ❌                  | ✅               | ❌               |
| ChineseBERT        | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| ConvBERT           | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| CTRL               | ✅                       | ❌                    | ❌                  | ❌               | ❌               |
| DistilBERT         | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| ELECTRA            | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| ERNIE              | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| ERNIE-CTM          | ❌                       | ✅                    | ❌                  | ❌               | ❌               |
| ERNIE-Doc          | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| ERNIE-GEN          | ❌                       | ❌                    | ❌                  | ✅               | ❌               |
| ERNIE-Gram         | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| ERNIE-M            | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| FNet               | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| Funnel-Transformer | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| GPT                | ✅                       | ✅                    | ❌                  | ✅               | ❌               |
| LayoutLM           | ✅                       | ✅                    | ❌                  | ❌               | ❌               |
| LayoutLMv2         | ❌                       | ✅                    | ❌                  | ❌               | ❌               |
| LayoutXLM          | ❌                       | ✅                    | ❌                  | ❌               | ❌               |
| LUKE               | ❌                       | ✅                    | ✅                  | ❌               | ❌               |
| mBART              | ✅                       | ❌                    | ✅                  | ❌               | ✅               |
| MegatronBERT       | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| MobileBERT         | ✅                       | ❌                    | ✅                  | ❌               | ❌               |
| MPNet              | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| NEZHA              | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| PP-MiniLM          | ✅                       | ❌                    | ❌                  | ❌               | ❌               |
| ProphetNet         | ❌                       | ❌                    | ❌                  | ✅               | ❌               |
| Reformer           | ✅                       | ❌                    | ✅                  | ❌               | ❌               |
| RemBERT            | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| RoBERTa            | ✅                       | ✅                    | ✅                  | ❌               | ✅               |
| RoFormer           | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| SKEP               | ✅                       | ✅                    | ❌                  | ❌               | ❌               |
| SqueezeBERT        | ✅                       | ✅                    | ✅                  | ❌               | ❌               |
| T5                 | ❌                       | ❌                    | ❌                  | ✅               | ❌               |
| TinyBERT           | ✅                       | ❌                    | ❌                  | ❌               | ❌               |
| UnifiedTransformer | ❌                       | ❌                    | ❌                  | ✅               | ❌               |
| XLNet              | ✅                       | ✅                    | ✅                  | ❌               | ✅               |

</div></details>

For more pretrained model usage, please refer to [Transformer API Docs](./docs/model_zoo/index.rst).

### Industrial End-to-end System

We provide high value scenarios including information extraction, semantic retrieval, question answering high-value.

For more details industrial cases please refer to [Applications](./applications).


#### 🔍 Neural Search System

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168514909-8817d79a-72c4-4be1-8080-93d1f682bb46.gif" width="400">
</div>


For more details please refer to [Neural Search](./applications/neural_search).

#### ❓ Question Answering System

We provide question answering pipeline which can support FAQ system, Document-level Visual Question answering system based on [🚀RocketQA](https://github.com/PaddlePaddle/RocketQA).

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168514868-1babe981-c675-4f89-9168-dd0a3eede315.gif" width="400">
</div>


For more details please refer to [Question Answering](./applications/question_answering) and [Document VQA](./applications/document_intelligence/doc_vqa).


#### 💌 Opinion Extraction and Sentiment Analysis

We build an opinion extraction system for product review and fine-grained sentiment analysis based on [SKEP](https://arxiv.org/abs/2005.05635) Model.

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407260-b7f92800-861c-4207-98f3-2291e0102bbe.png" width="300">
</div>


For more details please refer to [Sentiment Analysis](./applications/sentiment_analysis).

#### 🎙️ Speech Command Analysis

Integrated ASR Model, Information Extraction, we provide a speech command analysis pipeline that show how to use PaddleNLP and [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) to solve Speech + NLP real scenarios.

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168412618-04897a47-79c9-4fe7-a054-5dc1f6a1f75c.png" width="500">
</div>


For more details please refer to [Speech Command Analysis](./applications/speech_cmd_analysis).

### High Performance Distributed Training and Inference

#### ⚡ FastTokenizer: High Performance Text Preprocessing Library

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407921-b4395b1d-44bd-41a0-8c58-923ba2b703ef.png" width="400">
</div>

```python
AutoTokenizer.from_pretrained("ernie-3.0-medium-zh", use_fast=True)
```

Set `use_fast=True` to use C++ Tokenizer kernel to achieve 100x faster on text pre-processing. For more usage please refer to [FastTokenizer](./fast_tokenizer).

#### ⚡ FastGeneration: High Performance Generation Library

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407831-914dced0-3a5a-40b8-8a65-ec82bf13e53c.gif" width="400">
</div>

```python
model = GPTLMHeadModel.from_pretrained('gpt-cpm-large-cn')
...
outputs, _ = model.generate(
    input_ids=inputs_ids, max_length=10, decode_strategy='greedy_search',
    use_fast=True)
```

Set `use_fast=True` to achieve 5x speedup for Transformer, GPT, BART, PLATO, UniLM text generation. For more usage please refer to [FastGeneration](./fast_generation).

#### 🚀 Fleet: 4D Hybrid Distributed Training

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168515134-513f13e0-9902-40ef-98fa-528271dcccda.png" width="300">
</div>


For more super large-scale model pre-training details please refer to [GPT-3](./examples/language_model/gpt-3).


## Installation

### Prerequisites

* python >= 3.7
* paddlepaddle >= 2.3

More information about PaddlePaddle installation please refer to [PaddlePaddle's Website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html).

### Python pip Installation

```
pip install --upgrade paddlenlp
```

or you can install the latest develop branch code with the following command:

```shell
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

## Quick Start

**Taskflow** aims to provide off-the-shelf NLP pre-built task covering NLU and NLG scenario, in the meanwhile with extremely fast inference satisfying industrial applications.

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

## API Reference

- Support [LUGE](https://www.luge.ai/) dataset loading and compatible with Hugging Face [Datasets](https://huggingface.co/datasets). For more details please refer to [Dataset API](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html).
- Using Hugging Face style API to load 500+ selected transformer models and download with fast speed. For more information please refer to [Transformers API](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html).
- One-line of code to load pre-trained word embedding. For more usage please refer to [Embedding API](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/embeddings.html).

Please find all PaddleNLP API Reference from our [readthedocs](https://paddlenlp.readthedocs.io/).

## Community

### Slack

To connect with other users and contributors, welcome to join our [Slack channel](https://paddlenlp.slack.com/).

### WeChat

Scan the QR code below with your Wechat⬇️. You can access to official technical exchange group. Look forward to your participation.

<div align="center">
<img src="https://user-images.githubusercontent.com/11987277/245085922-0aa68d24-00ff-442e-9c53-2f1e898151ce.png" width="150" height="150" />
</div>



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

We have borrowed from Hugging Face's [Transformers](https://github.com/huggingface/transformers)🤗 excellent design on pretrained models usage, and we would like to express our gratitude to the authors of Hugging Face and its open source community.

## License

PaddleNLP is provided under the [Apache-2.0 License](./LICENSE).
