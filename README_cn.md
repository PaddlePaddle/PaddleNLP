**简体中文**🀄 | [English🌎](./README_en.md)

<p align="center">
  <img src="https://user-images.githubusercontent.com/1371212/175816733-8ec25eb0-9af3-4380-9218-27c154518258.png" align="middle"  width="500" />
</p>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleNLP?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.6.2+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleNLP?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/PaddleNLP?color=3af"></a>
    <a href="https://pypi.org/project/paddlenlp/"><img src="https://img.shields.io/pypi/dm/paddlenlp?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleNLP?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleNLP?color=ccf"></a>
</p>


<h4 align="center">
  <a href=#特性> 特性 </a> |
  <a href=#安装> 安装 </a> |
  <a href=#快速开始> 快速开始 </a> |
  <a href=#api文档> API文档 </a> |
  <a href=#社区交流> 社区交流 </a>
</h4>

**PaddleNLP**是一款**简单易用**且**功能强大**的自然语言处理开发库。聚合业界**优质预训练模型**并提供**开箱即用**的开发体验，覆盖NLP多场景的模型库搭配**产业实践范例**可满足开发者**灵活定制**的需求。

## News 📢
* 🔥 **2022.10.27 发布 [PaddleNLP v2.4.2](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.4.2)**
  * NLG能力扩充：新增📄[**基于Pegasus的中文文本摘要方案**](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_summarization/pegasus)，效果领先；新增❓[**问题生成解决方案**](./examples/question_generation)，提供基于业界领先模型UNIMO-Text和大规模多领域问题生成数据集训练的通用问题生成预训练模型。均支持Taskflow一键调用，支持FasterGeneration高性能推理，训练推理部署全流程打通。
  * 发布 🖼[**PPDiffusers**](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers)：支持跨模态（如图像与语音）训练和推理的扩散模型（Diffusion Model）工具箱，可快速体验、二次开发 **Stable Diffusion**，持续支持更多模型。

* 🔥 **2022.10.14 发布 [PaddleNLP v2.4.1](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.4.1)**
  * 🧾 发布多语言跨模态布局增强文档智能大模型 [**ERNIE-Layout**](./model_zoo/ernie-layout/)，刷新11项任务SOTA。同步发布基于ERNIE-Layout的**文档抽取问答模型DocPrompt** 🔖，精准理解文档图片布局与语义信息，轻松应对各类业务场景。

* 🔥 **2022.9.6 发布 [PaddleNLP v2.4](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.4.0)**
  * 💎 NLP工具：**[NLP 流水线系统 Pipelines](./pipelines)** 发布，支持快速搭建搜索引擎、问答系统，可扩展支持各类NLP系统，让解决 NLP 任务像搭积木一样便捷、灵活、高效！
  * 💢 产业应用：新增 **[文本分类全流程应用方案](./applications/text_classification)** ，覆盖多分类、多标签、层次分类各类场景，支持 **小样本学习** 和 **TrustAI** 可信计算模型训练与调优；[**通用信息抽取 UIE 能力升级**](./model_zoo/uie)，发布 **UIE-M**，支持中英文混合抽取，新增**UIE 数据蒸馏**方案，打破 UIE 推理瓶颈，推理速度提升 100 倍以上；
  * 🍭 AIGC 内容生成：新增代码生成 SOTA 模型[**CodeGen**](./examples/code_generation/codegen)，支持多种编程语言代码生成；集成[**文图生成潮流模型**](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/taskflow.md#%E6%96%87%E5%9B%BE%E7%94%9F%E6%88%90) DALL·E Mini、Disco Diffusion、Stable Diffusion，更多趣玩模型等你来玩；新增[**中文文本摘要应用**](./applications/text_summarization)，基于大规模语料的中文摘要模型首次发布，可支持 Taskflow 一键调用和定制训练；
  * 💪 框架升级：[**模型自动压缩 API**](./docs/compression.md) 发布，自动对模型进行裁减和量化，大幅降低模型压缩技术使用门槛；[**小样本 Prompt**](./applications/text_classification/multi_class/few-shot)能力发布，集成 PET、P-Tuning、RGL 等经典算法。


## 社区交流

- **💥 直播回放：10.25-10.28**，PaddleNLP研发团队详解PaddleNLP v2.4新发功能，并带来前沿技术分享，以及企业落地实践。微信扫描下方二维码，关注公众号并填写问卷后进入官方交流群，获取直播回放与10G重磅NLP学习大礼包。

  <div align="center">
  <img src="https://user-images.githubusercontent.com/11793384/197535273-75db1fb3-3a34-4bd2-bac3-bcacb1b986a8.jpg" width="150" height="150" />
  </div>

## 特性

#### <a href=#开箱即用的nlp工具集> 📦 开箱即用的NLP工具集 </a>

#### <a href=#丰富完备的中文模型库> 🤗 丰富完备的中文模型库 </a>

#### <a href=#产业级端到端系统范例> 🎛️ 产业级端到端系统范例 </a>

#### <a href=#高性能分布式训练与推理> 🚀 高性能分布式训练与推理 </a>


### 开箱即用的NLP工具集

Taskflow提供丰富的**📦开箱即用**的产业级NLP预置模型，覆盖自然语言理解与生成两大场景，提供**💪产业级的效果**与**⚡️极致的推理性能**。

![taskflow1](https://user-images.githubusercontent.com/11793384/159693816-fda35221-9751-43bb-b05c-7fc77571dd76.gif)

Taskflow最新集成了文生图的趣玩应用，三行代码体验 **Stable Diffusion**
```python
from paddlenlp import Taskflow
text_to_image = Taskflow("text_to_image", model="CompVis/stable-diffusion-v1-4")
image_list = text_to_image('"In the morning light,Chinese ancient buildings in the mountains,Magnificent and fantastic John Howe landscape,lake,clouds,farm,Fairy tale,light effect,Dream,Greg Rutkowski,James Gurney,artstation"')
```
<img width="300" alt="image" src="https://user-images.githubusercontent.com/16698950/194882669-f7cc7c98-d63a-45f4-99c1-0514c6712368.png">

更多使用方法可参考[Taskflow文档](./docs/model_zoo/taskflow.md)。

### 丰富完备的中文模型库

#### 🀄 业界最全的中文预训练模型

精选 45+ 个网络结构和 500+ 个预训练模型参数，涵盖业界最全的中文预训练模型：既包括文心NLP大模型的ERNIE、PLATO等，也覆盖BERT、GPT、RoBERTa、T5等主流结构。通过`AutoModel` API一键⚡**高速下载**⚡。

```python
from paddlenlp.transformers import *

ernie = AutoModel.from_pretrained('ernie-3.0-medium-zh')
bert = AutoModel.from_pretrained('bert-wwm-chinese')
albert = AutoModel.from_pretrained('albert-chinese-tiny')
roberta = AutoModel.from_pretrained('roberta-wwm-ext')
electra = AutoModel.from_pretrained('chinese-electra-small')
gpt = AutoModelForPretraining.from_pretrained('gpt-cpm-large-cn')
```

针对预训练模型计算瓶颈，可以使用API一键使用文心ERNIE-Tiny全系列轻量化模型，降低预训练模型部署难度。

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

对预训练模型应用范式如语义表示、文本分类、句对匹配、序列标注、问答等，提供统一的API体验。

```python
import paddle
from paddlenlp.transformers import *

tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')
text = tokenizer('自然语言处理')

# 语义表示
model = AutoModel.from_pretrained('ernie-3.0-medium-zh')
sequence_output, pooled_output = model(input_ids=paddle.to_tensor([text['input_ids']]))
# 文本分类 & 句对匹配
model = AutoModelForSequenceClassification.from_pretrained('ernie-3.0-medium-zh')
# 序列标注
model = AutoModelForTokenClassification.from_pretrained('ernie-3.0-medium-zh')
# 问答
model = AutoModelForQuestionAnswering.from_pretrained('ernie-3.0-medium-zh')
```

#### 💯 全场景覆盖的应用示例

覆盖从学术到产业的NLP应用示例，涵盖NLP基础技术、NLP系统应用以及拓展应用。全面基于飞桨核心框架2.0全新API体系开发，为开发者提供飞桨文本领域的最佳实践。

精选预训练模型示例可参考[Model Zoo](./model_zoo)，更多场景示例文档可参考[examples目录](./examples)。更有免费算力支持的[AI Studio](https://aistudio.baidu.com)平台的[Notbook交互式教程](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)提供实践。

<details><summary> PaddleNLP预训练模型适用任务汇总（<b>点击展开详情</b>）</summary><div>

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

可参考[Transformer 文档](/docs/model_zoo/index.rst) 查看目前支持的预训练模型结构、参数和详细用法。

### 产业级端到端系统范例

PaddleNLP针对信息抽取、语义检索、智能问答、情感分析等高频NLP场景，提供了端到端系统范例，打通*数据标注*-*模型训练*-*模型调优*-*预测部署*全流程，持续降低NLP技术产业落地门槛。更多详细的系统级产业范例使用说明请参考[Applications](./applications)。

#### 🔍 语义检索系统

针对无监督数据、有监督数据等多种数据情况，结合SimCSE、In-batch Negatives、ERNIE-Gram单塔模型等，推出前沿的语义检索方案，包含召回、排序环节，打通训练、调优、高效向量检索引擎建库和查询全流程。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168514909-8817d79a-72c4-4be1-8080-93d1f682bb46.gif" width="400">
</div>


更多使用说明请参考[语义检索系统](./applications/neural_search)。

#### ❓ 智能问答系统

基于[🚀RocketQA](https://github.com/PaddlePaddle/RocketQA)技术的检索式问答系统，支持FAQ问答、说明书问答等多种业务场景。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168514868-1babe981-c675-4f89-9168-dd0a3eede315.gif" width="400">
</div>


更多使用说明请参考[智能问答系统](./applications/question_answering)与[文档智能问答](./applications/document_intelligence/doc_vqa)

#### 💌 评论观点抽取与情感分析

基于情感知识增强预训练模型SKEP，针对产品评论进行评价维度和观点抽取，以及细粒度的情感分析。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407260-b7f92800-861c-4207-98f3-2291e0102bbe.png" width="400">
</div>

更多使用说明请参考[情感分析](./applications/sentiment_analysis)。

#### 🎙️ 智能语音指令解析

集成了[PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)和[百度开放平台](https://ai.baidu.com/)的的语音识别和[UIE](./model_zoo/uie)通用信息抽取等技术，打造智能一体化的语音指令解析系统范例，该方案可应用于智能语音填单、智能语音交互、智能语音检索等场景，提高人机交互效率。

<div align="center">
    <img src="https://user-images.githubusercontent.com/16698950/168589100-a6c6f346-97bb-47b2-ac26-8d50e71fddc5.png" width="400">
</div>

更多使用说明请参考[智能语音指令解析](./applications/speech_cmd_analysis)。

### 高性能分布式训练与推理

#### ⚡ FasterTokenizer：高性能文本处理库

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407921-b4395b1d-44bd-41a0-8c58-923ba2b703ef.png" width="400">
</div>

```python
AutoTokenizer.from_pretrained("ernie-3.0-medium-zh", use_faster=True)
```

为了实现更极致的模型部署性能，安装FastTokenizers后只需在`AutoTokenizer` API上打开 `use_faster=True`选项，即可调用C++实现的高性能分词算子，轻松获得超Python百余倍的文本处理加速，更多使用说明可参考[FasterTokenizer文档](./faster_tokenizer)。

#### ⚡️ FasterGeneration：高性能生成加速库

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407831-914dced0-3a5a-40b8-8a65-ec82bf13e53c.gif" width="400">
</div>

```python
model = GPTLMHeadModel.from_pretrained('gpt-cpm-large-cn')
...
outputs, _ = model.generate(
    input_ids=inputs_ids, max_length=10, decode_strategy='greedy_search',
    use_faster=True)
```

简单地在`generate()`API上打开`use_faster=True`选项，轻松在Transformer、GPT、BART、PLATO、UniLM等生成式预训练模型上获得5倍以上GPU加速，更多使用说明可参考[FasterGeneration文档](./faster_generation)。

#### 🚀 Fleet：飞桨4D混合并行分布式训练技术

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168515134-513f13e0-9902-40ef-98fa-528271dcccda.png" width="300">
</div>


更多关于千亿级AI模型的分布式训练使用说明可参考[GPT-3](./examples/language_model/gpt-3)。

## 安装

### 环境依赖

- python >= 3.6
- paddlepaddle >= 2.2

### pip安装

```shell
pip install --upgrade paddlenlp
```

更多关于PaddlePaddle和PaddleNLP安装的详细教程请查看[Installation](./docs/get_started/installation.rst)。

## 快速开始

这里以信息抽取-命名实体识别任务，UIE模型为例，来说明如何快速使用PaddleNLP:

### 一键预测

PaddleNLP提供[一键预测功能](./docs/model_zoo/taskflow.md)，无需训练，直接输入数据即可开放域抽取结果：

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow

>>> schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
>>> ie = Taskflow('information_extraction', schema=schema)
>>> pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))
[{'时间': [{'end': 6,
          'probability': 0.9857378532924486,
          'start': 0,
          'text': '2月8日上午'}],
  '赛事名称': [{'end': 23,
            'probability': 0.8503089953268272,
            'start': 6,
            'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
  '选手': [{'end': 31,
          'probability': 0.8981548639781138,
          'start': 28,
          'text': '谷爱凌'}]}]
```

### 小样本学习

如果对一键预测效果不满意，也可以使用少量数据进行模型精调，进一步提升特定场景的效果，详见[UIE小样本定制训练](./model_zoo/uie/)。

更多PaddleNLP内容可参考：
- [精选模型库](./model_zoo)，包含优质预训练模型的端到端全流程使用。
- [多场景示例](./examples)，了解如何使用PaddleNLP解决NLP多种技术问题，包含基础技术、系统应用与拓展应用。
- [交互式教程](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)，在🆓免费算力平台AI Studio上快速学习PaddleNLP。


## API文档

PaddleNLP提供全流程的文本领域API，可大幅提升NLP任务建模的效率：

- 支持[千言](https://www.luge.ai)等丰富中文数据集加载的[Dataset API](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html)。
- 提供🤗Hugging Face Style的API，支持 **500+** 优质预训练模型加载的[Transformers API](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html)。
- 提供30+多语言词向量的[Embedding API](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/embeddings.html)

更多使用方法请参考[API文档](https://paddlenlp.readthedocs.io/zh/latest/)。


## Citation

如果PaddleNLP对您的研究有帮助，欢迎引用

```
@misc{=paddlenlp,
    title={PaddleNLP: An Easy-to-use and High Performance NLP Library},
    author={PaddleNLP Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleNLP}},
    year={2021}
}
```

## Acknowledge

我们借鉴了Hugging Face的[Transformers](https://github.com/huggingface/transformers)🤗关于预训练模型使用的优秀设计，在此对Hugging Face作者及其开源社区表示感谢。

## License

PaddleNLP遵循[Apache-2.0开源协议](./LICENSE)。
