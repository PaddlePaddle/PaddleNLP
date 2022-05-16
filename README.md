简体中文 | [English](./README_en.md)

<p align="center">
  <img src="./docs/imgs/paddlenlp.png" align="middle"  width="500" />
</p>




------------------------------------------------------------------------------------------

[![PyPI - PaddleNLP Version](https://img.shields.io/pypi/v/paddlenlp.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/paddlenlp/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paddlenlp)](https://pypi.org/project/paddlenlp/)
[![PyPI Status](https://pepy.tech/badge/paddlenlp/month)](https://pepy.tech/project/paddlenlp)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
![GitHub](https://img.shields.io/github/license/paddlepaddle/paddlenlp)

<h4 align="center">
  <a href=#特性> 特性 </a> |
  <a href=#安装> 安装 </a> |
  <a href=#QuickStart> 快速开始 </a> |
  <a href=#API文档> API文档 </a> |
  <a href=#社区交流> 社区交流 </a> 
</h4>

## News  <img src="./docs/imgs/news_icon.png" width="40"/>

* 🔥 2021.5.18-19直播课，解读通用信息抽取技术**UIE**和**ERNIE 3.0**轻量级模型能力，欢迎报名来交流

  <div align="center">
  <img src="https://user-images.githubusercontent.com/11793384/168411900-d9f3d777-99ab-4b5c-8cdc-ef747a48b864.jpg" width="188" height="188" />
  </div>

* 🔥 2022.5.16 PaddleNLP [release/2.3](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.3.0)

  * 新增文心大模型 [ERNIE 3.0](./model_zoo/ernie-3.0)，在CLUE Benchmark上实现同规模模型中文最佳效果；新增中文医疗领域预训练模型 [ERNIE-Health](./model_zoo/ernie-health)；新增超大规模百亿（11B）开放域对话预训练模型 PLATO-XL（英文），并提供FasterGeneration高性能GPU加速，相比上版本推理速度加速2.7倍。
  * 通用信息抽取技术 [UIE](./model_zoo/uie)发布，单个模型可以同时支持命名实体识别、关系抽取、事件抽取、情感分析等任务；

* 2022.3.21 PaddleNLP [release/2.2.5](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.2.5) 一键预测工具[Taskflow](./docs/model_zoo/taskflow.md)全新升级！欢迎体验更丰富的功能、更便捷的使用方式；新推出适合不同场景的中文分词、命名实体识别模式！

* 2021.12.28 PaddleNLP [release/2.2.2](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.2.2) 发布语义检索、问答、评论观点抽取和情感倾向分析 [产业化案例](./applications)，快速搭建系统！配套视频课程[直通车](https://aistudio.baidu.com/aistudio/course/introduce/24902)！

## 特性

PaddleNLP是飞桨自然语言处理开发库，旨在提升开发者在文本领域的开发效率，并提供丰富的NLP应用示例。具备以下四大特性：

#### <img src="https://user-images.githubusercontent.com/11793384/168454776-2075cc68-9402-4b0b-8723-5be0a315ddc3.png" width="20" height="20" /><a href=#开箱即用的NLP能力> 开箱即用的NLP能力 </a>

#### <img src="https://user-images.githubusercontent.com/11793384/168454751-f111d8b4-a16a-4e36-b9de-3af8a2f00714.png" width="20" height="20" /><a href=#丰富完备的中文模型库> 丰富完备的中文模型库 </a> 

#### <img src="https://user-images.githubusercontent.com/11793384/168454721-0ac49e17-22db-4074-ba20-940365daf9f6.png" width="20" height="20" /><a href=#产业级端到端系统范例> 产业级端到端系统范例 </a> 

#### <img src="https://user-images.githubusercontent.com/11793384/168454587-8b5a0f63-3d4b-4339-be47-f3ad7ef9e16c.png" width="20" height="20" /><a href=#高性能分布式训练与推理> 高性能分布式训练与推理 </a> 


### 开箱即用的NLP能力

Taskflow提供丰富的**开箱即用**的产业级NLP预置模型，覆盖自然语言理解与生成两大场景，提供**产业级的效果**与**极致的推理性能**。

![taskflow1](https://user-images.githubusercontent.com/11793384/159693816-fda35221-9751-43bb-b05c-7fc77571dd76.gif)

更多使用方法可参考[Taskflow文档](./docs/model_zoo/taskflow.md)。

### 丰富完备的中文模型库

#### 业界最全的中文预训练模型

精选 45+ 个网络结构和 500+ 个预训练模型参数，涵盖业界最全的中文预训练模型，既包括文心NLP大模型的ERNIE、PLATO等，也覆盖BERT、GPT、RoBERTa、T5等主流结构。通过AutoModel API一键⚡高速下载⚡。

```python
from paddlenlp.transformers import *

ernie = AutoModel.from_pretrained('ernie-3.0-base-zh')
bert = AutoModel.from_pretrained('bert-wwm-chinese')
albert = AutoModel.from_pretrained('albert-chinese-tiny')
roberta = AutoModel.from_pretrained('roberta-wwm-ext')
electra = AutoModel.from_pretrained('chinese-electra-small')
gpt = AutoModelForPretraining.from_pretrained('gpt-cpm-large-cn')
```

<details><summary>&emsp;对预训练模型应用范式如语义表示、文本分类、句对匹配、序列标注、问答等，提供统一的API体验（可展开详情）</summary><div>

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

</div></details>

<details><summary>&emsp;PaddleNLP预训练模型适用任务汇总（可展开详情）</summary><div>

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

#### 全场景覆盖的应用示例

覆盖从学术到产业的NLP应用示例，涵盖NLP基础技术、NLP系统应用以及拓展应用。全面基于飞桨核心框架2.0全新API体系开发，为开发者提供飞桨文本领域的最佳实践。

精选预训练模型示例可参考[Model Zoo](./model_zoo)，更多场景示例文档可参考[examples目录](./examples)。更有免费算力支持的[AI Studio](https://aistudio.baidu.com)平台的[Notbook交互式教程](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)提供实践。


### 产业级端到端系统范例

PaddleNLP针对信息抽取、语义检索、智能问答、情感分析等高频NLP技术产经，提供端到端系统范例，打通数据标注-模型训练-调优-预测部署全流程，持续降低NLP技术产业落地门槛，更多详细的系统级产业范例使用说明请参考[Applications](./applications)。

#### 智能语音指令解析

集成了业界领先的语音识别（Automatic Speech Recognition, ASR）、信息抽取（Information Extraction, IE）等技术，打造智能一体化的语音指令系统，广泛应用于智能语音填单、智能语音交互、智能语音检索、手机APP语音唤醒等场景，提高人机交互效率。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168412618-04897a47-79c9-4fe7-a054-5dc1f6a1f75c.png" width="500">
</div>


更多使用说明请参考[智能语音指令解析](./applications/speech_cmd_analysis)。

#### 语义检索系统

针对无监督数据、有监督数据等多种数据情况，结合SimCSE、In-batch Negatives、ERNIE-Gram单塔模型等，推出前沿的语义检索方案，包含召回、排序环节，打通训练、调优、高效向量检索引擎建库和查询全流程。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168514909-8817d79a-72c4-4be1-8080-93d1f682bb46.gif" width="500">
</div>


更多使用说明请参考[语义检索系统](./applications/neural_search)。

#### 智能问答系统

推出基于语义检索技术的问答系统，支持FAQ问答、说明书问答等多种业务场景。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168514868-1babe981-c675-4f89-9168-dd0a3eede315.gif" width="500">
</div>


更多使用说明请参考[智能问答系统](./applications/question_answering)。


#### 评论观点抽取与情感分析

基于情感知识增强预训练模型SKEP，针对产品评论进行评价维度和观点抽取，以及细粒度的情感分析。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407260-b7f92800-861c-4207-98f3-2291e0102bbe.png" width="400">
</div>


更多使用说明请参考[情感分析](./applications/sentiment_analysis)。

### 高性能分布式训练与推理

#### 飞桨4D混合并行分布式训练技术

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168515134-513f13e0-9902-40ef-98fa-528271dcccda.png" height="400" width="500">
</div>


更多关于千亿级AI模型的分布式训练使用说明可参考[GPT-3](./examples/language_model/gpt-3)。

#### 高性能文本处理库 FasterTokenizers

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407921-b4395b1d-44bd-41a0-8c58-923ba2b703ef.png" width="500">
</div>


更多内容可参考[FasterTokenizers文档](./faster_tokenizers)。

#### 高性能生成加速组件 FasterGeneration

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407831-914dced0-3a5a-40b8-8a65-ec82bf13e53c.gif" width="500">
</div>


更多内容可参考[FasterGeneration文档](./faster_generation)。

## 社区交流👬

- 微信扫描二维码并填写问卷之后，加入交流群领取福利
  - 获取5月18-19日每晚20:30《产业级通用信息抽取技术UIE+ERNIE轻量级模型》直播课链接
  - 10G重磅NLP学习大礼包：

  <div align="center">
  <img src="https://user-images.githubusercontent.com/11793384/168411900-d9f3d777-99ab-4b5c-8cdc-ef747a48b864.jpg" width="188" height="188" />
  </div>


## 安装

### 环境依赖

- python >= 3.6
- paddlepaddle >= 2.2

### pip安装

```shell
pip install --upgrade paddlenlp
```

更多关于PaddlePaddle和PaddleNLP安装的详细教程请查看[Installation](./docs/get_started/installation.rst)。

## QuickStart

这里以信息抽取-命名实体识别任务，UIE模型为例，来说明如何快速使用PaddleNLP:

- 一键预测

PaddleNLP提供[一键预测功能](./docs/model_zoo/taskflow.md)，无需训练，直接输入数据，即可得到预测结果，以情感分析任务为例：

```python
from pprint import pprint
from paddlenlp import Taskflow

schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
ie = Taskflow('information_extraction', schema=schema)
pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))
>>> [{'时间': [{'end': 6,
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

- 定制训练

如果对一键预测效果不满意，也可以进行模型微调，这里对UIE模型进行微调，以进一步提升命名实体识别的准确率：

```python
from paddlenlp.transformers import ErniePretrainedModel，AutoTokenizer
```

完整微调代码，可参考[UIE微调](./model_zoo/uie/)

更多内容可参考：[多场景示例](./examples)，[PaddleNLP on AI Studio](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)。


## API文档

PaddleNLP提供全流程的文本领域API，可大幅提升NLP任务建模的效率：

- 支持丰富中文数据集加载的[Dataset API](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html)；
- 灵活高效地完成数据预处理的[Data API](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.data.html)；
- 提供500+预训练模型的[Transformers API](./docs/model_zoo/transformers.rst)。    

更多使用方法请参考[API文档](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.data.html)。


## 版本更新

更多版本更新说明请查看[ChangeLog](./docs/changelog.md)

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
