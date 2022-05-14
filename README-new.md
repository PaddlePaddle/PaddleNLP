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

<h4 align="left">
  
  <a href=#特性> 特性 </a> |
  <a href=#安装> 安装 </a> |
  <a href=#QuickStart> QuickStart </a> |
  <a href=#API文档> API文档 </a> |
  <a href=#社区交流> 社区交流 </a> 
</h4>

## News  <img src="./docs/imgs/news_icon.png" width="40"/>

* 🔥 2021.5.18-19直播课，解读信息抽取UIE和ERNIE 3.0轻量级模型能力，欢迎报名来交流
  <div align="center">
  <img src="https://user-images.githubusercontent.com/11793384/168411900-d9f3d777-99ab-4b5c-8cdc-ef747a48b864.jpg" width="188" height="188" />
</div>

* 🔥 2022.5.16 PaddleNLP [release/2.3](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.3.0rc0)
  * 新增百度文心大模型 [ERNIE 3.0](./model_zoo/ernie-3.0)，在CLUE Benchmark上实现同规模模型中文最佳效果；新增中文医疗领域预训练模型 [ERNIE-Health](./model_zoo/ernie-health)；新增超大规模百亿（11B）开放域对话预训练模型 PLATO-XL（英文），并提供FasterGeneration高性能GPU加速，相比上版本推理速度加速2.7倍。
  * 通用信息抽取技术 [UIE](./model_zoo/uie)发布，单个模型可以同时支持命名实体识别、关系抽取、事件抽取、情感分析等任务；
 
* 2022.3.21 PaddleNLP [release/2.2.5](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.2.5)
  * 一键预测工具[Taskflow](./docs/model_zoo/taskflow.md)全新升级！欢迎体验更丰富的功能、更便捷的使用方式；新推出适合不同场景的中文分词、命名实体识别模式！
  
* 2021.12.28 PaddleNLP [release/2.2.2](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.2.2)
  * 新发语义检索、问答、评论观点抽取和情感倾向分析 [产业化案例](./applications)，快速搭建系统！配套视频课程[直通车](https://aistudio.baidu.com/aistudio/course/introduce/24902)！

* 2021.12.11 PaddleNLP [release/2.2.0](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.2.0)
  * 发布面向生成任务的高性能加速组件[FasterGeneration](./examples/faster/faster_generation)

* 2021.12.11 PaddleNLP [release/2.1.0](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.1.0)
  * 新增文本纠错、摘要、语义匹配、模型压缩等应用示例。 


## 特性

PaddleNLP是飞桨自然语言处理开发库，旨在提升开发者在文本领域的开发效率，并提供丰富的NLP应用示例。具备以下四大特性：
- <a href=#特性1-开箱即用的一键预测能力> 开箱即用的一键预测能力 </a>
- <a href=#特性2-中文最强模型库> 中文最强模型库 </a> 
- <a href=#特性3-场景系统技术方案> 场景系统技术方案 </a> 
- <a href=#特性4-高性能训练与部署能力> 高性能训练与部署能力 </a> 
  
## 社区交流👬
- 微信扫描二维码并填写问卷之后，加入交流群领取福利
  - 获取5月18-19日每晚20:30《产业级通用信息抽取技术UIE+ERNIE轻量级模型》直播课链接
  - 10G重磅NLP学习大礼包：

<div align="center">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/release/2.2/docs/imgs/wechat.png" width="188" height="188" />
</div>

## 特性1-开箱即用的一键预测能力

Taskflow旨在提供**开箱即用**的产业级NLP预置任务能力，覆盖自然语言理解与生成两大场景，提供**产业级的效果**与**极致的预测性能**。

![taskflow1](https://user-images.githubusercontent.com/11793384/159693816-fda35221-9751-43bb-b05c-7fc77571dd76.gif)

更多使用方法请参考[Taskflow文档](./docs/model_zoo/taskflow.md)。

## 特性2-中文最强模型库

- **Transformer 预训练模型**

覆盖 **45+** 个网络结构和 **500+** 个预训练模型参数，⭐️⭐️ 国内下载速度快!⭐️⭐️ 既包括百度自研的预训练模型如ERNIE系列, PLATO, SKEP等，也涵盖业界主流的中文预训练模型如BERT，GPT，RoBERTa，T5等。使用AutoModel可以下载不同网络结构的预训练模型。欢迎开发者加入贡献更多预训练模型！

统一通过调用`paddlenlp.transformers`使用：

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

tokenizer = AutoTokenizer.from_pretrained('ernie-1.0')
text = tokenizer('自然语言处理')

# 语义表示
model = AutoModel.from_pretrained('ernie-1.0')
sequence_output, pooled_output = model(input_ids=paddle.to_tensor([text['input_ids']]))
# 文本分类 & 句对匹配
model = AutoModelForSequenceClassification.from_pretrained('ernie-1.0')
# 序列标注
model = AutoModelForTokenClassification.from_pretrained('ernie-1.0')
# 问答
model = AutoModelForQuestionAnswering.from_pretrained('ernie-1.0')
```
</div></details>

<details><summary>&emsp;Transformer预训练模型适用任务汇总 （可展开详情）</summary><div>

 | Model      |    Sequence Classification | Token Classification | Question Answering | Text Generation | Multiple Choice |                                              
| :--------------------------------- | -------------------------------- | -------- | -------- | -------- | ---------- | 
|ALBERT_             | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
  |BART_               | ✅                      | ✅                   | ✅                 | ✅              | ❌              |
  |BERT_               | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
|BigBird_            | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
|Blenderbot_         | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
|Blenderbot-Small_   | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
|ChineseBert_        | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
|ConvBert_           | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
|CTRL_               | ✅                      | ❌                   | ❌                 | ❌              | ❌              |
|DistilBert_         | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
|ELECTRA_            | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
|ERNIE_              | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
|ERNIE-CTM_          | ❌                      | ✅                   | ❌                 | ❌              | ❌              |
|ERNIE-DOC_          | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
|ERNIE-GEN_          | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
|ERNIE-GRAM_         | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
|ERNIE-M_            | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
|FNet_               | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
|Funnel_             | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
|GPT_                | ✅                      | ✅                   | ❌                 | ✅              | ❌              |
|LayoutLM_           | ✅                      | ✅                   | ❌                 | ❌              | ❌              |
|LayoutLMV2_         | ❌                      | ✅                   | ❌                 | ❌              | ❌              |
|LayoutXLM_          | ❌                      | ✅                   | ❌                 | ❌              | ❌              |
|Luke_               | ❌                      | ✅                   | ✅                 | ❌              | ❌              |
|MBart_              | ✅                      | ❌                   | ✅                 | ❌              | ✅              |
|MegatronBert_       | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
|MobileBert_         | ✅                      | ❌                   | ✅                 | ❌              | ❌              |
|MPNet_              | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
|NeZha_              | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
|PPMiniLM_           | ✅                      | ❌                   | ❌                 | ❌              | ❌              |
|ProphetNet_         | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
|Reformer_           | ✅                      | ❌                   | ✅                 | ❌              | ❌              |
|RemBert_            | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
|RoBERTa_            | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
|RoFormer_           | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
|SKEP_               | ✅                      | ✅                   | ❌                 | ❌              | ❌              |
|SqueezeBert_        | ✅                      | ✅                   | ✅                 | ❌              | ❌              |
|T5_                 | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
|TinyBert_           | ✅                      | ❌                   | ❌                 | ❌              | ❌              |
|UnifiedTransformer_ | ❌                      | ❌                   | ❌                 | ✅              | ❌              |
|XLNet_              | ✅                      | ✅                   | ✅                 | ❌              | ✅              |
  
</div></details>

请参考[Transformer 文档](/docs/model_zoo/index.rst) 查看目前支持的预训练模型结构、参数和详细用法。

- **模型应用示例，覆盖NLP全场景**

覆盖从学术到产业级的NLP[应用示例](#多场景的应用示例)，涵盖NLP基础技术、NLP系统应用以及相关拓展应用。全面基于飞桨核心框架2.0全新API体系开发，为开发者提供飞桨文本领域的最佳实践。
多场景示例文档请参考[example文档](./docs/model_zoo/examples.md)、[Notbook交互式教程](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)。


## 特性3-场景系统技术方案

PaddleNLP针对语义检索、问答、情感分析清楚，推出场景系统技术方案，打通数据标注-模型训练-调优-预测部署全流程。

- **语音指令解析和关键信息抽取**

描述：

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168412618-04897a47-79c9-4fe7-a054-5dc1f6a1f75c.png" width="400">
</div>

更多请参考[语音指令解析和信息抽取案例](./applications/speech_cmd_analysis)。

- **语义检索系统**

描述：

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407426-fc363513-8a78-4092-9bc0-db108244366f.png" width="400">
</div>

更多请参考[语义检索](./applications/neural_search)。

- **问答系统**

描述：

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407211-edb14045-15f9-4e0b-8339-d1ffa386ee6e.gif" width="400">
</div>

更多请参考[问答系统](./applications/question_answering)。


- **产品评论维度、观点抽取和细粒度情感分析**

描述：

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407260-b7f92800-861c-4207-98f3-2291e0102bbe.png" width="400">
</div>

更多请参考[情感分析](./applications/sentiment_analysis)。

## 特性4-高性能训练与部署能力

- **高性能Transformer类文本分词器：FasterTokenizer**
<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407921-b4395b1d-44bd-41a0-8c58-923ba2b703ef.png" width="600">
</div>

更多内容请参考[FasterTokenizer文档](./examples/faster/faster_tokenizer)

- **面向生成任务的高性能加速组件：FasterGeneration**

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168407831-914dced0-3a5a-40b8-8a65-ec82bf13e53c.gif" width="600">
</div>

更多内容请参考[FasterGeneration文档](./examples/faster/faster_generation)

- **蒸馏、剪裁、量化等级联模型压缩技术**

PaddleNLP 联合 PaddleSlim 通过模型蒸馏、剪裁、量化等级联模型压缩技术发布中文特色小模型 PP-MiniLM(6L768H) 及压缩方案，保证模型精度的同时模型推理速度达 BERT(12L768H) 的 8.88 倍，参数量相比减少 52%，模型精度在中文语言理解评测基准 CLUE 高 0.62。

PP-MiniLM 压缩方案以面向预训练模型的任务无关知识蒸馏(Task-agnostic Distillation)技术、裁剪(Pruning)技术、量化(Quantization)技术为核心，使得 PP-MiniLM 又快、又准、又小。

| Model                         | #Params   | #FLOPs    | Speedup (w/o FasterTokenizer)   | AFQMC     | TNEWS     | IFLYTEK   | CMNLI     | OCNLI     | CLUEWSC2020 | CSL       | Avg       |
| ----------------------------- | --------- | --------- | ---------------- | --------- | --------- | --------- | --------- | --------- | ----------- | --------- | --------- |
| BERT-base, Chinese            | 102.3M    | 10.87B    | 1.00x            | 74.14     | 56.81     | 61.10     | 81.19     | 74.85     | 79.93       | 81.47     | 72.78     |
| TinyBERT<sub>6,</sub> Chinese | 59.7M     | 5.44B     | 1.90x            | 72.59     | 55.70     | 57.64     | 79.57     | 73.97     | 76.32       | 80.00     | 70.83     |
| UER-py RoBERTa L6-H768        | 59.7M     | 5.44B     | 1.90x            | 69.62     | **66.45** | 59.91     | 76.89     | 71.36     | 71.05       | **82.87** | 71.16     |
| RBT6, Chinese                 | 59.7M     | 5.44B     | 1.90x            | 73.93     | 56.63     | 59.79     | 79.28     | 73.12     | 77.30       | 80.80     | 71.55     |
| ERNIE-Tiny                    | 90.7M     | 4.83B     | 2.22x            | 71.55     | 58.34     | 61.41     | 76.81     | 71.46     | 72.04       | 79.13     | 70.11     |
| PP-MiniLM                     | 59.7M     | 5.44B     | 2.15x (1.90x)     | 74.14     | 57.43     | **61.75** | 81.01     | **76.17** | 86.18       | 79.17     | **73.69** |
| PP-MiniLM + 裁剪              | **49.1M** | **4.08B** | 2.74x (2.48x)     | 73.91     | 57.44     | 61.64     | 81.10     | 75.59     | **85.86**   | 78.53     | 73.44     |
| PP-MiniLM + 量化              | 59.8M     | -         | 7.34x (4.63x)     | **74.19** | 57.13     | 61.10     | **81.20** | 76.10     | 85.20       | 78.03     | 73.28     |
| PP-MiniLM + 裁剪 + 量化       | **49.2M** | -         | **8.88x** (5.36x) | 74.00     | 57.37     | 61.33     | 81.09     | 75.56     | 85.85       | 78.57     | 73.40     |

详情请参考[压缩方案文档](./examples/model_compression/pp-minilm)。


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

这里以情感倾向分析任务，SKEP模型为例，来说明如何快速使用PaddleNLP:

- 一键预测

PaddleNLP提供[一键预测功能](./docs/model_zoo/taskflow.md)，无需训练，直接输入数据，即可得到预测结果，以情感分析任务为例：

```python
from paddlenlp import Taskflow
senta = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")
senta("作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。")
>>> [{'text': '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。', 'label': 'positive', 'score': 0.984320878982544}]
```

- 定制训练

如果对一键预测效果不满意，也可以进行模型微调，这里对SKEP模型进行微调，以进一步提升情感倾向预测的准确率：

```python
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
model = SkepForSequenceClassification.from_pretrained("skep_ernie_1.0_large_ch", num_classes=3)
tokenizer = SkepTokenizer.from_pretrained("skep_ernie_1.0_large_ch")
```
完整微调代码，可参考[SKEP微调](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/sentiment_analysis/skep)

更多内容可参考：[多场景示例](./docs/model_zoo/examples.md)、[PaddleNLP on AI Studio](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)。


## API文档

PaddleNLP提供全流程的文本领域API，可大幅提升NLP任务建模的效率：
- 支持丰富中文数据集加载的[Dataset API](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html)；
- 灵活高效地完成数据预处理的[Data API](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.data.html)；
- 提供500+预训练模型的[Transformers API](./docs/model_zoo/transformers.rst)。    

更多使用方法请参考[API文档](./docs/model_zoo/api.md)


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
