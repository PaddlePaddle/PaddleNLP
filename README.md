简体中文 | [English](./README_en.md)

<p align="center">
  <img src="./docs/imgs/paddlenlp.png" width="520" height ="100" />
</p>

------------------------------------------------------------------------------------------

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

## 简介

PaddleNLP 2.0拥有**覆盖多场景的模型库**、**简洁易用的全流程API**与**动静统一的高性能分布式训练**能力，旨在为飞桨开发者提升文本领域建模效率，并提供基于PaddlePaddle 2.0的NLP领域最佳实践。

## 特性

- **覆盖多场景的模型库**
  - PaddleNLP集成了RNN与Transformer等多种主流模型结构，涵盖从[词向量](./examples/word_embedding/)、[词法分析](./examples/lexical_analysis/)、[命名实体识别](./examples/information_extraction/msra_ner/)、[语义表示](./examples/language_model/)等NLP基础技术，到[文本分类](./examples/text_classification/)、[文本匹配](./examples/text_matching/)、[文本生成](./examples/text_generation/)、[文本图学习](./examples/text_graph/erniesage/)、[信息抽取](./examples/information_extraction)等NLP核心技术。同时针对[机器翻译](./examples/machine_translation/)、[通用对话](./examples/dialogue/)、[阅读理解](./examples/machine_reading_comprehension/)等系统应用提供相应核心组件与预训练模型。更多详细介绍请查看[PaddleNLP应用示例](./examples/)。


- **简洁易用的全流程API**
  - 深度兼容飞桨2.0的[高层API](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/quick_start/high_level_api/high_level_api.html)体系，内置可复用的文本建模模块([Embedding](./docs/embeddings.md), [CRF](./paddlenlp/layers/crf.py), [Seq2Vec](./paddlenlp/seq2vec/encoder.py), [Transformer](./docs/transformers.md))，可大幅度减少在数据处理、模型组网、训练与评估、推理部署环节的开发量，提升NLP任务迭代与落地的效率。

- **动静统一的高性能分布式训练**
  - 基于飞桨2.0核心框架『动静统一』的特性与领先的混合精度优化策略，结合Fleet分布式训练API，可充分利用GPU集群资源，高效完成大规模预训练模型的分布式训练。


## 安装

### 环境依赖

- python >= 3.6
- paddlepaddle >= 2.0.0

### pip安装

```
pip install paddlenlp\>=2.0.0rc
```

## 快速开始

### 数据集快速加载

```python
from paddlenlp.datasets import load_dataset

train_ds, dev_ds, test_ds = load_dataset("chnsenticorp", splits=["train", "dev", "test"])
```

可参考[Dataset文档](./docs/datasets.md)查看更多数据集。

### 一键加载中文词向量

```python
from paddlenlp.embeddings import TokenEmbedding

wordemb = TokenEmbedding("w2v.baidu_encyclopedia.target.word-word.dim300")
print(wordemb.cosine_sim("国王", "王后"))
>>> 0.63395125
wordemb.cosine_sim("艺术", "火车")
>>> 0.14792643
```

内置50+中文词向量，更多使用方法请参考[Embedding文档](./examples/word_embedding/README.md)。


### 一键加载高质量中文预训练模型

```python
from paddlenlp.transformers import ErnieModel, BertModel, RobertaModel, ElectraModel, GPT2ForPretraining

ernie = ErnieModel.from_pretrained('ernie-1.0')
bert = BertModel.from_pretrained('bert-wwm-chinese')
roberta = RobertaModel.from_pretrained('roberta-wwm-ext')
electra = ElectraModel.from_pretrained('chinese-electra-small')
gpt2 = GPT2ForPretraining.from_pretrained('gpt2-base-cn')
```

### 便捷获取文本特征

```python
import paddle
from paddlenlp.transformers import ErnieTokenizer, ErnieModel

tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
model = ErnieModel.from_pretrained('ernie-1.0')

text = tokenizer('自然语言处理')
pooled_output, sequence_output = model(input_ids=paddle.to_tensor([text['input_ids']]))
```

请参考[Transformer API文档](./docs/transformers.md)查看目前支持的预训练模型。

## 模型库及其应用

PaddleNLP模型库整体介绍请参考文档[PaddleNLP Model Zoo](./docs/model_zoo.md)。
模型应用场景介绍请参考[PaddleNLP Examples](./examples/)。

- [词向量](./examples/word_embedding/)
- [词法分析](./examples/lexical_analysis/)
- [命名实体识别](./examples/information_extraction/msra_ner/)
- [语言模型](./examples/language_model/)
- [文本分类](./examples/text_classification/)
- [文本生成](./examples/text_generation/)
- [语义匹配](./examples/text_matching/)
- [文本图学习](./examples/text_graph/erniesage/)
- [信息抽取](./examples/information_extraction/)
- [通用对话](./examples/dialogue/)
- [机器翻译](./examples/machine_translation/)
- [阅读理解](./examples/machine_reading_comprehension/)

## 进阶应用

- [模型压缩](./examples/model_compression/)

## API 使用文档

- [Transformer API](./docs/transformers.md)
  * 基于Transformer结构相关的预训练模型API，包含ERNIE, BERT, RoBERTa, Electra等主流经典结构和下游任务。
- [Data API](./docs/data.md)
  * 文本数据处理Pipeline的相关API说明。
- [Dataset API](./docs/datasets.md)
  * 数据集相关API，包含自定义数据集，数据集贡献与数据集快速加载等功能说明。
- [Embedding API](./docs/embeddings.md)
  * 词向量相关API，支持一键快速加载包预训练的中文词向量，VisulDL高维可视化等功能说明。
- [Metrics API](./docs/metrics.md)
  * 针对NLP场景的评估指标说明，与飞桨2.0框架高层API兼容。


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

- 欢迎您加入PaddleNLP的SIG社区，贡献优秀的模型实现、公开数据集、教程与案例、外围小工具。
- 现在就加入PaddleNLP的QQ技术交流群，一起交流NLP技术吧！⬇️

<div align="center">
  <img src="./docs/imgs/qq.png" width="200" height="200" />
</div>  


## License

PaddleNLP遵循[Apache-2.0开源协议](./LICENSE)。
