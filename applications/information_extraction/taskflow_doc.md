# UIE Taskflow使用指南

**目录**
- [1. 功能简介](#1)
- [2. 应用示例](#2)
- [3. 文档信息抽取](#3)
  - [3.1 实体抽取](#31)
  - [3.2 关系抽取](#32)
  - [3.3 使用说明](#33)
  - [3.4 更多配置](#34)

<a name="1"></a>

## 1. 功能简介

```paddlenlp.Taskflow```提供文本及文档的通用信息抽取、评价观点抽取等能力，可抽取多种类型的信息，包括但不限于命名实体识别（如人名、地名、机构名等）、关系（如电影的导演、歌曲的发行时间等）、事件（如某路口发生车祸、某地发生地震等）、以及评价维度、观点词、情感倾向等信息。用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本或文档中的对应信息。**实现开箱即用，并满足各类信息抽取需求**

<a name="2"></a>

## 2. 应用示例

UIE不限定行业领域和抽取目标，以下是一些通过Taskflow实现开箱即用的行业示例：

- 医疗场景-专病结构化

![image](https://user-images.githubusercontent.com/40840292/169017581-93c8ee44-856d-4d17-970c-b6138d10f8bc.png)

- 法律场景-判决书抽取

![image](https://user-images.githubusercontent.com/40840292/169017863-442c50f1-bfd4-47d0-8d95-8b1d53cfba3c.png)

- 金融场景-收入证明、招股书抽取

![image](https://user-images.githubusercontent.com/40840292/169017982-e521ddf6-d233-41f3-974e-6f40f8f2edbc.png)

- 公安场景-事故报告抽取

![image](https://user-images.githubusercontent.com/40840292/169018340-31efc1bf-f54d-43f7-b62a-8f7ce9bf0536.png)

- 旅游场景-宣传册、手册抽取

![image](https://user-images.githubusercontent.com/40840292/169018113-c937eb0b-9fd7-4ecc-8615-bcdde2dac81d.png)

<a name="3"></a>

## 3. 文档信息抽取

UIE-X支持端到端的文档信息抽取，schema配置和输出形式与[文本信息抽取](#3)相同，下面是一些UIE-X的使用场景示例。

<a name="31"></a>

#### 3.1 实体抽取

- 证件

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/203457596-8dbc9241-833d-4b0e-9291-f134a790d0e1.jpeg height=350 width=500 hspace='10'/>
</div>

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow
>>> schema = ['姓名', '性别', '学校']
>>> ie = Taskflow("information_extraction", schema=schema, model="uie-x-base")
>>> pprint(ie({"doc": "./cases/student_id.jpeg"}))
[{'姓名': [{'end': 16,
          'probability': 0.6738645758156565,
          'start': 14,
          'text': '吴磊'}],
  '学校': [{'end': 9,
          'probability': 0.8072635771428587,
          'start': 0,
          'text': '四川省成都列五中学'}],
  '性别': [{'end': 20,
          'probability': 0.840880616278028,
          'start': 19,
          'text': '男'}]}]
```

- 单据、票据

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/203457719-84a70241-607e-4bb1-ab4c-3d9beee9e254.jpeg height=800 width=500 hspace='10'/>
</div>

```python
>>> schema = ['收发货人', '进口口岸', '进口日期', '申报日期', '提运单号']
>>> ie.set_schema(schema)
>>> pprint(ie({"doc": "./cases/custom.jpeg"}))
[{'提运单号': [{'end': 197,
            'probability': 0.985449746345779,
            'start': 188,
            'text': '769428175'}],
  '收发货人': [{'end': 95,
            'probability': 0.5309611882936238,
            'start': 82,
            'text': '上海新尚实国际贸易有限公司'}],
  '申报日期': [{'end': 140,
            'probability': 0.9262545894964944,
            'start': 130,
            'text': '2017-02-23'}],
  '进口口岸': [{'end': 120,
            'probability': 0.9799873036392412,
            'start': 111,
            'text': '洋山港区-2248'}],
  '进口日期': [{'end': 130,
            'probability': 0.8883286952976022,
            'start': 120,
            'text': '2017-02-24'}]}]
```


<a name="32"></a>

#### 3.2 关系抽取

- 单据、票据

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/203457817-76fe638a-3277-4619-9066-d1dffd52c5d4.jpg height=400 width=650 hspace='10'/>
</div>

```python
>>> schema = {'项目名': '单价'}
>>> ie.set_schema(schema)
>>> pprint(ie({'doc': "./cases/medical.png"}))
[{'项目名': [{'end': 178,
           'probability': 0.6219053739514173,
           'relations': {'单价': [{'end': 142,
                                 'probability': 0.5131751051094824,
                                 'start': 136,
                                 'text': '26.000'}]},
           'start': 174,
           'text': '头针治疗'},
          {'end': 158,
           'probability': 0.576683802116392,
           'relations': {'单价': [{'end': 164,
                                 'probability': 0.9167558518149299,
                                 'start': 158,
                                 'text': '54.000'}]},
           'start': 152,
           'text': '特殊穴位针刺'},
          {'end': 136,
           'probability': 0.666216406596444,
           'relations': {'单价': [{'end': 142,
                                 'probability': 0.7802317480024143,
                                 'start': 136,
                                 'text': '26.000'}]},
           'start': 131,
           'text': '腕踝针治疗'}]}]
```

<a name="33"></a>

#### 3.3 使用说明

- 输入格式

UIE-X支持图片路径、http图片链接、base64的输入形式，支持图片和PDF两种文档格式。

```python
[
    {'doc': './invoice.jpg'},
    {'doc': 'https://user-images.githubusercontent.com/40840292/203457719-84a70241-607e-4bb1-ab4c-3d9beee9e254.jpeg'}
]
```

**NOTE**: 多页PDF输入只抽取第一页的结果，UIE-X比较适合单证文档（如票据、单据等）的信息提取，目前不适合过长或多页的文档。

<a name="34"></a>

#### 3.4 更多配置

```python
>>> from paddlenlp import Taskflow

>>> ie = Taskflow('information_extraction',
                  schema="",
                  schema_lang="ch",
                  ocr_lang="ch",
                  batch_size=16,
                  model='uie-x-base',
                  layout_analysis=False,
                  position_prob=0.5,
                  precision='fp32',
                  use_fast=False)
```

* `schema`：定义任务抽取目标，可参考开箱即用中不同任务的调用示例进行配置。
* `schema_lang`：设置schema的语言，默认为`ch`, 可选有`ch`和`en`。因为中英schema的构造有所不同，因此需要指定schema的语言。
* `ocr_lang`：选择PaddleOCR的语言，`ch`可在中英混合的图片中使用，`en`在英文图片上的效果更好，默认为`ch`。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为16。
* `model`：选择任务使用的模型，默认为`uie-base`，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`, `uie-nano`和`uie-medical-base`, `uie-base-en`，`uie-x-base`。
* `layout_analysis`：是否使用PPStructure对文档进行布局分析以优化布局信息的排序，默认为False。
* `position_prob`：模型对于span的起始位置/终止位置的结果概率在0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span的最终概率输出为起始位置概率和终止位置概率的乘积。
* `precision`：选择模型精度，默认为`fp32`，可选有`fp16`和`fp32`。`fp16`推理速度更快。如果选择`fp16`，请先确保机器正确安装NVIDIA相关驱动和基础软件，**确保CUDA>=11.2，cuDNN>=8.1.1**，初次使用需按照提示安装相关依赖。其次，需要确保GPU设备的CUDA计算能力（CUDA Compute Capability）大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)。
* `use_fast`: 使用C++实现的高性能分词算子FastTokenizer进行文本预处理加速。需要通过`pip install fast-tokenizer-python`安装FastTokenizer库后方可使用。默认为`False`。更多使用说明可参考[FastTokenizer文档](../../fast_tokenizer)。
