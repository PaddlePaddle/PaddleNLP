# UIE Taskflow 使用指南

**目录**
- [1. 功能简介](#1)
- [2. 文档信息抽取](#2)
  - [2.1 实体抽取](#21)
  - [2.2 关系抽取](#22)
  - [2.3 跨任务使用](#23)
  - [2.4 输入说明](#24)
  - [2.5 使用技巧](#25)
  - [2.6 结果可视化](#26)
  - [2.7 更多配置](#27)

<a name="1"></a>

## 1. 功能简介

```paddlenlp.Taskflow```提供文本及文档的通用信息抽取、评价观点抽取等能力，可抽取多种类型的信息，包括但不限于命名实体识别（如人名、地名、机构名等）、关系（如电影的导演、歌曲的发行时间等）、事件（如某路口发生车祸、某地发生地震等）、以及评价维度、观点词、情感倾向等信息。用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本或文档中的对应信息。**实现开箱即用，并满足各类信息抽取需求**

<a name="2"></a>

## 2. 文档信息抽取

本章节主要介绍 Taskflow 的文档抽取功能，以下示例图片[下载链接](https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/cases.zip)。

<a name="21"></a>

#### 2.1 实体抽取

实体抽取，又称命名实体识别（Named Entity Recognition，简称 NER），是指识别文本中具有特定意义的实体。在开放域信息抽取中，抽取的类别没有限制，用户可以自己定义。

- 报关单

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/206112148-82e26dad-4a77-40e3-bc11-f877047aeb87.png height=700 width=450 hspace='10'/>
</div>

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow
>>> schema = ["收发货人", "进口口岸", "进口日期", "运输方式", "征免性质", "境内目的地", "运输工具名称", "包装种类", "件数", "合同协议号"]
>>> ie = Taskflow("information_extraction", schema=schema, model="uie-x-base")
>>> pprint(ie({"doc": "./cases/custom.jpeg"}))
[{'件数': [{'bbox': [[826, 1062, 926, 1121]],
        'end': 312,
        'probability': 0.9832498761402597,
        'start': 308,
        'text': '1142'}],
'包装种类': [{'bbox': [[1214, 1066, 1310, 1121]],
            'end': 314,
            'probability': 0.9995648138860567,
            'start': 312,
            'text': '纸箱'}],
'合同协议号': [{'bbox': [[151, 1077, 258, 1117]],
            'end': 319,
            'probability': 0.9984179437542124,
            'start': 314,
            'text': '33035'}],
'境内目的地': [{'bbox': [[1966, 872, 2095, 923]],
            'end': 275,
            'probability': 0.9975541483111243,
            'start': 272,
            'text': '上海市'}],
'征免性质': [{'bbox': [[1583, 770, 1756, 821]],
            'end': 242,
            'probability': 0.9950633161231508,
            'start': 238,
            'text': '一般征税'}],
'收发货人': [{'bbox': [[321, 533, 841, 580]],
            'end': 95,
            'probability': 0.4772132061042136,
            'start': 82,
            'text': '上海新尚实国际贸易有限公司'},
        {'bbox': [[306, 584, 516, 624]],
            'end': 150,
            'probability': 0.33807074572195006,
            'start': 140,
            'text': '31222609K9'}],
'运输工具名称': [{'bbox': [[1306, 672, 1516, 712], [1549, 668, 1645, 712]],
            'end': 190,
            'probability': 0.6692050414718089,
            'start': 174,
            'text': 'E. R. TIANAN004E'}],
'运输方式': [{'bbox': [[1070, 664, 1240, 715]],
            'end': 174,
            'probability': 0.9994416347044179,
            'start': 170,
            'text': '永路运输'}],
'进口口岸': [{'bbox': [[1070, 566, 1346, 617]],
            'end': 120,
            'probability': 0.9945697196994345,
            'start': 111,
            'text': '洋山港区-2248'}],
'进口日期': [{'bbox': [[1726, 569, 1933, 610]],
            'end': 130,
            'probability': 0.9804819494073627,
            'start': 120,
            'text': '2017-02-24'}]}]
```

- 证件


<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/206114081-8c82e2a2-0c88-4ca3-9651-b12c94266be9.png height=400 width=700 hspace='10'/>
</div>

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow
>>> schema = ["Name", "Date of birth", "Issue date"]
>>> ie = Taskflow("information_extraction", schema=schema, model="uie-x-base", ocr_lang="en", schema_lang="en")
>>> pprint(ie({"doc": "./cases/license.jpeg"}))
```

<a name="22"></a>

#### 2.2 关系抽取

关系抽取（Relation Extraction，简称 RE），是指从文本中识别实体并抽取实体之间的语义关系，进而获取三元组信息，即<主体，谓语，客体>。

- 表格

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/206115688-30de315a-8fd4-4125-a3c3-8cb05c6e39e5.png height=180 width=600 hspace='10'/>
</div>

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow
>>> schema = {"姓名": ["招聘单位", "报考岗位"]}
>>>> ie = Taskflow("information_extraction", schema=schema, model="uie-x-base")
>>> pprint(ie({"doc": "./cases/table.png"}))
```

<a name="23"></a>

#### 2.3 跨任务使用

- 实体、关系多任务抽取

对文档进行实体+关系抽取，schema 构造如下：

```text
schema = [
    "Total GBP",
    "No.",
    "Date",
    "Customer No.",
    "Subtotal without VAT",
    {
        "Description": [
            "Quantity",
            "Amount"
        ]
    }
]
```

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/206120861-13b475dc-9a78-43bc-9dec-91f331db2ddf.png height=400 width=650 hspace='10'/>
</div>

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow

>>> schema = ["Total GBP", "No.", "Date", "Customer No.", "Subtotal without VAT", {"Description": ["Quantity", "Amount"]}]
>>> ie = Taskflow("information_extraction", schema=schema, model="uie-x-base", ocr_lang="en", schema_lang="en")
>>> pprint(ie({"doc": "./cases/delivery_note.png"}))
```

<a name="24"></a>

#### 2.4 输入说明

- 输入格式

文档抽取 UIE-X 支持图片路径、http 图片链接、base64的输入形式，支持图片和 PDF 两种文档格式。文本抽取可以通过`text`指定输入文本。

```python
[
    {'text': '2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！'},
    {'doc': './cases/custom.jpg'},
    {'doc': 'https://user-images.githubusercontent.com/40840292/203457719-84a70241-607e-4bb1-ab4c-3d9beee9e254.jpeg'}
]
```

**NOTE**: 多页 PDF 输入目前只抽取第一页的结果，UIE-X 比较适合单证文档（如票据、单据等）的信息提取，目前还不适合过长或多页的文档。

- 使用自己的 layout / OCR 作为输入

```python
layout = [
    ([68.0, 12.0, 167.0, 70.0], '名次'),
    ([464.0, 13.0, 559.0, 67.0], '球员'),
    ([833.0, 15.0, 1054.0, 64.0], '总出场时间'),
    ......
]
ie({"doc": doc_path, 'layout': layout})
```

<a name="25"></a>

#### 2.5 使用技巧

- 使用 PP-Structure 版面分析功能

OCR 中识别出来的文字会按照左上到右下进行排序，对于分栏、表格内有多行文本等情况我们推荐使用版面分析功能``layout_analysis=True``以优化文字排序并增强抽取效果。以下例子仅举例版面分析功能的使用场景，实际场景一般需要标注微调。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/206139057-aedec98f-683c-4648-999d-81ce5ea04a86.png height=250 width=500 hspace='10'/>
</div>

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow

>>> schema = "中标候选人名称"
>>> ie = Taskflow("information_extraction", schema=schema, model="uie-x-base", layout_analysis=True)
>>> pprint(ie({"doc": "https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fwww.xuyiwater.com%2Fwp-content%2Fuploads%2F2021%2F06%2F1-4.jpg&refer=http%3A%2F%2Fwww.xuyiwater.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1672994926&t=2a4a3fedf6999a34ccde190f97bcfa47"}))
```

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/206137978-3a69e7e2-dc2e-4d11-98b7-25911b0375a0.png height=350 width=600 hspace='10'/>
</div>

```python
>>> schema = "抗血小板药物的用药指征"
>>> ie.set_schema(schema)
>>> pprint(ie({"doc": "./cases/drug.webp"}))
```

<a name="26"></a>

#### 2.6 结果可视化

- OCR 识别结果可视化：

```python
>>> from paddlenlp.utils.doc_parser import DocParser

>>> doc_parser = DocParser(ocr_lang="en")
>>> doc_path = "./cases/business_card.png"
>>> parsed_doc = doc_parser.parse({"doc": doc_path})
>>> doc_parser.write_image_with_results(
        doc_path,
        layout=parsed_doc['layout'],
        save_path="ocr_result.png")
```

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/206168103-0a37eab0-bb36-4eec-bd51-b3f85838b40c.png height=350 width=600 hspace='10'/>
</div>

- 抽取结果可视化：

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow
>>> from paddlenlp.utils.doc_parser import DocParser

>>> doc_path = "./cases/business_card.png"
>>> schema = ["人名", "职位", "号码", "邮箱地址", "网址", "地址", "邮编"]
>>> ie = Taskflow("information_extraction", schema=schema, model="uie-x-base", ocr_lang="en")

>>> results = ie({"doc": doc_path})

>>> DocParser.write_image_with_results(
        doc_path,
        result=results[0],
        save_path="image_show.png")
```

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/206168852-c32c34c4-f245-4116-a244-390e55c13383.png height=350 width=600 hspace='10'/>
</div>

<a name="27"></a>

#### 2.7 更多配置

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
* `schema_lang`：设置 schema 的语言，默认为`ch`, 可选有`ch`和`en`。因为中英 schema 的构造有所不同，因此需要指定 schema 的语言。
* `ocr_lang`：选择 PaddleOCR 的语言，`ch`可在中英混合的图片中使用，`en`在英文图片上的效果更好，默认为`ch`。
* `batch_size`：批处理大小，请结合机器情况进行调整，默认为16。
* `model`：选择任务使用的模型，默认为`uie-base`，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`, `uie-nano`和`uie-medical-base`, `uie-base-en`，`uie-x-base`。
* `layout_analysis`：是否使用 PP-Structure 对文档进行布局分析以优化布局信息的排序，默认为 False。
* `position_prob`：模型对于 span 的起始位置/终止位置的结果概率在0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span 的最终概率输出为起始位置概率和终止位置概率的乘积。
* `precision`：选择模型精度，默认为`fp32`，可选有`fp16`和`fp32`。`fp16`推理速度更快，支持 GPU 和 NPU 硬件环境。如果选择`fp16`，在 GPU 硬件环境下，请先确保机器正确安装 NVIDIA 相关驱动和基础软件，**确保 CUDA>=11.2，cuDNN>=8.1.1**，初次使用需按照提示安装相关依赖。其次，需要确保 GPU 设备的 CUDA 计算能力（CUDA Compute Capability）大于7.0，典型的设备包括 V100、T4、A10、A100、GTX 20系列和30系列显卡等。更多关于 CUDA Compute Capability 和精度支持情况请参考 NVIDIA 文档：[GPU 硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)。

## References
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**
- **[PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/ppstructure)**
