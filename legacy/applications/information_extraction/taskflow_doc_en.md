# UIE Taskflow User Guide

**Table of contents**
- [1. Introduction](#1)
- [2. Document Information Extraction](#2)
   - [2.1 Entity Extraction](#21)
   - [2.2 Relation Extraction](#22)
   - [2.3 Multi-Task Extraction](#23)
   - [2.4 Input Format](#24)
   - [2.5 Tips](#25)
   - [2.6 Visualization](#26)
   - [2.7 More Configuration](#27)

<a name="1"></a>

## 1. Introduction

```paddlenlp.Taskflow``` provides general information extraction of text and documents, evaluation opinion extraction and other capabilities, and can extract various types of information, including but not limited to named entities (such as person name, place name, organization name, etc.), relations (such as the director of the movie, the release time of the song, etc.), events (such as a car accident at a certain intersection, an earthquake in a certain place, etc.), and information such as product reviews, opinions, and sentiments. Users can use natural language to customize the extraction target, and can uniformly extract the corresponding information in the input text or document without training.

<a name="2"></a>

## 2. Document Information Extraction

This section introduces the document extraction capability of Taskflow with the following example picture [download link](https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/cases.zip).

<a name="21"></a>

#### 2.1 Entity Extraction

Entity extraction, also known as Named Entity Recognition (NER for short), refers to identifying entities with specific meanings in text. UIE adopts the open-domain approach where the entity category is not fixed and the users can define them by through natural language.

- Example: Customs Declaration Form

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

- Example: Driver's License

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

#### 2.2 Relation Extraction

Relation Extraction refers to identifying entities from text and extracting the semantic relationship between entities, and then obtaining triple information, namely <subject, predicate, object>.

- Example: Extracting relations from a table

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

#### 2.3 Multi-Task Extraction

To extract entities and relation from documents simultaneously, you may set the schema structure as following:

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

#### 2.4 Input Format

For document information extraction, UIE-X supports image paths, http image links, base64 input form, and image and PDF document formats. In the input dict, `text` indicates text input and `doc` refer to the document input.

```python
[
    {'text': '2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！'},
    {'doc': './cases/custom.jpg'},
    {'doc': 'https://user-images.githubusercontent.com/40840292/203457719-84a70241-607e-4bb1-ab4c-3d9beee9e254.jpeg'}
]
```

**NOTE**: Multi-page PDF input currently only extracts the results of the first page. UIE-X is more suitable for information extraction of document documents (such as bills, receipts, etc.), but it is not suitable for documents that are too long or multi-page.

- Using custom OCR input

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

#### 2.5 Tips

- Using PP-Structure layout analysis function

The text recognized in OCR will be sorted from top left to bottom right. For cases such as column division and multiple lines of text in the table, we recommend using the layout analysis function ``layout_analysis=True`` to optimize text sorting and enhance the extraction effect. The following example is only an example of the usage scenario of the layout analysis function, and the actual scenario generally needs to be marked and fine-tuned.

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

#### 2.6 Visualization

- Visualization of OCR recognition results:

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

- Visualization of extraction results:

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

#### 2.7 More Configuration

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
                  precision='fp32')
```

* `schema`: Define the task extraction target, which can be configured by referring to the calling examples of different tasks in the out-of-the-box.
* `schema_lang`: Set the language of the schema, the default is `ch`, optional `ch` and `en`. Because the structure of the Chinese and English schemas is different, the language of the schema needs to be specified.
* `ocr_lang`: Select the language of PaddleOCR, `ch` can be used in mixed Chinese and English images, `en` works better on English images, the default is `ch`.
* `batch_size`: batch size, please adjust according to the machine situation, the default is 16.
* `model`: select the model used by the task, the default is `uie-base`, optional `uie-base`, `uie-medium`, `uie-mini`, `uie-micro`, `uie-nano` ` and `uie-medical-base`, `uie-base-en`, `uie-x-base`.
* `layout_analysis`: Whether to use PP-Structure to analyze the layout of the document to optimize the sorting of layout information, the default is False.
* `position_prob`: The result probability of the model for the start position/end position of the span is between 0 and 1, and the returned result removes the results less than this threshold, the default is 0.5, and the final probability output of the span is the start position probability and end position The product of the position probabilities.
* `precision`: select the model precision, the default is `fp32`, optional `fp16` and `fp32`. `fp16` inference is faster, support GPU and NPU hardware. If you choose `fp16` and GPU hardware, please ensure that the machine is correctly installed with NVIDIA-related drivers and basic software. **Ensure that CUDA>=11.2, cuDNN>=8.1.1**. For the first time use, you need to follow the prompts to install the relevant dependencies. Secondly, it is necessary to ensure that the CUDA Compute Capability of the GPU device is greater than 7.0. Typical devices include V100, T4, A10, A100, GTX 20 series and 30 series graphics cards, etc. For more information about CUDA Compute Capability and precision support, please refer to NVIDIA documentation: [GPU Hardware and Supported Precision Comparison Table](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix).

## References
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**
- **[PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/ppstructure)**
