# 文档智能应用

**目录**
- [1. 文档智能应用简介](#文档智能应用简介)
- [2. 技术特色介绍](#技术特色介绍)
  - [2.1 多语言跨模态训练基座](#多语言跨模态训练基座)
  - [2.2 多场景覆盖](#多场景覆盖)
- [3. 快速开始](#快速开始)
  - [3.1 开箱即用](#开箱即用)
  - [3.2 产业级流程方案](#产业级流程方案)

## 1. 文档智能应用简介

文档智能（DI, Document Intelligence）主要指**对于网页、数字文档或扫描文档所包含的文本以及丰富的排版格式等信息，通过人工智能技术进行理解、分类、提取以及信息归纳**的过程。文档智能技术广泛应用于金融、保险、能源、物流、医疗等行业，常见的应用场景包括财务报销单、招聘简历、企业财报、合同文书、动产登记证、法律判决书、物流单据等多模态文档的关键信息抽取、文档解析、文档比对等。

在实际应用中，需要解决文档格式繁杂、布局多样、信息模态多样、需求开放、业务数据少等多重难题。针对文档智能领域的痛点和难点，PaddleNLP将持续开源一系列产业实践范例，解决开发者们实际应用难题。

<div align="center">
    <img width="1000" height="270" alt="文档智能技术一般流程" src="https://user-images.githubusercontent.com/40840292/196361583-6b1c66d1-6a9b-4193-949a-71e2d420a82a.png">
</div>

<a name="技术特色介绍"></a>

## 2. 技术特色介绍

<a name="多语言跨模态训练基座"></a>

### 2.1 多语言跨模态训练基座

近期，百度文心文档智能，基于多语言跨模态布局增强的文档智能大模型[ERNIE-Layout](http://arxiv.org/abs/2210.06155)，刷新了五类11项文档智能任务效果。依托文心ERNIE大模型，基于布局知识增强技术，融合文本、图像、布局等信息进行联合建模，能够对多模态文档（如文档图片、PDF 文件、扫描件等）进行深度理解与分析，为各类上层应用提供SOTA模型底座。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/196373896-597f6178-4c78-41a1-bb12-796546644b32.png width="600"/>
</div>

<a name="多场景覆盖"></a>

### 2.2 多场景覆盖

以下是文档智能技术的一些应用场景展示：

- 发票抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/196118171-fd3e49a0-b9f1-4536-a904-c48f709a2dec.png height=350 width=1000 hspace='10'/>
</div>

- 海报抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195610368-04230855-62de-439e-b708-2c195b70461f.png height=600 width=1000 hspace='15'/>
</div>

- 网页抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195611613-bdbe692e-d7f2-4a2b-b548-1a933463b0b9.png height=350 width=1000 hspace='10'/>
</div>


- 表格抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195610692-8367f1c8-32c2-4b5d-9514-a149795cf609.png height=350 width=1000 hspace='10'/>
</div>


- 试卷抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195823294-d891d95a-2ef8-4519-be59-0fedb96c00de.png height=700 width=1000 hspace='10'/>
</div>


- 英文票据多语种（中、英、日、泰、西班牙、俄语）抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195610820-7fb88608-b317-45fc-a6ab-97bf3b20a4ac.png height=400 width=1000 hspace='15'/>
</div>

- 中文票据多语种（中简、中繁、英、日、法语）抽取问答

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195611075-9323ce9f-134b-4657-ab1c-f4892075d909.png height=350 width=1000 hspace='15'/>
</div>

- Demo图片可在此[下载](https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/demo.zip)

<a name="快速开始"></a>

## 3. 快速开始

<a name="开箱即用"></a>

### 3.1 开箱即用

开源DocPrompt开放文档抽取问答模型，以ERNIE-Layout为底座，可精准理解图文信息，推理学习附加知识，准备捕捉图片、PDF等多模态文档中的每个细节。

🧾 通过[Huggingface网页](https://huggingface.co/spaces/PaddlePaddle/ERNIE-Layout)体验DocPrompt功能：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/195749427-864d7744-1fd1-455e-99c6-53a260776483.jpg height=700 width=1100 hspace='10'/>
</div>

#### Taskflow

通过``paddlenlp.Taskflow``三行代码调用DocPrompt功能，具备多语言文档抽取问答能力，部分应用场景展示如下：

- 输入格式

```
[
  {"doc": "./invoice.jpg", "prompt": ["发票号码是多少?", "校验码是多少?"]},
  {"doc": "./resume.png", "prompt": ["五百丁本次想要担任的是什么职位?", "五百丁是在哪里上的大学?", "大学学的是什么专业?"]}
]
```

默认使用PaddleOCR进行OCR识别，同时支持用户通过``word_boxes``传入自己的OCR结果，格式为``List[str, List[float, float, float, float]]``。

```
[
  {"doc": doc_path, "prompt": prompt, "word_boxes": word_boxes}
]
```

- 支持单条、批量预测

  - 支持本地图片路径输入

  <div align="center">
      <img src=https://user-images.githubusercontent.com/40840292/194748579-f9e8aa86-7f65-4827-bfae-824c037228b3.png height=800 hspace='20'/>
  </div>

  ```python
  >>> from pprint import pprint
  >>> from paddlenlp import Taskflow

  >>> docprompt = Taskflow("document_intelligence")
  >>> pprint(docprompt([{"doc": "./resume.png", "prompt": ["五百丁本次想要担任的是什么职位?", "五百丁是在哪里上的大学?", "大学学的是什么专业?"]}]))
  [{'prompt': '五百丁本次想要担任的是什么职位?',
    'result': [{'end': 7, 'prob': 1.0, 'start': 4, 'value': '客户经理'}]},
  {'prompt': '五百丁是在哪里上的大学?',
    'result': [{'end': 37, 'prob': 1.0, 'start': 31, 'value': '广州五百丁学院'}]},
  {'prompt': '大学学的是什么专业?',
    'result': [{'end': 44, 'prob': 0.82, 'start': 38, 'value': '金融学(本科）'}]}]
  ```

  - http图片链接输入

  <div align="center">
      <img src=https://user-images.githubusercontent.com/40840292/194748592-e20b2a5f-d36b-46fb-8057-86755d188af0.jpg height=400 hspace='10'/>
  </div>

  ```python
  >>> from pprint import pprint
  >>> from paddlenlp import Taskflow

  >>> docprompt = Taskflow("document_intelligence")
  >>> pprint(docprompt([{"doc": "https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/images/invoice.jpg", "prompt": ["发票号码是多少?", "校验码是多少?"]}]))
  [{'prompt': '发票号码是多少?',
    'result': [{'end': 2, 'prob': 0.74, 'start': 2, 'value': 'No44527206'}]},
  {'prompt': '校验码是多少?',
    'result': [{'end': 233,
                'prob': 1.0,
                'start': 231,
                'value': '01107 555427109891646'}]}]
  ```

- 可配置参数说明
  * `batch_size`：批处理大小，请结合机器情况进行调整，默认为1。
  * `lang`：选择PaddleOCR的语言，`ch`可在中英混合的图片中使用，`en`在英文图片上的效果更好，默认为`ch`。
  * `topn`: 如果模型识别出多个结果，将返回前n个概率值最高的结果，默认为1。

<a name="产业级流程方案"></a>

### 3.2 产业级流程方案

针对文档智能领域的痛点和难点，PaddleNLP将持续开源一系列文档智能产业实践范例，解决开发者们实际应用难题。

- 👉 [汽车说明书跨模态智能问答](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/document_intelligence/doc_vqa#readme)

更多：百度TextMind智能文档分析平台可提供包括文档信息抽取、文本内容审查、企业文档管理、文档格式解析、文档内容比对等全方位一站式的文档智能服务，已形成一套完整的企业文档场景化解决方案，满足银行、券商、法律、能源、传媒、通信、物流等不同行业和场景的文档处理需求，以AI助力企业的办公智能化升级和数字化转型。欢迎深度交流与商业合作，了解详情：https://ai.baidu.com/tech/nlp/Textanalysis

## References

- [文档智能：数据集、模型和应用](http://jcip.cipsc.org.cn/CN/abstract/abstract3331.shtml)

- [ERNIE-Layout: Layout-Knowledge Enhanced Multi-modal Pre-training for Document Understanding](http://arxiv.org/abs/2210.06155)
