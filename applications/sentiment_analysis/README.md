# 情感分析应用

**目录**
- [1. 情感分析应用简介](#1)
- [2. 技术特色介绍](#2)
  - [2.1 提供强大训练基座，覆盖情感分析多项基础能力](#2.1)
  - [2.2 用户友好的情感分析方案，从输入数据直达分析结果可视化](#2.2)
  - [2.3 支持定制面向垂域的情感分析能力，解决同义属性聚合以及隐性观点抽取](#2.3)
  - [2.4 提供面向属性、观点和情感极性抽取的阶段式定制方案](#2.4)
- [3. 快速开始](#3)

<a name="1"></a>

## **1. 情感分析应用简介**
情感分析（sentiment analysis）是近年来国内外研究的热点，旨在对带有情感色彩的主观性文本进行分析、处理、归纳和推理，其广泛应用于消费决策、舆情分析、个性化推荐等领域，具有很高的商业价值。

情感分析包含较多的任务，按分析粒度可以分为篇章级的情感分析（Document-Level Sentiment Classification）、语句级的情感分析（Sentence-Level Sentiment Classification）和属性级的情感分析（Aspect-Level Sentiment Classification）。其中属性级的情感分析又包含多项子任务，例如属性抽取（Aspect Term Extraction）、观点抽取（Opinion Term Extraction）、属性级情感分析（Aspect Based Sentiment Classification）等。可以看到，情感分析包含较多的子方向，因此较难将其简单归为单一的技术领域，往往从不同的角度将其划归到不同的方向。如果单纯地判别文本的倾向性，可以将其看作是一个分类任务；如果要从观点句中抽取相关的要素（属性、观点等），则是一个信息抽取任务。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/199726520-812962ec-b4ae-4250-a903-4b33a07deff9.png" />
</div>
<br>

在业务实践中，往往需要针对文本评论分析多项较为细粒度的情感信息，例如文本评论中的属性、观点、情感极性等信息。当前常见的建模方式有阶段式和端到端式的建模方法，其中阶段式建模是比较经典的建模方式，即先抽取一部分信息，再根据抽取的信息抽取其他信息，例如先抽取属性、再抽取该属性对应的观点，最后抽取其对应的情感极性。

PaddleNLP情感分析应用立足真实企业用户对情感分析方面的需求，同时针对情感分析领域的痛点和难点，基于前沿模型开源了细粒度的情感分析解决方案，助力开发者快速分析业务相关产品或服务的用户感受

<a name="2"></a>

## **2. 技术特色介绍**

<a name="2.1"></a>

### 2.1 提供强大训练基座，覆盖情感分析多项基础能力

PaddleNLP情感分析应用基于中文通用信息抽取模型UIE，利用大量情感分析数据进行训练，增强了模型对于情感知识的处理能力，支持整句情感分类、属性-情感抽取（Aspect-Sentiment Extraction, 简记为A-S）,属性-观点（Aspect-Opinion Extraction, 简记为A-O,属性-情感-观点抽取（Aspect-Sentiment-Opinion, 简记为A-S-O）等基础情感分析能力。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/199965793-f0933baa-5b82-47da-9271-ba36642119f8.png" />
</div>
<br>

<a name="2.2"></a>

### **2.2 用户友好的情感分析方案，从输入数据直达分析结果可视化**

#### **2.2.1 属性/观点分析**
通过属性信息，可以查看客户对于产品/服务的重点关注方面；通过观点信息，可以查看客户对于产品/服务整体的直观印象。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/199973186-be978c42-dc92-40f1-b493-e122ac9bdc6e.png" />
</div>
<br>

#### **2.2.2 属性+观点分析**
结合属性和观点两者信息，可以更加具体的展现客户对于产品/服务的详细观点，分析某个属性的优劣，从而能够帮助商家更有针对性地改善或提高自己的产品/服务质量。

**全部属性+观点的内容分析**
<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/199974942-8e55aabd-6c35-48ec-8f6d-3270b67b299c.png"/>
</div>

同时为更加方便地帮助商家发现自己产品/服务的问题，也支持按照积极和消极两个方面单独分析。

**正向属性+观点的内容分析**

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/199976730-b72f653d-e5b9-487e-98bd-ceed821be0fb.png"/>
</div>

**负向属性+观点的内容分析**

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/199977230-4d4eb7db-3ebf-4858-883a-5fa61e8542ce.png"/>
</div>

#### **2.2.3 属性+情感极性分析**
挖掘客户对于产品/服务针对属性的情感极性，帮助商家直观地查看客户对于产品/服务的某些属性的印象。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/200213177-0342bec4-5955-4ab9-9e98-5e4ef8e1a35e.png"/>
</div>

#### **2.2.4 固定属性和观点分析**
通过指定属性，更加细致查看客户对于产品/服务某个属性的观点。可以帮助商家更加细粒度地分析客户对于产品/服务的某个属性的印象。下面图片示例中，展示了客户对于属性"房间"的观点。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/200213998-e646c422-7ab5-48ae-9e28-d6068cdf7b8f.png"/>
</div>

<a name="2.3"></a>

### **2.3 支持定制面向垂域的情感分析能力，解决同义属性聚合以及隐性观点抽取**
考虑到用户在对业务数据进行情感分析时，往往聚焦于某个特定场景或领域，为满足用户更高的情感分析要求，本项目除了预先设定的通用情感分析能力之外，同时支持进一步地微调，以在当前业务侧获取更好的效果。

为方便用户快速定制业务侧情感分析，本项目提供了数据标注及样本构建 - 模型训练 - 模型测试 - 模型预测及效果展示等全流程指导，同时在用户提供属性同义词表以及隐性观点词表的情况下，还能进一步加强模型对于属性聚合和隐性观点抽取的能力。

**属性聚合示例：**
<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/203913660-ac95caad-c5e2-43c5-b291-6208babd58d3.png />
</div>

**隐性观点抽取示例：**
<div align="center">
    <img src=https://user-images.githubusercontent.com/35913314/203913490-a6fbf0aa-1f9c-476d-83c7-ea4604ab94d0.png />
</div>

<a name="2.4"></a>

### **2.4 提供面向属性、观点和情感极性抽取的阶段式定制方案**

除了抽取式建模方式，本项目同时提供了基于序列标注和情感极性分类的阶段式分析方案，该方案默认基于情感知识增强的模型 SKEP 进行情感分析，同时也可以更新为其他模型，方便使用。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/204823817-697910c0-cf22-4e0e-8b8d-1b0278c2dbeb.png" />
</div>

<a name="3"></a>

## **3. 快速开始**

- 👉 [通用情感分析抽取](./unified_sentiment_extraction/README)

- 👉 [阶段式属性、观点和情感极性抽取](./ASO_analysis/README)
