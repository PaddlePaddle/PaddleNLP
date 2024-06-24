# 产业级端到端系统范例

## 1、简介

PaddleNLP 从预训练模型库出发，提供了经典预训练模型在主流 NLP 任务上丰富的[应用示例](../examples)，满足了大量开发者的学习科研与基础应用需求。

针对更广泛的产业落地需求、更复杂的 NLP 场景任务，PaddleNLP 推出**产业级端到端系统范例库**（下文简称产业范例），提供单个模型之上的产业解决方案。

- 最强模型与实践———产业范例针对具体业务场景，提供最佳模型（组合），兼顾模型精度与性能，降低开发者模型选型成本；
- 全流程———打通数据标注-模型训练-模型调优-模型压缩—预测部署全流程，帮助开发者更低成本得完成产业落地。

## 2、基于 Pipelines 构建产业范例，加速落地

在面向不同场景任务建设一系列产业方案的过程中，不难发现，从技术基础设施角度看：

（1）NLP系统都可以抽象为由多个基础组件串接而成的流水线系统；
（2）多个NLP流水线系统可共享使用相同的基础组件。

因此，PaddleNLP 逐渐孵化出了一套 NLP 流水线系统 [Pipelines](../pipelines)，将各个 NLP 复杂系统的通用模块抽象封装为标准组件，支持开发者通过配置文件对标准组件进行组合，仅需几分钟即可定制化构建智能系统，让解决NLP任务像搭积木一样便捷、灵活、高效。同时，Pipelines 中预置了前沿的预训练模型和算法，在研发效率、模型效果和性能方面提供多重保障。因此，Pipelines 能够大幅加快开发者使用飞桨落地的效率。


<div>
    <img src="https://user-images.githubusercontent.com/11793384/212836991-d9132e46-b5bf-4389-80e1-4f9dee32f1fe.png" width="90%" length="90%">
</div>

<br>

**PaddleNLP 提供了多个版本的产业范例:**

- 如果你希望快速体验、直接应用、从零搭建一套完整系统，推荐使用 **Pipelines 版本**。这里集成了训练好的模型，无需关心模型训练细节；提供 Docker 环境，可快速一键部署端到端系统；打通前端 Demo 界面，便于直观展示、分析、调试效果。
- 如果你希望使用自己的业务数据进行二次开发，推荐使用`./applications`目录下的**可定制版本**，训练好的模型可以直接集成进 Pipelines 中进行使用。
- 也可以使用 [AI Studio](https://aistudio.baidu.com/aistudio/index) 在线 Jupyter Notebook 快速体验，有 GPU 算力哦。

| 场景任务   | Pipelines版本地址 | 可定制版本地址 | Notebook |
| :--------------- | ------- | ------- | ------- |
| **检索**| [字面+语义检索](../pipelines/examples/semantic-search) | [语义检索](./neural_search) | [基于Pipelines搭建检索系统](https://aistudio.baidu.com/aistudio/projectdetail/4442670)<br>[二次开发语义检索](https://aistudio.baidu.com/aistudio/projectdetail/3351784) |
| **问答** | [FAQ问答](../pipelines/examples/FAQ/)<br>[无监督检索式问答](../pipelines/examples/unsupervised-question-answering)<br>[有监督检索式问答](../pipelines/examples/question-answering) | [FAQ问答](./question_answering/supervised_qa)<br>[无监督检索式问答](./question_answering/unsupervised_qa) | [基于Pipelines搭建FAQ问答系统](https://aistudio.baidu.com/aistudio/projectdetail/4465498)<br>[基于Pipelines搭建抽取式问答系统](https://aistudio.baidu.com/aistudio/projectdetail/4442857)<br>[FAQ政务问答](https://aistudio.baidu.com/aistudio/projectdetail/3678873)<br>[FAQ保险问答](https://aistudio.baidu.com/aistudio/projectdetail/3882519) |
| **文本分类**| 暂无 | [文本分类](./text_classification)  | [对话意图识别](https://aistudio.baidu.com/aistudio/projectdetail/2017202)<br>[法律文本多标签分类](https://aistudio.baidu.com/aistudio/projectdetail/3996601)<br>[层次分类](https://aistudio.baidu.com/aistudio/projectdetail/4568985) |
| **通用文本分类** | 暂无 | [通用文本分类](./zero_shot_text_classification) |  |
| **通用信息抽取** | 暂无 | [通用信息抽取](./information_extraction) | [UIE快速体验](https://aistudio.baidu.com/aistudio/projectdetail/3914778)<br>[UIE微调实体抽取](https://aistudio.baidu.com/aistudio/projectdetail/4038499)<br>[UIE微调关系抽取](https://aistudio.baidu.com/aistudio/projectdetail/4371345)<br>[UIE-X快速体验](https://aistudio.baidu.com/aistudio/projectdetail/5017442)<br>[UIE-X微调](https://aistudio.baidu.com/aistudio/projectdetail/5261592) |
| **情感分析**  | [情感分析](../pipelines/examples/sentiment_analysis)  | [情感分析](./sentiment_analysis) |  [情感分析](https://aistudio.baidu.com/aistudio/projectdetail/5318177)|
| **文档智能**  | [文档抽取问答](../pipelines/examples/document-intelligence) |  [跨模态文档问答](./document_intelligence/doc_vqa)| [文档抽取问答](https://aistudio.baidu.com/aistudio/projectdetail/4881278)<br>[汽车说明书问答](https://aistudio.baidu.com/aistudio/projectdetail/4049663)  |
| **文生图**  | [文生图系统](../pipelines/examples/text_to_image)  | 可参考[PPDiffusers](../ppdiffusers) |   |
| **语音指令解析**  | 暂无 | [语音指令解析](./speech_cmd_analysis) | [语音指令解析](https://aistudio.baidu.com/aistudio/projectdetail/4399703) |
| **文本摘要**  | 暂无 | [文本摘要](./text_summarization) | [文本摘要](https://aistudio.baidu.com/aistudio/projectdetail/4903667) |

## 3、典型范例介绍

#### 📄 通用信息抽取系统

- 首个产业级通用信息抽取方案 UIE，面向纯文本，实现多任务统一建模，提供强大的零样本抽取和少样本快速迁移能力；
- 首个兼具文本及文档抽取能力、多语言、开放域的信息抽取方案 UIE-X，基于 [ERNIE-Layout](../model_zoo/ernie-layout) 跨模态布局增强预训练模型，集成 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 的 PP-OCR、PP-Structure 版面分析能力，小样本文档信息抽取效果领先。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/213365046-69967745-b4a8-4435-98fb-c34f68cd22e9.png" width="60%" length="60%">
</div>


详细使用说明请参考[通用信息抽取系统](./information_extraction)，更多：[UIE 解读](https://mp.weixin.qq.com/s/-hHz8knHIKKqKCBTke7i5A)、[UIE-X 解读](https://zhuanlan.zhihu.com/p/592422623)。

#### 🔍 语义检索系统

- 前沿算法———基于 SimCSE、In-batch Negatives、ERNIE Pairwise、RocketQA Pointwise 等提供针对无监督、有监督等多种数据情况的多样化方案；
- 全流程———覆盖召回、排序环节，集成主流 ANN 引擎，同时兼容 ElasticSearch 字面检索模式，提供多路召回方案。打通训练、调优、高效向量检索引擎建库和查询全流程。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/213134465-30cae5fd-4cd1-4e5b-a1cb-fa55c72980a7.gif" width="60%" length="60%">
</div>

详细使用说明请参考[语义检索系统](./neural_search)。

#### ❓ 智能问答系统

- 端到端问答技术 [🚀RocketQA](https://github.com/PaddlePaddle/RocketQA)，首个中文端到端问答模型，基于知识增强的预训练模型ERNIE和百万量级的人工标注数据集DuReader训练得到，效果优异；
- 覆盖有监督（如 FAQ 问答）、无监督（自动生成 QA 对，生成的问答对语料可以通过无监督的方式构建检索式问答系统）等多种情况，适用各类业务场景。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/168514868-1babe981-c675-4f89-9168-dd0a3eede315.gif" width="60%" length="60%">
</div>


详细使用说明请参考[智能问答系统](./question_answering)与[文档智能问答](./document_intelligence/doc_vqa)。

#### 📚 通用文本分类

- 基于“任务架构统一、通用能力共享”的通用文本分类技术 UTC，实了良好的零/少样本迁移能力，实现大一统诸多任务的开放域分类，可支持情感分析、意图识别、语义匹配、蕴含推理等各种可转换为分类问题的 NLU 任务。

<div align="center">
    <img src="https://user-images.githubusercontent.com/11793384/213347595-e9c08bd1-3d32-4519-9a52-31fb69b841e8.png" width="60%" length="60%">
</div>

<br>

详细使用说明请参考[通用文本分类](./zero_shot_text_classification)，更多：[文章解读](https://mp.weixin.qq.com/s/VV-nYv4y1r7oipJnURRL5w)。


#### 🗂 文本分类

- 场景方案全覆盖––––开源预训练模型-微调、提示学习、基于语义索引等多种分类技术方案，满足不同场景需求，涵盖多分类（multi-class）、多标签（multi-label）、层次分类（hierarchical）三类任务；
- 模型高效调优––––强强结合数据增强能力与可信增强技术，解决脏数据、标注数据欠缺、数据不平衡等问题，大幅提升模型效果。

<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/186378697-630d3590-4e67-49a0-8d5f-7cabd9daa894.png" width="60%" length="60%">
</div>

<br>

详细使用说明请参考[文本分类](./text_classification)，更多：[文章解读](https://mp.weixin.qq.com/s/tas7yM8vapxwtlJt-MRZdg)。

#### 💌 评论观点抽取与情感分析

- 经典方案：基于情感知识增强预训练模型SKEP，两阶段式抽取和分类，首先通过序列标注的方式定位属性词和观点词，然后进行属性集情感分类；
- 前沿方案：基于UIE的情感分析方案采用 Prompt Learning 的方式进行情感信息抽取，精度更高。支持语句级和属性级情感分析，解决同义属性聚合、隐性观点抽取难点，并提供可视化分析能力。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/200259473-434888f7-c0ac-4253-ab23-ede1628e6ba2.png" width="60%" length="60%">
</div>
<br>

详细使用说明请参考[情感分析](./sentiment_analysis)，更多：[文章解读](https://mp.weixin.qq.com/s/QAHjIRG9zxpYfM6YPRQ-9w)。

#### 🎙️ 智能语音指令解析

- 集成了[PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)和[百度开放平台](https://ai.baidu.com/)的语音识别和[UIE](./model_zoo/uie)通用信息抽取等技术，打造智能一体化的语音指令解析系统范例，该方案可应用于智能语音填单、智能语音交互、智能语音检索等场景，提高人机交互效率。

<div align="center">
    <img src="https://user-images.githubusercontent.com/16698950/168589100-a6c6f346-97bb-47b2-ac26-8d50e71fddc5.png" width="400">
</div>

详细使用说明请参考[智能语音指令解析](./applications/speech_cmd_analysis)。
