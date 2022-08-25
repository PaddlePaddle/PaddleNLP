# 文本分类应用

**目录**
   * [文本分类应用简介](#文本分类应用简介)
   * [文本分类场景](#文本分类场景)
   * [文本分类技术方案](#文本分类技术方案)
   * [文本分类应用全流程方案](#文本分类应用全流程方案)
   * [文本分类快速开始](#文本分类快速开始)

## 文本分类应用简介
文本分类应用针对**多分类、多标签、层次分类等高频分类场景提供多种端到端应用方案，打通数据标注-模型训练-模型调优-模型压缩-预测部署全流程**，满足开发者多种分类需求，有效缩短开发周期，降低文本分类落地技术门槛。

文本分类简单来说就是对给定的一个句子或一段文本使用分类模型分类，文本分类任务广泛应用于长短文本分类、情感分析、新闻分类、事件类别分类、政务数据分类、商品信息分类、商品类目预测、文章分类、论文类别分类、专利分类、案件描述分类、罪名分类、意图分类、论文专利分类、邮件自动标签、评论正负识别、药物反应分类、对话分类、税种识别、来电信息自动分类、投诉分类、广告检测、敏感违法内容检测、内容安全检测、舆情分析、话题标记等各类日常或专业领域中。

虽然文本分类在各个领域中有广泛的成功实践应用，但学习开发成本高、调优困难、落地门槛高等问题使部分开发者望而却步。PaddleNLP文本分类应用旨在**助力开发者简单高效实现文本分类模型训练、调优、上线，拒绝让技术成为AI开发应用的门槛！**

**PaddleNLP文本分类应用亮点**

- **方案全面🎓：** 涵盖多分类、多标签、层次分类等高频分类场景，提供预训练模型微调、提示学习（小样本学习）、语义索引三种端到端全流程分类方案，满足开发者多种分类落地需求。
- **分析高效✊：** 文本分类应用依托[TrustAI](https://github.com/PaddlePaddle/TrustAI)可信增强能力和[数据增强API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)，提供模型分析模块助力开发者实现模型分析，并提供稀疏数据筛选、脏数据清洗、数据增强等多种解决方案。
- **效果领先🏃：** 使用在中文领域内模型效果和模型计算效率有突出效果的ERNIE 3.0轻量级模型作为训练基座，ERNIE 3.0轻量级模型学习海量的中文数据与知识，具有广泛成熟的实践应用。
- **低门槛操作👶：** 开发者**无需机器学习背景知识**，仅需提供指定格式的标注分类数据，一行命令即可开启文本分类训练。



## 文本分类场景

文本分类场景可以根据标签类型分为多分类（multi class）、多标签（multi label）、层次分类（hierarchical）等三种场景，接下来我们将以下图的新闻文本分类为例介绍三种分类场景的区别。

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/186378697-630d3590-4e67-49a0-8d5f-7cabd9daa894.png />
</div>

- **多分类🚶：** 数据集的标签集含有两个或两个以上的类别，所有输入句子/文本有且只有一个标签。在文本多分类场景中，我们需要预测输入句子/文本最可能来自 `n` 个标签类别中的哪一个类别。以上图多分类中新闻文本为例，该新闻文本的标签为 `娱乐`。快速开启多分类任务参见  👉 [多分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_class#readme)

- **多标签👫 ：** 数据集的标签集含有两个或两个以上的类别，输入句子/文本具有一个或多个标签。在文本多标签任务中，我们需要预测输入句子/文本可能来自 `n` 个标签类别中的哪几个类别。以上图多标签中新闻文本为例，该新闻文本具有 `相机` 和 `芯片` 两个标签。快速开启多标签任务参见  👉 [多标签指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_label#readme) 。

- **层次分类👪 ：** 数据集的标签集具有多级标签且标签之间具有层级结构关系，输入句子/文本具有一个或多个标签。在文本层次分类任务中，我们需要预测输入句子/文本可能来自于不同级标签类别中的某一个或几个类别。以上图层次分类中新闻文本为例（新闻为根节点），该新闻一级分类标签为 `体育`，二级分类标签为 `足球`。快速开启层次分类任务参见 👉 [层次分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/hierarchical#readme) 。

## 文本分类技术方案
### 模型基座：预训练模型

近年来，大量的研究表明在超大规模的语料采用无监督或者弱监督的方式训练模型，模型能够获得语言相关的知识。预训练模型学习到的文本语义表示能够避免从零开始训练模型，同时有利于下游自然语言处理(NLP)任务。预训练模型与具体的文本分类任务的关系可以直观地理解为，**预训练模型已经懂得了相关句法、语义的语言知识，用具体任务数据训练使得预训练模型”更懂”这个任务**，在预训练过程中学到的知识基础使学习文本分类任务事半功倍。

文本分类应用使用**ERNIE 3.0轻量级模型作为预训练模型**，ERNIE 3.0 轻量级模型是文心大模型ERNIE 3.0基础上通过在线蒸馏技术得到的轻量级模型。下面是ERNIE 3.0 效果-时延图，可以看到ERNIE 3.0 轻量级模型在精度和性能上的综合表现已全面领先于 UER-py、Huawei-Noah 以及 HFL 的中文模型，具体的测评细节可以见[ERNIE 3.0 效果和性能测评文档](../../model_zoo/ernie-3.0)。

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/186376051-6c3ca239-1e31-4c0a-bdbe-547439234ddb.png width="600"/>
</div>

### 方案一：预训练模型微调

【方案选择】对于大多数任务，我们推荐使用**预训练模型微调作为首选的文本分类方案**，预训练模型微调提供了数据标注-模型训练-模型分析-模型压缩-预测部署全流程，有效减少开发时间，低成本迁移至实际应用场景。

【方案介绍】预训练模型不能直接在文本分类任务上使用，预训练模型微调在预训练模型 `[CLS]` 输出向量后接入线性层分类器，用具体任务数据进行微调训练文本分类器，使预训练模型”更懂”这个任务。

[PaddleNLP预训练模型](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer)包含了 `ERNIE`、`BERT`、`RoBERTa`等40多个主流预训练模型，500多个模型权重，只需一行代码即可加载用于文本分类微调的多种预训练模型。下面以ERNIE 3.0 中文base模型为例，演示如何加载预训练模型：

```shell
from paddlenlp.transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained( "ernie-3.0-base-zh", num_classes=10)
```

【快速开始】
- 快速开启多分类任务参见 👉 [预训练模型微调-多分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_class#readme)
- 快速开启多标签分类任务参见 👉 [预训练模型微调-多标签分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_label#readme)
- 快速开启层次分类任务参见 👉 [预训练模型微调-层次分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/hierarchical#readme)

### 方案二：提示学习

【方案选择】提示学习（Prompt Learning）适用于**标注成本高、标注样本较少的小样本场景**文本分类任务。在小样本场景中，相比于预训练模型微调学习，提示学习能取得更好的效果。对于标注样本充足、标注成本较低的场景，我们仍旧推荐使用充足的标注样本进行文本分类[预训练模型微调](#预训练模型微调)。

【方案介绍】**提示学习的主要思想是将文本分类任务转换为构造提示中掩码 `[MASK]` 的分类预测任务**，也即在掩码 `[MASK]`向量后接入线性层分类器预测掩码位置可能的字或词。提示学习使用待预测字的预训练向量来初始化分类器参数（如果待预测的是词，则为词中所有字的预训练向量平均值），充分利用预训练语言模型学习到的特征和标签文本，从而降低样本需求。提示学习同时提供[R-Drop](https://arxiv.org/abs/2106.14448) 和[RGL](https://aclanthology.org/2022.findings-naacl.81/) 策略，帮助提示模型效果。

我们以下图情感二分类任务为例来具体介绍提示学习流程，分类任务标签分为 `0:负向` 和 `1:正向` 。在文本加入构造提示 `我[MASK]喜欢。` ，将情感分类任务转化为预测掩码 `[MASK]` 的待预测字是 `不` 还是 `很`。具体实现方法是在掩码`[MASK]`的输出向量后接入线性分类器（二分类），然后用`不`和`很`的预训练向量来初始化分类器进行训练，分类器预测分类为 `0：不` 或 `1：很` 对应原始标签 `0:负向` 或 `1:正向`。而预训练模型微调则是在预训练模型`[CLS]`向量接入随机初始化线性分类器进行训练，分类器直接预测分类为 `0:负向` 或 `1:正向`。

<div align="center">
    <img src=https://user-images.githubusercontent.com/25607475/183909263-6ead8871-699c-4c2d-951f-e33eddcfdd9c.png width=800 height=300 />
</div>

【快速开始】
- 快速开启多分类任务参见 👉 [提示学习(小样本)-多分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_class/few-shot#readme)
- 快速开启多标签分类任务参见 👉 [提示学习(小样本)-多标签分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_lable/few-shot#readme)
- 快速开启层次分类任务参见 👉 [提示学习(小样本)-层次分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/hierarchical/few-shot#readme)

### 方案三：语义索引

【方案选择】基于语义索引的文本分类方案**适用于标签类别不固定的场景**，对于新增标签类别或新的相关分类任务无需重新训练，模型仍然能获得较好预测效果，方案具有良好的拓展性。

【方案介绍】语义索引目标是从海量候选召回集中快速、准确地召回一批与输入文本语义相关的文本。基于语义索引的文本分类方法具体来说是将标签集作为召回目标集，召回与输入文本语义相似的标签作为文本的标签类别。

【快速开始】
- 快速开启多分类任务参见 👉 [语义索引-多分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_class/retrieval_based#readme)
- 快速开启多标签分类任务参见 👉 [语义索引-多标签分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_lable/retrieval_based#readme)
- 快速开启层次分类任务参见 👉 [语义索引-层次分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/hierarchical/retrieval_based#readme)

## 文本分类应用全流程方案

<div align="center">
    <img width="1238" alt="image" src="https://user-images.githubusercontent.com/63761690/186386181-7cdf3015-3e6c-4ffe-9512-95ba707fd00c.png">
</div>


如上图所示，我们将预训练模型微调为例介绍文本分类应用的全流程：

**1.数据准备阶段**

- 我们根据文本分类任务选择对应的场景目录: [多分类场景目录](./multi_class)、
 [多标签场景目录](./multi_label)、[层次分类场景目录](./hierarchical)。

- 如果没有已标注的数据集，我们推荐doccano数据标注工具进行标注，详见[文本分类标注指南](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/text_classification/doccano.md)。如果已有标注好的本地数据集，我们需要根据不同任务要求将数据集整理为文档要求的格式，详见各分类场景文档。

**2.模型训练**

- 数据准备完成后，开始进行预训练模型微调训练。可以根据实际数据调整可配置参数，选择使用GPU或CPU进行模型训练，脚本默认保存在开发集最佳表现模型参数。

- 训练结束后，使用模型分析(analysis)模块对分析模型表现，同时模型分析(analysis)模块依托[TrustAI](https://github.com/PaddlePaddle/TrustAI)可信增强能力和[数据增强API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)提供稀疏数据筛选、脏数据清洗、数据增强三种优化方案帮助提升模型效果。

- 模型训练、调优完成后，可以通过预测脚本加载最佳模型参数，打印模型预测结果。

**3.模型部署**

- 现实部署场景需要同时考虑模型的精度和性能表现。基于压缩API的模型裁剪能够进一步压缩模型体积，此外模型裁剪去掉了部分冗余参数的扰动，增加了模型的泛化能力，在部分任务预测精度得到提高。

- 模型部署需要将保存的最佳模型参数（动态图参数）导出成静态图参数，用于后续的推理部署。p.s.模型裁剪之后会默认导出静态图模型

- 文本分类应用提供了离线部署，并且支持在GPU设备使用FP16，在CPU设备使用动态量化的低精度加速推理；同时提供基于Paddle Serving的在线服务化部署，详见各分类场景文档中模型部署介绍。

## 文本分类快速开始

- 快速开启多分类 👉 [多分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_class#readme)

- 快速开启多标签分类 👉 [多标签指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_label#readme)

- 快速开启层次分类 👉 [层次分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/hierarchical#readme)
