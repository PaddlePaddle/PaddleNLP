# 文本分类应用

**目录**
- [1. 文本分类应用简介](#文本分类应用简介)
- [2. 技术特色介绍](#技术特色介绍)
  - [2.1 文本分类方案全覆盖](#文本分类方案全覆盖)
  - [2.2 更懂中文的训练基座](#更懂中文的训练基座)
  - [2.3 高效模型调优方案](#高效模型调优方案)
  - [2.4 产业级全流程方案](#产业级全流程方案)
- [3. 快速开始](#快速开始)
- [4. 常用中文分类数据集](#常用中文分类数据集)

<a name="文本分类应用简介"></a>

## 1. 文本分类应用简介
文本分类应用针对**多分类、多标签、层次分类等高频场景开源了产业级分类应用方案**，打通数据标注-模型训练-模型调优-模型压缩-预测部署全流程，旨在解决细分场景应用的痛点和难点，快速实现文本分类产品落地。

文本分类简单来说就是对给定的一个句子或一段文本使用分类模型分类。虽然文本分类在金融、医疗、法律、工业等领域都有广泛的成功实践应用，但如何选择合适的方案和预训练模型、数据标注质量差、效果调优困难、AI 入门成本高、如何高效训练部署等问题使部分开发者望而却步。针对文本分类领域的痛点和难点，PaddleNLP 文本分类应用提出了多种前沿解决方案，助力开发者简单高效实现文本分类数据标注、训练、调优、上线，降低文本分类落地技术门槛。

<div align="center">
    <img width="700" alt="文本分类落地难点" src="https://user-images.githubusercontent.com/63761690/189114119-4a1b0bd5-a604-4a34-a63b-7b27519eaf09.png">
</div>

**文本分类应用技术特色：**

- **方案全面🎓：** 涵盖多分类、多标签、层次分类等高频分类场景，提供预训练模型微调、提示学习（小样本学习）、语义索引三种端到端全流程分类方案，满足开发者多样文本分类落地需求。
- **效果领先🏃：** 使用在中文领域内模型效果和模型计算效率有突出效果的 ERNIE 3.0 轻量级系列模型作为训练基座，ERNIE 3.0 轻量级系列提供多种尺寸的预训练模型满足不同需求，具有广泛成熟的实践应用性。
- **高效调优✊：** 文本分类应用依托[TrustAI](https://github.com/PaddlePaddle/TrustAI)可信增强能力和[数据增强 API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)，提供模型分析模块助力开发者实现模型分析，并提供稀疏数据筛选、脏数据清洗、数据增强等多种解决方案。
- **简单易用👶：** 开发者**无需机器学习背景知识**，仅需提供指定格式的标注分类数据，一行命令即可开启文本分类训练，轻松完成上线部署，不再让技术成为文本分类的门槛。

<a name="技术特色介绍"></a>

## 2. 技术特色介绍

<a name="文本分类方案全覆盖"></a>

### 2.1 文本分类方案全覆盖

<div align="center">
    <img width="900" alt="image" src="https://user-images.githubusercontent.com/63761690/189114232-bb706af4-45a9-4e63-8857-76945a63d081.png">
</div>

#### 2.1.1 分类场景齐全

文本分类应用涵盖多分类（multi class）、多标签（multi label）、层次分类（hierarchical）三种场景，接下来我们将以下图的新闻文本分类为例介绍三种分类场景的区别。

<div align="center">
    <img width="900" alt="image" src=https://user-images.githubusercontent.com/63761690/186378697-630d3590-4e67-49a0-8d5f-7cabd9daa894.png />
</div>

- **多分类🚶：** 数据集的标签集含有两个或两个以上的类别，所有输入句子/文本有且只有一个标签。在文本多分类场景中，我们需要预测输入句子/文本最可能来自 `n` 个标签类别中的哪一个类别。以上图多分类中新闻文本为例，该新闻文本的标签为 `娱乐`。快速开启多分类任务参见  👉 [多分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_class#readme)

- **多标签👫 ：** 数据集的标签集含有两个或两个以上的类别，输入句子/文本具有一个或多个标签。在文本多标签任务中，我们需要预测输入句子/文本可能来自 `n` 个标签类别中的哪几个类别。以上图多标签中新闻文本为例，该新闻文本具有 `相机` 和 `芯片` 两个标签。快速开启多标签任务参见  👉 [多标签指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_label#readme) 。

- **层次分类👪 ：** 数据集的标签集具有多级标签且标签之间具有层级结构关系，输入句子/文本具有一个或多个标签。在文本层次分类任务中，我们需要预测输入句子/文本可能来自于不同级标签类别中的某一个或几个类别。以上图层次分类中新闻文本为例（新闻为根节点），该新闻一级分类标签为 `体育`，二级分类标签为 `足球`。快速开启层次分类任务参见 👉 [层次分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/hierarchical#readme) 。


#### 2.1.2 多方案满足定制需求

#### 方案一：预训练模型微调

【方案选择】对于大多数任务，我们推荐使用**预训练模型微调作为首选的文本分类方案**，预训练模型微调提供了数据标注-模型训练-模型分析-模型压缩-预测部署全流程，有效减少开发时间，低成本迁移至实际应用场景。

【方案介绍】ERNIE 3.0 轻量级模型不能直接在文本分类任务上使用，预训练模型微调在预训练模型 `[CLS]` 输出向量后接入线性层作为文本分类器，用具体任务数据进行微调训练文本分类器，使预训练模型”更懂”这个任务。

【方案效果】下表展示在多标签任务 CAIL2019—婚姻家庭要素提取数据集中 ERNIE 3.0 系列轻量级模型效果评测。


<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/189115968-c1d14ed3-dbdd-4a84-ac11-e9eaa447d40f.png width=800 height=300 />
</div>


【快速开始】
- 快速开启多分类任务参见 👉 [预训练模型微调-多分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_class#readme)
- 快速开启多标签分类任务参见 👉 [预训练模型微调-多标签分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_label#readme)
- 快速开启层次分类任务参见 👉 [预训练模型微调-层次分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/hierarchical#readme)

#### 方案二：提示学习

【方案选择】提示学习（Prompt Learning）适用于**标注成本高、标注样本较少的文本分类场景**。在小样本场景中，相比于预训练模型微调学习，提示学习能取得更好的效果。对于标注样本充足、标注成本较低的场景，我们仍旧推荐使用充足的标注样本进行文本分类[预训练模型微调](#预训练模型微调)。

【方案介绍】**提示学习的主要思想是将文本分类任务转换为构造提示中掩码 `[MASK]` 的分类预测任务**，也即在掩码 `[MASK]`向量后接入线性层分类器预测掩码位置可能的字或词。提示学习使用待预测字的预训练向量来初始化分类器参数（如果待预测的是词，则为词中所有字的预训练向量平均值），充分利用预训练语言模型学习到的特征和标签文本，从而降低样本需求。提示学习同时提供[ R-Drop](https://arxiv.org/abs/2106.14448) 和 [RGL](https://aclanthology.org/2022.findings-naacl.81/) 策略，帮助提升模型效果。

我们以下图情感二分类任务为例来具体介绍提示学习流程，分类任务标签分为 `0:负向` 和 `1:正向` 。在文本加入构造提示 `我[MASK]喜欢。` ，将情感分类任务转化为预测掩码 `[MASK]` 的待预测字是 `不` 还是 `很`。具体实现方法是在掩码`[MASK]`的输出向量后接入线性分类器（二分类），然后用`不`和`很`的预训练向量来初始化分类器进行训练，分类器预测分类为 `0：不` 或 `1：很` 对应原始标签 `0:负向` 或 `1:正向`。而预训练模型微调则是在预训练模型`[CLS]`向量接入随机初始化线性分类器进行训练，分类器直接预测分类为 `0:负向` 或 `1:正向`。

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/189114324-376025b6-8f4e-4d94-a135-953f53f20636.png width=800 height=300 />
</div>

【方案效果】我们比较预训练模型微调与提示学习在多分类、多标签、层次分类小样本场景的模型表现（多分类精度为准确率，多标签和层次分类精度为 Macro F1值），可以看到在样本较少的情况下，提示学习比预训练模型微调有明显优势。

<div align="center">
    <img width="600" alt="文本分类落地难点" src="https://user-images.githubusercontent.com/63761690/189114445-ee0dd6af-f102-4708-9f46-c630c572dfd3.png">
</div>



【快速开始】

更多测评和使用细节详见各场景文档：
- 快速开启多分类任务参见 👉 [提示学习(小样本)-多分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_class/few-shot#readme)
- 快速开启多标签分类任务参见 👉 [提示学习(小样本)-多标签分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_label/few-shot#readme)
- 快速开启层次分类任务参见 👉 [提示学习(小样本)-层次分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/hierarchical/few-shot#readme)

#### 方案三：语义索引

【方案选择】基于语义索引的文本分类方案**适用于标签类别不固定的场景**，对于新增标签类别或新的相关分类任务无需重新训练，模型仍然能获得较好预测效果，方案具有良好的拓展性。

【方案介绍】语义索引目标是从海量候选召回集中快速、准确地召回一批与输入文本语义相关的文本。基于语义索引的文本分类方法具体来说是将标签集作为召回目标集，召回与输入文本语义相似的标签作为文本的标签类别。

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/189114541-2278b7f7-1af6-470d-a300-28e7e902b6a8.png width=800 height=300 />
</div>

【快速开始】
- 快速开启多分类任务参见 👉 [语义索引-多分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_class/retrieval_based#readme)
- 快速开启多标签分类任务参见 👉 [语义索引-多标签分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_label/retrieval_based#readme)
- 快速开启层次分类任务参见 👉 [语义索引-层次分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/hierarchical/retrieval_based#readme)



<a name="更懂中文的训练基座"></a>

### 2.2 更懂中文的训练基座

近年来，大量的研究表明在超大规模的语料采用无监督或者弱监督的方式训练模型，模型能够获得语言相关的知识。预训练模型学习到的文本语义表示能够避免从零开始训练模型，同时有利于下游自然语言处理(NLP)任务。预训练模型与具体的文本分类任务的关系可以直观地理解为，**预训练模型已经懂得了相关句法、语义的语言知识，用具体任务数据训练使得预训练模型”更懂”这个任务**，在预训练过程中学到的知识基础使学习文本分类任务事半功倍。

文本分类应用使用**ERNIE 3.0轻量级模型作为预训练模型**，ERNIE 3.0 轻量级模型是文心大模型 ERNIE 3.0基础上通过在线蒸馏技术得到的轻量级模型。下面是 ERNIE 3.0 效果-时延图，ERNIE 3.0 轻量级模型在精度和性能上的综合表现已全面领先于 UER-py、Huawei-Noah 以及 HFL 的中文模型，具体的测评细节可以见[ERNIE 3.0 效果和性能测评文档](../../model_zoo/ernie-3.0)。

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/186376051-6c3ca239-1e31-4c0a-bdbe-547439234ddb.png width="600"/>
</div>



<a name="高效模型调优方案"></a>

### 2.3 高效模型调优方案

有这么一句话在业界广泛流传，"数据决定了机器学习的上限，而模型和算法只是逼近这个上限"，可见数据质量的重要性。文本分类应用依托[TrustAI](https://github.com/PaddlePaddle/TrustAI)可信增强能力和[数据增强 API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)开源了模型分析模块，针对标注数据质量不高、训练数据覆盖不足、样本数量少等文本分类常见数据痛点，提供稀疏数据筛选、脏数据清洗、数据增强三种数据优化方案，解决训练数据缺陷问题，用低成本方式获得大幅度的效果提升。


- **稀疏数据筛选**基于特征相似度的实例级证据分析方法挖掘待预测数据中缺乏证据支持的数据（也即稀疏数据），并进行有选择的训练集数据增强或针对性筛选未标注数据进行标注来解决稀疏数据问题，有效提升模型表现。
<div align="center">
    <img width="1000" alt="文本分类落地难点" src="https://user-images.githubusercontent.com/63761690/189114644-c0d21801-dd6c-4530-b3a3-5f5a568a7a22.png">
</div>

我们采用在多分类、多标签、层次分类场景中评测稀疏数据-数据增强策略和稀疏数据-数据标注策略，下图表明稀疏数据筛选方案在各场景能够有效提高模型表现（多分类精度为准确率，多标签和层次分类精度为 Macro F1值）。

<div align="center">
    <img width="600" alt="文本分类落地难点" src="https://user-images.githubusercontent.com/63761690/189114660-820d1471-4907-4c73-a118-c494200afff0.png">
</div>


- **脏数据清洗**基于表示点方法的实例级证据分析方法，计算训练数据对模型的影响分数，分数高的训练数据表明对模型影响大，这些数据有较大概率为脏数据（标注错误样本）。脏数据清洗方案通过高效识别训练集中脏数据（也即标注质量差的数据），有效降低人力检查成本。

<div align="center">
    <img width="1000" alt="文本分类落地难点" src="https://user-images.githubusercontent.com/63761690/189114677-9a2f5232-9551-4e10-a215-54ddb7ca1f33.png">
</div>

我们采用在多分类、多标签、层次分类场景中评测脏数据清洗方案，实验表明方案能够高效筛选出训练集中脏数据，提高模型表现（多分类精度为准确率，多标签和层次分类精度为 Macro F1值）。

<div align="center">
    <img width="600" alt="文本分类落地难点" src="https://user-images.githubusercontent.com/63761690/189114695-90cef2fd-955d-4243-9fc4-78f2dcf7fb6c.png">
</div>


- **数据增强**在数据量较少的情况下能够通过增加数据集多样性，提升模型效果。PaddleNLP 内置[数据增强 API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)，支持词替换、词删除、词插入、词置换、基于上下文生成词（MLM 预测）、TF-IDF 等多种数据增强策略。数据增强方案提供一行命令，快速完成数据集增强。以 CAIL2019—婚姻家庭要素提取数据子集（500条）为例，我们在数据集应用多种数据增强策略，策略效果如下表。

<div align="center">
    <img width="600" alt="文本分类落地难点" src="https://user-images.githubusercontent.com/63761690/189115071-40152f7c-7c90-41b3-a70b-4d4aabc5b715.png">
</div>


【快速开始】

更多使用方法和测评细节详见各场景模型分析模块：

- 体验模型分析模块 👉 [多分类-模型分析模块](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_class/analysis)
- 体验模型分析模块 👉 [多标签-模型分析模块](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_label/analysis)
- 体验模型分析模块 👉 [层次分类-模型分析模块](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/hierarchical/analysis)

<a name="产业级全流程方案"></a>

### 2.4 产业级全流程方案

文本分类应用提供了简单易用的数据标注-模型训练-模型调优-模型压缩-预测部署全流程方案，我们将以预训练模型微调方案为例介绍文本分类应用的全流程：

<div align="center">
    <img width="900" alt="image" src="https://user-images.githubusercontent.com/63761690/189115101-20cbaa00-e549-425b-b047-61bac2a5e39f.png">
</div>
<div align="center">
    <font size ="2">
    文本分类应用全流程示意图
     </font>
</div>


**1.数据准备阶段**

- 我们根据文本分类任务选择对应的场景目录: [多分类场景目录](./multi_class)、
 [多标签场景目录](./multi_label)、[层次分类场景目录](./hierarchical)。

- 如果没有已标注的数据集，我们推荐 doccano 数据标注工具进行标注，详见[文本分类标注指南](./doccano.md)。如果已有标注好的本地数据集，我们需要根据不同任务要求将数据集整理为文档要求的格式，详见各分类场景文档。

**2.模型训练**

- 数据准备完成后，开始进行预训练模型微调训练。可以根据实际数据调整可配置参数，选择使用 GPU 或 CPU 进行模型训练，脚本默认保存在开发集最佳表现模型参数。

- 训练结束后，使用模型分析(analysis)模块对分析模型表现，同时模型分析(analysis)模块依托[TrustAI](https://github.com/PaddlePaddle/TrustAI)可信增强能力和[数据增强 API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/dataaug.md)提供稀疏数据筛选、脏数据清洗、数据增强三种优化方案帮助提升模型效果。

- 模型训练、调优完成后，可以通过预测脚本加载最佳模型参数，打印模型预测结果。

**3.模型部署**

- 现实部署场景需要同时考虑模型的精度和性能表现，文本分类应用接入 PaddleNLP 模型压缩 API 。采用了 DynaBERT 中宽度自适应裁剪策略，对预训练模型多头注意力机制中的头（Head ）进行重要性排序，保证更重要的头（Head ）不容易被裁掉，然后用原模型作为蒸馏过程中的教师模型，宽度更小的模型作为学生模型，蒸馏得到的学生模型就是我们裁剪得到的模型。实验表明模型裁剪能够有效缩小模型体积、减少内存占用、提升推理速度。模型裁剪去掉了部分冗余参数的扰动，增加了模型的泛化能力，在部分任务中预测精度得到提高。

<div align="center">
    <img width="900" alt="image" src="https://user-images.githubusercontent.com/63761690/189115124-2f429043-3145-4bf8-9969-47580a706037.png">
</div>

- 模型部署需要将保存的最佳模型参数（动态图参数）导出成静态图参数，用于后续的推理部署。p.s.模型裁剪之后会默认导出静态图模型

- 文本分类应用提供了离线部署，并且支持在 GPU 设备使用 FP16，在 CPU 设备使用动态量化的低精度加速推理；同时提供基于 Paddle Serving 的在线服务化部署，详见各分类场景文档中模型部署介绍。


<a name="快速开始"></a>

## 3. 快速开始

- 快速开启多分类 👉 [多分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_class#readme)

- 快速开启多标签分类 👉 [多标签指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_label#readme)

- 快速开启层次分类 👉 [层次分类指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/hierarchical#readme)

<a name="常用中文分类数据集"></a>

## 4. 常用中文分类数据集

**多分类数据集：**

- [THUCNews 新闻分类数据集](http://thuctc.thunlp.org/)

- [百科问答分类数据集](https://github.com/brightmart/nlp_chinese_corpus#3%E7%99%BE%E7%A7%91%E7%B1%BB%E9%97%AE%E7%AD%94json%E7%89%88baike2018qa)

- [头条新闻标题数据集 TNEWS](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)

- [复旦新闻文本数据集](https://www.heywhale.com/mw/dataset/5d3a9c86cf76a600360edd04)

- [IFLYTEK app 应用描述分类数据集](https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip)

- [CAIL 2022事件检测](http://cail.cipsc.org.cn/task1.html?raceID=1&cail_tag=2022)

**情感分类数据集(多分类):**

- [亚马逊商品评论情感数据集](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/yf_amazon/intro.ipynb)

- [财经新闻情感分类数据集](https://github.com/wwwxmu/Dataset-of-financial-news-sentiment-classification)

- [ChnSentiCorp 酒店评论情感分类数据集](https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/ChnSentiCorp_htl_all)

- [外卖评论情感分类数据集](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb)

- [weibo 情感二分类数据集](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb)

- [weibo 情感四分类数据集](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb)

- [商品评论情感分类数据集](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb)

- [电影评论情感分类数据集](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/dmsc_v2/intro.ipynb)

- [大众点评分类数据集](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/yf_dianping/intro.ipynb)

**多标签数据集:**

- [学生评语分类数据集](https://github.com/FBI1314/textClassification/tree/master/multilabel_text_classfication/data)

- [CAIL2019婚姻要素识别](https://aistudio.baidu.com/aistudio/projectdetail/3996601)

- [CAIL2018 刑期预测、法条预测、罪名预测](https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip)

**层次分类数据集:**

- [头条新闻标题分类-TNEWS 的升级版](https://github.com/aceimnorstuvwxz/toutiao-multilevel-text-classfication-dataset)

- [网页层次分类数据集](https://csri.scu.edu.cn/info/1012/2827.htm)

- [医学意图数据集(CMID)](https://github.com/liutongyang/CMID)

- [2020语言与智能技术竞赛事件分类](https://github.com/percent4/keras_bert_multi_label_cls/tree/master/data)
