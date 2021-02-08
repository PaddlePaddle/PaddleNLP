# PaddleMRC
PaddleMRC，全称为Paddle Machine Reading Comprehension，集合了百度在阅读理解领域相关的模型，工具，开源数据等一系列工作。包括DuReader (百度开源的基于真实搜索用户行为的中文大规模阅读理解数据集)，KT-Net (结合知识的阅读理解模型，SQuAD以及ReCoRD曾排名第一), D-Net (预训练-微调框架，在EMNLP2019 MRQA国际阅读理解评测获得第一)，等。

## 机器阅读理解任务简介
机器阅读理解 (Machine Reading Comprehension) 是指让机器阅读文本，然后回答和阅读内容相关的问题。其技术可以使计算机具备从文本数据中获取知识并回答问题的能力，是构建通用人工智能的关键技术之一。简单来说，就是根据给定材料和问题，让机器给出正确答案。阅读理解是自然语言处理和人工智能领域的重要前沿课题，对于提升机器智能水平、使机器具有持续知识获取能力具有重要价值，近年来受到学术界和工业界的广泛关注。

## [DuReader](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2018-DuReader)

### DuReader数据集
DuReader是一个大规模、面向真实应用、由人类生成的中文阅读理解数据集。DuReader聚焦于真实世界中的不限定领域的问答任务。相较于其他阅读理解数据集，DuReader的优势包括:

 - 问题来自于真实的搜索日志
 - 文章内容来自于真实网页
 - 答案由人类生成
 - 面向真实应用场景
 - 标注更加丰富细致

更多关于DuReader数据集的详细信息可在[DuReader官网](https://ai.baidu.com//broad/subordinate?dataset=dureader)找到。

### DuReader基线系统

DuReader基线系统利用[PaddlePaddle](http://paddlepaddle.org)深度学习框架，针对**DuReader阅读理解数据集**实现并升级了一个经典的阅读理解模型 —— BiDAF.


## [KT-Net](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2019-KTNET)
KT-NET是百度NLP提出的具有开创性意义的语言表示与知识表示的深度融合模型。该模型同时借助语言和知识的力量进一步提升了机器阅读理解的效果，获得的成果包括

 - ReCoRD榜单排名第一（截至2019.5.14）
 - SQuAD1.1榜单上单模型排名第一（截至2019.5.14）
 - 被ACL 2019录用为长文 ([文章链接](https://www.aclweb.org/anthology/P19-1226/))

此外，KT-NET具备很强的通用性，不仅适用于机器阅读理解任务，对其他形式的语言理解任务，如自然语言推断、复述识别、语义相似度判断等均有帮助。


## [D-NET](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/MRQA2019-D-NET)
D-NET是一个以提升**阅读理解模型泛化能力**为目标的“预训练-微调”框架。D-NET的特点包括：

- 利用多个预训练模型 (ERNIE2.0, XL-NET, BERT)，增强模型的语义表示能力
- 在微调阶段引入多任务、多领域的学习策略 (基于[PALM](https://github.com/PaddlePaddle/PALM)多任务学习框架)，有效的提升了模型在不同领域的泛化能力

百度利用D-NET框架在EMNLP 2019 [MRQA](https://mrqa.github.io/shared)国际阅读理解评测中以超过第二名近两个百分点的成绩夺得冠军，同时，在全部12个测试数据集中的10个排名第一。
