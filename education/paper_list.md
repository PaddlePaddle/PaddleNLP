## NLP 论文资料

以下是一些NLP常用的模型简介、论文及链接。

#### 1、词向量和语言模型

**NNLM**（2003）

论文：[A Neural Probabilistic Language Model](http://www.iro.umontreal.ca/~lisa/pointeurs/BengioDucharmeVincentJauvin_jmlr.pdf)

简介：本论文提出一种词的分布式表示来克服维数灾难，该方法允许每一个训练语句给模型提供关于语义相邻句子的指数级别数量的信息。NNLM能够使用n-gram建模更远的关系，并且考虑到了词语之间的相似性。

**Word2Vec**（2013）

论文：[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

简介： 本文提出了CBOW模型和Skip-Gram模型，用来学习word vector。并且设计了一种验证词向量效果的测试数据，从semantic和syntactic两个维度上进行测试。作者是来自Google的Tomas Mikolov，也是Word2Vec开源软件的作者。

代码链接：[https://github.com/danielfrg/word2vec](https://github.com/danielfrg/word2vec)

**GloVe**（2014）

论文：[GloVe: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162/)

简介：本文提出的GloVe是一个基于**全局词频统计**的词表征工具，它可以把一个单词表达成一个由实数组成的向量，这些向量捕捉到了单词之间一些语义特性，比如相似性（similarity）、类比性（analogy）等。向量之间的欧式距离可以反映出词之前的相似性。

代码链接：[https://github.com/maciejkula/glove-python](https://github.com/maciejkula/glove-python)

#### 2、预训练模型

**ELMo**（2018）

论文：[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)

简介：ELMo只预训练language model，而word embedding是通过输入的句子实时输出的， 这样可以得到与上下文相关的动态word embedding，很大程度上缓解了歧义的发生。

代码链接：[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/elmo](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/elmo)

**Bert**（2018）

论文：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

简介：BERT:  Bidirectional Encoder Representations from Transformers，是Google提出的NLP预训练方法，在大型文本语料库（如维基百科）上训练通用的语言理解模型，然后将该模型用于我们关心的下游NLP任务（如分类、阅读理解）。 BERT是第一个备受关注的无监督，深度双向系统，可以说指引了预训练模型的方向。

代码链接：[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert)

[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/bert](

**GPT**（2018）

论文：[Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)

简介：GPT是较早得使用Transformer模型进行预训练的模型。此前，大多数深度学习模型需要大量人工标注数据，而有限的标注资源则限制了模型在多个领域内的应用，因此急需从无标注的数据中学习语义信息。

https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/bert)

**GPT2**（2019）

论文：[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

简介：GPT2 与 GPT 的模型结构差别不大，但是采用了更大的数据集进行实验。模型在一个数百万级别的WebText的数据集上进行非监督训练。

代码链接：[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/gpt](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/gpt)

**ERNIE 2.0**（2019）

论文：[ERNIE 2.0: A Continual Pre-training Framework for Language Understanding](https://arxiv.org/abs/1907.12412)  

简介：预训练语言模型带来NLP领域的巨大飞跃，本文所提出的ERNIE 2.0除了能够捕获预训练语言模型中常见的句子或者词的共现之外，更重要的是能够捕获词汇、句法和语义信息。ERNIE 2.0的预训练是持续性地多任务的增量学习。本文的模型在16个自然语言处理任务上(包括GLUE benchmarks和多个中文任务）都超越BERT和XLNet。

**ERNIE**（2019）

论文：[ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129)

简介：本文提出的ERNIE 分为抽取知识信息与训练语言模型两大步骤，作者首先识别文本中的命名实体，然后将实体与知识图谱中的实体进行匹配。通过知识嵌入算法（例如 TransE）编码 KG 的图结构，并将多信息实体嵌入作为ERNIE的输入。ERNIE 将知识模块的实体表征整合到语义模块的隐藏层中。为了更好地融合文本和知识特征，作者设计了一种新型预训练目标，即随机 Mask 掉一些命名实体，并要求模型从知识图谱中选择合适的实体以完成对齐。

代码链接：[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/ernie](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/ernie-1.0)

**DistilBERT**（2019）

论文：[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)

简介：DistilBERT利用知识蒸馏的技术，在不过分降低性能的情况下，减少模型大小和推理速度。

代码链接：[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/distilbert](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/distilbert)

**ALBERT**（2019）

论文：[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)

简介：ALBERT为了解决模型参数量大以及训练时间过长的问题。ALBERT最小的参数只有十几兆, 效果只比BERT低1-2个点。模型有三个创新点：一是将embedding的参数进行了因式分解，二是跨层的参数共享，三是抛弃了原来的NSP任务，使用SOP任务。

代码链接：[https://github.com/google-research/ALBERT.](https://github.com/google-research/ALBERT)

PaddleNLP代码链接：[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/albert](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/albert)

**RoBERTa**（2019）

论文：[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

简介： RoBERTa 模型可以说是BERT 的改进版本，具有更大的模型参数量、更大bacth size、更多的训练数据。RoBERTa采用动态掩码，以及新的文本编码方式。RoBERTa建立在BERT的语言掩蔽策略的基础上，删除了BERT的"预测下一个句子”训练目标。

代码链接：[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/roberta][https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/roberta]

**XLM-RoBERTa**（2019）

论文：[Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)

简介：本文的XLM-R(XLM-RoBERTa)证明了使用大规模多语言预训练的模型可以显著提升跨语言迁移任务的性能。XLM-RoBERTa在技术上等同于XLM+RoBERTa，在数据上使用100种语言、2.5TB文本进行训练。XLM-RoBERTa在四个跨语言理解基准测试中取得了迄今为止最好的结果。

代码链接：[https://github.com/pytorch/fairseq](https://github.com/pytorch/fairseq)

**T5**（2019）

论文：[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

简介：迁移学习是一种在自然语言处理中强大的技术，首先要针对大数据进行预训练，然后再针对下游任务进行微调。谷歌发布的T5预训练模型在Glue、SuperGlue以及Squad任务上都超越了BERT，其最主要的贡献：一是提出一个通用框架，二是公开了C4数据集。

代码链接：[https://github.com/google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)

**XLNet**（2019）

论文：[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf)

简介：本文提出的XLNet具有建模双向上下文的能力，是一种广义自回归预处理训练方法，实验中，在问答，自然语言推理、情感分析和文档排序等多个任务中，XLNet的性能均优于BERT。

代码链接：[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/xlnet](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/xlnet)

**ERNIE-ViL**（2020）

论文：[ERNIE-ViL: Knowledge Enhanced Vision-Language Representations Through Scene Graph](https://arxiv.org/abs/2006.16934)

简介：百度在多模态语义理解领域取得突破，提出知识增强视觉-语言预训练模型 ERNIE-ViL，首次将场景图（Scene Graph）知识融入多模态预训练，在 5 项多模态任务上刷新世界最好效果，并在多模态领域权威榜单 VCR 上超越微软、谷歌、Facebook 等机构。此次突破充分借助飞桨深度学习平台分布式训练领先优势。

**ERNIE-UNIMO**（2020）

论文：[UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning](https://arxiv.org/abs/2012.15409)

简介：UNIMO可以有效地适应单模态和多模态理解和生成任务。利用大规模文本和图像语料库来提高视觉和文本理解以及跨模式对比学习（CMCL）能力，可以将文本和视觉信息对齐到统一的语义空间中，适用于文本-图像多模态任务。

**ERNIE-DOC**（2020）

论文：[ERNIE-DOC: The Retrospective Long-Document Modeling Transformer](https://arxiv.org/abs/2012.15688)

简介：本文提出了一个预训练语言模型 ERNIE-DOC，基于Recurrence Transformers的文档级语言预训练模型。本模型采用了两种机制：回溯式feed机制和增强的循环机制，使模型获取更长的上下文长度信息。

**ERNIE-Gram**（2020）

论文：[ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding](https://arxiv.org/abs/2010.12148)

简介：ERNIE-Gram使用一个生成器模型来采样n-gram标识，直接使用n-gram标识来进行屏蔽和预测，在多个任务中取得SOTA效果。

代码链接：[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie_gram](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie_gram)
