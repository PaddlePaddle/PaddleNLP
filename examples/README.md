# PaddleNLP 范例

在本目录（https://github.com/PaddlePaddle/PaddleNLP ）下，包含了一些范例。涵盖了大多数常见NLP任务，是入门NLP和PaddleNLP的学习资料，也可以作为工作中上手NLP的基线参考实现。


## PaddleNLP的例子清单

| **目录**  | **主题**                                           | 简要说明                                                      |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| benchmark/glue   | GLUE Benchmark                                    | GLUE是当今使用最为普遍的自然语言理解评测基准数据集，评测数据涵盖新闻、电影、百科等许多领域，其中有简单的句子，也有困难的句子。本项目是 GLUE评测任务 在 Paddle 2.0上的开源实现                       |
| dependency_parsing/ddparser       | 句法分析     | 提供了一个基于ddparser的句法分析任务实现示例。      |
| dialogue | 该目录下涵盖了多个对话系统相关的例子| 如dgu 对话通用理解模型, LIC 2021对话比赛基线, PLATO-2 开放域机器人 UnifiedTransformer 适合对话生成任务的Transfer网络。|
| few_shot       | 小样本学习                                           | 提供简单易用、全面、前沿的 FSL 策略库，如P-tuning，EFL等。           |
| information_extraction    | 信息抽取 | 提供了多个数据集上的信息抽取基线实现。包含快递单信息抽取， MSRA-NER 数据集命名实体识别，LIC2021 DuIE 关系抽取基线，LIC2021 DuEE 事件抽取基线 |
| language_model       | 语言模型    | 提供了多个语言模型的PaddleNLP实现。如bert, bigbird, electra，elmo, gpt等等。也提供了支持语言模型在垂直了类领域数据上继续训练的工具包。   |
| lexical_analysis       | 词法分析    | 词法分析任务的输入是一个句子，而输出是句子中的词边界和词性、实体类别。这个例子基于双向GRU和CRF实现 |
| machine_reading_comprehension       | 机器阅读理解    | 提供了多个机器阅读理解数据集， 如SQuAD，DuReader以及它们对应的实现。 |
| machine_translation       | 机器翻译    | 提供了一个带Attention机制的，基于LSTM的多层RNN Seq2Seq翻译模型，以及一个基于Transformer的翻译模型 |
| model_compression       | 模型压缩    | 提供了一些大模型压缩，知识蒸馏的工具包。比如将Bert蒸馏到双向LSTM，对Bert网络进行压缩和蒸馏等。 |
| semantic_indexing       | 语义索引    | 语义索引技术是搜索引擎、推荐系统、广告系统在召回阶段的核心技术之一，语义索引库提供了前沿语义索引策略的训练、语义索引模型的效果评估方案、支持用户基于我们开源的语义索引模型进行文本 Pair 的相似度计算或者 Embedding 语义表示抽取。 |
| sentiment_analysis       | 情感分析    | 提供了两个情感分析任务实现示例，分别使用传统的TextCNN模型和情感预训练模型SKEP。 |
| simultaneous_translation/stacl/ | 同声传译    | 基于机器翻译领域主流模型 Transformer网络结构的同传模型STACL的PaddlePaddle 实现，包含模型训练，预测以及使用自定义数据等内容 |
| text_classification       | 文本分类    | 提供了多个文本分类任务示例，基于传统序列模型的二分类，基于预训练模型的二分类和基于预训练模型的多标签文本分类。 |
| text_generation       | 文本生成    | 包含ERNIE-Gen面向生成任务的预训练+微调模型框架，以及一个使用传统Seq2Seq模型的对对联应用，以及文本的基于变分自动编码器的生成模型 |
| text_graph/erniesage       | 文本图模型    | 基于PaddleNLP的ErnieSage模型，可以同时建模文本语义与图结构信息。 |
| text_matching      | 文本匹配    | 提供了多个不同算法的文本匹配算法实现。可以应用于搜索，推荐系统排序，召回等场景。 |
| text_summarization/pointer_summarizer       | 文本摘要    | 提供了一个基于指针生成网络的文本摘要算法实现。 |
| text_to_knowledge       | 语言模型    | 是百度解语的开源。解语（Text to Knowledge）是首个覆盖中文全词类的知识库（百科知识树）及知识标注框架，拥有可描述所有中文词汇的词类体系、中文知识标注工具集，以及更适用于中文挖掘任务的预训练语言模型 |
| text_to_sql       | 表格问答 / Text2SQL     | 提供了两个Text2SQL的例子实现。其中一个是IGSQL模型， 一个是RAT-SQL模型|
| time_series/tcn       | 时间序列预测    | 一个使用时间卷积网络TCN进行预测的例子 |
| word_embedding       | 词向量模型    | 提供了一个利用领域数据集提升词向量效果的例子。这个例子利用ChnSentiCorp数据集提升了预置词向量在分类任务上的准确性。 |
