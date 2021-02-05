# PaddleNLP Model Zoo

PaddleNLP提供了丰富的模型结构，包含经典的RNN类模型结构，与Transformer类模型及其预训练模型。

## RNN类模型

| 模型    |  简介   |
| ------ | ------ |
| [BiGRU-CRF](../examples/lexical_analysis) | BiGRU-CRF是一个经典的词法分析模型，可用于中文分词、词性标注和命名实体识别等任务。    |
| [BoW](../examples/text_classification/rnn) | 最基础的序列特征提取模型，对序列内所有词向量进行线性求和或取平均的操作。    |
| [RNN/Bi-RNN](../examples/text_classification/rnn) | 单/双向RNN序列特征提取器，是变种的LSTM结构，计算量相比LSTM较少。    |
| [LSTM/Bi-LSTM](../examples/text_classification/rnn) | 单/双向LSTM序列特征提取器。  |
| [TextCNN](../examples/text_classification/rnn) | [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)  |
| [LSTM/Bi-LSTM with Attention](../examples/text_classification/rnn) | 带注意力机制的单/双向LSTM特征提取器。   |
| [GRU/Bi-GRU](../examples/text_classification/rnn) | 单/双向GRU序列特征提取器，是变种的LSTM结构，计算量相比LSTM较少。    |
| [TCN](../examples/time_series)|TCN(Temporal Convolutional Network)模型基于卷积的时间序列模型，通过因果卷积(Causal Convolution)和空洞卷积(Dilated Convolution) 特定的组合方式解决卷积不适合时间序列任务的问题，TCN具备并行度高，内存低等诸多优点，在某些时间序列任务上效果超过传统的RNN模型。|
| [RNNLM](../examples/language_model/rnnlm/) | [Recurrent neural network based language model](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)，RNN/LSTM等结构的经典语言模型。   |
| [ELMo](../examples/language_model/elmo/) | Embedding from Language Model(ELMo), 发表于NAACL2018的动态词向量开山之作。 |
| [SimNet](../examples/text_matching/simnet) |  SimNet是百度自研的文本匹配计算的框架，主要包括 BOW、CNN、RNN、MMDNN 等核心网络结构形式，已在百度各产品上广泛应用。|
| [LSTM Seq2Seq with Attention](../examples/machine_translation/seq2seq) | 使用编码器-解码器(Encoder-Decoder) 结构, 同时使用了Attention机制来加强Decoder和Encoder之间的信息交互，Seq2Seq 广泛应用于机器翻译，自动对话机器人，文档摘要自动生成，图片描述自动生成等任务中。|

## Transformer类模型

| 模型    |  简介   |
| ------ | ------ |
| [Transformer](../examples/machine_translation/transformer/) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)     |
| [Transformer-XL](../examples/language_model/transformer-xl/) | [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)     |
| [BERT](../examples/language_model/bert/) |[BERT(Bidirectional Encoder Representation from Transformers)](./examples/language_model/bert)      |
| [ERNIE](../examples/text_classification/rnn) | [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223)   |
| [ERNIE-Tiny](../examples/text_classification/rnn) | 百度自研的小型化ERNIE网络结构，采用浅层Transformer，加宽隐层参数，中文subword粒度词表结合蒸馏的方法使模型相比SOTA Before BERT 提升8.35%， 速度提升4.3倍。 |
| [ERNIE-GEN](../examples/text_generation/ernie-gen) | [ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation](https://arxiv.org/abs/2001.11314) ERNIE-GEN是百度发布的生成式预训练模型，通过Global-Attention的方式解决训练和预测曝光偏差的问题，同时使用Multi-Flow Attention机制来分别进行Global和Context信息的交互，同时通过片段生成的方式来增加语义相关性。    |
| [ERNIESage](../examples/text_graph/erniesage)| ERNIESage(ERNIE SAmple aggreGatE) 通过Graph(图)来构建自身节点和邻居节点的连接关系，将自身节点和邻居节点的关系构建成一个关联样本输入到ERNIE中，ERNIE作为聚合函数 (Aggregators) 来表征自身节点和邻居节点的语义关系，最终强化图中节点的语义表示。|
| [GPT-2](../examples/language_model/gpt2) |[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)   |
| [ELECTRA](../examples/language_model/electra/) | [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)   |
| [RoBERTa](../examples/text_classification/rnn) | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)   |
| [PLATO-2](../examples/dialogue/plato-2) | 百度自研领先的开放域对话预训练模型 [PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning](https://arxiv.org/abs/2006.16779) |
| [SentenceBERT](../examples/text_matching/sentence_transformers)| [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) |


更多模型应用场景介绍请参考[PaddleNLP Example](../examples/)
