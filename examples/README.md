# PaddleNLP 应用示例

[**PaddleNLP**](https://github.com/PaddlePaddle/PaddleNLP) 是基于 PaddlePaddle 深度学习框架开发的自然语言处理 (NLP) 工具，算法，模型和数据的开源项目。百度在 NLP 领域十几年的深厚积淀为 PaddleNLP 提供了强大的核心动力。PaddleNLP 提供较为丰富的模型库，基本涵盖了主流的NLP任务，因为模型库中使用了PaddleNLP提供的基础NLP工具，例如数据集处理，高层API，使得模型库的算法简洁易懂。

下面是 PaddleNLP 支持任务的具体信息，涵盖了 [**NLP基础技术**](#nlp基础技术)，[**NLP核心技术**](#nlp核心技术)， [**NLP系统应用**](#nlp系统应用)三大领域。同时随着NLP序列建模技术的成熟，我们还提供了更多的基于NLP序列建模技术的应用场景如[蛋白质二级结构预测](#蛋白质二级结构预测-protein-secondary-structure-prediction)以及进阶的[模型压缩](#模型压缩-model-compression)应用示例。

## NLP基础技术

### 词法分析 (Lexical Analysis)

| 模型   | 简介   |
| ------ | ----- |
| [BiGRU-CRF](./lexical_analysis) | 百度自主研发中文特色模型词法分析任务，集成了中文分词、词性标注和命名实体识别任务。输入是一个字符串，而输出是句子中的词边界和词性、实体类别。 |

### 词向量 (Word Embedding)

| 模型   | 简介   |
| ------ | ----- |
| [Word Embedding](./word_embedding) | 提供了丰富的中文预训练词向量，通过简单配置即可使用词向量来进行热启训练，能支持较多的中文场景下的训练任务的热启训练，加快训练收敛速度。|

### 命名实体识别 (Named Entity Recognition)

命名实体识别 (Named Entity Recognition，NER) 是NLP中一项非常基础的任务。NER是信息提取、问答系统、句法分析、机器翻译等众多NLP任务的重要基础工具。命名实体识别的准确度，决定了下游任务的效果，是NLP中非常重要的一个基础问题。
在NER任务提供了两种解决方案，一类LSTM/GRU + CRF(Conditional Random Field)，RNN类的模型来抽取底层文本的信息，而CRF(条件随机场)模型来学习底层Token之间的联系；另外一类是通过预训练模型，例如ERNIE，BERT模型，直接来预测Token的标签信息。
因为该类模型较为抽象，提供了一份快递单信息抽取的训练脚本给大家使用，具体的任务是通过两类的模型来抽取快递单的核心信息，例如地址，姓名，手机号码，具体的[快递单任务链接](./information_extraction/waybill_ie)。

| 模型   | 简介   |
| ------ | ----- |
| [BiGRU-CRF](./information_extraction/waybill_ie) |传统的序列标注模型，通过双向GRU模型能抽取文本序列的信息和联系，通过CRF模型来学习文本Token之间的联系，本示例通过Paddle 2.0高层API进行训练，代码结构简单易懂。 |
| [ERNIE/BERT Token Classification](./information_extraction/msra_ner) |通过预训练模型提供的强大的语义信息和ERNIE/BERT类模型的Self-Attention机制来覆盖Token之间的联系，直接通过BERT/ERNIE的序列分类模型来预测文本每个token的标签信息，模型结构简单，效果优异。|


### 语言模型 (Language Model)
在自然语言处理（NLP）领域中，语言模型预训练方法在多项NLP任务上都获得了不错的提升，广泛受到了各界的关注。在这里主要是提供了目前两种语言模型，一种是RNNLM模型，通过RNN网络来进行序列任务的预测；另外一种是ELMo模型，以双向 LSTM 为网路基本组件，以 Language Model 为训练目标，通过预训练得到通用的语义表示。

| 模型   | 简介   |
| ------ | ----- |
| [RNNLM](./language_model/rnnlm) | 序列任务常用的RNN网络，实现了一个两层的LSTM网络，然后LSTM的结果去预测下一个词出现的概率。是基于RNN的常规的语言模型。|
| [ELMo](./language_model/elmo) | [Deep contextualized word representations](https://allennlp.org/elmo), 发表于NAACL2018的动态词向量开山之作。|
| [Transformer-XL](./language_model/transformer-xl/) | [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)    |

### 预训练模型 (Pretrained Language Model)

PaddleNLP 提供了多种成熟的预训练模型技术，适用于自然语言理解(NLU)和自然语言生成(NLG)等多种场景。

| 模型    | 简介     |
| ------- | ------- |
| [BERT](./language_model/bert/) | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)     |
| [ERNIE](./text_classification/pretrained_models) | [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223)   |
| [ERNIE-Tiny](./text_classification/pretrained_models) | 百度自研的小型化ERNIE网络结构，采用浅层Transformer，加宽隐层参数，中文subword粒度词表结合蒸馏的方法使模型相比SOTA Before BERT 提升8.35%， 速度提升4.3倍。 |
| [ERNIE-GEN](./text_generation/ernie-gen) | [ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation](https://arxiv.org/abs/2001.11314) ERNIE-GEN是百度发布的生成式预训练模型，通过Global-Attention的方式解决训练和预测曝光偏差的问题，同时使用Multi-Flow Attention机制来分别进行Global和Context信息的交互，同时通过片段生成的方式来增加语义相关性。    |
| [ELECTRA](./language_model/electra/)  | [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) ELECTRA 创新性地引入GAN的思想对BERT预训练过程进行了改进，在和BERT具有相同的模型参数、预训练计算量一样的情况下，ELECTRA GLUE得分明显好。同时相比GPT、ELMo，在GLUE得分略好时，ELECTRA预训练模型只需要很少的参数和计算量。|
| [RoBERTa](./text_classification/pretrained_models) | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)   |
| [PLATO-2](./dialogue/plato-2) | 百度自研领先的开放域对话预训练模型 [PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning](https://arxiv.org/abs/2006.16779) |
| [GPT-2](./language_model/gpt) | [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)   |


## NLP核心技术

### 文本分类 (Text Classification)
文本分类是NLP和核心技术方向之一，在该任务上我们提供了两大类模型，一类是基于RNN类模型的传统轻量级的分类模型，一类是基于预训模型的分类模型。

| 模型  | 简介    |
| ----- | ------ |
| [RNN/CNN/GRU/LSTM](./text_classification/rnn) | 面向通用场景的文本分类模型，网络结构接入常见的RNN类模型，例如CNN, LSTM, GRU, RNN。|
| [ERNIE/BERT For Sequence Classification](./text_classification/pretrained_models) | 基于预训练模型的文本分类的模型，多达11种的预训练模型可供使用，其中有较多中文预训练模型，预训练模型切换简单，情感分析任务上效果突出。|

### 文本生成 (Text Generation)

文本生成是自然语言处理中一个重要的研究领域，具有广阔的应用前景。国内外已经有诸如Automated Insights、Narrative Science等文本生成系统投入使用，这些系统根据格式化数据或自然语言文本生成新闻、财报或者其他解释性文本。目前比较常见的文本生成任务两大类，文本写作和文本摘要。在这里主要提供百度自研的文本生成模型ERNIE-GEN, ERNIE-GEN是一种Multi-Flow结构的预训练和微调框架。ERNIE-GEN利用更少的参数量和数据，在摘要生成、问题生成、对话和生成式问答4个任务共5个数据集上取得了SOTA效果。我们基于ERNIE-GEN模型提供了一个自动关写诗的示例，来展示ERNIE-GEN的生成效果。

| 模型        | 简介      |
| --------- | ------------------ |
| [Seq2Seq](./text_generation/couplet) | 通过经典的Seq2Seq with Attention的网络结构，在中文对联数据集上完成文本生成的示例任务。更多关于模型细节可参考[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) |
| [VAE-Seq2Seq](./text_generation/vae-seq2seq) | 在传统的Seq2Seq框架基础上，通过VAE的方案完成文本生成任务。其中关于VAE的技术细节可参考[(Bowman et al., 2015) Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf) |
| [ERNIE-GEN](./text_generation/ernie-gen) | ERNIE-GEN是百度自研的生成式预训练模型，通过Global-Attention的方式解决训练和预测曝光偏差的问题，同时使用Multi-Flow Attention机制来分别进行Global和Context信息的交互，同时通过片段生成的方式来增加语义相关性。更多技术细节可参考[An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation](https://arxiv.org/abs/2001.11314) |


### 文本匹配 (Text Matching)

文本匹配一直是自然语言处理（NLP）领域一个基础且重要的方向，一般研究两段文本之间的关系。文本相似度计算、自然语言推理、问答系统、信息检索等，都可以看作针对不同数据和场景的文本匹配应用。在文本匹配的任务上提供了百度自研的SimNet语义匹配框架，以及基于Transformer结构的SentenceBERT模型。SimNet是一个计算短文本相似度的框架，主要包括 BOW、CNN、RNN、MMDNN 等核心网络结构形式，在百度各产品上广泛应用，提供语义相似度计算训练和预测框架，适用于信息检索、新闻推荐、智能客服等多个应用场景，帮助企业解决语义匹配问题。SentenceBERT模型是通过强大语义信息的预训练模型来表征句子的语义信息，通过比较两个句子的语义信息来判断两个句子是否匹配。

| 模型       | 简介                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [SimNet](./text_matching/simnet)| PaddleNLP提供的SimNet模型已经纳入了PaddleNLP的官方API中，用户直接调用API即完成一个SimNet模型的组网，在模型层面提供了Bow/CNN/LSTM/GRU等特征抽取方式, 灵活高，使用方便。|
| [SentenceTransformer](./text_matching/sentence_transformers)| 直接调用简易的预训练模型接口接口完成对Sentence的语义表示，同时提供了较多的中文预训练模型，可以根据任务的来选择相关参数。|

### 文本图学习 (Text Graph)

在很多工业应用中，往往出现一种特殊的图：Text Graph。顾名思义，图的节点属性由文本构成，而边的构建提供了结构信息。如搜索场景下的Text Graph，节点可由搜索词、网页标题、网页正文来表达，用户反馈和超链信息则可构成边关系。百度图学习PGL((Paddle Graph Learning)团队提出ERNIESage(ERNIE SAmple aggreGatE)模型同时建模文本语义与图结构信息，有效提升Text Graph的应用效果。图学习是深度学习领域目前的研究热点，如果想对图学习有更多的了解，可以访问[PGL Github链接](https://github.com/PaddlePaddle/PGL/)。

| 模型   | 简介     |
| ------- | ------- |
| [ERNIESage(ERNIE SAmple aggreGatE)](./text_graph/erniesage)| 通过Graph(图)来构建自身节点和邻居节点的连接关系，将自身节点和邻居节点的关系构建成一个关联样本输入到ERNIE中，ERNIE作为聚合函数（Aggregators）来表征自身节点和邻居节点的语义关系，最终强化图中节点的语义表示。|

### 文本信息抽取 (Information Extraction)

| 任务   | 简介     |
| ------- | ------- |
| [事件抽取](./information_extraction/DuEE)| 基于百度开放的大规模中文事件抽取数据集[DuEE](https://link.springer.com/chapter/10.1007/978-3-030-60457-8_44)，结合ERNIE预训练模型与序列标注模型完成高质量的事件抽取。|
| [关系抽取](./information_extraction/DuIE)| 基于百度开放的业界规模最大的Schema的中文关系抽取数据集[DuIE](http://tcci.ccf.org.cn/conference/2019/papers/EV10.pdf)，结合ERNIE预训练模型与序列标注模型完成文本关系抽取。|

### 文本知识挖掘 (Text to Knoledge)


## NLP系统应用

### 机器翻译 (Machine Translation)
机器翻译是计算语言学的一个分支，是人工智能的终极目标之一，具有重要的科学研究价值。在机器翻译的任务上，提供了两大类模型，一类是传统的 Sequence to Sequence任务，简称Seq2Seq，通过RNN类模型进行编码，解码；另外一类是Transformer类模型，通过Self-Attention机制来提升Encoder和Decoder的效果，Transformer模型的具体信息可以参考论文, [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。下面是具体的模型信息。
| 模型    | 简介     |
| ------ | ------- |
| [Seq2Seq](./machine_translation/seq2seq) | 使用编码器-解码器（Encoder-Decoder）结构, 同时使用了Attention机制来加强Decoder和Encoder之间的信息交互，Seq2Seq 广泛应用于机器翻译，自动对话机器人，文档摘要自动生成，图片描述自动生成等任务中。|
| [Transformer](./machine_translation/transformer) |基于PaddlePaddle框架的Transformer结构搭建的机器翻译模型，Transformer 计算并行度高，能解决学习长程依赖问题。并且模型框架集成了训练，验证，预测任务，功能完备，效果突出。|

### 同声传译（Simultaneous Translation）
同声传译（Simultaneous Translation）也隶属于机器翻译，它要求在句子完成之前进行翻译，同传模型STACL是针对同传场景提出的模型，它的Prefix-to-Prefix架构拥有预测能力，即在未看到源词的情况下仍然可以翻译出对应的目标词，克服了SOV（主宾谓）→SVO（主谓宾）等词序差异，同时Wait-k策略可以不需要全句的源句，直接预测目标句，可以实现任意的字级延迟，并保持较高的翻译质量。下面是具体的模型信息。
| 模型    | 简介     |
| ------ | ------- |
| [STACL](./simultaneous_translation/stacl) | 基于Transformer网络结构的同传模型STACL的PaddlePaddle 实现，STACL是针对同传提出的适用于所有同传场景的翻译模型，既能在未看到源词的情况下仍然翻译出对应的目标词，同时又能保持较高的翻译质量。|

### 阅读理解 (Machine Reading Comprehension)
机器阅读理解是近期自然语言处理领域的研究热点之一，也是人工智能在处理和理解人类语言进程中的一个长期目标。得益于深度学习技术和大规模标注数据集的发展，用端到端的神经网络来解决阅读理解任务取得了长足的进步。下面是具体的模型信息。

| 模型     | 简介     |
| ------ | -------- |
| [BERT/ERNIE For Question Answering](./machine_reading_comprehension/) |通过ERNIE/BERT等预训练模型的强大的语义表示能力，设置在阅读理解上面的下游任务，该模块主要是提供了多个数据集来验证BERT模型在阅读理解上的效果，数据集主要是包括了SQuAD，DuReader-robust，DuReader-yesno。同时提供了和相关阅读理解相关的Metric(指标)，用户可以简易的调用这些API，快速验证模型效果。|

### 对话系统 (Dialogue System)

对话系统 (Dialogue System) 常常需要根据应用场景的变化去解决多种多样的任务。任务的多样性（意图识别、槽填充、行为识别、状态追踪等等），以及领域训练数据的稀少给对话系统领域带来了诸多挑战。为此提供了基于BERT的对话通用理解模型 (DGU: Dialogue General Understanding)，该种训练范式在对话理解任务上取得比肩甚至超越各个领域业内最好的模型的效果，展现了学习一个通用对话理解模型的巨大潜力。下面是模型的信息：

| 模型   | 简介      |
| --------- | ------|
| [Dialogue General Understanding](./dialogue/dgu) | 提供基于BERT通用对话理解模型，通过对文本分类、序列标注等操作就可以完成对话中的意图识别，行文识别，状态跟踪等理解任务。|
| [PLATO-2](./dialogue/plato-2) | 百度自研领先的开放域对话预训练模型。[PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning](https://arxiv.org/abs/2006.16779) |

## 更多序列建模应用

### 时间序列预测 (Time Series Prediction)

时间序列是指按照时间先后顺序排列而成的序列，例如每日发电量、每小时营业额等组成的序列。通过分析时间序列中的发展过程、方向和趋势，我们可以预测下一段时间可能出现的情况。为了更好让大家了解时间序列预测任务，提供了基于2019年新冠疫情预测的任务示例，下面是具体的时间序列模型的信息：

| 模型     | 简介    |
| -------- | ------- |
| [TCN(Temporal Convolutional Network)](./time_series/tcn)|TCN模型基于卷积的时间序列模型，通过因果卷积(Causal Convolution)和空洞卷积(Dilated Convolution) 特定的组合方式解决卷积不适合时间序列任务的问题，TCN具备并行度高，内存低等诸多优点，在某些时间序列任务上效果已经超过传统的RNN模型。|

### 蛋白质二级结构预测 (Protein Secondary Structure Prediction)

| 模型     | 简介    |
| -------- | ------- |
| [TAPE](https://arxiv.org/abs/1906.08230) | 借鉴自然语言处理中对大量未标记的序列使用自监督学习的方式进行预训练（如Transformer, LSTM等序列建模结构），从而提取蛋白质中有用的生物学信息，并将这些信息迁移到其他带标签的任务，使得这些任务训练更快更稳定的收敛。更多详情可以参考[PaddleHelix蛋白质预训练模型](https://github.com/PaddlePaddle/PaddleHelix/blob/dev/apps/pretrained_protein/tape/README_cn.md#%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B)|

## 进阶应用

### 模型压缩 (Model Compression)

| 模型     | 简介    |
| -------- | ------- |
| [Distill-LSTM](./model_compression/distill_lstm/) | 基于[Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)论文策略的实现，将BERT中英文分类的下游模型知识通过蒸馏的方式迁移至LSTM的小模型结构中，取得比LSTM单独训练更好的效果。|
| [OFA-BERT](./model_compression/ofa/) | 基于PaddleSlim Once-For-ALL(OFA)策略对BERT在GLUE任务的下游模型进行压缩，在精度无损的情况下可减少33%参数量，达到模型小型化的提速的效果。 |
