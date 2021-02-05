# PaddleNLP Datasets API

PaddleNLP提供了

## 阅读理解

|  数据集名称   | 简介 | 调用方法 |
|  ----  | ----- | ------ |
|  [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) | 斯坦福问答数据集，包括SQaAD1.1和SQaAD2.0|`paddlenlp.datasets.SQuAD` |
|  [DuReader-yesno](https://aistudio.baidu.com/aistudio/competition/detail/49) | 千言数据集：阅读理解，判断答案极性|`paddlenlp.datasets.DuReaderYesNo` |
|  [DuReader-robust](https://aistudio.baidu.com/aistudio/competition/detail/49) | 千言数据集：阅读理解，答案原文抽取|`paddlenlp.datasets.DuReaderRobust` |

## 文本分类

| 数据集名称  | 简介 | 调用方法 |
| ----  | --------- | ------ |
|  [CoLA](https://nyu-mll.github.io/CoLA/) | 单句分类任务，二分类，判断句子是否合法| `paddlenlp.datasets.GlueCoLA`|
|  [SST-2](https://nlp.stanford.edu/sentiment/index.html) | 单句分类任务，二分类，判断句子情感极性| `paddlenlp.datasets.GlueSST2`|
|  [MRPC](https://microsoft.com/en-us/download/details.aspx?id=52398) | 句对匹配任务，二分类，判断句子对是否是相同意思| `paddlenlp.datasets.GlueMRPC`|
|  [STSB](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) | 计算句子对相似性，分数为1~5| `paddlenlp.datasets.GlueSTSB`|
|  [QQP](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) | 判定句子对是否等效，等效、不等效两种情况，二分类任务| `paddlenlp.datasets.GlueQQP`|
|  [MNLI](http://www.nyu.edu/projects/bowman/multinli/) | 句子对，一个前提，一个是假设。前提和假设的关系有三种情况：蕴含（entailment），矛盾（contradiction），中立（neutral）。句子对三分类问题| `paddlenlp.datasets.GlueMNLI`|
|  [QNLI](https://rajpurkar.github.io/SQuAD-explorer/) | 判断问题（question）和句子（sentence）是否蕴含，蕴含和不蕴含，二分类| `paddlenlp.datasets.GlueQNLI`|
|  [RTE](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment) | 判断句对是否蕴含，句子1和句子2是否互为蕴含，二分类任务| `paddlenlp.datasets.GlueRTE`|
|  [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html) | 判断句子对是否相关，相关或不相关，二分类任务| `paddlenlp.datasets.GlueWNLI`|
|  [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) | A Large-scale Chinese Question Matching Corpus 语义匹配数据集| `paddlenlp.datasets.LCQMC`|
|  [ChnSentiCorp](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/intro.ipynb) | 中文评论情感分析语料| `paddlenlp.datasets.ChnSentiCorp`|
|  [IMDB](https://www.imdb.com/interfaces/) | IMDB电影评论情感分析数据集| `paddle.text.datasets.Imdb`|
|  [Movielens](https://grouplens.org/datasets/movielens/) | Movielens 1-M电影评级数据集| `paddle.text.datasets.Movielens`|

## 序列标注

|  数据集名称   | 简介 | 调用方法 |
|  ----  | --------- | ------ |
|  [Conll05](https://www.cs.upc.edu/~srlconll/spec.html) | 语义角色标注数据集| `paddle.text.datasets.Conll05st`|
|  [MSRA_NER](https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra) | MSRA 命名实体识别数据集| `paddlenlp.datasets.MSRA_NER`|
|  [Express_Ner](https://aistudio.baidu.com/aistudio/projectdetail/131360?channelType=0&channel=-1) | 快递单命名实体识别数据集| [express_ner](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/examples/named_entity_recognition/express_ner/data)|

## 机器翻译

| 数据集名称  | 简介 | 调用方法 |
| ----  | --------- | ------ |
|  [IWSLT15](https://workshop2015.iwslt.org/) | IWSLT'15 English-Vietnamese data 英语-越南语翻译数据集| `paddlenlp.datasets.IWSLT15`|
|  [WMT14](http://www.statmt.org/wmt14/translation-task.html) | WMT14 EN-DE 英语-德语翻译数据集| `paddlenlp.datasets.WMT14ende`|

## 时序预测

| 数据集名称  | 简介 | 调用方法 |
| ----  | --------- | ------ |
|  [CSSE COVID-19](https://github.com/CSSEGISandData/COVID-19) |约翰·霍普金斯大学系统科学与工程中心新冠病例数据 | [time_series](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/examples/time_series)|
|  [UCIHousing](https://archive.ics.uci.edu/ml/datasets/Housing) | 波士顿房价预测数据集 | `paddle.text.datasets.UCIHousing`|

## 语料库

| 数据集名称  | 简介 | 调用方法 |
| ----  | --------- | ------ |
|  [yahoo](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&guccounter=1) | 雅虎英文语料库 | [VAE](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/examples/text_generation/vae-seq2seq)|
|  [PTB](http://www.fit.vutbr.cz/~imikolov/rnnlm/) | Penn Treebank Dataset | `paddlenlp.datasets.PTB`|
|  [1 Billon words](https://opensource.google/projects/lm-benchmark) | 1 Billion Word Language Model Benchmark R13 Output 基准语料库| [ELMo](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/examples/language_model/elmo)|
