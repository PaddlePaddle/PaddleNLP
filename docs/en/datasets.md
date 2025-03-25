# PaddleNLP Datasets API

PaddleNLP provides a quick API for loading the following datasets. When using, please **add splits information** as needed:

## Reading Comprehension

| Dataset Name | Description | Method Call |
| ------------ | ----------- | ----------- |
| [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) | Stanford Question Answering Dataset, including SQuAD1.1 and SQuAD2.0 | `paddlenlp.datasets.load_dataset('squad')` |
| [DuReader-yesno](https://aistudio.baidu.com/aistudio/competition/detail/49) | Qianyan Dataset: Reading comprehension, determining answer polarity | `paddlenlp.datasets.load_dataset('dureader_yesno')` |
| [DuReader-robust](https://aistudio.baidu.com/aistudio/competition/detail/49) | Qianyan Dataset: Reading comprehension, answer extraction from original text | `paddlenlp.datasets.load_dataset('dureader_robust')` |
| [CMRC2018](https://hfl-rc.github.io/cmrc2018/) | The Second "iFLYTEK Cup" Chinese Machine Reading Comprehension Evaluation Dataset | `paddlenlp.datasets.load_dataset('cmrc2018')` |
| [DRCD](https://github.com/DRCKnowledgeTeam/DRCD) | Delta Reading Comprehension Dataset | `paddlenlp.datasets.load_dataset('drcd')` |

## Text Classification

| Dataset Name | Description | Method Call |
| ------------ | ----------- | ----------- |
| [CoLA](https://nyu-mll.github.io/CoLA/) | Single sentence classification task, binary classification, determining sentence grammaticality | `paddlenlp.datasets.load_dataset('glue','cola')` |
| [SST-2](https://nlp.stanford.edu/sentiment/index.html) | Single sentence classification task, binary classification, determining sentence sentiment polarity | `paddlenlp.datasets.load_dataset('glue','sst-2')` |
| [MRPC](https://microsoft.com/en-us/download/details.aspx?id=52398) | Sentence pair matching task, binary classification, determining if sentence pairs have the same meaning | `paddlenlp.datasets.load_dataset('glue','mrpc')` |
| [STSB](https://huggingface.co/datasets/mteb/stsbenchmark-sts) | Calculating sentence pair similarity, score ranges from 1 to 5 | `paddlenlp.datasets.load_dataset('glue','sts-b')` |
|
|  [QQP](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) | Determine whether sentence pairs are equivalent, with two cases: equivalent and not equivalent, a binary classification task | `paddlenlp.datasets.load_dataset('glue','qqp')` |
|  [MNLI](http://www.nyu.edu/projects/bowman/multinli/) | Sentence pairs consisting of a premise and a hypothesis. The relationship between the premise and hypothesis can be one of three: entailment, contradiction, or neutral. A three-class classification problem for sentence pairs | `paddlenlp.datasets.load_dataset('glue','mnli')` |
|  [QNLI](https://rajpurkar.github.io/SQuAD-explorer/) | Determine whether a question and a sentence entail each other, with entailment and non-entailment as the two classes, a binary classification task | `paddlenlp.datasets.load_dataset('glue','qnli')` |
|  [RTE](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment) | Determine whether sentence pairs entail each other, whether sentence 1 and sentence 2 are mutually entailing, a binary classification task | `paddlenlp.datasets.load_dataset('glue','rte')` |
|  [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html) | Determine whether sentence pairs are related, with related and unrelated as the two classes, a binary classification task | `paddlenlp.datasets.load_dataset('glue','wnli')` |
|  [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) | A Large-scale Chinese Question Matching Corpus for semantic matching | `paddlenlp.datasets.load_dataset('lcqmc')` |
|  [ChnSentiCorp](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/intro.ipynb) | Chinese sentiment analysis corpus for reviews | `paddlenlp.datasets.load_dataset('chnsenticorp')` |

## Sequence Labeling

|  Dataset Name   | Description | Loading Method |
|  ----  | --------- | ------ |
|  [MSRA_NER](https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra) | MSRA Named Entity Recognition dataset | `paddlenlp.datasets.load_dataset('msra_ner')` |
|  [People's Daily](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily) | People's Daily Named Entity Recognition dataset |
`paddlenlp.datasets.load_dataset('peoples_daily_ner')`|

## Machine Translation

| Dataset Name  | Description | Usage |
| ----  | --------- | ------ |
|  [IWSLT15](https://workshop2015.iwslt.org/) | IWSLT'15 English-Vietnamese data, English-Vietnamese translation dataset | `paddlenlp.datasets.load_dataset('iwslt15')`|
|  [WMT14ENDE](http://www.statmt.org/wmt14/translation-task.html) | WMT14 EN-DE English-German translation dataset with BPE tokenization | `paddlenlp.datasets.load_dataset('wmt14ende')`|

## Simultaneous Interpretation

| Dataset Name  | Description | Usage |
| ----  | --------- | ------ |
|  [BSTC](https://aistudio.baidu.com/aistudio/competition/detail/44/) | Thousand Words Dataset: Simultaneous Interpretation, including transcription_translation and ASR | `paddlenlp.datasets.load_dataset('bstc', 'asr')`|

## Text Generation

| Dataset Name  | Description | Usage |
| ----  | --------- | ------ |
|  [Poetry](https://github.com/chinese-poetry/chinese-poetry) | Classical Chinese poetry collection dataset | `paddlenlp.datasets.load_dataset('poetry')`|
|  [Couplet](https://github.com/v-zich/couplet-clean-dataset) | Chinese couplet dataset | `paddlenlp.datasets.load_dataset('couplet')`|

## Corpora

| Dataset Name  | Description | Usage |
| ----  | --------- | ------ |
|  [PTB](http://www.fit.vutbr.cz/~imikolov/rnnlm/) | Penn Treebank Dataset | `paddlenlp.datasets.load_dataset('ptb')`|
|  [Yahoo Answer 100k](https://arxiv.org/pdf/1702.08139.pdf)  | Sampled 100K from Yahoo Answer | `paddlenlp.datasets.load_dataset('yahoo_answer_100k')`|
