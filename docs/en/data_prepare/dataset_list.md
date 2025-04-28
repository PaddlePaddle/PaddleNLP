# PaddleNLP Datasets API

PaddleNLP provides easy-to-use APIs for the following datasets. Please add **splits** information as needed:

## Reading Comprehension

| Dataset Name | Description | Method |
| ---- | ----- | ------ |
| [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) | Stanford Question Answering Dataset (SQuAD1.1 and SQuAD2.0) | `paddlenlp.datasets.load_dataset('squad')` |
| [DuReader-yesno](https://aistudio.baidu.com/aistudio/competition/detail/49) | Dureader Yes-No: Polarity judgment for reading comprehension | `paddlenlp.datasets.load_dataset('dureader_yesno')` |
| [DuReader-robust](https://aistudio.baidu.com/aistudio/competition/detail/49) | Dureader Robust: Answer extraction for reading comprehension | `paddlenlp.datasets.load_dataset('dureader_robust')` |
| [CMRC2018](https://hfl-rc.github.io/cmrc2018/) | Chinese Machine Reading Comprehension 2018 | `paddlenlp.datasets.load_dataset('cmrc2018')` |
| [DRCD](https://github.com/DRCKnowledgeTeam/DRCD) | Delta Reading Comprehension Dataset | `paddlenlp.datasets.load_dataset('drcd')` |
| [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) | Trivia Question Answering Dataset | `paddlenlp.datasets.load_dataset('triviaqa')` |
| [C3](https://dataset.org/c3/) | Multiple-choice Reading Comprehension | `paddlenlp.datasets.load_dataset('c3')` |

## Text Classification

| Dataset Name | Description | Method |
| ---- | --------- | ------ |
| [CoLA](https://nyu-mll.github.io/CoLA/) | Single-sentence classification, binary (grammatical correctness) | `paddlenlp.datasets.load_dataset('glue','cola')` |
| [SST-2](https://nlp.stanford.edu/sentiment/index.html) | Single-sentence classification, binary (sentiment analysis) | `paddlenlp.datasets.load_dataset('glue','sst-2')` |
| [MRPC](https://microsoft.com/en-us/download/details.aspx?id=52398) | Sentence pair classification, binary (paraphrase detection) | `paddlenlp.datasets.load_dataset('glue','mrpc')` |
|  [STSB](https://huggingface.co/datasets/mteb/stsbenchmark-sts) | Calculate sentence pair similarity, score ranges from 1 to 5 | `paddlenlp.datasets.load_dataset('glue','sts-b')` |
|  [QQP](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) | Determine if sentence pairs are equivalent, with two categories: equivalent and non-equivalent (binary classification) | `paddlenlp.datasets.load_dataset('glue','qqp')` |
|  [MNLI](http://www.nyu.edu/projects/bowman/multinli/) | Sentence pairs with premise and hypothesis. Relationships between premise and hypothesis fall into three categories: entailment, contradiction, neutral (3-class classification) | `paddlenlp.datasets.load_dataset('glue','mnli')` |
|  [QNLI](https://rajpurkar.github.io/SQuAD-explorer/) | Determine if question and sentence are entailed, with two categories: entailment and not_entailed (binary classification) | `paddlenlp.datasets.load_dataset('glue','qnli')` |
|  [RTE](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment) | Judge if sentence pairs entail each other, with two categories: entailment and not_entailed (binary classification) | `paddlenlp.datasets.load_dataset('glue','rte')` |
|  [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html) | Determine if sentence pairs are related, with two categories: related and unrelated (binary classification) | `paddlenlp.datasets.load_dataset('glue','wnli')` |
|  [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) | A Large-scale Chinese Question Matching Corpus (semantic matching dataset) | `paddlenlp.datasets.load_dataset('lcqmc')` |
|  [ChnSentiCorp](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/intro.ipynb) | Chinese review sentiment analysis corpus | `paddlenlp.datasets.load_dataset('chnsenticorp')` |
|  [COTE-DP](https://aistudio.baidu.com/aistudio/competition/detail/50/?isFromLuge=1) | Chinese opinion extraction corpus | `paddlenlp.datasets.load_dataset('cote', 'dp')` |
|  [SE-ABSA16_PHNS](https://aistudio.baidu.com/aistudio/competition/detail/50/?isFromLuge=1) | Chinese Aspect-based Sentiment Analysis Corpus | `paddlenlp.datasets.load_dataset('seabsa16', 'phns')`|
|  [AFQMC](https://github.com/CLUEbenchmark/CLUE) | Ant Financial Semantic Similarity Dataset (1: similar, 0: dissimilar) | `paddlenlp.datasets.load_dataset('clue', 'afqmc')`|
|  [TNEWS](https://github.com/CLUEbenchmark/CLUE) | Toutiao Chinese News Headlines Classification (15 categories) | `paddlenlp.datasets.load_dataset('clue', 'tnews')`|
|  [IFLYTEK](https://github.com/CLUEbenchmark/CLUE) | Long Text Classification (119 categories) | `paddlenlp.datasets.load_dataset('clue', 'iflytek')`|
|  [OCNLI](https://github.com/cluebenchmark/OCNLI) | Original Chinese Natural Language Inference (three-way classification) | `paddlenlp.datasets.load_dataset('clue', 'ocnli')`|
|  [CMNLI ](https://github.com/CLUEbenchmark/CLUE) | Chinese Language Understanding and Inference (entailment/contradiction/neutral) | `paddlenlp.datasets.load_dataset('clue', 'cmnli')`|
|  [CLUEWSC2020](https://github.com/CLUEbenchmark/CLUE) | Chinese Winograd Schema Challenge (coreference resolution) | `paddlenlp.datasets.load_dataset('clue', 'cluewsc2020')`|
|  [CSL](https://github.com/P01son6415/CSL) | Chinese Scientific Literature Keyword Recognition (binary classification) | `paddlenlp.datasets.load_dataset('clue', 'csl')`|
|  [EPRSTMT](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets)  | E-commerce Product Review Sentiment Analysis (Positive/Negative) | `paddlenlp.datasets.load_dataset('fewclue', 'eprstmt')`|
|  [CSLDCP](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets)  | Chinese Scientific Literature Discipline Classification (67 categories) | `paddlenlp.datasets.load_dataset('fewclue', 'csldcp')`|
| Dataset | Description | Command |
| --- | --- | --- |
| [CSLDCP](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets) | Chinese literature discipline classification from FewCLUE benchmark, 67 categories | `paddlenlp.datasets.load_dataset('fewclue', 'csldcp')` |
| [TNEWSF](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets) | Today's Headlines Chinese news (short text) classification from FewCLUE, 15 categories | `paddlenlp.datasets.load_dataset('fewclue', 'tnews')` |
| [IFLYTEK](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets) | Long text classification task from FewCLUE, 119 categories | `paddlenlp.datasets.load_dataset('fewclue', 'iflytek')` |
| [OCNLIF](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets) | Chinese natural language inference dataset from FewCLUE, sentence pair ternary classification | `paddlenlp.datasets.load_dataset('fewclue', 'ocnli')` |
| [BUSTM](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets) | Dialogue short text semantic matching dataset from FewCLUE, binary classification | `paddlenlp.datasets.load_dataset('fewclue', 'bustm')` |
| [CHIDF](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets) | Chinese idiom reading comprehension cloze from FewCLUE, predict correct idiom from 7 candidates | `paddlenlp.datasets.load_dataset('fewclue', 'chid')` |
| [CSLF](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets) | Paper keyword recognition from FewCLUE, binary classification for authentic keywords | `paddlenlp.datasets.load_dataset('fewclue', 'csl')` |
| [CLUEWSCF](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets) | WSC Winograd schema challenge Chinese version from FewCLUE, pronoun disambiguation task | `paddlenlp.datasets.load_dataset('fewclue', 'cluewsc')` |
| [THUCNews](https://github.com/gaussic/text-classification-cnn-rnn#%E6%95%B0%E6%8D%AE%E9%9B%86) | THUCNews Chinese news category classification | `paddlenlp.datasets.load_dataset('thucnews')` |
| [HYP](https://pan.webis.de/semeval19/semeval19-web/) | English political news sentiment classification corpus |
| Dataset Name | Description | How to Use |
| ---- | --------- | ------ |
| [ChnSentiCorp](https://paddlehub-dataset.bj.bcebos.com/chnsenticorp.tar.gz) | Chinese sentiment analysis dataset | `paddlenlp.datasets.load_dataset('chnsenticorp')` |
| [LCQMC](http://icrc.hitsz.edu.cn/info/1037/1146.htm) | Chinese question matching corpus, binary classification task | `paddlenlp.datasets.load_dataset('lcqmc', splits=['test', 'dev'])` |
| [NLPCC-DBQA](https://github.com/shuishen112/NLPCCDBQA) | Chinese database question answering dataset, binary classification task | `paddlenlp.datasets.load_dataset('nlpcc_dbqa')` |

## Natural Language Inference

| Dataset Name | Description | How to Use |
| ---- | --------- | ------ |
| [CMNLI](https://github.com/CLUEbenchmark/CLUE) | Chinese Multi-Genre NLI dataset | `paddlenlp.datasets.load_dataset('cmnli')` |
| [OCNLI](https://github.com/CLUEbenchmark/OCNLI) | Chinese Original NLI dataset | `paddlenlp.datasets.load_dataset('ocnli')` |
| [GLUE-MNLI](https://gluebenchmark.com/) | English Multi-Genre NLI dataset | `paddlenlp.datasets.load_dataset('glue', 'mnli')` |
| [GLUE-QNLI](https://gluebenchmark.com/) | English Question NLI dataset | `paddlenlp.datasets.load_dataset('glue', 'qnli')` |
| [GLUE-RTE](https://gluebenchmark.com/) | English Recognizing Textual Entailment dataset | `paddlenlp.datasets.load_dataset('glue', 'rte')` |
| [XNLI](https://github.com/facebookresearch/XNLI) | 15-language NLI dataset, 3-class task | `paddlenlp.datasets.load_dataset('xnli', 'ar')` |
| [XNLI_CN](https://github.com/facebookresearch/XNLI) | Chinese subset of XNLI, 3-class task | `paddlenlp.datasets.load_dataset('xnli_cn')` |

## Text Matching

| Dataset Name | Description | How to Use |
| ---- | --------- | ------ |
| [CAIL2019-SCM](https://github.com/china-ai-law-challenge/CAIL2019/tree/master/scm) | Similar legal case matching | `paddlenlp.datasets.load_dataset('cail2019_scm')` |

## Sequence Labeling

| Dataset Name | Description | How to Use |
| ---- | --------- | ------ |
| [MSRA_NER](https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra) | MSRA named entity recognition dataset | `paddlenlp.datasets.load_dataset('msra_ner')` |
| [People's Daily](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily) | People's Daily named entity recognition dataset | `paddlenlp.datasets.load_dataset('peoples_daily_ner')` |
| [CoNLL-2002](https://www.aclweb.org/anthology/W02-2024/) | Spanish and Dutch NER datasets | `paddlenlp.datasets.load_dataset('conll2002', 'es')` |

## Machine Translation

| Dataset Name | Description | How to Use |
| ---- | --------- | ------ |
| [IWSLT15](https://workshop2015.iwslt.org/) | IWSLT'15 English-Vietnamese translation dataset | `paddlenlp.datasets.load_dataset('iwslt15')` |
| [WMT14ENDE](http://www.statmt.org/wmt14/translation-task.html) | WMT14 EN-DE translation dataset with BPE tokenization | `paddlenlp.datasets.load_dataset('wmt14ende')` |
## Machine Simultaneous Translation

| Dataset Name | Description | Loading Method |
| ---- | --------- | ------ |
| [BSTC](https://aistudio.baidu.com/aistudio/competition/detail/44/) | Baidu Speech Translation Corpus, including transcription_translation and ASR | `paddlenlp.datasets.load_dataset('bstc', 'asr')` |

## Dialogue System

| Dataset Name | Description | Loading Method |
| ---- | --------- | ------ |
| [DuConv](https://aistudio.baidu.com/aistudio/competition/detail/48/) | Knowledge-aware Chinese Conversation Dataset | `paddlenlp.datasets.load_dataset('duconv')` |

## Text Generation

| Dataset Name | Description | Loading Method |
| ---- | --------- | ------ |
| [Poetry](https://github.com/chinese-poetry/chinese-poetry) | Classical Chinese Poetry Collection | `paddlenlp.datasets.load_dataset('poetry')` |
| [Couplet](https://github.com/v-zich/couplet-clean-dataset) | Chinese Couplet Dataset | `paddlenlp.datasets.load_dataset('couplet')` |
| [DuReaderQG](https://github.com/PaddlePaddle/Research/tree/master/NLP/DuReader-Robust-BASELINE) | Question Generation Dataset Based on DuReader | `paddlenlp.datasets.load_dataset('dureader_qg')` |
| [AdvertiseGen](https://github.com/ZhihongShao/Planning-based-Hierarchical-Variational-Model) | Chinese Advertising Copy Generation Dataset | `paddlenlp.datasets.load_dataset('advertisegen')` |
| [LCSTS_new](https://aclanthology.org/D15-1229.pdf) | Chinese Abstractive Summarization Dataset | `paddlenlp.datasets.load_dataset('lcsts_new')` |
| [CNN/Dailymail](https://github.com/abisee/cnn-dailymail) | English Abstractive Summarization Dataset | `paddlenlp.datasets.load_dataset('cnn_dailymail')` |

## Corpus

| Dataset Name | Description | Loading Method |
| ---- | --------- | ------ |
| [PTB](http://www.fit.vutbr.cz/~imikolov/rnnlm/) | Penn Treebank Dataset | `paddlenlp.datasets.load_dataset('ptb')` |
| Dataset Name | Description | Example Usage |
|--------------|-------------|---------------|
| [PTB](https://arxiv.org/abs/1603.04467) | Penn Treebank Dataset | `paddlenlp.datasets.load_dataset('ptb')` |
| [Yahoo Answer 100k](https://arxiv.org/pdf/1702.08139.pdf) | 100k samples from Yahoo Answer | `paddlenlp.datasets.load_dataset('yahoo_answer_100k')` |
