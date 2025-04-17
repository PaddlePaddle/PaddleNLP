# ERNIE 中文词表制作

ERNIE 是百度提出的大规模预训练模型，曾在中文场景下取得了 SOTA 效果。
PaddleNLP 致力于预训练开源工作，本文档提供了 ERNIE 词表的制作方法。

预训练全部流程的整体详细介绍文档，请参考[ERNIE 中文预训练介绍](../pretraining_introduction.md)。

**目录**
* [1. 数据获取](#数据获取)
* [2. 全字符中文词表制作](#中文词表制作)
    - [2.1 分析准备](#分析准备)
    - [2.2 文本字符统计](#文本字符统计)
    - [2.3 英文字符词表](#英文字符词表)
    - [2.4 合并词表](#合并词表)
* [3. 词表使用](#vocab_usage)
    - [3.1 转化为 jsonl 格式数据](#jsonl)
    - [3.2 TokenID 转化](#快速 TokenID 转化)
* [4. 参考](#ref)


<a name="数据获取"> </a>

## 1. 数据获取


**WuDaoCorpus2.0 Base 语料**

WuDaoCorpora 是悟道爬取的中文大规模语料。整体数量为3TB，目前开源的部分为 WuDaoCorpus2.0 bases 数据集，大小为200GB。用户请参考[这里](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/tools/preprocess/docs/WuDaoCorpusBase.md)获取原始文本数据。


**CLUECorpus2020 语料**

CLUECorpus2020 过对 Common Crawl 的中文部分进行语料清洗得到。开源部分提供了约200G 左右的语料文本，详细介绍见[官网](https://github.com/CLUEbenchmark/CLUECorpus2020#%E6%95%B0%E6%8D%AE%E4%B8%8B%E8%BD%BD)，用户参考[这里](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/tools/preprocess/docs/CLUECorpus2020.md)获取原始文本数据。




<a name="全字符中文词表制作"> </a>

## 2. 全字符中文词表制作

词表的制作有两种方案：

第一种，词表组合方案
1. 统计字符
2. 制作英文词表
3. 合并词表

第二种，预处理后直接生成，方案
1. 文本预处理（中文加空格，文本 normalize）
2. 使用 sentencepeice 制作词表

第二种方案需要对文本先使用`BasicTokenizer`切分一遍语料。
第一种方案，自定义程度高，但存在一些局限性。本项目采用了第一种方案，详细介绍如下：

### 2.1 分析准备
词表大小： 这里我们考虑的因素主要有两个
- 已有模型对照：
    - ERNIE 3.0系列模型的词表，词表大小为 40000 左右。
- 预训练数据存储占用：
    - 文本 token id 化后，希望使用 uint16表示，此时表示的最大字符为65536。
    - 同时考虑到 ERNIE 虽然是字模型，我们的仍然需要 `##中` 之类的中文字符表示分词信息。假设使用中文全字符20902(0x4E00-0x9FA5)个字符，那么剩余 vocab 大小不能超过 44634。

综上，本项目决定采用 40000 左右的 vocab 容量。
其中：
- 中文全字符 `20902`
- 英文字符 `17000`
- 其他字符约 `2000` 左右


### 2.2 文本字符统计
首先第一步是对文本字符进行统计。字符统计的目的主要是添加常用的中文字符、特殊字符。

由于语料文本过大，我们随机选取 10G 左右的原始文本进行了字符统计。
```
python gen_char.py path_to_corpus.txt
```
可以在本地文件夹得到`char_dict.pickle`字符频率文件。同时我们也提供了自己统计的词频文件，方便用户复现：
```
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/char_dict.pickle
```

### 2.3 英文字符词表
基于字符的词频统计，使得英文字符也切割为字母，为此我们需要添加英文词表。
英文部分，我们使用了 [WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)  数据集，来构造词表。
下载解压数据，使用 BPE 切词
```
wget  https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
python gen_vocab.py ./wikitext-103-raw/wiki.train.raw
```
即可产生英文部分的词表。这里我们也提供了处理好的 vocab 方便用户验证。
```
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/eng.vocab
```


### 2.4 合并词表

目前我们得到了字符统计表，和英文字符词表。下一步，我们将词表进行合并。

将`char_dict.pickle`，`eng.vocab`放置到当前目录，使用下面命令
```
python merge_vocab.py
```
即可在 当前 目录生成 vocab.txt 得到最终词表。

此阶段需要注意的一些问题是：
1. 对于一些日文、谚文文字字符，需要进行 normalize
2. 添加 special_tokens

### 2.5 问题遗留
本项目采用的第一种方式，即拼接产出的词表，对连续非中、英文字符文本，会出现 UNK 的情况。
如 issue: [#2927](https://github.com/PaddlePaddle/PaddleNLP/issues/2927)、 [#2585](https://github.com/PaddlePaddle/PaddleNLP/issues/2585)。本项目做了两点改进:

1. 对 Symbol 字符默认添加空格，变成独立字符
2. 对 日文、谚文 在合并词表阶段默认添加 ## 字符。

虽然有上述两点修复，任然无法避免 [#2927](https://github.com/PaddlePaddle/PaddleNLP/issues/2927) 现象。
彻底解决的话，建议使用第二种方式制作 vocab 文件。

### 2.6 方案二：预处理后直接生成
此方案没有被采用，这里也简单说明一下具体的方案：
1. 对语料使用 BasicTokenizer 转换
```python
from paddlenlp.transformers import
tokenizer = BasicTokenizer()
basic_toknizer = lambda x: " ".join(tokenizer.tokenize(x))
# 对语料使用 basic_toknizer 转换
# 并存储为新的语料 afer_basic_toknizer_corpus.txt
```
2. 处理转换后的语料
```shell
python gen_vocab.py afer_basic_toknizer_corpus.txt
```
对处理好的 vocab 文件手动替换一些`<pad> -> [PAD]`之类的 special_tokens，即可产出词表。


<a name="vocab_usage"></a>
## 3. 词表使用

<a name="josnl"> </a>

## 3.1 转化为 jsonl 格式数据

本文档以 WuDao 数据为例，对数据进行分词：

```shell
python ../preprocess/words_segmentation.py \
    --input_path ./WuDaoCorpus2.0_base_200G \
    --workers 40  \
    --data_format wudao \
    --cn_seg_func seg \
    --output_path ./wudao_lac_cut \
```

文本转化完成后。我们使用 `../data_tools/trans_to_json.py`重新转换为 jsonl 格式（分词完毕）。
```shell
python ../preprocess/trans_to_json.py  \
    --input_path ./wudao_lac_cut \
    --output_path wudao_corpus_200g_0623.jsonl \
    --workers 40 \
```

<a name="快速 TokenID 转化"> </a>

## 3.2 Token ID 转化

语料、新建的词表准备妥当后，我们可以开始进行最后的数据 ID 转化。

```
python -u  ../preprocess/create_pretraining_data.py \
    --model_name /path/to/your/vocab.txt \
    --tokenizer_name ErnieTokenizer \
    --input_path wudao_corpus_200g_0623.jsonl \
    --split_sentences \
    --chinese \
    --cn_whole_word_segment \
    --cn_seg_func jieba \
    --cn_splited \
    --output_prefix wudao_corpus_200g_0623 \
    --workers 48 \
    --log_interval 10000
```

- 我们提前分词好了，所以加上了 `cn_splited`，否则不需要使用此选项。
- model_name 指定为我们准备的词表路径。也可以更换为其他 ERNIE 系列模型，如: `ernie-3.0-base-zh`
- workers 表示转化的线程数目

转化后的数据如下，使用这份数据，即可开始 ERNIE 预训练
```
-rw-rw-r-- 1 500 501 129G Jul  4 03:39 wudao_200g_0703_ids.npy
-rw-rw-r-- 1 500 501 6.4G Jul  4 03:39 wudao_200g_0703_idx.npz
```

<a name='ref'></a>
## 4. 参考

感谢 CLUE，WuDao 提供的开源文本语料，参考资料：
- Xu, L., Zhang, X. and Dong, Q., 2020. CLUECorpus2020: A large-scale Chinese corpus for pre-training language model. arXiv preprint arXiv:2003.01355.
- Yuan, S., Zhao, H., Du, Z., Ding, M., Liu, X., Cen, Y., Zou, X., Yang, Z. and Tang, J., 2021. Wudaocorpora: A super large-scale chinese corpora for pre-training language models. AI Open, 2, pp.65-68.
- https://github.com/CLUEbenchmark/CLUECorpus2020
- https://resource.wudaoai.cn
