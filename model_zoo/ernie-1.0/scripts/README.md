# **大规模** **开源** **中文** 语料预训练-<small>从零开始构建预训练模型</small>

ERNIE是百度提出的大规模预训练模型，曾在中文场景下取得了SOTA效果。
PaddleNLP致力于预训练开源工作，使用开源中文语料CLUE、WuDao 总共400GB，发布大规模开源语料预训练全流程。从零开始，轻松构建预训练模型。

本项目，从数据下载，词表制作，数据转化，模型训练，所有流程，完全开源开放，可复现。
并训练发布开源最优的模型参数。

接下来将从下面几个方面，详细介绍整个数据制作全流程，从零开始，构建一个预训练模型。

**目录**
* [1. **大规模**中文数据](#大规模中文数据)
* [2. **高精准**中文分词](#高精准中文分词)
* [3. **全字符**中文词表制作](#中文中文词表制作)
* [4. **快速**Token ID 转化](#快速TokenID转化)
* [5. 参考](#参考)


<a name="大规模中文数据"> </a>

## 1. 大规模中文数据

**CLUECorpus2020语料**

CLUECorpus2020 过对Common Crawl的中文部分进行语料清洗得到。开源部分提供了约200G左右的语料文本，详细介绍见[官网](https://github.com/CLUEbenchmark/CLUECorpus2020#%E6%95%B0%E6%8D%AE%E4%B8%8B%E8%BD%BD)，用户可以通过邮件申请下载，方式如下：
> 数据下载
> 申请方式： 将使用语料研究目的和用途，计划、研究机构和申请者介绍，发送到邮箱，并承诺不向第三方提供。
>
> 邮箱: CLUEbenchmark@163.com，标题是：CLUECorpus2020 200G语料库

**WuDaoCorpus2.0 Base 语料**

WuDaoCorpora是悟道爬取的中文大规模语料。整体数量为3TB，目前开源的部分为WuDaoCorpus2.0 bases数据集，大小为200GB。
用户微信登录[官网](https://resource.wudaoai.cn/home)，即可直接下载数据。下载好的压缩数据约 64GB
```
64GB WuDaoCorpus2.0_base_200G.rar
```

<a name="高精准中文分词"> </a>

## 2. 高精准中文分词

ERNIE 使用知识嵌入的方式进行预训练，如何尽可能精确的从原始文本中提取知识，直接关系预训练模型的效果。
目前PaddleNLP常用的分词方式的有`jieba`，`lac`，`Wordtag`，
效果、速度对比表格如下，假设CPU使用40线程，GPU使用16卡，处理200G文本：

| 切词方式 | 效果 | 速度 | 预估耗时
|-|-|-|-|
| jieba | 一般 | 607 KB/s |  2.5 h |
| lac   | 好 | 106 KB/s | 13.9 h
| wordtag| 最好 | 0.94 KB/s | 159 D (GPU)|

综合考虑分词的效果与速度，我们选择百度的LAC作为我们的文本分词工具。

本文档以WuDao数据为例，对数据进行分词：

```shell
python wudao_process.py \
    --input_path WuDaoCorpus2.0_base_200G \
    --workers 40  \
    --ouput_path ./wudao_lac_cut \
```
注：预训练需要实现 SOP( Sentence Order Predict) 任务，在分词的同时，我们使用 简单规则 进行了文本断句。

文本转化完成后。我们使用 `../data_tools/trans_to_json.py`重新转换为jsonl格式（分词完毕）。
```shell
python ../data_tools/trans_to_json.py  \
    --input_path ./wudao_lac_cut \
    --output_path wudao_corpus_200g_0623.jsonl \
    --workers 40 \
    --no-shuffle
```

<a name="全字符中文词表制作"> </a>

## 3. 全字符中文词表制作

词表的制作有两种方案：

第一种，词表组合方案
1. 统计字符
2. 制作英文词表
3. 合并词表

第二种，预处理后直接生成，方案
1. 文本预处理（中文加空格，文本normalize）
2. 使用sentencepeice制作词表

第二种方案需要对文本先使用`BasicTokenizer`切分一遍语料。
第一种方案，自定义程度高，但存在一些局限性。本项目采用了第一种方案，详细介绍如下：

### 分析准备
词表大小： 这里我们考虑的因素主要有两个
- 已有模型对照：
    - ERNIE 3.0系列模型的词表，词表大小为 40000 左右。
- 预训练数据存储占用：
    - 文本token id化后，希望使用uint16表示，此时表示的最大字符为65536。
    - 同时考虑到ERNIE虽然是字模型，我们的仍然需要 `##中` 之类的中文字符表示分词信息。假设使用中文全字符20902(0x4E00, 0x9FA5)个字符，那么剩余 vocab 大小不能超过 44634。

综上，本项目决定采用 40000 左右的 vocab 容量。
其中：
- 中文全字符 `20902`
- 英文字符 `17000`
- 其他字符约 `2000` 左右


### 文本字符统计
首先第一步是对文本字符进行统计。字符统计的目的主要是添加常用的中文字符、特殊字符。

由于语料文本过大，我们随机选取 10G 左右的原始文本进行了字符统计。
```
python gen_char.py path_to_corpus.txt
```
可以在本地文件夹得到`char_dict.pickle`字符频率文件。同时我们也提供了自己统计的词频文件，方便用户复现：
```
wget https://xxx.bos/data/char_dict.pickle
```

### 英文字符词表
基于字符的词频统计，使得英文字符也切割为字母，为此我们需要添加英文词表。
英文部分，我们使用了 [WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)  数据集，来构造词表。
下载解压数据，使用BPE切词
```
wget  https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
python gen_vocab.py ./wikitext-103-raw/wiki.train.raw
```
即可产生英文部分的词表。这里我们也提供了处理好的 vocab 方便用户验证。
```
wget  https://xxx.bos/data/eng.vocab
```


### 合并词表

目前我们得到了字符统计表，和英文字符词表。下一步，我们将词表进行合并。

将`char_dict.pickle`，`eng.vocab`放置到当前目录，使用下面命令
```
python merge_vocab.py
```
即可在 当前 目录生成 vocab.txt 得到最终词表。

此阶段需要注意的一些问题是：
1. 对于一些日文、谚文文字字符，需要进行 normalize
2. 添加special_tokens

### 问题遗留
本项目采用的第一种方式，即拼接产出的词表，对连续非中、英文字符文本，会出现UNK的情况。
如issue: [#2927](https://github.com/PaddlePaddle/PaddleNLP/issues/2927)、 [#2585](https://github.com/PaddlePaddle/PaddleNLP/issues/2585)。本项目做了两点改进:

1. 对 Symbol 字符默认添加空格，变成独立字符
2. 对 日文、谚文 在合并词表阶段默认添加 ## 字符。

虽然有上述两点修复，任然无法避免 [#2927](https://github.com/PaddlePaddle/PaddleNLP/issues/2927) 现象。
彻底解决的话，建议使用第二种方式制作vocab文件。

### 方案二：预处理后直接生成
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
对处理好的vocab文件手动替换一些`<pad> -> [PAD]`之类的special_tokens，即可产出词表。


<a name="快速TokenID转化"> </a>

## 4. 快速Token ID 转化

预料、词表准备妥当后，我们可以开始进行最后的数据ID转化。

- 高效的 Multiprocessing 多进程实现
- 使用内存BytesIO存储ID数据

由于转换的逻辑复杂，需要定义`class Converter`对象来进行转化处理。如果每次处理新的文本，都实例化一次class对象，速度瓶颈会在处理函数的实例化。
我们使用了提前multiprocessing.Pool的`initializer`，对处理函数进行提前实例化，提高处理效率。

处理后的token id数量巨大，可以达到数百Billion，如果使用普通的数据结构，如python的list保存，会出现存储瓶颈，不仅占用空间大，list对象还需要重新分配内存空间。这里我们采用了 BytesIO 的方式，类似写入内存文件的方式，速度快，可以非常方便转化为numpy文件保存。

使用 Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz CPU测试，40线程，处理速度 8+MB/s，约7个小时左右，即可完成 200GB 文本转化为ID.

```
python -u  ../data_tools/create_pretraining_data.py \
    --model_name ./vocab_path/vocab.txt \
    --tokenizer_name ErnieTokenizer \
    --input_path wudao_corpus_200g_0623.jsonl \
    --split_sentences\
    --chinese \
    --cn_splited \
    --cn_whole_word_segment \
    --output_prefix wudao_200g_0703 \
    --workers 40 \
    --log_interval 1000
```
转化后的数据如下，使用这份数据，即可开始ERNIE预训练
```
-rw-rw-r-- 1 500 501 129G Jul  4 03:39 wudao_200g_0703_ids.npy
-rw-rw-r-- 1 500 501 6.4G Jul  4 03:39 wudao_200g_0703_idx.npz
```

## 5. 参考
感谢CLUE，WuDao提供的开源文本语料，参考资料：
- Xu, L., Zhang, X. and Dong, Q., 2020. CLUECorpus2020: A large-scale Chinese corpus for pre-training language model. arXiv preprint arXiv:2003.01355.
- Yuan, S., Zhao, H., Du, Z., Ding, M., Liu, X., Cen, Y., Zou, X., Yang, Z. and Tang, J., 2021. Wudaocorpora: A super large-scale chinese corpora for pre-training language models. AI Open, 2, pp.65-68.
- https://github.com/CLUEbenchmark/CLUECorpus2020
- https://resource.wudaoai.cn
