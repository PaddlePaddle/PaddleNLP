# ERNIE 数据制作全流程


## 数据下载

### CLUECorpus2020  & WuDaoCorpus2.0 Base 数据集

CLUE 提供了约200G左右的语料文本，详细介绍见[官网](https://github.com/CLUEbenchmark/CLUECorpus2020#%E6%95%B0%E6%8D%AE%E4%B8%8B%E8%BD%BD)，用户可以申请下载，方式如下：
> 数据下载
> 申请方式： 将使用语料研究目的和用途，计划、研究机构和申请者介绍，发送到邮箱，并承诺不向第三方提供。
>
> 邮箱: CLUEbenchmark@163.com，标题是：CLUECorpus2020 200G语料库


WuDaoCorpora是悟道爬取的中文大规模预料。整体数量为3TB，目前开源的部分为WuDaoCorpus2.0 bases数据集，大小为200GB。
用户微信登录[官网](https://resource.wudaoai.cn/home)，即可直接下载数据。

### 中文预料分词

ERNIE 使用知识嵌入的方式进行预训练，如何尽可能精确的从原始文本中提取知识，直接关系预训练模型的效果。
目前采用的分词方式的有jieba，lac，Wordtag，效果以此
速度对比，假设CPU使用40线程，GPU使用16卡：

| 切词方式 | 效果 | 速度 | 耗时（处理200G）
|-|-|-|-|
| jieba | 一般 | 607 KB/s |  2.5 h |
| lac   | 好 | 106 KB/s | 13.9 h
| wordtag| 最好 | 0.94 KB/s | 159D|

综合考虑分词的效果与速度，我们选择百度的LAC作为我们的分词工具。




## 词表制作

1. 统计字符
2. 制作英文词表
3. 合并词表

注：此方法拼接产出的词表容易出现UNK的情况。
如issue[2927](https://github.com/PaddlePaddle/PaddleNLP/issues/2927)

### 英文部分，下载了 WikiText 数据
