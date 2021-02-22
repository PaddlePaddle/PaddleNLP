# PaddleNLP Embedding API

## 介绍

PaddleNLP提供多个开源的预训练词向量模型，用户仅需在使用`paddlenlp.embeddings.TokenEmbedding`时，指定预训练模型的名称，即可加载相对应的预训练模型。以下为PaddleNLP所支持的预训练Embedding模型，其名称用作`paddlenlp.embeddings.TokenEmbedding`的参数。
- 命名方式为：\${训练模型}.\${语料}.\${词向量类型}.\${co-occurrence type}.dim\${维度}。
- 模型有三种，分别是Word2Vec(w2v, skip-gram), GloVe(glove)和FastText(fasttext)。

在[使用方式](#使用方式)这一节中，将介绍如何通过模型名称使用`paddlenlp.embeddings.TokenEmbedding`加载预训练模型。


## 中文词向量

以下预训练词向量由[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)提供。

根据不同类型的上下文为每个语料训练多个目标词向量，第二列开始表示不同类型的上下文。以下为上下文类别：

* Word表示训练时目标词预测的上下文是一个Word。
* Word + N-gram表示训练时目标词预测的上下文是一个Word或者Ngram，其中bigram表示2-grams，ngram.1-2表示1-gram或者2-grams。
* Word + Character表示训练时目标词预测的上下文是一个Word或者Character，其中word-character.char1-2表示上下文是1个或2个Character。
* Word + Character + Ngram表示训练时目标词预测的上下文是一个Word、Character或者Ngram。bigram-char表示上下文是2-grams或者1个Character。

| 语料 | Word | Word + N-gram | Word + Character | Word + Character + N-gram |
| ------------------------------------------- | ----   | ---- | ----   | ---- |
| Baidu Encyclopedia 百度百科                 | w2v.baidu_encyclopedia.target.word-word.dim300 | w2v.baidu_encyclopedia.target.word-ngram.1-2.dim300 | w2v.baidu_encyclopedia.target.word-character.char1-2.dim300 | w2v.baidu_encyclopedia.target.bigram-char.dim300 |
| Wikipedia_zh 中文维基百科                   | w2v.wiki.target.word-word.dim300 | w2v.wiki.target.word-bigram.dim300 | w2v.wiki.target.word-char.dim300 | w2v.wiki.target.bigram-char.dim300 |
| People's Daily News 人民日报                | w2v.people_daily.target.word-word.dim300 | w2v.people_daily.target.word-bigram.dim300 | w2v.people_daily.target.word-char.dim300 | w2v.people_daily.target.bigram-char.dim300 |
| Sogou News 搜狗新闻                         | w2v.sogou.target.word-word.dim300 | w2v.sogou.target.word-bigram.dim300 | w2v.sogou.target.word-char.dim300 | w2v.sogou.target.bigram-char.dim300 |
| Financial News 金融新闻                     | w2v.financial.target.word-word.dim300 | w2v.financial.target.word-bigram.dim300 | w2v.financial.target.word-char.dim300 | w2v.financial.target.bigram-char.dim300 |
| Zhihu_QA 知乎问答                           | w2v.zhihu.target.word-word.dim300 | w2v.zhihu.target.word-bigram.dim300 | w2v.zhihu.target.word-char.dim300 | w2v.zhihu.target.bigram-char.dim300 |
| Weibo 微博                                  | w2v.weibo.target.word-word.dim300 | w2v.weibo.target.word-bigram.dim300 | w2v.weibo.target.word-char.dim300 | w2v.weibo.target.bigram-char.dim300 |
| Literature 文学作品                         | w2v.literature.target.word-word.dim300 | w2v.literature.target.word-bigram.dim300 | w2v.literature.target.word-char.dim300 | w2v.literature.target.bigram-char.dim300 |
| Complete Library in Four Sections 四库全书  | w2v.sikuquanshu.target.word-word.dim300 | w2v.sikuquanshu.target.word-bigram.dim300 | 无 | 无 |
| Mixed-large 综合                            | w2v.mixed-large.target.word-word.dim300 | 暂无 | w2v.mixed-large.target.word-word.dim300 | 暂无 |

特别地，对于百度百科语料，在不同的 Co-occurrence类型下分别提供了目标词与上下文向量：

| Co-occurrence 类型          | 目标词向量 | 上下文词向量  |
| --------------------------- | ------   | ---- |
|    Word → Word              | w2v.baidu_encyclopedia.target.word-word.dim300     |   w2v.baidu_encyclopedia.context.word-word.dim300    |
|    Word → Ngram (1-2)       |  w2v.baidu_encyclopedia.target.word-ngram.1-2.dim300    |   w2v.baidu_encyclopedia.context.word-ngram.1-2.dim300    |
|    Word → Ngram (1-3)       |  w2v.baidu_encyclopedia.target.word-ngram.1-3.dim300    |   w2v.baidu_encyclopedia.context.word-ngram.1-3.dim300    |
|    Ngram (1-2) → Ngram (1-2)|  w2v.baidu_encyclopedia.target.word-ngram.2-2.dim300   |   w2v.baidu_encyclopedia.target.word-ngram.2-2.dim300    |
|    Word → Character (1)     |  w2v.baidu_encyclopedia.target.word-character.char1-1.dim300    |  w2v.baidu_encyclopedia.context.word-character.char1-1.dim300     |
|    Word → Character (1-2)   |  w2v.baidu_encyclopedia.target.word-character.char1-2.dim300    |  w2v.baidu_encyclopedia.context.word-character.char1-2.dim300     |
|    Word → Character (1-4)   |  w2v.baidu_encyclopedia.target.word-character.char1-4.dim300    |  w2v.baidu_encyclopedia.context.word-character.char1-4.dim300     |
|    Word → Word (left/right) |   w2v.baidu_encyclopedia.target.word-wordLR.dim300   |   w2v.baidu_encyclopedia.context.word-wordLR.dim300    |
|    Word → Word (distance)   |   w2v.baidu_encyclopedia.target.word-wordPosition.dim300   |   w2v.baidu_encyclopedia.context.word-wordPosition.dim300    |

## 英文词向量

### Word2Vec

| 语料 | 名称 |
|------|------|
| Google News | w2v.google_news.target.word-word.dim300.en |

### GloVe

| 语料                | 25维     | 50维      | 100维    | 200维    | 300 维   |
| -----------------   | ------   |  ------   | ------   | ------   | ------   |
| Wiki2014 + GigaWord | 无 | glove.wiki2014-gigaword.target.word-word.dim50.en | glove.wiki2014-gigaword.target.word-word.dim100.en | glove.wiki2014-gigaword.target.word-word.dim200.en | glove.wiki2014-gigaword.target.word-word.dim300.en |
| Twitter             | glove.twitter.target.word-word.dim25.en | glove.twitter.target.word-word.dim50.en | glove.twitter.target.word-word.dim100.en | glove.twitter.target.word-word.dim200.en | 无 |

### FastText

| 语料 | 名称 |
|------|------|
| Wiki2017 | fasttext.wiki-news.target.word-word.dim300.en |
| Crawl    | fasttext.crawl.target.word-word.dim300.en |

## 使用方式

以上所述的模型名称可直接以参数形式传入`padddlenlp.embeddings.TokenEmbedding`，加载相对应的模型。比如要加载语料为Wiki2017，通过FastText训练的预训练模型（`fasttext.wiki-news.target.word-word.dim300.en`），只需执行以下代码：

```python
import paddle
from paddlenlp.embeddings import TokenEmbedding

token_embedding = TokenEmbedding(embedding_name="fasttext.wiki-news.target.word-word.dim300.en")
```

## 模型信息

| 模型 | 文件大小 | 词表大小 |
|-----|---------|---------|
| w2v.baidu_encyclopedia.target.word-word.dim300                         | 678.21 MB  | 635965 |
| w2v.baidu_encyclopedia.target.word-character.char1-1.dim300            | 679.15 MB  | 636038 |
| w2v.baidu_encyclopedia.target.word-character.char1-2.dim300            | 679.30 MB  | 636038 |
| w2v.baidu_encyclopedia.target.word-character.char1-4.dim300            | 679.51 MB  | 636038 |
| w2v.baidu_encyclopedia.target.word-ngram.1-2.dim300                    | 679.48 MB  | 635977 |
| w2v.baidu_encyclopedia.target.word-ngram.1-3.dim300                    | 671.27 MB  | 628669 |
| w2v.baidu_encyclopedia.target.word-ngram.2-2.dim300                    | 7.28 GB    | 6969069 |
| w2v.baidu_encyclopedia.target.word-wordLR.dim300                       | 678.22 MB  | 635958 |
| w2v.baidu_encyclopedia.target.word-wordPosition.dim300                 | 679.32 MB  | 636038 |
| w2v.baidu_encyclopedia.target.bigram-char.dim300                       | 679.29 MB  | 635976 |
| w2v.baidu_encyclopedia.context.word-word.dim300                        | 677.74 MB  | 635952 |
| w2v.baidu_encyclopedia.context.word-character.char1-1.dim300           | 678.65 MB  | 636200 |
| w2v.baidu_encyclopedia.context.word-character.char1-2.dim300           | 844.23 MB  | 792631 |
| w2v.baidu_encyclopedia.context.word-character.char1-4.dim300           | 1.16 GB    | 1117461 |
| w2v.baidu_encyclopedia.context.word-ngram.1-2.dim300                   | 7.25 GB    | 6967598 |
| w2v.baidu_encyclopedia.context.word-ngram.1-3.dim300                   | 5.21 GB    | 5000001 |
| w2v.baidu_encyclopedia.context.word-ngram.2-2.dim300                   | 7.26 GB    | 6968998 |
| w2v.baidu_encyclopedia.context.word-wordLR.dim300                      | 1.32 GB    | 1271031 |
| w2v.baidu_encyclopedia.context.word-wordPosition.dim300                | 6.47 GB    | 6293920 |
| w2v.wiki.target.bigram-char.dim300                                     | 375.98 MB  | 352274 |
| w2v.wiki.target.word-char.dim300                                       | 375.52 MB  | 352223 |
| w2v.wiki.target.word-word.dim300                                       | 374.95 MB  | 352219 |
| w2v.wiki.target.word-bigram.dim300                                     | 375.72 MB  | 352219 |
| w2v.people_daily.target.bigram-char.dim300                             | 379.96 MB  | 356055 |
| w2v.people_daily.target.word-char.dim300                               | 379.45 MB  | 355998 |
| w2v.people_daily.target.word-word.dim300                               | 378.93 MB  | 355989 |
| w2v.people_daily.target.word-bigram.dim300                             | 379.68 MB  | 355991 |
| w2v.weibo.target.bigram-char.dim300                                    | 208.24 MB  | 195199 |
| w2v.weibo.target.word-char.dim300                                      | 208.03 MB  | 195204 |
| w2v.weibo.target.word-word.dim300                                      | 207.94 MB  | 195204 |
| w2v.weibo.target.word-bigram.dim300                                    | 208.19 MB  | 195204 |
| w2v.sogou.target.bigram-char.dim300                                    | 389.81 MB  | 365112 |
| w2v.sogou.target.word-char.dim300                                      | 389.89 MB  | 365078 |
| w2v.sogou.target.word-word.dim300                                      | 388.66 MB  | 364992 |
| w2v.sogou.target.word-bigram.dim300                                    | 388.66 MB  | 364994 |
| w2v.zhihu.target.bigram-char.dim300                                    | 277.35 MB  | 259755 |
| w2v.zhihu.target.word-char.dim300                                      | 277.40 MB  | 259940 |
| w2v.zhihu.target.word-word.dim300                                      | 276.98 MB  | 259871 |
| w2v.zhihu.target.word-bigram.dim300                                    | 277.53 MB  | 259885 |
| w2v.financial.target.bigram-char.dim300                                | 499.52 MB  | 467163 |
| w2v.financial.target.word-char.dim300                                  | 499.17 MB  | 467343 |
| w2v.financial.target.word-word.dim300                                  | 498.94 MB  | 467324 |
| w2v.financial.target.word-bigram.dim300                                | 499.54 MB  | 467331 |
| w2v.literature.target.bigram-char.dim300                               | 200.69 MB  | 187975 |
| w2v.literature.target.word-char.dim300                                 | 200.44 MB  | 187980 |
| w2v.literature.target.word-word.dim300                                 | 200.28 MB  | 187961 |
| w2v.literature.target.word-bigram.dim300                               | 200.59 MB  | 187962 |
| w2v.sikuquanshu.target.word-word.dim300                                | 20.70 MB   | 19529 |
| w2v.sikuquanshu.target.word-bigram.dim300                              | 20.77 MB   | 19529 |
| w2v.mixed-large.target.word-char.dim300                                | 1.35 GB    | 1292552 |
| w2v.mixed-large.target.word-word.dim300                                | 1.35 GB    | 1292483 |
| w2v.google_news.target.word-word.dim300.en                             | 1.61 GB    | 3000000 |
| glove.wiki2014-gigaword.target.word-word.dim50.en                      | 73.45 MB   | 400002 |
| glove.wiki2014-gigaword.target.word-word.dim100.en                     | 143.30 MB  | 400002 |
| glove.wiki2014-gigaword.target.word-word.dim200.en                     | 282.97 MB  | 400002 |
| glove.wiki2014-gigaword.target.word-word.dim300.en                     | 422.83 MB  | 400002 |
| glove.twitter.target.word-word.dim25.en                                | 116.92 MB  | 1193516 |
| glove.twitter.target.word-word.dim50.en                                | 221.64 MB  | 1193516 |
| glove.twitter.target.word-word.dim100.en                               | 431.08 MB  | 1193516 |
| glove.twitter.target.word-word.dim200.en                               | 848.56 MB  | 1193516 |
| fasttext.wiki-news.target.word-word.dim300.en                          | 541.63 MB  | 999996 |
| fasttext.crawl.target.word-word.dim300.en                              | 1.19 GB    | 2000002 |

## 致谢
- 感谢 [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)提供Word2Vec中文预训练词向量。
- 感谢 [GloVe Project](https://nlp.stanford.edu/projects/glove)提供的GloVe英文预训练词向量。
- 感谢 [FastText Project](https://fasttext.cc/docs/en/english-vectors.html)提供的英文预训练词向量。

## 参考论文
- Li, Shen, et al. "Analogical reasoning on chinese morphological and semantic relations." arXiv preprint arXiv:1805.06504 (2018).
- Qiu, Yuanyuan, et al. "Revisiting correlations between intrinsic and extrinsic evaluations of word embeddings." Chinese Computational Linguistics and Natural Language Processing Based on Naturally Annotated Big Data. Springer, Cham, 2018. 209-221.
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.
- T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, A. Joulin. Advances in Pre-Training Distributed Word Representations.
