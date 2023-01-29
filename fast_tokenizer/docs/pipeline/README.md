# FastTokenizer Pipeline

当我们使用 Tokenizer 的`Tokenizer.encode` 或者 `Tokenizer.encode_batch` 方法进行分词时，会经历如下四个阶段：Normalize、PreTokenize、 Model 以及 PostProcess。针对这四个阶段，FastTokenizer 提供 Normalizer、PreTokenizer、Model以及PostProcessor四个组件分别完成四个阶段所需要的工作。下面将详细介绍四大组件具体负责的工作。

## Normalizer

Normalizer 组件主要用于将原始字符串标准化，输出标准化的字符串，常见的标准化字符串操作有大小写转换、半角全角转换等。

## PreTokenizer

PreTokenizer 组件主要使用简单的分词方法，将标准化的字符串进行预切词，得到较大粒度的词组，例如按照标点、空格方式进行分词。

## Model

Model 组件是 FastTokenizer 核心模块，用于将粗粒度词组按照一定的算法进行切分，得到细粒度的Token，目前支持的切词算法包括 FastWordPiece、WordPiece、BPE 以及 Unigram。

## PostProcessor

PostProcess 组件为后处理，主要执行 Transformer 类模型的文本序列的处理，比如添加 [SEP] 等特殊 Token。
