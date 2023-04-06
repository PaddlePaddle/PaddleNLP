# ErnieFastTokenizer分词示例

FastTokenizer库在C++、Python端提供ErnieFastTokenizer接口，用户只需传入模型相应的词表即可调用该接口，完成高效分词操作。该接口底层使用`WordPiece`算法进行分词。针对`WordPiece`算法，FastTokenizer实现了"Fast WordPiece Tokenization"提出的基于`MinMaxMatch`的`FastWordPiece`算法。原有`WordPiece`算法的时间复杂度与序列长度为二次方关系，在对长文本进行分词操作时，时间开销比较大。而`FastWordPiece`算法通过`Aho–Corasick `算法将`WordPiece`算法的时间复杂度降低为与序列长度的线性关系，大大提升了分词效率。`ErnieFastTokenizer`除了支持ERNIE模型的分词以外，还支持其他基于`WordPiece`算法分词的模型，比如`BERT`, `TinyBert`等，详细的模型列表如下：

## 支持的模型列表

- ERNIE
- BERT
- TinyBERT
- ERNIE Gram
- ERNIE ViL

## 详细分词示例文档

[C++ 分词示例](./cpp/README.md)

[Python 分词示例](./python/README.md)

## 参考文献

- Xinying Song, Alex Salcianuet al. "Fast WordPiece Tokenization", EMNLP, 2021
