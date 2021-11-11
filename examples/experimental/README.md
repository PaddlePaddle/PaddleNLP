
# 高性能文本处理——FasterTokenizer

## 概览

近几年NLP Transformer类的模型发展迅速，各个NLP的基础技术和核心应用的核心技术基本上都被Transformer类的核心技术所替换。
学术上，目前各个NLP任务领域的SOTA效果基本都是由Transformer类刷新，但在落地应用上还面临上线困难的问题。
Transformer类文本预处理部分，主要存在以下两个因素影响Transformer训推一体的部署：

* 文本预处理部分复杂，不具备训推一体的部署体验，C++端需要重新开发，成本高。
  训练侧多为Python实现，C++部署时需要重新进行迁移与对齐，目前业界缺少标准高效的C++参考实现
* 文本预处理效率Python实现与C++实现存在数量级上的差距，对产业实践场景有较大的价值。
  在服务部署时Tokenizer的性能也是推动NLP相关模型的一个性能瓶颈，尤其是小型化模型部署如ERNIE-Tiny等，文本预处理耗时占总体预测时间高达30%。


基于以上两点原因，我们将Transformer类文本预处理部分内置成Paddle底层算子——FasterTokenizer。
FasterTokenizer底层为C++实现，同时提供了python接口调用。其可以将文本转化为模型数值化输入。
同时，用户可以将其导出为模型的一部分，直接用于部署推理。从而实现Transformer训推一体。



## FasterTokenizer 切词

目前，PaddleNLP 2.2版本提供了FasterTokenizer python API接口。


   模型名                           |  FasterTokenizer
---------------------------------- | :------:
ERNIE, Chinese                     | `FasterTokenizer.from_pretrained("ernie-1.0")`
ERNIE 2.0 Base, English            | `FasterTokenizer.from_pretrained("ernie-2.0-en")`
ERNIE 2.0 Large, English           | `FasterTokenizer.from_pretrained("ernie-2.9-large-en")`
BERT-Base, Uncased                 | `FasterTokenizer.from_pretrained("bert-base-uncased")`
BERT-Large, Uncased                | `FasterTokenizer.from_pretrained("bert-large-uncased")`
BERT-Base, Cased                   | `FasterTokenizer.from_pretrained("bert-base-cased")`
BERT-Large, Cased                  | `FasterTokenizer.from_pretrained("bert-large-cased")`
BERT-Base, Multilingual Cased      | `FasterTokenizer.from_pretrained("bert-base-multilingual-cased")`
BERT-Base, Chinese                 | `FasterTokenizer.from_pretrained("bert-base-chinese")`
BERT-WWM, Chinese                  | `FasterTokenizer.from_pretrained("bert-wwm-chinese")`
BERT-WWM-Ext, Chinese              | `FasterTokenizer.from_pretrained("bert-wwm-ext-chinese")`
RoBERTa-WWM-Ext, Chinese           | `FasterTokenizer.from_pretrained("roberta-wwm-ext")`
RoBERTa-WWM-Ext-Large, Chinese     | `FasterTokenizer.from_pretrained("roberta-wwm-ext-large")`



使用方式如下：

```python
tokenizer = FasterTokenizer.from_pretrained("ernie-1.0")
text = '小说是文学的一种样式，一般描写人物故事，塑造多种多样的人物形象，但亦有例外。'
input_ids, token_type_ids = tokenizer([text], max_seq_len=50)
```

同时通过对比FasterTokenizer、[HuggingFace Tokenizer(use_fast=True)](https://github.com/huggingface/tokenizers)以及 [Tensorflow Text BertTokenizer](https://www.tensorflow.org/text/api_docs/python/text/BertTokenizer)的性能。
实验结果发现FasterTokenizer性能远远超过其他Tokenizer， 并且高达HuggingFace Tokenizer(use_fast=True)性能20倍。详细参考[性能测试脚本](./faster_tokenizer/perf.py)。

## FasterERNIE

如上所述，FasterTokenizer可以直接作为模型的一部分，即可以直接输入文本到模型就可以完成模型训练。基于FasterTokenizer，
paddlenlp 2.2版本实现了`FasterErnieModel`。

示例中提供了FasterERNIE用于[文本分类](./faster_ernie/seq_cls)和[序列标注](./faster_ernie/token_cls)任务。
