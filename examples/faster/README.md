# Faster PaddleNLP：更快更易用的文本领域全流程开发

PaddleNLP 2.2版本提供了文本领域全流程更快的开发体验，包含了以下三大组件：

* **FasterTokenizer**： 极致优化的高性能文本处理算子
* **FasterERNIE**：预训练模型的全流程优化
* **FasterGeneration**：高性能生成任务加速

## FasterTokenizer：高性能文本处理

近几年NLP Transformer类的模型发展迅速，各个NLP的基础技术和核心应用的核心技术基本上都被Transformer类的核心技术所替换。
学术上，目前各个NLP任务领域的SOTA效果基本都是由Transformer类刷新，但在落地应用上还面临上线困难的问题。
Transformer类文本预处理部分，主要存在以下两个因素影响Transformer训推一体的部署：

* 文本预处理部分逻辑复杂，C++端需要重新开发，成本高。
  训练侧多为Python实现，进行产业级部署时需要重新进行迁移与对齐，目前业界缺少通用且高效的C++参考实现。
* 文本预处理效率Python实现与C++实现存在数量级上的差距，对产业实践场景有较大的价值。
  在服务部署时Tokenizer的性能也是推动NLP相关模型的一个性能瓶颈，尤其是小型化模型部署如ERNIE-Tiny等，文本预处理耗时占总体预测时间高达30%。

基于以上两点原因，我们将常用预训练模型的文本预处理部分内置成Paddle底层算子——FasterTokenizer。
FasterTokenizer底层为C++实现，同时提供了python接口调用。其可以将文本转化为模型数值化输入。
同时，用户可以将其导出为模型的一部分，直接用于部署推理。实现了预训练模型包含高性能文本处理训推一体开发体验。

### Usage

目前，PaddleNLP 2.2版本提供了FasterTokenizer python API接口。

支持模型                                                  |  FasterTokenizer API Usage
-------------------------------------------------------- | :------:
ERNIE, Chinese                                           | `FasterTokenizer.from_pretrained("ernie-1.0")`
ERNIE 2.0 Base, English                                  | `FasterTokenizer.from_pretrained("ernie-2.0-en")`
ERNIE 2.0 Large, English                                 | `FasterTokenizer.from_pretrained("ernie-2.0-large-en")`
BERT-Base, Uncased                                       | `FasterTokenizer.from_pretrained("bert-base-uncased")`
BERT-Large, Uncased                                      | `FasterTokenizer.from_pretrained("bert-large-uncased")`
BERT-base, Cased                                         | `FasterTokenizer.from_pretrained("bert-base-cased")`
BERT-Large, Cased                                        | `FasterTokenizer.from_pretrained("bert-large-cased")`
BERT-Base, Multilingual Cased                            | `FasterTokenizer.from_pretrained("bert-base-multilingual-cased")`
BERT-Base, Chinese                                       | `FasterTokenizer.from_pretrained("bert-base-chinese")`
BERT-Base (Whole Word Masking), Chinese                  | `FasterTokenizer.from_pretrained("bert-wwm-chinese")`
BERT-Base ((Whole Word Masking, EXT Data), Chinese       | `FasterTokenizer.from_pretrained("bert-wwm-ext-chinese")`
RoBERTa-Base (Whole Word Masking, EXT Data), Chinese     | `FasterTokenizer.from_pretrained("roberta-wwm-ext")`
RoBERTa-Large (Whole Word Masking, EXT Data), Chinese    | `FasterTokenizer.from_pretrained("roberta-wwm-ext-large")`


使用方式如下：

```python
from paddlenlp.experimental import FasterTokenizer
from paddlenlp.experimental import to_tensor
tokenizer = FasterTokenizer.from_pretrained("ernie-1.0")
text = '小说是文学的一种样式，一般描写人物故事，塑造多种多样的人物形象，但亦有例外。'
input_ids, token_type_ids = tokenizer([text], max_seq_len=50)
```

### Performance

通过对比Paddle FasterTokenizer、[HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)以及 [TensorFlow Text BertTokenizer](https://www.tensorflow.org/text/api_docs/python/text/BertTokenizer)的性能。
实验结果发现FasterTokenizer性能远远超过其他同类Tokenizer实现，更多详细实验记录请参考[性能测试脚本perf.py](./faster_tokenizer/perf.py)。

TODO: (Add Perf figure)

## FasterERNIE：预训练模型的全流程优化

FasterERNIE内置了FasterTokenizer的实现，使模型计算图包含了高性能文本处理算子，在文本领域任务上提供了更加简洁易用的训推一体开发体验，同时Python部署具备更快的推理性能。
基于Paddle 2.2的Fused TransformerEncoder API功能，可以在NVDIA GPU上提供更快的训练与推理优化。
综合FasterTokenizer与Fuse TransformerEncoder API，PaddleNLP 2.2版本提供了`FasterErnieModel`实现，包含了从文本处理、极致优化的训练与推理性能，且产业级部署代码更简洁的开发示例：

* [文本分类：FasterERNIEForSequenceClassification](./faster_ernie/seq_cls)
* [序列标注：FasterERNIEForTokenizerClassification](./faster_ernie/token_cls)

TODO(add training and inference perf data)

**NOTE**: FasterERNIE当前处于实验阶段，不排除后续会出现API变更。

## FasterGeneration：高性能生成任务加速

[FasterGeneration](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/faster/faster_generation)是PaddleNLP v2.2版本加入的一个高性能推理功能，可实现基于CUDA的序列解码。该功能可以用于多种生成类的预训练NLP模型，例如GPT、BART、UnifiedTransformer等，并且支持多种解码策略。因此该功能主要适用于机器翻译，文本续写，文本摘要，对话生成等任务。

功能底层依托于[FasterTransformer](https://github.com/NVIDIA/FasterTransformer)，该库专门针对Transformer系列模型及各种解码策略进行了优化。功能顶层封装于`model.generate`函数。功能的开启和关闭通过传入`use_faster`参数进行控制（默认为关闭状态）。通过调用generate函数，用户可以简单实现模型的高性能推理功能。
