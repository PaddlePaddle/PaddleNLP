# FasterTokenizer

------------------------------------------------------------------------------------------

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleNLP?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.6.2+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleNLP?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/PaddleNLP?color=3af"></a>
    <a href="https://pypi.org/project/paddlenlp/"><img src="https://img.shields.io/pypi/dm/paddlenlp?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleNLP?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleNLP?color=ccf"></a>
</p>
FasterTokenizer是一款简单易用、功能强大的跨平台高性能文本预处理库，集成业界多个常用的Tokenizer实现，支持不同NLP场景下的文本预处理功能，如文本分类、阅读理解，序列标注等。结合PaddleNLP Tokenizer模块，为用户在训练、推理阶段提供高效通用的文本预处理能力。

## 特性

- 高性能。由于底层采用C++实现，所以其性能远高于目前常规Python实现的Tokenizer。在文本分类任务上，FasterTokenizer对比Python版本Tokenizer加速比最高可达20倍。
- 跨平台。FasterTokenizer可在不同的系统平台上使用，目前已支持Windows x64，Linux x64以及MacOS 10.14+平台上使用。
- 多编程语言支持。FasterTokenizer提供在C++、Python语言上开发的能力。
- 灵活性强。用户可以通过指定不同的FasterTokenizer组件定制满足需求的Tokenizer。

## 快速开始

下面将介绍Python版本FasterTokenizer的使用方式，C++版本的使用方式可参考[FasterTokenizer C++ Demo](./faster_tokenizer/demo/README.md)。

### 前置依赖

- Windows 64位系统
- Linux x64系统
- MacOS 10.14+系统（m1芯片的MacOS，需要使用x86_64版本的Anaconda作为python环境方可安装使用）
- Python 3.6 ~ 3.9

### 安装FasterTokenizer

```python
pip install faster_tokenizer
```

### FasterTokenizer使用示例

- 准备词表

```shell
# Linux或者Mac用户可直接执行以下命令下载测试的词表，Windows 用户可在浏览器上下载到本地。
wget https://bj.bcebos.com/paddlenlp/models/transformers/ernie/vocab.txt
```

- 切词示例

FasterTokenizer库内置NLP任务常用的Tokenizer，如ErnieFasterTokenizer。下面将展示FasterTokenizer的简单用法。

```python
from faster_tokenizer import ErnieFasterTokenizer, models
# 1. 加载词表
vocab = models.WordPiece.read_file("ernie_vocab.txt")
# 2. 实例化ErnieFasterTokenizer对象
faster_tokenizer = ErnieFasterTokenizer(vocab)
# 3. 切词
output = faster_tokenizer.encode("我爱中国")
# 4. 输出结果
print("ids: ", output.ids)
print("type_ids: ", output.type_ids)
print("tokens: ", output.tokens)
print("offsets: ", output.offsets)
print("attention_mask: ", output.attention_mask)
```

### FasterTokenizer在PaddleNLP Tokenizer模块加速示例

PaddleNLP Tokenizer模块可简单地应用在模型训练以及推理部署的文本预处理阶段，并通过`AutoTokenizer.from_pretrained`方式实例化相应的Tokenizer。其中`AutoTokenizer`默认加载得到的Tokenizer是常规Python实现的Tokenizer，其性能会低于C++实现的FasterTokenizer。为了提升PaddleNLP Tokenizer模块性能，目前PaddleNLP Tokenizer模块已经支持使用FasterTokenizer作为Tokenizer的后端加速切词阶段。在现有的Tokenizer加载接口中，仅需添加`use_faster=True`这一关键词参数，其余代码保持不变，即可加载Faster版本的Tokenizer，代码示例如下：

```python
from paddlenlp.transformers import AutoTokenizer

# 默认加载Python版本的Tokenizer
tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')
# 打开use_faster开关，可加载Faster版本Tokenizer
faster_tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh', use_faster=True)

text1 = tokenizer('自然语言处理')
text2 = faster_tokenizer('自然语言处理')

print(text1)
print(text2)
```

目前PaddleNLP已支持BERT、ERNIE、TinyBERT以及ERNIE-M 4种Tokenizer的Faster版本，其余模型的Tokenizer暂不支持Faster版本。

## FAQ

Q：我在AutoTokenizer.from_pretrained接口上已经打开`use_faster=True`开关，为什么文本预处理阶段性能上好像没有任何变化？

A：在有三种情况下，打开`use_faster=True`开关可能无法提升性能：
  1. 没有安装faster_tokenizer。若在没有安装faster_tokenizer库的情况下打开`use_faster`开关，PaddleNLP会给出以下warning："Can't find the faster_tokenizer package, please ensure install faster_tokenizer correctly. "。

  2. 加载的Tokenizer类型暂不支持Faster版本。目前支持4种Tokenizer的Faster版本，分别是BERT、ERNIE、TinyBERT以及ERNIE-M Tokenizer。若加载不支持Faster版本的Tokenizer情况下打开`use_faster`开关，PaddleNLP会给出以下warning："The tokenizer XXX doesn't have the faster version. Please check the map paddlenlp.transformers.auto.tokenizer.FASTER_TOKENIZER_MAPPING_NAMES to see which faster tokenizers are currently supported."

  3. 待切词文本长度过短（如文本平均长度小于5）。这种情况下切词开销可能不是整个文本预处理的性能瓶颈，导致在使用FasterTokenizer后仍无法提升整体性能。

## 相关文档

[FasterTokenizer编译指南](docs/compile/README.md)
