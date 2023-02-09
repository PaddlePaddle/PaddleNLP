
# ⚡ FastTokenizer：高性能文本处理库

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


FastTokenizer 是一款简单易用、功能强大的跨平台高性能文本预处理库，集成业界多个常用的 Tokenizer 实现，支持不同 NLP 场景下的文本预处理功能，如文本分类、阅读理解，序列标注等。在 Python 端结合 PaddleNLP Tokenizer 模块，为用户在训练、推理阶段提供高效通用的文本预处理能力。

![fast-tokenizer-framework](https://user-images.githubusercontent.com/10826371/215277623-1fcd84e5-1cc7-43a8-a33c-890103cf1cc8.png)

## 特性

- 高性能。底层采用 C++ 实现，并且集成高性能分词算法 `FastWordPiece` [1]，其性能远高于常规 Python 实现的 Tokenizer。在文本分类任务上，FastTokenizer 对比 Python 版本 Tokenizer 加速比最高可达20倍。支持多线程加速多文本批处理分词。默认使用单线程分词。
- 可拓展性强。用户可以通过指定不同的 `Normalizer`, `PreTokenizer`, `Model` 以及 `PostProcessor` 组件自定义 Tokenizer。可在 [FastTokenizer Pipeline](docs/pipeline/README.md) 文档了解更多关于组件的介绍以及使用方式。
- 跨平台。FastTokenizer 可在不同的系统平台上使用，目前已支持 Windows x64，Linux x64 以及 MacOS 10.14+ 平台上使用。
- 支持多编程语言。FastTokenizer 提供在 [C++](./docs/cpp/README.md)、[Python](./docs/python/README.md) 语言上开发的能力。
- 包含所有预处理。覆盖绝大部分 Transformer 模型的 Tokenizer 所需要的功能，包括特殊 Tokens 的拼接、截断等。输入的原始文本经过 FastTokenizer 处理后得到的结果可直接输入到 Transformer 类模型。

## 快速开始

下面将介绍 Python 版本 FastTokenizer 的使用方式，C++ 版本的使用方式可参考 [FastTokenizer C++ 库使用教程](./docs/cpp/README.md)。

### 环境依赖

- Windows 64位系统
- Linux x64系统
- MacOS 10.14+系统（ m1 芯片的 MacOS，需要使用 x86_64 版本的 Anaconda 作为 python 环境方可安装使用）
- Python 3.6 ~ 3.10

### 安装 FastTokenizer

```python
pip install fast-tokenizer-python
```

### FastTokenizer 使用示例

- 准备词表

```shell
# Linux或者Mac用户可直接执行以下命令下载测试的词表，Windows 用户可在浏览器上下载到本地。
wget https://bj.bcebos.com/paddlenlp/models/transformers/ernie/vocab.txt
```

- 切词示例

FastTokenizer 库内置 NLP 任务常用的 Tokenizer，如 ErnieFastTokenizer。下面将展示 FastTokenizer 的简单用法。

```python
import fast_tokenizer
from fast_tokenizer import ErnieFastTokenizer, models

# 0.（可选）设置线程数
fast_tokenizer.set_thread_num(1)
# 1. 加载词表
vocab = models.WordPiece.read_file("ernie_vocab.txt")
# 2. 实例化 ErnieFastTokenizer 对象
fast_tokenizer = ErnieFastTokenizer(vocab)
# 3. 切词
output = fast_tokenizer.encode("我爱中国")
# 4. 输出结果
print("ids: ", output.ids)
print("type_ids: ", output.type_ids)
print("tokens: ", output.tokens)
print("offsets: ", output.offsets)
print("attention_mask: ", output.attention_mask)

# 5. 示例输出
# ids:  [1, 75, 329, 12, 20, 2]
# type_ids:  [0, 0, 0, 0, 0, 0]
# tokens:  ['[CLS]', '我', '爱', '中', '国', '[SEP]']
# offsets:  [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (0, 0)]
# attention_mask:  [1, 1, 1, 1, 1, 1]
```

### FastTokenizer 在 PaddleNLP Tokenizer 模块加速示例

PaddleNLP Tokenizer 模块可简单地应用在模型训练以及推理部署的文本预处理阶段，并通过 `AutoTokenizer.from_pretrained` 方式实例化相应的 Tokenizer 。其中 `AutoTokenizer` 默认加载得到的 Tokenizer 是常规 Python 实现的 Tokenizer，其性能会低于 C++ 实现的 FastTokenizer。为了提升 PaddleNLP Tokenizer 模块性能，目前 PaddleNLP Tokenizer 模块已经支持使用 FastTokenizer 作为 Tokenizer 的后端加速切词阶段。在现有的 Tokenizer 加载接口中，仅需添加 `use_fast=True` 这一关键词参数，其余代码保持不变，即可加载 Fast 版本的 Tokenizer，代码示例如下：

```python
from paddlenlp.transformers import AutoTokenizer

# 默认加载Python版本的Tokenizer
tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')
# 打开use_fast开关，可加载Fast版本Tokenizer
fast_tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh', use_fast=True)

text1 = tokenizer('自然语言处理')
text2 = fast_tokenizer('自然语言处理')

print(text1)
print(text2)

# 示例输出
# {'input_ids': [1, 67, 187, 405, 545, 239, 38, 2], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0]}
# {'input_ids': [1, 67, 187, 405, 545, 239, 38, 2], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0]}

```

目前 PaddleNLP 已支持 BERT、ERNIE、TinyBERT 以及 ERNIE-M 4种 Tokenizer 的 Fast 版本，其余模型的 Tokenizer 暂不支持 Fast 版本。

## FAQ

**Q：我在 AutoTokenizer.from_pretrained 接口上已经打开 `use_fast=True` 开关，为什么文本预处理阶段性能上好像没有任何变化？**

A：在有三种情况下，打开 `use_fast=True` 开关可能无法提升性能：
  1. 没有安装 fast_tokenizer 。若在没有安装 fast_tokenizer 库的情况下打开 `use_fast` 开关，PaddleNLP 会给出以下warning："Can't find the fast_tokenizer package, please ensure install fast_tokenizer correctly. "。

  2. 加载的 Tokenizer 类型暂不支持 Fast 版本。目前支持4种 Tokenizer 的 Fast 版本，分别是 BERT、ERNIE、TinyBERT 以及 ERNIE-M Tokenizer。若加载不支持 Fast 版本的 Tokenizer 情况下打开 `use_fast` 开关，PaddleNLP 会给出以下warning："The tokenizer XXX doesn't have the fast version. Please check the map paddlenlp.transformers.auto.tokenizer.FAST_TOKENIZER_MAPPING_NAMES to see which fast tokenizers are currently supported."

  3. 待切词文本长度过短（如文本平均长度小于5）。这种情况下切词开销可能不是整个文本预处理的性能瓶颈，导致在使用 FastTokenizer 后仍无法提升整体性能。

**Q：如何使用多线程加速分词？**

A：可以通过调用 `fast_tokenizer.set_thread_num(xxx)` 使用多线程进行分词。需要谨慎开启多线程加速分词，在以下场景下可以考虑开启多线程：
  1. CPU资源充足。若在推理阶段使用CPU进行推理，开启多线程分词可能会出现资源竞争情况，从而影响推理阶段的性能。

  2. 文本的批大小较大。若批大小比较小，开启多线程可能不会得到任何加速效果，并且可能会因为线程调度导致延时增长。建议批大小大于4的时候再考虑开启多线程分词。

  3. 文本长度较长。若文本长度较短，开启多线程可能不会得到任何加速效果，并且可能会因为线程调度导致延时增长。建议文本平均长度大于16的时候再考虑开启多线程分词。

**Q：Windows 上编译、运行示例出错。** 相关issue：[issues 4673](https://github.com/PaddlePaddle/PaddleNLP/issues/4673)。

A：FastTokenizer 支持 Linux、Windows 以及 MacOS 系统上运行，同一示例可以在不同的操作系统上运行。如果出现在其他系统编译运行没错，但在 Windows 上编译或者运行示例出错的问题，大概率是编译过程中遇到中文字符的编码问题，FastTokenizer 要求字符集必须为 UTF-8。可以参考Visual Studio的官方文档，设置源字符集为/utf-8解决：[/utf-8（将源字符集和执行字符集设置为 UTF-8）](https://learn.microsoft.com/zh-cn/cpp/build/reference/utf-8-set-source-and-executable-character-sets-to-utf-8?view=msvc-170)。

## 参考文献

- [1] Xinying Song, Alex Salcianuet al. "Fast WordPiece Tokenization", EMNLP, 2021

## 相关文档

[FastTokenizer Pipeline](docs/pipeline/README.md)

[FastTokenizer 编译指南](docs/compile/README.md)

[FastTokenizer C++ 库使用教程](./docs/cpp/README.md)

[FastTokenizer Python 库使用教程](./docs/python/README.md)
