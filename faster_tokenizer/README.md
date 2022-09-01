# FasterTokenizer
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
FasterTokenizer是一款简单易用且功能强大的高性能文本预处理库，集成业界多个常用的Tokenizer实现，支持不同NLP场景下的文本预处理功能，如文本分类、阅读理解，序列标注等。结合PaddleNLP Tokenizer模块为用户提供快速版本通用文本预处理能力。

## 特性

- 高性能。在文本分类任务上，FasterTokenizer对比Python版本Tokenizer加速比最高可达20倍。
- 跨平台。FasterTokenizer可在不同的系统平台上使用，目前已支持Windows x64，Linux x64以及MacOS 10.14+平台上使用（m1芯片的Mac系统，需要使用x86_64版本的Anaconda作为python环境方可安装使用）。
- 多编程语言支持。FasterTokenizer提供在C++、Python语言上开发的能力。
- 灵活性强。用户可以通过指定不同的FasterTokenizer组件定制满足需求的Tokenizer。

## 切词流水线

## 快速开始

### PaddleNLP Tokenizer模块加速示例
