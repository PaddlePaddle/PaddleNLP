# Qwen

## 1.模型介绍

[通义千问（Qwen）](https://arxiv.org/abs/2205.01068) 是阿里云研发的通义千问大模型系列的模型, 有 70 亿和 140 亿两个规模。Qwen是基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。

**支持模型权重:**
| Model             |
|-------------------|
| qwen/qwen-7b      |
| qwen/qwen-7b-chat |
| qwen/qwen-14b     |
| qwen/qwen-14b-chat|
| qwen/qwen-72b     |
| qwen/qwen-72b-chat|

[通义千问（Qwen1.5-MoE）](https://qwenlm.github.io/blog/qwen-moe/) 是阿里云研发的通义千问MoE模型。Qwen1.5-MoE基于Transformer架构，采用了专家混合（MoE）架构，这些模型通过密集型语言模型升级改造而来。例如，Qwen1.5-MoE-A2.7B就是从Qwen-1.8B升级改造而来的。它总共有143亿个参数，但在运行时仅激活27亿个参数，却实现了与Qwen1.5-7B相近的性能，而训练资源仅为其25%。

**支持模型权重:**
| Model (qwen-1.5)       |
|------------------------|
| qwen/qwen1.5-moe-a2.7b |

## 2. 模型精调
请参考[LLM全流程工具介绍](../README.md)
