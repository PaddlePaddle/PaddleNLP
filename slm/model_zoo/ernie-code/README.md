# ERNIE-Code

[ACL 2023 (Findings)](https://aclanthology.org/2023.findings-acl.676/) | [arXiv](https://arxiv.org/pdf/2212.06742) | [BibTex](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/model_zoo/ernie-code#bibtex) | [English version](https://github.com/PaddlePaddle/PaddleNLP/blob/release/2.8/model_zoo/ernie-code/README.en.md)

![ernie-code-comp](https://github.com/KB-Ding/PaddleNLP/assets/13767887/2a550b46-a7d5-416d-b300-83cce7044be4)

[ERNIE-Code: Beyond English-Centric Cross-lingual Pretraining for Programming Languages](https://aclanthology.org/2023.findings-acl.676.pdf)


ERNIE-Code 是一个多自然语言、多编程语言的统一代码语言模型（Code LLM），支持116种自然语言和6+种编程语言。采用了两种预训练方法来进行跨语言预训练：
- Span-Corruption Language Modeling (SCLM) 从单语言的自然语言或编程语言中进行掩码语言学习；
- Pivot-based Translation Language Modeling (PTLM)，将多自然语言到多编程语言的映射 规约为，以英语为枢轴(pivot)的多自然语言到英语、和英语到多编程语言的联合学习。

ERNIE-Code 在代码智能的各种下游任务中，包括代码到多自然语言、多自然语言到代码、代码到代码、多自然语言文档翻译等任务，优于以前的多语言代码和文本模型（例如 mT5 和 CodeT5），同时在多自然语言的代码摘要和文档翻译等任务上具备较好的的 zero-shot prompt 能力。

详细请参考[这里](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/model_zoo/ernie-code).
