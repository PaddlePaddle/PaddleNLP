# ERNIE-Code

[ACL 2023 (Findings)](https://aclanthology.org/2023.findings-acl.676/) | [arXiv](https://arxiv.org/pdf/2212.06742) | [BibTex](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-code/README.md#bibtex) | [English version](./README.en.md)

![ernie-code-comp](https://github.com/KB-Ding/PaddleNLP/assets/13767887/2a550b46-a7d5-416d-b300-83cce7044be4)

[ERNIE-Code: Beyond English-Centric Cross-lingual Pretraining for Programming Languages](https://aclanthology.org/2023.findings-acl.676.pdf)


ERNIE-Code是一个多自然语言、多编程语言的统一代码语言模型（Code LLM），支持116种自然语言和6+种编程语言。采用了两种预训练方法来进行跨语言预训练：
- Span-Corruption Language Modeling (SCLM) 从单语言的自然语言或编程语言中进行掩码语言学习；
- Pivot-based Translation Language Modeling (PTLM)，将多自然语言到多编程语言的映射 规约为，以英语为枢轴(pivot)的多自然语言到英语、和英语到多编程语言的联合学习。

ERNIE-Code在代码智能的各种下游任务中，包括代码到多自然语言、多自然语言到代码、代码到代码、多自然语言文档翻译等任务，优于以前的多语言代码和文本模型（例如mT5 和 CodeT5），同时在多自然语言的代码摘要和文档翻译等任务上具备较好的的zero-shot prompt能力。

## 快速开始

本项目是ERNIE-Code的PaddlePaddle实现，包括模型预测和权重转换。以下是该示例的简要目录结构和说明：

```text
├── README.md               # 文档
├── predict.py              # 前向预测示例
├── converter.py            # 权重转换脚本
```

### 多语言文本到代码/代码到文本

本项目提供了一个简单的多语言代码/文本生成的演示。启动命令如下：

```shell
python predict.py \
  --input 'BadZipFileのAliasは、古い Python バージョンとの互換性のために。' \
  --target_lang 'code' \
  --source_prefix 'translate Japanese to Python: \n' \
  --max_length 1024 \
  --num_beams 3 \
  --device 'gpu'
```

配置文件中参数的解释：
- `input`：输入的文本序列。
- `target_lang`：目标语言，可设置为'text'或'code'。
- `source_prefix`：提示词Prompt。
- `max_length`：输入/输出文本的最大长度。
- `num_beams`：解码时每个时间步保留的beam大小（用于束搜索）。
- `device`：运行设备，可设置为'cpu'或'gpu'。



### Zero-shot示例
- 多语言代码到文本生成（zero-shot）

![code-to-text-examples](https://github.com/KB-Ding/PaddleNLP/assets/13767887/7dbf225e-e6be-401d-9f6c-f733e2f68f76)

![zh_code-to-text_examples-1](https://github.com/KB-Ding/PaddleNLP/assets/13767887/2d1ba091-f43c-4f3e-95c6-0038ede9e63e)

- 计算机术语翻译（zero-shot）

![zero-shot-mt-examples](https://github.com/KB-Ding/PaddleNLP/assets/13767887/8be1a977-fa21-4a46-86ba-136fa8276a1a)


## BibTeX
```
@inproceedings{chai-etal-2023-ernie,
    title = "{ERNIE}-Code: Beyond {E}nglish-Centric Cross-lingual Pretraining for Programming Languages",
    author = "Chai, Yekun  and
      Wang, Shuohuan  and
      Pang, Chao  and
      Sun, Yu  and
      Tian, Hao  and
      Wu, Hua",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.676",
    pages = "10628--10650",
    abstract = "Software engineers working with the same programming language (PL) may speak different natural languages (NLs) and vice versa, erecting huge barriers to communication and working efficiency. Recent studies have demonstrated the effectiveness of generative pre-training in computer programs, yet they are always English-centric. In this work, we step towards bridging the gap between multilingual NLs and multilingual PLs for large language models (LLMs). We release ERNIE-Code, a unified pre-trained language model for 116 NLs and 6 PLs. We employ two methods for universal cross-lingual pre-training: span-corruption language modeling that learns patterns from monolingual NL or PL; and pivot-based translation language modeling that relies on parallel data of many NLs and PLs. Extensive results show that ERNIE-Code outperforms previous multilingual LLMs for PL or NL across a wide range of end tasks of code intelligence, including multilingual code-to-text, text-to-code, code-to-code, and text-to-text generation. We further show its advantage of zero-shot prompting on multilingual code summarization and text-to-text translation. We release our code and pre-trained checkpoints.",
}
```
