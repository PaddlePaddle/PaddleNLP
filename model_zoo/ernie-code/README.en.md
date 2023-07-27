# ERNIE-Code

[ACL 2023 (Findings)](https://aclanthology.org/2023.findings-acl.676/) | [arXiv](https://arxiv.org/pdf/2212.06742) | [BibTex](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-code/README.md#bibtex) | [中文版](./README.md)

![ernie-code-comp](https://github.com/KB-Ding/PaddleNLP/assets/13767887/2a550b46-a7d5-416d-b300-83cce7044be4)

[ERNIE-Code: Beyond English-Centric Cross-lingual Pretraining for Programming Languages](https://aclanthology.org/2023.findings-acl.676.pdf)


ERNIE-Code is a unified large language model (LLM) that connects 116 natural languages with 6 programming languages. We employ two pre-training methods for universal cross-lingual pre-training: span-corruption language modeling that learns patterns from monolingual NL or PL; and pivot-based translation language modeling that relies on parallel data of many NLs and PLs. Extensive results show that ERNIE-Code outperforms previous multilingual LLMs for PL or NL across a wide range of end tasks of code intelligence, including multilingual code-to-text, text-to-code, code-to-code, and text-to-text generation. We further show its advantage of zero-shot prompting on multilingual code summarization and text-to-text translation.

## Quick Start

This project is the PaddlePaddle implementation of the ERINE-Code, including model prediction and weight conversion. The brief directory structure and description of this example are as follows:

```text
├── README.md               # Documentation
├── predict.py              # Forward prediction demo
├── converter.py            # Weight conversion script
```

### Multilingual Text-to-Code / Code-to-Text

This project provides a simple demo for multlingual code/text generation. The startup command is as follows:

```shell
python predict.py \
  --input 'BadZipFileのAliasは、古い Python バージョンとの互換性のために。' \
  --target_lang 'code' \
  --source_prefix 'translate Japanese to Python: \n' \
  --max_length 1024 \
  --num_beams 3 \
  --device 'gpu'
```

Explanation of parameters in the configuration file:
- `input`:The input sequence.
- `target_lang`: The target language, which can be set to 'text' or 'code'.
- `source_prefix`: The prompt.
- `max_length`: The maximum length of input/output text.
- `num_beams`: The number of beams to keep at each decoding step (for beam search).
- `device`: The running device, which can be set to 'cpu' or 'gpu'.


### Zero-shot Examples
- Multilingual code-to-text generation (zero-shot)

![code-to-text-examples](https://github.com/KB-Ding/PaddleNLP/assets/13767887/7dbf225e-e6be-401d-9f6c-f733e2f68f76)

![zh_code-to-text_examples-1](https://github.com/KB-Ding/PaddleNLP/assets/13767887/2d1ba091-f43c-4f3e-95c6-0038ede9e63e)

- Multilingual text-to-text translation (zero-shot)

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
