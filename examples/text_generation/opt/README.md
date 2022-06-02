# [OPT: Open Pre-trained Transformer Language Models](http://arxiv.org/abs/2205.01068)

## 摘要

Meta AI近期公开了开放预训练模型（OPT-175B），这是一个拥有1750亿参数的语言模型，基于公开可用数据集训练，让更多的社区参与到理解这一基础新技术中来。对于这种规模的语言技术系统来说。

## 文本生成测试
```sh
python demo.py
```

模型生成使用到的参数释义如下：
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。
- `max_predict_len` 表示最大生成的句子长度。
- `repetition_penalty` 表示生成重复token的惩罚参数。

## 生成结果样例

TODO: to be update

```

```
