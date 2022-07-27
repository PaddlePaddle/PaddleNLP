# CodeGen: A Conversational Paradigm for Program Synthesis

## 模型简介

[CodeGen](https://arxiv.org/pdf/2203.13474.pdf) （A Conversational Paradigm for Program Synthesis）提出了一种通过大型语言模型进行对话式程序生成的方法，将编写规范和程序的过程转换为用户和系统之间的多回合对话。它把程序生成看作一个序列预测问题，用自然语言表达规范，并有条件地对所期望的程序进行抽样。同时，CodeGen（16B）在HumanEval benchmark上已经超过[OpenAI's Codex](https://arxiv.org/pdf/2107.03374.pdf)。

本项目展示如何调用CodeGen来进行代码生成。

## 快速开始

### 环境依赖

  - python >= 3.6
  - paddlepaddle >= 2.3.0
  - paddlenlp >= 2.3.4

### 代码调用

```python
import re
import paddle
from paddlenlp.transformers import CodeGenTokenizer, CodeGenForCausalLM

# The supported models are shown in the following table
model_name = 'Salesforce/codegen-350M-mono'
# Init tokenizer
tokenizer = CodeGenTokenizer.from_pretrained(model_name)
# Init model
model = CodeGenForCausalLM.from_pretrained(model_name)
inputs = tokenizer(["def hello_world():"])
inputs = {k: paddle.to_tensor(v) for (k, v) in inputs.items()}
# Generate
output, score = model.generate(inputs['input_ids'],
                               max_length=128,
                               decode_strategy='sampling',
                               top_k=5,
                               repetition_penalty=1.1,
                               temperature=0.6)
# Decode the result
print(
    re.split(
        "\nclass|\ndef|\n#|\n@|\nprint|\nif",
        tokenizer.decode(output[0],
                         skip_special_tokens=True,
                         spaces_between_special_tokens=False))[0].rstrip())
```

其中参数释义如下：
- `max_length` 解码的最大长度，默认128。
- `decode_strategy` 解码的策略，默认sampling。
- `top_k` 解码参数top_k，默认5。
- `repetition_penalty` 解码重复惩罚系数，默认1.1。
- `temperature` 解码参数temperature，默认0.6。

模型列表
| 模型名称                           | 说明                         |
| :--------------------------------- | -------------------------------- |
| Salesforce/codegen-350M-mono             | 基于Python数据集BIGPYTHON训练  |
| Salesforce/codegen-2B-mono             | 基于Python数据集BIGPYTHON训练  |
| Salesforce/codegen-6B-mono             | 基于Python数据集BIGPYTHON训练  |
| Salesforce/codegen-16B-mono             | 基于Python数据集BIGPYTHON训练  |
| Salesforce/codegen-350M-nl             | 基于自然语言数据集THEPILE训练  |
| Salesforce/codegen-2B-nl             | 基于自然语言数据集THEPILE训练  |
| Salesforce/codegen-6B-nl             | 基于自然语言数据集THEPILE训练  |
| Salesforce/codegen-16B-nl             | 基于自然语言数据集THEPILE训练  |
| Salesforce/codegen-350M-multi             | 基于多编程语言数据集BIGQUERY训练  |
| Salesforce/codegen-2B-multi            | 基于多编程语言数据集BIGQUERY训练  |
| Salesforce/codegen-6B-multi             | 基于多编程语言数据集BIGQUERY训练  |
| Salesforce/codegen-16B-multi             | 基于多编程语言数据集BIGQUERY训练  |

### TaskFlow调用
参考[TaskFlow文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/taskflow.md)
