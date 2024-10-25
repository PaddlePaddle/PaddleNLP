# [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/1909.05858.pdf)

## 摘要

Meta AI 实验室高调宣布，将开放自己的 OPT（Open Pretrained Transformer，预训练变换模型）预训练模型，并贡献出所有代码，此模型对标 GPT3，从模型性能、多个下有任务以及小样本中都取得了与 GPT-3可比的成绩，PaddleNLP 也是及时接入此模型，各位开发者只需要简单的调用即可使用此大模型。

## 文本生成测试
```sh
python demo.py
```
模型生成使用到的参数释义如下：
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。
- `max_predict_len` 表示最大生成的句子长度。

## 生成结果样例

```
input text:
Question:If x is 2 and y is 5, what is x+y?
Answer: 7

Question: if x is 12 and y is 9, what is x+y?
Answer:21

Question: if x is 3 and y is 4, what is x+y?

output text:
Answer:7

Question: if x is
==================================================
input text:
a chat between a curious human and Statue of Liberty.
Human: What is your name?
Statue: I am statue of liberty.

Human: where do you live?
Statue: New york city.

Human: how long have you lived there?

output text:
Statue: I have lived here for a long
==================================================
input text:
Chinese: 我想回家。
English: I want to go home.

Chinese: 我不知道。
English: I don't know.

Chinese: 我饿了。
English: I am hungry.

Chinese: 我累了。

output text:
English: I am tired.

Chinese:
==================================================
```
