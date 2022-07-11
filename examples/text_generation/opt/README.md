# [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/1909.05858.pdf)

## 摘要

Meta AI 实验室高调宣布，将开放自己的语言大模型 OPT（Open Pretrained Transformer，预训练变换模型），并贡献出所有代码，而这次Meta AI 直接开源千亿参数的OPT模型，对标GPT3，模型性能方面，在多个任务，不管是zero-shot还是multi-shot中都取得了与GPT-3可比的成绩，PaddleNLP也是及时将此大模型接入PaddleNLP，各位开发者只需要简单的调用即可使用此大模型。

## 文本生成测试
```sh
python demo.py
```
模型生成使用到的参数释义如下：
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。
- `max_predict_len` 表示最大生成的句子长度。
- `repetition_penalty` 表示生成重复token的惩罚参数，详细信息可查看[这篇论文](https://arxiv.org/pdf/1909.05858.pdf)。

## 生成结果样例

```
input text: Diet English : I lost 10 kgs! ; German :
output text:  >I'm a vegetarian, but I don't eat meat.
>I'm a vegan, but I don't eat meat.  You're not a vegetarian because you are eating meat?</s>
==================================================
input text: Reviews Rating: 5.0
output text: /10

I have been using this product for a few months now and I am very happy with it! It is so easy to use, the scent is great and the price is right. The only downside is that you can't get any of the other ingredients in the bottle (which is not an issue). However, if you are looking for something more natural, this is definitely your best bet.

This product has been used on my skin since I was little. I love how it smells and tastes like fresh fruit. I also love how it makes me feel better after a long day
==================================================
input text: Questions Q: What is the capital of India?
output text:
A: The capital of India is Mumbai. It has a population of about 1,000 million people and it is one of the most populous cities in the world. There are more than 100 million people living there. And that's why we have to be very careful with our investments because they can't just go away. They need to grow their economy so that they don't lose money. So I think that's what we're doing. We've got to make sure that we invest in the right way.
Q: How do you see the growth rate for the next five years
==================================================
input text: Books Weary with toil, I haste me to my bed,
output text:  and lay down on the floor.
"I am not going to sleep," said I, "for I have been so tired that I cannot bear to go out."
"You are not going to sleep?" cried he, "and you will never be able to sleep again."
"No; but I shall not let myself be afraid of it. It is a very good thing for me to be in peace, and I do not want to be frightened by it. But I must make sure that I am not too much disturbed by it. If I should
==================================================
```
