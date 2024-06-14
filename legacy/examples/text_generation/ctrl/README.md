# [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/pdf/1909.05858.pdf)

## 摘要
大规模语言模型显示出很有前景的文本生成能力，但用户无法轻松控制生成文本的特定方面。我们发布了CTRL，一个包含 16.3 亿个参数的条件转换器语言模型，经过训练以调节控制样式、内容和特定任务行为的控制代码。 控制代码源自与原始文本自然共同出现的结构，保留了无监督学习的优势，同时对文本生成提供了更明确的控制。 这些代码还允许CTRL预测训练数据的哪些部分最有可能给定序列。 这提供了一种通过基于模型的来源归因分析大量数据的潜在方法。 我们在 https://github.com/salesforce/ctrl 上发布了多个全尺寸、预训练版本的CTRL。

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
output text: Ich habe zehn Kilogramm abgenommen!

 Als ich das erste Mal mit meinem Smartphone war, war es ein wenig schwierig zu finden, wo man die App herunterladen kann. Aber jetzt ist sie da.

 Das Smartphone hat mich auch sehr beeindruckt. Es machte mir viel Spaß. Und so funktioniert mein Leben heute ganz einfach und ohne große Probleme.

 Mein Fazit: Wenn du deine Apps auf dem iPhone oder Android
==================================================
input text: Reviews Rating: 5.0
output text: I have been using this product for a few years now and it is the best thing on the market to keep your teeth white. It does not taste bad at all like some of these other products do. The only problem with this product is that you need to use it every day or else they will start coming back in after about 2 weeks. But if you do that, then it's worth it. You can also buy them from Amazon but shipping takes forever. So just make sure you order enough so you don't run out.
 Rating: 5.0
 This stuff works great. My dentist recommended it, and I'm glad he did. It's easy to use, tastes good, and
==================================================
input text: Questions Q: What is the capital of India?
output text: A: mumbai.
 Q: Who was a British politician who served as Prime Minister from 1922 to 1924?
 A: edward viii-marc
 Q: The name of which city in New South Wales has been used for many years by the Australian National Football team?
 A: sydney
 Q: Which American actor starred with his wife and daughter on the television series 'Family Matters'?
 A: james coburn
 Q: In what year did the first edition of this book appear?
 A: 1962
 Q: How long does it take to make one pound of sausage?
==================================================
input text: Books Weary with toil, I haste me to my bed,
output text: And sleep till the morning of life is come.
 The sun has risen and his beams are bright,
 But still he shines upon a world forlorn;
 He sees no more its joys or griefs below,
 Nor hears their murmur as they pass below.
 My heart grows weary for the world's delight,
 For all that makes it dear in human eyes;
 It feels like one who wanders through an empty land,
 With nothing left but desolation there.
 O God! how long shall this be mine abode,
 Where every joy hath passed away from me?
 How long, O God, must I thus wander here,
 In sorrow
==================================================
```
