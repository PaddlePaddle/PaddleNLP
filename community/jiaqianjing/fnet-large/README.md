# 详细介绍
**介绍**：该模型是 large 版本的 FNet 模型。论文：https://arxiv.org/pdf/2105.03824v3.pdf

**网络结构**： 使用傅里叶变换，替换了传统 Transformer 中 attention 的部分进行特征提取，从而加速模型的计算。

**适用下游任务**：与传统的 bert 类似，适合 NLP 常见的下游网络，如：文本分类，序列标注等。

# 使用示例

```python
import paddle
import paddlenlp as ppnlp

text = "Replace me by any text you'd like."
tokenizer = ppnlp.transformers.FNetTokenizer.from_pretrained('fnet-large')
model = ppnlp.transformers.FNetModel.from_pretrained('fnet-large')

model.eval()
inputs = tokenizer(text)
inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
outputs = model(**inputs)

logits = outputs[0]
array = logits.numpy()
print("prediction_logits shape:{}".format(array.shape))
print("prediction_logits:{}".format(array))
```

# 权重来源

huggingface [google/fnet-large](https://huggingface.co/google/fnet-large/tree/main)
