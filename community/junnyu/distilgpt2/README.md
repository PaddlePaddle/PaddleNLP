# 详细介绍
# DistilGPT2
**介绍**： DistilGPT2 英语语言模型使用 OpenWebTextCorpus（OpenAI 的 WebText 数据集），使用 GPT2 的最小版本的进行了预训练。 该模型有 6 层、768 个维度和 12 个头，总计 82M 参数（相比之下 GPT2 的参数为 124M）。 平均而言，DistilGPT2 比 GPT2 快两倍。
在 WikiText-103 基准测试中，GPT2 在测试集上的困惑度为 16.3，而 DistilGPT2 的困惑度为 21.1（在训练集上进行微调后）。

**模型结构**： **`GPTLMHeadModel`**，GPT模型。

**适用下游任务**：**文本生成**。

# 使用示例

```python

import numpy as np

import paddle
from paddlenlp.transformers import GPTTokenizer, GPTLMHeadModel
from paddlenlp.utils.log import logger


class Demo:
    def __init__(self, model_name_or_path="junnyu/distilgpt2", max_predict_len=32):

        self.tokenizer = GPTTokenizer.from_pretrained(model_name_or_path)
        logger.info("Loading the model parameters, please wait...")
        self.max_predict_len = max_predict_len
        self.model = GPTLMHeadModel.from_pretrained(
            model_name_or_path, eol_token_id=self.tokenizer.eol_token_id
        )
        self.model.eval()
        logger.info("Model loaded.")

    @paddle.no_grad()
    def predict(self, text="My name is Teven and I am"):
        ids = self.tokenizer(text)["input_ids"]
        input_ids = paddle.to_tensor(np.array(ids).reshape(1, -1).astype("int64"))
        out = self.model.generate(
            input_ids,
            max_length=self.max_predict_len,
            repetition_penalty=1.2,
            temperature=0,
        )[0][0]
        out = [int(x) for x in out.numpy().reshape([-1])]
        print(text + self.tokenizer.convert_ids_to_string(out))

demo = Demo(model_name_or_path="junnyu/distilgpt2",max_predict_len=64)
demo.predict(text="My name is Teven and I am")

# My name is Teven and I am a member of the team.
# I have been playing with my friends since we were little, so it was nice to see them play together in our home town on Saturday night! We are very excited about this opportunity for us as well!!<|endoftext|>
```

# 权重来源

https://huggingface.co/distilgpt2
