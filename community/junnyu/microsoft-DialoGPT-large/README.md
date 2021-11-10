# 详细介绍
# microsoft-DialoGPT-small
**介绍**： 最先进的大规模预训练响应生成模型 (DialoGPT)
DialoGPT 是一种用于多轮对话的 SOTA 大规模预训练对话响应生成模型。 人类评估结果表明，DialoGPT 生成的响应与单轮对话图灵测试下的人类响应质量相当。 该模型是在来自 Reddit 讨论的 147M 多轮对话上训练的。

**模型结构**： **`GPTLMHeadModel`**，GPT模型。

**适用下游任务**：**文本生成**。

# 使用示例

```python
import numpy as np

import paddle
from paddlenlp.transformers import GPTTokenizer, GPTLMHeadModel
from paddlenlp.utils.log import logger


class Demo:
    def __init__(self, model_name_or_path="junnyu/microsoft-DialoGPT-small", max_predict_len=32):

        self.tokenizer = GPTTokenizer.from_pretrained(model_name_or_path)
        logger.info("Loading the model parameters, please wait...")
        self.max_predict_len = max_predict_len
        self.model = GPTLMHeadModel.from_pretrained(
            model_name_or_path, eol_token_id=self.tokenizer.eol_token_id
        )
        self.model.eval()

        logger.info("Model loaded.")

    @paddle.no_grad()
    def predict(self):
        # Let's chat for 5 lines
        for step in range(5):
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            ids = self.tokenizer(input(">> User:"))["input_ids"] + [self.tokenizer.eos_token_id]
            new_user_input_ids = paddle.to_tensor(np.array(ids).reshape(1, -1).astype("int64"))

            # append the new user input tokens to the chat history
            bot_input_ids = paddle.concat([chat_history_ids, new_user_input_ids], axis=-1) if step > 0 else new_user_input_ids


            # generated a response while limiting the total chat history to 1000 tokens,
            chat_history_ids = self.model.generate(bot_input_ids, max_length=self.max_predict_len, pad_token_id=self.tokenizer.eos_token_id,decode_strategy="sampling",top_k=5,)[0]

            # pretty print last ouput tokens from bot
            print("DialoGPT: {}".format(self.tokenizer.convert_ids_to_string(chat_history_ids[0].tolist()).replace("<|endoftext|>","")))
            chat_history_ids = paddle.concat([new_user_input_ids, chat_history_ids], axis=-1)

demo = Demo(model_name_or_path="junnyu/microsoft-DialoGPT-large")
demo.predict()

# >> User: Does money buy happiness?
# DialoGPT: No , but it can buy you a better life .
# >> User: What is the best way to buy happiness ?
# DialoGPT: A job , money , and a better life .
# >> User: This is so difficult !
# DialoGPT: Just get a job , money , and a better life . Then you can buy happiness .
# >> User: Oh, thank you!
# DialoGPT: No problem , friend .
```

# 权重来源

https://huggingface.co/microsoft/DialoGPT-large
