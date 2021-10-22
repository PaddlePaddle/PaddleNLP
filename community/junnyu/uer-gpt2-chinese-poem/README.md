# 详细介绍
# Chinese Poem GPT2 Model
**介绍**： 该模型用于生成中国古诗词。训练数据包含 80 万首中国古诗词，由 chinese-poetry 和 Poetry 项目收集。

**模型结构**： **`GPTLMHeadModel`**，GPT模型。

**适用下游任务**：**诗歌文本生成**。

# 使用示例

```python
import numpy as np

import paddle
from paddlenlp.transformers import BertTokenizer, GPTForGreedyGeneration
from paddlenlp.utils.log import logger


class Demo:
    def __init__(self, model_name_or_path="junnyu/uer-gpt2-chinese-poem", max_predict_len=32):
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        logger.info("Loading the model parameters, please wait...")
        self.model = GPTForGreedyGeneration.from_pretrained(
            model_name_or_path,
            max_predict_len=max_predict_len,
            eol_token_id=self.tokenizer.pad_token_id,
        )
        self.model.eval()
        logger.info("Model loaded.")

    # prediction function
    @paddle.no_grad()
    def dictation_poetry_cn(self, front):
        # don't add [SEP] token.
        ids = self.tokenizer(front)["input_ids"][:-1]
        input_ids = paddle.to_tensor(np.array(ids).reshape(1, -1).astype("int64"))
        out = self.model(input_ids)
        out = [int(x) for x in out.numpy().reshape([-1])]
        logger.info(
            self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(out)
            )
        )


demo = Demo(model_name_or_path="junnyu/uer-gpt2-chinese-poem")
demo.dictation_poetry_cn("大漠")

# [CLS] 大 漠 风 沙 暗 ， 长 城 日 月 寒 。 汉 兵 天 上 至 ， 胡 马 雪 中 看 。 壮 士 心 逾 勇 ， 孤 军 气 不 残
```

# 权重来源

https://huggingface.co/uer/gpt2-chinese-poem
