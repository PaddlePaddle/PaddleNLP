# 简介
**介绍**：mengzi-T5-base 与 T5 结构相同，不包含下游任务，需要在特定任务上 Finetune 后使用。
适用于文案生成、新闻生成等可控文本生成任务；与 GPT 定位不同，不适合文本续写。

# 使用示例：
```python
from paddlenlp import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained('Langboat/mengzi-t5-base')
model = T5ForConditionalGeneration.from_pretrained('Langboat/mengzi-t5-base')
```

# 权重来源
https://huggingface.co/Langboat/mengzi-t5-base
