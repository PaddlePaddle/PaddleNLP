# LLaMA

## 1. 模型介绍

**支持模型权重:**

| Model                            |
| ---------------------------------|
| facebook/llama-7b                 |
| facebook/llama-13b                |
| facebook/llama-30b                |
| facebook/llama-65b                |
| meta-llama/Llama-2-7b             |
| meta-llama/Llama-2-7b-chat        |
| meta-llama/Llama-2-13b            |
| meta-llama/Llama-2-13b-chat       |
| meta-llama/Llama-2-70b            |
| meta-llama/Llama-2-70b-chat       |
| ziqingyang/chinese-llama-7b       |
| ziqingyang/chinese-llama-13b      |
| ziqingyang/chinese-alpaca-7b      |
| ziqingyang/chinese-alpaca-13b     |
| idea-ccnl/ziya-llama-13b-v1       |
| linly-ai/chinese-llama-2-7b       |
| linly-ai/chinese-llama-2-13b      |
| baichuan-inc/Baichuan-7B          |
| baichuan-inc/Baichuan-13B-Base    |
| baichuan-inc/Baichuan-13B-Chat    |
| baichuan-inc/Baichuan2-7B-Base    |
| baichuan-inc/Baichuan2-7B-Chat    |
| baichuan-inc/Baichuan2-13B-Base   |
| baichuan-inc/Baichuan2-13B-Chat   |
| FlagAlpha/Llama2-Chinese-7b-Chat  |
| FlagAlpha/Llama2-Chinese-13b-Chat |



使用方法：

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")
```

## 2. 模型协议

LLaMA 模型的权重的使用则需要遵循[License](../../paddlenlp/transformers/llama/LICENSE)。

Llama2 模型的权重的使用则需要遵循[License](../../paddlenlp/transformers/llama/Llama2.LICENSE)。


## 3. 预训练

请参考[LLM全流程工具介绍](../README.md)

## 4. 模型精调
请参考[LLM全流程工具介绍](../README.md)
