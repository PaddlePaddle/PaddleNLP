# LLaMA

## 1. Model Overview

**Supported Models:**

| Model                                |
|--------------------------------------|
| facebook/llama-7b                    |
| facebook/llama-13b                   |
| facebook/llama-30b                   |
| facebook/llama-65b                   |
| meta-llama/Llama-2-7b                |
| meta-llama/Llama-2-7b-chat           |
| meta-llama/Llama-2-13b               |
| meta-llama/Llama-2-13b-chat          |
| meta-llama/Llama-2-70b               |
| meta-llama/Llama-2-70b-chat          |
| meta-llama/Meta-Llama-3-8B           |
| meta-llama/Meta-Llama-3-8B-Instruct  |
| meta-llama/Meta-Llama-3-70B          |
| meta-llama/Meta-Llama-3-70B-Instruct |
| ziqingyang/chinese-llama-7b          |
| ziqingyang/chinese-llama-13b         |
| ziqingyang/chinese-alpaca-7b         |
| ziqingyang/chinese-alpaca-13b        |
| idea-ccnl/ziya-llama-13b-v1          |
| linly-ai/chinese-llama-2-7b          |
| linly-ai/chinese-llama-2-13b         |
| baichuan-inc/Baichuan-7B             |
| baichuan-inc/Baichuan-13B-Base       |
| baichuan-inc/Baichuan-13B-Chat       |
| baichuan-inc/Baichuan2-7B-Base       |
| baichuan-inc/Baichuan2-7B-Chat       |
| baichuan-inc/Baichuan2-13B-Base      |
| baichuan-inc/Baichuan2-13B-Chat      |
| FlagAlpha/Llama2-Chinese-7b-Chat     |
| FlagAlpha/Llama2-Chinese-13b-Chat    |

Usage:

Refer to `llama.cpp`...
```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")
```

## 2. Model License

The use of LLaMA model weights requires compliance with the [License](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/llama/LICENSE).

The use of Llama2 model weights requires compliance with the [License](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/llama/Llama2.LICENSE).
