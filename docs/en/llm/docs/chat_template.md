# Multi-Turn Dialogue Fine-Tuning Tutorial

With the increasing availability of open-source Chat-type models, PaddleNLP has integrated [Llama](../config/llama), [Qwen](../config/qwen), [ChatGLM](../config/chatglm) and other model series, while also supporting [Multi-Turn Dialogue Prompt Template Inference](https://paddlenlp.readthedocs.io/zh/latest/get_started/chat_template.html). By simply calling the `apply_chat_template` function, we can construct prompts that concatenate dialogue history and the user's latest query according to each model's specified rules, enabling customized prompt-based inference for different models.

Moreover, there is a growing need for fine-tuning multi-turn dialogue training. As different models have distinct template construction rules for multi-turn dialogues, we designed `chat_template` to standardize pre-processing differences on the training side.

### How to Construct `chat_template`

Simply add a `chat_template` configuration to enable multi-turn dialogue fine-tuning training for the model. Taking the `qwen-14b-chat` configuration file as an example:

> The following configuration references: https://huggingface.co/Qwen/Qwen-14B-Chat/blob/main/qwen_generation_utils.py#L119

```json
{
    "system": "You are a helpful assistant.",
    "conversation": ["\n<|im_start|>user\n{{user}}<|im_end|>\n<|im_start|>assistant\n", "{{bot}}<|im_end|>"],
    "query": "\n<|im_start|>user\n{{query}}<|im_end|>\n<|im_start|>assistant\n",
}
```

Key Notes:

1. The configuration file is named `chat_template.json` by default.
2. The `query` and `conversation` fields are mandatory in the `chat_template.json` configuration. Their contents are highly similar, mainly designed to handle inference and training scenarios respectively: `query` is used solely for inference, while both `query` and `conversation` are used for training.
3. During both training and inference, special token markers (e.g., bos_token, eos_token, and custom markers like `<|im_start|>`) are added to the text. Therefore, tokenization based on `chat_template` does not add special tokens, meaning the `add_special_tokens` parameter in the tokenizer must always be set to `False`.
4. The `conversation` field must be an array with exactly two elements, corresponding to the User and Bot dialogue content respectively. The former does not participate in loss calculation during training, while the latter does.
5. During training, the length of the `system` text cannot exceed `max_length`. For single-turn dialogues, truncation is performed at the token level using pseudo-code: `(system_tokens + conversation_tokens)[:max_length]`.
# How to Use `chat_template` for Training

Using the `qwen-14b-chat` base model as an example, first we need to adjust the training data to ensure the following format:

```json
{"src": ["user-1", "user-2", ..., "user-n"], "tgt": ["bot-1", "bot-2", ..., "bot-n"]}
...
```

Next, pass the constructed `chat_template.json` file to the `llm/run_finetune.py` module:

* Using the model's built-in chat-template

> Note: Not all models support chat-template. PaddleNLP is actively working on compatibility. You can check if a model supports chat-template by verifying the presence of the `chat_template.json` file.

```shell
python run_finetune.py ... --model_name_or_path qwen/qwen-7b-chat --chat_template qwen/qwen-7b-chat
```

When the `chat_template` parameter matches `model_name_or_path`, the system will automatically use the model's built-in `chat_template.json` file.

* Using a custom chat-template

```shell
python run_finetune.py ... --chat_template ./qwen_14b_chat_template.json
```

1. When `chat_template` and `model_name_or_path` parameters are identical, the system defaults to using the model's built-in `chat_template.json`.
2. When `chat_template` specifies a file path, the system uses the template configuration from that file.
3. When `chat_template` is unspecified, the system will not use any chat-template configuration during training.

# How to Customize System Prompt

To dynamically adjust the system prompt during training or inference:

1. Ensure the `chat_template.json` file's system configuration contains Jinja2 variable placeholders (e.g., `{{user}}` in `<|im_start|>user\n{{user}}<|im_end|>`), while maintaining default parameters. For example:

> Developers must manually modify `chat_template.json` to enable dynamic system prompt adjustments.

```json
{
  "system": "<|im_start|>system\n{{system_prompt}}<|im_end|>\n",
  "conversations": [
    {"role": "user", "content": "<|im_start|>user\n{{content}}<|im_end|>\n"},
    {"role": "assistant", "content": "<|im_start|>assistant\n{{content}}<|im_end|>\n"}
  ]
}
```

2. When initializing the tokenizer, pass the `system_prompt` parameter:

```python
from paddlenlp.transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-7b-chat", system_prompt="You are a helpful assistant.")
```

This allows dynamic modification of the system prompt during both training and inference.
diff
{
-    "system": "You are a helpful assistant.",
+    "system": "{{system | 'You are a helpful assistant.'}}",
    "conversation": ["\n<|im_start|>user\n{{user}}<|im_end|>\n<|im_start|>assistant\n", "{{bot}}<|im_end|>"],
    "query": "\n<|im_start|>user\n{{query}}<|im_end|>\n<|im_start|>assistant\n",
}
```

2. The training text data requires configuration of the `context` field to pass in the `system` field. Sample data format:

```json
{"src": ["user-1", "user-2", ..., "user-n"], "tgt": ["bot-1", "bot-2", ..., "bot-n"], "context": {"system": "You are an AI assistant skilled at task completion"}}
...
```

When rendering the chat_template, use the `context` data as Jinja2 context variables. This allows customizing the system prompt for each training data instance.
