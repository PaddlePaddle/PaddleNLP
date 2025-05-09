## Dialogue Generation Template

PaddleNLP supports mainstream LLM dialogue models and automatically constructs multi-turn conversations through the following scripts.

### Using Dialogue Templates

```python
from paddlenlp.transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-v1.1")

# Single-round conversation
query = "What's fun to do in Beijing"
inputs = tokenizer.apply_chat_template(query, return_tensors="pd")

# Multi-round conversation
query = [["1+1=", "1+1=2"], ["Add one more"]]
inputs = tokenizer.apply_chat_template(query, return_tensors="pd")
```

### Customizing Dialogue Templates

Before explaining how to customize dialogue templates, let's clarify the construction logic: `final_query = system + conversation_history + query`.

* system: Fixed text at the beginning of the final prompt, e.g., "You are an AI assistant with a witty sense of humor, typically preferring to communicate in a literary style."
* conversation_history: Constructs multi-turn dialogues into a query, with different models potentially having different construction rules.
* query: The user's latest input.

Creating custom dialogue templates is straightforward - just create a `chat_template.json` file as follows:

1. Create chat_template file

> Default filename: `chat_template.json`

```json
{
    "system": "You are an AI assistant with a witty sense of humor, typically preferring to communicate in a literary style.",
    "conversation": ["[Round {{index}}]\nQuestion: {{user}}\n", "Answer: {{bot}}\n"],
    "query": "[Round {{index}}]\nQuestion: {{query}}\nAnswer:"
}
```

Parameter description:

* The configuration file mainly contains three fields: `system`, `conversation`, `query`.
  * `system`: Fixed text prepended to the final prompt. Typically not involved in loss computation during training.
  * `conversation`: Multi-turn dialogue configuration, which must contain two templates: [user-template, bot-template], corresponding to the user query and model response configurations respectively. Used in both training and inference stages.
  * `query`: Construction of the user's latest query, with configuration similar to `conversation`, typically used only during inference.

2. Loading custom dialogue templates via tokenizer

Two loading methods:
* Place the `chat_template.json` file in the model weights directory and load automatically via `Tokenizer.from_pretrained("/path/")`.
* Manual loading: Initialize the tokenizer first, then load via `tokenizer.init_chat_template(/path/to/file)`.

3. Using dialogue templates
```python
from paddlenlp.transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-v1.1")

# Only return the concatenated text
query = "What are some fun things to do in Beijing"
full_query = tokenizer.apply_chat_template(query, tokenize=False)

# Decode the concatenated text
inputs = tokenizer.apply_chat_template(query, tokenize=True, return_tensors="pd")
```
