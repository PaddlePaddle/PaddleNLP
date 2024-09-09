## 对话生成模板

PaddleNLP 支持主流LLM 对话模型，同时支持自动构建多轮对话，可通过以下脚本。

### 使用对话模板

```python
from paddlenlp.transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-v1.1")

# 单论对话
query = "北京有什么好玩的"
inputs = tokenizer.apply_chat_template(query, return_tensors="pd")

# 多轮对话
query = [["1+1=", "1+1=2"], ["再加一"]]
inputs = tokenizer.apply_chat_template(query, return_tensors="pd")
```

### 自定义对话模板

在介绍如何自定义对话模板之前，介绍对话模板构造的逻辑：`final_query = system + conversation_history + query`。

* system: 在最终 prompt 最前面的固定文本，比如：你是一个人工智能助手，风趣幽默，通常喜欢用比较文艺的语言风格跟人们沟通。
* conversation_history: 将多轮对话构造成一个 query，不同模型通常会有不同的构造规则。
* query: 用户最新的输入。

构建自定义对话模板非常简单，只需要创建一个 `chat_template.json` 文件即可，如下所示：

1. 创建 chat_template 文件

> 文件名默认为：`chat_template.json`

```json
{
    "system": "你是一个人工智能助手，风趣幽默，通常喜欢用比较文艺的语言风格跟人们沟通。",
    "conversation": ["[Round {{index}}]\n问：{{user}}\n", "答：{{bot}}\n"],
    "query": "[Round {{index}}]\n问：{{query}}\n答："
}
```

参数介绍

* 配置文件当前主要有三个字段：`system`, `conversation`, `query`。
  * `system`: 在最终 prompt 构造时拼接到最前面固定的文本。通常不参与训练中 loss 的计算。
  * `conversation`: 多轮对话的配置，且必须为两个配置：[user-template, bot-template]，分别对应多轮对话中用户 query 的配置和模型回复 answer 的配置。可用于训练和推理两个阶段。
  * `query`: 用户最新 query 的构造，配置内容和 `conversation` 大体一致，且通常仅用于推理。

2. 通过 tokenizer 加载自定义对话模板

可通过两种方式加载：
* 将 `chat_template.json` 文件放到权重文件夹下，通过 Tokenizer.from_pretrained("/path/") 进行自动加载。
* 手动加载：先初始化tokenizer，再通过 `tokenizer.init_chat_template(/path/to/file)` 函数加载。

3. 使用对话模板

```python
from paddlenlp.transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-v1.1")

# 仅返回拼接后的文本
query = "北京有什么好玩的"
full_query = tokenizer.apply_chat_template(query, tokenize=False)

# 对拼接后的文本解码
inputs = tokenizer.apply_chat_template(query, tokenize=True, return_tensors="pd")
```
