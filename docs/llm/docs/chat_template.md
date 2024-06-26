# 多轮对话精调教程

当前开源Chat 类型模型越来越多，PaddleNLP 已经集成了 [Llama](../config/llama)、[Qwen](../config/qwen)、[ChatGLM](../config/chatglm) 等系列模型，也支持[多轮对话 Prompt Template 推理](https://paddlenlp.readthedocs.io/zh/latest/get_started/chat_template.html)，只需要调用`apply_chat_template` 函数即可构造将对话历史和用户最新 query 按照模型指定规则拼接到一起，实现不同模型的定制化 Prompt 规则推理。

此外多轮对话训练精调的应用场景也是越来越多，不同模型的多轮对话模板构造规则都不一致，为了在训练侧标准化前处理上的区别，设计了`chat_template`来解决此问题。

### 如何构造 `chat_template`

只需要添加一个 chat_template 的配置即可为该模型添加相应的多轮对话精调训练支持，以`qwen-14b-chat`配置文件

> 以下配置参考：https://huggingface.co/Qwen/Qwen-14B-Chat/blob/main/qwen_generation_utils.py#L119

```json
{
    "system": "You are a helpful assistant.",
    "conversation": ["\n<|im_start|>user\n{{user}}<|im_end|>\n<|im_start|>assistant\n", "{{bot}}<|im_end|>"],
    "query": "\n<|im_start|>user\n{{query}}<|im_end|>\n<|im_start|>assistant\n",
}
```

注意点：

1. 配置文件名默认为：`chat_template.json`。
1. 对于 `chat_template.json`配置文件 `query`和`conversation`字段为必选项，且内容非常类似，主要是为应对推理和训练两种场景设计使用：query 只用于推理，query 和 conversation 用于训练。
1. 由于训练和推理过程中会在文本中添加 独特token 标记，其中包括 bos_token, eos_token 以及像上述的 <|im_start|> 自定义标记等，故基于 chat_template 的分词是不会添加 special_token，也就是说 tokenizer 中的 `add_special_tokens` 参数始终要设置为 `False`。
1. `conversation`字段为数组，且必须为两个元素，分别对应着 User 和 Bot 的对话内容，前者在训练过程中不参与 loss 的计算，后者的参与 Loss 的计算。
1. 在训练过程中，system 文本的长度不可大于 `max_length`，当对话轮次只有一轮时，基于 token 长度来截断，伪代码为：`(system_tokens + conversation_tokens)[:max_length]`；否则将基于对话轮次来截断，详细来说就是在计算训练 token 总长度时，会从后往前计算每一轮的对话长度，如果截止当前的对话（包含 User 和 Bot 的总 tokens 长度）token 长度大于 `max_length`，此时将当前对话轮次给截断，也不计算后续历史对话数据，直接构造训练数据。
1. 在训练过程中，system 必须存在，不能被截断。

#### 如何使用 `chat_template` 进行训练

以`qwen-14b-chat`基座模型为例，首先需要调整的是训练数据部分，需要保证如下格式：

```json
{"src": ["user-1", "user-2", ..., "user-n"], "tgt": ["bot-1", "bot-2", ..., "bot-n"]}
...
```

其次就是将构造好的`chat_template.json`文件传入到 `llm/run_finetune.py` 模块当中：

* 使用模型自带chat-template

> 并不是所有的模型支持chat-template，PaddleNLP 正在全力支持，可根据是否有下载 `chat_template.json` 文件来判断该模型是否支持 chat-template。

```shell
python run_finetune.py ... --model_name_or_path qwen/qwen-7b-chat --chat_template qwen/qwen-7b-chat
```

此时当 `chat_template` 参数和 `model_name_or_path` 参数一致时，此时将默认使用模型自带的chat_template.json` 文件。

* 使用自定义 chat-template

```shell
python run_finetune.py ... --chat_template ./qwen_14b_chat_template.json
```

1. 当 `chat_template` 参数和 `model_name_or_path` 参数一致时，此时将默认使用模型自带的 `chat_template.json` 文件。
1. 当 `chat_template` 参数为文件路径时，此时将使用该文件中的 `chat_template` 配置。
1. 当 `chat_template` 参数为空时，此时不使用 `chat_template` 配置进行训练。

#### 如何自定义system prompt

如果想要在训练或者推理的过程中动态调整 system prompt，需要进行以下调整：

1. 则需要保证 `chat_template.json` 文件中的 system 配置是包含jinja2 中的变量占位符（比如：`<|im_start|>user\n{{user}}<|im_end|>` 中的 {{user}} 就是一个变量占位符），同时尽量让其保留默认参数，比如上述配置可调整成：

> 需要开发者手动调整 `chat_template.json` 实现动态调整 system prompt。

```diff
{
-    "system": "You are a helpful assistant.",
+    "system": "{{system | 'You are a helpful assistant.'}}",
    "conversation": ["\n<|im_start|>user\n{{user}}<|im_end|>\n<|im_start|>assistant\n", "{{bot}}<|im_end|>"],
    "query": "\n<|im_start|>user\n{{query}}<|im_end|>\n<|im_start|>assistant\n",
}
```

2. 训练文本数据中需要配置 `context` 字段将 `system` 字段给传递进去，示例数据为：

```json
{"src": ["user-1", "user-2", ..., "user-n"], "tgt": ["bot-1", "bot-2", ..., "bot-n"], "context": {"system": "你是一个擅长做任务的人工智能助手"}}
...
```

在渲染 chat_template 的时候将以上数据中的`context` 作为jinja2 的上下文数据，这样就可以在训练数据集中定制每个训练数据的 system prompt。
