
## Function Call

Function Call 由于引入了 Tool 相关的概念,此时就涉及到如下问题:

1. 调用什么工具?
1. 调用工具的输入参数是什么?
1. 调用工具的输出内容是什么?

以上问题表现在消息如何构造,此时借鉴[文心一言SDK消息数据格式](https://yiyan.baidu.com/developer/doc)并与其保持对齐,故此时有如下三种数据类型:

1. system message

`role` 为 `system`, 内容放在 `content` 字段下, 示例数据如下所示:

```json
{"role": "system", "content": "You are a helpful assistant."}
```

2. user message

`role` 为 `user`, 内容放在 `content` 字段下, 示例数据如下所示:

```json
{"role": "user", "content": "北京现在什么天气?"}
```

3. assistant message

`role` 为 `assistant`, 内容放在 `content` 字段下, 示例数据如下所示:

```json
{"role": "assistant", "content": "对不起,我没办法提供北京的实时天气数据,您可以自行百度搜索北京天气数据."}
```
4. assistant function_call message

此消息数据与以上的区别在于: 此为ReAct类型消息,带有 function_call 字段,其内容主要包含:

* thoughts: 调用此工具的思考
* parameters: 调用此工具的所有输入参数.
* name: 工具的名称,用来标识需要调用哪个工具.

示例数据如下所示:

```json
{"role": "assistant", "function_call": {"thoughts": "用户想要查询北京的天气,我应该调用 weather 工具来完成用户的查询.", "name": "weather", "arguments": "{'location': '北京'}"}}
```

LLM 生成了此类消息之后,会暂停生成，将结果返回给客户端，调用工具，然后将结果返回给工具继续执行.

5. function message

当 LLM 通过 assistant function_call message 成功调用某工具之后,此时将工具返回结果保存成 function message 再输入给模型,示例数据如下所示:

```json
{"role": "function", "content": '{"code": 200, data: "工具返回的具体内容,json string 的格式数据."}'}
```

### 原始数据格式

```json
{
    "messages": []
}
```

### 训练数据集

Function Call 的训练数据集和评估数据集的数据格式不一致,故需要针对于训练数据集单独处理,通常为 `/{model}/convert_fc_train.py` ,例如 qwen 的模型为: `./qwen/convert_fc_train.py` 文件.

转化命令为: `python ./qwen/convert_fc_train.py ./path/to/origin/train.json ./path/to/train/train.json`

此时即可生成可用于训练的 `train.json` 文件.

### 评估数据集

由于评估数据集采用原始数据格式,此时不用做任何处理即可直接进行推理.

> 注意: 此时 train.json 和 dev.json 文件格式不一致,属于正常现象.
