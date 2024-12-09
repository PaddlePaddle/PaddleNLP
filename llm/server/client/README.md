# 客户端使用方式

## 简介

服务化部署客户端提供命令行接口和Python接口，可以快速调用服务化后端部署的LLM模型服务。

## 安装

源码安装
```
pip install .
```

## 命令行接口

首先通过环境变量设置模型服务模式、模型服务URL、模型ID，然后使用命令行接口调用模型服务。

| 参数 | 说明 | 是否必填 | 默认值 |
| --- | --- | --- | --- |
| FASTDEPLOY_MODEL_URL | 模型服务部署的IP地址和端口，格式为`x.x.x.x:xxx`。 | 是 | |

```
export FASTDEPLOY_MODEL_URL="x.x.x.x:xxx"

# 流式接口
fdclient stream_generate "你好?"

# 非流式接口
fdclient generate "你好，你是谁?"
```

## Python接口

首先通过Python代码设置模型服务URL(hostname+port)，然后使用Python接口调用模型服务。

| 参数 | 说明 | 是否必填 | 默认值 |
| --- | --- | --- | --- |
| hostname+port | 模型服务部署的IP地址和端口，格式为`x.x.x.x。 | 是 | |


```
from fastdeploy_client.chatbot import ChatBot

hostname = "x.x.x.x"
port = xxx

# 流式接口，stream_generate api的参数说明见附录
chatbot = ChatBot(hostname=hostname, port=port)
stream_result = chatbot.stream_generate("你好", topp=0.8)
for res in stream_result:
    print(res)

# 非流式接口，generate api的参数说明见附录
chatbot = ChatBot(hostname=hostname, port=port)
result = chatbot.generate("你好", topp=0.8)
print(result)
```

### 接口说明
```
ChatBot.stream_generate(message,
                        max_dec_len=1024,
                        min_dec_len=2,
                        topp=0.0,
                        temperature=1.0,
                        frequency_score=0.0,
                        penalty_score=1.0,
                        presence_score=0.0,
                        eos_token_ids=254186)

# 此函数返回一个iterator，其中每个元素为一个dict, 例如：{"token": "好的", "is_end": 0}
# 其中token为生成的字符，is_end表明是否为生成的最后一个字符（0表示否，1表示是）
# 注意：当生成结果出错时，返回错误信息；不同模型的eos_token_ids不同
```

```
ChatBot.generate(message,
                 max_dec_len=1024,
                 min_dec_len=2,
                 topp=0.0,
                 temperature=1.0,
                 frequency_score=0.0,
                 penalty_score=1.0,
                 presence_score=0.0,
                 eos_token_ids=254186)

# 此函数返回一个，例如：{"results": "好的，我知道了。"}，其中results即为生成结果
# 注意：当生成结果出错时，返回错误信息；不同模型的eos_token_ids不同
```

### 参数说明

| 字段名 | 字段类型 | 说明 | 是否必填 | 默认值 | 备注 |
| :---: | :-----: | :---: | :---: | :-----: | :----: |
| req_id |  str  | 请求ID，用于标识一个请求。建议设置req_id，保证其唯一性   | 否 | 随机id | 如果推理服务中同时有两个相同req_id的请求，会返回req_id重复的错误信息 |
| text   | str  | 请求的文本 | 是 | 无 |  |
| max_dec_len | int  | 最大生成token的长度，如果请求的文本token长度加上max_dec_len大于模型的max_seq_len，会返回长度超限的错误信息 | 否 | max_seq_len减去文本token长度 |  |
| min_dec_len | int | 最小生成token的长度，最小是1 | 否 | 1 |  |
| topp | float | 控制随机性参数，数值越大则随机性越大，范围是0~1 | 否 | 0.7 |  |
| temperature | float | 控制随机性参数，数值越小随机性越大，需要大于 0 | 否 | 0.95 |  |
| frequency_score | float | 频率分数 | 否 | 0 |  |
| penalty_score | float | 惩罚分数 | 否 | 1 |  |
| presence_score | float | 存在分数 | 否 | 0 |  |
| stream | bool | 是否流式返回 | 否 | False |  |
| return_all_tokens | bool | 是否一次性返回所有结果 | 否 | False | 与stream参数差异见表后备注 |
| timeout | int | 请求等待的超时时间，单位是秒 | 否 | 300 |  |

* 在正确配置PUSH_MODE_HTTP_PORT字段下，服务支持 GRPC 和 HTTP 两种请求服务
  * stream 参数仅对 HTTP 请求生效
  * return_all_tokens 参数对 GRPC 和 HTTP 请求均有效
