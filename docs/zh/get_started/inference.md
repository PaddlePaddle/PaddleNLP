# 推理服务化快速上手

我们提供了一套基于动态图推理的简单易用 UI 服务化部署方法，用户可以快速部署服务化推理。

请确保，在部署前请确保已正确安装 PaddeNLP，clone 本 repo 下位置代码。以及自定义算子库。本部署的服务是兼容 OpenAI API 接口

Clone PaddleNLP 到本地
```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git && cd PaddleNLP/llm # 如已clone或下载PaddleNLP可跳过
```


环境准备

```
python >= 3.9
gradio
flask
paddlenlp_ops (可选，高性能自定义加速算子， 安装参考 https://paddlenlp.readthedocs.io/zh/latest/llm/docs/predict/installation.html)
```

服务化部署,单卡脚本如下:
```bash
python  ./predict/flask_server.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --port 8010 \
    --flask_port 8011 \
    --dtype "float16"
```
用户也可以使用 paddle.distributed.launch 启动多卡推理。

其中参数如下：
- port: Gradio UI 服务端口号，默认8010。
- flask_port: Flask 服务端口号，默认8011。

其他参数请参见推理文档中推理参数配置。


### 使用模型
**图形化界面**:
- 打开 http://127.0.0.1:8010 即可使用 gradio 图形化界面，即可开启对话。

**API 访问:**
- 您也可用通过 flask 服务化 API 的形式访问服务:

**1.** 您可以直接使用 curl, 开始对话
```
curl 127.0.0.1:8011/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{"message": [{"role": "user", "content": "你好"}]}'
```

**2.** 可以使用 OpenAI 客户端调用：
```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8011/v1/",
)

# Completion API
stream = True
completion = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "PaddleNLP好厉害！这句话的感情色彩是？"}
    ],
    max_tokens=1024,
    stream=stream,
)

if stream:
    for c in completion:
        print(c.choices[0].delta.content, end="")
else:
    print(completion.choices[0].message.content)
```


**3.** 还可以参考：`./predict/request_flask_server.py` 文件使用脚本调用。
```bash
# 在 PaddleNLP/llm 目录下
python predict/request_flask_server.py
```
