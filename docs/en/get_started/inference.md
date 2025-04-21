# Quick Start for Inference Service Deployment

We provide a simple and easy-to-use UI-based service deployment method based on dynamic graph inference, allowing users to quickly deploy service-based inference.

Please ensure that before deployment, you have properly installed PaddleNLP, cloned the code from this repo, and installed custom operator libraries. The deployed service is compatible with OpenAI API interfaces.

Clone PaddleNLP locally:
```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git && cd PaddleNLP/llm # Skip if already cloned or downloaded
```

Environment preparation:

```
python >= 3.9
gradio
flask
paddlenlp_ops (optional, high-performance custom acceleration operators, installation reference here)
```

For service deployment on single GPU, use the following script:
```bash
python ./predict/flask_server.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --port 8010 \
    --flask_port 8011 \
    --dtype "float16"
```

Users can also use paddle.distributed.launch to start multi-GPU inference.

Parameters:
- port: Gradio UI service port number, default 8010
- flask_port: Flask service port number, default 8011

Other parameters please refer to the inference documentation for configuration.

Graphical interface:
- Visit http://127.0.0.1:8010 to use the gradio interface for conversations. API access: You can also access the service via flask API.

Accessing the service:

1. You can directly use curl to start a conversation:
```
curl 127.0.0.1:8011/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{"message": [{"role": "user", "content": "你好"}]}'
```

2. You can use the OpenAI client to call:
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
3. You can also refer to the script invocation using the ./predict/request_flask_server.py file.
```bash
# Under the PaddleNLP/llm directory
python predict/request_flask_server.py
```
