# torch2paddle

## 转化 Pytorch 权重
PaddleNLP 提供了可自动将 PyTorch 相关的权重转化为 Paddle 权重的接口，代码如下：

```python
from paddlenlp.transformers import AutoModelForCausalLM

AutoModelForCausalLM.from_pretrained("/path/to/pytorch/model", convert_from_torch=True, dtype="float16")
```

> dtype 为转化权重的真实 dtype 数据类型，通常为：float16, bloat16 和 float32。

以上代码可自动加载 pytorch 权重并转化为对应 paddle 权重保存在 `/path/to/pytorch/model` 目录下。

## 合并 Pytorch 分片权重

当前 PaddleNLP 仅支持转化单个 Pytorch 权重：`pytorch_model.bin`文件。所以当Pytorch 权重为分片权重时，需要将其合并，合并脚本如下所示：

```python
import torch, os
state_dict = {}

files = [file for file in os.list("./path/to/pytorch/weight") if file.startswith("pytorch_model-")]

for file in files:
    state_dict.update(torch.load(file))

torch.save(state_dict, "pytorch_model.bin")
```

## 支持模型列表

以下为支持权重自动转化的系列模型列表：

| 模型       | 是否支持 |
|------------|----------|
| AlBert     | ✅        |
| Bart       | ✅        |
| Bert       | ✅        |
| Bloom      | ✅        |
| Clip       | ✅        |
| DistilBert | ✅        |
| Electra    | ✅        |
| ErnieCode  | ✅        |
| GLM        | ✅        |
| Gpt        | ✅        |
| Llama      | ✅        |
| Mt5        | ✅        |
| Opt        | ✅        |
| Qwen       | ✅        |
| Roberta    | ✅        |
| Roformer   | ✅        |
| RW         | ✅        |
| T5         | ✅        |
