# Torch2Paddle Weight Conversion Tutorial

## Convert PyTorch Weights
PaddleNLP provides an interface to automatically convert PyTorch weights to Paddle weights. The code is as follows:

```python
from paddlenlp.transformers import AutoModelForCausalLM

AutoModelForCausalLM.from_pretrained("/path/to/pytorch/model", convert_from_torch=True, dtype="float16")
```

> The dtype parameter specifies the actual data type of converted weights, typically: float16, bfloat16 and float32.

The above code will automatically load PyTorch weights and convert them into corresponding Paddle weights, saved in the `/path/to/pytorch/model` directory.

## Merge Sharded PyTorch Weights

Currently, PaddleNLP only supports converting single PyTorch weight files: `pytorch_model.bin`. When PyTorch weights are sharded, they need to be merged first. The merging script is shown below:

```python
import torch, os
state_dict = {}

files = [file for file in os.list("./path/to/pytorch/weight") if file.startswith("pytorch_model-")]

for file in files:
    state_dict.update(torch.load(file))

torch.save(state_dict, "pytorch_model.bin")
```

## Supported Models List

The following table shows the list of supported models for automatic weight conversion:

| Model       | Supported |
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
