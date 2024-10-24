# 获取和转换 Paddle 版 MiniGPT4 权重

## 1. 准备 MiniGPT4 中所有模块的权重

你需要下载3个权重，以获取最终 MiniGPT4的权重，分别是：
- Pretrained MiniGPT-4
- Vicuna Weight
- Blip2 Weight

### 1.1 下载 MiniGPT4 的预训练权重

根据你准备的 Vicuna 模型版本，下载预训练的 MiniGPT4 权重。

|                               Checkpoint Aligned with Vicuna 7B                                |                                Checkpoint Aligned with Vicuna 13B                                 |
|:----------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|
| [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) | [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) |

### 1.2准备 ViT and Qformer 权重
MiniGPT4中使用的 ViT 和 Qformer Weight 来自 blip2-flan-t5-xxl，这个 weight 在 PaddleNLP 中进行了转换。 所以你可以从 PaddleNLP 下载它，你有两种下载方式进行下载：

#### 1.2.1 通过 paddlenlp 方式加载
直接通过 paddlenlp 的模型加载方法进行下载，下载后一般会存入 `PPNLP_HOME` 指定的目录。

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import paddle
from paddlenlp.transformers import Blip2Model, Blip2VisionModel, Blip2VisionConfig, Blip2QFormerConfig, Blip2QFormerModel

Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xxl")
```

#### 1.2.2 直接点击下载
可以直接进行点击下载：

| blip2-flan-t5-xxl 权重 |                                                    点击下载                                                    |
|:----------------------:|:--------------------------------------------------------------------------------------------------------------:|
|  model_state.pdparams  | [Download](https://paddlenlp.bj.bcebos.com/models/community/Salesforce/blip2-flan-t5-xxl/model_state.pdparams) |

### 1.3 准备 Vicuna 权重

这里需要下载两个权重：Vicuna delta Weight 和 huggingface-formated Llama Weight。 然后你应该结合这两个重量来获得可以使用的 Vicuna 权重。

#### 1.3.1 下载 Vicuna delta 权重

这里展示两种 Vicuna delta 权重，请根据需要选择一种并点击下载。

|                          vicuna-7b-delta-v0                           |                     vicuna-13b-delta-v0                      |
|:---------------------------------------------------------------------:|:------------------------------------------------------------:|
| [Download](https://huggingface.co/lmsys/vicuna-7b-delta-v0/tree/main) | [Download](https://huggingface.co/lmsys/vicuna-13b-delta-v0) |

#### 1.3.2 根据以上选择的 vicuna delta 权重，下载 相应的 llama 权重。

|                                  llama-7b                                  |                             llama-13b                             |
|:--------------------------------------------------------------------------:|:-----------------------------------------------------------------:|
| [Download](https://huggingface.co/baffo32/decapoda-research-llama-7B-hf/tree/main) | [Download](https://huggingface.co/yahma/llama-13b-hf/tree/main) |


#### 1.3.3 结合上面的两个权重，得到可以使用的 vicuna 权重
- 为组合如上两个权重，请安装以下工具：

```shell
pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
```
- 运行以下命令，获取最终可用的 vicuna 权重

```shell
python -m fastchat.model.apply_delta --base /path/to/llama-13bOR7b-hf/  --target /path/to/save/working/vicuna-13b/weight/  --delta /path/to/vicuna-13bOR7b-delta-v0/
```

## 2. 将多个 pytorch 子权重文件合并为一个权重文件

Pytorch 版的权重文件可能是由多个子权重文件组合而成，为使用 PaddleNLP 进行加载并自动转换为 Paddle 版，需要将其合并为一个文件：

### 2.1 下载 MiniGPT 库
在开始之前，请确保已经下载了 [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4.git) 库：

```
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
```

### 2.2 获取完整的 vicuna 权重
进入到 MiniGPT4文件夹，执行以下代码，获取完整的 vicuna 权重文件：
```python
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["FLAGS_use_cuda_managed_memory"]="true"

import torch
from minigpt4.models.modeling_llama import LlamaForCausalLM

llama_model = LlamaForCausalLM.from_pretrained("/path/to/save/working/vicuna-13b/")
torch.save(llama_model.state_dict(), "/path/to/save/working/vicuna-13b/pytorch_model.bin")
```

## 3. 合并以上所有权重，获取最终的 Paddle 版 MiniGPT4 权重
这里提供了一个合并以上权重的脚本，你可以通过设置相关权重路径 以获取最终的 MiniGPT4 权重。

```shell
python merge_weight.py \
    --blip2_path "your dir name of blip2" \
    --vicuna_path "your dir name of vicuna" \
    --minigpt4_path "your ckpt path of minigpt4" \
    --save_path "your dir name saving the final minigpt4"
```

**参数说明**：
- `blip2_path`： 存放 blip2 权重的目录名
- `vicuna_path`： 存放 vicuna_path 权重的目录名
- `minigpt4_path`： 存放 blip2 权重的文件地址，比如./prerained_minigpt4_7b.pth
- `save_path`： 保存 Paddle 版 MiniGPT3 权重的目录名

## 3. More Reference

- [MiniGPT Official Site](https://github.com/Vision-CAIR/MiniGPT-4)
