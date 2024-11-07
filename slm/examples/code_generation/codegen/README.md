# 代码生成：写代码的 AI 助理

**目录**
- [代码生成](#代码生成)
  - [简介](#简介)
    - [特色](#特色)
  - [效果展示](#效果展示)
  - [Github Copilot 插件配置](#GithubCopilot 插件配置)
    - [环境依赖](#环境依赖)
    - [代码结构说明](#代码结构说明)
    - [启动服务](#启动服务)
      - [配置参数](#配置参数说明)
    - [测试服务](#测试服务)
    - [配置插件](#配置插件)
    - [注意事项](#注意事项)
  - [训练定制](#训练定制)
    - [数据准备](#数据准备)
      - [从本地文件创建数据集](#从本地文件创建数据集)
    - [模型训练](#模型训练)
  - [TaskFlow 调用](#TaskFlow 调用)
  - [更多使用案例](#更多使用案例)
  - [模型列表](#模型列表)
  - [References](#references)


## 简介
代码生成是根据编程人员的输入，生成出编程人员想要的代码，能够帮助编程人员甚至独立生成代码，提高编程效率。


### 特色

本项目是基于预训练语言模型 CodeGen 的代码生成，具有以下优势：
- **效果领先**。CodeGen（16B）在 HumanEval benchmark 上评估指标已经超过[OpenAI's Codex](https://arxiv.org/pdf/2107.03374.pdf)。
- **免费的 Github Copilot**。支持通过 Github Copilot 调用该模型，让你免费体验代码 AI 助理。
- **支持自定义数据集训练**。可增加自己的代码数据加以微调，让其更智能。
- **开箱即用**。本项目提供 TaskFlow 接口，无需训练，仅需几行代码便可预测。


## 效果展示

- Github Copilot 代码提示效果展示
<p align="center">
<img src="https://user-images.githubusercontent.com/24390500/189046785-6c04a3c3-ce89-4684-9aff-a7dc2e7a7041.gif"/> <br />
</p>

- 解算法题效果展示。求解无重复字符的最长子串的长度
```python
from paddlenlp import Taskflow

prompt = "def lengthOfLongestSubstring(self, s: str) -> int:"
codegen = Taskflow("code_generation", model="Salesforce/codegen-2B-mono",decode_strategy="greedy_search", repetition_penalty=1.0)
print(codegen(prompt))
```
结果输出为：
```python
        if not s:
            return 0

        start = 0
        end = 0
        max_len = 0

        while end < len(s):
            if s[end] not in s[start:end]:
                max_len = max(max_len, end - start + 1)
                end += 1
            else:
                start += 1

        return max_len
```
<p align="center">
<img src="https://user-images.githubusercontent.com/24390500/182512164-946d959c-57b1-49e6-b9a5-be47281d1ee2.png"/> <br />
</p>


## Jupyter Lab 插件配置

请参考[codegenJupyterLabExt](https://github.com/chenqianhe/codegenJupyterLabExt), 感谢生态开发者[@chenqianhe](https://github.com/chenqianhe)的贡献！👏👏

## GithubCopilot 插件配置

**以 VS Code 的插件为例**

### 环境依赖
- PaddleNLP >= 2.4.0
- PaddlePaddle >= 2.3.1

其他依赖：`pip install -r requirements.txt`

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
codegen/
├── requirements.txt # 环境依赖
├── codegen_server.py # server启动脚本
├── run_clm.py # 训练评估脚本
├── run_clm.sh # 启动脚本
└── README.md # 说明文档
```

### 启动服务

```python
python codegen_server.py
```

##### 配置参数说明
在 codegen_server.py 中配置如下参数：
- `model_name_or_path`：模型名，默认为 "Salesforce/codegen-350M-mono"
- `device`：运行设备，默认为"gpu"
- `temperature`：解码参数 temperature，默认为0.5
- `top_k`：解码参数 top_k，默认为10
- `top_p`：解码参数 top_p，默认为1.0
- `repetition_penalty`：解码重复惩罚项，默认为1.0
- `min_length`：生成的最小长度，默认为0
- `max_length`：生成的最大长度，默认为16
- `decode_strategy`：解码策略，默认为"greedy_search"
- `use_fast`：是否使用 FastGeneration，可加速推理，默认为 True
- `use_fp16_decoding`：是否使用 fp16推理，可节省显存和加速推理，默认为 True

### 测试服务
```python
import openai
openai.api_key = 'dummy'
openai.api_base = 'http://127.0.0.1:8978'
result = openai.Completion.create(
    engine='codegen', prompt='def hello', max_tokens=16, temperature=0.1)
print(result)
'''
<OpenAIObject text_completion id=cmpl-dmhoeHmcw9DJ4NeqOJDQVKv3iivJ0 at 0x7fe7a81d42c0> JSON: {
  "id": "cmpl-dmhoeHmcw9DJ4NeqOJDQVKv3iivJ0",
  "choices": [
    {
      "text": "_world():\n    print(\"Hello World!\")\n\n\n#",
      "index": 0,
      "finish_reason": "stop",
      "logprobs": null,
    }
  ],
  "usage": {
    "completion_tokens": null,
    "prompt_tokens": null,
    "total_tokens": null
  }
}
'''
```
**注意**：如果要从本地访问服务器，`127.0.0.1`需要换成服务器的对外 IP。


### 配置插件
打开用户设置（[settings.json](https://code.visualstudio.com/docs/getstarted/settings#_settings-file-locations)），增加一行配置
```json
    "github.copilot.advanced": {
        "debug.overrideEngine": "codegen",
        "debug.testOverrideProxyUrl": "http://127.0.0.1:8978",
        "debug.overrideProxyUrl": "http://127.0.0.1:8978"
    },
```
接下来就可以愉快地使用了😊。


#### 注意事项
- 如果使用 FastGeneration，需要设置[codegen_server.py](#配置参数说明)中`use_fast=True`，第一次推理会涉及到编译，会耗费一些时间。
- 如果要使用自己训练好的模型，可以设置[codegen_server.py](#配置参数说明)中`model_name_or_path`为本地模型路径。
- 如果要从本地访问服务器，上述的`127.0.0.1`需要换成服务器的对外 IP。
- 如果出现下方的提示和报错，则说明 FastGeneration 没有启动成功，需要定位下失败的原因。或者也可设置`use_fast=False`，不启动 FastGeneration 加速，但推理速度会较慢。
```shell
  FastGeneration is not available, and the original version would be used instead.
```
```shell
  RuntimeError: (NotFound) There are no kernels which are registered in the unsqueeze2 operator.
  [Hint: Expected kernels_iter != all_op_kernels.end(), but received kernels_iter == all_op_kernels.end().] (at /home/Paddle/paddle/fluid/imperative/prepared_operator.cc:341)
  [operator < unsqueeze2 > error]
```
- 本代码也支持插件[fauxpilot](https://marketplace.visualstudio.com/items?itemName=Venthe.fauxpilot)，感谢[@linonetwo](https://github.com/linonetwo)测试。`settings.json`中配置"fauxpilot.server": "http://服务器 ip:8978/v1/engines"

## 训练定制

### 数据准备

#### 从本地文件创建数据集

在许多情况，我们需要使用本地数据集来训练我们的代码生成模型，本项目支持使用固定格式本地数据集文件进行训练。

本地数据集文件格式如下：
- train.json/test.json 文件格式：
每行为一个 jsonline
```text
{
    "code": "from paddlenlp.transformers import CodeGenForCausalLM\n\n\nmodel = CodeGenForCausalLM.from_pretrained('Salesforce/codegen-2B-mono')\n"
}
```

更多数据集读取格式详见[数据集加载](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_load.html#)和[自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)。


### 模型训练
运行如下命令即可在样例训练集上进行 finetune，并在样例验证集上进行验证。

```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus 0,1 run_clm.py \
            --model_name_or_path Salesforce/codegen-350M-mono \
            --block_size 1024 \
            --output_dir output \
            --train_file train.json \
            --validation_file test.json \
            --num_train_epochs 5 \
            --logging_steps 10 \
            --save_steps 1000 \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 2 \
            --learning_rate 1e-4 \
            --warmup_ratio 0.1 \
            --do_train \
            --do_eval \
            --device gpu
```
使用多卡训练可以指定多个 GPU 卡号，例如 --gpus "0,1"

关键参数释义如下：
- `gpus` 指示了训练所用的 GPU 卡号。
- `model_name_or_path` 指示了 finetune 使用的具体预训练模型，可以是 PaddleNLP 提供的预训练模型（详见[模型列表](#模型列表)），或者是本地的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含 paddle 预训练模型 model_state.pdparams。如果使用 PaddleNLP 提供的预训练模型，可以选择下面其中之一。
- `block_size` 表示训练时候数据被拆分的块数。
- `output_dir` 表示模型的保存路径。
- `train_file` 本地训练数据地址，数据格式必须与`dataset_name`所指数据集格式相同。
- `validation_file` 本地测试数据地址，数据格式必须与`dataset_name`所指数据集格式相同。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `per_device_train_batch_size` 表示训练时**每张卡**上的样本数目。
- `per_device_eval_batch_size` 表示测试时**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `warmup_ratio` 表示学习率逐渐升高到基础学习率（即上面配置的 learning_rate）所需要的迭代数占总步数的比例，最早的使用可以参考[这篇论文](https://arxiv.org/pdf/1706.02677.pdf)。
- `do_train` 表示是否训练。
- `do_eval` 表示是否评测。
- `device` 表示使用的设备，从 gpu 和 cpu 中选择。

可通过`bash run_clm.sh`启动训练，更多参数详情和参数的默认值请参考`run_clm.py`。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
./output/
│── model_config.json
│── model_state.pdparams
│── tokenizer_config.json
│── special_tokens_map.json
│── added_tokens.json
│── vocab.json
│── merges.txt
└── ...
```

**NOTE:** 如需恢复模型训练，`model_name_or_path`配置本地模型的目录地址即可。


## TaskFlow 调用
参考[TaskFlow 文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/taskflow.md)

## 更多使用案例

- 根据注释/功能描述写代码

```python
import re
import paddle
from paddlenlp.transformers import CodeGenTokenizer, CodeGenForCausalLM

# The supported models are shown in the following table
model_name = 'Salesforce/codegen-2B-mono'
# Init tokenizer
tokenizer = CodeGenTokenizer.from_pretrained(model_name)
# Init model
model = CodeGenForCausalLM.from_pretrained(model_name)

prompt = "# this function prints hello world"
inputs = tokenizer([prompt])
inputs = {k: paddle.to_tensor(v) for (k, v) in inputs.items()}
# Generate
output, score = model.generate(inputs['input_ids'],
                               max_length=128,
                               decode_strategy='greedy_search')
# Decode the result
print(
    tokenizer.decode(output[0],
                     truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
                     skip_special_tokens=True,
                     spaces_between_special_tokens=False))
```
结果输出为：
```python
def hello_world():
    print("Hello World")

hello_world()
```

## 模型列表
模型列表
| 模型名称                           | 说明                         |
| :--------------------------------- | -------------------------------- |
| Salesforce/codegen-350M-mono             | 基于 Python 数据集 BIGPYTHON 训练  |
| Salesforce/codegen-2B-mono             | 基于 Python 数据集 BIGPYTHON 训练  |
| Salesforce/codegen-6B-mono             | 基于 Python 数据集 BIGPYTHON 训练  |
| Salesforce/codegen-16B-mono             | 基于 Python 数据集 BIGPYTHON 训练  |
| Salesforce/codegen-350M-nl             | 基于自然语言数据集 THEPILE 训练  |
| Salesforce/codegen-2B-nl             | 基于自然语言数据集 THEPILE 训练  |
| Salesforce/codegen-6B-nl             | 基于自然语言数据集 THEPILE 训练  |
| Salesforce/codegen-16B-nl             | 基于自然语言数据集 THEPILE 训练  |
| Salesforce/codegen-350M-multi             | 基于多编程语言数据集 BIGQUERY 训练  |
| Salesforce/codegen-2B-multi            | 基于多编程语言数据集 BIGQUERY 训练  |
| Salesforce/codegen-6B-multi             | 基于多编程语言数据集 BIGQUERY 训练  |
| Salesforce/codegen-16B-multi             | 基于多编程语言数据集 BIGQUERY 训练  |

## References
- Nijkamp, Erik, et al. "A conversational paradigm for program synthesis." arXiv preprint arXiv:2203.13474 (2022).
- [https://github.com/features/copilot/](https://github.com/features/copilot/)
- [https://github.com/AndPuQing/Papilot](https://github.com/AndPuQing/Papilot)
