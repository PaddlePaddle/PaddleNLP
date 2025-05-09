# 自动并行使用说明
本 README 详细介绍了如何使用自动并行进行大模型的预训练、SFT（监督微调）、LoRA（低秩适应）、DPO（直接偏好优化）以及推理。

## 目录
- [自动并行使用说明](#自动并行使用说明)
  - [目录](#目录)
  - [当前支持模型](#当前支持模型)
  - [环境准备](#环境准备)
  - [预训练](#预训练)
    - [数据准备](#数据准备)
    - [启动预训练](#启动预训练)
  - [监督微调(SFT)](#监督微调 sft)
    - [数据准备](#数据准备-1)
    - [启动微调](#启动微调)
  - [低秩适应（LoRA）](#低秩适应 lora)
  - [DPO](#dpo)
  - [推理](#推理)
    - [动态图推理](#动态图推理)
    - [静态图推理](#静态图推理)
  - [FAQ](#faq)

## 当前支持模型
| Model | Pretrain | SFT |  LoRA | DPO |
|-------|----------|-----|-----|-----|
| GPT-3 |    ✅    |  🚧   |  🚧  | 🚧   |
| Llama |    ✅    |  ✅   |  ✅  | ✅   |
| Qwen  |    ✅    |  🚧   |  🚧  | 🚧   |
| DeepSeek-V3| ✅   |  🚧   |  🚧  | 🚧   |

- ✅: Supported
- 🚧: In Progress

注：当前提供的 DeepSeek-v3模型配置脚本为一个规模较小的示例 demo（调小了网络层数），以支持在单机8卡的环境下运行，如果你想运行完整671B 规模的 DeepSeek-v3，需要将层数配置为61层，并对应地调整并行策略。当前自动并行提供的 deepseek-v3版本中，暂未集成 FP8、DeepEP 等优化策略。

## 环境准备

1.安装 PaddlePaddle 最新版本

首先，您需要安装最新的`Paddle`， 推荐使用`Nightly`版本。访问 [Paddle 官网](https://www.paddlepaddle.org.cn/install/quick?docurl=undefined) 获取安装指导。

2.Paddle 安装验证

```python
import paddle
print(paddle.utils.run_check())
```
3.安装 PaddleNLP 及自定义算子

请访问[PaddleNLP 安装教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/zh/get_started/installation.rst)获取安装指导。


## 预训练

### 数据准备

项目提供了预先处理好的数据方便用户测试模型，下载到 `data` 目录下：

```shell
mkdir -p data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.{bin,idx}
```

### 启动预训练

#### GPU 启动预训练

- 动态图模式

```python
# Llama pretrain example
# assume that cur dir is auto_parallel
# cd ${PaddleNLP_Path}/llm/auto_parallel/
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7"            \
    --log_dir "llama_auto_3d"           \
    ./llama/run_pretrain_auto.py ./llama/pretrain_argument.json
```

该配置下运行`facebook/llama-7b`预训练任务，并行策略为 MP2-PP2-DP2，分片策略为 Stage1。
更多可配置参数，请参考`ModelArguments`, `DataArguments`, `PreTrainingArguments`。

- 动转静模式
<br>追加 `to_static`参数

#### XPU 启动预训练

除了 GPU，XPU 也支持自动并行，目前支持 llama 模型 7b 和 13b，更多模型支持正在开发中。

用户可以使用 `PaddleNLP/llm/auto_parallel/llama` 目录下的 `run_llama2_7b_xpu.sh` 和 `run_llama2_13b_xpu.sh` 脚本启动 XPU 上的预训练任务。

```shell
# cd ${PaddleNLP_Path}/llm/auto_parallel/llama
bash run_llama2_7b_xpu.sh
# or
bash run_llama2_13b_xpu.sh
```

Llama 7b 并行策略为 DP8，分片策略为 Stage1。Llama 13b 并行策略为 DP2-PP4，分片策略为 Stage1。


## 监督微调(SFT)
### 数据准备

项目提供预处理好的精调数据方便用户测试模型，下载并解压到`data`目录下：

```shell
wget -O AdvertiseGen.tar.gz https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz
tar -xvf AdvertiseGen.tar.gz
```

### 启动微调

- 动态图模式
```python
# Llama finetune example
# assume that cur dir is auto_parallel
# cd ${PaddleNLP_Path}/llm/auto_parallel/
python -u -m paddle.distributed.launch \
  --gpus "0,1,2,3,4,5,6,7" \
  ./run_finetune_auto.py ./llama/finetune_argument.json
```
该配置下运行`Meta-Llama-3.1-8B-Instruct`任务，并行策略为 MP2-PP2-DP2，分片策略为 Stage2。
更多可配置参数，请参考`GenerateArgument`, `ModelAutoConfig`, `ReftArgument`, `DataConfig`, `SFTAutoConfig`。

- 动转静模式
<br>追加`to_static`参数

## 低秩适应（LoRA）
在 SFT 基础上启用，开启`lora`, `lora_rank`参数。
更多的参数，可以参考[model_config.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/trl/model_config.py)。

## DPO
### 数据准备
为了方便测试，我们将 [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) 的数据集处理成对应的数据集格式，可以在 PaddleNLP/llm 目录下运行：
```shell
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized.tar.gz
tar -zxvf ultrafeedback_binarized.tar.gz
```

### 启动 DPO 训练
可以在 PaddleNLP/llm/auto_parallel/llama 目录下运行：
```shell
bash llama_dpo_with_api.sh
```
同样，可以通过配置`to_static`开关控制是否使用动转静模式。

## 推理
推理流程包括：动态图推理，动转静导出模型 -> 静态图推理。

### 动态图推理
当前自动并行任务保存的模型参数已支持用于动态图推理。以动态图自动并行训练（DP2-MP2-PP2）为例：
- 分布式 ckpt 合并为单卡模型参数

```python
import paddle
import paddle.distributed as dist

ckpt_path='/path/for/dist_ckpt'
# offload=1, 参数 offload 到 CPU，减少显存占用
# prefix="model" 参数可用于过滤掉非模型参数，例如 optimizer 状态等
merged_state_dict = dist.checkpoint.load_state_dict.load_merged_state_dict(ckpt_path, offload=1, prefix="model")
paddle.save(merged_state_dict, 'model_state.pdparams')

# 上述合并的模型参数格式为Paddle原生格式，如需转换为unified checkpoint格式(safetensors)，或需获取模型参数的index文件，继续执行如下代码：
python PaddleNLP/llm/auto_parallel/utils/convert_to_safetensors.py --input_path input_path  [--output_path output_path] [--split_num split_num] [--offload] [--as_safetensors]

# 参数介绍
--input_path: 输入的单卡模型参数路径
--output_path: 可选，输出模型参数路径，默认为'./temp'
--split_num: 可选，输出的模型参数分片数，默认为 1
--offload: 可选，选项用于控制是否将参数 offload 到 CPU
--as_safetensors: 可选，选项用于控制是否将模型参数转换为 safetensors 格式
```

- 动态图推理
<br>请参考[大模型推理教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/inference.md)。

### 静态图推理
动转静导出模型、静态图推理步骤请参考 [LLaMA 系列大模型运行文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/llama.md)。

## FAQ

Q1: 出现 OOM 如何调整?
- 减少 batch_size
- 开启 fuse_attention_ffn, fuse_flash_qkv
