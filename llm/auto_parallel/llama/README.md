# LLaMA 自动并行使用说明
本 README 详细介绍了如何使用 LLaMA 自动并行进行大模型的预训练、SFT（监督微调）、LoRA（低秩适应）、DPO（直接偏好优化）以及推理。

## 目录
- [LLaMA 自动并行使用说明](#llama-自动并行使用说明)
  - [目录](#目录)
  - [环境准备](#环境准备)
  - [自动并行策略配置](#自动并行策略配置)
  - [预训练](#预训练)
    - [数据准备](#数据准备)
    - [启动预训练](#启动预训练)
  - [监督微调(SFT)](#监督微调sft)
    - [数据准备](#数据准备-1)
    - [启动微调](#启动微调)
  - [低秩适应（LoRA）](#低秩适应lora)
  - [DPO](#dpo)
  - [PPO](#ppo)
  - [推理](#推理)
    - [动态图推理](#动态图推理)
    - [静态图推理](#静态图推理)
  - [FAQ](#faq)


## 环境准备
1.安装 PaddlePaddle 最新版本

首先，您需要安装最新的`Paddle`， 推荐使用`3.0-rc`版本。访问 [Paddle 官网](https://www.paddlepaddle.org.cn/install/quick?docurl=undefined) 获取安装指导。

2.验证安装

```python
import paddle
print(paddle.utils.run_check())
```
3.安装 PaddleNLP

请访问[PaddleNLP 安装教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)获取安装指导。

## 自动并行策略配置
当前自动并行支持多种并行策略，包括数据并行（DP）、模型并行（MP）、流水线并行（PP）以及混合 ND 并行策略。
- 自动并行基础API在组网中定义分布式状态:
```python
self.gate_proj.weight = dist.shard_tensor(
    self.gate_proj.weight,
    get_mesh(self.ipp),
    [dist.Replicate(), dist.Shard(1)],
)
```

- 自动并行中层API通过配置指定并行策略:
``` python
import paddle.distributed as dist
def auto_dist_config(self, prefix=""):
    config = {
        "sp_config": {
            "parallelize_plan": {
                "llama.layers.*.self_attn.qkv_proj": dist.ColWiseParallel(),
            },
        },
        "mp_config": {
            "parallelize_plan": {
                "llama.embed_tokens": dist.RowWiseParallel(),
            },
        },
        "pp_config": {"split_spec": "llama.layers", "global_spec": "llama.global_layer"},
    }

    return config
```

>详细的配置使用说明可以参考[Paddle自动并行使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/paddle_v3_features/auto_parallel_cn.html)和[Paddle分布式API](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/Overview_cn.html)。


## 预训练
### 数据准备
项目提供了预先处理好的数据方便用户测试模型，下载到 `data` 目录下：
```shell
# llama 模型数据下载
mkdir -p data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.{bin,idx}
```
### 启动预训练

预训练脚本位于[run_pretrain_auto.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/auto_parallel/llama/run_pretrain_auto.py)。

- 动态图模式(8卡 A100代码示例)
<br>启动 shell 脚本**llama_with_api.sh**可以默认进行8卡，DP2-MP2-PP2的并行策略的预训练任务。更多可配置参数，请参考[PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/trainer.html)文档。

- 动转静模式
<br>追加 `--to_static=true`参数


## 监督微调(SFT)
### 数据准备
项目提供预处理好的精调数据方便用户测试模型，下载并解压到`data`目录下：
```shell
wget -O AdvertiseGen.tar.gz https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz
tar -xvf AdvertiseGen.tar.gz
```

### 启动微调
SFT 训练脚本位于[run_finetune_auto.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/auto_parallel/run_finetune_auto.py)。

- 动态图模式(8卡 A100代码示例)
<br>启动 shell 脚本**llama_finetune_with_api.sh**可以默认进行8卡，DP2-MP2-PP2的并行策略的微调任务。更多可配置参数，请参考[PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/trainer.html)文档。

- 动转静模式
<br>追加`--to_static=true`参数

## 低秩适应（LoRA）
在 SFT 基础上启用 LoRA 参数：
```bash
# 追加以下参数
--lora true \
--lora_rank 8
```
更多的参数以及说明，可以参考[model_config.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/trl/model_config.py)。

## DPO
TODO

## PPO
TODO

## 推理
推理流程包括：动态图推理 -> 动转静导出模型 -> 静态图推理。

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
