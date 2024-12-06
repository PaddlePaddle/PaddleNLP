# RLHF PPO

提供了基于强化学习 PPO 算法对 LLM 进行人类偏好对齐的代码及完整使用示例，支持**3D 分布式并行训练以及 rollout 阶段使用预测优化进行生成加速**。其中 PPO 代码实现细节参考了 [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf)（PKU Beaver） 中的 PPO 实现，支持 reward normalization、pretraining loss 等常用的 PPO 稳定训练策略；示例使用 PKU-Alignment/safe-rlhf 提供的部分数据集和模型。后续将持续完善扩展，支持更好效果、更低成本、更高性能、更大规模的 RLHF 能力。

## 快速开始

项目整体组织结构如下：

```
.
├── reward_main.py               # reward model训练脚本
├── reward_config.json           # reward model训练配置文件
├── reward_trainer.py            # reward训练执行器py脚本
├── ppo_main.py                  # RLHF训练脚本
├── ppo_config.json              # RLHF训练配置文件
├── ppo_trainer.py               # RLHF训练执行器py脚本
├── ppo_config.json              # RLHF训练配置文件
├── trainer_utils.py             # Trainer补丁及工具py脚本
├── infer_utils.py               # 生成加速工具py脚本
├── data                         # 数据集相关目录
│ └── base.py                    # 数据集基类及工具py文件
│ └── alpaca.py                  # alpaca(raw)数据集py文件
│ └── safe_rlhf.py               # safe_rlhf(raw)数据集py文件
│ └── preference.py              # 偏好数据集py文件
│ └── prompt_only.py             # prompt only数据集py文件
│ └── supervised.py              # supervised数据集py文件
├── models                       # 模型相关目录
│ └── score_model_utils.py       # score model基类及工具py文件
│ └── score_model.py             # score model模型定义py文件
│ └── ppo_model_utils.py         # PPO loss等模型策略py文件
│ └── pp_model_utils.py          # 流水线并行补丁及工具py文件
│ └── model_pp.py                # 流水线并行模型py文件
│ └── infer_model_utils.py       # 预测加速模型补丁及工具py文件
└── README.md
```

### 环境准备

- Python >= 3.10
- PaddlePaddle >= 2.6.0
- PaddleNLP 最新版本

如需使用生成加速功能，需要安装 [paddlenlp_ops](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/csrc) ，请使用 `git clone https://github.com/PaddlePaddle/PaddleNLP.git` 克隆 PaddleNLP 代码库并且将 PaddleNLP/llm 目录的路径加入 PYTHONPATH（后续将进行完善）。安装 paddlenlp_ops 后训练时将直接开启生成加速（开启流水线并行时不支持生成加速），否则使用原生动态图进行生成。

### 数据准备

PPO 训练包括 Supervised Fine-Tuning、Reward Model Fine-Tuning、RLHF 三个阶段（可见下文训练部分），会涉及到多个数据集，下面分别介绍并给出自定义数据的方法。

#### Supervised Fine-Tuning 数据
同[LLM 精调](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm#2-%E7%B2%BE%E8%B0%83)，可以直接参考对应内容进行数据准备。

#### Reward Model Fine-Tuning 数据
Reward Model Fine-Tuning 阶段需要使用人类偏好数据。示例使用 PKU-Alignment/safe-rlhf 提供的 [PKU-Alignment/PKU-SafeRLHF-30K](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-30K) 数据集，下面是其中一条样本，这里使用其中的`prompt、response_0、response_1、better_response_id`字段来组织偏好数据（safe 字段，该数据集将 helpful 和 harmless 分开标注，示例这里使用其 helpful 标注）。

```json
{
    "prompt": "Is it possible to purchase a complete chess set at a pawn shop?",
    "response_0": "Yes, it is possible to purchase a complete chess set at a pawn shop.",
    "response_1": "Yes, it is possible to purchase a complete chess set at a pawn shop. Pawn shops may carry chess sets that have been either inherited or previously borrowed. There is also a possibility of finding individual chess pieces such as rooks, knights, bishops, queens, and kings. If you cannot find a complete set, you may be able to purchase the various chess pieces separately.",
    "is_response_0_safe": true,
    "is_response_1_safe": true,
    "better_response_id": 1,
    "safer_response_id": 1
}
```

#### RLHF 数据
RLHF 阶段使用 prompt only 数据，另外可以可选的提供额外的监督数据用于构建 LM 损失约束 RLHF 训练。示例使用 [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) 数据集（同样是人类偏好数据集，这里只使用其 prompt 字段并对 prompt 去重）。此外还使用了 [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)的数据来构建额外的损失项。

以上示例数据集在训练时将自动下载缓存使用。

#### 自定义数据
数据定义围绕 `RawSample` 和 `RawDataset` 两个预置的类进行；`RawSample` 提供了数据样本级别接入协议规范，`RawDataset` 提供了数据集级别接入协议规范；按照下面介绍的规范接入，即可通过预置的 `SupervisedDataset`、`PreferenceDataset`、`PromptOnlyDataset` 三类 RLHF 训练所需数据接口来使用自定义数据。

自定义数据集需要：
- 继承 `RawDataset` ，并定义类属性 `NAME` 用于注册数据集。
- 实现 `__init__` 方法（加载数据），`__getitem__` 方法（根据 index 获取样本并转换为 `RawSample` 对象返回）、`__len__` 方法（数据集大小）。

示例如下：

```python
from datasets import load_dataset
from data import RawDataset, RawSample

class MyRawDataset(RawDataset):
    NAME = 'my-dataset-name'

    def __init__(self, path=None) -> None:
        # Load a dataset from Hugging Face or any other data source
        # self.data = load_dataset(path or 'my-organization/my-dataset')['train']
        self.data = [{
            'col1': 'question',
            'col2': 'answer1',
            'col3': 'answer2',
            'col4': 1,  # score of answer1
            'col5': 2  # score of answer2
        }] * 10  # dummy data for example

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        # Construct a `RawSample` dictionary from your custom dataset item
        return RawSample(
            input=data['col1'],
            answer=data['col2'],
            other_answer=data['col3'],
            better=float(data['col4']) > float(data['col5']),
        )

    def __len__(self) -> int:
        return len(self.data)  # dataset size
```

其中 `RawSample` 是整个 RLHF 训练过程用到的几种数据类型的超集，如下所示，其可以桥接各训练阶段所需样本类型。在自定义数据时，对于 SFT 数据使用 `RawSample` 的 `(input, answer)` 字段；对于人类偏好数据使用 `RawSample` 的 `(input, answer, other_answer, better)` 字段；对于 prompt only 数据，使用 `RawSample` 的 `(input)`字段。

```python
class RawSample(TypedDict, total=False):
    """Raw sample type.

    For SupervisedDataset, should provide (input, answer) or (dialogue).
    For PreferenceDataset, should provide (input, answer, other_answer, better).
    For SafetyPreferenceDataset, should provide (input, answer, other_answer, safer, is_safe, is_other_safe).
    For PromptOnlyDataset, should provide (input).

    When input is a list, it would be processed as a dialogue.
    """

    # Texts
    input: NotRequired[str | list[str]]  # either `input` or `dialogue` should be provided
    """User input text."""
    answer: NotRequired[str]
    """Assistant answer text."""
    other_answer: NotRequired[str]
    """Other assistant answer text via resampling."""
    dialogue: NotRequired[list[str]]  # either `input` or `dialogue` should be provided
    """Dialogue history."""

    # Flags
    better: NotRequired[bool]
    """Whether ``answer`` is better than ``other_answer``."""
    safer: NotRequired[bool]
    """Whether ``answer`` is safer than ``other_answer``."""
    is_safe: NotRequired[bool]
    """Whether ``answer`` is safe."""
    is_other_safe: NotRequired[bool]
    """Whether ``other_answer`` is safe."""
```

如此定义的数据集将可以通过预置接口根据 `NAME` 来使用，当前内置支持`"PKU-SafeRLHF/train", "PKU-SafeRLHF/test", "PKU-SafeRLHF-30K/train", "PKU-SafeRLHF-30K/test", "PKU-SafeRLHF-10K/train", "alpaca"` 几个数据集。另外还支持使用多个数据集并指定数据比例，我们可以按照需要为每个阶段训练准备多份数据集。示例如下：

```python
from paddlenlp.transformers import AutoTokenizer
from data import PreferenceDataset

tokenizer = AutoTokenizer.from_pretrained('facebook/llama-7b')
dataset = PreferenceDataset({
    'alpaca': 0.75,
    'my-dataset-name': 0.5
}, tokenizer)
```

### 训练

PPO 完整的训练过程包括以下 3 个阶段，如下图所示（来自[DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat)）：

<p align="center">
  <img src="https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/assets/image/ppo_trainer.png?raw=true" align="middle" width = "600" />
</p>

1. Supervised Fine-Tuning (SFT)

同[LLM 精调](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm#2-%E7%B2%BE%E8%B0%83)，可以直接参考对应内容进行训练并使用其产出模型。

2. Reward Model Fine-Tuning

使用 `reward_main.py` 脚本根据 `reward_config.json` 训练奖励模型

```
python -u -m paddle.distributed.launch reward_main.py ./reward_config.json
```

`reward_config.json` 中的绝大部分参数释义同[LLM 精调](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm#2-%E7%B2%BE%E8%B0%83)，不再赘述；稍有区别的是 `train_datasets`/`eval_datasets` 分别使用数据集定义注册时的`NAME`属性给出训练和验证集。另外对于奖励模型训练有以下特殊参数配置及释义（使用 PKU-Alignment/PKU-SafeRLHF 中的默认值）：

- `normalize_score_during_training`：是否在训练过程中对奖励进行 normalize，默认为 `False`。
- `normalizer_type`：使用 normalizer 时计算 mean、var 的方式，可选`"RunningMeanStd", "ExponentialMovingAverage"`。
- `normalizer_momentum`：使用 `ExponentialMovingAverage` normalizer 时指定的 momentum ，默认为 `0.9`。
- `loss_type`：使用 token 级或是 sequence 级 loss 进行奖励模型训练，可选`"token-wise", "sequence-wise"`，默认为 `"sequence-wise"`。
- `regularization`：奖励模型训练目标中对奖励的正则化系数，默认为 `0.001`。

3. RLHF：

RLHF 阶段需要 actor model、reference model、critic model、reward model 四个模型；actor-model/reference-model 使用 SFT 模型进行 initialize/frozen；critic-model/reward-model 使用 reward 模型进行 initialize/frozen (另外注意若 SFT 使用 LoRA 请先将 LoRA 权重合并）。这里使用 PKU-Alignment/PKU-SafeRLHF 提供的 SFT 模型（[PKU-Alignment/alpaca-7b-reproduced](https://huggingface.co/PKU-Alignment/alpaca-7b-reproduced)）和 reward 模型（[PKU-Alignment/beaver-7b-v1.0-reward](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0-reward)，注意该模型只关注 helpful 未考量 harmless）作为示例，使用 `ppo_main.py` 脚本根据 `ppo_config.json` 进行 RLHF 训练。

```
# 类型提升 warning 暂时通过 loglevel 屏蔽，待后续修复
GLOG_minloglevel=2 python -u -m paddle.distributed.launch ppo_main.py ./ppo_config.json
```

`ppo_config.json` 中的绝大部分参数释义同[LLM 精调](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm#2-%E7%B2%BE%E8%B0%83)，不再赘述，重点给出以下参数配置及释义（使用 PKU-Alignment/PKU-SafeRLHF 中的默认值）：

- `train_datasets`：使用数据集定义注册时的`NAME`属性给出训练集。
- `eval_datasets`：使用数据集定义注册时的`NAME`属性给出验证集。
- `ptx_datasets`：使用数据集定义注册时的`NAME`属性给出 ptx-loss 使用的数据集，未提供时将不使用 ptx-loss。
- `actor_model_name_or_path`：actor-model/reference-model 用来 initialize/frozen 的模型名称或目录。
- `reward_model_name_or_path`：reward-model 的模型名称或目录。
- `reward_critic_model_name_or_path`：critic-model 的模型名称或目录，未提供时将使用`reward_model_name_or_path`进行 critic-model 的初始化。
- `per_device_prompt_batch_size`：训练时 prompt only 数据集读取用于 rollout 生成的批次大小（每张卡）。
- `per_device_train_batch_size`：根据 prompt 进行生成及训练使用的批次大小（每张卡）。
- `num_return_sequences`：生成时每个 prompt 生成的回复个数，即 `GenerationConfig.num_return_sequences`，所有回复都将用来训练。
- `temperature`：生成采样时使用的 `temperature` ，即 `GenerationConfig.temperature`。
- `top_p`：生成采样时 top-p-filtering 阈值，即 `GenerationConfig.top_p`。
- `repetition_penalty`：生成采样时长度惩罚系数，即 `GenerationConfig.repetition_penalty`。
- `update_iters`：一次生成的数据被使用的次数。
- `kl_coeff`：对 reward 进行 KL-Penalty 的系数。
- `clip_range_score`：对 reward 进行裁剪的阈值。
- `clip_range_value`：critic model（value function）对当前 sequence 的新值与 Experience Buffer 中旧值的差距超过该范围将进行裁剪。
- `clip_range_ratio`：将当前 sequence 的新概率与 Experience Buffer 中旧概率比值裁剪到`(1-clip_range_ratio, 1+clip_range_ratio)`范围（PPO-Clip）。
- `ptx_coeff`： 预训练损失项 ptx-loss 的系数。

另外所有 [`TrainingArguments` 支持参数配置](https://paddlenlp.readthedocs.io/zh/latest/trainer.html#trainingarguments)将为 actor-model 和 critic-model 的训练复用（如`sharding_stage`），除单独提供了 `critic_learning_rate/critic_weight_decay/critic_lr_scheduler_type/critic_warmup_ratio/critic_recompute` 这些参数支持为 critic-model 训练单独指定相应配置。actor-model 和 critic-model 的 checkpoints 将分别保存在 `outpt_dir` 所指定目录的 policy 和 value 文件夹下。

此外为了支持更高性、更大规模的 RLHF 训练提供了以下特殊参数配置，可以按需使用：
- `use_fusemt`：安装 paddlenlp_ops 后将在 rollout 生成时开启生成加速（开启流水线并行时不支持生成加速），通过此设置可以禁用生成加速。
- `eval_mode`：支持为空或者设置为 "single"、"tensor_parallel"；通常可以在使用流水线并行训练时设置为"tensor_parallel"，以此在 rollout 生成阶段使用非流水线并行模型并进行生成加速。
- `offload_level`：支持设置为"freeze_model"、"optimizer"、"train_model"或者同时使用(空格分隔），分别指示 reward+reference 两个冻结模型、actor+critic 两个训练模型的优化器状态和模型参数的 offload/reload，用于在不同阶段 model/optimizer 使用结束后及时 offload 并在下次使用时 reload 相应参数权重以节省显存。

另外注意，在使用流水线并行时（pipeline_parallel_degree 大于1）建议将 `dataloader_drop_last` 设置为 true, 以此避免不同 batch size 带来的问题。




### 推理

训练完成后可以直接使用 `outpt_dir` 所指定目录中 policy 文件夹下的 checkpoints 按照[LLM 推理](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm#4-%E6%8E%A8%E7%90%86)部分的介绍来进行推理，请参考相应部分内容。

## Acknowledge

我们借鉴了[PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf)（PKU Beaver）的优秀设计实现，在此对其作者表示感谢。

## 参考文献
- Zheng R, Dou S, Gao S, et al. Secrets of rlhf in large language models part i: Ppo[J]. arXiv preprint arXiv:2307.04964, 2023.
- Dai J, Pan X, Sun R, et al. Safe rlhf: Safe reinforcement learning from human feedback[J]. arXiv preprint arXiv:2310.12773, 2023.
