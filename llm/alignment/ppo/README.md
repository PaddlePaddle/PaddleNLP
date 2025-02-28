# PPO

PPO（Proximal Policy Optimization，近端策略优化）是一种强化学习算法，旨在通过优化策略来最大化累积奖励。PPO 算法结合了 Policy Gradient 和‌TRPO 的优点，通过使用随机梯度上升优化一个“替代”目标函数，实现小批量更新，而不是每个数据样本只进行一次梯度更新。

以下是详细的使用文档和示例：

## 环境依赖

* 训练环境：在 python3.9的环境下安装, 可以使用如下脚本安装
```bash
bash -x scripts/install_train_env.sh gpu
```

## 数据协议

数据格式以`data/rlhf_train_data_test.jsonl`为例。

### 字段说明

- src (list(str)): 用户对话内容，可能会包含 markup 内容，如 [<search-res>]；
- tgt (list(str)): 除了最后一轮的系统多轮回复内容，以对话轮次排列，可能会包含 markup 内容，如 [<search>]；注意：len(tgt)==len(src)-1

### 数据示例

```json
{
    "src": [
        "需要你帮我写几个有创意的广告语来打开市场。",
        "目标用户是年轻人，追求时尚、个性和自我。"
    ],
    "tgt": [
        "当然！我很乐意帮助你创作几个有创意的广告语来推广你的新洗发露。请告诉我一些关于你的产品的特点，目标受众以及你希望传达的核心信息，我会根据这些信息为你提供几个创意的广告语。"
    ]
}
```

## 训练

```shell
bash scripts/ppo.sh
```

其中参数释义如下：

- `train_task_config`: 训练数据 config, 请以`config/task_ppo.json`为例
- `eval_task_config`: 评估数据 config, 请以`config/task_ppo.json`为例
- `ptx_task_config`: SFT 辅助数据, 请以`config/task_sft.json`为例，默认为""
- `actor_model_name_or_path`: PPO 中 actor-model 和 reference-model 模型本地的模型路径
- `reward_model_name_or_path`: PPO 中 reward-model 和 critic-model 模型本地的模型路径
- `use_fusemt`: 是否通过 FustMT 加速生成，默认为 True
- `use_flash_attention`: 是否启用 FlashAttention-2，默认为 False
- `output_dir`: 模型参数保存目录
- `max_seq_len`: 输入数据的最大长度，默认为 4096
- `max_dec_len`: 最大生成长度
- `min_dec_len`: 最小生成长度
- `top_p`: 生成解码超参数
- `temperature`: 生成解码超参数
- `repetition_penalty`: 生成解码超参数
- `num_return_sequences`: 生成解码超参数
- `min_learning_rate`: Actor 模型的最小学习率
- `critic_learning_rate`: Critic 模型的最小学习率
- `recompute`: Actor 模型是否使用重计算策略，开启后可节省训练显存
- `critic_recompute`: Critic 模型是否使用重计算策略，开启后可节省训练显存
- `recompute_granularity` Actor 模型的重计算的粒度，可选项为`core_attn`和`full`. `core_attn`速度快但是显存占用，`full`速度慢但是显存占用低
- `critic_recompute_granularity` Critic 模型重计算的粒度，可选项为`core_attn`和`full`. `core_attn`速度快但是显存占用，`full`速度慢但是显存占用低
- `warmup_ratio`: Actor 模型用于从 0 到 `learning_rate` 的线性 warmup 的总训练步骤的比例
- `critic_warmup_ratio`: Critic 模型用于从 0 到 `critic_learning_rate` 的线性 warmup 的总训练步骤的比例
- `lr_scheduler_type`: Actor 模型要使用的学习率调度策略。 (`str`, 可选, 默认为 `"linear"`)
- `critic_lr_scheduler_type`: Critic 模型要使用的学习率调度策略。 (`str`, 可选, 默认为 `"linear"`)
- `weight_decay`: Actor 模型除了所有 bias 和 LayerNorm 权重之外，应用于所有层的权重衰减数值。（`float`，可选，默认为 0.0）
- `critic_weight_decay`: Critic 模型除了所有 bias 和 LayerNorm 权重之外，应用于所有层的权重衰减数值。（`float`，可选，默认为 0.0）
- `max_prompt_len`: 生成样本时的最大生成长度， max_length 调大会增加生成时间，并且增加显存占用。注意：
max_dec_len + max_prompt_len 应当小于 max_seq_len。
- `per_device_prompt_batch_size`: PPO 生成样本时的批处理大小，同 micro batch size，即满足 global_batch_size = dp（data parallel）* sharding * micro batch size。batch_size 调大会增加生成时间，并且增加显存占用
- `per_device_train_batch_size`: 训练 batch 大小, 当前为了优化性能设为1，请避免更改
- `per_device_eval_batch_size`: 评估 batch 大小。
- `max_steps`: 总的训练步数
- `eval_steps`: 模型评估的间隔步数
- `max_evaluate_steps`: 模型单次评估的最大步数
- `logging_steps`: 训练日志打印的间隔步数
- `save_steps`: 模型参数保存的间隔步数
- `weight_decay`: 权重衰减数值
- `do_train`: 是否进行训练任务
- `do_eval`: 是否进行评估任务
- `fp16`: 使用 float16 精度进行模型训练和推理。
- `bf16`: 使用 bfloat16 精度进行模型训练和推理。
- `fp16_opt_level`: float16 精度训练模式，`O2`表示纯 float16 训练
