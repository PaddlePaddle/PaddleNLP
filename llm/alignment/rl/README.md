# GRPO && REINFORCE++

GRPO（Group Relative Policy Optimization，组相对策略优化）是 PPO（Proximal Policy Optimization，近端策略优化）算法的一种变体。与 PPO 不同，GRPO 省略了价值函数估计器。在 GRPO 中，对于每个状态 $s$，算法会从当前策略 $\pi_{\theta_{t}}$ 中采样多个动作 $a_{1}, \dots, a_{G}$。然后，GRPO 计算这些动作相对于组内其他动作的“组相对优势”（group-relative advantage），以此作为优化策略的依据。
REINFORCE++ 是经典 REINFORCE 算法的改进版本，通过融合 PPO 的关键优化技术并移除 Critic Model，实现了更加简洁高效的策略优化。相比于传统的 REINFORCE，REINFORCE++ 在 Token-Level KL 惩罚、PPO-Clip、优势标准化、奖励裁剪和奖励标准化等关键技术上进行了改进，从而提高了训练过程的效率和稳定性。

以下是详细的使用文档和示例：

## 环境依赖

* 训练环境：
1. 参考 [Paddle 官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)安装 PaddlePaddle-GPU, 要求 PaddlePaddle>=3.0
2. clone 并安装 PaddleNLP
```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```
3. 安装 paddlenlp_ops 推理算子，参考 PaddleNLP/csrc 进行安装（必需）
```shell
cd your_PaddleNLP_path/csrc
python setup_cuda.py install
```
4. 安装 fused_ln 和 fast_ln 训练算子，参考 PaddleNLP/slm/model_zoo/gpt-3/external_ops (必须)
```shell
cd your_PaddleNLP_path/slm/model_zoo/gpt-3/external_ops
python setup.py install
```

## 支持模型

|   模型系列    | 模型名称                                                                                                                                                                                                                                                                      |
|:-------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    Qwen1.5    | Qwen/Qwen1.5-0.5B, Qwen/Qwen1.5-0.5B-Chat, Qwen/Qwen1.5-1.8B, Qwen/Qwen1.5-1.8B-Chat, Qwen/Qwen1.5-4B, Qwen/Qwen1.5-4B-Chat, Qwen/Qwen1.5-7B, Qwen/Qwen1.5-7B-Chat, Qwen/Qwen1.5-14B, Qwen/Qwen1.5-14B-Chat, Qwen/Qwen1.5-32B, Qwen/Qwen1.5-32B-Chat                          |
|     Qwen2     | Qwen/Qwen2-0.5B, Qwen/Qwen2-0.5B-Instruct, Qwen/Qwen2-1.5B, Qwen/Qwen2-1.5B-Instruct, Qwen/Qwen2-7B, Qwen/Qwen2-7B-Instruct, Qwen/Qwen2-72B, Qwen/Qwen2-72B-Instruct, Qwen/Qwen2-57B-A14B, Qwen/Qwen2-57B-A14B-Instruct                                                       |
|  Qwen2-Math   | Qwen/Qwen2-Math-1.5B, Qwen/Qwen2-Math-1.5B-Instruct, Qwen/Qwen2-Math-7B, Qwen/Qwen2-Math-7B-Instruct                                                                                                                                                                          |
|    Qwen2.5    | Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B, Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B, Qwen/Qwen2.5-14B-Instruct, Qwen/Qwen2.5-32B, Qwen/Qwen2.5-32B-Instruct, |
| Qwen2.5-Math  | Qwen/Qwen2.5-Math-1.5B, Qwen/Qwen2.5-Math-1.5B-Instruct, Qwen/Qwen2.5-Math-7B, Qwen/Qwen2.5-Math-7B-Instruct                                                                                                                                                                  |
| Qwen2.5-Coder | Qwen/Qwen2.5-Coder-1.5B, Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2.5-Coder-7B, Qwen/Qwen2.5-Coder-7B-Instruct                                                                                                                                                              |

## 数据协议

### 字段说明

- src (list(str)): 经过 chat_template 处理后的 prompt 输入；或者根据需要自己拼接构造 prompt；
- tgt (list(str)): 标签内容；

### 数据示例

```json
{
    "src": ["<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\nA very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 3 inhabitants: Michael, Zoey, and Ethan. Michael was heard saying, \"Ethan is a knight if and only if Michael is a knight\". \"Zoey is a knight or Ethan is a knight,\" Zoey mentioned. Ethan asserted: \"Michael is a knave if and only if Zoey is a knave\". So who is a knight and who is a knave?\n<|im_end|>\n<|im_start|>assistant\n<think>"],
    "tgt": ["(1) Michael is a knight\n(2) Zoey is a knight\n(3) Ethan is a knight"]
}
```


### GRPO & REINFORCE++ 数据准备
我们提供了一版使用 `Qwen/Qwen2.5-7B-Instruct-1M` 的`chat template`预处理后的[KK 数据集](https://hf-mirror.com/datasets/K-and-K/knights-and-knaves)。
```
wget https://paddlenlp.bj.bcebos.com/datasets/examples/ppo-kk.tgz && tar zxf ppo-kk.tgz
```

## 训练

### GRPO && REINFORCE++ 训练配置

我们采用的配置文件放置在`llm/config/qwen/grpo_argument.yaml`中，同时我们提供了详细参数释义如下：
- `rl_algorithm`: 使用的强化学习算法，支持`grpo`、`reinforce_plus_plus`
- `actor_model_name_or_path`: actor-model 和 reference-model 模型本地的模型路径
- `reward_model_name_or_path`: reward 模型的名称或本地路径
- `use_rm_server`: 是否使用 reward model server，设置为`False`时需要提供`reward_model_name_or_path`
- `reward_server`: reward model server 的 URL 地址, 比如`http://127.0.0.1:8731`
- `logging_dir`: 日志保存的文件夹
- `logging_steps`: 训练日志打印的间隔步数
- `output_dir`: 模型参数保存目录
- `report_to`: 训练可视化工具，支持 "all"、"wandb"、"tensorboard"、"visualdl"、"none"
- `wandb_http_proxy`: 连接 wandb 使用的 HTTP 代理
- `run_name`: 实验名称
- `train_datasets`: 训练集路径
- `eval_datasets`: 验证集路径
- `prompt_key`: 数据集中 query 对应的字段名
- `response_key`: 数据集中 response 对应的字段名
- `dataloader_drop_last`: dataloader 是否丢弃最后不完整的 batch
- `balance_batch`: 该参数用于指定是否在数据并行场景下，对批次内的 token 数量进行均衡分配。若设置为 True，系统将尝试在不同并行设备间平衡 token 的分布；若设置为 False（默认值），则不进行此类均衡操作。
- `use_remove_padding`: 此参数决定是否在训练过程中去除输入数据中的 padding 部分。启用该选项（设置为 True）可有效提高训练过程中有效 token 的占比，从而提升训练效率；若设置为 False（默认值），则保留输入数据中的 padding。
- `tensor_parallel_degree`: 张量并行度
- `sequence_parallel`: 是否启用序列并行
- `sharding_parallel_degree`: sharding 并行度
- `sharding`: 分片策略，支持 "stage1" 或 "stage2"
- `sharding_parallel_config`: sharding 并行配置
- `pipeline_parallel_degree`: 流水线并行度
- `virtual_pp_degree`: 虚拟流水线并行度
- `max_prompt_len`: 生成样本时的最大生成长度， max_length 调大会增加生成时间，并且增加显存占用。注意：
max_dec_len + max_prompt_len 应当小于 max_seq_len。
- `max_dec_len`: 最大生成长度
- `min_dec_len`: 最小生成长度
- `top_p`: 生成解码超参数
- `temperature`: 生成解码超参数
- `repetition_penalty`: 生成解码超参数
- `rollout_max_num_seqs`: 单次推理可以处理的最大序列数
- `rollout_quant_type`: 量化类型，例如 "weight_only_int8"
- `seed`: 随机种子
- `global_batch_size`: 一次（一个 step）推理（rollout)采样的 prompt 数量
- `global_mini_batch_size`: actor model 更新一次参数训练的 prompt 数量
- `rollout_n`: 一个 prompt 采样的 response 数量
- `update_iters`: 同一批数据训练次数
- `per_device_logprob_batch_size`: 计算 log_probs 时，一个 batch 的样本数量
- `per_device_reward_batch_size`: critic model 计算 loss 与反向传播时，一个 batch 的的样本数量
- `per_device_value_batch_size`: critic model 前向计算 values 时，一个 batch 的的样本数量
- `per_device_train_batch_size`: actor model 计算 loss 与反向传播时，一个 batch 的样本数量
- `num_train_epochs`: 训练的 epoch 数
- `max_length`: 训练时的最大长度，应大于 `max_prompt_len` 和 `max_dec_len` 之和
- `learning_rate`: 学习率
- `lr_scheduler_type`: Actor 模型要使用的学习率调度策略。 (`str`, 可选, 默认为`linear`)
- `weight_decay`: AdamW 优化器的权重衰减
- `adam_beta1`: AdamW 优化器的 beta1
- `adam_beta2`: AdamW 优化器的 beta2
- `adam_epsilon`: AdamW 优化器的 epsilon
- `max_grad_norm`: 梯度裁剪的最大值
- `max_steps`: 总的训练步数
- `save_steps`: 模型参数保存的间隔步数
- `ignore_save_lr_and_optim`: 是否忽略保存学习率和优化器状态
- `kl_coeff`: KL 惩罚系数
- `kl_loss_coeff`: KL Loss 系数
- `pg_loss_coeff`: 策略梯度损失系数
- `entropy_coeff`: entropy loss 系数
- `clip_range_ratio`: PPO-Clip 裁剪阈值
- `clip_range_ratio_low`: PPO-Clip 裁剪下限阈值
- `clip_range_ratio_high`: PPO-Clip 裁剪上限阈值
- `clip_range_score`: reward 的剪切范围，reward 会被限制在 [-clip_range_score, clip_range_score] 范围内
- `clip_range_value`: value 模型输出的剪切范围，value 会被限制在 [-clip_range_value, clip_range_value] 范围内
- `normalize_reward`: 是否使用 reward 标准化
- `normalize_advantage`: 是否使用 advantage 标准化
- `use_fp32_compute`: 是否使用 fp32 来计算 log_prob、reward、advantage 和 loss
- `do_eval`: 是否进行评估
- `per_device_eval_batch_size`: 估 batch 大小
- `evaluation_strategy`: 评估策略，例如 `steps`
- `eval_steps`: 模型评估的间隔步数
- `use_flash_attention`: 是否启用 FlashAttention-2，默认为 False
- `use_fused_rms_norm`: 是否使用融合的 RMSNorm 算子，需安装 fused_ln
- `recompute`: Actor 模型是否使用重计算策略，开启后可节省训练显存
- `recompute_granularity`: Actor 模型的重计算的粒度，可选项为`core_attn`和`full`. `core_attn`速度快但是显存占用，`full`速度慢但是显存占用低
- `bf16`: 使用 bfloat16 精度进行模型训练和推理。
- `fp16_opt_level`: float16 精度训练模式，`O2`表示纯 float16 训练
- `amp_custom_black_list`: 自定义 AMP 黑名单
- `amp_custom_white_list`: 自定义 AMP 白名单



### GRPO 训练命令
```shell
cd your_PaddleNLP_path/llm/alignment/rl
```

```shell
# 启动 reward server
python reward_server.py
```

```shell
export PYTHONPATH=your_PaddleNLP_path/:$PYTHONPATH
export PYTHONPATH=your_PaddleNLP_path/llm:$PYTHONPATH

export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_dataloader_use_file_descriptor=False
export HF_DATASETS_DOWNLOAD_TIMEOUT=1
export FLAGS_gemm_use_half_precision_compute_type=False
export FLAGS_force_cublaslt_no_reduced_precision_reduction=True

export FLAGS_mla_use_tensorcore=0
export FLAGS_cascade_attention_max_partition_size=2048

python -u -m paddle.distributed.launch --devices "0,1,2,3" run_rl.py ../../config/qwen/grpo_argument.yaml

# QWEN32B 2k prompt + 30k response 9台8x80G 显卡训练命令如下：
# python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_rl.py ../../config/qwen/grpo_32b_argument.yaml
```
我们提供根据上述脚本可复现的[wandb 日志](https://api.wandb.ai/links/junyu/5jiulhem)。


### Reinforce++ 训练命令
```shell
cd your_PaddleNLP_path/llm/alignment/rl
```

```shell
# 启动 reward server
python reward_server.py
```

```shell
export PYTHONPATH=your_PaddleNLP_path/:$PYTHONPATH
export PYTHONPATH=your_PaddleNLP_path/llm:$PYTHONPATH

export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_dataloader_use_file_descriptor=False
export HF_DATASETS_DOWNLOAD_TIMEOUT=1
export FLAGS_gemm_use_half_precision_compute_type=False
export FLAGS_force_cublaslt_no_reduced_precision_reduction=True

export FLAGS_mla_use_tensorcore=0
export FLAGS_cascade_attention_max_partition_size=2048

python -u -m paddle.distributed.launch --devices "0,1,2,3" run_rl.py ../../config/qwen/reinforce_plus_plus_argument.yaml
```
我们提供根据上述脚本可复现的[wandb 日志](https://api.wandb.ai/links/ainlp66-netflix/injcw3ra)。

### 在线监控
在`grpo_argument.yaml`和`reinforce_plus_plus_argument.yaml`中设置的输出目录为`"logging_dir": "vdl_log"`, 可以通过以下命令查看训练过程
```shell
visualdl --logdir vdl_log --host 0.0.0.0
```

也支持 wandb 等多种监控，可设置`"logging_dir": "wandb"`，需要提前安装好 wandb 依赖并登录。
