# 飞桨大模型套件 PRM 文档
## 1.算法介绍
基于过程的奖励模型（PRM, Process Reward Model），对推理过程中的每个步骤与最终答案都提供奖励信号，可以应对基于结果的奖励模型（ORM, Outcome Reward Model）对 FP ——即过程错误，答案正确——会错误地给予正向奖励的问题。

实现角度，采用 next token prediction 方式，在每个 step 的末尾进行二分类，判断当前 step 的正误。

## 2.快速开始
接下来我们将以**Llama 3**为例介绍如何使用统一脚本进行 DPO。
### 2.1 环境准备
- PaddlePaddle 3.0-beta
- PaddleNLP   develop

git clone 代码到本地，即可开始。

```bash
    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    # pip install ./PaddleNLP 使用develop版本
    cd PaddleNLP/llm
    # 到达运行目录
```
### 2.2 数据准备
我们支持的 PRM 数据格式是 json 文件，每个元素包含以下字段：
 - `src` : `str, List(str)`, 用户对话内容；
 - `tgt` : `str, List(str)`, 系统回复内容；
 - `responses` : `List(str)`, 包含每个推理 step 的回复；
 - `labels` : `List(str)`, 包含每个推理 step 的标签（包含一个正向标记与一个负向标记），列表长度需要和 `responses` 列表长度一致；

样例数据：

[
    {
        "src": [
            "Tony has $87. He needs to buy some cheese, which costs $7 a pound and a pound of beef that costs $5 a pound. After buying the beef and his cheese, he has $61 left. How many pounds of cheese did he buy?"
        ],
        "tgt": [],
        "responses": [
            "Step 1: He bought 7 / 5 = <<7/5=1.4>>1.4 pounds of beef.",
            "Step 2: He spent 7 + 5 = <<7+5=12>>12 on beef and cheese.",
            "Step 3: So, he spent 12 - 87 = 75.",
            "Step 4: That means he bought 87 - 75 = <<87-75=12>>12 pounds of cheese. The answer is: 12"
        ],
        "labels": [
            "+",
            "+",
            "-",
            "-"
        ]
    },
    ...
]

为了方便测试，我们将[Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd)数据集处理成对应的数据集格式，使用方式如下：

```bash
wget https://bj.bcebos.com/paddlenlp/datasets/examples/math-shepherd.tar.gz
tar -zxvf math-shepherd.tar.gz
```
### 2.3 PRM 训练

```bash
# PRM 启动命令参考
python -u -m paddle.distributed.launch --gpus "0,1,2,3" ./llm/alignment/rm/flashmask/run_reward.py ./llm/config/mistral/prm_flashmask_argument.json
```

## 3. PRM 参数介绍

### 训练参数（TrainingArguments）
 - `output_dir`: 用于保存相关文件的目录，包括模型、checkpoint、分词器文件、评估结果等，默认为 `"./checkpoints/dpo_ckpts"`；
 - `per_device_train_batch_size`: 每个设备上的训练批处理大小，默认为 `1`；
 - `gradient_accumulation_steps`: 梯度累积步数，默认为 `8`，表示每 `8` 个步数进行一次参数更新；
 - `per_device_eval_batch_size`: 每个设备上的验证批处理大小，默认为 `1`；
 - `num_train_epochs`: 模型训练的轮次，默认为 `1`；
 - `max_steps`: 训练的最大步数，默认为 `100`；
 - `learning_rate`: 优化器的初始学习率，默认为 `1e-06`；
 - `warmup_steps`: warmup 的步数，默认为0。当 warmup_steps>0时，会覆盖 warmup_ratio 的设置，默认为 `10`；
 - `logging_steps`: 日志记录的步数间隔，默认为 `1`；
 - `evaluation_strategy`: 评估策略。"no"：训练期间不进行评估；"steps"：在每 eval_steps 结束进行；"epoch"：在每个 epoch 结束时进行；
 - `save_strategy`: 保存策略。"no"：训练期间不进行评估；"steps"：在每 eval_steps 结束进行；"epoch"：在每个 epoch 结束时进行；
 - `eval_steps`: 评估的步数间隔，默认为 `100`；
 - `save_steps`: 模型保存的步数间隔，默认为 `500`；
 - `bf16`: 是否需要开启 BF16训练，开启 BF16训练可以加速训练，默认为 `True`；
 - `fp16_opt_level`: 可设置 O1或者 O2，在 O1 级别下，在白名单中的算子将使用 float16/bfloat16 计算，在黑名单中的算子将使用 float32 计算。在 O2 级别下，模型的参数被转换为 float16/bfloat16， 如果算子的浮点型输入全是 float16/bfloat16，算子才会采用 float16/bfloat16 计算，若任意浮点型输入是 float32 类型，算子将采用 float32 计算。默认为 O1。默认为 `"O2"`；
 - `do_train`: 是否开启训练，默认为 `True`；
 - `do_eval`: 是否开启评估，默认为 `True`；
 - `load_best_model_at_end`: 是否在训练结束时加载最优模型，默认为 `True`；
 - `tensor_parallel_degree`: 此参数 tensor_parallel_degree 表示将一层 transformer 结构的份数，该方法对通信开销较大,但可以节约显存，建议 tensor_parallel_degree<=8, 尽量使用机器内部通信；
 - `sharding_parallel_degree`: 分组参数切片的数据并行大小；
 - `sharding`: 是否使用 Sharding 数据并行功能，默认为 `stage1`；
 - `recompute`: 重计算，暂支持 full 策略。开启后可降低显存以达到增大 batch size 的目的，full recompute 降低速度大约30%；
 - `recompute_granularity`: 重计算粒度，可设置为`full`或`full_attn`或`core_attn`；
 - `unified_checkpoint`: 是否使用统一的 checkpoint，默认为 `True`。
 # PRM 参数
 - `process_reward`: 是否开启 PRM 训练，PRM 为`True`，ORM 为 `False`。

### 数据参数（DataArgument）
 - `train_dataset_path`: 训练集数据路径；
 - `dev_dataset_path`: 验证集数据路径；
 - `max_seq_le`: 输入序列的最大长度，默认为 `4096`；
 - `max_prompt_len`: 输入提示的最大长度，默认为 `2048`；
 - `autotuner_benchmark`: 是否启用 autotuner 基准测试，默认为 `False`；
 - `benchmark`: 是否开启基准测试，默认为 `False`；
 - `zero_padding`: 是否使用 zero padding，默认为 `True`；
 - `greedy_zero_padding`: 是否使用 greedy zero padding，打开有利于降低 padding 比例，默认为 `False`；
 - `lazy`: 是否返回`MapDataset` 或者`IterDataset`。`True`代表`IterDataset`，`False`代表`MapDataset`。数据集较大是建议打开 lazy，注意 lazy 为 True 数据集不 shuffle。

### 模型参数（ModelArgumen）
 - `model_name_or_path`: 使用的预训练模型名称或者本地的模型路径，每个模型支持模型权重详见各模型目录；
 - `tokenizer_name_or_path`: 分词器的预训练名称或路径，如果与模型不同；
 - `use_flash_attention`: 模型是否使用 FlashAttention，默认为 `False`；
 - `recompute_granularity`: 重计算的粒度，默认为 `"full"`；
 - `flash_mask`: 是否使用 FlashMask，需要在 FlashAttention 打开的基础上设置；
 - `virtual_pp_degree`: 虚拟流水线并行度，默认为 `1`；
 # PRM 参数
 - `placeholder_token`: 每个推理步骤最后用于 PRM 打分占位的 token，在当前模型的 tokenizer 的长度应为1，默认为 `ки`（需配合数据集使用）；
 - `reward_tokens`: 标识 PRM 打分，应由逗号分隔的两个 token 组成的字符串，第一个为正向标记，第二个为负向标记，默认为`"+,-"`（需配合数据集使用）。
