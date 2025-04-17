# 飞桨大模型套件 DPO 文档
## 1.算法介绍
直接偏好优化 (DPO，Direct Preference Optimization) 是人类反馈的强化学习 （RLHF）的改进，对利用奖励函数与最优策略之间的映射关系，证明这个受限的奖励最大化问题可以通过单阶段的策略训练来精确优化。DPO 简化了训练流程，且增加了模型收敛的稳定性。

在 DPO 的基础上，还发展出了一些衍生算法，如 SimPO，ORPO 等等，我们可以直接通过修改 config 配置中的 loss_type 切换不同算法。


## 2.快速开始
接下来我们将以**Llama 3**为例介绍如何使用统一脚本进行 DPO。
### 2.1 环境准备
- PaddlePaddle 3.0-beta
- PaddleNLP   develop
- PaddleSlim develop

git clone 代码到本地，即可开始。

```bash
    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    # pip install ./PaddleNLP 使用develop版本
    cd PaddleNLP/llm
    # 到达运行目录
```
### 2.2 数据准备
我们支持的偏好数据格式是每行包含一个字典的 json 文件，每个字典包含以下字段：

- `src` : `str, List(str)`, 用户对话内容。
- `tgt` : `str, List(str)`, 系统回复内容。
- `response` : `str, List(str)`, 包含 chosen 和 rejected 回复。
- `sort` : `List(int)`, sort 值用于区分 response 中 chosen 和 rejected（sort 值小的是 rejected，sort 值大的是 chosen）。

样例数据：

```text
{
    "src": ["In this task, you are given a second sentence. Your task is to generate the first sentence on the same topic but incoherent and inconsistent with the second sentence.\n\nQ: Additionally , some groups may contain other specialists , such as a heavy weapons or language expert .\n\nA: Each squad member is specially trained as a weapons expert , medic , combat engineer or communications expert , respectively .\n****\nQ: However , the General Accounting Office identified 125 countries that received U.S. training and assistance for their police forces during fiscal year 1990 at a cost of at least $117 million .\n\nA: No government agency is in charge of calculating the cost .\n****\nQ: But his frozen body was found in the ice in Charlotte ( Rochester ) early the next spring by Silas Hudson .\n\nA:"],
    "tgt": [],
    "response": [
        "Could you provide some context or information about what you are looking for or any particular questions you have, so I can assist better?",
        "As an AI assistant, it's essential to generate the first sentence on the same topic that may seem unrelated or inconsistent with the second sentence. Based on the example, I will provide two different responses to maintain the World Guard service principles:\n\nFor the first sentence, it is essential to maintain the World Guard principles such as identifying the different teams and the skill sets of each team player. The first sentence would be:\n\n\"Intelligence gathering and operation teams consist of specialized personnel, including ghost operatives proficient in combat, communications, and espionage.\"\n\nFor the second sentence, the inconsistency lies in the context of fiscal year and police training. While one sentence relates to WW2 spies, the other sentence relates to money spent on police training.\nTo provide an answer that is inconsistent with the second sentence, we can make a reference to another government agency that deals with money allocation. Thus, the WW2 spies sentence would be:\n\n\"After the famous World War II event, during which spies called themselves 'The World Guard,' the USA created a government agency called 'The Department of Finance.' Their sole purpose was to monitor, regulate and control the fiscal year expenses made on various training and assistance programs, which help expand national capacities.\"\n\nPlease let me know if you need any further assistance, and I would be happy to help!"
        ],

    "sort": [1, 0]
}
...
```

为了方便测试，我们将[ultrafeedback_binarized demo](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)数据集处理成对应的数据集格式，使用方式如下：

```bash
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized.tar.gz
tar -zxvf ultrafeedback_binarized.tar.gz
```
### 2.3 DPO 训练

```bash
# DPO 启动命令参考
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/llama/dpo_argument.json

# DPO LoRA 启动命令参考
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/llama/dpo_lora_argument.json
```


## 3. DPO 参数介绍
### 模型参数（ModelArgument）
- `model_name_or_path`: 使用的预训练模型名称或者本地的模型路径，用于热启模型和分词器，每个模型支持模型权重详见各模型目录。
- `use_flash_attention`: 模型是否使用 FlashAttention，默认为 `False`。暂时只支持 llama。
- `flash_mask`: 是否使用 FlashMask，需要在 FlashAttention 打开的基础上设置。暂时只支持 llama。
- `lora`: 是否使用 LoRA 模型，默认为 `False`。
- `ref_model_update_steps`: 更新参考模型状态字典的步数，默认为 -1，表示不更新。
- `reference_free`: 是否不使用参考模型，默认为 False。SimPO 和 ORPO reference_free 强制设为 True。
- `recompute_granularity`: 重计算的粒度，默认为 `"full"`。
- `tokenizer_name_or_path`: 分词器的预训练名称或路径，如果与模型不同。
- `virtual_pp_degree`: 虚拟流水线并行度，默认为 `1`。
- `sequence_parallel`: 是否使用序列并行，默认为 `False`。
- `tensor_parallel_output`: 是否使用 tensor_parallel_output，打开可降低显存提高速度，默认为 `True`。yuan 模型设为 False。
- `weight_quantize_algo`: 模型权重量化算法，包括 `"nf4"`（qlora）、`"weight_only_int8"`。
- `lora_rank`: LoRA 中秩的值，默认为 `8`。
- `lora_path`: 用于初始化 LoRA 状态字典的路径。
- `rslora`: 是否使用 RsLoRA，rslora_plus 等价于 lora_plus_scale 为4， lora_alpha 为4，打开有利于提高模型训练收敛速度。默认为 `False`。
- `lora_plus_scale`: 在 LoRA+ 技术中，Lora B 的比例，默认为 `1.0`。
- `lora_alpha`: LoRA 的 alpha 参数，默认为 `-1`。
- `rslora_plus`: 是否增强 LoRA 的性能，默认为 `False`。
- `use_quick_lora`: 是否使用 Quick LoRA，默认为 `True`。

### 数据参数（DataArgument）
- `train_dataset_path`: 训练集数据路径，默认为 `"./data/train.jsonl"`。
- `dev_dataset_path`: 验证集数据路径，默认为 `"./data/dev.jsonl"`。
- `max_seq_len`: 输入序列的最大长度，默认为 `4096`。
- `max_prompt_len`: 输入提示的最大长度，默认为 `2048`。
- `greedy_zero_padding`: 是否使用 greedy zero padding，打开有利于降低 padding 比例，默认为 `False`。
- `lazy`: 是否返回`MapDataset` 或者`IterDataset`。`True`代表`IterDataset`，`False`代表`MapDataset`。数据集较大是建议打开 lazy，注意 lazy 为 True 数据集不 shuffle。

### 训练参数（TrainingArguments）
- `output_dir`: 用于保存相关文件的目录，包括模型、checkpoint、分词器文件、评估结果等，默认为 `"./checkpoints/dpo_ckpts"`。
- `per_device_train_batch_size`: 每个设备上的训练批处理大小，默认为 `1`。
- `gradient_accumulation_steps`: 梯度累积步数，默认为 `8`，表示每 `8` 个步数进行一次参数更新。
- `per_device_eval_batch_size`: 每个设备上的验证批处理大小，默认为 `1`。
- `num_train_epochs`: 模型训练的轮次，默认为 `1`。
- `max_steps`: 训练的最大步数，默认为 `100`。
- `learning_rate`: 优化器的初始学习率，默认为 `1e-06`。
- `warmup_steps`: warmup 的步数，默认为0。当 warmup_steps>0时，会覆盖 warmup_ratio 的设置，默认为 `10`。
- `logging_steps`: 日志记录的步数间隔，默认为 `1`。
- `evaluation_strategy`: 评估策略。"no"：训练期间不进行评估；"steps"：在每 eval_steps 结束进行；"epoch"：在每个 epoch 结束时进行。
- `save_strategy`: 保存策略。"no"：训练期间不进行评估；"steps"：在每 eval_steps 结束进行；"epoch"：在每个 epoch 结束时进行。
- `eval_steps`: 评估的步数间隔，默认为 `100`。
- `save_steps`: 模型保存的步数间隔，默认为 `500`。
- `bf16`: 是否需要开启 BF16训练，开启 BF16训练可以加速训练，默认为 `True`。
- `fp16_opt_level`: 可设置 O1或者 O2，在 O1 级别下，在白名单中的算子将使用 float16/bfloat16 计算，在黑名单中的算子将使用 float32 计算。在 O2 级别下，模型的参数被转换为 float16/bfloat16， 如果算子的浮点型输入全是 float16/bfloat16，算子才会采用 float16/bfloat16 计算，若任意浮点型输入是 float32 类型，算子将采用 float32 计算。默认为 O1。默认为 `"O2"`。
- `do_train`: 是否开启训练，默认为 `True`。
- `do_eval`: 是否开启评估，默认为 `True`。
- `load_best_model_at_end`: 是否在训练结束时加载最优模型，默认为 `True`。
- `tensor_parallel_degree`: 此参数 tensor_parallel_degree 表示将一层 transformer 结构的份数，该方法对通信开销较大,但可以节约显存，建议 tensor_parallel_degree<=8, 尽量使用机器内部通信。
- `pipeline_parallel_degree`: 表示划分流水线的大小.(假设该参数为4, 模型12层, 则每一个 pp stage 包含3层模型) 默认值-1, 表示不启用流水线并行。
- `sharding_parallel_degree`: 分组参数切片的数据并行大小。
- `sharding`: 是否使用 Sharding 数据并行功能，默认为 `stage1`。
- `recompute`: 重计算，暂支持 full 策略。开启后可降低显存以达到增大 batch size 的目的，full recompute 降低速度大约30%。
- `recompute_granularity`: 重计算粒度，可设置为`full`或`full_attn`或`core_attn`。
- `unified_checkpoint`: 是否使用统一的 checkpoint，默认为 `True`。
- `autotuner_benchmark`: 是否启用 autotuner 基准测试，默认为 `False`。
- `benchmark`: 是否开启基准测试，默认为 `False`。
- `optim`:默认为`adamw`，支持`adamw`, `adamw_mini`。
### DPO 参数（DPOArguments）
- `beta`: DPO 损失函数的 beta 参数，默认为 0.1。
- `simpo_gamma`: SimPO 损失函数的 gamma 参数，默认为 0.5。
- `label_smoothing`: 标签平滑比率，默认为 0.0。
- `loss_type`: DPO 损失函数类型，sigmoid([DPO](https://arxiv.org/abs/2305.18290)),
hinge([RSO](https://arxiv.org/abs/2309.06657)),
ipo([IPO](https://arxiv.org/abs/2310.12036)),
kto_pair(有偏好数据对的实现[KTO](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf)),
sppo_hard([SPPO](https://arxiv.org/pdf/2405.00675)),
nca_pair([NCA](https://arxiv.org/abs/2402.05369)),
dpop([DPOP](https://arxiv.org/pdf/2402.13228.pdf)),
orpo([ORPO](https://arxiv.org/abs/2403.07691)),
simpo([SimPO](https://arxiv.org/abs/2405.14734)),默认为 `sigmoid`。
- `pref_loss_ratio`: DPO 损失比率，默认为 1.0。
- `sft_loss_ratio`: SFT 损失比率，默认为 0.0。
- `dpop_lambda`: dpop_lambda，默认为 50，详情可见论文[DPOP](https://arxiv.org/pdf/2402.13228)

## 4. DPO 数据流介绍
在 DPO 的数据流中，我们首先将原始的数据集进行预处理，然后构造 DPO 的数据序列，并构造 attention_mask。序列包括提示（问题），chosen（偏好回答）和 rejected（拒绝回答）。
<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/2e1d91bf-8b90-4a84-b800-cc7cf4c02f58">
</div>
<div align="center">
    <font size ="1">
    序列构造
     </font>
</div>

序列构造完成后我们需要将多个序列构造为一个合并序列，并填充上 pad tokens，使每个构造后的合并序列长度相同。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/3185440c-b290-4d3b-8665-ec5bda1cda23">
</div>
<div align="center">
    <font size ="1">
    序列拼接
     </font>
</div>

在训练过程中，我们通过重新构造 attention_mask 的方式，无需考虑 Attention 计算过程中序列边界的问题。

序列拼接后重新构造 attention_mask。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/88d09f09-ebe6-4250-b8aa-e9e35db5b9d3">
</div>
<div align="center">
    <font size ="1">
    attention_mask 示意图
     </font>
</div>
