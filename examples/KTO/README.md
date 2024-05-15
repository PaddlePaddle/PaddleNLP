## KTO

ContextualAI最近提出了一种有趣的替代方案，称为Kahneman-Tversky优化（KTO），它完全根据被标记为“好”或“坏”的单个例子来定义损失函数（例如👍 或👎 在聊天UI中看到的图标），不需要像DPO那样需要成对的偏好数据。这些标签在实践中更容易获得，KTO是一种很有前途的方法，可以不断更新生产环境中运行的模型。以下是KTO模型的Paddle实现。

## 快速开始

项目的整体组织结构如下：

```
├── kto_config.py # KTO训练参数
├── kto.py # 训练主函数
├── kto_trainer.py # KTO算法的trainer实现
├── model_base.py # 模型需要使用的基函数
├── model_config.py # 模型的配置参数
├── README.md # README文档
└── utils.py # 工具类
```

### 环境准备

- Python >= 3.9.16
- PaddlePaddle >= 2.6.0
- PaddleNLP >= 2.8.0

### 数据准备

数据集使用的是`trl-lib/kto-mix-14k`，可以直接使用datasets库进行加载：

```
>>> from datasets import load_dataset
>>> train_ds = load_dataset("trl-lib/kto-mix-14k")
>>> train_ds
DatasetDict({
    train: Dataset({
        features: ['prompt', 'completion', 'label'],
        num_rows: 13500
    })
    test: Dataset({
        features: ['prompt', 'completion', 'label'],
        num_rows: 1500
    })
})
>>> train_ds['train'][0]
{'prompt': [{'content': 'Q:Question: how old julio cesar chavez when he fought de la hoya I found the following answer on Google: He holds records for most successful consecutive defenses of world titles (27), most title fights (37), most title-fight victories (31) and he is after Joe Louis with (23) for most title defenses won by knockout (21). Is that a correct answer? Yes or no.\nA:', 'role': 'user'}], 'completion': [{'content': " Yes, the information you found on Google is correct. Julio César Chávez holds several records related to world title defenses and victories, and he is considered one of the greatest boxers in history. Here is a detailed answer to your question:\n\nJulio César Chávez was born on July 12, 1962, in Ciudad Obregón, Sonora, Mexico. He began boxing at a young age and quickly made a name for himself in the sport, winning his first world title in 1984 when he defeated Mario Miranda for the WBC super featherweight championship.\n\nOver the course of his career, Chávez would go on to hold titles in three different weight classes (super featherweight, lightweight, and junior welterweight) and defend his titles a record 27 times, including 21 times by knockout. This record for most successful consecutive defenses of world titles has never been broken, and it is a testament to Chávez's skill, determination, and durability as a fighter.\n\nIn addition to his record for most successful title defenses, Chávez also holds the records for most title fights (37) and most title-fight victories (31). These records are also unbroken and demonstrate Chávez's consistent success and dominance in the ring.\n\nChávez's impressive record and achievements have earned him a place among the greatest boxers of all time, and he is often compared to other boxing legends such as Joe Louis, who holds the record for most title defenses won by knockout (23). While Chávez did not quite match Louis's record for most title defenses won by knockout, he is still widely regarded as one of the greatest boxers in history and a true legend of the sport.\n\nIn conclusion, the information you found on Google is correct: Julio César Chávez holds several records related to world title defenses and victories, and he is considered one of the greatest boxers in history. His impressive record and achievements have earned him a place among the greatest boxers of all time, and he will always be remembered as a true legend of the sport.", 'role': 'assistant'}], 'label': True}
```
### 训练

lora单卡训练：

```
python kto.py \
    --model_name_or_path=Llama-2-7b-chat-hf \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 200 \
    --output_dir=kto-aligned-model-lora \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --logging_first_step \
    --use_peft \
    --data_seed 16 \
    --lora_r=16 \
    --lora_alpha=16 \
    --bf16 \
    --do_eval \
    --evaluation_strategy steps \
    --recompute
```

- `model_name_or_path`: 基座模型的名称。
- `per_device_train_batch_size`: 根据 prompt 进行生成及训练使用的批次大小（每张卡）。
- `num_train_epochs`: 模型训练的轮数。
- `learning_rate`: 训练的学习率。
- `lr_scheduler_type`: scheduler类型，可选linear和cosine。
- `gradient_accumulation_steps`: 模型参数梯度累积的步数，可用于扩大 batch size。实际的 batch_size = per_device_train_batch_size * gradient_accumulation_steps。
- `logging_steps`: 训练日志打印间隔。
- `eval_steps`: 训练评估间隔步数。
- `output_dir`: 模型的保存路径。
- `warmup_ratio`: warmup步数占总步数的比例。
- `report_to`: 日志输出工具，包含wandb，tensorboard，visualdl。
- `logging_first_step`: 是否记录和评估第一个 `global_step`。（`bool`，可选，默认为`False`）
- `use_peft`: 是否使用lora。
- `data_seed`: 数据集的种子随机数。
- `lora_r`: LoRA 算法中rank（秩）的值，默认为8。
- `lora_alpha`: LoRA 算法的alpha的缩放参数。
- `bf16`: 是否使用 bf16 混合精度训练。
- `do_eval`: 是否需要评估。
- `evaluation_strategy`: 评估策略，默认为no。"no"：训练期间不进行评估；"steps"：在每eval_steps结束进行；"epoch"：在每个 epoch 结束时进行。
- `recompute`: 是否使用recompute训练，重计算transformer结构。

多卡训练：
```
python -u  -m paddle.distributed.launch --gpus "2,3,4,5" kto.py \
    --model_name_or_path=Llama-2-7b-chat-hf \
    --per_device_train_batch_size 4 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 100 \
    --output_dir=kto-aligned-model \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --data_seed 16 \
    --do_eval \
    --evaluation_strategy steps \
    --logging_first_step \
    --sharding "stage2" \
    --bf16 \
    --fp16_opt_level O2 \
    --sharding_parallel_degree 4 \
    --recompute
```

- `model_name_or_path`: 基座模型的名称。
- `per_device_train_batch_size`: 根据 prompt 进行生成及训练使用的批次大小（每张卡）。
- `num_train_epochs`: 模型训练的轮数。
- `learning_rate`: 训练的学习率。
- `lr_scheduler_type`: scheduler类型，可选linear和cosine。
- `gradient_accumulation_steps`: 模型参数梯度累积的步数，可用于扩大 batch size。实际的 batch_size = per_device_train_batch_size * gradient_accumulation_steps。
- `logging_steps`: 训练日志打印间隔。
- `eval_steps`: 训练评估间隔步数。
- `output_dir`: 模型的保存路径。
- `warmup_ratio`: warmup步数占总步数的比例。
- `report_to`: 日志输出工具，包含wandb，tensorboard，visualdl。
- `data_seed`: 数据集的种子随机数。
- `do_eval`: 是否需要评估。
- `evaluation_strategy`: 评估策略，默认为no。"no"：训练期间不进行评估；"steps"：在每eval_steps结束进行；"epoch"：在每个 epoch 结束时进行。
- `logging_first_step`: 是否记录和评估第一个 `global_step`。（`bool`，可选，默认为`False`）
- `bf16`: 是否使用 bf16 混合精度训练。
- `fp16_opt_level`: 混合精度策略，支持O1 自动混合精度，O2 pure fp16精度训练。
- `sharding_parallel_degree`: sharding_parallel_degree 表示sharding发生在多少路数据流之间。
- `sharding`: 是否使用Paddle的Sharding数据并行功能，用户的参数。支持sharding `stage1`, `stage2` or `stage3`。
- `recompute`: 是否使用重计算训练。可以节省显存。

## 推理
模型的推理请参考[推理](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm#4-%E6%8E%A8%E7%90%86)

## 服务化部署

模型的服务化部署请参考[服务化部署](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm#5-%E6%9C%8D%E5%8A%A1%E5%8C%96%E9%83%A8%E7%BD%B2)

## Acknowledge

我们借鉴了[trl](https://github.com/huggingface/trl/tree/main)的优秀设计实现，在此对其作者表示感谢。

## 参考文献

[1] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306). CoRR abs/2402.01306 (2024)
