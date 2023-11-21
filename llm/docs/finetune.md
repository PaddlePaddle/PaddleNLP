## 精调
目前精调统一脚本只支持[LLaMA v1/v2](./llama)、[ChatGLM-6B](./chatglm)、[ChatGLM2-6B](./chatglm2)、[Bloom](./bloom)、[OPT](./opt)、[Qwen](./qwen)，其他模型精调使用详见对应模型目录。接下来我们将以**Llama 2**为例介绍如何使用统一脚本进行SFT、LoRA、Prefix Tuning。更多LoRA、Prefix Tuning请参见[PEFT文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/peft.md)。

### 精调训练数据格式

为了方便用户测试，我们也提供示例数据集[广告生成数据集](https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz)，用户也可以仿照数据集的格式制作自己的数据集进行精调。我们支持的数据格式是每行包含一个字典，每个字典包含以下字段：

- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。

样例数据：
```
{"src": "类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结", "tgt": "裙身处采用立体蝴蝶结装饰辅以蓝色条带点缀，令衣身造型饱满富有层次的同时为其注入一丝甜美气息。将女孩清新娇俏的一面衬托而出。"}
...
```



### SFT

SFT（Supervised Fine-Tuning）依托飞桨提出的[4D混合分布式并行](https://ai.baidu.com/forum/topic/show/987996)能力，支持使用Trainer API轻松切换数据并行(DP)、[张量并行（TP, Tensor Parallelism）](https://arxiv.org/abs/1909.08053)、[流水线并行（PP, Pipeline Parallelism）](https://arxiv.org/abs/1811.06965)（目前仅支持Llama）等多种分布式训练策略。

4D 混合并行策略的最佳配置实践如图下所示，在单机内使用通信量较大，适合使用机器内卡间通信的张量并行（张量并行又称模型并行，MP）和分组参数切片（Sharding）的2D组合策略；训练千亿规模模型时，叠加流水线并行策略使用多台机器共同分担；同时叠加数据并行来增加并发数量，提升训练速度。
<div align="center">
    <img src="https://ai.bdstatic.com/file/63F5EBB1E188457ABAFD311CFC1D8658" width=50% height=50%>
</div>

```
# 张量并行分布式训练（常用）
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./llama/sft_argument.json

# 目前ChatGLM2、OPT不支持张量并行，默认使用Sharding策略（Paddle 2.5.1支持Sharding Stage2，Sharding Stage3需要使用Paddle develop版本）
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./chatglm2/sft_argument.json

# 张量并行&流水线并行分布式训练（目前仅支持Llama）
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./llama/sft_pp_argument.json
```

### LoRA

Transformer模型中包含许多Linear层需要进行密集的矩阵乘法计算，而这些通常具有全秩(full rank)特性。[LoRA](https://arxiv.org/abs/2106.09685)提出冻结预训练的权重矩阵, 通过引入两个低 rank 矩阵 $AB$(图中橙色的两个矩阵) 来近似权重的更新过程 $W_0+\Delta W=W_0+B A$ , 其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$，实验表明将输入表达随机投影到较小的子空间模型仍然可以有效地学习下游任务，并大幅降低计算的显存需求。


<div align="center">
<img src=https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/63d56558-247a-4a8d-a6ca-121c820f7534 width=50% height=50% />
</div>


PaddleNLP LoRA API支持数据并行、张量并行等多种分布式训练策略，可以通过控制`tensor_parallel_degree` 调整并行训练策略。LoRA策略默认应用在所有Linear层，可拓展至**单机LoRA微调千亿模型**。


```
# 单卡训练
python  finetune_generation.py ./llama/lora_argument.json

# 张量并行分布式训练（ChatGLM2、OPT不支持张量并行）
# 将lora_argument.json中tensor_parallel_degree修改为2
python  -u  -m paddle.distributed.launch --gpus "0,1"  finetune_generation.py ./llama/lora_argument.json
```


### Prefix Tuning

[Prefix Tuning](https://arxiv.org/abs/2101.00190)受提示学习（Prompt learning）的影响，加入的一部分 Prefix Embedding 作为连续型提示进行训练。Prefix Embedding是由专门的 Prefix Encoder 网络生成的数个张量，会以 `past_key_value` 的方式被插入到语言模型每一层的 hidden_state 之前。

<div align="center">
<img src=https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/8baf6943-4540-4c02-8540-35f977acc077 width=40% height=40% />
</div>

PaddleNLP Prefix Tuning API支持数据并行（DP）、张量并行（TP）等多种分布式训练策略，可以通过控制`tensor_parallel_degree` 调整并行训练策略。
```
# 单卡训练
python  finetune_generation.py ./llama/pt_argument.json

# 张量并行分布式训练（ChatGLM2、OPT不支持张量并行）
# 将pt_argument.json中tensor_parallel_degree修改为2
python  -u  -m paddle.distributed.launch --gpus "0,1"  finetune_generation.py ./llama/pt_argument.json
```
### 精调参数介绍
<summary>&emsp; 模型参数（ModelArgument） </summary><div>

- `model_name_or_path`: 预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。每个模型**支持模型权重**详见各模型目录。
- `use_flash_attention`: 模型是否使用FlashAttention2，默认为False。
- `lora`: 是否开启LoRA微调策略，默认为False。
- `lora_path`: LoRA参数和配置路径，对LoRA参数进行初始化，默认为None。
- `lora_rank`: LoRA算法中rank（秩）的值，默认为8。
- `prefix_tuning`: 是否使用Prefix Tuning策略，默认为False。
- `num_prefix_tokens`: Prefix Tuning策略中Prefix Token数量，默认为128。
- `from_aistudio`: 模型权重是否从Aistudio下载，默认为False。
- `save_to_aistudio`: 模型权重是否保存到Aistudio，默认为False。
- `aistudio_repo_id`: 模型权重保存到Aistudio的repo id，默认为None。
- `aistudio_repo_private`: 模型权重保存到Aistudio的repo是否为私有，默认为True。
- `aistudio_repo_license`: 模型权重保存到Aistudio的repo license，默认为"Apache License 2.0"。
- `aistudio_token`: 模型权重保存到Aistudio的token，默认为None。如果save_to_aistudio为True，且环境变量没有设置相应token，必须传入。
- `neftune`: 是否使用[NEFT](https://arxiv.org/abs/2310.05914)，进行微调。默认为False。
- `neftune_noise_alpha`: NEFT alpha参数，默认为5.0。

</div>

<summary>&emsp; 数据参数（DataArgument）</summary><div>

- `dataset_name_or_path`: 本地数据集目录或内置数据集名称，默认为None。脚本已适配单文件和多文件，会自己寻找`dataset_name_or_path/train.json` 或者 `dataset_name_or_path/train/*.json`作为训练集文件, 以及`dataset_name_or_path/dev.json` 或者 `dataset_name_or_path/dev/*.json`作为验证集文件。
- `task_name`: 用于选择内置数据集中的具体任务，默认为None。
- `eval_with_do_generation`: 在模型效果评估的时候是否调用model.generate,默认为False。设置为True时，指标为ppl, accuracy；设置为False时，指标为BLEU4/Rouge，建议将`metric_for_best_model`设为bleu4。
- `save_generation_output`: 当`eval_with_do_generation`设为True，是否将生成结果保存在`generated_output.json`文件中，默认为False。
- `intokens`:是否使用InToken数据流（减少Padding冗余计算，大幅提升有效Token计算效率），默认为False。当`eval_with_do_generation`设为True,评估过程不支持InToken数据流。。
- `src_length`: 模型输入上下文最大token长度，默认为1024。
- `max_length`:模型输入（上下文+生成内容）的最大token长度, 默认为2048。当`intokens`设为True的时候，同时也为InToken数据流模型训练输入最大长度，通常建议设为模型允许输入最大长度，同时`per_device_train_batch_size`设为1，使用`gradient_accumulation_steps`控制batch size。
- `lazy`:设置为False则使用`MapDataset`，设置为True则使用`IterDataset`，默认为False。对于数据量较大的时候建议设为True，`IterDataset`可以避免一次性将所有数据读入内存，注意需要设置`max_steps`并且`evaluation_strategy`和`save_strategy`设为`steps`

</div>


<summary>&emsp; 生成参数（GenerateArgument）</summary><div>

注：以下参数仅在`eval_with_do_generation`为True，调用model.generate()时生效。

- `top_k`: “采样”策略中为 top-k 过滤保留的最高概率标记的数量。默认为1，等价于贪心策略。
- `top_p`:“采样”策略中 top-p 过滤的累积概率。默认为1.0，表示不起作用。
</div>

<summary>&emsp; 训练参数（TrainingArguments）</summary><div>

以下仅介绍TrainingArguments部分常用参数，详情请参见[TrainingArguments文档](https://paddlenlp.readthedocs.io/zh/latest/trainer.html)。

- `output_dir`: 用于保存相关的文件目录，主要包括模型相关文件、训练过程中的checkpoint、分词器相关文件、评估的结果文件，默认为None。
- `per_device_train_batch_size`: 训练集训练过程批处理大小，对应 micro batch size，默认为8。该参数需要根据具体的数据集来设定，该参数越大，占用显存越高，训练代价越大；反之，占用显存越小，训练速度越快。
- `gradient_accumulation_steps`:梯度累积步数，顾名思义，就是将多次计算得到的梯度值进行累加，然后一次性进行参数更新，默认为1。等效于将原有训练batch size*gradient_accumulation_steps。
- `per_device_eval_batch_size`: 验证集批处理大小，对应 micro batch size，默认为8。该参数越大，占用显存越高；该参数越小，占用显存越低。
- `eval_accumulation_steps`:在将结果移动到CPU之前，累积输出张量的预测步骤数。如果如果未设置，则在移动到CPU之前，整个预测都会在GPU上累积（速度更快需要更多的显存），默认为None。
- `num_train_epochs`:模型训练的轮次，默认为3。
- `learning_rate`:优化器的初始学习率，默认为 5e-05。
- `warmup_steps`: warmup的步数，默认为0。当warmup_steps>0时，会覆盖warmup_ratio的设置。
- `logging_steps`: 日志打印的频率，仅当logging_strategy=="step"生效，默认为 500。如果希望看到较快的日志反馈或者即时的训练的速度，可以减小logging_steps。
- `evaluation_strategy`: 评估策略，默认为no。"no"：训练期间不进行评估；"steps"：在每eval_steps结束进行；"epoch"：在每个 epoch 结束时进行。
- `save_strategy`: 保存策略，默认为no。"no"：训练期间不进行评估；"steps"：在每eval_steps结束进行；"epoch"：在每个 epoch 结束时进行。
- `fp16`: 是否需要开启FP16训练，开启FP16训练可以加速训练，默认为False。
- `bf16`: 是否需要开启BF16训练，开启BF16训练可以加速训练，默认为False。
- `fp16_opt_level`: 可设置O1或者O2，在 O1 级别下，在白名单中的算子将使用 float16/bfloat16 计算，在黑名单中的算子将使用 float32 计算。在 O2 级别下，模型的参数被转换为 float16/bfloat16， 如果算子的浮点型输入全是 float16/bfloat16，算子才会采用 float16/bfloat16 计算，若任意浮点型输入是 float32 类型，算子将采用 float32 计算。默认为O1。
- `do_train`: 是否打开训练，默认为False。
- `do_eval`: 是否打开评估，默认为False。
- `disable_tqdm`: 是否关掉tqdm的进度条，默认为False。如果需要预估整体的训练时长，可以打开该配置，实时观察训练进度。
- `load_best_model_at_end`: 训练结束后是否加载最优模型，通常与`metric_for_best_model`配合使用,默认为False。
- `metric_for_best_model`: 最优模型指标，如"accuarcy"等，用于比较模型好坏，默认为None。
- `recompute`: 重计算，暂支持full策略。开启后可降低显存以达到增大batch size的目的，默认为False。
- `save_total_limit`: 保留checkpoint的个数，老的checkpoint会被删除，默认为None。
- `tensor_parallel_degree`: 此参数tensor_parallel_degree表示将一层transformer结构的份数，该方法对通信开销较大, 建议 tensor_parallel_degree<=8, 尽量使用机器内部通信。默认为-1，表示不启用张量并行。
- `pipeline_parallel_degree`: 表示划分流水线的大小.(假设该参数为4, 模型12层, 则每一个pp stage 包含3层模型) 默认值-1, 表示不启用流水线并行。

</div>


### 张量并行参数合并

我们使用张量并行（TP，Tensor Parallelism）训练过程中，为了节省TP参数合并时间通常在中间checkpoint将参数存储为多个TP参数分片，可以使用提供的分片合并参数脚本进行参数合并。

```
python merge_tp_params.py \
    --model_name_or_path ./checkpoints/llama_sft_ckpts/checkpoint-100
```

<summary>&emsp; 脚本参数介绍</summary><div>
- `model_name_or_path`: 必须，本地的TP模型参数路径，默认为None。
- `device`: 运行环境，默认为gpu。
</div>

### LoRA 参数合并

为了后续的**压缩**和**静态图推理**方便，我们提供LoRA参数合并脚本，可以将LoRA参数合并到主干模型并保存相应的权重。
```
python merge_lora_params.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --lora_path ./checkpoints/llama_lora_ckpts
```
<summary>&emsp; 脚本参数介绍</summary><div>

- `model_name_or_path`: 必须，预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。
- `lora_path`: LoRA参数和配置路径，对LoRA参数进行初始化，默认为None。
- `merge_model_path`: 必须，合并参数后保存路径，默认为None。
- `device`: 运行环境，默认为gpu。
</div>