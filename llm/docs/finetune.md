# 飞桨大模型套件精调文档

## 1.精调特色介绍

- **Zero Padding策略**

模型的输入是定长序列数据，每个文本的序列长度不一样，所以是变长的序列，一般的做法是使用pad token进行填充，通常会占训练token 50%或更多。Zero Padding策略提出在单条数据中拼接多个文本为长文本，使用attention_mask保证精度对齐。通常使用Zero Padding策略时会将batch size设为1，训练过程中没有pad token参与计算，有效提高模型训练效率。精调训练只需要添加一个`zero_padding`为`True`的配置，即可开启Zero Padding训练。


<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/fe1090f4-e4ed-4e4c-b029-d4f8672f8df2">
</div>
<div align="center">
    <font size ="1">
    Zero Padding策略图示意，能够有效减少无效Pad Token进行训练
     </font>
</div>



- **PEFT结合低比特和分布式策略**

PEFT(Parameter-Efficient Fine-Tuning)相比于全量参数大大降低了所需的显存资源，但对于百亿级别的模型对训练资源仍然要求很高。为了减少显存资源占用，PEFT中提供将16位浮点数的主干模型转化为4比特或8比特的量化模型，只有当权重参与计算时才将低比特的主干模型反量化为浮点数模型。PaddleNLP中提供量化为**INT4、INT8、NF4、FP4**等多种低比特数据类型。

对于千亿参数级别的模型，PEFT配合低比特策略并不能在单卡训练。PaddleNLP中支持上述所有PEFT策略包含低比特策略使用数据并行（data parallel）、张量并行（tensor parallel）、流水线并行（pipeline parallel）策略、分组参数切分并行
（Sharding）。PEFT、低比特策略、分布式能力三者组合，PaddleNLP在有限计算资源下，可以将模型微调拓展到单机(80G * 8)**千亿参数级别**。
<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/698ef0b3-4424-4a23-942f-1bcc88be812e">
</div>

- **统一对话模板**

当前开源Chat 类型模型越来越多，PaddleNLP 已经集成了 [LLaMA/LLaMA2](../llama)、[Baichuan/Baichuan2](../llama)、[ChatGLM](../chatglm)、[ChatGLM2/ChatGLM3](./chatglm2)、[Qwen](../qwen)、[Bloom](../bloom)、[GPT-3](./gpt-3)等系列模型，也支持[多轮对话 Prompt Template 推理](https://paddlenlp.readthedocs.io/zh/latest/get_started/chat_template.html)，只需要调用`apply_chat_template` 函数即可构造将对话历史和用户最新 query 按照模型指定规则拼接到一起，实现不同模型的定制化 Prompt 规则推理。

此外多轮对话训练精调的应用场景也是越来越多，不同模型的多轮对话模板构造规则都不一致，为了在训练侧标准化前处理上的区别，设计了`chat_template`来解决此问题。只需要添加一个`chat_template` 的配置即可为该模型添加相应的多轮对话精调训练支持，具体的配置可看[多轮对话文档](./chat_template.md)。

## 2. 快速开始

接下来我们将以**Llama 2**为例介绍如何使用统一脚本进行SFT、LoRA、Prefix Tuning。
### 2.1 环境准备

- PaddlePaddle develop
- PaddleNLP  develop
- PaddleSlim develop

git clone 代码到本地，即可开始。

```bash
    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    # pip install ./PaddleNLP 使用develop版本
    cd PaddleNLP/llm
    # 到达运行目录
```

### 2.2 精调数据准备

为了方便用户测试，我们也提供示例数据集[广告生成数据集](https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz)，用户也可以仿照数据集的格式制作自己的数据集进行精调。我们支持的数据格式是每行包含一个字典，每个字典包含以下字段：

- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。

样例数据：
```
{"src": "类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结", "tgt": "裙身处采用立体蝴蝶结装饰辅以蓝色条带点缀，令衣身造型饱满富有层次的同时为其注入一丝甜美气息。将女孩清新娇俏的一面衬托而出。"}
...
```

### 2.3 SFT

SFT（Supervised Fine-Tuning）模型全参微调依托飞桨提出的[4D混合分布式并行](https://ai.baidu.com/forum/topic/show/987996)能力，支持使用Trainer API轻松切换数据并行(DP)、[张量并行（TP, Tensor Parallelism）](https://arxiv.org/abs/1909.08053)、[流水线并行（PP, Pipeline Parallelism）](https://arxiv.org/abs/1811.06965)（目前仅支持Llama）等多种分布式训练策略。

```
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/sft_argument.json
```

1. `zero_padding`设为True有助于提高训练效率。建议将`per_device_train_batch_size`设为1，使用`gradient_accumulation_steps`控制batch size，适当调整`max_length`取值。
2. 设置`use_flash_attention`为True使用FlashAttention。
3. SFT API支持4D并行策略，可以通过控制`tensor_parallel_degree`、`pipeline_parallel_degree`、 `sharding`、`sharding_parallel_degree`调整

### 2.4 LoRA

```
# 单卡训练
python  run_finetune.py ./config/llama/lora_argument.json

# 张量并行分布式训练
python  -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7"  run_finetune.py ./config/llama/lora_argument.json
```

**Note:**
1. `zero_padding`设为True有助于提高训练效率。建议将`per_device_train_batch_size`设为1，使用`gradient_accumulation_steps`控制batch size，适当调整`max_length`取值。
2. LoRA策略默认应用在所有Linear层
3. 可以通过设置`weight_quantize_algo`将主干模型量化低比特，例如'weight_only_int4','weight_only_int8'，'nf4'或'fp4'。具体参考精调参数介绍
4. 设置`use_flash_attention`为True使用FlashAttention。
5. LoRA API支持4D并行策略，可以通过控制`tensor_parallel_degree`、`pipeline_parallel_degree`、 `sharding`、`sharding_parallel_degree`调整并行训练策略，可拓展至**单机LoRA微调千亿模型**。
6. LoRA策略默认应用在所有Linear层。
7. 可以通过修改`lora_rank`改变LoRA算法中rank（秩）的值。

### 2.5 Prefix Tuning

```
# 单卡训练
python  run_finetune.py ./llama/pt_argument.json

# 张量并行分布式训练
python  -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7"  run_finetune.py ./llama/pt_argument.json
```

**Note:**
1. `zero_padding`设为True有助于提高训练效率。建议将`per_device_train_batch_size`设为1，使用`gradient_accumulation_steps`控制batch size，适当调整`max_length`取值。
2. 可以通过设置`weight_quantize_algo`将主干模型量化低比特，例如'weight_only_int4','weight_only_int8'，'nf4'或'fp4'。具体参考精调参数介绍
3. 设置`use_flash_attention`为True使用FlashAttention。
4. Prefix Tuning API支持4D并行策略，可以通过控制`tensor_parallel_degree`、`pipeline_parallel_degree`、 `sharding`、`sharding_parallel_degree`调整并行训练策略，可拓展至**单机LoRA微调千亿模型**。
5. 可以通过`num_prefix_tokens`控制Prefix Tuning策略中Prefix Token数量。


## 3.精调参数介绍
<summary>&emsp; 模型参数（ModelArgument） </summary><div>

- `model_name_or_path`: 预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。每个模型**支持模型权重**详见各模型目录。
- `use_flash_attention`: 模型是否使用FlashAttention，默认为False。
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
- `zero_padding`:是否使用Zero Padding数据流（减少Padding冗余计算，大幅提升有效Token计算效率），默认为False。当`eval_with_do_generation`设为True,评估过程不支持Zero Padding数据流。。
- `src_length`: 模型输入上下文最大token长度，默认为1024。
- `max_length`:模型输入（上下文+生成内容）的最大token长度, 默认为2048。当`zero_padding`设为True的时候，同时也为Zero Padding数据流模型训练输入最大长度，通常建议设为模型允许输入最大长度，同时`per_device_train_batch_size`设为1，使用`gradient_accumulation_steps`控制batch size。
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


## 4.分布式策略参数合并

**如果开启unified_checkpoint则不需要合参**。我们使用张量并行（TP，Tensor Parallelism）和 流水线并行（PP，Pipeline Parallelism）训练过程中，为了节省TP参数合并时间通常在中间checkpoint将参数存储为多个TP和PP参数分片，可以使用提供的分片合并参数脚本进行参数合并。

```
python merge_tp_and_pp_params.py \
    --model_name_or_path ./checkpoints/llama_sft_ckpts/checkpoint-100 \
    --pp 2 --tp 4
```

<summary>&emsp; 脚本参数介绍</summary><div>
- `model_name_or_path`: 必须，本地的TP模型参数路径，默认为None。
- `device`: 运行环境，默认为gpu。
</div>

## 5.LoRA 参数合并

为了后续的**压缩**和**静态图推理**方便，我们提供LoRA参数合并脚本，可以将LoRA参数合并到主干模型并保存相应的权重。
```
python merge_lora_params.py \
    --model_name_or_path ./checkpoints/sft_ckpts \
    --lora_path ./checkpoints/lora_ckpts \
    --output_path ./checkpoints/lora_merge \
    --device "gpu" \
    --safe_serialization True
```

<summary>&emsp; 脚本参数介绍</summary><div>

- `lora_path`: LoRA参数和配置路径，对LoRA参数进行初始化，默认为None。
- `model_name_or_path`: 必须，主干模型参数路径，默认为None。
- `merge_model_path`: 必须，合并参数后保存路径，默认为None。
- `device`: 运行环境，默认为gpu。
- `safe_serialization`: 是否保存为safetensor格式，默认为True。
</div>
