# 飞桨大语言模型
大模型全流程工具基于PaddlePaddle的4D分布式并行能力旨在提供高性能、灵活易用大模型工具，可以根据自己的需求轻易来定制化百亿和千亿大模型训练，同时支持高性能的压缩推理和服务化，最终使用大模型能力提升业务效果。

| Model | Pretrain | SFT | LoRA | PrefixTuning | Generation | Quantization |
| --- | --- | --- | --- | --- | --- | --- |
| [LLaMA v1/v2](./llama) | ✅  | ✅ | ✅ | ✅ | ✅ | ✅  |
| [ChatGLM-6B](./chatglm) |  N/A |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |
| [ChatGLM2-6B](./chatglm2) |  N/A |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |
| [Bloom](./bloom) | N/A | ✅ | ✅ | ✅ | ✅ | ✅ |
| [GPT-3](./gpt-3) |   ✅  |  ✅  |  ✅  |  WIP  | ✅    | WIP |
| [OPT](./opt) | WIP | ✅ | ✅ | WIP|  ✅ | WIP |
| [GLM](./glm) |N/A | ✅ | ✅ | WIP|  ✅ | WIP |
| [Qwen](./qwen) |N/A | ✅ | ✅ | ✅ |  ✅ | WIP |


# LLM全流程工具介绍
我们提供了模型预训练、精调（SFT、LoRA、PrefixTuning）、量化、动态图推理、服务化部署全流程脚本，开发者可以根据自己的需求定制化自己的大语言模型。

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/009bbb4e-baee-4c4a-a52e-94ac44c73c90">
</div>

<div align="center">
    <font size ="1">
    LLM全流程工具流程图（上图：PaddleNLP 2.6进展 下图：最终目标）
     </font>
</div>

## 1. 环境准备

- PaddlePaddle >= 2.5.1
- PaddleNLP >= 2.6.0
- tiktoken (仅 Qwen 需要)

## 2. 预训练
[LLaMA v1/v2](./llama)、[GPT-3](./gpt-3) 目录中提供了模型预训练的数据准备和训练细节，后续我们将支持更多的模型预训练。

## 3. 精调
目前精调统一脚本只支持[LLaMA v1/v2](./llama)、[ChatGLM-6B](./chatglm)、[ChatGLM2-6B](./chatglm2)、[Bloom](./bloom)、[OPT](./opt)、[Qwen](./qwen)，其他模型精调使用详见对应模型目录。接下来我们将以**Llama 2**为例介绍如何使用统一脚本进行SFT、LoRA、Prefix Tuning。更多LoRA、Prefix Tuning请参见[PEFT文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/peft.md)。

### 3.1 精调训练数据格式

为了方便用户测试，我们也提供示例数据集[广告生成数据集](https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz)，用户也可以仿照数据集的格式制作自己的数据集进行精调。我们支持的数据格式是每行包含一个字典，每个字典包含以下字段：

- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。

样例数据：
```
{"src": "类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结", "tgt": "裙身处采用立体蝴蝶结装饰辅以蓝色条带点缀，令衣身造型饱满富有层次的同时为其注入一丝甜美气息。将女孩清新娇俏的一面衬托而出。"}
...
```



### 3.2 SFT
SFT(Supervised Fine-Tuning)依托飞桨提出的[4D混合分布式并行](https://ai.baidu.com/forum/topic/show/987996)能力，支持使用Trainer API轻松切换数据并行(DP)、[张量并行（TP, Tensor Parallelism）](https://arxiv.org/abs/1909.08053)、[流水线并行（PP, Pipeline Parallelism）](https://arxiv.org/abs/1811.06965)（目前仅支持Llama）等多种分布式训练策略。

4D 混合并行策略如何组合？如图所示，在单机内使用通信量较大，适合使用机器内的卡间通信的张量并行（张量并行又称模型并行，MP）和分组参数切片（Sharding）的2D组合策略；训练千亿规模模型时，叠加流水线并行策略使用多台机器共同分担；同时叠加数据并行来增加并发数量，提升训练速度。
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

### 3.3 LoRA

Transformer模型中包含许多Linear层需要进行密集的矩阵乘法计算，而这些通常具有全秩(full rank)。[LoRA](https://arxiv.org/abs/2106.09685)提出冻结预训练的权重矩阵, 通过引入两个低 rank 矩阵 $AB$(图中橙色的两个矩阵) 来近似权重的更新过程 $W_0+\Delta W=W_0+B A$ , 其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$，实验表面将输入表达随机投影到较小的子空间模型仍然可以有效地学习下游任务还可以节约大量的计算显存需求。


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


### 3.4 Prefix Tuning

[Prefix Tuning](https://arxiv.org/abs/2101.00190)受提示学习（Prompt learning）的影响，加入的一部分 prefix embedding 作为连续型提示进行训练。prefix embedding是由专门的 prefix encoder 网络生成的数个张量，会以 past_key_value的方式被插入到语言模型每一层的 hidden_state之前。

<div align="center">
<img src=https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/8baf6943-4540-4c02-8540-35f977acc077 width=40% height=40% />
</div>

PaddleNLP Prefix Tuning API支持数据并行、张量并行等多种分布式训练策略，可以通过控制`tensor_parallel_degree` 调整并行训练策略。
```
# 单卡训练
python  finetune_generation.py ./llama/pt_argument.json

# 张量并行分布式训练（ChatGLM2、OPT不支持张量并行）
# 将pt_argument.json中tensor_parallel_degree修改为2
python  -u  -m paddle.distributed.launch --gpus "0,1"  finetune_generation.py ./llama/pt_argument.json
```
### 3.5 精调参数介绍
<details><summary>&emsp; 模型参数(ModelArgument) </summary><div>

- `model_name_or_path`: 预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。每个模型**支持模型权重**详见各模型目录。
- `lora`: 是否开启LoRA微调策略，默认为False。
- `lora_path`: LoRA参数和配置路径，对LoRA参数进行初始化，默认为None。
- `lora_rank`: LoRA算法中rank（秩）的值，默认为8。
- `prefix_tuning`: 是否使用Prefix Tuning策略，默认为False。
- `num_prefix_tokens`: Prefix Tuning策略中Prefix Token数量，默认为128。

</div></details>

<details><summary>&emsp; 数据参数(DataArgument) </summary><div>

- `dataset_name_or_path`: 本地数据集目录或内置数据集名称，默认为None。脚本已适配单文件和多文件，会自己寻找`dataset_name_or_path/train.json` 或者 `dataset_name_or_path/train/*.json`作为训练集文件, 以及`dataset_name_or_path/dev.json` 或者 `dataset_name_or_path/dev/*.json`作为验证集文件。
- `task_name`: 用于选择内置数据集中的具体任务，默认为None。
- `eval_with_do_generation`: 在模型效果评估的时候是否调用model.generate,默认为False。设置为True时，指标为ppl, accuracy；设置为False时，指标为BLEU4/Rouge，建议将`metric_for_best_model`设为bleu4。
- `save_generation_output`: 当`eval_with_do_generation`设为True，是否将生成结果保存在`generated_output.json`文件中，默认为False。
- `intokens`:是否使用InToken数据流（减少Padding冗余计算，大幅提升有效Token计算效率），默认为False。当`eval_with_do_generation`设为True,评估过程不支持InToken数据流。。
- `src_length`: 模型输入上下文最大token长度，默认为1024。
- `max_length`:模型输入（上下文+生成内容）的最大token长度, 默认为2048。当`intokens`设为True的时候，同时也为InToken数据流模型训练输入最大长度，通常建议设为模型允许输入最大长度，同时`per_device_train_batch_size`设为1，使用`gradient_accumulation_steps`控制batch size。
- `lazy`:设置为False则使用`MapDataset`，设置为True则使用`IterDataset`，默认为False。对于数据量较大的时候建议设为True，`IterDataset`可以避免一次性将所有数据读入内存，注意需要设置`max_steps`并且`evaluation_strategy`和`save_strategy`设为`steps`

</div></details>


<details><summary>&emsp; 生成参数(GenerateArgument) </summary><div>

注：以下参数仅在`eval_with_do_generation`为True，调用model.generate()时生效。

- `top_k`: “采样”策略中为 top-k 过滤保留的最高概率标记的数量。默认为1，等价于贪心策略。
- `top_p`:“采样”策略中 top-p 过滤的累积概率。默认为1.0，表示不起作用。
</div></details>

<details><summary>&emsp; 训练参数(TrainingArguments) </summary><div>

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

</div></details>


### 3.6 张量并行参数合并
我们使用张量并行(TP，Tensor Parallelism)训练过程中，为了节省TP参数合并时间往往在中间checkpoint将参数存储为多个TP参数分片，可以使用提供的分片合并参数脚本进行参数合并。

```
python merge_tp_params.py \
    --model_name_or_path ./checkpoints/llama_sft_ckpts/checkpoint-100
```

<details><summary>&emsp; 脚本参数介绍</summary><div>
- `model_name_or_path`: 必须，本地的TP模型参数路径，默认为None。
- `device`: 运行环境，默认为gpu。
</div></details>

### 3.7 LoRA参数合并
为了后续的**压缩**和**静态图推理**方便，我们提供LoRA参数合并脚本，可以将LoRA参数合并到主干模型并保存相应的权重。
```
python merge_lora_params.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --lora_path ./checkpoints/llama_lora_ckpts
```
<details><summary>&emsp; 脚本参数介绍</summary><div>

- `model_name_or_path`: 必须，预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。
- `lora_path`: LoRA参数和配置路径，对LoRA参数进行初始化，默认为None。
- `merge_model_path`: 必须，合并参数后保存路径，默认为None。
- `device`: 运行环境，默认为gpu。
</div></details>

## 4. 模型推理

### 4.1 动态图推理

```shell
# 预训练&SFT动态图模型推理
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --batch_size 1 \
    --data_file ./data/dev.json \
    --dtype "float16" \
    --mode "dynamic"

# LoRA动态图模型推理
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --batch_size 1 \
    --data_file ./data/dev.json \
    --lora_path ./checkpoints/llama_lora_ckpts \
    --mode "dynamic"

# Prefix Tuning动态图模型推理
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --batch_size 1 \
    --data_file ./data/dev.json \
    --prefix_path ./checkpoints/llama_pt_ckpts \
    --mode "dynamic"
```

### 4.2 静态图推理

```shell
# 首先需要运行一下命令将动态图导出为静态图
# LoRA需要先合并参数，详见3.7LoRA参数合并
# Prefix Tuning暂不支持
python export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --output_path ./inference \
    --dtype float16


# 静态图模型推理
python predictor.py \
    --model_name_or_path inference \
    --batch_size 1 \
    --data_file ./data/dev.json \
    --dtype "float16" \
    --mode "static"
```

### 4.3 InferenceModel 动态图推理

```shell
# InferenceModel 动态图推理
# LoRA需要先合并参数，详见3.7LoRA参数合并
# Prefix Tuning暂不支持
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --dtype float16 \
    --max_length 1024 \
    --mode "dynamic" \
    --inference_model
```

### 4.4 InferenceModel 静态图推理

```shell
# 首先需要运行一下命令将InferenceModel动态图导出为静态图
# LoRA需要先合并参数，详见3.7LoRA参数合并
# Prefix Tuning暂不支持
python export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --output_path ./inference \
    --dtype float16 \
    --inference_model

# InferenceModel 静态图推理
python predictor.py \
    --model_name_or_path ./inference \
    --dtype float16 \
    --max_length 1024 \
    --output_file "infer.json" \
    --mode "static" \
    --inference_model
```


### 4.5 参数介绍

<details><summary>&emsp; 脚本参数介绍 </summary><div>

- `model_name_or_path`: 必须，预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。
- `batch_size`: 批处理大小，默认为8。该参数越大，占用显存越高；该参数越小，占用显存越低。
- `src_length`: 模型输入上下文最大token长度，默认为1024。
- `max_length`:模型输入（上下文+生成内容）的最大token长度, 默认为2048。
- `lora_path`: LoRA参数和配置路径，对LoRA参数进行初始化，默认为None。
- `prefix_path`: Prefix Tuning参数和配置路径，对Prefix Tuning参数进行初始化，默认为None。
- `top_k`: “采样”策略中为 top-k 过滤保留的最高概率标记的数量。默认为1，等价于贪心策略。
- `top_p`:“采样”策略中 top-p 过滤的累积概率。默认为1.0，表示不起作用。
- `temperature`:“采样”策略中会对输出logit除以temperature。默认为1.0，表示不起作用。
- `data_file`:必须，待推理json文件，默认为None。
- `output_file`:保存推理结果文件名，默认为output.json。
- `device`: 运行环境，默认为gpu。
- `dtype`: 模型参数dtype，默认为None。如果没有传入`lora_path`、`prefix_path`则必须传入
- `model_type`: 初始化不同类型模型，gpt-3: GPTForCausalLM; ernie-3.5-se: Ernie35ForCausalLM; 默认为 None。
- `mode`: 使用动态图或者静态图推理，值为：[dynamic, static]，默认为 dynamic。
- `inference_model`: 是否使用InferenceModel 推理，默认值为 False。

</div></details>

## 5. 服务化部署

### 5.1 环境准备

- python >= 3.8
- gradio
- flask

### 5.2 Flask & Gradio UI服务化部署

我们提供了一套简单易用的UI服务化部署脚本:


```
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" flask_server.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --port 8010 \
    --flask_port 8011 \
    --src_length 1024 \
    --dtype "float16"
```

<details><summary>&emsp; 脚本参数介绍</summary><div>


- `port`: Gradio UI 服务端口号，默认8011。
- `flask_port`: Flask服务端口号，默认8010。
- 其他参数请参见动态图推理中参数。

</div></details>

## 6. 量化

**注**：量化后模型暂不支持推理，相关开源工作正在进行中，敬请期待。
量化算法可以将模型输入和模型权重用更低比特数值表示，能够有效减少内存占用和计算开销。下面我们提供PTQ、GPTQ两种量化算法结合**PaddleSlim自研策略**进行量化，更多技术细节详见[量化策略详细教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/quant/advanced_quantization.md)

### 6.1 环境安装
- PaddleSlim develop版本
- PaddlePaddle develop版本

### 6.2 数据准备

量化中默认使用训练集作为校正（Calibartion）数据集，开发集作为评估数据集。如果希望使用其他数据作为校正数据集，则在数据目录下新增`quant.json`文件，文件格式请参照精调训练数据格式。

### 6.3 PTQ量化

```
python  finetune_generation.py ./llama/ptq_argument.json
```

### 6.4 GPTQ量化

```
python  finetune_generation.py ./llama/gptq_argument.json
```

### 6.5 量化参数介绍

<details><summary>&emsp; 量化参数(QuantArgument)</summary><div>

- `quant_type`: PTQ,QAT量化类型，默认为A8W8。支持A8W8,WINT4，WINT8：A8W8指对激活（输入）进行INT8量化，对模型权重进行INT8量化；WINT4指仅对模型权重进行INT4量化，后续使用WeightOnly进行推理；WINT8指仅对模型权重进行INT8量化，后续使用WeightOnly进行推理。
- `do_ptq`: 是否进行PTQ量化，默认为False。
- `ptq_step`: PTQ量化步数，也即模型前向次数，默认为32。
- `shift`: 是否在PTQ量化前进行[Shift策略](https://arxiv.org/abs/2304.09145)，默认为False。使用Shift策略需要设`do_ptq`为True。
- `shift_all_linear`: 是否对模型中所有Linear层应用Shift，如果为True，将会对非LayerNorm-Linear组合的Linear进行Shift，并且添加两个op，默认为False
- `shift_sampler`: Shift策略使用的sampler，默认为none。可选none，ema：none指直接利用MinMax计算Shift中的零点；ema指使用指数平均计算Shift中零点。
- `shift_step`: Shift采样步数，也即模型前向次数，默认为32。
- `smooth`: 是否在PTQ量化前进行[SmoothQuant策略](https://arxiv.org/abs/2211.10438)，默认为False。使用Smooth策略需要设`do_ptq`为True。
- `smooth_all_linears`: 是否对模型中所有Linear层应用Smooth，如果为True，将会对非LayerNorm-Linear组合的Linear进行Smooth，并且添加两个op，默认为False
- `smooth_sampler`: Smooth策略使用的sampler，默认为none，可选none，multi_step。multi_step会保存多轮前向结果进行计算，需要更大的显存。
- `smooth_step`: Smooth采样步数，也即模型前向次数，默认为32。
- `smooth_piecewise_search`: Smooth是否进行分段搜索,默认为False。分段搜索根据数值大小将激活分成K段，对于每一段进行alhpa和scale的搜索。
- `smooth_k_piece`: 使用分段搜索功能时分段数量，默认为3。根据经验建议10B模型设置为3，100B模型设置为6。
- `smooth_search_piece`: 使用分段搜索功能时，是否搜索分段数量，默认为False。设为True时，`smooth_k_piece`建议设为6，搜索分段数量耗时较长，如需加速Smooth过程建议关闭。
- `do_gptq`: 是否进行GPTQ量化，GPTQ对模型进行WINT4量化，相比于普通PTQ量化精度更高，量化时间较长。默认为False。
- `gptq_step`: GPTQ量化步数，也即模型前向次数，默认为8。
</div></details>


<details><summary>&emsp; 其他参数</summary><div>

- `per_device_train_batch_size`: 量化前向批大小，默认为8。量化过程只有模型前向，相比于普通训练需要显存较少。

- 更多参数详见精调参数介绍。

</div></details>
