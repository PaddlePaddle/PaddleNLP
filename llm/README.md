# 1. 飞桨大语言模型
飞桨大模型套件PaddleFleetX是基于PaddlePaddle的4D分布式并行能力的大模型全流程套件，旨在提供高性能、灵活易用大模型工具，可以根据自己的需求轻易来定制化百亿和千亿大模型训练，同时支持高性能的压缩推理和服务化，最终使用大模型能力提升业务效果。


# 2. 全流程适配情况
| Model | Pretrain | SFT | LoRA | PrefixTuning | Model Compression | Generation |
| --- | --- | --- | --- | --- | --- | --- |
| [LLaMA v1/v2](./llama) | ✅  | ✅ | ✅ | ✅ | ✅ | ✅  |
| [ChatGLM-6B v1](./chatglm) |  NA |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |
| [ChatGLM-6B v2](./chatglm_v2) |  NA |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |
| [Bloom](./bloom) | NA | ✅ | ✅ | ✅ | ✅ | ✅ |
| [GPT-3](./gpt-3) |   ✅  |  ✅  |  ✅  |  WIP  |  WIP  |  ✅  |
| [OPT](./opt) | WIP | ✅ | ✅ | WIP| WIP| ✅ |
| [GLM](./glm) | NA | ✅ | ✅ | WIP| WIP| ✅ |

# 3. 大预言模型全流程工具介绍
我们提供了模型预训练、精调（SFT、LoRA、PrefixTuning）、量化、动态图推理、静态图推理、服务化部署全流程脚本，开发者可以根据自己的需求定制化自己的大语言模型。

## 3.1 预训练
[LLaMA v1/v2](./llama)、[GPT-3](./gpt-3) 目录中提供了模型预训练的数据准备和训练细节，后续我们将支持更多的模型预训练。

## 3.2 精调
目前精调统一脚本只支持[LLaMA v1/v2](./llama)、[ChatGLM-6B](./chatglm)、[ChatGLM-6B v2](./chatglm_v2)、[Bloom](./bloom)、[OPT](./opt)，其他模型精调使用详见对应模型目录。接下来我们将以**ChatGLM-6B v2**为例介绍如何使用统一脚本进行SFT、[LoRA、Prefix Tuning](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/peft.md)。

### 3.2.1 精调训练数据格式

为了方便用户测试，我们也提供示例数据集[广告生成数据集](https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz)，用户也可以仿照数据集的格式制作自己的数据集进行精调。我们支持的数据格式是每行包含一个字典，每个字典包含以下字段：
- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。

### 3.2.2 SFT
SFT(Supervised Fine-Tuning)支持数据并行(DP)、[张量并行（TP, Tensor Parallelism）](https://arxiv.org/abs/1909.08053)、[流水线并行（PP, Pipeline Parallelism）](https://arxiv.org/abs/1811.06965)（仅支持Llama）等多种分布式训练策略，可以通过控制`tensor_parallel_degree`, `pipeline_parallel_degree`调整并行训练策略。
```
# 张量并行分布式训练（常用）
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./chatglm_v2/sft_argument.json

# 流水线并行分布式训练（仅支持Llama）
# 将lora_argument.json中pipeline_parallel_degree修改为4
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./chatglm_v2/sft_argument.json
```

### 3.2.3 LoRA

[LoRA](https://arxiv.org/abs/2106.09685)支持数据并行、张量并行等多种分布式训练策略，可以通过控制`tensor_parallel_degree` 调整并行训练策略。LoRA策略默认应用在所有Linear层。
```
# 单卡训练
python  finetune_generation.py ./chatglm_v2/lora_argument.json

# 张量并行分布式训练
# 将lora_argument.json中tensor_parallel_degree修改为2
python  -u  -m paddle.distributed.launch --gpus "0,1"  finetune_generation.py ./chatglm_v2/lora_argument.json
```


### 3.2.4 Prefix Tuning
[Prefix Tuning](https://arxiv.org/abs/2101.00190)支持数据并行、张量并行等多种分布式训练策略，可以通过控制`tensor_parallel_degree` 调整并行训练策略。
```
# 单卡训练
python  finetune_generation.py ./chatglm_v2/pt_argument.json

# 张量并行分布式训练
# 将pt_argument.json中tensor_parallel_degree修改为2
python  -u  -m paddle.distributed.launch --gpus "0,1"  finetune_generation.py ./chatglm_v2/pt_argument.json
```
### 3.2.5 精调参数介绍

**模型参数(ModelArgument)：**

- `model_name_or_path`: 预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。
- `lora`: 是否开启LoRA微调策略，默认为False。
- `lora_path`: LoRA参数和配置路径，对LoRA参数进行初始化，默认为None。
- `lora_rank`: LoRA算法中rank（秩）的值，默认为8。
- `prefix_tuning`: 是否使用Prefix Tuning策略，默认为False。
- `num_prefix_tokens`: Prefix Tuning策略中Prefix Token数量，默认为128。

**数据参数(DataArgument)：**
- `dataset_name_or_path`: 本地数据集目录或内置数据集名称，默认为None。
- `task_name`: 用于选择内置数据集中的具体任务，默认为None。
- `src_length`: 模型输入上下文最大长度，默认为1024。
- `tgt_length`:模型生成文本最大长度，默认为1024。
- `eval_with_do_generation`: 在模型效果评估的时候是否调用model.generate,默认为False。设置为True时，指标为ppl, accuracy；设置为False时，指标为BLEU4/Rouge，建议将`metric_for_best_model`设为bleu4。
- `save_generation_output`: 当`eval_with_do_generation`设为True，是否将生成结果保存在`generated_output.json`文件中，默认为False。
- `intokens`:是否使用InToken数据流（减少Padding冗余计算，大幅提升有效Token计算效率），默认为False。当`eval_with_do_generation`设为True,评估过程不支持InToken数据流。
- `intokens_max_length`: InToken数据流模型训练最大长度，默认为2048。

**生成参数(GenerateArgument):**

注：以下参数仅在`eval_with_do_generation`为True，调用model.generate()时生效。

- `top_k`: “采样”策略中为 top-k 过滤保留的最高概率标记的数量。默认为1，等价于贪心策略。
- `top_p`:“采样”策略中 top-p 过滤的累积概率。默认为1.0，表示不起作用。

**训练参数(TrainingArguments)：**

以下仅介绍TrainingArguments部分常用参数，详情请参见[TrainingArguments文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md#trainingarguments-%E5%8F%82%E6%95%B0%E4%BB%8B%E7%BB%8D)。

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


### 3.2.6 张量并行参数合并
我们使用张量并行(TP，Tensor Parallelism)训练过程中，为了节省TP参数合并时间往往在中间checkpoint将参数存储为多个TP参数分片，可以使用提供的分片合并参数脚本进行参数合并。

```
python merge_tp_params.py --model_name_or_path ./checkpoint --merge_model_path ./checkpoint_merge --dtype "float16" --with_tokenizer
```

**参数：**
- `model_name_or_path`: 必须，预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。
- `merge_model_path`: 必须，合并参数后保存路径，默认为None。
- `dtype`: 必须，模型参数dtype，默认为None。
- `with_tokenizer`: 是否同时保存分词器，默认为False。
- `device`: 运行环境，默认为gpu。

### 3.2.7 LoRA参数合并
为了后续的压缩和静态图推理方便，我们提供LoRA参数合并脚本，可以将LoRA参数合并到主干模型并保存相应的权重。
```
python merge_lora_params.py --model_name_or_path THUDM/chatglm2-6b --lora_path ./checkpoint
```
**参数：**
- `model_name_or_path`: 必须，预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。
- `lora_path`: LoRA参数和配置路径，对LoRA参数进行初始化，默认为None。
- `merge_model_path`: 必须，合并参数后保存路径，默认为None。
- `device`: 运行环境，默认为gpu。

## 3.3 动态图推理

```
python predict_generation.py \
    --model_name_or_path THUDM/chatglm2-6b \
    --batch_size 1 \
    --data_file ./data/dev.json \
    --dtype "float16"
```

**参数：**
- `model_name_or_path`: 必须，预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。
- `batch_size`: 批处理大小，默认为8。该参数越大，占用显存越高；该参数越小，占用显存越低。
- `src_length`: 模型输入上下文最大长度，默认为1024。
- `tgt_length`:模型生成文本最大长度，默认为1024。
- `lora_path`: LoRA参数和配置路径，对LoRA参数进行初始化，默认为None。
- `prefix_path`: Prefix Tuning参数和配置路径，对Prefix Tuning参数进行初始化，默认为None。
- `top_k`: “采样”策略中为 top-k 过滤保留的最高概率标记的数量。默认为1，等价于贪心策略。
- `top_p`:“采样”策略中 top-p 过滤的累积概率。默认为1.0，表示不起作用。
- `temperature`:“采样”策略中会对输出logit除以temperature。默认为1.0，表示不起作用。
- `data_file`:必须，待推理json文件，默认为None。
- `output_file`:保存推理结果文件名，默认为output.json。
- `device`: 运行环境，默认为gpu。
- `dtype`: 模型参数dtype，默认为None。如果没有传入`lora_path`、`prefix_path`则必须传入
- `gpt`: 是否使用GPTForCausalLM模型，默认为False。

## 3.4 服务化部署

### 3.4.1 Flask & Gradio UI服务化部署

我们提供了一套简单易用的UI服务化部署脚本:


```
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" flask_server.py \
    --model_name_or_path THUDM/chatglm2-6b \
    --port 8010 \
    --flask_port 8011 \
    --src_length 1024 \
    --dtype "float16"
```

**参数：**
其他参数请参见动态图推理中参数。
- `port`: Gradio UI 服务端口号，默认8011。
- `flask_port`: Flask服务端口号，默认8010。

## 3.5 压缩

### 3.5.1 PTQ量化

### 3.5.2 GPTQ量化
