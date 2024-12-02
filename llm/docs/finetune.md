# 飞桨大模型套件精调文档

## 1.飞桨精调特色
大模型精调（Supervised Fine-Tuning，SFT）作为大语言模型（LLM）的重要一环，其主要目标是使模型能够遵循指令输出预期回答，有效提升通用模型在特定的领域和应用场景的效果，更好的满足大模型的个性化应用。
是一种用于改进和定制预训练大语言模型的方法。

- 易用并行策略：支持纯数据并行（Data Parallelism）、分组参数切片的数据并行（Sharding Parallelism）、张量模型并行（Tensor Parallelism）、流水线模型并行（Pipeline Parallelism）、序列并行(Sequence parallelism)。
- 多种精度训练：16/32bit 全量精调、4/8/16bit LoRA 精调、混合量化 LoRA 精调。
- 性能极致优化：FlashAttention-2、FlashMask、Greedy Zero Padding。
- 先进精调策略：LoRA+、PiSSA、rsLoRA、NEFTune、VeRA。

更多算法原理细节详见[飞桨大模型常见算法文档](algorithm_overview.md)

## 2.大模型精调介绍

下面我们将介绍 SFT 常用的技术：
<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/4556e9f0-d855-418f-914f-bcecccce6dba">
</div>
<div align="center">
    <font size ="1">
    飞桨与 Huggingface Transformers 微调性能比对
     </font>
</div>

- 全量精调: 最常用的 SFT 技术，在指令数据集上重新训练预训练模型的所有参数。这种方法通常能提供最佳结果，但需要大量的计算资源。

- LoRA: 低秩适配（Low-Rank Adaptation）是最常用参数高效微调(PEFT，Parameter-Efficient Fine-Tuning)技术。它不是重新训练整个模型，而是冻结权重，并在每个目标线性层引入低秩矩阵。这使得 LoRA 所需训练的参数数量大幅减少（少于1%），从而减少了内存使用和训练时间。

- QLoRA:量化感知低秩适配（Quantized Low-Rank Adaptation）与标准 LoRA 相比，它可额外减少多达33%的内存使用，使其在 GPU 内存受限的情况下尤为有用。QLoRA 通常比普通 LoRA 多花费约20%的时间，但其显著的内存节省使其在 GPU 内存有限的情况下成为唯一可行的选择。


## 3. 快速开始

接下来我们将以**Llama 3**为例介绍如何使用统一脚本进行 SFT 全参精调和 LoRA 精调。
### 3.1 环境准备

- PaddlePaddle 3.0-beta
- PaddleNLP   3.0.0b2
- PaddleSlim develop

git clone 代码到本地，即可开始。

```bash
    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    # pip install ./PaddleNLP 使用develop版本
    cd PaddleNLP/llm
    # 到达运行目录
```

### 3.2 精调数据准备

为了方便用户测试，我们也提供示例数据集[广告生成数据集](https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz)，用户也可以仿照数据集的格式制作自己的数据集进行精调。我们支持的数据格式是每行包含一个字典，每个字典包含以下字段：

- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。

样例数据：
```
{"src": "类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结", "tgt": "裙身处采用立体蝴蝶结装饰辅以蓝色条带点缀，令衣身造型饱满富有层次的同时为其注入一丝甜美气息。将女孩清新娇俏的一面衬托而出。"}
...
```

### 3.3 全参精调

```
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/sft_argument.json
```

1. `zero_padding`和`greedy_zero_padding`同时设为 True 有助于提高训练效率。建议将`per_device_train_batch_size`设为1，使用`gradient_accumulation_steps`控制 batch size，适当调整`max_length`取值。
2. 设置`use_flash_attention`为 True 使用 FlashAttention。在 FlashAttention 打开的基础上设置`flash_mask`为 True 使用 FlashMask。
3. SFT API 支持4D 并行策略，可以通过控制`tensor_parallel_degree`、`pipeline_parallel_degree`、 `sharding`、`sharding_parallel_degree`调整

### 3.4 LoRA/QLoRA

```
# 单卡LoRA
python  run_finetune.py ./config/llama/lora_argument.json

# 单卡QLoRA
python  run_finetune.py ./config/llama/qlora_argument.json

# 多卡LoRA
python  -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7"  run_finetune.py ./config/llama/lora_argument.json

# 多卡QLoRA
python  -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7"  run_finetune.py ./config/llama/qlora_argument.json
```

**Note:**
1. `zero_padding`和`greedy_zero_padding`同时设为 True 有助于提高训练效率。建议将`per_device_train_batch_size`设为1，使用`gradient_accumulation_steps`控制 batch size，适当调整`max_length`取值。
2. LoRA 策略默认应用在所有 Linear 层。
3. 可以通过设置`weight_quantize_algo`将主干模型量化低比特，例如'weight_only_int4','weight_only_int8'，'nf4'或'fp4'。具体参考精调参数介绍
4. 设置`use_flash_attention`为 True 使用 FlashAttention。在 FlashAttention 打开的基础上设置`flash_mask`为 True 使用 FlashMask。
5. LoRA API 支持4D 并行策略，可以通过控制`tensor_parallel_degree`、`pipeline_parallel_degree`、 `sharding`、`sharding_parallel_degree`调整并行训练策略，可拓展至**单机 LoRA 微调千亿模型**。

### 3.5 LoRA 参数合并

为了后续的**压缩**和**静态图推理**方便，我们提供 LoRA 参数合并脚本，可以将 LoRA 参数合并到主干模型并保存相应的权重。
```
python merge_lora_params.py \
    --model_name_or_path ./checkpoints/sft_ckpts \
    --lora_path ./checkpoints/lora_ckpts \
    --output_path ./checkpoints/lora_merge \
    --device "gpu" \
    --safe_serialization True
```

<summary>&emsp; 脚本参数介绍</summary><div>

- `lora_path`: LoRA 参数和配置路径，对 LoRA 参数进行初始化，默认为 None。
- `model_name_or_path`: 必须，主干模型参数路径，默认为 None。
- `merge_model_path`: 必须，合并参数后保存路径，默认为 None。
- `device`: 运行环境，默认为 gpu。
- `safe_serialization`: 是否保存为 safetensor 格式，默认为 True。
</div>

## 4.精调参数介绍
<summary>&emsp; 模型参数（ModelArgument） </summary><div>

- `model_name_or_path`: 预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为 None。每个模型**支持模型权重**详见各模型目录。
- `use_flash_attention`: 模型是否使用 FlashAttention，默认为 False。
- `flash_mask`: 模型是否使用 FlashMask，默认为 False。请在 FlashAttention 打开的基础上设置。
- `lora`: 是否开启 LoRA 微调策略，默认为 False。
- `lora_path`: LoRA 参数和配置路径，对 LoRA 参数进行初始化，默认为 None。
- `lora_rank`: LoRA 算法中 rank（秩）的值，默认为8。
- `rslora`: 是否使用 rsLoRA 算法。
- `lora_plus_scale`:  是否使用 LoRA+，设置 B 与 A 的学习率比例。
- `neftune`: 是否使用[NEFT](https://arxiv.org/abs/2310.05914)，进行微调。默认为 False。
- `neftune_noise_alpha`: NEFT alpha 参数，默认为5.0。
- `vera`: 是否开启 VeRA 微调策略，默认为 False。
- `vera_rank`: VeRA 算法中 rank（秩）的值，默认为8。
- `lokr`: 是否开启 LoKr 微调策略，默认为 False。
- `lokr_rank`: LoKr 算法中 rank（秩）的值，默认为8。
- `use_long_sequence_strategies`: 是否使用长序列扩展策略，默认为 False。
- `strategy_type`: 长序列扩展策略的类型，默认为 None。
- `strategy_name`: 长序列扩展策略的具体名称，默认为 None。
- `rope_scaling_factor`: 应用 RoPE 扩展策略时的缩放因子。
- `lora_use_mixer`: 是否开启 MosLoRA 策略。
</div>

<summary>&emsp; 数据参数（DataArgument）</summary><div>

- `dataset_name_or_path`: 本地数据集目录或内置数据集名称，默认为 None。脚本已适配单文件和多文件，会自己寻找`dataset_name_or_path/train.json` 或者 `dataset_name_or_path/train/*.json`作为训练集文件, 以及`dataset_name_or_path/dev.json` 或者 `dataset_name_or_path/dev/*.json`作为验证集文件。
- `zero_padding`:是否使用 Zero Padding 数据流（减少 Padding 冗余计算，大幅提升有效 Token 计算效率），默认为 False。当`eval_with_do_generation`设为 True,评估过程不支持 Zero Padding 数据流。
- `greedy_zero_padding`:贪心 Zero Padding 数据流，默认为 False。请在`zero_padding`设为 True 的基础上打开。
- `src_length`: 模型输入上下文最大 token 长度，默认为1024。
- `max_length`:模型输入（上下文+生成内容）的最大 token 长度, 默认为2048。当`zero_padding`设为 True 的时候，同时也为 Zero Padding 数据流模型训练输入最大长度，通常建议设为模型允许输入最大长度，同时`per_device_train_batch_size`设为1，使用`gradient_accumulation_steps`控制 batch size。
- `lazy`:设置为 False 则使用`MapDataset`，设置为 True 则使用`IterDataset`，默认为 False。对于数据量较大的时候建议设为 True，`IterDataset`可以避免一次性将所有数据读入内存，注意需要设置`max_steps`并且`evaluation_strategy`和`save_strategy`设为`steps`
- `autoregressive`: 是否使用自回归生成，即训练数据为无监督数据，默认为 False。
- `use_pose_convert`: 是否使用 PoSE 算法的数据处理，默认为 False。

</div>


<summary>&emsp; 生成参数（GenerateArgument）</summary><div>

注：以下参数仅在`eval_with_do_generation`为 True，调用 model.generate()时生效。

- `top_k`: “采样”策略中为 top-k 过滤保留的最高概率标记的数量。默认为1，等价于贪心策略。
- `top_p`:“采样”策略中 top-p 过滤的累积概率。默认为1.0，表示不起作用。
</div>

<summary>&emsp; 训练参数（TrainingArguments）</summary><div>

以下仅介绍 TrainingArguments 部分常用参数，详情请参见[TrainingArguments 文档](https://paddlenlp.readthedocs.io/zh/latest/trainer.html)。

- `output_dir`: 用于保存相关的文件目录，主要包括模型相关文件、训练过程中的 checkpoint、分词器相关文件、评估的结果文件，默认为 None。
- `per_device_train_batch_size`: 训练集训练过程批处理大小，对应 micro batch size，默认为8。该参数需要根据具体的数据集来设定，该参数越大，占用显存越高，训练代价越大；反之，占用显存越小，训练速度越快。
- `gradient_accumulation_steps`:梯度累积步数，顾名思义，就是将多次计算得到的梯度值进行累加，然后一次性进行参数更新，默认为1。等效于将原有训练 batch size*gradient_accumulation_steps。
- `per_device_eval_batch_size`: 验证集批处理大小，对应 micro batch size，默认为8。该参数越大，占用显存越高；该参数越小，占用显存越低。
- `num_train_epochs`:模型训练的轮次，默认为3。
- `learning_rate`:优化器的初始学习率，默认为 5e-05。
- `warmup_steps`: warmup 的步数，默认为0。当 warmup_steps>0时，会覆盖 warmup_ratio 的设置。
- `evaluation_strategy`: 评估策略，默认为 no。"no"：训练期间不进行评估；"steps"：在每 eval_steps 结束进行；"epoch"：在每个 epoch 结束时进行。
- `save_strategy`: 保存策略，默认为 no。"no"：训练期间不进行评估；"steps"：在每 eval_steps 结束进行；"epoch"：在每个 epoch 结束时进行。
- `fp16`: 是否需要开启 FP16训练，开启 FP16训练可以加速训练，默认为 False。
- `bf16`: 是否需要开启 BF16训练，开启 BF16训练可以加速训练，默认为 False。
- `fp16_opt_level`: 可设置 O1或者 O2，在 O1 级别下，在白名单中的算子将使用 float16/bfloat16 计算，在黑名单中的算子将使用 float32 计算。在 O2 级别下，模型的参数被转换为 float16/bfloat16， 如果算子的浮点型输入全是 float16/bfloat16，算子才会采用 float16/bfloat16 计算，若任意浮点型输入是 float32 类型，算子将采用 float32 计算。默认为 O1。
- `do_train`: 是否打开训练，默认为 False。
- `do_eval`: 是否打开评估，默认为 False。
- `recompute`: 重计算，暂支持 full 策略。开启后可降低显存以达到增大 batch size 的目的，默认为 False。
- `tensor_parallel_degree`: 此参数 tensor_parallel_degree 表示将一层 transformer 结构的份数，该方法对通信开销较大, 建议 tensor_parallel_degree<=8, 尽量使用机器内部通信。默认为-1，表示不启用张量并行。
- `pipeline_parallel_degree`: 表示划分流水线的大小.(假设该参数为4, 模型12层, 则每一个 pp stage 包含3层模型) 默认值-1, 表示不启用流水线并行。
- `sharding_parallel_degree`: 表示分组参数切片的数据并行大小. 默认值1, 表示不启用分组参数切片的数据并行。
- `sharding`:是否使用 Paddle 的 Sharding 数据并行功能，用户的参数。支持 sharding `stage1`, `stage2` or `stage3`。其中`stage2``stage3`可以和`offload`组合使用。
- `optim`:默认为`adamw`，支持`adamw`, `adamw_mini`。
</div>



<summary>&emsp; 表征微调(ReFT)参数（ReftArgument） </summary><div>

- `model_name_or_path`: 预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为 None。每个模型**支持模型权重**详见各模型目录。
- `layers`: 干预模型的那些层，默认为 all, 干预所有层。
- `position`: 干预哪些位置的 token，默认为 f7, 干预前7个 token。
- `intervention_type`: 干预网络的类型，默认为 LoReftIntervention。
- `rank`: 干预网络的低秩，默认为 8。
- `act_fn`: 干预网络中的激活函数，默认为 linear。
- `add_bias`: 干预网络中是否添加偏置，默认为 False。
- `dropout`:  干预网络中的 Dropout rate，默认为 0.00。
</div>
