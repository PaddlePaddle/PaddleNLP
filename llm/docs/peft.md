# PaddleNLP PEFT API

PaddleNLP PEFT API 提供单卡/分布式 LoRA 和 Prefix-Tuning，用户定义好模型，数据集, 以及相应的配置，就可以快速使用 PEFT 适配模型进行低参数模型微调。

## 预备知识
### LoRA
<div align="center">
<img src=https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/63d56558-247a-4a8d-a6ca-121c820f7534 width=30% height=30% />
</div>

大模型网络中有很多的线性层，里面需要进行密集的矩阵乘法计算，而这些通常具有全秩(full rank)，较难优化计算。LoRA 论文的研究中表明, 将输入表达随机投影到较小的子空间不仅任然可以有效地学习还可以节约大量的计算显存需求。具体做法：对于预训练的权重矩阵, 通过引入两个低 rank 矩阵 $AB$(图中橙色的两个矩阵) 来近似权重的更新过程 $W_0+\Delta W=W_0+B A$ , 其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$, $r$ 远小于原权重矩阵的 rank 。训练期间, $W_0$ 参数冻结, 只对 $\mathrm{A}$ 和 $\mathrm{B}$ 两个矩阵进行梯度更新，前向传播公式如下:

$$
h=W_{0}x+BAx
$$

由于训练参数的减少，训练过程会减少很多中间变量的存储，由此节约大量的训练显存消耗。
更多算法细节参考 LoRA[论文](https://arxiv.org/abs/2106.09685)

### Prefix-tuning

<div align="center">
<img src=https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/8baf6943-4540-4c02-8540-35f977acc077 width=40% height=40% />
</div>


Prefix-tuning 是一个针对 NLG 类型下游任务的轻量级微调方案，受提示学习（Prompt learning）的影响，加入的一部分 prefix embedding 作为连续型提示进行训练。prefix embedding 是由专门的 prefix encoder 网络生成的数个张量，会以 past_key_value 的方式被插入到语言模型每一层的 hidden_state 之前。和 LoRA 类似，它也会冻结整个预训练模型的所有参数权重，只对 prefix embedding 进行梯度更新，因此训练参数量只有常规 SFT 的 0.1%。Prefix-tuning 可以在全样本下获得与 SFT 比肩的训练效果，在小样本环境下甚至可以超越 SFT。更多算法细节参考
Prefix-tuning[论文](https://arxiv.org/abs/2101.00190)

## 快速开始
### LoRA

1. 要对 model 进行 LoRA 微调，首先需要定义 LoRAConfig， 再通过 LoRAConfig 对 LoRAModel 进行构建，再通过 mark_only_lora_as_trainable 函数冻结主干参数：
```python
    from paddlenlp.peft import LoRAConfig, LoRAModel
    from paddlenlp.transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained('facebook/llama-7b')
    target_modules = [".*q_proj.*", ".*v_proj.*", ".*k_proj.*"]
    lora_rank = 8
    lora_config = LoRAConfig(
        target_modules=target_modules,
        r=lora_rank,
        lora_alpha=2 * lora_rank,
    )
    model = LoRAModel(model, lora_config)
    model.mark_only_lora_as_trainable()
    model.print_trainable_parameters()
```

2. 模型的保存和载入

LoRAModel 的保存和载入和普通的 model 没有太大区别，都是通过 save_pretrained/from_pretrained 调用
```python
    # 保存
    model.save_pretrained('lora_path')
```
Paddle 会将 LoRAModel 的矩阵 AB 权重保存为 lora_mode_state.pdparams 文件，LoRAConfig 配置保存为 lora_config.json 文件在 lora_path 目录下。
之后当需要载入模型权重进行推理时，则直接进行 from_pretrained 即可。
```python
      from paddlenlp.transformers import AutoModelForCausalLM
    + from paddlenlp.peft import LoRAModel, LoRAConfig

    # 载入
    + config = LoRAConfig.from_pretrained('lora_path')
      model = AutoModelForCausalLM.from_pretrained('facebook/llama-7b')
    + model = LoRAModel.from_pretrained(model, 'lora_path')
      model.eval()
```

### class LoRAConfig

```text
Parameters:

    --r
                        默认为 8，LoRA A/B 矩阵秩。

    --target_modules
                        指定哪些 module 需要适配 LoRA 算法，格式为module 的名字
                        或正则表达式的 List，比如, ['q', 'v'] 或者 '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'

    --trainable_modules
                        指定除LoRA参数外的需要进行梯度更新参数的 modules，格式为module 的名字
                        或正则表达式的 List，比如, ['q', 'v'] 或者 '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'

    --lora_alpha
                        默认为 8，LoRA算法的 alpha 值，int 类型

    --lora_dropout
                        默认为 0.0，dropout的比例设置，float 类型

    --merge_weights
                        默认为 False，接口将被废弃。请使用model.merge()或model.unmerge()替代。

    --trainable_bias
                        指定可训练的 bias, 可选项 ['lora', 'all']

    --enable_lora_list
                        指定是否需要使用`MergedLoRALinear`，如果不指定则默认使用
                        `LoRALinear`

    --tensor_parallel_degree
                        默认为-1，多 GPU 并行的控制参数，传入tensor_parallel_degree 必须与 base model保持一致

    --dtype
                        LoRA矩阵参数类型设置

    --head_dim
                        多头注意力的头数，只有`LoRAMergedLinear`和
                        `ColumnParallelLoRAMergedLinear`使用
```
### class LoRAModel

```text
Parameters:

    --model
                        指定 base model，必须是 nn.Layer 类型的对象

    --lora_config
                        指定 LoRAConfig 用于配置 LoRAModel

key function:

    -mark_only_lora_as_trainable()

        其作用是将模型中与LoRA相关的的一些层标记为可训练，而其他层则标记为不可训练。


    -save_pretrained(save_directory, merge_tensor_parallel)
        --save_directory
                        保存目录的路径
        --merge_tensor_parallel
                        是否合并多卡参数,默认为True

        如果merge_tensor_parallel为真且模型的配置中的张量并行度大于1，则获取可训练的state_dict，并使用_merge_trainable_tensor_parallel方法合并张量并行训练的state_dict。如果merge_tensor_parallel为真且模型的张量并行度大于1，只有主进程会进行保存操作。


    -from_pretrained(model, lora_path)
        --model
                        要加载LORA权重参数的model对象
        --lora_path
                        保存LORA权重参数和 config 的路径

        该函数用于从预先训练的模型中加载LORA权重参数，并将其设置到给定的模型中，以便在后续的任务中使用该模型进行预测或训练。


    -print_trainable_parameters()

        该函数会遍历整个权重参数列表，对于每个权重参数weight，统计所有进行梯度更新的参数，最后将信息打印出来。
```


### Prefix-tuning
1. 设置 Prefix-tuning 参数
```python
    from paddlenlp.transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained('facebook/llama-7b')

    prefix_config = PrefixConfig(
        num_prefix_tokens=64,
        num_attention_heads=model.config.n_head,
        num_hidden_layers=model.config.n_layer,
        hidden_size=model.config.hidden_size,
        prefix_projection=False,
        prefix_projection_hidden_size=model.config.hidden_size
    )
    model = PrefixModelForCausalLM(model=model, prefix_config=prefix_config)
    model.mark_only_prefix_as_trainable()
    model.print_trainable_parameters()
```

2. 模型的保存和载入

和 LoRAModel 一致，通过 save_pretrained/from_pretrained 调用
```python
    # 保存
    model.save_pretrained('prefix_path')
```
Paddle 会将 PrefixModel 中用到的 prefix_encoder(里面包含 Embedding layer 和 Linear layers)网络模型权重，PrefixConfig 配置保存为 prefix_config.json 文件在 prefix_path 路径下。
之后当需要载入模型权重进行推理时，则直接进行 from_pretrained 即可。
```python
      from paddlenlp.transformers import AutoModelForCausalLM
    + from paddlenlp.peft import PrefixModel, PrefixConfig

    # 载入
    + config = PrefixConfig.from_pretrained('prefix_path')
      model = AutoModelForCausalLM.from_pretrained('facebook/llama-7b')
    + model = PrefixModelForCausalLM.from_pretrained(model, 'prefix_path')
      model.eval()
```

### class PrefixConfig
```text
Parameters:

    --prefix_dropout
                        默认为 0.0，prefix projection dropout比例设置，float 类型

    --num_prefix_tokens
                        prefix tokens个数设定，int 类型

    --num_attention_heads
                        注意力头数设置，int 类型

    --multi_query_group_num
                        multi query group 个数设置，int 类型

    --num_hidden_layers
                        base model 的 layer层数设置，int 类型

    --hidden_size
                        base model 的 hidden size 设置，int 类型

    --prefix_projection
                        默认为 False，是否对 prefix tokens 进行 projection 操作，bool 类型

    --prefix_projection_hidden_size
                        如果 prefix_projection 设置为 True，则在这里设置
                        projection 操作的 hidden size，int 类型

    --tensor_parallel_degree
                        默认为-1，多 GPU 并行的控制参数

    --dtype
                        prefix embeddings 参数类型设置

```

### class PrefixModelForCausalLM
```text
Parameters:

    --model
                        指定 base model，必须是 nn.Layer 类型的对象

    --prefix_config
                        指定 PrefixConfig 用于配置 PrefixModelForCausalLM

    --postprocess_past_key_value
                        指定对 past_key_value 进行后处理的函数

    --pad_attention_mask
                        指定处理新增的 prefix embedding 的 pad_attention_mask函数

key function

    -mark_only_prefix_as_trainable()

        其作用是只把模型中的 Prefix embedding 和 Prefix projection 层标记为可训练，而其他层参数冻结。

    -save_pretrained(save_directory, merge_tensor_parallel)
        --save_directory
                        保存目录的路径
        --merge_tensor_parallel
                        是否合并多卡参数，默认为True

        如果merge_tensor_parallel为真且模型的配置中的张量并行度大于1，则获取可训练的state_dict，并使用_merge_trainable_tensor_parallel方法合并张量并行训练的state_dict。如果merge_tensor_parallel为真且模型的张量并行度大于1，只有主进程会进行保存操作。

    -from_pretrained(model, prefix_path, postprocess_past_key_value, pad_attention_mask)
        --model
                        要加载Prefix权重参数的model对象
        --prefix_path
                        保存Prefix权重参数和 config 文件的路径
        --postprocess_past_key_value
                        功能同 PrefixModelForCausalLM 构造参数
        --pad_attention_mask
                        功能同 PrefixModelForCausalLM 构造参数

        该函数用于从预先训练的模型中加载Prefix权重参数，并将其设置到给定的模型中，以便在后续的任务中使用该模型进行预测或训练。

    -print_trainable_parameters()

        该函数会遍历整个权重参数列表，对于每个权重参数weight，统计所有进行梯度更新的参数，最后将信息打印出来。
```

更详细的使用可以参考[finetuning 脚本](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/run_finetune.py)版本, 以及对应的启动脚本编写方式（写在 [README.md](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/README.md)文件中)。
