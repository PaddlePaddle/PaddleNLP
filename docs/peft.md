# PaddleNLP PEFT API

PaddleNLP PEFT API提供单卡/分布式LoRA和Prefix-Tuning，用户定义好模型，数据集, 以及相应的配置，就可以快速使用PEFT适配模型进行低参数模型微调。

# 预备知识
## LoRA
<div align="center">
<img src=https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/63d56558-247a-4a8d-a6ca-121c820f7534 width=60% height=60% />
</div>

大模型网络中有很多的线性层，里面需要进行密集的矩阵乘法计算，而这些通常具有全秩(full rank)，较难优化计算。LoRA论文的研究中表明, 将输入表达随机投影到较小的子空间不仅任然可以有效地学习还可以节约大量的计算显存需求。具体做法：对于预训练的权重矩阵, 通过引入两个低 rank 矩阵 $AB$(图中橙色的两个矩阵) 来近似权重的更新过程 $W_0+\Delta W=W_0+B A$ , 其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$, $r$ 远小于原权重矩阵的 rank 。训练期间, $W_0$ 参数冻结, 只对 $\mathrm{A}$ 和 $\mathrm{B}$ 两个矩阵进行梯度更新，前向传播公式如下:
$$
h=W_0 x+B A x
$$
由于训练参数的减少，训练过程会减少很多中间变量的存储，由此节约大量的训练显存消耗。
更多算法细节参考LoRA[论文](https://arxiv.org/abs/2106.09685)

## Prefix-tuning

<div align="center">
<img src=https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/8baf6943-4540-4c02-8540-35f977acc077 width=70% height=70% />
</div>


Prefix-tuning是一个针对NLG类型下游任务的轻量级微调方案，受提示学习（Prompt learning）的影响，加入的一部分 prefix embedding 作为连续型提示进行训练。prefix embedding是由专门的 prefix encoder 网络生成的数个张量，会以 past_key_value的方式被插入到语言模型每一层的 hidden_state之前。和 LoRA 类似，它也会冻结整个预训练模型的所有参数权重，只对prefix embedding进行梯度更新，因此训练参数量只有常规 SFT 的 0.1%。Prefix-tuning可以在全样本下获得与 SFT 比肩的训练效果，在小样本环境下甚至可以超越 SFT。更多算法细节参考
Prefix-tuning[论文](https://arxiv.org/abs/2101.00190)

# 快速开始
## LoRA

1. LoRAConfig是构建 LoRAModel 的必要参数，里面将对 LoRA 算法的一系列参数进行指定：
    - 第一步需要先指定 LoRA 模块替换的 target modules，后续 PEFT 库将自动将对应的模块替换成 LoRA 专用的 Linear 层。
```python
    from paddlenlp.peft import LoRAConfig, LoRAModel

    target_modules = [".*q_proj.*", ".*v_proj.*", ".*k_proj.*"]
    lora_rank = 8
    lora_config = LoRAConfig(
        target_modules=target_modules,
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        merge_weights=True,
        dtype=paddle.float32,
    )
```

2. 对原有模型适配 LoRA 配置
    - 对 model 适配 LoRA config，内部会进行target modules中相关 Layers 到对应LoRA Layers 的自动替换。
```python
    from paddlenlp.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained('facebook/llama-7b')
    model = LoRAModel(model, lora_config)
```

4. 设置只有 LoRA 参数需要训练(**重要**)
    - 这里我们将冻结原有llama模型的所有参数，只把 LoRA A和 B 矩阵设置需要训练。
```python
    model.mark_only_lora_as_trainable()
    # 这一步会将当前整个模型所有的参数和进行训练的参数量打印出来，可以进行校对设置是否正确
    model.print_trainable_parameters()
```

5. 模型的保存和载入

LoRAModel的保存和载入和普通的 model 没有太大区别，都是通过 save_pretrained/from_pretrained调用
```python
    # 保存
    model.save_pretrained('lora_path')
```
Paddle会将 LoRAModel 的矩阵 AB 权重保存为lora_mode_state.pdparams文件，LoRAConfig 配置保存为 lora_config.json 文件在 lora_path 目录下。
之后当需要载入模型权重进行推理时，则直接进行 from_pretrained即可。
```python
      from paddlenlp.transformers import AutoModelForCausalLM
    + from paddlenlp.peft import LoRAModel, LoRAConfig

    # 载入
    + config = LoRAConfig.from_pretrained('lora_path')
      model = AutoModelForCausalLM.from_pretrained('facebook/llama-7b')
    + model = LoRAModel.from_pretrained(model, 'lora_path')
      model.eval()
```
## LoRAConfig 参数介绍
```python
    --r
                        LoRA A/B 矩阵秩。

    --target_modules
                        指定哪些 module 需要适配 LoRA 算法，格式为module 的名字
                        或正则表达式的 List，比如, ['q', 'v'] 或者 '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'

    --trainable_modules
                        指定哪些 module 参数需要进行梯度更新，格式为module 的名字
                        或正则表达式的 List，比如, ['q', 'v'] 或者 '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'

    --lora_alpha
                        LoRA算法的 alpha 值，int 类型

    --lora_dropout
                        dropout的比例设置，float 类型

    --merge_weights
                        是否进行base model 权重和 LoRA 权重的合参操作，bool 类型

    --trainable_bias
                        为 LoRAModel 指定bias 参数

    --enable_lora_list
                        指定是否需要使用`MergedLoRALinear`，如果不指定则默认使用
                        `LoRALinear`

    --tensor_parallel_degree
                        多 GPU 并行的控制参数，默认设置为 1，代表不使用并行

    --dtype
                        LoRA矩阵参数类型设置

    --head_dim
                        多头注意力的头数，只有`LoRAMergedLinear`和
                        `ColumnParallelLoRAMergedLinear`使用
```

## Prefix-tuning
1. 设置Prefix-tuning参数
    - 对 PrefixConfig 进行配置, 我们需要指定prefix tokens个数，这里我们设置成 64 个。
    - prefix_projection用于指定是否对 prefix embeddings 进行进一步的 Linear 线性映射，打开会增加模型计算，但也会增强 prefix 的拟合能力, 这里我们用的默认设置 False。
    - num_attention_heads, num_hidden_layers等参数需要与 base model 保持一致，所以我们推荐先对 base model 进行初始化，再以下面的方式给 PrefixConfig 传参数。
```python
    from paddlenlp.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained('facebook/llama-7b')
    prefix_config = PrefixConfig(
        num_prefix_tokens=64,
        num_attention_heads=model.config.n_head,
        num_hidden_layers=model.config.n_layer,
        hidden_size=model.config.hidden_size,
        prefix_projection=False,
        prefix_projection_hidden_size=model.config.hidden_size,
    )
```

2. 对原有模型适配 Prefix-tuning 配置
    - 对 model 适配 Prefix config，内部会重载 base model 的 forward 函数，每个输入样本进行 forward 前，先以 past_key_value的方式为模型插入 prefix embeddings。
```python
    model = PrefixModelForCausalLM(model=model, prefix_config=prefix_config)
```

3. 设置只有 Prefix-tuning 参数需要训练(**重要**)
    - 这里我们将冻结原有llama模型的所有参数，只把 Prefix Embedding Layer (可能还有 projection layers，如果打开 prefix_projection选项) 设置需要训练。
```python
    model.mark_only_prefix_as_trainable()
    # 这一步会将当前整个模型所有的参数和进行训练的参数量打印出来，可以进行校对设置是否正确
    model.print_trainable_parameters()
```

5. 模型的保存和载入

和 LoRAModel 一致，通过 save_pretrained/from_pretrained调用
```python
    # 保存
    model.save_pretrained('prefix_path')
```
Paddle会将 PrefixModel 中用到的 prefix_encoder(里面包含 Embedding layer 和 Linear layers)网络模型权重，PrefixConfig 配置保存为 prefix_config.json 文件在 prefix_path 路径下。
之后当需要载入模型权重进行推理时，则直接进行 from_pretrained即可。
```python
      from paddlenlp.transformers import AutoModelForCausalLM
    + from paddlenlp.peft import PrefixModel, PrefixConfig

    # 载入
    + config = PrefixConfig.from_pretrained('prefix_path')
      model = AutoModelForCausalLM.from_pretrained('facebook/llama-7b')
    + model = PrefixModel.from_pretrained(model, 'prefix_path')
      model.eval()
```

## PrefixConfig 参数介绍
```python
    --prefix_dropout
                        prefix projection dropout比例设置，float 类型

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
                        是否对 prefix tokens 进行 projection 操作，bool 类型

    --prefix_projection_hidden_size
                        如果 prefix_projection 设置为 True，则在这里设置
                        projection 操作的 hidden size，int 类型

    --tensor_parallel_degree
                        多 GPU 并行的控制参数，默认设置为 1，代表不使用并行

    --dtype
                        prefix embeddings 参数类型设置

```

更详细的使用可以参考[Llama LoRA](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/llama/finetune_generation.py)版本, 以及对应的启动脚本编写方式（写在 [README.md](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/llama/README.md)文件中)。
