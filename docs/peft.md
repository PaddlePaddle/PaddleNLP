# PaddleNLP Peft API

PaddleNLP Peft提供了LoRA和 Prefix-tuning 训练API，针对训练过程的通用训练配置做了封装，比如：

### LoRA tuning
- LoRA target modules, rank配置
- LoRA多 GPU 并行训练配置
### Prefix tuning
- Prefix token个数配置
- Prefix多 GPU 并行训练配置

用户定义好模型，数据集, 已经相应的配置，就可以使用Peft API高效快速为原模型适配 LoRA 和 prefix。

# 预备知识
## LoRA tuning
LoRA tuning(LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS)[论文](https://arxiv.org/pdf/2106.09685.pdf)
## Prefix tuning
Prefix tuning[论文](https://arxiv.org/pdf/2101.00190.pdf)

# LoRAConfig

1. LoRAConfig是构建 LoRAModel 的必要参数，里面将对 LoRA 算法的一系列参数进行指定：
- 第一步需要先指定 LoRA 模块替换的 target modules，后续 Peft 库将自动将对应的模块替换成 LoRA 专用的 Linear 层。
- 对 LoRAConfig 进行配置，主要是传入上面定制好的 target modules 和 LoRA 算法中的相关参数，如 rank，alpha。
- target_modules中指定了layer name中带有q_proj, v_proj, k_proj等字段的 layers, 这些是 Llama-based 的相关大模型共用的模型字段名，如果需要对其他类型模型进行转换则需要另外指定对应的字段名。
```python
    from paddlenlp.peft import LoRAConfig, LoRAModel

    target_modules = [
        ".*q_proj.*",
        ".*v_proj.*",
        ".*k_proj.*",
        ".*gate_proj.*",
        ".*up_proj.*",
        ".*o_proj.*",
        ".*down_proj.*",
    ]
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
    model = AutoModelForCausalLM.from_pretrained(
        'facebook/llama-7b'
    )
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

**通用分布式能力**
对于通用的分布式能力, PaddleNLP主要做了自动支持 LoRA Linear的RowParallel和 ColumnParallel，在上面单 gpu 的基础上，只需要更改 LoRAConfig 和 model 的少量配置即可实现多 GPU 并行训练.
### based model 声明部分我们加入多 GPU 的配置
```python
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    +   tensor_parallel_output=False,
    +   tensor_parallel_degree=training_args.tensor_parallel_degree,
    +   tensor_parallel_rank=training_args.tensor_parallel_rank,
    )
```
    - tensor_parallel_output设置为 False，表示 model 最终的输出会在多卡之间进行 reduce/gather 操作，这样每张卡最后的输出都是一致的且等于单 GPU 前向的输出
    - tensor_parallel_degree设置模型的并行度，既模型将在多少个 GPU 上进行并行，注意：并行度必须是模型多头注意力头数的因数。
    - tensor_parallel_rank设置本训练进程是第几个训练进程，如果 4gpu 并行，那么进程编号为0，1，2，3
### LoRAConfig部分添加多 GPU 的配置
```python
    lora_config = LoRAConfig(
        target_modules=target_modules,
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        merge_weights=True,
    +   tensor_parallel_degree=training_args.tensor_parallel_degree,
        dtype=paddle.float32,
    )
```

## Prefix finetuning
1. 设置Prefix参数
    - 对 PrefixConfig 进行配置, 我们需要指定prefix tokens个数，这里我们设置成 64 个。
    - prefix_projection用于指定是否对 prefix embeddings 进行进一步的 Linear 线性映射，打开会增加模型计算，但也会增强 prefix 的拟合能力, 这里我们用的默认设置 False。
    - num_attention_heads, num_hidden_layers等参数需要与 base model 保持一致，所以我们推荐先对 base model 进行初始化，再以下面的方式给 PrefixConfig 传参数。
```python
    from paddlenlp.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        'facebook/llama-7b'
    )
    prefix_config = PrefixConfig(
        num_prefix_tokens=64,
        num_attention_heads=model.config.n_head,
        num_hidden_layers=model.config.n_layer,
        hidden_size=model.config.hidden_size,
        prefix_projection=False,
        prefix_projection_hidden_size=model.config.hidden_size,
    )
```

2. 对原有模型适配 Prefix 配置
    - 对 model 适配 Prefix config，内部会重载 base model 的 forward 函数，每个输入样本进行 forward 前，先以 past_key_value的方式为模型插入 prefix embeddings。
```python
    model = PrefixModelForCausalLM(
        model=model,
        prefix_config=prefix_config,
    )
```

3. 设置只有 Prefix 参数需要训练(**重要**)
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

## Prefix进阶分布式能力使用介绍

**通用分布式能力**
对于通用的分布式能力, PaddleNLP可配置 Prefix 模块内的prefix_embeddings 和 prefix_proj_0/1 layers并行化，为了实现这一点我们需要在上面单卡操作的基础上额外对 PrefixConfig 和 base model 进行配置.
### based model 部分我们加入多 GPU 的配置
```python
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    +   tensor_parallel_output=False,
    +   tensor_parallel_degree=training_args.tensor_parallel_degree,
    +   tensor_parallel_rank=training_args.tensor_parallel_rank,
    )
```
### PrefixConfig 部分加入多 GPU 的配置
```python
    prefix_config = PrefixConfig(
        num_prefix_tokens=64,
        num_attention_heads=model.config.n_head,
        num_hidden_layers=model.config.n_layer,
        hidden_size=model.config.hidden_size,
        prefix_projection=False,
        prefix_projection_hidden_size=model.config.hidden_size,
    +   tensor_parallel_degree=training_args.tensor_parallel_degree,
    )

更详细的使用可以参考[Llama Lora tuning](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/llama/finetune_generation.py)版本, 以及对应的启动脚本编写方式（写在 [README.md](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/llama/README.md)文件中)。
