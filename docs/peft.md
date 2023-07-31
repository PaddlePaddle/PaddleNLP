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

# Peft基本使用方法介绍

下面是用户使用 Peft API对 llama 模型分别进行LoRA和 Prefix finetune任务的简单示例，这里以数据集`squad`为例。
更详细的使用可以参考[Llama Lora tuning](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/llama/finetune_generation.py)版本, 以及对应的启动脚本编写方式（写在 [README.md](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/llama/README.md)文件中)。

1. 导入需要用到的头文件, 定义数据集 Argument 类。
    - 主要是模型、Tokenizer
    - 还有Peft组件
        - 其中`LoRAModel`和`PrefixModelForCausalLM`分别是 LoRA 和 Prefix 的 Model 类, 配合各自对应的配置类`LoRAConfig`,`PrefixConfig`，便可进行普通模型到 LoRAModel/PrefixModel 的转换。
        - `load_dataset` 用于 squad 数据集对象的构建。
        - `LlamaTrainer` 是Llama-based模型的 Trainer 工具类
```python
from functools import partial
import paddle
from paddlenlp.datasets import load_dataset
from data import convert_example
from utils import LlamaTrainer
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.trainer import (
    PdArgumentParser,
    TrainingArguments,
)
@dataclass
class DataArgument:

    data_name: str = field(default=None, metadata={"help": "The name of data."})
    task_name: str = field(default=None, metadata={"help": "The name of task."})
    dataset_path: str = field(default=None, metadata={"help": "The file name of train dataset."})
    src_length: int = field(default=512, metadata={"help": "The max length of source text."})
    tgt_length: int = field(default=256, metadata={"help": "The max length of target text."})
```

## LoRA finetuning
1. 初始化 base model, dataset
    - 这里因为我们是基于 Llama-7b 继续进行 LoRA 微调，所以在这里载入 Llama-7b 的相关 model 和 tokenizer
```python
    # 生成 data_args, training_args
    parser = PdArgumentParser((DataArgument, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    # 定义 base model 和 tokenizer
    model = AutoModelForCausalLM.from_pretrained(training_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name_or_path)
    # 生成dataset
    train_ds = load_dataset(data_args.task_name, splits=["train_v1"])
    trans_func = partial(convert_example, tokenizer=tokenizer, data_args=data_args)
    train_ds = train_ds.map(partial(trans_func))
```
2. 设置LoRA参数
    - 第一步需要先指定 LoRA 模块替换的 target modules，后续 Peft 库将自动将对应的模块替换成 LoRA 专用的 Linear 层。
    - 对 LoRAConfig 进行配置，主要是传入上面定制好的 target modules 和 LoRA 算法中的相关参数，如 rank，alpha。
    - target_modules中指定了layer name中带有q_proj, v_proj, k_proj等字段的 layers, 这些是 Llama-based 的相关大模型共用的模型字段名，如果需要对其他类型模型进行转换则需要另外指定对应的字段名。
```python
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

3. 对原有模型适配 LoRA 配置
    - 对 model 适配 LoRA config，内部会进行target modules中相关 Layers 到对应LoRA Layers 的自动替换。
```python
    # 如果不需要从之前 LoRA checkpoint载入继续训练
    model = LoRAModel(model, lora_config)
    # 从之前 LoRA checkpoint载入继续训练则使用 from_pretrained函数，lora_path为 LoRA checkpoint 的本地保存地址
    model = LoRAModel.from_pretrained(model=model, lora_path=lora_path)
```

4. 设置只有 LoRA 参数需要训练(**重要**)
    - 这里我们将冻结原有llama模型的所有参数，只把 LoRA A和 B 矩阵设置需要训练。
```python
    model.mark_only_lora_as_trainable()
    # 这一步会将当前整个模型所有的参数和进行训练的参数量打印出来，可以进行校对设置是否正确
    model.print_trainable_parameters()
```

5. 初始化 LlamaTrainer，进行训练
```python
    trainer = LlamaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
```

6. 启动方式(单卡)
```shell
python lora_finetune_llama.py \
    --output_dir ./checkpoints/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path facebook/llama-7b\
    --task_name squad \
    --max_steps 100 \
    --num_train_epochs 500 \
    --learning_rate 3e-4 \
    --warmup_steps 30 \
    --save_strategy steps \
    --save_steps 150 \
    --src_length 1024 \
    --tgt_length 1024 \
    --do_train \
    --overwrite_output_dir

```

## LoRA进阶分布式能力使用介绍

**通用分布式能力**
对于通用的分布式能力, PaddleNLP主要做了自动支持 LoRA Linear的RowParallel和 ColumnParallel，在上面单 gpu 的基础上，只需要更改 LoRAConfig 和 model 的少量配置即可实现多 GPU 并行训练.
### based model 声明部分我们加入多 GPU 的配置
```python
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        tensor_parallel_output=False,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
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
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        dtype=paddle.float32,
    )
```

用户使用以下启动脚本即可切换为LoRA多卡并行训练。
```shell
python -m paddle.distruted.launch --devices "0,1" lora_finetuning_llama.py \
    --output_dir ./checkpoints/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path facebook/llama-7b\
    --task_name squad \
    --max_steps 100 \
    --num_train_epochs 500 \
    --learning_rate 3e-4 \
    --warmup_steps 30 \
    --save_strategy steps \
    --save_steps 150 \
    --src_length 1024 \
    --tgt_length 1024 \
    --do_train \
    --tensor_parallel_degree 2  \
    --overwrite_output_dir

```

## Prefix finetuning
1. 同样需要先初始化 base model, dataset, 步骤同 LoRA tuning
```python
    # 生成 data_args, training_args
    parser = PdArgumentParser((DataArgument, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    # 定义 base model 和 tokenizer
    model = AutoModelForCausalLM.from_pretrained(training_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name_or_path)
    # 生成dataset
    train_ds = load_dataset(data_args.task_name, splits=["train_v1"])
    trans_func = partial(convert_example, tokenizer=tokenizer, data_args=data_args)
    train_ds = train_ds.map(partial(trans_func))
```
2. 设置Prefix参数
    - 对 LoRAConfig 进行配置, 这里我们可以指定prefix tokens个数，这里我们设置成 64 个。
    - prefix_projection用于指定是否对 prefix embeddings 进行进一步的 Linear 线性映射，打开会增加模型计算，但也会增强 prefix 的拟合能力, 这里我们用的默认设置 False。
```python
    prefix_config = PrefixConfig(
        num_prefix_tokens=64,
        num_attention_heads=model.config.n_head,
        num_hidden_layers=model.config.n_layer,
        hidden_size=model.config.hidden_size,
        prefix_projection=False,
        prefix_projection_hidden_size=model.config.hidden_size,
    )
```

3. 对原有模型适配 Prefix 配置
    - 对 model 适配 Prefix config，内部会重载 base model 的 forward 函数，每个输入样本进行 forward 前，先以 past_key_value的方式为模型插入 prefix embeddings。
```python
    # 如果不需要从之前 Prefix checkpoint载入继续训练
    model = PrefixModelForCausalLM(
        model=model,
        prefix_config=prefix_config,
    )
    # 从之前 Prefix checkpoint载入继续训练则使用 from_pretrained函数，prefix_path为 Prefix checkpoint 的本地保存地址
    model = PrefixModelForCausalLM.from_pretrained(model=model, prefix_path=prefix_path)
```

4. 设置只有 Prefix 参数需要训练(**重要**)
    - 这里我们将冻结原有llama模型的所有参数，只把 Prefix Embedding Layer (可能还有 projection layers，如果打开 prefix_projection选项) 设置需要训练。
```python
    model.mark_only_prefix_as_trainable()
    # 这一步会将当前整个模型所有的参数和进行训练的参数量打印出来，可以进行校对设置是否正确
    model.print_trainable_parameters()
```

5. 初始化 LlamaTrainer，进行训练
```python
    trainer = LlamaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
```

6. 启动方式(单卡)
```shell
python prefix_finetuning_llama.py \
    --output_dir ./checkpoints/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path facebook/llama-7b \
    --task_name squad \
    --num_train_epochs 2 \
    --learning_rate 3e-5 \
    --warmup_steps 30 \
    --src_length 1024 \
    --tgt_length 1024 \
    --do_train \
    --overwrite_output_dir

```

## Prefix进阶分布式能力使用介绍

**通用分布式能力**
对于通用的分布式能力, PaddleNLP可配置 Prefix 模块内的prefix_embeddings 和 prefix_proj_0/1 layers并行化，为了实现这一点我们需要在上面单卡操作的基础上额外对 PrefixConfig 和 base model 进行配置.
### based model 部分我们加入多 GPU 的配置
```python
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        tensor_parallel_output=False,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
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
        tensor_parallel_degree=training_args.tensor_parallel_degree,
    )

接着使用以下启动脚本即可切换为Prefix多卡并行训练。
```shell
python -m paddle.distruted.launch --devices "0,1" prefix_finetuning_llama.py \
    --output_dir ./checkpoints/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path facebook/llama-7b\
    --task_name squad \
    --max_steps 100 \
    --num_train_epochs 500 \
    --learning_rate 3e-4 \
    --warmup_steps 30 \
    --save_strategy steps \
    --save_steps 150 \
    --src_length 1024 \
    --tgt_length 1024 \
    --do_train \
    --tensor_parallel_degree 2  \
    --overwrite_output_dir
```
