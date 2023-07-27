# Bloom

## 模型介绍

BLOOM是一种自回归大型语言模型(LLM)，在大量文本数据上训练从而生生成目标文本，同时它能够支持46种语言和13种编程语言的文本交互。BLOOM 主要基于文本生成任务训练而成，可以很好的完成文本续写任务，此外 BloomZ 系列模型加入了 Instruction Tuning，因为可以。


## 模型 Finetune

支持单个模型进行模型并行的生成式微调，示例脚本如下所示：

```shell
python -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py \
    --model_name_or_path bigscience/bloom-560m \
    --task_name_or_path "dureader_qg" \
    --output_dir ./checkpoints/bloom-560m \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 3e-5 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 1024 \
    --tgt_length 1024 \
    --fp16 \
    --fp16_opt_level O2 \
    --do_train \
    --do_eval \
    --disable_tqdm True \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --eval_with_do_generation False \
    --tensor_parallel_degree 4 \
    --recompute \
    --save_total_limit 1 \
    --scale_loss 32768 \
    --overwrite_output_dir
```

支持大模型的模型并行微调，设置 `tensor_parallel_degree` 就是模型并行的并行度

支持单个模型进行单卡LoRA微调，示例脚本如下所示：

```shell
python finetune_generation.py \
    --model_name_or_path bigscience/bloom-560m \
    --task_name_or_path "dureader_qg" \
    --output_dir ./checkpoints/bloom-560m \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 2 \
    --learning_rate 3e-4 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 1024 \
    --tgt_length 1024 \
    --fp16 \
    --fp16_opt_level O2 \
    --do_train \
    --do_eval \
    --disable_tqdm True \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --eval_with_do_generation False \
    --recompute \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --lora True \
    --lora_rank 8
```

支持单个模型进行单卡Prefix微调，示例脚本如下所示：

```shell
python finetune_generation.py \
    --model_name_or_path bigscience/bloom-560m \
    --task_name_or_path "dureader_qg" \
    --output_dir ./checkpoints/bloom-560m \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 2 \
    --learning_rate 3e-4 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 1024 \
    --tgt_length 1024 \
    --fp16 \
    --fp16_opt_level O2 \
    --do_train \
    --do_eval \
    --disable_tqdm True \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --eval_with_do_generation False \
    --recompute \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --prefix_tuning True \
    --num_prefix_tokens 64
```

其中参数释义如下：

- `model_name_or_path`: 预训练模型内置名称或者模型所在目录，默认为`facebook/llama-7b`。
- `num_train_epochs`: 要执行的训练 epoch 总数（如果不是整数，将在停止训练之前执行最后一个 epoch
的小数部分百分比）。
- `max_steps`: 模型训练步数。
- `label_smoothing`: 标签平滑参数。
- `learning_rate`: 参数更新的学习率。
- `warmup_steps`: 学习率热启的步数。
- `eval_steps`: 模型评估的间隔步数。
- `logging_steps`: 训练日志打印的间隔步数。
- `save_steps`: 模型参数保存的间隔步数。
- `save_total_limit`: 模型 checkpoint 保存的份数。
- `output_dir`: 模型参数保存目录。
- `src_length`: 上下文的最大输入长度，默认为128.
- `tgt_length`: 生成文本的最大长度，默认为160.
- `gradient_accumulation_steps`: 模型参数梯度累积的步数，可用于扩大 batch size。实际的 batch_size = per_device_train_batch_size * gradient_accumulation_steps。
- `fp16`: 使用 float16 精度进行模型训练和推理。
- `fp16_opt_level`: float16 精度训练模式，`O2`表示纯 float16 训练。
- `recompute`: 使用重计算策略，开启后可节省训练显存。
- `do_train`: 是否训练模型。
- `do_eval`: 是否评估模型。
- `tensor_parallel_degree`: 模型并行数量。
- `do_generation`: 在评估的时候是否调用model.generate,默认为False。
- `lora`: 是否使用LoRA技术。
- `lora_path`: 初始化lora参数和配置文件路径。
- `lora_rank`: lora 算法中rank（秩）的值。
- `merge_weights`: 是否合并原始模型和Lora模型的权重。
- `prefix_tuning`: 是否使用Prefix技术。
- `num_prefix_tokens`: prefix tuning算法中前缀token数量。

## 模型动态图预测


### 多分片模型预测
通过设置多卡预测即可对保留下来的checkpoints进行预测

```shell
python -m paddle.distributed.launch --gpus "0,1,2,3" predict_generation.py \
  --model_name_or_path checkpoints/bloom-560m/checkpoint-20 \
  --save_onepiece_model_path ./save
```

同时可以将多分片的模型参数merge成一个大分片参数，设置 `save_onepiece_model_path` 即可以进行参数merge

### 单分片模型预测
对merge后的单分片模型也可以进行直接预测，脚本如下
```shell
 python predict_generation.py --model_name_or_path ./save
```

### LoRA微调模型预测
对merge后的单分片模型也可以进行直接预测，脚本如下
```shell
 python predict_generation.py
    --model_name_or_path bigscience/bloom-560m \
    --lora_path ./checkpoints/bloom-560m
```

### Prefix微调模型预测
对merge后的单分片模型也可以进行直接预测，脚本如下
```shell
 python predict_generation.py
    --model_name_or_path bigscience/bloom-560m \
    --prefix_path ./checkpoints/bloom-560m
```

## 模型导出

当在指定数据集上 finetune 过后可导出模型部署，此时将会体验到paddle内置的加速优化，针对于不同任务提供了相同的导出脚本：


```shell
python export_generation_model.py --model_name_or_path ./save  --output_path inference/bloom
```
**NOTICE**: 动转静输入的动态图参数必须要是单分片参数checkpoint

当在指定数据集上进行 LoRA finetune 后的导出脚本：


```shell
python export_generation_model.py
    --model_name_or_path bigscience/bloom-560m
    --output_path inference/bloom
    --lora_path ./checkpoints/bloom-560m
```

## 模型部署
对动转静的后的模型可以进行静态图部署，具体执行脚本如下：

```shell
python infer_generation.py --model_dir inference/ --model_prefix bloom
```

## 模型评估

我们提供了对[WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)、[LAMBADA](https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl)两种数据集的评估脚本, 并将数据放置在data 目录下， 使用如下命令启动评估：

> 模型评估脚本相关脚本放置在 [Makefile](./Makefile) 中，可通过执行`make run_eval*`等命令执行对应评估命令。

1. WikiText数据集评估

* 单卡评估

```bash
make run_eval
# or
python run_eval.py \
    --model_type bloom \
    --model_name_or_path "bigscience/bloom-560m" \
    --batch_size 8 \
    --eval_path ./data/wikitext-103/wiki.valid.tokens
```

* 多卡评估

```bash
make run_eval_tps
# or
python -m paddle.distributed.launch --gpus "3,4,5,6" python run_eval.py \
    --model_type bloom \
    --model_name_or_path "bigscience/bloom-560m" \
    --batch_size 8 \
    --tensor_parallel_degree 4 \
    --eval_path ./data/wikitext-103/wiki.valid.tokens
```

2. LAMBADA数据集评估

```bash
python run_eval.py \
    --model_type bloom \
    --model_name_or_path "bigscience/bloom-560m" \
    --batch_size 8 \
    --eval_path ./data/lambada_test.jsonl \
    --cloze_eval
```

3. 176B 模型评估

当前 Bloom（176B）模型权重基于`auto_dist{rank}.pdparams`的命令方式加载，故在此提供以下脚本执行动态图评估脚本：

> 评估不同数据集只需要调整参数`eval_path`。

```bash
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" python run_eval.py \
    --model_type bloom \
    --model_name_or_path "/path/to/auto_dist/pdparams" \
    --batch_size 8 \
    --dtype "bfloat16" \
    --tensor_parallel_degree 8 \
    --eval_path ./data/lambada_test.jsonl \
    --cloze_eval \
    --load_autodist
```
