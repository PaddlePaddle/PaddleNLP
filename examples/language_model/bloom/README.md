# Bloom

## 模型介绍

BLOOM是一种自回归大型语言模型(LLM)，在大量文本数据上训练从而生生成目标文本，同时它能够支持46种语言和13种编程语言的文本交互。BLOOM 主要基于文本生成任务训练而成，可以很好的完成文本续写任务，此外 BloomZ 系列模型加入了 Instruction Tuning，因为可以。

## 文本生成

* 单卡小模型生成

```bash
python run_generation.py --model_name_or_path "bigscience/bloom-560m"
```

参数说明：
- `model_name_or_path`: 模型名称, 例如：`bigscience/bloom-560m`, `bigscience/bloom-3b`, `bigscience/bloom-7b1`等。

## 模型 Finetune

此模型也支持在生成式任务微调，示例脚本如下所示：

```shell
python -u -m paddle.distributed.launch --gpus "0" finetune_generation.py \
    --model_type bloom \
    --model_name_or_path "bigscience/bloom-560m" \
    --tokenizer_name_or_path "bigscience/bloom-560m" \
    --input_dir "old" \
    --output_dir "output_generate" \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --max_steps 50000 \
    --decay_steps 320 \
    --device gpu \
    --eval_freq  100 \
    --save_steps 100 \
    --logging_freq 10 \
    --warmup_rate 0.01 \
    --scale_loss 1024 \
    --global_batch_size 256\
    --micro_batch_size 4\
    --max_lr 5e-4 \
    --min_lr 1e-4 \
    --dp_degree 1 \
    --mp_degree 1 \
    --sharding_degree 1 \
    --use_pure_fp16 False\
    --use_recompute True\
    --sharding_stage 2
```

此外也提供了在 glue 任务上的微调代码，执行脚本如下所示：

```shell
python -u -m paddle.distributed.launch --gpus "0" finetune_glue.py \
    --model_type bloom \
    --model_name_or_path "bigscience/bloom-560m" \
    --tokenizer_name_or_path "bigscience/bloom-560m" \
    --input_dir "old" \
    --output_dir "output_glue" \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --max_steps 50000 \
    --decay_steps 320 \
    --device gpu \
    --eval_freq  100 \
    --save_steps 100 \
    --logging_freq 10 \
    --warmup_rate 0.01 \
    --scale_loss 1024 \
    --global_batch_size 256\
    --micro_batch_size 4\
    --max_lr 5e-4 \
    --min_lr 1e-4 \
    --dp_degree 1 \
    --mp_degree 1 \
    --sharding_degree 1 \
    --use_pure_fp16 False\
    --use_recompute True\
    --sharding_stage 2
```

## 模型导出

当在指定数据集上 finetune 过后可导出模型部署，此时将会体验到paddle内置的加速优化，针对于不同任务提供了相同的导出脚本：

* 导出生成模型

```shell
python export_generation_model.py --model_name_or_path "output_generation" --output_path "export_generation"
```

* 导出分类模型

```shell
python export_glue_model.py --model_name_or_path "output_glue" --output_path "export_glue"
```

## 模型评估

我们提供了对[WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)、[LAMBADA](https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl)两种数据集的评估脚本, 并将数据放置在data 目录下， 使用如下命令启动评估：

1. WikiText数据集评估
```bash

CUDA_VISIBLE_DEVICES="1" python run_eval.py \
    --model_type bloom \
    --model_name_or_path "bigscience/bloom-560m" \
    --tokenizer_name_or_path "bigscience/bloom-560m" \
    --input_dir "old" \
    --output_dir "output_glue" \
    --batch_size 8 \
    --eval_path ./data/wikitext-103/wiki.valid.tokens
```

2. LAMBADA数据集评估
```bash
# 覆盖default.yaml中的eval_path配置字段
python run_eval.py \
    --model_type bloom \
    --model_name_or_path "bigscience/bloom-560m" \
    --tokenizer_name_or_path "bigscience/bloom-560m" \
    --input_dir "old" \
    --output_dir "output_glue" \
    --batch_size 8 \
    --eval_path ./data/./lambada_test.jsonl \
    --cloze_eval
```
