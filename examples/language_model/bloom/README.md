# Bloom

## 模型介绍

BLOOM是一种自回归大型语言模型(LLM)，在大量文本数据上训练从而生生成目标文本，同时它能够支持46种语言和13种编程语言的文本交互。BLOOM 主要基于文本生成任务训练而成，可以很好的完成文本续写任务，此外 BloomZ 系列模型加入了 Instruction Tuning，因为可以。


## 模型 Finetune

支持单个模型进行模型并行的生成式微调，示例脚本如下所示：

```shell
python -m paddle.distributed.launch --log_dir our_log --gpus "0,1,2,3" finetune_generation.py \
    --model_name_or_path bigscience/bloom-560m \
    --num_train_epochs 4 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.06 \
    --weight_decay 0.1 \
    --label_smoothing 0.1 \
    --save_steps 100 \
    --logging_steps 1 \
    --eval_steps 100 \
    --output_dir ./checkpoints/bloom-560m \
    --src_length 500 \
    --tgt_length 100 \
    --min_tgt_length 0 \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --max_grad_norm 1.0 \
    --scale_loss 32768 \
    --lr_scheduler_type linear \
    --do_train \
    --do_eval \
    --fp16 \
    --fp16_opt_level O2 \
    --recompute \
    --tensor_parallel_degree 4
```

支持大模型的模型并行微调，设置 `tensor_parallel_degree` 就是模型并行的并行度

```shell
python -m paddle.distributed.launch --log_dir our_log --gpus "0,1,2,3" finetune_generation.py \
    --model_name_or_path bigscience/bloom-560m \
    --num_train_epochs 4 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.06 \
    --weight_decay 0.1 \
    --label_smoothing 0.1 \
    --save_steps 20 \
    --logging_steps 1 \
    --eval_steps 20 \
    --output_dir ./checkpoints/bloom-560m \
    --src_length 500 \
    --tgt_length 100 \
    --min_tgt_length 0 \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --max_grad_norm 1.0 \
    --scale_loss 32768 \
    --lr_scheduler_type linear \
    --do_train \
    --do_eval \
    --fp16 \
    --fp16_opt_level O2 \
    --recompute \
    --tensor_parallel_degree 4
```
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

## 模型导出

当在指定数据集上 finetune 过后可导出模型部署，此时将会体验到paddle内置的加速优化，针对于不同任务提供了相同的导出脚本：


```shell
python export_generation_model.py --model_name_or_path ./save  --output_path inference/bloom
```
**NOTICE**: 动转静输入的动态图参数必须要是单分片参数checkpoint

## 模型部署
对动转静的后的模型可以进行静态图部署，具体执行脚本如下：

```shell
python infer_generation.py --model_dir inference/ --model_prefix bloom
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
