# OPT

[OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068) 是以自回归填空作为训练目标的通用语言模型，可用于各类理解和生成任务。

本示例提供了 OPT 模型的生成任务微调流程。


## 快速开始

### SQuAD 知识问答微调

本示例中以SQuAD 数据集为示例，通过融合context、question来生成answer，从而让模型能够文本中生成指定的答案，执行脚本如下所示：

```
python -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py \
    --model_name_or_path facebook/opt-125m \
    --num_train_epochs 4 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.06 \
    --weight_decay 0.1 \
    --label_smoothing 0.1 \
    --save_steps 10000 \
    --logging_steps 1 \
    --eval_steps 4 \
    --output_dir ./checkpoints/opt \
    --src_length 608 \
    --tgt_length 160 \
    --min_tgt_length 55 \
    --length_penalty 0.7 \
    --no_repeat_ngram_size 3 \
    --num_beams 5 \
    --select_topk True \
    --per_device_eval_batch_size 2 \
    --per_device_train_batch_size 2 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type linear \
    --fp16 \
    --recompute \
    --do_train \
    --do_eval
```

其中参数释义如下：

- `model_name_or_path`: 预训练模型内置名称或者模型所在目录，默认为`facebook/opt-125m`。
- `src_length`: 上下文的最大输入长度，默认为608.
- `tgt_length`: 生成文本的最大长度，默认为160.
- `min_tgt_length`: 生成文本的最小长度，默认为55.
- `length_penalty`: 生成解码时的长度惩罚因子，默认为0.7.
- `num_beams`: 搜索方向数量，默认为5。
- `label_smoothing`: 标签平滑因子，默认为0.1.
- `lr_decay_ratio`: 学习率衰减因子，默认为0.1.
