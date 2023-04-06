# GLM

[General Language Model (GLM)](https://arxiv.org/abs/2103.10360) 是以自回归填空作为训练目标的通用语言模型，可用于各类理解和生成任务。

本示例提供了 GLM 模型的生成任务微调流程，适用于 GLM-Large-Chinese, GLM-10B-Chinese 模型。

## 摘要

现有预训练框架包括以 BERT 为代表的自编码模型，以 GPT 为代表的自回归模型和以 T5 为代表的编码-解码模型。但这些框架均不能完全支持自然语言理解、无条件生成和条件生成这三类主要任务。为了解决这一问题，我们提出了基于自回归填空任务的通用语言模型（GLM）。GLM 使用 2D 位置编码和任意顺序预测改进了填空预训练过程，在自然语言理解任务上超越了 BERT 和 T5。同时，GLM 的预训练过程基于多种任务，填空长度和数量各不相同。在自然语言理解、无条件生成和条件生成任务上，GLM 均超过了具有相同参数规模和训练数据量的 BERT、T5 和 GPT 模型。除此之外，GLM 还以 BERT Large 1.25 倍参数量的规模取得了当前最优的效果，证明了其在不同下游任务上良好的泛化能力。


## 快速开始

### DuReaderQG 问题生成任务

# Large 模型单卡训练脚本

```
python finetune_generation.py \
--model_name_or_path THUDM/glm-large-chinese \
--num_train_epochs 4 \
--learning_rate 3e-5 \
--warmup_ratio 0.06 \
--weight_decay 0.1 \
--label_smoothing 0.1 \
--save_steps 10000 \
--logging_steps 1 \
--eval_steps 4 \
--output_dir ./checkpoints/glm-large-chinese \
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

- `model_name_or_path`: 预训练模型内置名称或者模型所在目录，默认为`THUDM/glm-large-chinese`。
- `src_length`: 上下文的最大输入长度，默认为608.
- `tgt_length`: 生成文本的最大长度，默认为160.
- `min_tgt_length`: 生成文本的最小长度，默认为55.
- `length_penalty`: 生成解码时的长度惩罚因子，默认为0.7.
- `num_beams`: 搜索方向数量，默认为5。
- `label_smoothing`: 标签平滑因子，默认为0.1.
- `lr_decay_ratio`: 学习率衰减因子，默认为0.1.

# Large 模型多卡训练脚本（模型并行策略）

```
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" finetune_generation.py \
--model_name_or_path THUDM/glm-large-chinese \
--num_train_epochs 4 \
--learning_rate 3e-5 \
--warmup_ratio 0.06 \
--weight_decay 0.1 \
--label_smoothing 0.1 \
--save_steps 10000 \
--logging_steps 1 \
--eval_steps 4 \
--output_dir ./checkpoints/glm-large-chinese \
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
--do_eval \
--tensor_parallel_degree 8
```
