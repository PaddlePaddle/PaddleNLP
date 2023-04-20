# GLM

[General Language Model (GLM)](https://arxiv.org/abs/2103.10360) 是以自回归填空作为训练目标的通用语言模型，可用于各类理解和生成任务。

本示例提供了 GLM 模型的生成任务微调流程，适用于 GLM-Large-Chinese, GLM-10B-Chinese 模型。

## 摘要

现有预训练框架包括以 BERT 为代表的自编码模型，以 GPT 为代表的自回归模型和以 T5 为代表的编码-解码模型。但这些框架均不能完全支持自然语言理解、无条件生成和条件生成这三类主要任务。为了解决这一问题，我们提出了基于自回归填空任务的通用语言模型（GLM）。GLM 使用 2D 位置编码和任意顺序预测改进了填空预训练过程，在自然语言理解任务上超越了 BERT 和 T5。同时，GLM 的预训练过程基于多种任务，填空长度和数量各不相同。在自然语言理解、无条件生成和条件生成任务上，GLM 均超过了具有相同参数规模和训练数据量的 BERT、T5 和 GPT 模型。除此之外，GLM 还以 BERT Large 1.25 倍参数量的规模取得了当前最优的效果，证明了其在不同下游任务上良好的泛化能力。

## 环境依赖
目前版本支持的功能较多，建议使用paddlepaddle develop版本以获得较好体验。下面给出了cuda 11.2的paddle安装方法。更多其他版本，请参考[官网首页](https://www.paddlepaddle.org.cn/)下载。
```
python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

## DuReaderQG 问题生成任务

### Large 模型单卡训练脚本

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

### Large 模型多卡训练脚本（模型并行策略）

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

## 模型导出
使用`export_generation_model.py`脚本，传入我们需要的模型地址，和输出地址即可。如果需要导出`float16`参数的模型，请指定`paddle_dtype`参数为`float16`。
```
python export_generation_model.py \
   --model_name_or_path ./checkpoints/glm-large-chinese \
   --output_path ./checkpoints/infer/glm
```

## 模型推理 (c++推理)
需要依赖` pip install fastdeploy-gpu-python==0.0.0 -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html` (cpu请安装`fastdeploy-python`)
```
python infer_generation.py \
    --model_path  ./checkpoints/infer \
    --model_prefix glm
```

## 模型预测（python）
可以将模型python前向与推理结果比较：
```
python predict_generation.py \
    --model_name_or_path  ./checkpoints/glm-large-chinese
```
当ckpt为使用的`tensor parallel`存储为多分片格式时，也可使用此脚本预测，或者合并为一个单分片权重
例如下面4分片的例子（此模型为`glm-10b-chinese`）
```
(base) root@localhost glm $ ll ./checkpoints/glm-large-chinese/checkpoint-100/
total 130G
drwxr-xr-x 2 root root 4.0K Apr  7 18:21 ./
drwxr-xr-x 4 root root 4.0K Apr  7 20:02 ../
-rw-r--r-- 1 root root  201 Apr  7 18:20 added_tokens.json
-rw-r--r-- 1 root root 998K Apr  7 18:20 cog-pretrain.model
-rw-r--r-- 1 root root  892 Apr  7 18:20 config.json
-rw-r--r-- 1 root root 4.7G Apr  7 18:20 model_state.tp00.pdparams
-rw-r--r-- 1 root root 4.7G Apr  7 18:20 model_state.tp01.pdparams
-rw-r--r-- 1 root root 4.7G Apr  7 18:20 model_state.tp02.pdparams
-rw-r--r-- 1 root root 4.7G Apr  7 18:20 model_state.tp03.pdparams
```
设置 merge_tensor_parallel_path，可以将merge好的参数存储到对应位置。不过不设置此参数，将只跑前向预测。
```
python -m paddle.distributed.launch --gpus 0,1,2,3 predict_generation.py \
    --model_name_or_path  ./checkpoints/glm-large-chinese/checkpoint-100/ \
    --merge_tensor_parallel_path  ./checkpoints/glm-merged
```
