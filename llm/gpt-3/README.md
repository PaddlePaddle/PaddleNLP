# GPT

## 1. 模型介绍

GPT-3是一种预训练语言模型，它能够模拟人类语言思维和表达。GPT-3拥有巨大的参数，包含了1750亿个参数，这使得它具有强大的语言理解和生成能力。它可以完成的任务包括文本生成、文本摘要、回答问题、翻译、阅读理解等。GPT-3的预训练过程使用了大量的语料库，包括互联网上的大量文本。它通过分析这些文本，学习如何生成和理解人类语言。GPT-3在自然语言处理领域具有很高的影响力，它可以模拟人类对话和生成文本，这使得它在许多应用领域都有广泛的应用，比如智能客服、自然语言处理、游戏设计等。

## 2. 预训练

预训练数据制作参考[此处](../../model_zoo/ernie-1.0/preprocess/docs/OpenWebText2.md)

为了方便用户运行测试本模型，本项目提供了处理好的100k条doc的训练样本：
```shell
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
```

将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：

```
mkdir data
mv gpt_en_dataset_300m_ids.npy ./data
mv gpt_en_dataset_300m_idx.npz ./data
```

注意：
1. 需要paddle develop版本训练，需要安装`pip install tool_helpers visualdl==2.5.3`等相关缺失whl包
2. `use_flash_attention` 需要在A100机器开启，否则loss可能不正常（很快变成0.00x,非常小不正常）。建议使用cuda11.8环境。

使用下面脚本,即可在gpt2-medium-en的基础上,继续训练.
```shell
task_name="gpt3_hybrid"
export PYTHONPATH="../../PaddleNLP/"
export FLAGS_cudnn_deterministic=True
log_dir="log"
rm -rf $log_dir

python -u  -m paddle.distributed.launch \
    --gpus "0" \
    --log_dir ${log_dir} \
    run_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path gpt2-medium-en \
    --tokenizer_name_or_path gpt2-medium-en \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 1 \
    --fuse_attention_qkv 1 \
    --use_flash_attention 0 \
    --fp16  \
    --fp16_opt_level "O2"  \
    --scale_loss 1024 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --max_steps 10000 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1\
    --dataloader_num_workers 1 \
    --sharding "stage2" \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --recompute 1 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --do_eval \
    --device "gpu"
```

其中参数释义如下：

- `model_name_or_path`: 预训练模型内置名称或者模型所在目录，默认为`gpt2-medium-en`。
- `num_train_epochs`: 要执行的训练 epoch 总数（如果不是整数，将在停止训练之前执行最后一个 epoch
的小数部分百分比）。
- `max_steps`: 模型训练步数。
- `learning_rate`: 参数更新的学习率。
- `warmup_steps`: 学习率热启的步数。
- `eval_steps`: 模型评估的间隔步数。
- `logging_steps`: 训练日志打印的间隔步数。
- `save_steps`: 模型参数保存的间隔步数。
- `output_dir`: 模型参数保存目录。
- `src_length`: 上下文的最大输入长度，默认为128.
- `tgt_length`: 生成文本的最大长度，默认为160.
- `gradient_accumulation_steps`: 模型参数梯度累积的步数，可用于扩大 batch size。实际的 batch_size = per_device_train_batch_size * gradient_accumulation_steps。
- `fuse_attention_qkv`：在MultiHeadAttention中使用qkv线性层融合
- `use_flash_attention`：使用flash attention技术，注意此处需要在A100机器开启
- `fp16`: 使用 float16 精度进行模型训练和推理。
- `fp16_opt_level`: float16 精度训练模式，`O2`表示纯 float16 训练。
- `recompute`: 使用重计算策略，开启后可节省训练显存。
- `do_train`: 是否训练模型。
- `do_eval`: 是否评估模型。
- `tensor_parallel_degree`: 模型并行数量。
- `lora`: 是否使用LoRA技术。
- `task_name`: 内置数据集任务名。

<a name="1"></a>


## 3. 微调
### SFT

```shell
task_name="gpt3_hybrid"
export PYTHONPATH="../../PaddleNLP/"
export FLAGS_cudnn_deterministic=True
log_dir="log"
rm -rf $log_dir

python -u  -m paddle.distributed.launch \
    --gpus "0" \
    --log_dir ${log_dir} \
    finetune_generation.py \
    --model_type "gpt" \
    --model_name_or_path gpt2-medium-en \
    --output_dir "output/$task_name" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 1 \
    --fp16  \
    --fp16_opt_level "O2"  \
    --scale_loss 1024 \
    --learning_rate 0.00001 \
    --max_steps 10000 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1\
    --dataloader_num_workers 1 \
    --sharding "stage2" \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --recompute 1 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --do_eval \
    --device "gpu"
```

### LoRA

```shell
export PYTHONPATH="../../PaddleNLP/"
export FLAGS_cudnn_deterministic=True
log_dir="log"
rm -rf $log_dir

python finetune_generation.py \
    --model_type "gpt" \
    --model_name_or_path gpt2-medium-en \
    --output_dir "output/$task_name" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 1 \
    --fp16  \
    --fp16_opt_level "O2"  \
    --scale_loss 1024 \
    --learning_rate 3e-4 \
    --max_steps 10000 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1\
    --dataloader_num_workers 1 \
    --sharding "stage2" \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --recompute 1 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --do_eval \
    --device "gpu" \
    --lora
```


## 3. 动态图推理

```shell
python predict_generation.py

```
