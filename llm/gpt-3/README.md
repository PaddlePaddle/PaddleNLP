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
2. `use_flash_attention` 需要在A100机器开启。建议使用cuda11.8环境。

使用下面脚本,即可在gpt2-medium-en的基础上,继续训练.
```shell
task_name="gpt3_hybrid"
export PYTHONPATH="../../PaddleNLP/"
export FLAGS_cudnn_deterministic=True
log_dir="log"
rm -rf $log_dir

python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir ${log_dir} \
    run_pretrain.py \
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
    --sequence_parallel 0 \
    --fuse_attention_qkv 0 \
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
    --continue_training \
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
- `tokenizer_name_or_path`: tokenizer名称或者tokenizer所在目录，默认为`gpt2-medium-en`。
- `input_dir`: 预训练数据所在目录。
- `output_dir`: 模型参数及日志保存目录。
- `split`: 预训练数据切分比例，默认为949,50,1。
- `max_seq_length`: 预训练最大序列长度，默认为1024。
- `per_device_train_batch_size`: 单卡训练batch_size大小，默认为1。
- `per_device_eval_batch_size`: 单卡评估batch_size大小，默认为1。
- `tensor_parallel_degree`: 模型并行数量。
- `pipeline_parallel_degree`: 流水线并行数量。
- `sequence_parallel`: 序列并行数量。需要当`tensor_parallel_degree>1`时，使用序列并行。注意：当模型规模较小、batch_size较小、sequence_length较小时，不建议使用序列并行。
- `fuse_attention_qkv`：在MultiHeadAttention中使用qkv线性层融合
- `use_flash_attention`：使用flash attention技术，注意此处需要在A100机器开启, 建议使用cuda11.8环境。
- `fp16`: 使用 float16 精度进行模型训练和推理。
- `fp16_opt_level`: float16 精度训练模式，`O2`表示纯 float16 训练。
- `scale_loss`: float16 精度训练时，损失值的缩放比例。微调时建议使用1024，预训练时建议调大。
- `learning_rate`: 参数更新的学习率。
- `min_learning_rate`: 最小学习率。
- `max_steps`: 模型训练步数。
- `save_steps`: 模型参数保存的间隔步数。
- `weight_decay`: 权重衰减系数。
- `warmup_ratio`: warmup比例。
- `max_grad_norm`: 梯度裁剪系数。
- `logging_steps`: 训练日志打印的间隔步数。
- `continue_training`: 是否继续训练模型。
- `dataloader_num_workers`: dataloader进程数。
- `sharding`: sharding切分策略，包含stage1、stage2、stage3。
- `eval_steps`: 模型评估的间隔步数。
- `recompute`: 使用重计算策略，开启后可节省训练显存。
- `gradient_accumulation_steps`: 模型参数梯度累积的步数，可用于扩大 batch size。实际的 batch_size = per_device_train_batch_size * gradient_accumulation_steps。
- `do_train`: 是否训练模型。
- `do_eval`: 是否评估模型。
- `lora`: 是否使用LoRA技术。

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
    --model_name_or_path gpt2-medium-en \
    --output_dir "output/$task_name" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 1 \
    --sequence_parallel 0 \
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
    --model_name_or_path gpt2-medium-en \
    --output_dir "output/$task_name" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 1 \
    --sequence_parallel 0 \
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
