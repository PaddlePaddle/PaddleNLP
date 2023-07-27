# GPT inplementation

**目录**

- [1. 预训练](#0)
- [2. 微调](#1)
- [3. 模型预测](#2)
- [4. 动转静](#3)
- [5. 模型推理](#4)

<a name="0"></a>

## 预训练

预训练数据制作参考[此处](../model_zoo/ernie-1.0/preprocess/docs/OpenWebText2.md)

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

使用下面脚本,即可在llama-7b的基础上,继续训练.
```shell
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
注意：
1. 需要paddle develop版本训练，需要安装`pip install tool_helpers visualdl==2.5.3`等相关缺失whl包
2. `use_flash_attn` 需要在A100机器开启，否则loss可能不正常（很快变成0.00x,非常小不正常）。建议使用cuda11.8环境。

<a name="1"></a>


## 微调

```shell
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

## lora微调

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


<a name="2"></a>

## 模型预测

```shell
python predict_generation.py

```

<a name="3"></a>

## 动转静

```shell
python export_generation_model.py
```


<a name="4"></a>

## 模型推理

```shell
python infer_generation.py \
    --model_dir inference \
    --model_prefix gpt
```
