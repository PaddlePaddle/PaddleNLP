# ERNIE-3.5-SE

## 1. 模型介绍

我们采用了 Attention 和 FFN 并行的 Parallel Transformer 的实现方式，将 FFN 和 Attention 层进行并行计算。通过这样的设计，我们可以把 Attention 和 FFN 需要的线形层计算进行算子融合，降低 kernel 调用以及通讯次数，提升并行训练的效率。并且我们发现第一层的 FFN 和最后一层的 Attn 作用不大，因此采用了“掐头去尾”策略，将底层的 FFN 的计算量挪到模型的顶层，在同 FLOPs 下效果和传统 Transformer 结构一致，但有更好的训练速度和吞吐。

<table>
<tr>
 <td><img src="https://github.com/PaddlePaddle/PaddleNLP/assets/16911935/89ca3093-4039-44c7-abce-4a47de6af1f6" height="300"> </td>
 <td><img src="https://github.com/PaddlePaddle/PaddleNLP/assets/16911935/3c89a72d-34b8-4711-b13e-d31063fc92d3" height="300"> </td>
</tr>
<tr>
 <td> Parallel Transformer </td>
 <td> “掐头去尾”策略 </td>
</tr>
</table>


* Rope Embedding+[随机位置编码](https://aclanthology.org/2023.acl-short.161)：我们采用的旋转位置编码 Rope，并且为了有较好的模型外推能力，我们保留了线形层的 Bias。为了提供长文外推能力，我们通过随机间隔取 Position Ids，让模型能够有训短推长的能力。

<img src="https://github.com/PaddlePaddle/PaddleNLP/assets/20554008/423622c1-aed9-4ea9-83b0-d5d3efbaf35b" title="随机位置编码" height="300">

* Sequence Length Warmup：通过动态调整前期训练的序列长度，提升模型的收敛效率。


## 2. 预训练

预训练数据制作参考[此处](../../tools/preprocess/docs/OpenWebText2.md)

为了方便用户运行测试本模型，本项目提供了处理好的100k 条 doc 的训练样本：
```shell
wget https://bj.bcebos.com/paddlenlp/models/transformers/ernie/data/ernie_openwebtext_100k_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/ernie/data/ernie_openwebtext_100k_idx.npz
```

将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：

```
mkdir data
mv ernie_openwebtext_100k_ids.npy ./data
mv ernie_openwebtext_100k_idx.npz ./data
```

使用下面脚本,即可启动 ernie-3.5-se-3b 的预训练，也可直接参考 run_trainer_stage2.sh。
```shell
task_name="ernie35_hybrid"
python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name""_log" \
    run_pretrain.py \
    --model_type "ernie" \
    --model_name_or_path "baidu/ernie-3.5-se-3b" \
    --tokenizer_name_or_path "ernie-tokenizer" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention 1 \
    --use_fused_ln 1 \
    --bf16 \
    --fp16_opt_level "O2"  \
    --scale_loss 512 \
    --learning_rate 0.0003 \
    --min_learning_rate 0.00003 \
    --lr_scheduler_type "cosine" \
    --max_steps 300000 \
    --save_steps 200 \
    --adam_beta2 0.95 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --max_grad_norm 1.0 \
    --logging_steps 2 \
    --dataloader_num_workers 0 \
    --sharding "stage2" \
    --sharding_parallel_degree 8 \
    --eval_steps 200 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --continue_training 0\
    --recompute 1 \
    --do_train \
    --do_eval \
    --save_total_limit 10 \
    --device "gpu"
```
注意：
1. 需要 paddle develop 版本训练，需要安装`pip install fast_dataindex visualdl==2.5.3`等相关缺失 whl 包
2. `use_flash_attention` 需要在 A100机器开启，否则 loss 可能不正常（很快变成0.00x,非常小不正常）。建议使用 cuda11.8环境。
3. `continue_training` 表示从现有的预训练模型加载训练，如果需要从头开始预训练模型，则设置为0。
4. `use_fused_ln` 需要安装[此目录](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/gpt-3/external_ops)下的自定义 OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置 PYTHONPATH
5. 当前脚本为 sharding 版本，需要4D 并行训练（数据、sharding、张量、流水线并行）的用户，可另外调整相关参数。



## 3. 精调

### SFT
```shell
python -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    finetune_generation.py \
    --output_dir "output_sft/$task_name" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path <PATH_TO_CKPT> \
    --task_name squad \
    --num_train_epochs 2 \
    --learning_rate 3e-5 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 1024 \
    --tgt_length 1024 \
    --bf16 \
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
    --sharding "stage2" \
    --sharding_parallel_degree 8
```

### LoRA
```shell
python finetune_generation.py \
    --output_dir ./checkpoints/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path <PATH_TO_CKPT> \
    --task_name squad \
    --num_train_epochs 2 \
    --learning_rate 3e-4 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --src_length 1024 \
    --tgt_length 1024 \
    --bf16 \
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

其中参数释义如下：

- `model_name_or_path`: 预训练模型内置名称或者模型所在目录.
- `num_train_epochs`: 要执行的训练 epoch 总数（如果不是整数，将在停止训练之前执行最后一个 epoch
的小数部分百分比）。
- `max_steps`: 模型训练步数。
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
- `bf16`: 使用 bfloat16 精度进行模型训练和推理。
- `fp16_opt_level`: bfloat16 精度训练模式，`O2`表示纯 bfloat16 训练。
- `recompute`: 使用重计算策略，开启后可节省训练显存。
- `do_train`: 是否训练模型。
- `do_eval`: 是否评估模型。
- `tensor_parallel_degree`: 模型并行数量。
- `eval_with_do_generation`: 在评估的时候是否调用 model.generate,默认为 False。
- `lora`: 是否使用 LoRA 技术。
- `merge_weights`: 是否合并原始模型和 LoRA 模型的权重。
- `lora_rank`: LoRA 算法中 rank（秩）的值，默认为8。
- `lora_path`: LoRA 参数和配置路径，对 LoRA 参数进行初始化。
- `task_name`: 内置数据集任务名
- `data_name`: 内置数据集名，定义数据集名必须同时定义数据集任务名
- `dataset_path`: 自定义数据集路径。


## 4. 动态图预测

```shell
python predict_generation.py \
    --model_name_or_path <PATH_TO_CKPT> \
    --tokenizer_name_or_path ernie-tokenizer
```
