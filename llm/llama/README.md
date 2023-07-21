# LLaMA inplementation

**目录**

- [1. 预训练](#0)
- [2. 微调](#1)
- [3. 模型预测](#2)
- [4. 动转静](#3)
- [5. 模型推理](#4)

## 协议

Llama 模型的权重的使用则需要遵循[License](../../../paddlenlp/transformers/llama/LICENSE)。


<a name="0"></a>

## 预训练

预训练数据制作参考[此处](../../../model_zoo/ernie-1.0/preprocess/docs/OpenWebText2.md)

为了方便用户运行测试本模型，本项目提供了处理好的100k条doc的训练样本：
```shell
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz
```

将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：

```
mkdir data
mv llama_openwebtext_100k_ids.npy ./data
mv llama_openwebtext_100k_idx.npz ./data
```

使用下面脚本,即可在llama-7b的基础上,继续训练.
```shell
task_name="llama_hybid"
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name""_log" \
    run_pretrain.py \
    --model_type "llama" \
    --model_name_or_path "facebook/llama-7b" \
    --tokenizer_name_or_path "facebook/llama-7b" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention 1 \
    --use_fused_rms_norm 0 \
    --fp16  \
    --fp16_opt_level "O2"  \
    --scale_loss 1024 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --lr_scheduler_type "cosine" \
    --max_steps 10000 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 20\
    --dataloader_num_workers 1 \
    --sharding "stage2" \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --continue_training 1\
    --recompute 1 \
    --do_train \
    --do_eval \
    --device "gpu"
```
注意：
1. 需要paddle develop版本训练，需要安装`pip install tool_helpers visualdl==2.5.3`等相关缺失whl包
2. `use_flash_attention` 需要在A100机器开启，否则loss可能不正常（很快变成0.00x,非常小不正常）。建议使用cuda11.8环境。
3. `continue_training` 表示从现有的预训练模型加载训练。7b模型初始loss大概为1.99x, 随机初始化模型loss从11.x左右下降。
4. `use_fused_rms_norm` 需要安装[此目录](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/gpt-3/external_ops)下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
5. 当前脚本为sharding版本，需要4D并行训练（数据、sharding、张量、流水线并行）的用户，请参考 `run_trainer_tp4pp2.sh`脚本。

<a name="1"></a>

## 微调

```shell
python -u  -m paddle.distributed.fleet.launch \
    --gpus "0,1,2,3" finetune_generation.py \
    --output_dir ./checkpoints/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path facebook/llama-7b \
    --task_name squad \
    --num_train_epochs 2 \
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
    --recompute \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --tensor_parallel_degree 4
```

### 单卡LoRA微调

```shell
python finetune_generation.py \
    --output_dir ./checkpoints/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path facebook/llama-7b \
    --task_name squad \
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

### 单卡Prefix微调

```shell
python finetune_generation.py \
    --output_dir ./checkpoints/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path facebook/llama-7b \
    --task_name squad \
    --num_train_epochs 2 \
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
- `eval_with_do_generation`: 在评估的时候是否调用model.generate,默认为False。
- `lora`: 是否使用LoRA技术。
- `merge_weights`: 是否合并原始模型和Lora模型的权重。
- `lora_rank`: lora 算法中rank（秩）的值，默认为8。
- `lora_path`: lora参数和配置路径，对lora参数进行初始化。
- `qat`: 是否使用qat对模型进行量化
- `qat_type`: qat量化类型，支持A8W8, W4, A8W4。默认为A8W8。
- `prefix_tuning`: 是否使用Prefix技术。
- `num_prefix_tokens`: prefix tuning算法中前缀token数量。
- `task_name`: 内置数据集任务名
- `data_name`: 内置数据集名，定义数据集名必须同时定义数据集任务名
- `dataset_path`: 自定义数据集路径。

## 流水线并行
```shell
python -u  -m paddle.distributed.launch \
    --gpus "4,5,6,7"   finetune_generation.py \
    --model_name_or_path __internal_testing__/tiny-random-llama \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 2 \
    --pipeline_parallel_config "disable_p2p_cache_shape" \
    --overwrite_output_dir \
    --output_dir ./checkpoints/ \
    --logging_steps 1 \
    --disable_tqdm 1 \
    --eval_steps 100 \
    --eval_with_do_generation 0 \
    --fp16 0\
    --fp16_opt_level O2 \
    --recompute 0 \
    --learning_rate 3e-5 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --warmup_steps 20
```

<a name="2"></a>

## 模型预测

```shell
python predict_generation.py \
    --model_name_or_path ./checkpoints/
```

当ckpt为使用的tensor parallel存储为多分片格式时，也可使用此脚本预测，或者合并为一个单分片权重 例如下面4分片的例子（此模型为glm-10b-chinese）

```shell
-rw-r--r-- 1 root root  523 Apr 13 11:46 config.json
-rw-r--r-- 1 root root 3.2G Apr 13 11:46 model_state.tp00.pdparams
-rw-r--r-- 1 root root 3.2G Apr 13 11:46 model_state.tp01.pdparams
-rw-r--r-- 1 root root 3.2G Apr 13 11:46 model_state.tp02.pdparams
-rw-r--r-- 1 root root 3.2G Apr 13 11:46 model_state.tp03.pdparams
```

设置 merge_tensor_parallel_path，可以将merge好的参数存储到对应位置。不过不设置此参数，将只跑前向预测。

```shell
python -m paddle.distributed.launch --gpus 0,1,2,3 predict_generation.py \
    --model_name_or_path  ./checkpoints/checkpoint-100/ \
    --merge_tensor_parallel_path  ./checkpoints/llama-merged
```

### LoRA微调模型预测
对merge后的单分片模型也可以进行直接预测，脚本如下
```shell
 python predict_generation.py
    --model_name_or_path facebook/llama-7b \
    --lora_path ./checkpoints
```

### Prefix微调模型预测
对merge后的单分片模型也可以进行直接预测，脚本如下
```shell
 python predict_generation.py
    --model_name_or_path facebook/llama-7b \
    --prefix_path ./checkpoints
```

<a name="3"></a>

## 动转静

```shell
python export_generation_model.py \
    --model_path checkpoints/ \
    --output_path inference/llama
```

当在指定数据集上进行 LoRA finetune 后的导出脚本：


```shell
python export_generation_model.py
    --model_name_or_path facebook/llama-7b
    --output_path inference/llama
    --lora_path ./checkpoints
```

<a name="4"></a>

## 模型推理

```shell
python infer_generation.py \
    --model_dir inference \
    --model_prefix llama
```

结果：

```text
answer: linebacker context: The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2½ sacks, and two forced fumbles. </s>

question: What was von Miller's job title?
--------------------
answer: five context: The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2½ sacks, and two forced fumbles. </s>

question: How many total tackles did von Miller get in the Super Bowl?
--------------------
```

## 服务化推理

提供基于 UI 服务化推理，以下命令将会：

1. 启动多卡模型服务，并让其常驻显存，等待执行。
2. 启动 Flask 服务，监听外部请求。
3. 启动 Gradio UI 服务，提供可视化交互界面。

```shell
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" server.py \
    --model_name_or_path facebook/llama-7b \
    --port 8010 \
    --flask_port 8011 \
    --src_length 100
```
python predict_generation.py --model_name_or_path idea-ccnl/ziya-llama-13b-v1 --data_file /root/paddlejob/work/eb_data/hcg/dev.json
