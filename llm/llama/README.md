# LLaMA

## 1. 模型介绍

**支持模型权重:**

| Model                            |
| ---------------------------------|
| facebook/llama-7b                 |
| facebook/llama-13b                |
| facebook/llama-30b                |
| facebook/llama-65b                |
| meta-llama/Llama-2-7b             |
| meta-llama/Llama-2-7b-chat        |
| meta-llama/Llama-2-13b            |
| meta-llama/Llama-2-13b-chat       |
| meta-llama/Llama-2-70b            |
| meta-llama/Llama-2-70b-chat       |
| ziqingyang/chinese-llama-7b       |
| ziqingyang/chinese-llama-13b      |
| ziqingyang/chinese-alpaca-7b      |
| ziqingyang/chinese-alpaca-13b     |
| idea-ccnl/ziya-llama-13b-v1       |
| linly-ai/chinese-llama-2-7b       |
| baichuan-inc/Baichuan-7B          |
| baichuan-inc/Baichuan-13B-Base    |
| baichuan-inc/Baichuan-13B-Chat    |
| baichuan-inc/Baichuan2-7B-Base    |
| baichuan-inc/Baichuan2-7B-Chat    |
| baichuan-inc/Baichuan2-13B-Base   |
| baichuan-inc/Baichuan2-13B-Chat   |
| FlagAlpha/Llama2-Chinese-7b-Chat  |
| FlagAlpha/Llama2-Chinese-13b-Chat |



使用方法：

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")
```

## 2. 模型协议

LLaMA 模型的权重的使用则需要遵循[License](../../paddlenlp/transformers/llama/LICENSE)。

Llama2 模型的权重的使用则需要遵循[License](../../paddlenlp/transformers/llama/Llama2.LICENSE)。


## 3. 预训练

预训练数据制作参考[此处](../../model_zoo/ernie-1.0/preprocess/docs/OpenWebText2.md)

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
task_name_or_path="llama_hybrid"
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name_or_path""_log" \
    run_pretrain.py \
    --model_type "llama" \
    --model_name_or_path "facebook/llama-7b" \
    --tokenizer_name_or_path "facebook/llama-7b" \
    --input_dir "./data" \
    --output_dir "output/$task_name_or_path" \
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
2. `use_flash_attention` 需要在A100机器开启，建议使用cuda11.8环境。
3. `continue_training` 表示从现有的预训练模型加载训练。7b模型初始loss大概为1.99x, 随机初始化模型loss从11.x左右下降。
4. `use_fused_rms_norm` 需要安装[此目录](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/gpt-3/external_ops)下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
5. 当前脚本为sharding版本，需要4D并行训练（数据、sharding、张量、流水线并行）的用户，请参考 `run_trainer_tp4pp2.sh`脚本。

## 4. 模型精调
请参考[LLM全流程工具介绍](../README.md)
