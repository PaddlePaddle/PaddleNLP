# Auto Parallel User Guide

This README provides detailed instructions on how to use auto parallel for large model pretraining, SFT (Supervised Fine-Tuning), LoRA (Low-Rank Adaptation), DPO (Direct Preference Optimization), and inference.

## Table of Contents
- [Auto Parallel User Guide](#auto-parallel-user-guide)
  - [Table of Contents](#table-of-contents)
  - [Currently Supported Models](#currently-supported-models)
  - [Environment Setup](#environment-setup)
  - [Pretraining](#pretraining)
    - [Data Preparation](#data-preparation)
    - [Start Pretraining](#start-pretraining)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
    - [Data Preparation](#data-preparation-1)
    - [Start Fine-Tuning](#start-fine-tuning)
  - [Low-Rank Adaptation (LoRA)](#low-rank-adaptation-lora)
  - [DPO](#dpo)
  - [Inference](#inference)
    - [Dynamic Graph Inference](#dynamic-graph-inference)
    - [Static Graph Inference](#static-graph-inference)
  - [FAQ](#faq)

## Currently Supported Models
| Model | Pretrain | SFT | LoRA | DPO |
|-------|----------|-----|-----|-----|
| GPT-3 |    ✅    |  🚧   |  🚧  | 🚧   |
| Llama |    ✅    |  ✅   |  ✅  | ✅   |
| Qwen  |    ✅    |  🚧   |  🚧  | 🚧   |
| DeepSeek-V3| ✅   |  🚧   |  🚧  | 🚧   |

- ✅: Supported
- 🚧: In Progress

Note: The current DeepSeek-v3 model configuration provided is a small-scale example demo (with reduced network layers) to support running on single-node 8-GPU environments. If you want to run the full 671B-scale DeepSeek-v3, you need to configure 61 layers and adjust the parallel strategy accordingly. The current auto parallel version of deepseek-v3 does not yet integrate FP8, DeepEP and other optimization strategies.

## Environment Setup

1. Install the latest version of PaddlePaddle

First, you need to install the latest `Paddle`, recommended to use the `Nightly` version. Visit [Paddle Official Website](https://www.paddlepaddle.org.cn/install/quick?docurl=undefined) for installation instructions.

2. Verify Paddle Installation

```python
import paddle
print(paddle.utils.run_check())
```

3. Install PaddleNLP and Custom Operators

Please refer to [PaddleNLP Installation Guide](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/zh/get_started/installation.rst) for installation instructions.

## Pretraining

### Data Preparation

We provide preprocessed data for user testing. Download to the `data` directory:

```shell
mkdir -p data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.{bin,idx}
```

### Start Pretraining

#### GPU Pretraining Launch

- Dynamic Graph Mode

```shell
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" llm/llama/pretrain/pretrain_auto.py \
    --model_name_or_path llm/llama/pretrain/llama-7b-en \
    --tokenizer_name_or_path llm/llama/pretrain/llama-7b-en \
    --input_dir ./data \
    --output_dir ./pretrain/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --warmup_steps 2000 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --save_steps 2000 \
    --dataloader_num_workers 8 \
    --sharding parallel \
    --fp16
```

- Static Graph Mode

```shell
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" llm/llama/pretrain/pretrain_auto_static.py \
    --model_name_or_path llm/llama/pretrain/llama-7b-en \
    --tokenizer_name_or_path llm/llama/pretrain/llama-7b-en \
    --input_dir ./data \
    --output_dir ./pretrain/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --warmup_steps 2000 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --save_steps 2000 \
    --dataloader_num_workers 8 \
    --fp16
```

## Supervised Fine-Tuning (SFT)

### Data Preparation

Download SFT data to the `sft_data` directory:

```shell
mkdir -p sft_data && cd sft_data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/sft/data/alpaca_data_en_cleaned.json
```

### Start Fine-Tuning

```shell
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" llm/llama/sft/train.py \
    --model_name_or_path llm/llama/pretrain/llama-7b-en \
    --tokenizer_name_or_path llm/llama/pretrain/llama-7b-en \
    --data_path ./sft_data/alpaca_data_en_cleaned.json \
    --output_dir ./sft/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --warmup_steps 2000 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --save_steps 2000 \
    --dataloader_num_workers 8 \
    --fp16
```

## Low-Rank Adaptation (LoRA)

```shell
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" llm/llama/lora/train.py \
    --model_name_or_path llm/llama/pretrain/llama-7b-en \
    --tokenizer_name_or_path llm/llama/pretrain/llama-7b-en \
    --data_path ./sft_data/alpaca_data_en_cleaned.json \
    --output_dir ./lora/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --warmup_steps 2000 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --save_steps 2000 \
    --dataloader_num_workers 8 \
    --fp16
```

## DPO

```shell
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" llm/llama/dpo/train.py \
    --model_name_or_path llm/llama/pretrain/llama-7b-en \
    --tokenizer_name_or_path llm/llama/pretrain/llama-7b-en \
    --data_path ./dpo_data/ \
    --output_dir ./dpo/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --warmup_steps 2000 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --save_steps 2000 \
    --dataloader_num_workers 8 \
    --fp16
```

## Inference

### Dynamic Graph Inference

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("llm/llama/pretrain/llama-7b-en")
tokenizer = AutoTokenizer.from_pretrained("llm/llama/pretrain/llama-7b-en")

input_text = "Describe the meaning of life:"
inputs = tokenizer(input_text, return_tensors="pd")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### Static Graph Inference

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("llm/llama/pretrain/llama-7b-en", use_reorder_sequence=True)
tokenizer = AutoTokenizer.from_pretrained("llm/llama/pretrain/llama-7b-en")

input_text = "Explain machine learning:"
inputs = tokenizer(input_text, return_tensors="pd")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## FAQ

**Q: How to adjust the parallel strategy?**
A: Modify the `parallel_strategy` parameter in the configuration file. Example:
```yaml
parallel_config:
  pp_degree: 2
  mp_degree: 4
  vpp_degree: 1
  num_micro_batches: 4
  tensor_parallel_config:
    tensor_partitioning: True
```
```python
# Llama pretrain example
# assume that cur dir is auto_parallel
# cd ${PaddleNLP_Path}/llm/auto_parallel/
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7"            \
    --log_dir "llama_auto_3d"           \
    ./llama/run_pretrain_auto.py ./llama/pretrain_argument.json
```

This configuration runs the `facebook/llama-7b` pretraining task with a parallel strategy of MP2-PP2-DP2 and a sharding strategy of Stage1.
For more configurable parameters, please refer to `ModelArguments`, `DataArguments`, and `PreTrainingArguments`.

- Dynamic to Static Mode
<br>Add the `to_static` parameter

#### XPU Launch Pretraining

In addition to GPUs, XPU also supports automatic parallelization. Currently, it supports the 7b and 13b variants of the llama model, with more models under active development.

Users can utilize the `run_llama2_7b_xpu.sh` and `run_llama2_13b_xpu.sh` scripts in the `PaddleNLP/llm/auto_parallel/llama` directory to launch XPU-based pretraining tasks.

```shell
# cd ${PaddleNLP_Path}/llm/auto_parallel/llama
bash run_llama2_7b_xpu.sh
# or
bash run_llama2_13b_xpu.sh
```

The parallel strategy for Llama 7b is DP8 with Stage1 sharding. For Llama 13b, the parallel strategy is DP2-PP4 with Stage1 sharding.

## Supervised Fine-Tuning (SFT)
### Data Preparation

The project provides preprocessed fine-tuning data for user testing. Download and extract to the `data` directory:

```shell
wget -O AdvertiseGen.tar.gz https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz
tar -xvf AdvertiseGen.tar.gz
```

### Launch Fine-Tuning

- Dynamic Graph Mode
```python
# Llama finetune example
# assume that cur dir is auto_parallel
# cd ${PaddleNLP_Path}/llm/auto_parallel/
python -u -m paddle.distributed.launch \
  --gpus "0,1,2,3,4,5,6,7" \
  ./run_finetune_auto.py ./llama/finetune_argument.json
```
This configuration runs the `Meta-Llama-3.1-8B-Instruct` task with a parallel strategy of MP2-PP2-DP2 and Stage2 sharding.
For more configurable parameters, please refer to `GenerateArgument`, `ModelAutoConfig`, `ReftArgument`, `DataConfig`, and `SFTAutoConfig`.

- Dynamic to Static Mode
<br>Add the `to_static` parameter
## Low-Rank Adaptation (LoRA)
Enable LoRA on top of SFT by setting `lora` and `lora_rank` parameters. For more parameters, refer to [model_config.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/trl/model_config.py).

## DPO
### Data Preparation
For testing convenience, we preprocess the [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) dataset into the required format. Run the following command in the PaddleNLP/llm directory:
```shell
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized.tar.gz
tar -zxvf ultrafeedback_binarized.tar.gz
```

### Launch DPO Training
Run the following script in the PaddleNLP/llm/auto_parallel/llama directory:
```shell
bash llama_dpo_with_api.sh
```
The `to_static` parameter controls whether to enable the dynamic-to-static graph conversion mode.

## Inference
The inference workflow includes: dynamic graph inference, dynamic-to-static graph model export → static graph inference.

### Dynamic Graph Inference
The current automatically parallelized model checkpoints support dynamic graph inference. Taking dynamic graph automatic parallel training (DP2-MP2-PP2) as an example:
- Merge distributed checkpoints into single-GPU model parameters
```python
import paddle
import paddle.distributed as dist

ckpt_path='/path/for/dist_ckpt'
# offload=1, offload parameters to CPU to reduce memory usage
# prefix="model" can be used to filter out non-model parameters, such as optimizer states
merged_state_dict = dist.checkpoint.load_state_dict.load_merged_state_dict(ckpt_path, offload=1, prefix="model")
paddle.save(merged_state_dict, 'model_state.pdparams')

# The merged model parameters above are in Paddle native format. To convert to unified checkpoint format (safetensors), or to obtain the index file for model parameters, continue with the following code:
python PaddleNLP/llm/auto_parallel/utils/convert_to_safetensors.py --input_path input_path  [--output_path output_path] [--split_num split_num] [--offload] [--as_safetensors]

# Parameter description
--input_path: Path to input single-card model parameters
--output_path: Optional, output model parameter path, defaults to './temp'
--split_num: Optional, number of output model parameter shards, defaults to 1
--offload: Optional, controls whether to offload parameters to CPU
--as_safetensors: Optional, controls whether to convert model parameters to safetensors format
```

- Dynamic Graph Inference
<br>Please refer to [Large Model Inference Tutorial](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/inference.md).

### Static Graph Inference
For model export via dynamic-to-static and static graph inference steps, please refer to [LLaMA Series Large Model Operation Guide](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/llama.md).

## FAQ

Q1: How to adjust when OOM occurs?
- Reduce batch_size
- Enable fuse_attention_ffn, fuse_flash_qkv
