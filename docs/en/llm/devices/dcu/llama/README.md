# Running llama2-13b Model on Hygon DCU with PaddleNLP
Paddle framework and PaddleNLP suite have undergone deep adaptation and optimization on Hygon DCU products, achieving high consistency with GPU in large model training and inference, with leading levels in accuracy and performance.

Hygon DCU products offer multiple technical advantages in the PaddleNLP combined suite:

- **Full support for 4D hybrid parallel distributed training, flexibly adapting to various training strategies.**
- **Various high-performance fused operators to enhance training and inference performance.**
- **Optimized communication libraries to mask distributed training and inference latency.**

## 🚀 Quick Start 🚀

### Environment Preparation:

#### 1. Hardware Platform

| Chip Type | DTK Version |
| --- | --- |
| K100_AI | 24.04.1 |

**This example uses an 8-card machine and demonstrates the workflow through fine-tuning training + inference. Use the hy-smi command to view DCU information in the runtime environment, as shown below:**
```
$ hy-smi

============================ System Management Interface =============================
======================================================================================
DCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      DCU%      Mode
0       49.0C    118.0W     auto     800.0W     0%         0%        Normal
1       48.0C    120.0W     auto     800.0W     0%         0%        Normal
2       53.0C    116.0W     auto     800.0W     0%         0%        Normal
3       49.0C    138.0W     auto     800.0W     0%         0%        Normal
======================================================================================
=================================== End of SMI Log ===================================
```

#### 2. Environment Setup:
It is recommended to run using docker. The provided docker image can be pulled, and new versions of DTK required for this project can be downloaded and installed from [Hygon Developer Community](https://developer.hpccube.com/tool/). The docker environment uses dtk-24.04.1 by default.

(1) Pull the image
```
# Note: This image is for development environment only and does not contain precompiled Paddle packages
docker pull registry.baidubce.com/device/paddle-dcu:dtk24.04.1-kylinv10-gcc82
```
(2) Start the container with the following command:
```
docker run -it \
    --network=host \
    --name=paddle_llama \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size=128G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -u root \
    --ulimit stack=-1:-1 \
    --ulimit memlock=-1:-1 \
    -v $(pwd):/workspace \
    -v /opt/hyhal:/opt/hyhal \
    registry.baidubce.com/device/paddle-dcu:dtk24.04.1-kylinv10-gcc82 \
    /bin/bash
```

(3) Install PaddlePaddle
```
# PaddlePaddle deep learning framework provides fundamental computing capabilities
python -m pip install paddlepaddle-dcu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/dcu/
```

(4) Clone PaddleNLP repository and install dependencies
```
# Use develop branch of PaddleNLP
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/llm # Navigate to execution directory
pip install -r ../requirements.txt
```

(5) Install paddlenlp_ops
```
# PaddleNLP repository contains specialized operators for RMS
cd slm/model_zoo/gpt-3/external_ops
python setup.py install
```

## 3.Fine-tuning:
- **Note:** Perform the following operations in the llm directory.

### Dataset Preparation
We provide a demo dataset for debugging:
```
wget https://bj.bcebos.com/paddlenlp/datasets/examples/alpaca_demo.gz
tar -xvf alpaca_demo.gz
```

The supported fine-tuning data format is a json file where each line contains a dictionary with the following fields:
- `src`: `str, List(str)`, represents the model's input instruction (prompt) that the model should execute.
- `tgt`: `str, List(str)`, represents the model's output.

Sample data:
```
{"src": "Type#Dress*Color#Blue*Style#Fresh*Pattern#Bow", "tgt": "The dress features 3D bow decorations with blue ribbon details, creating a full and layered silhouette while adding a touch of sweetness. This highlights the girl's fresh and charming appearance."}
...
# You can prepare your own fine-tuning data following this format.
```

### LoRA Fine-tuning

You can use the following script to start LoRA fine-tuning:
```bash
PYTHONPATH=.. python run_finetune.py dcu/llama/lora_argument.json
```

### SFT Fine-tuning
For Lsft fine-tuning training, you may refer to the following hyperparameters:

```bash
PYTHONPATH=.. python run_finetune.py dcu/llama/sft_argument.json
```

## 3. Pre-training
### Data Preparation
For detailed data preparation workflow, please refer to [here](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/tools/preprocess/README.md). Example: OpenWebText2 pre-training data preparation refers to [here](https://paddlenlp.readthedocs.io/zh/latest/llm/pretraining/data/).

For user convenience in testing, this project provides processed 100k doc training samples:

```bash
cd PaddleNLP/llm/
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx
cd .. && tree data
data
├── llama_openwebtext_100k.bin
└── llama_openwebtext_100k.idx
```

- **Note:** Distinguish path from fine-tuning dataset

### Execution Script
This training script can run on single-node or multi-node environments, with 8 DCU-K100AI-64G per node.

Parallel configuration uses TP 1, PP 8, with fp16 precision for pre-training.

Refer to the following script to start pre-training:

```bash
python -m paddle.distributed.launch \
    --gpus '0,1,2,3,4,5,6,7' \
    run_pretrain.py dcu/llama/pretrain_pp8.json
```

## 4. High-Performance Inference
The high-performance inference system integrates dynamic insertion and full-process operator fusion strategies, abstracting underlying implementation details while delivering out-of-the-box high-performance parallel inference capabilities. It dynamically allocates storage space for cachekv while maintaining high-performance inference and dynamic insertion, significantly saving memory to handle more queries simultaneously for throughput improvement.

(1) Environment Preparation

PaddleNLP provides high-performance custom operators for Transformer series to boost model performance during inference and decoding. Please install the custom operator library first:

```bash
# Install custom operators for DCU devices
cd PaddleNLP/csrc && python3 setup_hip.py install
```

(2) High-Performance Inference

Below are reference commands for high-performance inference with BlockAttention disabled and enabled respectively:

a. High-Performance Inference with BlockAttention Disabled

**Dynamic Graph:**

```bash
python predictor.py --model_name_or_path meta-llama/Llama-2-7b --inference_mode dynamic --block_attn false
```
```
# fp16
python3 ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --dtype float16 (For benchmarking: --batch_size 1 --src_length 3072 --max_length 1024 --benchmark)
# a8w8
python3 ./predict/predictor.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --dtype float16 (For benchmarking: --batch_size 1 --src_length 3072 --max_length 1024 --benchmark)
```

**Static Graph:**

```
# Step1: Export Static Graph
# fp16
python3 ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --output_path ./inference --dtype float16
# a8w8
python3 ./predict/export_model.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --output_path ./inference --dtype float16

# Step2: Static Graph Inference
python3 ./predict/predictor.py --model_name_or_path ./inference --inference_model --dtype float16 --mode static (For benchmarking: --batch_size 1 --src_length 3072 --max_length 1024 --benchmark)
```

b. Enable high-performance inference with BlockAttention

**Dynamic Graph:**

```
```
# fp16
python3 ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --dtype float16 --block_attn (For performance testing (optional): --batch_size 1 --src_length 3072 --max_length 1024 --benchmark)
# a8w8
python3 ./predict/predictor.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --dtype float16 --block_attn (For performance testing (optional): --batch_size 1 --src_length 3072 --max_length 1024 --benchmark)
# cachekv
python3 ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --dtype float16 --block_attn --cachekv_int8 (For performance testing (optional): --batch_size 1 --src_length 3072 --max_length 1024 --benchmark)
```

**Static Graph:**

```
```markdown
# Step1: Static Graph Exporting
# fp16
python3 ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --output_path ./inference --dtype float16 --block_attn
# a8w8
python3 ./predict/export_model.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --output_path ./inference --dtype float16 --block_attn
# cachekv
python3 ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --output_path ./inference --dtype float16 --block_attn --cachekv_int8

# Step2: Static Graph Inference
# fp16
python3 ./predict/predictor.py --model_name_or_path ./inference --inference_model --dtype float16 --mode static --block_attn (For benchmarking: --batch_size 1 --src_length 3072 --max_length 1024 --benchmark)
# a8w8
python3 ./predict/predictor.py --model_name_or_path ./inference --inference_model --dtype float16 --mode static --block_attn (For benchmarking: --batch_size 1 --src_length 3072 --max_length 1024 --benchmark)
# cachekv
python3 ./predict/predictor.py --model_name_or_path ./inference --inference_model --dtype float16 --mode static --cachekv_int8 --block_attn (For benchmarking: --batch_size 1 --src_length 3072 --max_length 1024 --benchmark)

## 5. Application Scenarios

(1). Algorithm Category

`Natural Language Processing`

(2). Hot Application Industries

`Healthcare, Education, Research, Finance`

## 6. Source Repository & Issue Reporting

- [https://developer.hpccube.com/codes/modelzoo/llama_paddle](https://developer.hpccube.com/codes/modelzoo/llama_paddle)

## 7. References

* [https://github.com/PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)
```
