# DeepSeek

This document demonstrates how to build and run the [DeepSeek](https://www.deepseek.com/) series of large models in PaddleNLP.

## Model Introduction

* The DeepSeek series of large models are efficient open-source language models developed by DeepSeek Inc., focusing on improving model inference efficiency and multi-scenario application capabilities.

* [DeepSeek V3](https://www.deepseek.com/): In December 2024, the first version of DeepSeek-V3 was released and open-sourced. DeepSeek-V3 is an MoE model with 671B parameters and 37B activated parameters.
* [DeepSeek R1](https://www.deepseek.com/): In January 2025, DeepSeek released DeepSeek-R1 and open-sourced the model weights.
* [DeepSeek R1 Distill Model](https://www.deepseek.com/): In January 2025, while open-sourcing the R1 model, DeepSeek also distilled and open-sourced 6 smaller models using DeepSeek-R1's outputs, namely Qwen1.5B, 7B, 14B, 32B, as well as Llama8B and 70B.

## Verified Models (CKPT)

|Model|
|:-|
|deepseek-ai/DeepSeek-V2-Chat|
|deepseek-ai/DeepSeek-V2-Lite-Chat|
|deepseek-ai/DeepSeek-V3|
|deepseek-ai/DeepSeek-R1|
|deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|
|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|
|deepseek-ai/DeepSeek-R1-Distill-Qwen-14B|
|deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|
|deepseek-ai/DeepSeek-R1-Distill-Llama-8B|
|deepseek-ai/DeepSeek-R1-Distill-Llama-70B|

## Pre-built Static Graphs

|Model Name|Precision|MTP|Number of Nodes|Static Graph Download model_name|
|:------|:-:|:-:|:-:|:-:|
| deepseek-ai/DeepSeek-R1  |weight_only_int4|No|1| deepseek-ai/DeepSeek-R1/weight_only_int4 |
| deepseek-ai/DeepSeek-R1  |weight_only_int4|Yes|1| deepseek-ai/DeepSeek-R1-MTP/weight_only_int4 |
| deepseek-ai/DeepSeek-R1  |a8w8_fp8_wint4|No|1| deepseek-ai/DeepSeek-R1/a8w8_fp8_wint4 |
| deepseek-ai/DeepSeek-R1  |a8w8_fp8_wint4|Yes|1| deepseek-ai/DeepSeek-R1-MTP/a8w8_fp8_wint4 |
| deepseek-ai/DeepSeek-R1  |weight_only_int8|No|2| deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8 |
| deepseek-ai/DeepSeek-R1  |weight_only_int8|Yes|2| deepseek-ai/DeepSeek-R1-MTP-2nodes/weight_only_int8 |
| deepseek-ai/DeepSeek-R1  |a8w8_fp8|No|2| deepseek-ai/DeepSeek-R1-2nodes/a8w8_fp8|
| deepseek-ai/DeepSeek-R1  |a8w8_fp8|Yes|2| deepseek-ai/DeepSeek-R1-MTP-2nodes/a8w8_fp8|
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |weight_only_int8|-|-| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B  |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Llama-8B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Llama-70B/weight_only_int8 |


## One-Click Inference Service

### deepseek-ai/DeepSeek-R1

Single-node WINT4-TP8 Inference
```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1/weight_only_int4"}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=8 && start_server $model_name && tail -f /dev/null'
```

Two-node WINT8-TP16 Inference

```shell
Ensure two machine nodes can ping each other
# First node (master)
ping 192.168.0.1
# Second node (slave)
ping 192.168.0.2
model_name=${model_name:-"deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8"}
export POD_0_IP=master_ip
export POD_IPS=master_ip,slave_ip # This environment variable needs to be consistent on both machines
# Default service port, modify via export if port conflicts occur
export SERVICE_HTTP_PORT=${PUSH_MODE_HTTP_PORT:-${SERVICE_HTTP_PORT:-"9965"}}
# MODEL_PATH # Model mounting path
```
```shell
# node1
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8"}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 && start_server $model_name && tail -f /dev/null'

# node2
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8"}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}"\
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 && start_server $model_name && tail -f /dev/null'
```

Dual-node FP8-TP16 Inference
```shell
# node1
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-2nodes/a8w8_fp8"}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 && start_server $model_name  && tail -f /dev/null'

# node2
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-2nodes/a8w8_fp8"}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export MP_NUM=16 && export MP_NNODE=2 && export POD_0_IP=192.168.0.1 && export POD_IPS=192.168.0.1,192.168.0.2 && start_server $model_name  && tail -f /dev/null'
```

To enable MTP mode, refer to the [speculative decoding section](./speculative_decoding.md).

### deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/weight_only_int8"}
docker run --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v /MODEL_PATH/:/models -e "model_name=${model_name}"\
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'bash start_server.sh $model_name && tail -f /dev/null'
```

### Request Serving

curl Request
```shell
curl ${ip}:9965/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
      "model":"default",
      "text":"Hello, how are you?"
  }'
```

OpenAI Request
```python
import openai
client = openai.Client(base_url=f"http://127.0.0.1:9965/v1/chat/completions", api_key="EMPTY_API_KEY")
# Non-streaming response
response = client.completions.create(
    model="default",
    prompt="Hello, how are you?",
  max_tokens=50,
  stream=False,
)

print(response)
print("\n")

# Streaming response
response = client.completions.create(
    model="default",
    prompt="Hello, how are you?",
  max_tokens=100,
  stream=True,
)

for chunk in response:
  if chunk.choices[0] is not None:
    print(chunk.choices[0].text, end='')
print("\n")
```

## Model Local Inference

### deepseek-ai/DeepSeek-R1
Single-node WINT4-TP8 Inference
```shell
# Dynamic graph inference
export MODEL_TAG=deepseek-ai/DeepSeek-R1
export QUANT_MODE=weight_only_int4
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=4096
export FLAGS_mla_use_tensorcore=1 # only support Hopper, Amper should be 0
export FLAGS_cascade_attention_max_partition_size=${TOTAL_MAX_LENGTH}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch \
  --gpus ${CUDA_VISIBLE_DEVICES} \
  predictor.py \
  --model_name_or_path ${MODEL_TAG} \
  --dtype bfloat16 \
  --mode dynamic \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --max_length ${MAX_DEC_LEN} \
  --mla_use_matrix_absorption 1


# Export model via dynamic to static
export MODEL_TAG=deepseek-ai/DeepSeek-R1
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=weight_only_int4
export TOTAL_MAX_LENGTH=8192
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch \
  --gpus ${CUDA_VISIBLE_DEVICES} \
  export_model.py \
  --model_name_or_path ${MODEL_TAG} \
  --output_path ${OUTPUT_PATH} \
  --dtype bfloat16 \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --mla_use_matrix_absorption 1


# Static graph inference
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=weight_only_int4
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=4096
export FLAGS_mla_use_tensorcore=1 # only support Hopper, Amper should be 0
export FLAGS_cascade_attention_max_partition_size=${TOTAL_MAX_LENGTH}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch \
  --gpus ${CUDA_VISIBLE_DEVICES} \
  predictor.py \
  --model_name_or_path ${OUTPUT_PATH} \
  --dtype bfloat16 \
  --mode static \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --max_length ${MAX_DEC_LEN} \
  --mla_use_matrix_absorption 1
```
Two-Node WINT8-TP16 Inference

```shell
To perform 2-node inference, ensure both machines can ping each other:
# First node (master)
ping 192.168.0.1
# Second node (slave)
ping 192.168.0.2
```
```shell
# Dynamic graph inference (commands are the same for node1 and node2)
export MODEL_TAG=deepseek-ai/DeepSeek-R1
export QUANT_MODE=weight_only_int8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=4096
export FLAGS_mla_use_tensorcore=1 # only support Hopper, Amper should be 0
export FLAGS_cascade_attention_max_partition_size=${TOTAL_MAX_LENGTH}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch \
  --gpus ${CUDA_VISIBLE_DEVICES} --ips "192.168.0.1,192.168.0.2"\
  predictor.py \
  --model_name_or_path ${MODEL_TAG} \
  --dtype bfloat16 \
  --mode dynamic \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --max_length ${MAX_DEC_LEN} \
  --mla_use_matrix_absorption 1


# Export model via dynamic to static (commands are the same for node1 and node2)
export MODEL_TAG=deepseek-ai/DeepSeek-R1
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=weight_only_int8
export TOTAL_MAX_LENGTH=8192
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch --ips "192.168.0.1,192.168.0.2"\
  --gpus ${CUDA_VISIBLE_DEVICES} \
  export_model.py \
  --model_name_or_path ${MODEL_TAG} \
  --output_path ${OUTPUT_PATH} \
  --dtype bfloat16 \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --mla_use_matrix_absorption 1


# Static graph inference (commands are the same for node1 and node2)
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=weight_only_int8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=4096
export FLAGS_mla_use_tensorcore=1 # only support Hopper, Amper should be 0
export FLAGS_cascade_attention_max_partition_size=${TOTAL_MAX_LENGTH}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch --ips "192.168.0.1,192.168.0.2"\
  --gpus ${CUDA_VISIBLE_DEVICES} \
  predictor.py \
  --model_name_or_path ${OUTPUT_PATH} \
  --dtype bfloat16 \
  --mode static \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --max_length ${MAX_DEC_LEN} \
  --mla_use_matrix_absorption 1
```
2-Node FP8-TP16 Inference

```shell
To launch 2-node inference, ensure the two machine nodes can ping each other
# First node (master)
ping 192.168.0.1
# Second node (slave)
ping 192.168.0.2
```
```shell
# Dynamic graph inference. Commands for node1 and node2 are identical.
export MODEL_TAG=deepseek-ai/DeepSeek-R1-FP8
export QUANT_MODE=a8w8_fp8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=4096
export FLAGS_mla_use_tensorcore=1 # Only support Hopper, Amper should be 0
export FLAGS_cascade_attention_max_partition_size=${TOTAL_MAX_LENGTH}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch --ips "192.168.0.1,192.168.0.2"\
  --gpus ${CUDA_VISIBLE_DEVICES} \
  predictor.py \
  --model_name_or_path ${MODEL_TAG} \
  --dtype bfloat16 \
  --mode dynamic \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --max_length ${MAX_DEC_LEN} \
  --mla_use_matrix_absorption 1 \
  --weight_block_size 128 128


# Export model via dynamic to static. Commands for node1 and node2 are identical.
export MODEL_TAG=deepseek-ai/DeepSeek-R1-FP8
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=a8w8_fp8
export TOTAL_MAX_LENGTH=8192
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch --ips "192.168.0.1,192.168.0.2"\
  --gpus ${CUDA_VISIBLE_DEVICES} \
  export_model.py \
  --model_name_or_path ${MODEL_TAG} \
  --output_path ${OUTPUT_PATH} \
  --dtype bfloat16 \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --mla_use_matrix_absorption 1 \
  --weight_block_size 128 128


# Static graph inference. Commands for node1 and node2 are identical.
export OUTPUT_PATH=/path/to/exported_model
export QUANT_MODE=a8w8_fp8
export TOTAL_MAX_LENGTH=8192
export MAX_DEC_LEN=4096
export FLAGS_mla_use_tensorcore=1 # Only support Hopper, Amper should be 0
export FLAGS_cascade_attention_max_partition_size=${TOTAL_MAX_LENGTH}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m paddle.distributed.launch --ips "192.168.0.1,192.168.0.2"\
  --gpus ${CUDA_VISIBLE_DEVICES} \
  predictor.py \
  --model_name_or_path ${OUTPUT_PATH} \
  --dtype bfloat16 \
  --mode static \
  --inference_model 1 \
  --append_attn 1 \
  --total_max_length ${TOTAL_MAX_LENGTH} \
  --quant_type ${QUANT_MODE} \
  --max_length ${MAX_DEC_LEN} \
  --mla_use_matrix_absorption 1 \
  --weight_block_size 128 128
```
### deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

Single-GPU WINT8 Inference

```shell
# Dynamic graph inference
python predictor.py --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type weight_only_int8


# Exporting the model with dynamic to static
python export_model.py --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type weight_only_int8


# Static graph inference
python predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1
```

## Benchmark

### vLLM & sglang Service Deployment
1. Install [vLLM main branch](https://docs.vllm.ai/en/latest/getting_started/installation.html) & [sglang v0.4.3.post4](https://docs.sglang.ai/start/install.html)

```shell
export VLLM_COMMIT=1253b1577408f7981d11495b1fda71cbcbe48dc4
git clone https://github.com/vllm-project/vllm.git && cd vllm && git checkout $VLLM_COMMIT
python3 setup.py bdsit_wheel
```
```shell
pip install "sglang[all]>=0.4.3.post4"
```

2. Deploy the service
```shell
VLLM_USE_FLASHINFER_SAMPLER=1 VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=FLASHMLA vllm serve deepseek-ai/DeepSeek-R1 --tensor-parallel-size 16 --trust-remote-code   --max-num-seqs 256 --max-model-len 4096 --max-seq-len-to-capture 256 --enforce-eager --disable-log-requests
```
```shell
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1 --tp 16 --dist-init-addr $IP --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 40000 --enable-torch-compile --torch-compile-max-bs 256 --disable-cuda-graph --quantization fp8 --enable-flashinfer-mla
```

3. Benchmark Testing
```shell
cd llm/benchmark/serving
bash run_benchmark_client.sh vllm
bash run_benchmark_client.sh sglang
```

## Acknowledgement
During the development of this project, we have learned from and benefited from several excellent open-source projects. We would like to express our sincere gratitude to the following projects and their contributors:

- [DeepSeek](https://github.com/deepseek-ai): As a significant contributor to open-source large models, providing high-quality model weights and optimization solutions for the community.
- [sglang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) which first provided DeepSeek model support, offering important references for our optimization implementations.
- And numerous other outstanding open-source projects: including but not limited to [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [FlashAttention](https://github.com/Dao-AILab/flash-attention), etc., which have provided us with valuable insights into hardware optimization.

The open-source spirit has propelled the development of AI technology, and our project likewise benefits from this ecosystem. We extend our gratitude once again to all contributors in the open-source community!
