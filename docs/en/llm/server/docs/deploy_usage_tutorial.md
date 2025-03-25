# High-Performance Deployment of Static Graphs

*This deployment tool is based on NVIDIA Triton, designed specifically for server-side large model serving. It provides service interfaces supporting gRPC and HTTP protocols, along with streaming token output capabilities. The underlying inference engine supports continuous batching, weight-only int8, post-training quantization (PTQ), and other acceleration strategies, delivering an easy-to-use and high-performance deployment experience.*

## Quick Deployment of Static Graphs

This method only supports [one-click runnable models listed here](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md) for instant inference service startup.

To avoid prolonged download times for large models, we provide an automatic download [script](#static-graph-download) that supports post-download service initiation. After entering the container, perform static graph download based on single/multi-node model scenarios.

`MODEL_PATH` specifies the model storage path (user-definable)
`model_name` specifies the model name to download. Supported models can be found in [this document](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)

Notes:
1. Ensure shm-size >= 5, otherwise service startup may fail
2. Verify model environment and hardware requirements before deployment. Refer to [documentation](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)

**A100 Deployment Example**
```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B/weight_only_int8"}
docker run -i --rm --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 /bin/bash \
-c -ex 'start_server $model_name && tail -f /dev/null'
```
## Deployment Environment Preparation

### Basic Environment
This serving deployment tool currently only supports deployment on Linux systems. Please ensure the system has proper GPU environment before deployment.

- Install Docker
  Refer to [Install Docker Engine](https://docs.docker.com/engine/install/) to install Docker environment for your corresponding Linux platform.

- Install NVIDIA Container Toolkit
  Refer to [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit) to learn and install NVIDIA Container Toolkit.

  After successful installation of NVIDIA Container Toolkit, refer to [Running a Sample Workload with Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html#running-a-sample-workload-with-docker) to verify if NVIDIA Container Toolkit works properly.

### Prepare Deployment Images

For deployment convenience, we provide images for CUDA 12.4 and CUDA 11.8. You can either directly pull the images or use our provided `Dockerfile` to [build custom images](#create-custom-images-using-dockerfile).

| CUDA Version | Supported Hardware Architectures | Image Address | Supported Typical Devices |
|:-------------|:---------------------------------:|:-------------:|:-------------------------:|
| CUDA 11.8    | 70 75 80 86 | ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda118-cudnn8-v2.1 | V100, T4, A100, A30, A10 |
| CUDA 12.4    | 80 86 89 90 | ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 | A100, A30, A10, L20, H20, H100 |

### Prepare Models

The exported models can be placed in any directory, such as `/home/workspace/models_dir`
This deployment tool provides an efficient deployment solution for PaddleNLP static graph models. For model static graph export solutions, please refer to: [DeepSeek](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/deepseek.md), [LLaMA](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/llama.md), [Qwen](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/qwen.md), [Mixtral](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/mixtral.md)...

```
cd /home/workspace/models_dir

# The exported model directory structure is as follows, theoretically seamlessly supporting PaddleNLP exported static graph models without modifying the model directory structure
# /opt/output/Serving/models
# ├── config.json                # Model configuration file
# ├── xxxx.model                 # Vocabulary model file
# ├── special_tokens_map.json    # Vocabulary configuration file
# ├── tokenizer_config.json      # Vocabulary configuration file
# └── rank_0                     # Directory storing model structure and weight files
#     ├── model.pdiparams
#     └── model.pdmodel or model.json # Paddle 3.0 version uses model.json, Paddle 2.x version uses model.pdmodel
```

#### Static Graph Download

In addition to supporting automatic download via setting `model_name` during startup, the service provides scripts for manual download. **During deployment, the environment variable `MODEL_DIR` must be specified as the model download storage path**

Script location: `/opt/output/download_model.py`

```
python download_model.py \
--model_name $model_name \
--dir $MODEL_PATH \
--nnodes 2 \
--mode "master" \
--speculate_model_path $MODEL_PATH
```

**Single-node Model Download**
Taking DeepSeek-R1 weight_only_int4 model as example:
```
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name="deepseek-ai/DeepSeek-R1/weight_only_int4"
python download_model.py --model_name $model_name --dir $MODEL_PATH --nnodes 1
```

**Multi-node Model Download**
Taking DeepSeek-R1 2-node weight_only_int8 model as example:
**node1** Master node
```
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name="deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8"
python download_model.py --model_name $model_name --dir $MODEL_PATH --nnodes 2 --mode "master"
```

**Node 2** (Slave Node)
```
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name="deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8"
python download_model.py --model_name $model_name --dir $MODEL_PATH --nnodes 2 --mode "slave"
```


**Parameter Description**

| Field Name | Type | Description | Mandatory | Default Value |
| :---: | :-----: | :---: | :---: | :-----: |
| model_name | str | Specified model name for download. Supported models can be found in [documentation](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md) | No | deepseek-ai/DeepSeek-R1/weight_only_int4 |
| dir | str | Model storage path | No | downloads |
| nnodes | int | Number of nodes | No | 1 |
| mode | str | Download mode for distinguishing different nodes in multi-machine setup | No | Only supports "master" and "slave" values |
| speculate_model_path | str | Speculative decoding model storage path | No | None |


### Create Container

Before creating the container, please check Docker version and GPU environment to ensure Docker supports `--gpus all` parameter.

Mount the model directory to the container. Default model mount path is `/models/`. The mount path can be customized via `MODEL_DIR` environment variable during service startup.
```
docker run --gpus all \
    --name paddlenlp_serving \
    --privileged \
    --cap-add=SYS_PTRACE \
    --network=host \
    --shm-size=5G \
    -v /home/workspace/models_dir:/models/ \
    -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.1 bash

# Enter container to verify GPU environment and model mount status
docker exec -it paddlenlp_serving /bin/bash
nvidia-smi
ls /models/
```

## Start Service

### Configure Parameters

Set the following environment variables according to requirements and hardware information
```shell
# Single/Multi-GPU Inference Configuration. Modify as needed.
## For single-GPU inference, using GPU 0, set the following environment variables.
export MP_NUM=1
export CUDA_VISIBLE_DEVICES=0

## For multi-GPU inference, besides meeting the 2-GPU requirements for model export, also set the following environment variables.
# export MP_NUM=2
# export CUDA_VISIBLE_DEVICES=0,1

# If the deployment scenario does not require streaming Token returns, configure the following switch
# The service will return all generated Tokens for each request at once
# Reduces pressure on the service to send Tokens incrementally
# Disabled by default
# export DISABLE_STREAMING=1

# Data service configuration. Modify HTTP_PORT, GRPC_PORT, METRICS_PORT and INFER_QUEUE_PORT as needed. (Verify port availability first)
export HEALTH_HTTP_PORT="8110"                         # Health check service http port (currently only used for health checks)
export SERVICE_GRPC_PORT="8811"                         # Model serving grpc port
export METRICS_HTTP_PORT="8722"                      # Monitoring metrics port for model service
export INTER_PROC_PORT="8813"                  # Internal communication port for model service
export SERVICE_HTTP_PORT="9965"               # HTTP port for service requests. Defaults to -1 (only GRPC supported) if not configured

# MAX_SEQ_LEN: The service will reject requests where input token count exceeds MAX_SEQ_LEN and return error
# MAX_DEC_LEN: The service will reject requests with max_dec_len/min_dec_len exceeding this parameter and return error
export MAX_SEQ_LEN=8192
export MAX_DEC_LEN=1024

export BATCH_SIZE="48"                          # Set maximum Batch Size - maximum concurrent requests the model can handle, should not exceed 128
export BLOCK_BS="5"                             # Maximum Query Batch Size for cached Blocks. Reduce this value if encountering out of memory errors
export BLOCK_RATIO="0.75"                       # Generally can be set to (average input tokens)/(average input + output tokens)

export MAX_CACHED_TASK_NUM="128"  # Maximum length of service cache queue. New requests will be rejected when queue reaches limit, default 128
# To enable HTTP interface, configure the following parameters
export PUSH_MODE_HTTP_WORKERS="1" # Number of HTTP service processes. Effective when PUSH_MODE_HTTP_PORT is configured, can be set up to 8, default 1
```

#### Multi-Machine Parameter Configuration
Additional service parameters compared to single-machine deployment
```
export POD_IPS=10.0.0.1,10.0.0.2
export POD_0_IP=10.0.0.1
export MP_NNODE=2 # Number of nodes set to 2 indicates 2-machine service
export MP_NUM=16 # Model sharding set to 16
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```
For more request parameters, please refer to [Model Configuration Parameters](#模型配置参数介绍)

### Service Startup
We provide two deployment solutions for model serving:
- Deploy with pre-saved models at specified path
- Auto-download static graph deployment

#### Single-machine Startup
Deploy with pre-saved models at specified path

```shell
export MODEL_DIR=${MODEL_DIR:-"/models"}
start_server

# Before restarting the service, stop it using stop_server
```
Auto-download static graph deployment
`model_name` specifies the model to download. For supported models, see [documentation](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)

```shell
model_name="deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8"
start_server $model_name

# Before restarting the service, stop it using stop_server
```

#### Multi-machine Startup
##### Sequential Startup
1. Start master node service
2. Start services on other nodes sequentially

Start command is same as single-machine

##### MPI Startup
MPI startup requires prior SSH configuration between machines

```
mpirun start_server

# Stop service
mpirun stop_server
```

### Service Health Check

```
# port should be the HEALTH_HTTP_PORT specified during service startup
  > Please ensure correct service IP and port before testing

Liveness probe: (Check if service can accept requests)
  http://127.0.0.1:8110/v2/health/live
Readiness probe: (Check if model is ready for inference)
  http://127.0.0.1:8110/v2/health/ready
```

## Service Testing
For multi-machine testing, execute on master node or replace IP with master node's IP

### HTTP Invocation
```python
import uuid
import json
import requests

ip = 127.0.0.1
service_http_port = "9965"    # Service configuration
url = f"http://{ip}:{service_http_port}/v1/chat/completions"

req_id = str(uuid.uuid1())
data_single = {
    "text": "Hello, how are you?",
    "req_id": req_id,
    "max_dec_len": 64,
    "stream": True,
}

# Stream per token
res = requests.post(url, json=data_single, stream=True)
for line in res.iter_lines():
    print(json.loads(line))

# Multi-turn dialogue
data_multi = {
    "messages": [
        {"role": "user", "content": "Hello, who are you"},
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    "req_id": req_id,
    "max_dec_len": 64,
    "stream": True,
}

# Stream per token
res = requests.post(url, json=data_multi, stream=True)
for line in res.iter_lines():
    print(json.loads(line))
```

For more request parameters, please refer to [Request Parameters](#请求参数介绍)

### Response Examples

```python
When stream is True, streaming returns:
    If normal, returns {'token': xxx, 'is_end': xxx, 'send_idx': xxx, ..., 'error_msg': '', 'error_code': 0}
    If error occurs, returns {'error_msg': xxx, 'error_code': xxx} with error_msg not empty and error_code non-zero

When stream is False, non-streaming returns:
    If normal, returns {'tokens_all': xxx, ..., 'error_msg': '', 'error_code': 0}
    If error occurs, returns {'error_msg': xxx, 'error_code': xxx} with error_msg not empty and error_code non-zero
```

### OpenAI Client

We provide support for the OpenAI client. Usage is as follows:
```python
import openai

ip = 127.0.0.1
service_http_port = "9965"    # Service configuration

client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1/chat/completions", api_key="EMPTY_API_KEY")

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

# Chat completion
# Non-streaming response
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Hello, who are you"},
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
    stream=False,
)

print(response)
print("\n")

# Streaming response
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Hello, who are you"},
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
    stream=True,
)

for chunk in response:
  if chunk.choices[0].delta is not None:
    print(chunk.choices[0].delta.content, end='')
print("\n")
```
## Creating Your Own Image Based on Dockerfile

To facilitate users in building custom services, we provide scripts for creating your own images based on dockerfile.

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/llm/server

docker build --network=host -f ./dockerfiles/Dockerfile_serving_cuda124_cudnn9 -t llm-serving-cu124-self .
```
After creating your own image, you can [create a container](#创建容器) based on it.

## Model Configuration Parameters

| Field Name | Type | Description | Required | Default Value | Remarks |
| :---: | :-----: | :---: | :---: | :-----: | :----: |
| MP_NNODE | int | Number of nodes | No | 1 | Should match the number of machines |
| MP_NUM | int | Model parallelism degree | No | 8 | CUDA_VISIBLE_DEVICES must be configured with corresponding GPU count |
| CUDA_VISIBLE_DEVICES | str | GPU indices | No | 0,1,2,3,4,5,6,7 |  |
| POD_IPS | str | IP addresses of multi-node cluster | No | None | Required for multi-node, example: "10.0.0.1,10.0.0.2" |
| POD_0_IP | str | Master node IP in multi-node cluster | No | None | Required for multi-node, example: "10.0.0.1" (must exist in POD_IPS) |
| HEALTH_HTTP_PORT | int | Health check service HTTP port | Yes | None | Currently only used for health checks (pre 3.0.0 images use HTTP_PORT) |
| SERVICE_GRPC_PORT | int | Model serving GRPC port | Yes | None | (pre 3.0.0 images use GRPC_PORT) |
| METRICS_HTTP_PORT | int | Monitoring metrics HTTP port | Yes | None | (pre 3.0.0 images use METRICS_PORT) |
| INTER_PROC_PORT | int | Internal process communication port | No | 56666 | (pre 3.0.0 images use INTER_QUEUE_PORT) |
| SERVICE_HTTP_PORT | int | HTTP port for service requests | No | 9965 | (pre 3.0.0 images use PUSH_MODE_HTTP_PORT) |
| DISABLE_STREAMING | int | Disable streaming response | No | 0 |  |
| MAX_SEQ_LEN | int | Maximum input sequence length | No | 8192 | Requests exceeding this limit will be rejected with error |
| MAX_DEC_LEN | int | Maximum decoder sequence length | No | 1024 | Requests with max_dec_len/min_dec_len exceeding this will be rejected |
| BATCH_SIZE | int | Maximum batch size | No | 50 | Maximum concurrent inputs the model can handle, cannot exceed 128 |
| BLOCK_BS | int | Maximum query batch size for cached blocks | No | 50 | Reduce this value if encountering out of memory errors |
| BLOCK_RATIO | float |  | No | 0.75 | Recommended to set as input length ratio |
| MAX_CACHED_TASK_NUM | int | Maximum cached tasks in queue | No | 128 | New requests will be rejected when queue reaches limit |
| PUSH_MODE_HTTP_WORKERS | int | Number of HTTP service workers | No | 1 | Effective when configured, increase for high concurrency (max recommended 8) |
| USE_WARMUP | int | Enable warmup | No | 0 |  |
| USE_HF_TOKENIZER | int | Use HuggingFace tokenizer | No | 0 |  |
| USE_CACHE_KV_INT8 | int | Enable INT8 for KV Cache | No | 0 | Set to 1 for c8 quantized models |
| MODEL_DIR | str | Model file path | No | /models/ |  |
| model_name | str | Model name | No | None | Used for static model downloads (refer to [#./static_models.md](#./static_models.md)) |
| OUTPUT_LOG_TO_CONSOLE | str | Redirect logs to console | No | 0 |  |

## GPU Memory Configuration Recommendations

* BLOCK_BS: Determines the number of cacheKV blocks. Total supported tokens = BLOCK_BS * INFER_MODEL_MAX_SEQ_LEN
  * Example: For 8K model with BLOCK_BS=40, supports 40*8=320K tokens
  * For 32K model, recommended BLOCK_BS=40/(32K/8K)=10 (consider input length variations, may reduce to 8-9)
* BLOCK_RATIO: Block allocation ratio between encoder/decoder. Recommended value: (avg_input_len+128)/(avg_input_len + avg_output_len)*EXTEND_RATIO (EXTEND_RATIO=1~1.3)
  * Example: avg_input=300, avg_output=1500 → (300+128)/(300+1500)≈0.25
* BATCH_SIZE: Generally < TOTAL_TOKENS / (avg_input_len + avg_output_len)

| GPU Memory | Deployed Model | Static Graph Weights | Nodes | Quantization Type | Context Length | MTP Enabled | MTP Quant Type | Recommended BLOCK_BS |
|------------|-----------------|----------------------|-------|-------------------|----------------|-------------|----------------|----------------|
| 80GB       | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1/a8w8_fp8_wint4 | 1 | a8w8_fp8_wint4 | 8K | No | - | 40 |
| 80GB       | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1-MTP/a8w8_fp8_wint4 | 1 | a8w8_fp8_wint4 | 8K | Yes | a8w8_fp8 | 36 |
| 80GB       | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1/weight_only_int4 | 1 | weight_only_int4 | 8K | No | - | 40 |
| 80GB       | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1-MTP/weight_only_int4 | 1 | weight_only_int4 | 8K | Yes | weight_only_int8 | 36 |
| 80GB       | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1-2nodes/a8w8_fp8 | 2 | a8w8_fp8 | 8K | No | - | 50 |
| 80GB       | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1-MTP-2nodes/a8w8_fp8 | 2 | a8w8_fp8 | 8K | Yes | a8w8_fp8 | 36 |
| 80GB       | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8 | 2 | weight_only_int8 | 8K | No | - | 40 |
| 80GB       | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1-MTP-2nodes/weight_only_int8 | 2 | weight_only_int8 | 8K | Yes | weight_only_int8 | 36 |

## Request Parameters

| Field Name | Type | Description | Required | Default Value | Remarks |
| :---: | :-----: | :---: | :---: | :-----: | :----: |
| req_id | str | Request ID (unique identifier) | No | Random ID | Duplicate req_id will return error |
| text | str | Input text | No | None | Either text or messages must be provided |
| messages | str | Multi-turn conversation context | No | None | Stored as list |
| max_dec_len | int | Maximum generated tokens | No | max_seq_len - input_tokens | Requests exceeding limit will return error |
| min_dec_len | int | Minimum generated tokens | No | 1 |  |
| topp | float | Top-p sampling (0-1) | No | 0.7 | Higher values increase randomness |
| temperature | float | Temperature (must >0) | No | 0.95 | Lower values reduce randomness |
| frequency_score | float | Frequency penalty | No | 0 |  |
| penalty_score | float | Repetition penalty | No | 1 |  |
| presence_score | float | Presence penalty | No | 0 |  |
| stream | bool | Stream response | No | False |  |
| timeout | int | Request timeout (seconds) | No | 300 |  |
| return_usage | bool | Return input/output token counts | No | False |  |

* Service supports both GRPC and HTTP requests
  * stream parameter only applies to HTTP requests
