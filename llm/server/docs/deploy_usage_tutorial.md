
## 目录

- [部署环境准备](#部署环境准备)
  - [基础环境](#基础环境)
  - [准备部署镜像](#准备部署镜像)
  - [准备模型](#准备模型)
  - [创建容器](#创建容器)
- [启动服务](#启动服务)
  - [配置参数](#配置参数)
  - [启动服务](#启动服务)
  - [服务状态查询](#服务状态查询)
- [服务测试](#服务测试)
  - [Python 客户端](#Python-客户端)
  - [HTTP调用](#HTTP调用)
  - [OpenAI 客户端](#OpenAI-客户端)
  - [返回示例](#返回示例)
- [基于dockerfile创建自己的镜像](#基于dockerfile创建自己的镜像)
- [模型配置参数介绍](#模型配置参数介绍)
- [请求参数介绍](#请求参数介绍)

## 部署环境准备

### 基础环境
  该服务化部署工具目前仅支持在 Linux 系统下部署，部署之前请确保系统有正确的 GPU 环境。

  - 安装 docker
    请参考 [Install Docker Engine](https://docs.docker.com/engine/install/) 选择对应的 Linux 平台安装 docker 环境。

  - 安装 NVIDIA Container Toolkit
    请参考 [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit) 了解并安装 NVIDIA Container Toolkit。

    NVIDIA Container Toolkit 安装成功后，参考 [Running a Sample Workload with Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html#running-a-sample-workload-with-docker) 测试 NVIDIA Container Toolkit 是否可以正常使用。

### 准备部署镜像

为了方便部署，我们提供了 cuda12.3 的镜像，可以直接拉取镜像，或者使用我们提供的 `Dockerfile` [构建自定义镜像](#基于dockerfile创建自己的镜像)
```
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:llm-serving-cuda123-cudnn9-v1.2
```

### 准备模型

该部署工具为 PaddleNLP 静态图模型提供了高效的部署方案，模型静态图导出方案请参考：[LLaMA](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/llama.md)、[Qwen](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/qwen.md)、[Mixtral](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/mixtral.md) ...

导出后的模型放在任意文件夹下，以 `/home/workspace/models_dir` 为例

```
cd /home/workspace/models_dir

# 导出的模型目录结构如下所示，理论上无缝支持 PaddleNLP 导出的静态图模型，无需修改模型目录结构
# /opt/output/Serving/models
# ├── config.json                # 模型配置文件
# ├── xxxx.model                 # 词表模型文件
# ├── special_tokens_map.json    # 词表配置文件
# ├── tokenizer_config.json      # 词表配置文件
# ├── rank_mapping.csv           # 多卡模型会有此文件，如为单卡模型，则无此文件（可选，仅在多卡部署模式下需要）
# └── rank_0                     # 保存模型结构和权重文件的目录
#     ├── model.pdiparams
#     └── model.pdmodel
```

### 创建容器

创建容器之前，请检查 Docker 版本和 GPU 环境，确保 Docker 支持 `--gpus all` 参数。

将模型目录挂载到容器中，默认模型挂载地址为 `/models/`，服务启动时可通过 `MODEL_DIR` 环境变量自定义挂载地址。
```
docker run --gpus all \
    --name paddlenlp_serving \
    --privileged \
    --cap-add=SYS_PTRACE \
    --network=host \
    --shm-size=5G \
    -v /home/workspace/models_dir:/models/ \
    -dit registry.baidubce.com/paddlepaddle/fastdeploy:llm-serving-cuda123-cudnn9-v1.2 bash

# 进入容器，检查GPU环境和模型挂载是否正常
docker exec -it paddlenlp_serving /bin/bash
nvidia-smi
ls /models/
```

## 启动服务

### 配置参数

根据需求和硬件信息，配置以下环境变量

```
# 单/多卡推理配置。自行修改。
## 如果是单卡推理，使用0卡，设置如下环境变量。
export MP_NUM=1
export CUDA_VISIBLE_DEVICES=0

## 如果是多卡推理，除了模型导出得满足2卡要求，同时设置如下环境变量。
# export MP_NUM=2
# export CUDA_VISIBLE_DEVICES=0,1

# 如部署场景无流式Token返回需求，可配置如下开关
# 服务将会将每个请求的所有生成Token一次性返回
# 降低服务逐个Token发送压力
# 默认关闭
# export DISABLE_STREAMING=1

# 配置数据服务。需要自行修改HTTP_PORT、GRPC_PORT、METRICS_PORT和INFER_QUEUE_PORT。(请事先检查端口可用)
export HTTP_PORT="8110"                         # 探活服务的http端口（当前仅用于健康检查、探活）
export GRPC_PORT="8811"                         # 模型推服务的grpc端口
export METRICS_PORT="8722"                      # 模型服务中监督指标的端口
export INFER_QUEUE_PORT="8813"                  # 模型服务内部使用的端口
export PUSH_MODE_HTTP_PORT="9965"               # 服务请求HTTP端口号，如不配置，默认为-1，即服务只支持GRPC协议

# MAX_SEQ_LEN: 服务会拒绝input token数量超过MAX_SEQ_LEN的请求，并返回错误提示
# MAX_DEC_LEN: 服务会拒绝请求中max_dec_len/min_dec_len超过此参数的请求，并返回错误提示
export MAX_SEQ_LEN=8192
export MAX_DEC_LEN=1024

export BATCH_SIZE="48"                          # 设置最大Batch Size，模型可同时并发处理的最大输入数量，不能高于128
export BLOCK_BS="5"                             # 缓存Block支持的最大Query Batch Size，如果出现out of memeory 错误，尝试减少该数值
export BLOCK_RATIO="0.75"                       # 一般可以设置成 输入平均Token数/（输入+输出平均Token数)

export MAX_CACHED_TASK_NUM="128"  # 服务缓存队列最大长度，队列达到上限后，会拒绝新的请求，默认128
# 开启HTTP接口配置如下参数
export PUSH_MODE_HTTP_WORKERS="1" # HTTP服务进程数，在 PUSH_MODE_HTTP_PORT 配置的情况下有效，最高设置到8即可，默认为1
```

更多请求参数请参考[模型配置参数介绍](#模型配置参数介绍)

### 启动服务

```
cd /opt/output/Serving
bash start_server.sh

# 重新启动服务前，需要停止服务，在/opt/output/Serving目录下执行 bash stop_server.sh
```

### 服务状态查询

```
# port为上面启动服务时候指定的HTTP_PORT
  > 测试前请确保服务IP和端口正确

live接口： (服务是否能正常接收请求）
  http://127.0.0.1:8110/v2/health/live
health接口：（模型是否准备好推理）
  http://127.0.0.1:8110/v2/health/ready
```

## 服务测试

### HTTP调用

提示：HTTP调用接口使用变量 PUSH_MODE_HTTP_PORT 配置！HTTP_PORT 仅用于探活接口使用！

```
import uuid
import json
import requests

push_mode_http_port = "9965"    # 服务配置的PUSH_MODE_HTTP_PORT
url = f"http://127.0.0.1:{push_mode_http_port}/v1/chat/completions"
req_id = str(uuid.uuid1())
data_single = {
    "text": "Hello, how are you?",
    "req_id": req_id,
    "max_dec_len": 64,
    "stream": True,
  }
# 逐token返回
res = requests.post(url, json=data_single, stream=True)
for line in res.iter_lines():
    print(json.loads(line))

# 多轮对话
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
# 逐token返回
res = requests.post(url, json=data_multi, stream=True)
for line in res.iter_lines():
    print(json.loads(line))
```

更多请求参数请参考[请求参数介绍](#请求参数介绍)

### 返回示例

```
如果stream为True，流式返回
    如果正常，返回{'token': xxx, 'is_end': xxx, 'send_idx': xxx, ..., 'error_msg': '', 'error_code': 0}
    如果异常，返回{'error_msg': xxx, 'error_code': xxx}，error_msg字段不为空，error_code字段不为0

如果stream为False，非流式返回
    如果正常，返回{'tokens_all': xxx, ..., 'error_msg': '', 'error_code': 0}
    如果异常，返回{'error_msg': xxx, 'error_code': xxx}，error_msg字段不为空，error_code字段不为0
```

### OpenAI 客户端

我们提供了 OpenAI 客户端的支持，使用方法如下：

提示：使用 OpenAI 客户端需要配置 `PUSH_MODE_HTTP_PORT`！

```
import openai

push_mode_http_port = "9965"    # 服务配置的PUSH_MODE_HTTP_PORT
client = openai.Client(base_url=f"http://127.0.0.1:{push_mode_http_port}/v1/chat/completions", api_key="EMPTY_API_KEY")

# 非流式返回
response = client.completions.create(
	model="default",
	prompt="Hello, how are you?",
  max_tokens=50,
  stream=False,
)

print(response)
print("\n")

# 流式返回
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
# 非流式返回
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

# 流式返回
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

## 基于dockerfile创建自己的镜像

为了方便用户构建自定义服务，我们提供了基于dockerfile创建自己的镜像的脚本。
```
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/llm/server

docker build --network=host -f ./dockerfiles/Dockerfile_serving_cuda123_cudnn9 -t llm-serving-cu123-self .
```
创建自己的镜像后，可以基于该镜像[创建容器](#创建容器)

## 模型配置参数介绍

| 字段名 | 字段类型 | 说明 | 是否必填 | 默认值 | 备注 |
| :---: | :-----: | :---: | :---: | :-----: | :----: |
| MP_NUM |  int  | 模型并行度 | 否 | 8 | CUDA_VISIBLE_DEVICES 需配置对应卡数 |
| CUDA_VISIBLE_DEVICES | str | 使用 GPU 编号 | 否 | 0,1,2,3,4,5,6,7 |  |
| HTTP_PORT | int | 探活服务的http端口 | 是 | 无 | 当前仅用于健康检查、探活 |
| GRPC_PORT | int | 模型推服务的grpc端口 | 是 | 无 |   |
| METRICS_PORT | int | 模型服务中监督指标的端口 | 是 | 无 |   |
| INFER_QUEUE_PORT | int | 模型服务内部使用的端口 | 否 | 56666 |   |
| PUSH_MODE_HTTP_PORT | int | 服务请求HTTP端口号 | 否 | -1 | 如不配置，服务只支持GRPC协议 |
| DISABLE_STREAMING | int | 是否使用流式返回 | 否 | 0 |  |
| MAX_SEQ_LEN | int | 最大输入序列长度 | 否 | 8192 | 服务会拒绝input token数量超过MAX_SEQ_LEN的请求，并返回错误提示 |
| MAX_DEC_LEN | int | 最大decoer序列长度 | 否 | 1024 | 服务会拒绝请求中max_dec_len/min_dec_len超过此参数的请求，并返回错误提示 |
| BATCH_SIZE | int | 最大Batch Size | 否 | 50 | 模型可同时并发处理的最大输入数量，不能高于128 |
| BLOCK_BS | int | 缓存Block支持的最大Query Batch Size | 否 | 50 | 如果出现out of memeory 错误，尝试减少该数值 |
| BLOCK_RATIO | float |  | 否 | 0.75 | 建议配置 输入平均Token数/（输入+输出平均Token数) |
| MAX_CACHED_TASK_NUM | int | 服务缓存队列最大长度 | 否 | 128 | 队列达到上限后，会拒绝新的请求 |
| PUSH_MODE_HTTP_WORKERS | int | HTTP服务进程数 | 否 | 1 | 在 PUSH_MODE_HTTP_PORT 配置的情况下有效，高并发下提高该数值，建议最高配置为8 |
| USE_WARMUP | int | 是否进行 warmup | 否 | 0 |  |
| USE_HF_TOKENIZER | int | 是否进行使用huggingface的词表 | 否 | 0 |   |
| USE_CACHE_KV_INT8 | int | 是否将INT8配置为KV Cache的类型 | 否 | 0 | c8量化模型需要配置为1 |
| MODEL_DIR | str | 模型文件路径 | 否 | /models/ |  |
| FD_MODEL_CONFIG_PATH | str | 模型config文件路径 | 否 | ${model_dir}/config.json |  |
| DISTRIBUTED_CONFIG | str | 模型分布式配置文件路径 | 否 | ${model_dir}/rank_mapping.csv |  |

## 请求参数介绍

| 字段名 | 字段类型 | 说明 | 是否必填 | 默认值 | 备注 |
| :---: | :-----: | :---: | :---: | :-----: | :----: |
| req_id |  str  | 请求ID，用于标识一个请求。建议设置req_id，保证其唯一性   | 否 | 随机id | 如果推理服务中同时有两个相同req_id的请求，会返回req_id重复的错误信息 |
| text   | str  | 请求的文本 | 否 | 无 | text 和 messages 必须有一个 |
| messages | str | 多轮对话文本 | 否 | 无 | 多轮对话以list方式存储 |
| max_dec_len | int  | 最大生成token的长度，如果请求的文本token长度加上max_dec_len大于模型的max_seq_len，会返回长度超限的错误信息 | 否 | max_seq_len减去文本token长度 |  |
| min_dec_len | int | 最小生成token的长度，最小是1 | 否 | 1 |  |
| topp | float | 控制随机性参数，数值越大则随机性越大，范围是0~1 | 否 | 0.7 |  |
| temperature | float | 控制随机性参数，数值越小随机性越大，需要大于 0 | 否 | 0.95 |  |
| frequency_score | float | 频率分数 | 否 | 0 |  |
| penalty_score | float | 惩罚分数 | 否 | 1 |  |
| presence_score | float | 存在分数 | 否 | 0 |  |
| stream | bool | 是否流式返回 | 否 | False |  |
| timeout | int | 请求等待的超时时间，单位是秒 | 否 | 300 |  |
| return_usage | bool | 是否返回输入、输出 token 数量 | 否 | False |  |

* 在正确配置PUSH_MODE_HTTP_PORT字段下，服务支持 GRPC 和 HTTP 两种请求服务
  * stream 参数仅对 HTTP 请求生效
