# 静态图高性能部署全流程


*该部署工具是基于英伟达 Triton 框架专为服务器场景的大模型服务化部署而设计。它提供了支持 gRPC、HTTP 协议的服务接口，以及流式 Token 输出能力。底层推理引擎支持连续批处理、weight only int8、后训练量化（PTQ）等加速优化策略，为用户带来易用且高性能的部署体验。*

## 静态图快速部署

该方法仅支持[可一键跑通的模型列表](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)中的模型进行一键启动推理服务。
  
为了避免模型过大导致的下载时间过长问题，我们直接提供了自动下载的[脚本](#静态图下载)，支持下载后再启动服务进行推理。进入容器后根据单机或多机模型进行静态图下载。

`MODEL_PATH` 为指定模型下载的存储路径，可自行指定
`model_name` 为指定下载模型名称，具体支持模型可查看[文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)

Note:
1. 请保证 shm-size >= 5，不然可能会导致服务启动失败
2. 部署前请确认模型所需要的环境和硬件，请参考[文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)

**A100部署示例**
```shell
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name=${model_name:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B/weight_only_int8"}
docker run  -i --rm  --gpus all --shm-size 32G --network=host --privileged --cap-add=SYS_PTRACE \
-v $MODEL_PATH:/models -e "model_name=${model_name}" \
-dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.3 /bin/bash \
-c -ex 'start_server $model_name && tail -f /dev/null'
```


## 部署环境准备

### 基础环境
  该服务化部署工具目前仅支持在 Linux 系统下部署，部署之前请确保系统有正确的 GPU 环境。

  - 安装 docker
    请参考 [Install Docker Engine](https://docs.docker.com/engine/install/) 选择对应的 Linux 平台安装 docker 环境。

  - 安装 NVIDIA Container Toolkit
    请参考 [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit) 了解并安装 NVIDIA Container Toolkit。

    NVIDIA Container Toolkit 安装成功后，参考 [Running a Sample Workload with Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html#running-a-sample-workload-with-docker) 测试 NVIDIA Container Toolkit 是否可以正常使用。

### 准备部署镜像

为了方便部署，我们提供了 cuda12.4 与 cuda 11.8 的镜像，可以直接拉取镜像，或者使用我们提供的 `Dockerfile` [构建自定义镜像](#基于-dockerfile-创建自己的镜像)


|cuda版本| 支持硬件架构|镜像地址|支持的典型设备|
|:------|:-:|:-:|:-:|
| cuda11.8 | 70 75 80 86 |ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda118-cudnn8-v2.3 |V100，T4，A100，A30，A10 |
| cuda12.4 | 80 86 89 90 |ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.3 |A100，A30，A10,L20，H20，H100 |


### 准备模型

导出后的模型放在任意文件夹下，以 `/home/workspace/models_dir` 为例
该部署工具为 PaddleNLP 静态图模型提供了高效的部署方案，模型静态图导出方案请参考：[DeepSeek](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/deepseek.md)、[LLaMA](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/llama.md)、[Qwen](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/qwen.md)、[Mixtral](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/mixtral.md) ...

```
cd /home/workspace/models_dir

# 导出的模型目录结构如下所示，理论上无缝支持 PaddleNLP 导出的静态图模型，无需修改模型目录结构
# /opt/output/Serving/models
# ├── config.json                # 模型配置文件
# ├── xxxx.model                 # 词表模型文件
# ├── special_tokens_map.json    # 词表配置文件
# ├── tokenizer_config.json      # 词表配置文件
# └── rank_0                     # 保存模型结构和权重文件的目录
#     ├── model.pdiparams
#     └── model.pdmodel 或者 model.json # Paddle 3.0 版本模型为model.json，Paddle 2.x 版本模型为model.pdmodel
```

#### 静态图下载

除了支持通过设置`model_name` 在启动时进行自动下载，服务提供脚本可以进行自行下载。**部署时需指定环境变量`MODEL_DIR` 为模型下载存储路径**

脚本所在路径`/opt/output/download_model.py` (当前路径为镜像内路径)

**注**：模型下载依赖当前镜像的版本号，若在镜像内启动下载脚本无需配置，若直接从代码仓库拉取需配置环境变量`tag`，当前默认下载的版本为3.0.0.b4

```
python download_model.py \
--model_name $model_name \
--dir $MODEL_PATH \
--nnodes 2 \
--mode "master" \
--speculate_model_path $MODEL_PATH 
```

**单机模型下载**
以DeepSeek-R1 weight_only_int4 模型为例
```
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name="deepseek-ai/DeepSeek-R1/weight_only_int4"
python download_model.py --model_name $model_name --dir $MODEL_PATH --nnodes 1
```
**多机模型下载**
以DeepSeek-R1 2机 weight_only_int8 模型为例
**node1** 主节点
```
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name="deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8"
python download_model.py --model_name $model_name --dir $MODEL_PATH --nnodes 2 --mode "master"
```
**node2** 副节点
```
export MODEL_PATH=${MODEL_PATH:-$PWD}
export model_name="deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8"
python download_model.py --model_name $model_name --dir $MODEL_PATH --nnodes 2 --mode "slave"
```


**参数说明**

| 字段名 | 字段类型 | 说明 | 是否必填 | 默认值 |
| :---: | :-----: | :---: | :---: | :-----: |
| model_name | str | 为指定下载模型名称，具体支持模型可查看[文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md) | 否 | deepseek-ai/DeepSeek-R1/weight_only_int4 |
| dir | str | 模型存储地址 | 否 | downloads |
| nnodes | int | 节点个数 | 否 | 1 |
| mode | str | 下载模式用于区分多机的不同节点 | 否 | 仅支持 master 和 slave 两个值 |
| speculate_model_path | str | 投机解码模型存储路径 | 否 | None |


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
    -dit ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlenlp:llm-serving-cuda124-cudnn9-v2.3 bash

# 进入容器，检查GPU环境和模型挂载是否正常
docker exec -it paddlenlp_serving /bin/bash
nvidia-smi
ls /models/
```

## 启动服务

### 配置参数

根据需求和硬件信息，配置以下环境变量

```shell
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
export HEALTH_HTTP_PORT="8110"                         # 探活服务的http端口（当前仅用于健康检查、探活）
export SERVICE_GRPC_PORT="8811"                         # 模型推服务的grpc端口
export METRICS_HTTP_PORT="8722"                      # 模型服务中监督指标的端口
export INTER_PROC_PORT="8813"                  # 模型服务内部使用的端口
export SERVICE_HTTP_PORT="9965"               # 服务请求HTTP端口号，如不配置，默认为-1，即服务只支持GRPC协议

# MAX_SEQ_LEN: 服务会拒绝input token数量超过MAX_SEQ_LEN的请求，并返回错误提示
# MAX_DEC_LEN: 服务会拒绝请求中max_dec_len/min_dec_len超过此参数的请求，并返回错误提示
export MAX_SEQ_LEN=8192
export MAX_DEC_LEN=1024

export BATCH_SIZE="48"                          # 设置最大Batch Size，模型可同时并发处理的最大输入数量，不能高于128
export BLOCK_BS="5"                             # 缓存Block支持的最大Query Batch Size，如果出现out of memory 错误，尝试减少该数值
export BLOCK_RATIO="0.75"                       # 一般可以设置成 输入平均Token数/（输入+输出平均Token数)

export MAX_CACHED_TASK_NUM="128"  # 服务缓存队列最大长度，队列达到上限后，会拒绝新的请求，默认128
# 开启HTTP接口配置如下参数
export PUSH_MODE_HTTP_WORKERS="1" # HTTP服务进程数，在 PUSH_MODE_HTTP_PORT 配置的情况下有效，最高设置到8即可，默认为1
```


#### 多机参数配置
相比单机部署新增服务参数

```
export POD_IPS=10.0.0.1,10.0.0.2
export POD_0_IP=10.0.0.1
export MP_NNODE=2 # 节点数为2表明为2机服务
export MP_NUM=16 # 模型分片为16
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```
更多请求参数请参考[模型配置参数介绍](#模型配置参数介绍)

### 启动服务
针对模型部署我们提供两种方案：
- 模型已保存在指定路径下进行部署
- 静态图自动下载部署

#### 单机启动
模型已保存在指定路径下进行部署

```shell
export MODEL_DIR=${MODEL_DIR:-"/models"}
start_server

# 重新启动服务前，需要停止服务，执行 stop_server
```
静态图自动下载部署  
`model_name` 为指定下载模型名称，具体支持模型可查看[文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md)

```shell
model_name="deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8"
start_server $model_name

# 重新启动服务前，需要停止服务，执行 stop_server
```
#### 多机启动
##### 依次启动服务
1. 启动 master node 主节点服务
2. 依次启动其他节点的服务

启动命令与单机相同

##### mpi启动
若使用mpi 进行启动需提前配置各机器的ssh 可以正常访问

```
mpirun start_server

# 停止服务
mpirun stop_server
```


### 服务状态查询

```
# port为上面启动服务时候指定的HEALTH_HTTP_PORT
  > 测试前请确保服务IP和端口正确

live接口： (服务是否能正常接收请求）
  http://127.0.0.1:8110/v2/health/live
health接口：（模型是否准备好推理）
  http://127.0.0.1:8110/v2/health/ready
```

## 服务测试
多机测试时需在主节点执行或者将ip 修改为主节点ip

### HTTP 调用


```python
import uuid
import json
import requests
ip = 127.0.0.1
service_http_port = "9965"    # 服务配置的
url = f"http://{ip}:{service_http_port}/v1/chat/completions"
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

```python
如果stream为True，流式返回
    如果正常，返回{'token': xxx, 'is_end': xxx, 'send_idx': xxx, ..., 'error_msg': '', 'error_code': 0}
    如果异常，返回{'error_msg': xxx, 'error_code': xxx}，error_msg字段不为空，error_code字段不为0

如果stream为False，非流式返回
    如果正常，返回{'tokens_all': xxx, ..., 'error_msg': '', 'error_code': 0}
    如果异常，返回{'error_msg': xxx, 'error_code': xxx}，error_msg字段不为空，error_code字段不为0
```

### OpenAI 客户端

我们提供了 OpenAI 客户端的支持，使用方法如下：


```python
import openai

ip = 127.0.0.1
service_http_port = "9965"    # 服务配置的

client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1/chat/completions", api_key="EMPTY_API_KEY")

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

## 基于 dockerfile 创建自己的镜像

为了方便用户构建自定义服务，我们提供了基于 dockerfile 创建自己的镜像的脚本。
```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/llm/server

docker build --network=host -f ./dockerfiles/Dockerfile_serving_cuda124_cudnn9 -t llm-serving-cu124-self .
```
创建自己的镜像后，可以基于该镜像[创建容器](#创建容器)

## 模型配置参数介绍

| 字段名 | 字段类型 | 说明 | 是否必填 | 默认值 | 备注 |
| :---: | :-----: | :---: | :---: | :-----: | :----: |
| MP_NNODE |  int  | 节点个数 | 否 | 1 | 与机器个数相同| 
| MP_NUM |  int  | 模型并行度 | 否 | 8 | CUDA_VISIBLE_DEVICES 需配置对应卡数 |
| CUDA_VISIBLE_DEVICES | str | 使用 GPU 编号 | 否 | 0,1,2,3,4,5,6,7 |  |
| POD_IPS | str | 多机每个结点的ip | 否 | 无 | 多机时必填，示例: "10.0.0.1,10.0.0.2" |
| POD_0_IP | str | 多机中主节点IP | 否 | 无 | 多机时必填，示例: "10.0.0.1" 该IP 必须存在在POD_IPS 中|
| HEALTH_HTTP_PORT | int | 探活服务的 http 端口 | 是 | 无 | 当前仅用于健康检查、探活 (3.0.0版本前镜像使用HTTP_PORT)|
| SERVICE_GRPC_PORT | int | 模型推服务的 grpc 端口 | 是 | 无 |   (3.0.0版本前镜像使用GRPC_PORT)| |
| METRICS_HTTP_PORT | int | 模型服务中监督指标的端口 | 是 | 无 |  (3.0.0版本前镜像使用METRICS_PORT) |
| INTER_PROC_PORT | int | 模型服务内部使用的端口 | 否 | 56666 |  (3.0.0版本前镜像使用INTER_QUEUE_PORT)  |
| SERVICE_HTTP_PORT | int | 服务请求 HTTP 端口号 | 否 | 9965 | (3.0.0版本前镜像使用PUSH_MODE_HTTP_PORT)  |
| DISABLE_STREAMING | int | 是否使用流式返回 | 否 | 0 |  |
| MAX_SEQ_LEN | int | 最大输入序列长度 | 否 | 8192 | 服务会拒绝 input token 数量超过 MAX_SEQ_LEN 的请求，并返回错误提示 |
| MAX_DEC_LEN | int | 最大 decoder 序列长度 | 否 | 1024 | 服务会拒绝请求中 max_dec_len/min_dec_len 超过此参数的请求，并返回错误提示 |
| BATCH_SIZE | int | 最大 Batch Size | 否 | 50 | 模型可同时并发处理的最大输入数量，不能高于128 |
| BLOCK_BS | int | 缓存 Block 支持的最大 Query Batch Size | 否 | 50 | 如果出现 out of memory 错误，尝试减少该数值 |
| BLOCK_RATIO | float |  | 否 | 0.75 | 建议设为输入长度占总长度的比例 |
| MAX_CACHED_TASK_NUM | int | 服务缓存队列最大长度 | 否 | 128 | 队列达到上限后，会拒绝新的请求 |
| PUSH_MODE_HTTP_WORKERS | int | HTTP 服务进程数 | 否 | 1 | 在  配置的情况下有效，高并发下提高该数值，建议最高配置为8 |
| USE_WARMUP | int | 是否进行 warmup | 否 | 0 |  |
| USE_HF_TOKENIZER | int | 是否进行使用 huggingface 的词表 | 否 | 0 |   |
| USE_CACHE_KV_INT8 | int | 是否将 INT8配置为 KV Cache 的类型 | 否 | 0 | c8量化模型需要配置为1 |
| MODEL_DIR | str | 模型文件路径 | 否 | /models/ |  |
| model_name | str | 模型名称 | 否 | 无 |  用于支持模型静态图下载，具体名称参考文档(#./static_models.md)|
| OUTPUT_LOG_TO_CONSOLE | str | 是否定向输出到console 文件中 | 否 | 0 |  |


## 显存相关参数推荐

* BLOCK_BS：设定 cacheKV block 的数量，服务支持的输入、输出的总 token 数为 NUM_TOTAL_TOKEN = BLOCK_BS * INFER_MODEL_MAX_SEQ_LEN
  * 例如 8K 的模型 BLOCK_BS 设为 40，则共可以支持 40*8 = 320K 的 token 长度
  * 例如 8K 的模型推荐配置 BLOCK_BS设为 40，运行 32K 的模型，支持的总 token 数不变，则 BLOCK_BS 设为 40 / (32K/8K) = 10，考虑到输入长度变化，影响激活峰值显存，可适当下调到 8 或 9
* BLOCK_RATIO: block 分配给 encoder 和 decoder 的比例，建议设为 (平均输入长度+128) /(平均输入长度 + 平均输出长度) * EXTEND_RATIO，其中 EXTEND_RATIO 是超参，在大部分情况中，设为1即可，如输入长度方差较大，可根据具体情况设为 1.1~1.3
  * 例如一份平均输入 300，平均输出 1500，则 block_ratio 推荐值为 (300 + 128) / (300 + 1500) ≈ 0.25
* BATCH_SIZE: 一般小于 NUM_TOTAL_TOKEN / (平均输入长度 + 平均输出长度)

| GPU 显存 | 部署模型         | 静态图权重                                       | 节点数 | 量化类型           | 导出长度 | 是否 MTP | MTP 量化类型      | 推荐 block_bs |
|----------|--------------------|------------------------------------------------|------|----------------|--------|--------|---------------|---------|
| 80GB     | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1/a8w8_fp8_wint4        | 1    | a8w8_fp8_wint4 | 8K     | 否     | -             | 40      |
| 80GB     | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1-MTP/a8w8_fp8_wint4    | 1    | a8w8_fp8_wint4 | 8K     | 是     | a8w8_fp8      | 36      |
| 80GB     | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1/weight_only_int4      | 1    | weight_only_int4 | 8K   | 否     | -             | 40      |
| 80GB     | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1-MTP/weight_only_int4  | 1    | weight_only_int4 | 8K   | 是     | weight_only_int8 | 36      |
| 80GB     | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1-2nodes/a8w8_fp8      | 2    | a8w8_fp8       | 8K     | 否     | -             | 50      |
| 80GB     | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1-MTP-2nodes/a8w8_fp8  | 2    | a8w8_fp8       | 8K     | 是     | a8w8_fp8      | 36      |
| 80GB     | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8 | 2  | weight_only_int8 | 8K   | 否     | -             | 40      |
| 80GB     | DeepSeek-V3/R1 | deepseek-ai/DeepSeek-R1-MTP-2nodes/weight_only_int8 | 2  | weight_only_int8 | 8K | 是     | weight_only_int8 | 36      |


## 请求参数介绍

| 字段名 | 字段类型 | 说明 | 是否必填 | 默认值 | 备注 |
| :---: | :-----: | :---: | :---: | :-----: | :----: |
| req_id |  str  | 请求 ID，用于标识一个请求。建议设置 req_id，保证其唯一性   | 否 | 随机 id | 如果推理服务中同时有两个相同 req_id 的请求，会返回 req_id 重复的错误信息 |
| text   | str  | 请求的文本 | 否 | 无 | text 和 messages 必须有一个 |
| messages | str | 多轮对话文本 | 否 | 无 | 多轮对话以 list 方式存储 |
| max_dec_len | int  | 最大生成 token 的长度，如果请求的文本 token 长度加上 max_dec_len 大于模型的 max_seq_len，会返回长度超限的错误信息 | 否 | max_seq_len 减去文本 token 长度 |  |
| min_dec_len | int | 最小生成 token 的长度，最小是1 | 否 | 1 |  |
| topp | float | 控制随机性参数，数值越大则随机性越大，范围是0~1 | 否 | 0.7 |  |
| temperature | float | 控制随机性参数，数值越小随机性越大，需要大于 0 | 否 | 0.95 |  |
| frequency_score | float | 频率分数 | 否 | 0 |  |
| penalty_score | float | 惩罚分数 | 否 | 1 |  |
| presence_score | float | 存在分数 | 否 | 0 |  |
| stream | bool | 是否流式返回 | 否 | False |  |
| timeout | int | 请求等待的超时时间，单位是秒 | 否 | 300 |  |
| return_usage | bool | 是否返回输入、输出 token 数量 | 否 | False |  |

* 服务支持 GRPC 和 HTTP 两种请求服务
  * stream 参数仅对 HTTP 请求生效
