# 基于Triton Inference Server的服务化部署

本文档将介绍如何使用[Triton Inference Server](https://github.com/triton-inference-server/server)工具部署ERNIE 3.0新闻分类和序列标注模型的pipeline在线服务。

## 目录
- [环境准备](#环境准备)
- [模型转换](#模型转换)
- [部署模型](#部署模型)
- [客户端请求](#客户端请求)

## 环境准备
需要[准备PaddleNLP的运行环境](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)和Triton Server的运行环境。

### 安装Triton Server
下载Triton Server镜像，并启动
```
# 拉取镜像
docker pull nvcr.io/nvidia/tritonserver:21.10-py3

# 启动容器
docker run  -it --net=host --name triton_server -v /path/triton/models:/models nvcr.io/nvidia/tritonserver:21.10-py3 bash
```
Triton版本号`21.10`可以根据自己的需求调整，各个Triton版本对应的Driver、CUDA、TRT和ONNX Runtime等后端版本可以参考[官网文档](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).注意其中的`NVIDIA Driver`行，如果NVIDIA Driver低于文档中要求，在启动运行时会报错

### 进入容器并准备PaddleNLP环境
整个服务的前后处理依赖PaddleNLP，需要在容器内安装相关python包
```
# 进入容器
docker exec -it triton_server bash

# 安装PaddleNLP
python3 -m pip install paddlenlp
```

### 安装FasterTokenizer文本处理加速库（可选）
如果部署环境是Linux，推荐安装faster_tokenizer可以得到更极致的文本处理效率，进一步提升服务性能。目前暂不支持Windows设备安装，将会在下个版本支持。
```
# 注意：在容器内安装
python3 -m pip install faster_tokenizer
```


## 模型获取和转换

使用Triton做服务化部署时，选择ONNX Runtime后端运行需要先将模型转换成ONN格式。

下载ERNIE 3.0的新闻分类模型、序列标注模型(如果有已训练好的模型，跳过此步骤):
```bash
# 下载并解压新闻分类模型
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_3.0/tnews_pruned_infer_model.zip
unzip tnews_pruned_infer_model.zip

# 下载并解压序列标注模型
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_3.0/msra_ner_pruned_infer_model.zip
unzip msra_ner_pruned_infer_model.zip
```

使用Paddle2ONNX将Paddle静态图模型转换为ONNX模型格式的命令如下，以下命令成功运行后，将会在当前目录下生成model.onnx模型文件。
```bash
# 模型地址根据实际填写即可
# 转换新闻分类模型
paddle2onnx --model_dir tnews_pruned_infer_model --model_filename float32.pdmodel --params_filename float32.pdiparams --save_file model.onnx --opset_version 13 --enable_onnx_checker True --enable_dev_version True

# 将转换好的ONNX模型移动到分类任务的模型仓库目录
mv model.onnx /models/ernie_seqcls_model/1

# 转换序列标注模型
paddle2onnx --model_dir  --model_filename msra_ner_pruned_infer_model float32.pdmodel --params_filename float32.pdiparams --save_file model.onnx --opset_version 13 --enable_onnx_checker True --enable_dev_version True

# 将转换好的ONNX模型移动到序列标注任务的模型仓库目录
mv model.onnx /models/ernie_tokencls_model/1
```
Paddle2ONNX的命令行参数说明请查阅：[Paddle2ONNX命令行参数说明](https://github.com/PaddlePaddle/Paddle2ONNX#%E5%8F%82%E6%95%B0%E9%80%89%E9%A1%B9)

模型下载转换好之后，分类任务的models目录结构如下:
```
models
├── ernie_seqcls                      # 分类任务的pipeline
│   ├── 1
│   └── config.pbtxt                  # 通过这个文件组合前后处理和模型推理
├── ernie_seqcls_model                # 分类任务的模型推理
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
├── ernie_seqcls_postprocess          # 分类任务后处理
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
└── ernie_tokenizer                   # 预处理分词
    ├── 1
    │   └── model.py
    └── config.pbtxt
```

## 部署模型
triton目录包含启动pipeline服务的配置和发送预测请求的代码，包括：

```
models                    # Triton启动需要的模型仓库，包含模型和服务配置文件
seq_cls_rpc_client.py     # 新闻分类任务发送pipeline预测请求的脚本
token_cls_rpc_client.py   # 序列标注任务发送pipeline预测请求的脚本
```

*注意*:启动服务时，Triton Server的每个python后端进程默认申请`64M`内存，默认启动的docker无法启动多个python后端节点。有两个解决方案：
- 1.启动容器时设置`shm-size`参数, 比如:`docker run  -it --net=host --name triton_server --shm-size="1g" -v /path/triton/models:/models nvcr.io/nvidia/tritonserver:21.10-py3 bash`
- 2.启动服务时设置python后端的`shm-default-byte-size`参数, 设置python后端的默认内存为10M： `tritonserver --model-repository=/models --backend-config=python,shm-default-byte-size=10485760`

### 分类任务
在容器内执行下面命令启动服务:
```
# 默认启动models下所有模型
tritonserver --model-repository=/models

# 可通过参数只启动分类任务
tritonserver --model-repository=/models --model-control-mode=explicit --load-model=ernie_seqcls
```
输出打印如下:
```
I0601 08:08:27.951220 8697 pinned_memory_manager.cc:240] Pinned memory pool is created at '0x7f5c1c000000' with size 268435456
I0601 08:08:27.953774 8697 cuda_memory_manager.cc:105] CUDA memory pool is created on device 0 with size 67108864
I0601 08:08:27.958255 8697 model_repository_manager.
...
I0613 08:59:20.577820 10021 server.cc:592]
+----------------------------+---------+--------+
| Model                      | Version | Status |
+----------------------------+---------+--------+
| ernie_seqcls               | 1       | READY  |
| ernie_seqcls_model         | 1       | READY  |
| ernie_seqcls_postprocess   | 1       | READY  |
| ernie_tokenizer            | 1       | READY  |
+----------------------------+---------+--------+
...
I0601 07:15:15.923270 8059 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0601 07:15:15.923604 8059 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0601 07:15:15.964984 8059 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

### 序列标注任务
在容器内执行下面命令启动序列标注服务:
```
tritonserver --model-repository=/models --model-control-mode=explicit --load-model=ernie_tokencls --backend-config=python,shm-default-byte-size=10485760
```
输出打印如下:
```
I0601 08:08:27.951220 8697 pinned_memory_manager.cc:240] Pinned memory pool is created at '0x7f5c1c000000' with size 268435456
I0601 08:08:27.953774 8697 cuda_memory_manager.cc:105] CUDA memory pool is created on device 0 with size 67108864
I0601 08:08:27.958255 8697 model_repository_manager.
...
I0613 08:59:20.577820 10021 server.cc:592]
+----------------------------+---------+--------+
| Model                      | Version | Status |
+----------------------------+---------+--------+
| ernie_tokencls             | 1       | READY  |
| ernie_tokencls_model       | 1       | READY  |
| ernie_tokencls_postprocess | 1       | READY  |
| ernie_tokenizer            | 1       | READY  |
+----------------------------+---------+--------+
...
I0601 07:15:15.923270 8059 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0601 07:15:15.923604 8059 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0601 07:15:15.964984 8059 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

## 客户端请求
客户端请求可以在本地执行脚本请求；也可以下载官方客户端镜像，在容器中执行。

本地执行脚本需要先安装依赖:
```
pip install grpcio
pip install tritonclient==2.10.0
```

拉取官网镜像并启动容器:
```
# 拉取镜像
docker pull nvcr.io/nvidia/tritonserver:21.10-py3-sdk

#启动容器
docker run  -it --net=host --name triton_client -v /path/to/triton:/triton_code nvcr.io/nvidia/tritonserver:21.10-py3-sdk bash
```

### 分类任务
注意执行客户端请求时关闭代理，并根据实际情况修改main函数中的ip地址(启动服务所在的机器)
```
python seq_cls_grpc_client.py
```
输出打印如下:
```
{'label': array([5, 9]), 'confidence': array([0.6425664 , 0.66534853], dtype=float32)}
{'label': array([4]), 'confidence': array([0.53198355], dtype=float32)}
acc: 0.5731
```

### 序列标注任务
注意执行客户端请求时关闭代理，并根据实际情况修改main函数中的ip地址(启动服务所在的机器)
```
python token_cls_grpc_client.py
```
输出打印如下:
```
input data: 北京的涮肉，重庆的火锅，成都的小吃都是极具特色的美食。
The model detects all entities:
entity: 北京   label: LOC   pos: [0, 1]
entity: 重庆   label: LOC   pos: [6, 7]
entity: 成都   label: LOC   pos: [12, 13]
input data: 原产玛雅故国的玉米，早已成为华夏大地主要粮食作物之一。
The model detects all entities:
entity: 玛雅   label: LOC   pos: [2, 3]
entity: 华夏   label: LOC   pos: [14, 15]
```
