# FastDeploy ERNIE 3.0 模型 Serving 部署示例


在服务化部署前，需确认

- 1. 服务化镜像的软硬件环境要求和镜像拉取命令请参考 [FastDeploy 服务化部署](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/README_CN.md)

## 准备模型

以下示例展示如何基于 FastDeploy 库完成 ERNIE 3.0 模型在 CLUE Benchmark 的 [AFQMC 数据集](https://github.com/CLUEbenchmark/CLUE)上进行文本分类任务以及 [MSRA_NER 数据集](https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra)上进行序列标注任务的**服务化部署**。按照[ERNIE 3.0 训练文档](../../README.md)分别训练并导出文本分类模型以及序列标注模型，并将导出的模型移动到 models 目录下相应位置。

模型移动好之后，文本分类任务的 models 目录结构如下:

```
models
├── ernie_seqcls                      # 分类任务的pipeline
│   ├── 1
│   └── config.pbtxt                  # 通过这个文件组合前后处理和模型推理
├── ernie_seqcls_model                # 分类任务的模型推理
│   ├── 1
│   │   ├── model.pdiparams
│   │   └── model.pdmodel
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

序列标注任务的 models 目录结构如下:

```
models
├── ernie_tokencls                      # 序列标注任务的 pipeline
│   ├── 1
│   └── config.pbtxt                    # 通过这个文件组合前后处理和模型推理
├── ernie_tokencls_model                # 序列标注任务的模型推理
│   ├── 1
│   │   ├── model.pdiparams
│   │   └── model.pdmodel
│   └── config.pbtxt
├── ernie_tokencls_postprocess          # 序列标注任务后处理
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
└── ernie_tokenizer                     # 预处理分词
    ├── 1
    │   └── model.py
    └── config.pbtxt
```

## 拉取并运行镜像

```
# x.y.z为镜像版本号，需参照serving文档替换为数字
# GPU镜像
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
# CPU镜像
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-cpu-only-21.10

# 运行
docker run  -it --net=host --name fastdeploy_server --shm-size="1g" -v /path/serving/models:/models registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-cpu-only-21.10 bash
```

## 部署模型

serving 目录包含启动 pipeline 服务的配置和发送预测请求的代码，包括：

```
models                    # 服务化启动需要的模型仓库，包含模型和服务配置文件
seq_cls_rpc_client.py     # AFQMC 分类任务发送 pipeline 预测请求的脚本
token_cls_rpc_client.py   # 序列标注任务发送 pipeline 预测请求的脚本
```

注意:启动服务时，Server 的每个 python 后端进程默认申请 64M 内存，默认启动的 docker 无法启动多个 python 后端节点。有两个解决方案：

1. 启动容器时设置 shm-size 参数, 比如: docker run  -it --net=host --name fastdeploy_server --shm-size="1g" -v /path/serving/models:/models registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10 bash

2. 启动服务时设置 python 后端的 shm-default-byte-size 参数, 设置 python 后端的默认内存为10M： tritonserver --model-repository=/models --backend-config=python,shm-default-byte-size=10485760

### 分类任务

在容器内执行下面命令启动服务:

```
# 默认启动models下所有模型
fastdeployserver --model-repository=/models

# 可通过参数只启动分类任务
fastdeployserver --model-repository=/models --model-control-mode=explicit --load-model=ernie_seqcls
```

输出打印如下:

```shell

I1019 09:41:15.375496 2823 model_repository_manager.cc:1183] successfully loaded 'ernie_tokenizer' version 1
I1019 09:41:15.375987 2823 model_repository_manager.cc:1022] loading: ernie_seqcls:1
I1019 09:41:15.477147 2823 model_repository_manager.cc:1183] successfully loaded 'ernie_seqcls' version 1
I1019 09:41:15.477325 2823 server.cc:522]
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

```shell
fastdeployserver --model-repository=/models --model-control-mode=explicit --load-model=ernie_tokencls --backend-config=python,shm-default-byte-size=10485760
```

输出打印如下:

```shell
I1019 09:41:15.375496 2823 model_repository_manager.cc:1183] successfully loaded 'ernie_tokenizer' version 1
I1019 09:41:15.375987 2823 model_repository_manager.cc:1022] loading: ernie_seqcls:1
I1019 09:41:15.477147 2823 model_repository_manager.cc:1183] successfully loaded 'ernie_seqcls' version 1
I1019 09:41:15.477325 2823 server.cc:522]
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

客户端请求可以在本地执行脚本请求；也可以在容器中执行。

本地执行脚本需要先安装依赖:

```shell

pip install grpcio
pip install tritonclient[all]

# 如果bash无法识别括号，可以使用如下指令安装:
pip install tritonclient\[all\]

```

### 分类任务

注意执行客户端请求时关闭代理，并根据实际情况修改main函数中的ip地址(启动服务所在的机器)

```shell
python seq_cls_grpc_client.py
```

输出打印如下:

```shell
{'label': array([5, 9]), 'confidence': array([0.6425664 , 0.66534853], dtype=float32)}
{'label': array([4]), 'confidence': array([0.53198355], dtype=float32)}
acc: 0.5731
```


### 序列标注任务

注意执行客户端请求时关闭代理，并根据实际情况修改main函数中的ip地址(启动服务所在的机器)


```shell
python token_cls_grpc_client.py
```

输出打印如下:

```shell

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

## 配置修改

当前分类任务(ernie_seqcls_model/config.pbtxt)默认配置在 CPU上 运行 OpenVINO 引擎; 序列标注任务默认配置在 GPU 上运行 PaddleInference 引擎。如果要在 CPU/GPU 或其他推理引擎上运行, 需要修改配置，详情请参考[配置文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/model_configuration.md)。
