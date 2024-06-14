# 基于Triton Inference Server的服务化部署指南

本文档将介绍如何使用[Triton Inference Server](https://github.com/triton-inference-server/server)工具部署基于ERNIE 3.0中文模型文本多分类的pipeline在线服务。

## 目录
- [服务端环境准备](#服务端环境准备)
- [模型获取和转换](#模型获取和转换)
- [部署模型](#部署模型)
- [客户端请求](#客户端请求)

## 服务端环境准备

### 安装Triton Server
拉取Triton Server镜像：
```shell
docker pull nvcr.io/nvidia/tritonserver:21.10-py3
```
启动容器：
```shell
docker run  -it --gpus all --net=host --name triton_server -v /path/triton/models:/models nvcr.io/nvidia/tritonserver:21.10-py3 bash
```

**NOTE:**

1. Triton版本号`21.10`可以根据自己的需求调整，各个Triton版本对应的Driver、CUDA、TRT和ONNX Runtime等后端版本可以参考[官网文档](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)。注意其中的`NVIDIA Driver`行，如果NVIDIA Driver低于文档中要求，在启动运行时会报错。

2. 可以使用`--gpus '"device=1"'`来指定GPU卡号，更多GPU指定方式请参见[Nvidia User Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration)


### 进入容器并准备PaddleNLP环境
整个服务的前后处理依赖PaddleNLP，需要在容器内安装相关python包

进入容器：
```shell
docker exec -it triton_server bash
```
安装PaddlePaddle、PaddleNLP
```shell
python3 -m pip install paddlepaddle-gpu paddlenlp -i https://mirror.baidu.com/pypi/simple
```

**NOTE:**

1. 默认开启百度镜像源来加速下载，如果您使用 HTTP 代理可以关闭(-i https://mirror.baidu.com/pypi/simple)

2. 环境中paddlepaddle-gpu或paddlepaddle版本应大于或等于2.2, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的PaddlePaddle下载命令。

3. 更多关于PaddleNLP安装的详细教程请查看[Installation](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)。


### 安装FastTokenizer文本处理加速库（可选）

> 重要提示：由于FastTokenizer长时间未得到维护，因此可能会遇到训练（基于Python实现的tokenizer）与部署（基于C++实现的tokenizer）阶段分词不一致的问题。为了确保稳定性和一致性，我们建议避免安装该库。

如果想要安装fast_tokenizer，以获得更高的文本处理效率，从而显著提升服务性能。您可以通过以下命令进行安装：
```shell
python3 -m pip install fast-tokenizer-python
```


## 模型获取和转换

使用Triton做服务化部署时，选择ONNX Runtime后端运行需要先将模型转换成ONNX格式。使用Paddle2ONNX将Paddle静态图模型转换为ONNX模型格式的命令如下，以下命令成功运行后，将会在当前目录下生成model.onnx模型文件。
```shell
paddle2onnx --model_dir ../../checkpoint/export --model_filename model.pdmodel --params_filename model.pdiparams --save_file model.onnx --opset_version 13 --enable_onnx_checker True --enable_dev_version True
```
创建空白目录/seqcls/1和seqcls_model/1，并将将转换好的ONNX模型移动到模型仓库目录
```shell
mkdir /models/seqcls/1
mkdir /models/seqcls_model/1
mv model.onnx /models/seqcls_model/1
```

Paddle2ONNX的命令行参数说明请查阅：[Paddle2ONNX命令行参数说明](https://github.com/PaddlePaddle/Paddle2ONNX#%E5%8F%82%E6%95%B0%E9%80%89%E9%A1%B9)

模型下载转换好之后，models目录结构如下:
```
models
├── seqcls
│   ├── 1
│   └── config.pbtxt
├── seqcls_model
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
├── seqcls_postprocess
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
└── tokenizer
    ├── 1
    │   └── model.py
    └── config.pbtxt
```

模型配置文件config.pbtxt配置细节请参见[Triton Server Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md)

## 部署模型

triton目录包含启动pipeline服务的配置和发送预测请求的代码，包括：

```
models                    # Triton启动需要的模型仓库，包含模型和服务配置文件
seqcls_grpc_client.py     # 分类任务发送pipeline预测请求的脚本
```

### 启动服务端

在容器内执行下面命令启动服务，默认启动models下所有模型:
```shell
tritonserver --model-repository=/models
```
也可以通过设定参数只启动单一任务服务：
```shell
tritonserver --model-repository=/models --model-control-mode=explicit --load-model=seqcls
```

**NOTE:**

启动服务时，Triton Server的每个python后端进程默认申请`64M`内存，默认启动的docker无法启动多个python后端节点。两个解决方案：

1. 启动容器时设置`shm-size`参数, 比如:`docker run  -it --net=host --name triton_server --shm-size="1g" -v /path/triton/models:/models nvcr.io/nvidia/tritonserver:21.10-py3 bash`

2. 启动服务时设置python后端的`shm-default-byte-size`参数, 设置python后端的默认内存为10M： `tritonserver --model-repository=/models --backend-config=python,shm-default-byte-size=10485760`

输出打印如下:

```
...
I0619 13:40:51.590901 5127 onnxruntime.cc:1999] TRITONBACKEND_Initialize: onnxruntime
I0619 13:40:51.590938 5127 onnxruntime.cc:2009] Triton TRITONBACKEND API version: 1.6
I0619 13:40:51.590947 5127 onnxruntime.cc:2015] 'onnxruntime' TRITONBACKEND API version: 1.6
I0619 13:40:51.623808 5127 openvino.cc:1193] TRITONBACKEND_Initialize: openvino
I0619 13:40:51.623862 5127 openvino.cc:1203] Triton TRITONBACKEND API version: 1.6
I0619 13:40:51.623868 5127 openvino.cc:1209] 'openvino' TRITONBACKEND API version: 1.6
I0619 13:40:52.980990 5127 pinned_memory_manager.cc:240] Pinned memory pool is created at '0x7f14d8000000' with size 268435456
...
I0619 13:43:33.360018 5127 server.cc:592]
+--------------------+---------+--------+
| Model              | Version | Status |
+--------------------+---------+--------+
| seqcls             | 1       | READY  |
| seqcls_model       | 1       | READY  |
| seqcls_postprocess | 1       | READY  |
| tokenizer          | 1       | READY  |
+--------------------+---------+--------+
...
I0619 13:43:33.365824 5127 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0619 13:43:33.366221 5127 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0619 13:43:33.409775 5127 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

## 客户端请求

### 客户端环境准备
客户端请求有两种方式，可以选择在本地执行脚本请求，或下载官方客户端镜像在容器中执行。

方式一：本地执行脚本，需要先安装依赖:
```shell
pip install grpcio
pip install tritonclient==2.10.0
```

方式二：拉取官网镜像并启动容器:
```shell
docker pull nvcr.io/nvidia/tritonserver:21.10-py3-sdk
docker run  -it --net=host --name triton_client -v /path/to/triton:/triton_code nvcr.io/nvidia/tritonserver:21.10-py3-sdk bash
```

### 启动客户端测试
注意执行客户端请求时关闭代理，并根据实际情况修改main函数中的ip地址(启动服务所在的机器)

```shell
python seqcls_grpc_client.py
```

输出打印如下:

```
text:  黑苦荞茶的功效与作用及食用方法
label:  功效作用
confidence:  0.984
--------------------
text:  交界痣会凸起吗
label:  疾病表述
confidence:  0.904
--------------------
text:  检查是否能怀孕挂什么科
label:  就医建议
confidence:  0.969
--------------------
text:  幼儿挑食的生理原因是
label:  病因分析
confidence:  0.495
--------------------
text:  鱼油怎么吃咬破吃还是直接咽下去
label:  其他
confidence:  0.850
--------------------
```
