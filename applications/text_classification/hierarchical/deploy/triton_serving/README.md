# 基于Triton Inference Server的服务化部署指南

本文档将介绍如何使用[Triton Inference Server](https://github.com/triton-inference-server/server)工具部署基于ERNIE 2.0英文模型文本层次分类的pipeline在线服务。

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


### 安装FasterTokenizers文本处理加速库（可选）

如果部署环境是Linux，推荐安装faster_tokenizers可以得到更极致的文本处理效率，进一步提升服务性能。目前暂不支持Windows设备安装，将会在下个版本支持。

在容器内安装 faster_tokenizers
```shell
python3 -m pip install faster_tokenizers
```


## 模型获取和转换

使用Triton做服务化部署时，选择ONNX Runtime后端运行需要先将模型转换成ONNX格式。


首先将保存的动态图参数导出成静态图参数，具体代码见[静态图导出脚本](../../export_model.py)。静态图参数保存在`output_path`指定路径中。运行方式：

```shell
python ../../export_model.py --params_path=../../checkpoint/model_state.pdparams --output_path=./wos_infer_model
```

使用Paddle2ONNX将Paddle静态图模型转换为ONNX模型格式的命令如下，以下命令成功运行后，将会在当前目录下生成model.onnx模型文件。

```shell
paddle2onnx --model_dir infer_model/ --model_filename float32.pdmodel --params_filename float32.pdiparams --save_file model.onnx --opset_version 13 --enable_onnx_checker True --enable_dev_version True
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

模型配置文件config.pbtxt配置细节请参见[Triton Server Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md)

## 部署模型

triton目录包含启动pipeline服务的配置和发送预测请求的代码，包括：

```
models                    # Triton启动需要的模型仓库，包含模型和服务配置文件
seqcls_grpc_client.py     # 层次分类任务发送pipeline预测请求的脚本
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

**NOTE:**

启动服务时，Triton Server的每个python后端进程默认申请`64M`内存，默认启动的docker无法启动多个python后端节点。两个解决方案：

1. 启动容器时设置`shm-size`参数, 比如:`docker run  -it --net=host --name triton_server --shm-size="1g" -v /path/triton/models:/models nvcr.io/nvidia/tritonserver:21.10-py3 bash`

2. 启动服务时设置python后端的`shm-default-byte-size`参数, 设置python后端的默认内存为10M： `tritonserver --model-repository=/models --backend-config=python,shm-default-byte-size=10485760`

## 客户端请求

### 客户端环境准备
客户端请求有两种方式，可以选择在在本地执行脚本请求，或下载官方客户端镜像在容器中执行。

方式一：本地执行脚本，需要先安装依赖:
```
pip install grpcio
pip install tritonclient==2.10.0
```

方式二：拉取官网镜像并启动容器:
```
docker pull nvcr.io/nvidia/tritonserver:21.10-py3-sdk
docker run  -it --net=host --name triton_client -v /path/to/triton:/triton_code nvcr.io/nvidia/tritonserver:21.10-py3-sdk bash
```

### 启动客户端测试
注意执行客户端请求时关闭代理，并根据实际情况修改main函数中的ip地址(启动服务所在的机器)

```
python seqcls_grpc_client.py
```

输出打印如下:

```
text:
a high degree of uncertainty associated with the emission inventory for china tends to degrade the performance of chemical transport models in predicting pm2.5 concentrations especially on a daily basis. in this study a novel machine learning algorithm, geographically -weighted gradient boosting machine (gw-gbm), was developed by improving gbm through building spatial smoothing kernels to weigh the loss function. this modification addressed the spatial nonstationarity of the relationships between pm2.5 concentrations and predictor variables such as aerosol optical depth (aod) and meteorological conditions. gw-gbm also overcame the estimation bias of pm2.5 concentrations due to missing aod retrievals, and thus potentially improved subsequent exposure analyses. gw-gbm showed good performance in predicting daily pm2.5 concentrations (r-2 = 0.76, rmse = 23.0 g/m(3)) even with partially missing aod data, which was better than the original gbm model (r-2 = 0.71, rmse = 25.3 g/m(3)). on the basis of the continuous spatiotemporal prediction of pm2.5 concentrations, it was predicted that 95% of the population lived in areas where the estimated annual mean pm2.5 concentration was higher than 35 g/m(3), and 45% of the population was exposed to pm2.5 >75 g/m(3) for over 100 days in 2014. gw-gbm accurately predicted continuous daily pm2.5 concentrations in china for assessing acute human health effects. (c) 2017 elsevier ltd. all rights reserved.
- level 1
label: 0  confidence: 0.997
- level 2
label: 8  confidence: 0.991
...
```
