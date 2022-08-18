# Paddle Serving服务化部署

本文档将介绍如何使用[Paddle Serving](https://github.com/PaddlePaddle/Serving/blob/develop/README_CN.md)工具部署自动文本摘要在线服务。

## 目录
- [Paddle Serving服务化部署](#paddle-serving服务化部署)
  - [目录](#目录)
  - [背景介绍](#背景介绍)
  - [环境准备](#环境准备)
    - [安装Paddle Serving](#安装paddle-serving)
    - [安装FasterTokenizer文本处理加速库（可选）](#安装fastertokenizer文本处理加速库可选)
  - [模型转换](#模型转换)
  - [pipeline部署](#pipeline部署)
    - [修改配置文件](#修改配置文件)
    - [server启动服务](#server启动服务)
    - [client发送服务请求](#client发送服务请求)

## 背景介绍
Paddle Serving 依托深度学习框架 PaddlePaddle 旨在帮助深度学习开发者和企业提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案，和多种经典预训练模型示例。集成高性能服务端推理引擎 Paddle Inference 和端侧引擎 Paddle Lite。设计并实现基于有向无环图(DAG) 的异步流水线高性能推理框架，具有多模型组合、异步调度、并发推理、动态批量、多卡多流推理、请求缓存等特性。

Paddle Serving Python端预测部署主要包含以下步骤：
- 环境准备
- 模型转换
- 部署模型

## 环境准备
### 安装Paddle Serving
安装client和serving app，用于向服务发送请求:
```shell
pip install paddle_serving_app paddle_serving_client
```
安装server，用于启动服务，根据服务器设备选择安装CPU server或GPU server：

- 安装CPU server
```shell
pip install paddle_serving_server
```
- 安装GPU server, 注意选择跟本地环境一致的命令
```shell
# CUDA10.2 + Cudnn7 + TensorRT6
pip install paddle-serving-server-gpu==0.8.3.post102 # -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA10.1 + TensorRT6
pip install paddle-serving-server-gpu==0.8.3.post101 # -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA11.2 + TensorRT8
pip install paddle-serving-server-gpu==0.8.3.post112 # -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**NOTE:**
- 可以开启国内清华镜像源来加速下载
- 如果要安装最新版本的PaddleServing参考[链接](https://github.com/PaddlePaddle/Serving/blob/develop/doc/Latest_Packages_CN.md)。


### 安装FasterTokenizer文本处理加速库（可选）
如果部署环境是Linux，推荐安装faster_tokenizer可以得到更极致的文本处理效率，进一步提升服务性能。目前暂不支持Windows设备安装，将会在下个版本支持。
```shell
pip install faster_tokenizer
```


## 模型转换

使用Paddle Serving做服务化部署时，需要将保存的inference模型转换为serving易于部署的模型。

用已安装的paddle_serving_client将静态图参数模型转换成serving格式。关于如何使用将训练后的动态图模型转为静态图模型详见[FasterTransformer加速及模型静态图导出](../../README.md)。

模型转换命令如下：
```shell
python -m paddle_serving_client.convert --dirname ./export_checkpoint \
                                        --model_filename unimo_text.pdmodel \
                                        --params_filename unimo_text.pdiparams \
                                        --serving_server ./deploy/paddle_serving/export_checkpoint_server \
                                        --serving_client ./deploy/paddle_serving/export_checkpoint_client
```
关键参数释义如下：
* `dirname`：模型文件夹地址。
* `model_filename`：模型文件名。
* `params_filename`：模型参数名。
* `serving_server`：server的模型文件和配置文件路径，默认"serving_server"。
* `serving_client`：client的配置文件路径，默认"serving_client"。

更多参数可通过以下命令查询：
```shell
python -m paddle_serving_client.convert --help
```
模型转换完成后，会在./delopy/paddle_serving文件夹多出export_checkpoint_server和export_checkpoint_client的文件夹，文件夹目录格式如下：
```
export_checkpoint_server/
├── unimo_text.pdiparams
├── unimo_text.pdmodel
├── serving_server_conf.prototxt
└── serving_server_conf.stream.prototxt
export_checkpoint_server/
├── serving_client_conf.prototxt
└── serving_client_conf.stream.prototxt
```

## pipeline部署

paddle_serving目录包含启动pipeline服务和发送预测请求的代码，包括：
```
paddle_serving/
├──config.yml        # 启动服务端的配置文件
├──pipeline_client.py     # 发送pipeline预测请求的脚本
└──pipeline_service.py        # 启动pipeline服务端的脚本
```

### 修改配置文件
目录中的`config.yml`文件解释了每一个参数的含义，可以根据实际需要修改其中的配置。

### server启动服务
修改好配置文件后，执行下面命令启动服务:
```shell
cd deploy/paddle_serving
# 启动服务
python pipeline_service.py
```
成功启动服务后，log.txt中会打印类似如下日志
```
--- Running analysis [ir_graph_to_program_pass]
I0818 09:56:03.768963 56655 analysis_predictor.cc:1035] ======= optimize end =======
I0818 09:56:03.769762 56655 naive_executor.cc:102] ---  skip [feed], feed -> seq_len
I0818 09:56:03.769776 56655 naive_executor.cc:102] ---  skip [feed], feed -> attention_mask
I0818 09:56:03.769781 56655 naive_executor.cc:102] ---  skip [feed], feed -> token_type_ids
I0818 09:56:03.769786 56655 naive_executor.cc:102] ---  skip [feed], feed -> input_ids
I0818 09:56:03.770205 56655 naive_executor.cc:102] ---  skip [_generated_var_3], fetch -> fetch
I0818 09:56:03.770215 56655 naive_executor.cc:102] ---  skip [slice_0.tmp_0], fetch -> fetch
[2022-08-18 09:56:03,772] [    INFO] - Already cached /root/.paddlenlp/models/unimo-text-1.0/unimo-text-1.0-vocab.txt
[2022-08-18 09:56:03,790] [    INFO] - tokenizer config file saved in /root/.paddlenlp/models/unimo-text-1.0/tokenizer_config.json
[2022-08-18 09:56:03,791] [    INFO] - Special tokens file saved in /root/.paddlenlp/models/unimo-text-1.0/special_tokens_map.json
[PipelineServicer] succ init
[OP Object] init success
[OP Object] init success
[OP Object] init success
[OP Object] init success
[OP Object] init success
[OP Object] init success
[OP Object] init success
[OP Object] init success
[OP Object] init success
[OP Object] init success
[OP Object] init success
2022/08/18 09:56:03 start proxy service
```

### client发送服务请求
执行以下命令发送文本摘要服务请求：
```shell
python pipeline_client.py
```
注意执行客户端请求时关闭代理，并根据实际情况修改server_url地址(启动服务所在的机器)。
