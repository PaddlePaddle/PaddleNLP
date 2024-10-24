# 基于 Paddle Serving 的服务化部署

本文档将介绍如何使用[Paddle Serving](https://github.com/PaddlePaddle/Serving/blob/develop/README_CN.md)工具搭建多标签在线服务部署。

## 目录
- [环境准备](#环境准备)
- [模型转换](#模型转换)
- [部署模型](#部署模型)

## 环境准备
需要准备 PaddleNLP 的运行环境和 Paddle Serving 的运行环境。

- python >= 3.6
- paddlepaddle >= 2.3
- paddlenlp >= 2.4

### 安装 PaddlePaddle

 环境中 paddlepaddle-gpu 或 paddlepaddle 版本应大于或等于2.3, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的 PaddlePaddle 下载命令。


### 安装 PaddleNLP

安装 PaddleNLP 默认开启百度镜像源来加速下载，如果您使用 HTTP 代理可以删去` -i https://mirror.baidu.com/pypi/simple` ，更多关于 PaddleNLP 安装的详细教程请查见[PaddleNLP 快速安装](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)。

```shell
python3 -m pip install --upgrade paddlenlp -i https://mirror.baidu.com/pypi/simple
```
### 安装 Paddle Serving
安装 client 和 serving app，用于向服务发送请求:
```
pip install paddle_serving_app paddle_serving_client
```
安装 serving，用于启动服务，根据服务器设备选择安装 CPU server 或 GPU server：

- 安装 CPU server
```shell
pip install paddle_serving_server
```
- 安装 GPU server, 注意选择跟本地环境一致的命令
```shell
# CUDA10.2 + Cudnn7 + TensorRT6
pip install paddle-serving-server-gpu==0.8.3.post102 -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA10.1 + TensorRT6
pip install paddle-serving-server-gpu==0.8.3.post101 -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA11.2 + TensorRT8
pip install paddle-serving-server-gpu==0.8.3.post112 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**NOTE:**
- 默认开启国内清华镜像源来加速下载，如果您使用 HTTP 代理可以关闭(-i https://pypi.tuna.tsinghua.edu.cn/simple)
- 更多 wheel 包请参考[serving 官网文档](https://github.com/PaddlePaddle/Serving/blob/develop/doc/Latest_Packages_CN.md)


## 模型转换

使用 Paddle Serving 做服务化部署时，需要将保存的 inference 模型转换为 serving 易于部署的模型。

用已安装的 paddle_serving_client 将静态图参数模型转换成 serving 格式。如何使用[静态图导出脚本](../../export_model.py)将训练后的模型转为静态图模型详见[模型静态图导出](../../README.md)，模型地址`dirname`，模型文件和参数名`model_filename`，`params_filename`根据实际填写即可。

```shell
python -m paddle_serving_client.convert --dirname ../../export --model_filename float32.pdmodel --params_filename float32.pdiparams
```

可以通过命令查参数含义：
```shell
python -m paddle_serving_client.convert --help
```

转换成功后的目录如下:
```
paddle_serving/
├──serving_server
│  ├── float32.pdiparams
│  ├── float32.pdmodel
│  ├── serving_server_conf.prototxt
│  └── serving_server_conf.stream.prototxt
└──serving_client
   ├── serving_client_conf.prototxt
   └── serving_client_conf.stream.prototxt
```

## 部署模型

serving 目录包含启动 pipeline 服务和发送预测请求的代码和模型，包括：

```
serving/
├──serving_server
│  ├── float32.pdiparams
│  ├── float32.pdmodel
│  ├── serving_server_conf.prototxt
│  └── serving_server_conf.stream.prototxt
├──config.yml        # 分类任务启动服务端的配置文件
├──rpc_client.py     # 分类任务发送pipeline预测请求的脚本
└──service.py        # 分类任务启动服务端的脚本
```

### 修改配置文件
目录中的`config.yml`文件解释了每一个参数的含义，可以根据实际需要修改其中的配置。比如：
```
# 修改模型目录为下载的模型目录或自己的模型目录:
model_config: serving_server =>  model_config: erine-3.0-tiny/serving_server

# 修改rpc端口号
rpc_port: 10231   =>   rpc_port: 9998

# 修改使用GPU推理为使用CPU推理:
device_type: 1    =>   device_type: 0

#开启MKLDNN加速
#use_mkldnn: False    =>   use_mkldnn: True

#Fetch结果列表，以serving_client/serving_client_conf.prototxt中fetch_var的alias_name为准
fetch_list: ["linear_147.tmp_1"]    =>   fetch_list: ["linear_75.tmp_1"]
```

### 分类任务
#### 启动服务
修改好配置文件后，执行下面命令启动服务:
```shell
python service.py --max_seq_length 128 --model_name "ernie-3.0-medium-zh"
```

可支持配置的参数：
* `max_seq_length`：分词器 tokenizer 使用的最大序列长度，ERNIE 模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `model_name`：选择预训练模型,可选"ernie-1.0-large-zh-cw","ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en","ernie-m-base","ernie-m-large"；默认为"ernie-3.0-medium-zh",根据实际使用的预训练模型选择。

输出打印如下:
```
[DAG] Succ init
[PipelineServicer] succ init
......
--- Running analysis [ir_graph_to_program_pass]
I0625 16:44:36.563802 40218 analysis_predictor.cc:1007] ======= optimize end =======
I0625 16:44:36.571702 40218 naive_executor.cc:102] ---  skip [feed], feed -> token_type_ids
I0625 16:44:36.571728 40218 naive_executor.cc:102] ---  skip [feed], feed -> input_ids
I0625 16:44:36.574352 40218 naive_executor.cc:102] ---  skip [linear_147.tmp_1], fetch -> fetch
[2022-06-25 16:44:37,546] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer'> to load 'ernie-3.0-medium-zh'.
[2022-06-25 16:44:37,546] [    INFO] - Already cached /root/.paddlenlp/models/ernie-3.0-medium-zh/ernie_3.0_base_zh_vocab.txt
[OP Object] init success
W0625 16:45:40.312942 40218 gpu_context.cc:278] Please NOTE: device: 3, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.2
W0625 16:45:40.316538 40218 gpu_context.cc:306] device: 3, cuDNN Version: 8.1.
```

#### 启动 rpc client 测试
注意执行客户端请求时关闭代理，并根据实际情况修改 server_url 地址(启动服务所在的机器)
```shell
python rpc_client.py
```
输出打印如下:
```
data:  五松新村房屋是被告婚前购买的；
label:  婚前个人财产
--------------------
data:  被告于2016年3月将车牌号为皖B×××××出售了2.7万元，被告通过原告偿还了齐荷花人民币2.6万元，原、被告尚欠齐荷花2万元。
label:  有夫妻共同财产,有夫妻共同债务
--------------------
data:  2、判令被告返还借婚姻索取的现金33万元，婚前个人存款10万元；
label:  婚前个人财产
--------------------
data:  一、判决原告于某某与被告杨某某离婚；
label:  准予离婚,法定离婚
```
#### 启动 http client 测试
注意执行客户端请求时关闭代理，并根据实际情况修改 server_url 地址(启动服务所在的机器)
```shell
python http_client.py
```
输出打印如下:
```
data:  五松新村房屋是被告婚前购买的；
label:  婚前个人财产
--------------------
data:  被告于2016年3月将车牌号为皖B×××××出售了2.7万元，被告通过原告偿还了齐荷花人民币2.6万元，原、被告尚欠齐荷花2万元。
label:  有夫妻共同财产,有夫妻共同债务
--------------------
data:  2、判令被告返还借婚姻索取的现金33万元，婚前个人存款10万元；
label:  婚前个人财产
--------------------
data:  一、判决原告于某某与被告杨某某离婚；
label:  准予离婚,法定离婚
```
