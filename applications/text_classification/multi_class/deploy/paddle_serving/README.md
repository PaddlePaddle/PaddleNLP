# 基于Paddle Serving的服务化部署

本文档将介绍如何使用[Paddle Serving](https://github.com/PaddlePaddle/Serving/blob/develop/README_CN.md)工具搭建多分类在线服务部署。

## 目录
- [环境准备](#环境准备)
- [模型转换](#模型转换)
- [部署模型](#部署模型)

## 环境准备
需要准备PaddleNLP的运行环境和Paddle Serving的运行环境。

- python >= 3.6
- paddlepaddle >= 2.3
- paddlenlp >= 2.4

### 安装PaddlePaddle

 环境中paddlepaddle-gpu或paddlepaddle版本应大于或等于2.3, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的PaddlePaddle下载命令。


### 安装PaddleNLP


安装PaddleNLP默认开启百度镜像源来加速下载，如果您使用 HTTP 代理可以删去` -i https://mirror.baidu.com/pypi/simple` ，更多关于PaddleNLP安装的详细教程请查见[PaddleNLP快速安装](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)。

```shell
python3 -m pip install --upgrade paddlenlp -i https://mirror.baidu.com/pypi/simple
```
### 安装Paddle Serving
安装client和serving app，用于向服务发送请求:
```shell
pip install paddle_serving_app paddle_serving_client
```
安装serving，用于启动服务，根据服务器设备选择安装CPU server或GPU server：

- 安装CPU server
```shell
pip install paddle_serving_server
```
- 安装GPU server, 注意选择跟本地环境一致的命令
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
- 更多wheel包请参考[serving官网文档](https://github.com/PaddlePaddle/Serving/blob/develop/doc/Latest_Packages_CN.md)


### 安装FastTokenizer文本处理加速库（可选）
推荐安装fast_tokenizer可以得到更极致的文本处理效率，进一步提升服务性能。
```shell
pip install fast-tokenizer-python
```


## 模型转换

使用Paddle Serving做服务化部署时，需要将保存的inference模型转换为serving易于部署的模型。

用已安装的paddle_serving_client将静态图参数模型转换成serving格式。如何使用[静态图导出脚本](../../export_model.py)将训练后的模型转为静态图模型详见[模型静态图导出](../../README.md)，模型地址`dirname`，模型文件和参数名`model_filename`，`params_filename`根据实际填写即可。

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

serving目录包含启动pipeline服务和发送预测请求的代码和模型，包括：

```
serving/
├──serving_server
│  ├── float32.pdiparams
│  ├── float32.pdmodel
│  ├── serving_server_conf.prototxt
│  └── serving_server_conf.stream.prototxt
├──config.yml        # 层次分类任务启动服务端的配置文件
├──rpc_client.py     # 层次分类任务发送pipeline预测请求的脚本
└──service.py        # 层次分类任务启动服务端的脚本

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
* `max_seq_length`：分词器tokenizer使用的最大序列长度，ERNIE模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `model_name`：选择预训练模型,可选"ernie-1.0-large-zh-cw","ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en","ernie-m-base","ernie-m-large"；默认为"ernie-3.0-medium-zh",根据实际使用的预训练模型选择。

输出打印如下:
```
[DAG] Succ init
[PipelineServicer] succ init
...
--- Running analysis [ir_graph_to_program_pass]
I0628 09:12:30.739985 74305 analysis_predictor.cc:1007] ======= optimize end =======
I0628 09:12:30.776288 74305 naive_executor.cc:102] ---  skip [feed], feed -> token_type_ids
I0628 09:12:30.779004 74305 naive_executor.cc:102] ---  skip [feed], feed -> input_ids
I0628 09:12:30.787542 74305 naive_executor.cc:102] ---  skip [linear_147.tmp_1], fetch -> fetch
[2022-06-28 09:12:32,879] [ WARNING] - Can't find the fast_tokenizer package, please ensure install fast_tokenizer correctly. You can install fast_tokenizer by `pip install fast-tokenizer-python`.
[2022-06-28 09:12:32,880] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer'> to load 'ernie-3.0-medium-zh'.
[2022-06-28 09:12:32,880] [    INFO] - Already cached /root/.paddlenlp/models/ernie-3.0-medium-zh/ernie_3.0_base_zh_vocab.txt
[OP Object] init success

```

#### 启动rpc client测试
注意执行客户端请求时关闭代理，并根据实际情况修改server_url地址(启动服务所在的机器)
```shell
python rpc_client.py
```
输出打印如下:
```
data:  黑苦荞茶的功效与作用及食用方法
label:  功效作用
--------------------
data:  交界痣会凸起吗
label:  疾病表述
--------------------
data:  检查是否能怀孕挂什么科
label:  就医建议
--------------------
data:  鱼油怎么吃咬破吃还是直接咽下去
label:  其他
--------------------
data:  幼儿挑食的生理原因是
label:  病因分析
--------------------

```

#### 启动http client测试
注意执行客户端请求时关闭代理，并根据实际情况修改server_url地址(启动服务所在的机器)
```shell
python http_client.py
```
输出打印如下:
```
data:  黑苦荞茶的功效与作用及食用方法
label:  功效作用
--------------------
data:  交界痣会凸起吗
label:  疾病表述
--------------------
data:  检查是否能怀孕挂什么科
label:  就医建议
--------------------
data:  鱼油怎么吃咬破吃还是直接咽下去
label:  其他
--------------------
data:  幼儿挑食的生理原因是
label:  病因分析
--------------------

```
