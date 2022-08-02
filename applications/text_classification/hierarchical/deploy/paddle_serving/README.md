# 基于Paddle Serving的服务化部署

本文档将介绍如何使用[Paddle Serving](https://github.com/PaddlePaddle/Serving/blob/develop/README_CN.md)工具部署基于ERNIE 2.0的层次分类部署pipeline在线服务。

## 目录
- [环境准备](#环境准备)
- [模型转换](#模型转换)
- [部署模型](#部署模型)

## 环境准备
需要[准备PaddleNLP的运行环境]()和Paddle Serving的运行环境。

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

### 安装FasterTokenizer文本处理加速库（可选）
如果部署环境是Linux，推荐安装faster_tokenizer可以得到更极致的文本处理效率，进一步提升服务性能。目前暂不支持Windows设备安装，将会在下个版本支持。
```shell
pip install faster_tokenizer
```


## 模型转换

使用Paddle Serving做服务化部署时，需要将保存的inference模型转换为serving易于部署的模型。

用已安装的paddle_serving_client将静态图参数模型转换成serving格式。如何使用[静态图导出脚本](../../export_model.py)将训练后的模型转为静态图模型详见[模型静态图导出](../../README.md)，模型地址--dirname根据实际填写即可。

```shell
python -m paddle_serving_client.convert --dirname ../../export --model_filename float32.pdmodel --params_filename float32.pdiparams
```
可以通过命令查参数含义：
```shell
python -m paddle_serving_client.convert --help
```
转换成功后的目录如下:
```
serving_server/
├── float32.pdiparams
├── float32.pdmodel
├── serving_server_conf.prototxt
└── serving_server_conf.stream.prototxt
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

# 修改rpc端口号为9998
rpc_port: 9998   =>   rpc_port: 9998

# 修改使用GPU推理为使用CPU推理:
device_type: 1    =>   device_type: 0

#Fetch结果列表，以serving_client/serving_client_conf.prototxt中fetch_var的alias_name为准
fetch_list: ["linear_113.tmp_1"]    =>   fetch_list: ["linear_147.tmp_1"]

#开启MKLDNN加速
#use_mkldnn: True    =>   use_mkldnn: True
```

### 分类任务
#### 启动服务
修改好配置文件后，执行下面命令启动服务:
```shell
python service.py
```
输出打印如下:
```
[DAG] Succ init
[PipelineServicer] succ init
......
--- Running analysis [ir_graph_to_program_pass]
I0624 06:31:00.891119 13138 analysis_predictor.cc:1007] ======= optimize end =======
I0624 06:31:00.899907 13138 naive_executor.cc:102] ---  skip [feed], feed -> token_type_ids
I0624 06:31:00.899941 13138 naive_executor.cc:102] ---  skip [feed], feed -> input_ids
I0624 06:31:00.902855 13138 naive_executor.cc:102] ---  skip [linear_147.tmp_1], fetch -> fetch
[2022-06-24 06:31:01,899] [ WARNING] - Can't find the faster_tokenizers package, please ensure install faster_tokenizers correctly. You can install faster_tokenizers by `pip install faster_tokenizers`(Currently only work for linux platform).
[2022-06-24 06:31:01,899] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer'> to load 'ernie-2.0-base-en'.
[2022-06-24 06:31:01,899] [    INFO] - Already cached /root/.paddlenlp/models/ernie-2.0-base-en/vocab.txt
[OP Object] init success
```

#### 启动client测试
注意执行客户端请求时关闭代理，并根据实际情况修改server_url地址(启动服务所在的机器)
```shell
python rpc_client.py
```
输出打印如下:
```
text:  b'a high degree of uncertainty associated with the emission inventory for china tends to degrade the performance of chemical transport models in predicting pm2.5 concentrations especially on a daily basis. in this study a novel machine learning algorithm, geographically -weighted gradient boosting machine (gw-gbm), was developed by improving gbm through building spatial smoothing kernels to weigh the loss function. this modification addressed the spatial nonstationarity of the relationships between pm2.5 concentrations and predictor variables such as aerosol optical depth (aod) and meteorological conditions. gw-gbm also overcame the estimation bias of pm2.5 concentrations due to missing aod retrievals, and thus potentially improved subsequent exposure analyses. gw-gbm showed good performance in predicting daily pm2.5 concentrations (r-2 = 0.76, rmse = 23.0 g/m(3)) even with partially missing aod data, which was better than the original gbm model (r-2 = 0.71, rmse = 25.3 g/m(3)). on the basis of the continuous spatiotemporal prediction of pm2.5 concentrations, it was predicted that 95% of the population lived in areas where the estimated annual mean pm2.5 concentration was higher than 35 g/m(3), and 45% of the population was exposed to pm2.5 >75 g/m(3) for over 100 days in 2014. gw-gbm accurately predicted continuous daily pm2.5 concentrations in china for assessing acute human health effects. (c) 2017 elsevier ltd. all rights reserved.'
label:  0,8
--------------------
...
```
