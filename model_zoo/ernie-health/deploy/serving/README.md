# 基于Paddle Serving的服务化部署

本文档将介绍如何使用[Paddle Serving](https://github.com/PaddlePaddle/Serving/blob/develop/README_CN.md)工具部署ERNIE Health文本分类和序列标注模型的pipeline在线服务。

## 目录
- [环境准备](#环境准备)
- [模型转换](#模型转换)
- [部署模型](#部署模型)

## 环境准备
需要[准备PaddleNLP的运行环境]()和Paddle Serving的运行环境。

### 安装Paddle Serving
安装指令如下，更多wheel包请参考[serving官网文档](https://github.com/PaddlePaddle/Serving/blob/develop/doc/Latest_Packages_CN.md)
```
# 安装client和serving app，用于向服务发送请求
pip install paddle_serving_app paddle_serving_clinet

# 安装serving，用于启动服务
# CPU server
pip install paddle_serving_server

# GPU server, 选择跟本地环境一致的命令:
# CUDA10.2 + Cudnn7 + TensorRT6
pip install paddle-serving-server-gpu==0.8.3.post102 -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA10.1 + TensorRT6
pip install paddle-serving-server-gpu==0.8.3.post101 -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA11.2 + TensorRT8
pip install paddle-serving-server-gpu==0.8.3.post112 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

默认开启国内清华镜像源来加速下载，如果您使用 HTTP 代理可以关闭(-i https://pypi.tuna.tsinghua.edu.cn/simple)


### 安装FasterTokenizer文本处理加速库（可选）
如果部署环境是Linux，推荐安装faster_tokenizer可以得到更极致的文本处理效率，进一步提升服务性能。目前暂不支持Windows设备安装，将会在下个版本支持。
```
pip install faster_tokenizers
```


## 模型转换

使用Paddle Serving做服务化部署时，需要将保存的inference模型转换为serving易于部署的模型。

下载ERNIE Health的文本分类、序列标注模型:

```
# 下载并解压文本分类模型
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_health/kuake_qic_infer_model.zip
unzip kuake_qic_infer_model.zip
# 下载并解压序列标注模型
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_health/cmeee_infer_model.zip
unzip cmeee_infer_model.zip
```

用已安装的paddle_serving_client将inference模型转换成serving格式。

```bash
# 模型地址根据实际填写即可
# 转换文本分类模型
python -m paddle_serving_client.convert --dirname ./KUAKE_QIC/ --model_filename inference.pdmodel --params_filename inference.pdiparams

# 转换序列标注模型
python -m paddle_serving_client.convert --dirname ./CMeEE/ --model_filename inference.pdmodel --params_filename inference.pdiparams

# 可通过命令查参数含义
python -m paddle_serving_client.convert --help
```
转换成功后的目录如下:
```
serving_server
├── inference.pdiparams
├── inference.pdmodel
├── serving_server_conf.prototxt
└── serving_server_conf.stream.prototxt
```

## 部署模型

serving目录包含启动pipeline服务和发送预测请求的代码，包括：

```
seq_cls_config.yml        # 文本分类任务启动服务端的配置文件
seq_cls_rpc_client.py     # 文本分类任务发送pipeline预测请求的脚本
seq_cls_service.py        # 文本分类任务启动服务端的脚本

token_cls_config.yml      # 序列标注任务启动服务端的配置文件
token_cls_rpc_client.py   # 序列标注任务发送pipeline预测请求的脚本
token_cls_service.py      # 序列标注任务启动服务端的脚本
```


### 修改配置文件
目录中的`seq_cls_config.yml`和`token_cls_config.yml`文件解释了每一个参数的含义，可以根据实际需要修改其中的配置。比如：
```
# 修改模型目录为下载的模型目录或自己的模型目录:
model_config: no_task_emb/serving_server =>  model_config: erine-health/serving_server

# 修改rpc端口号为9998
rpc_port: 9998   =>   rpc_port: 9998

# 修改使用GPU推理为使用CPU推理:
device_type: 1    =>   device_type: 0
```

### 分类任务
#### 启动服务
修改好配置文件后，执行下面命令启动服务:
```
python seq_cls_service.py
```
输出打印如下:
```
[DAG] Succ init
[PipelineServicer] succ init
--- Running analysis [ir_graph_build_pass]
......
--- Running analysis [ir_graph_to_program_pass]
I0606 16:24:02.272878 40602 analysis_predictor.cc:1007] ======= optimize end =======
I0606 16:24:02.288636 40602 naive_executor.cc:102] ---  skip [feed], feed -> position_ids
I0606 16:24:02.288674 40602 naive_executor.cc:102] ---  skip [feed], feed -> token_type_ids
I0606 16:24:02.288679 40602 naive_executor.cc:102] ---  skip [feed], feed -> input_ids
I0606 16:24:02.294317 40602 naive_executor.cc:102] ---  skip [linear_147.tmp_1], fetch -> fetch
[2022-06-06 16:24:03,672] [ WARNING] - The tokenizer <class 'paddlenlp.transformers.electra.tokenizer.ElectraTokenizer'> doesn't have the faster version. Please check the map `paddlenlp.transformers.auto.tokenizer.FASTER_TOKENIZER_MAPPING_NAMES` to see which faster tokenizers are currently supported.
[2022-06-06 16:24:03,672] [    INFO] - We are using <class 'paddlenlp.transformers.electra.tokenizer.ElectraTokenizer'> to load 'ernie-health-chinese'.
[2022-06-06 16:24:03,672] [    INFO] - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/ernie-health-chinese/vocab.txt
[OP Object] init success
```

#### 启动client测试
注意执行客户端请求时关闭代理，并根据实际情况修改init_client函数中的ip地址(启动服务所在的机器)
```
python seq_cls_rpc_client.py
```
输出打印如下:
```
{'label': array([1, 5]), 'confidence': array([0.99721164, 0.982297  ], dtype=float32)}
acc: 0.8260869565217391
```

### 实体识别任务
#### 启动服务
修改好配置文件后，执行下面命令启动服务:
```
python token_cls_service.py
```
输出打印如下:
```
[DAG] Succ init
[PipelineServicer] succ init
--- Running analysis [ir_graph_build_pass]
......
--- Running analysis [ir_graph_to_program_pass]
I0609 11:30:42.569332  1654 analysis_predictor.cc:1007] ======= optimize end =======
I0609 11:30:42.581693  1654 naive_executor.cc:102] ---  skip [feed], feed -> attention_mask
I0609 11:30:42.581728  1654 naive_executor.cc:102] ---  skip [feed], feed -> position_ids
I0609 11:30:42.581748  1654 naive_executor.cc:102] ---  skip [feed], feed -> token_type_ids
I0609 11:30:42.581754  1654 naive_executor.cc:102] ---  skip [feed], feed -> input_ids
I0609 11:30:42.587451  1654 naive_executor.cc:102] ---  skip [linear_146.tmp_1], fetch -> fetch
I0609 11:30:42.587489  1654 naive_executor.cc:102] ---  skip [linear_147.tmp_1], fetch -> fetch
[2022-06-09 11:30:43,702] [    INFO] - We are using <class 'paddlenlp.transformers.electra.tokenizer.ElectraTokenizer'> to load 'ernie-health-chinese'.
[2022-06-09 11:30:43,703] [    INFO] - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/ernie-health-chinese/vocab.txt
[OP Object] init success
```

#### 启动client测试
注意执行客户端请求时关闭代理，并根据实际情况修改init_client函数中的ip地址(启动服务所在的机器)
```
python token_cls_rpc_client.py
```
输出打印如下:
```
input data: 研究证实，细胞减少与肺内病变程度及肺内炎性病变吸收程度密切相关。
The model detects all entities:
type: bod , position: (5, 7) , name: 细胞
type: dis , position: (17, 23) , name: 肺内炎性病变
type: bod , position: (10, 11) , name: 肺
-----------------------------
input data: 可为不规则发热、稽留热或弛张热，但以不规则发热为多，可能与患儿应用退热药物导致热型不规律有关。
The model detects all entities:
type: sym , position: (12, 15) , name: 弛张热
type: sym , position: (8, 11) , name: 稽留热
type: sym , position: (2, 7) , name: 不规则发热
type: sym , position: (18, 23) , name: 不规则发热
-----------------------------
```
