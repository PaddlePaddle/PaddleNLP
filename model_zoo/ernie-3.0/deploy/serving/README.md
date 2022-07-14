# 基于Paddle Serving的服务化部署

本文档将介绍如何使用[Paddle Serving](https://github.com/PaddlePaddle/Serving/blob/develop/README_CN.md)工具部署ERNIE 3.0新闻分类和序列标注模型的pipeline在线服务。

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
pip install paddle_serving_app paddle_serving_client

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
pip install faster_tokenizer
```


## 模型转换

使用Paddle Serving做服务化部署时，需要将保存的inference模型转换为serving易于部署的模型。

下载ERNIE 3.0的新闻分类、序列标注模型:

```bash
# 下载并解压新闻分类模型
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_3.0/tnews_pruned_infer_model.zip
unzip tnews_pruned_infer_model.zip
# 下载并解压序列标注模型
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_3.0/msra_ner_pruned_infer_model.zip
unzip msra_ner_pruned_infer_model.zip
```

用已安装的paddle_serving_client将inference模型转换成serving格式。

```bash
# 模型地址根据实际填写即可
# 转换新闻分类模型
python -m paddle_serving_client.convert --dirname tnews_pruned_infer_model --model_filename float32.pdmodel --params_filename float32.pdiparams

# 转换序列标注模型
python -m paddle_serving_client.convert --dirname msra_ner_pruned_infer_model --model_filename float32.pdmodel --params_filename float32.pdiparams

# 可通过命令查参数含义
python -m paddle_serving_client.convert --help
```
转换成功后的目录如下:
```
serving_server
├── float32.pdiparams
├── float32.pdmodel
├── serving_server_conf.prototxt
└── serving_server_conf.stream.prototxt
```

## 部署模型

serving目录包含启动pipeline服务和发送预测请求的代码，包括：

```
seq_cls_config.yml        # 新闻分类任务启动服务端的配置文件
seq_cls_rpc_client.py     # 新闻分类任务发送pipeline预测请求的脚本
seq_cls_service.py        # 新闻分类任务启动服务端的脚本

token_cls_config.yml      # 序列标注任务启动服务端的配置文件
token_cls_rpc_client.py   # 序列标注任务发送pipeline预测请求的脚本
token_cls_service.py      # 序列标注任务启动服务端的脚本
```


### 修改配置文件
目录中的`seq_cls_config.yml`和`token_cls_config.yml`文件解释了每一个参数的含义，可以根据实际需要修改其中的配置。比如：
```
# 修改模型目录为下载的模型目录或自己的模型目录:
model_config: no_task_emb/serving_server =>  model_config: erine-3.0-tiny/serving_server

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
I0515 05:36:48.316895 62364 analysis_predictor.cc:714] ======= optimize end =======
I0515 05:36:48.320442 62364 naive_executor.cc:98] ---  skip [feed], feed -> token_type_ids
I0515 05:36:48.320463 62364 naive_executor.cc:98] ---  skip [feed], feed -> input_ids
I0515 05:36:48.321842 62364 naive_executor.cc:98] ---  skip [linear_113.tmp_1], fetch -> fetch
[2022-05-15 05:36:49,316] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer'> to load 'ernie-3.0-medium-zh'.
[2022-05-15 05:36:49,317] [    INFO] - Already cached /vdb1/home/heliqi/.paddlenlp/models/ernie-3.0-medium-zh/ernie_3.0_medium_zh_vocab.txt
[OP Object] init success
```

#### 启动client测试
注意执行客户端请求时关闭代理，并根据实际情况修改init_client函数中的ip地址(启动服务所在的机器)
```
python seq_cls_rpc_client.py
```
输出打印如下:
```
{'label': array([6, 2]), 'confidence': array([0.5543532, 0.9495907], dtype=float32)}acc: 0.5745
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
I0515 05:36:48.316895 62364 analysis_predictor.cc:714] ======= optimize end =======
I0515 05:36:48.320442 62364 naive_executor.cc:98] ---  skip [feed], feed -> token_type_ids
I0515 05:36:48.320463 62364 naive_executor.cc:98] ---  skip [feed], feed -> input_ids
I0515 05:36:48.321842 62364 naive_executor.cc:98] ---  skip [linear_113.tmp_1], fetch -> fetch
[2022-05-15 05:36:49,316] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer'> to load 'ernie-3.0-medium-zh'.
[2022-05-15 05:36:49,317] [    INFO] - Already cached /vdb1/home/heliqi/.paddlenlp/models/ernie-3.0-medium-zh/ernie_3.0_medium_zh_vocab.txt
[OP Object] init success
```

#### 启动client测试
注意执行客户端请求时关闭代理，并根据实际情况修改init_client函数中的ip地址(启动服务所在的机器)
```
python token_cls_rpc_client.py
```
输出打印如下:
```
input data: 北京的涮肉，重庆的火锅，成都的小吃都是极具特色的美食。
The model detects all entities:
entity: 北京   label: LOC   pos: [0, 1]
entity: 重庆   label: LOC   pos: [6, 7]
entity: 成都   label: LOC   pos: [12, 13]
-----------------------------
input data: 原产玛雅故国的玉米，早已成为华夏大地主要粮食作物之一。
The model detects all entities:
entity: 玛雅   label: LOC   pos: [2, 3]
entity: 华夏   label: LOC   pos: [14, 15]
-----------------------------
PipelineClient::predict pack_data time:1652593013.713769
PipelineClient::predict before time:1652593013.7141528
input data: ['从', '首', '都', '利', '隆', '圭', '乘', '车', '向', '湖', '边', '小', '镇', '萨', '利', '马', '进', '发', '时', '，', '不', '到', '１', '０', '０', '公', '里', '的', '道', '路', '上', '坑', '坑', '洼', '洼', '，', '又', '逢', '阵', '雨', '迷', '蒙', '，', '令', '人', '不', '时', '发', '出', '路', '难', '行', '的', '慨', '叹', '。']
The model detects all entities:
entity: 利隆圭   label: LOC   pos: [3, 5]
entity: 萨利马   label: LOC   pos: [13, 15]
-----------------------------
```
