# 基于Paddle Serving的服务化部署

本文档将介绍如何使用[Paddle Serving](https://github.com/PaddlePaddle/Serving/blob/develop/README_CN.md)工具部署基于ERNIE 3.0的分类部署pipeline在线服务。

## 目录
- [环境准备](#环境准备)
- [模型转换](#模型转换)
- [部署模型](#部署模型)

## 环境准备
需要[准备PaddleNLP的运行环境]()和Paddle Serving的运行环境。

### 安装Paddle Serving
安装client和serving app，用于向服务发送请求:
```
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

用已安装的paddle_serving_client将静态图参数模型转换成serving格式。如何使用[静态图导出脚本](../../export_model.py)将训练后的模型转为静态图模型详见[模型静态图导出](../../README.md),模型地址--dirname根据实际填写即可。

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
├──config.yml        # 分类任务启动服务端的配置文件
├──rpc_client.py     # 分类任务发送pipeline预测请求的脚本
└──service.py        # 分类任务启动服务端的脚本
```

### 修改配置文件
目录中的`config.yml`文件解释了每一个参数的含义，可以根据实际需要修改其中的配置。比如：
```
# 修改模型目录为下载的模型目录或自己的模型目录:
model_config: serving_server =>  model_config: erine-3.0-tiny/serving_server

# 修改rpc端口号为9998:
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
I0625 16:44:36.563802 40218 analysis_predictor.cc:1007] ======= optimize end =======
I0625 16:44:36.571702 40218 naive_executor.cc:102] ---  skip [feed], feed -> token_type_ids
I0625 16:44:36.571728 40218 naive_executor.cc:102] ---  skip [feed], feed -> input_ids
I0625 16:44:36.574352 40218 naive_executor.cc:102] ---  skip [linear_147.tmp_1], fetch -> fetch
[2022-06-25 16:44:37,545] [ WARNING] - Can't find the faster_tokenizers package, please ensure install faster_tokenizers correctly. You can install faster_tokenizers by `pip install faster_tokenizers`(Currently only work for linux platform).
[2022-06-25 16:44:37,546] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer'> to load 'ernie-3.0-base-zh'.
[2022-06-25 16:44:37,546] [    INFO] - Already cached /root/.paddlenlp/models/ernie-3.0-base-zh/ernie_3.0_base_zh_vocab.txt
[OP Object] init success
W0625 16:45:40.312942 40218 gpu_context.cc:278] Please NOTE: device: 3, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.2
W0625 16:45:40.316538 40218 gpu_context.cc:306] device: 3, cuDNN Version: 8.1.
```

#### 启动client测试
注意执行客户端请求时关闭代理，并根据实际情况修改server_url地址(启动服务所在的机器)
```shell
python rpc_client.py
```
输出打印如下:
```
data:  经审理查明，2012年4月5日19时许，被告人王某在杭州市下城区朝晖路农贸市场门口贩卖盗版光碟、淫秽光碟时被民警当场抓获，并当场查获其贩卖的各类光碟5515张，其中5280张某属非法出版物、235张某属淫秽物品。上述事实，被告人王某在庭审中亦无异议，且有经庭审举证、质证的扣押物品清单、赃物照片、公安行政处罚决定书、抓获经过及户籍证明等书证；证人胡某、徐某的证言；出版物鉴定书、淫秽物品审查鉴定书及检查笔录等证据证实，足以认定。
label:  32,158,187
--------------------
data:  榆林市榆阳区人民检察院指控：2015年11月22日2时许，被告人王某某在自己经营的榆阳区长城福源招待所内，介绍并容留杨某向刘某某、白某向乔某某提供性服务各一次。
label:  26,27
--------------------
data:  静乐县人民检察院指控，2014年8月30日15时许，静乐县苏坊村村民张某某因占地问题去苏坊村半切沟静静铁路第五标施工地点阻拦施工时，遭被告人王某某阻止，张某某打电话叫来儿子李某某，李某某看到张某某躺在地上，打了王某某一耳光。于是王某某指使工人殴打李某某，致李某某受伤。经忻州市公安司法鉴定中心鉴定，李某某的损伤评定为轻伤一级。李某某被打伤后，被告人王某某为逃避法律追究，找到任某某，指使任某某作实施××的伪证，并承诺每月给1万元。同时王某某指使工人王某甲、韩某某去丰润派出所作由任某某打伤李某某的伪证，导致任某某被静乐县公安局以涉嫌××罪刑事拘留。公诉机关认为，被告人王某某的行为触犯了《中华人民共和国刑法》××、《中华人民共和国刑法》××××之规定，应以××罪和××罪追究其刑事责任，数罪并罚。
label:  0,61
--------------------
```
