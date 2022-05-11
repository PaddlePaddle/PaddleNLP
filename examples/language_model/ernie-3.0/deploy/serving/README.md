# ERNIE-3.0服务化部署

## 环境准备

### 安装Paddle Serving，
安装指令如下，更多wheel包请参考[serving官网文档](https://github.com/PaddlePaddle/Serving/blob/develop/doc/Latest_Packages_CN.md)
```
pip install paddle_serving_app paddle_serving_clinet

# CPU server
pip install paddle_serving_server

# GPU server, 需要确认环境再选择执行哪一条:
# CUDA10.2 + Cudnn7 + TensorRT6
pip install paddle-serving-server-gpu==0.8.3.post102 -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA10.1 + TensorRT6
pip install paddle-serving-server-gpu==0.8.3.post101 -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA11.2 + TensorRT8
pip install paddle-serving-server-gpu==0.8.3.post112 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

默认开启国内清华镜像源来加速下载，如果您使用 HTTP 代理可以关闭(-i https://pypi.tuna.tsinghua.edu.cn/simple)


### 安装Paddle库
更多Paddle库下载安装可参考[Paddle官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)
```
# CPU 环境请执行
pip3 install paddlepaddle

# GPU CUDA 环境(默认CUDA10.2)
pip3 install paddlepaddle-gpu
```

## 准备模型和数据
下载[Erine-3.0模型](TODO)

### 转换模型
如果是上述下载或者训练导出的静态图推理模型(含`xx.pdmodel`和‘xx.pdiparams’)，需要转换成serving模型
```
# 模型地址根据实际填写即可
python -m paddle_serving_client.convert --dirname models/erinie-3.0 --model_filename infer.pdmodel --params_filename infer.pdiparams

# 可通过指令查看参数含义
python -m paddle_serving_client.convert --help
```
转换成功后的目录如下:
```
serving_server
├── infer.pdiparams
├── infer.pdmodel
├── serving_server_conf.prototxt
└── serving_server_conf.stream.prototxt
```


## 服务化部署模型
### 修改配置文件
目录中的`xx_config.yml`文件解释了每一个参数的含义，可以根据实际需要修改其中的配置。比如：
```
#修改模型目录为下载的模型目录或自己的模型目录:
model_config: no_task_emb/serving_server =>  model_config: erine-3.0-tiny

#修改rpc端口号为9998
rpc_port: 9998   =>   rpc_port: 9998

#修改处理客户端请求的最大并发进程数为6:
worker_num: 4    =>   worker_num: 6
```

### 分类任务
#### 启动服务
修改好配置文件后，执行下面指令启动服务:
```
python seq_cls_service.py
```

#### 启动client测试
注意执行客户端请求时关闭代理，并根据实际情况修改init_client函数中的ip地址(启动服务所在的机器)
```
python seq_cls_rpc_client.py
```

### 实体识别任务
#### 启动服务
修改好配置文件后，执行下面指令启动服务:
```
python token_cls_service.py
```

#### 启动client测试
注意执行客户端请求时关闭代理，并根据实际情况修改init_client函数中的ip地址(启动服务所在的机器)
```
python seq_cls_rpc_client.py
```
会输出以下打印:
```
input data: 在过去的五年中，致公党在邓小平理论指引下，遵循社会主义初级阶段的基本路线，努力实践致公党十大提出的发挥参政党职能、加强自身建设的基本任务。
The model detects all entities:
entity: 公党在   label: ORG   pos: [9, 11]
entity: 小平理   label: PER   pos: [13, 15]
entity: 公党十大提   label: ORG   pos: [42, 46]
-----------------------------
input data: 今年７月１日我国政府恢复对香港行使主权，标志着“一国两制”构想的巨大成功，标志着中国人民在祖国统一大业的道路上迈出了重要的一步。
The model detects all entities:
entity: 港行   label: LOC   pos: [14, 15]
entity: 国人   label: LOC   pos: [41, 42]
-----------------------------
input data: ['中', '共', '中', '央', '致', '中', '国', '致', '公', '党', '十', '一', '大', '的', '贺', '词', '各', '位', '代', '表', '、', '各', '位', '同', '志', '：', '在', '中', '国', '致', '公', '党', '第', '十', '一', '次', '全', '国', '代', '表', '大', '会', '隆', '重', '召', '开', '之', '际', '，', '中', '国', '共', '产', '党', '中', '央', '委', '员', '会', '谨', '向', '大', '会', '表', '示', '热', '烈', '的', '祝', '贺', '，', '向', '致', '公', '党', '的', '同', '志', '们', '致', '以', '亲', '切', '的', '问', '候', '！']
The model detects all entities:
entity: 共中央致   label: ORG   pos: [1, 4]
entity: 国致公党十一大的   label: ORG   pos: [6, 13]
entity: 国致公党第十一次全国代表大会隆   label: ORG   pos: [28, 42]
entity: 国共产党中央委员会谨   label: ORG   pos: [50, 59]
entity: 公   label: ORG   pos: [73, 73]
-----------------------------
```
