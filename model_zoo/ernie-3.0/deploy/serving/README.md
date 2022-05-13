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
如果是链接中下载的部署模型或训练导出的静态图推理模型(含`xx.pdmodel`和`xx.pdiparams`)，需要转换成serving模型
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
# 修改模型目录为下载的模型目录或自己的模型目录:
model_config: no_task_emb/serving_server =>  model_config: erine-3.0-tiny/serving_server

# 修改rpc端口号为9998
rpc_port: 9998   =>   rpc_port: 9998

# 修改使用GPU推理为使用CPU推理:
device_type: 1    =>   device_type: 0
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
输出打印如下:
```
{'label': array([6, 2]), 'confidence': array([4.9473147, 5.7493963], dtype=float32)}
acc: 0.5745
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
输出打印如下:
```
input data: 古老的文明，使我们引以为豪，彼此钦佩。
The model detects all entities:
-----------------------------
input data: 原产玛雅故国的玉米，早已成为华夏大地主要粮食作物之一。
The model detects all entities:
entity: 玛雅   label: LOC   pos: [2, 3]
entity: 华夏   label: LOC   pos: [14, 15]
-----------------------------
input data: ['从', '首', '都', '利', '隆', '圭', '乘', '车', '向', '湖', '边', '小', '镇', '萨', '利', '马', '进', '发', '时', '，', '不', '到', '１', '０', '０', '公', '里', '的', '道', '路', '上', '坑', '坑', '洼', '洼', '，', '又', '逢', '阵', '雨', '迷', '蒙', '，', '令', '人', '不', '时', '发', '出', '路', '难', '行', '的', '慨', '叹', '。']
The model detects all entities:
entity: 利隆圭   label: LOC   pos: [3, 5]
entity: 萨利马   label: LOC   pos: [13, 15]
-----------------------------
```
