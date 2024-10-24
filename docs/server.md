# PaddleNLP SimpleSevring

PaddleNLP SimpleServing 是基于 unicorn 封装的模型部署服务化工具，该服务化工具具备灵活、易用的特性，可以简易部署预训练模型和预训练模型工具 Taskflow，PaddleNLP SimpleServing 具备以下两个特性：
  - 易用：一行代码即可部署预训练模型和预训练工具 Taskflow
  - 灵活：Handler 机制可以快速定制化服务化部署方式


## Tasflow 部署

Taskflow 是 PaddleNLP 预训练模型工具，具备开箱即用的特性，同时 Taskflow 可以支持加载微调后的模型，基于 Taskflow 的服务化方式可以进一步降低使用者的部署难度。PaddleNLP SimpleServing 基于这样的设计需求，设计了一套基于 Taskflow 的快速部署方式。下面从 server 搭建，client 发送请求来详细介绍使用方式。

### server 搭建

下面是 Taskflow 搭建服务的简易代码

```python
schema = ['出发地', '目的地', '费用', '时间']
uie = Taskflow("information_extraction", schema=schema)
app = SimpleServer()
app.register_taskflow('taskflow/uie', uie)
```
这里主要是使用 `SimpleServer` 服务类来注册 Taskflow Server，下面我们具体介绍一下 `register_taskflow` 相关参数

```text
def register_taskflow(
    task_name,
    task,
    taskflow_handler=None)

task_name(str)：
      服务化的名称，最终的服务化的URL: https://host:port/{task_name}
task(paddlenlp.Taskflow or list(paddlenlp.Taskflow)):
      Taskflow的实例对象，将想要注册的Taskflow任务注册进去，可以是多个Taskflow实例来支持多卡服务化
taskflow_handler(paddlenlp.server.BaseTaskflowHandler, 可选):
      Taskflow句柄处理类，可以自定义处理类来定制化Taskflow服务，默认为None，是默认的TaskflowHandler
```
### 多卡服务化(可选)
在机器环境里面如果有多卡，那就可以 register taskflow 服务化时，可以注册多个 Taskflow 实例，在服务化处理请求的过程中做了负载均衡，保证机器设备利用率充分利用，下面是具体的使用例子
```python
schema = ['出发地', '目的地', '费用', '时间']
uie1 = Taskflow("information_extraction", schema=schema, device_id=0)
uie2 = Taskflow("information_extraction", schema=schema, device_id=1)
app = SimpleServer()
app.register_taskflow('taskflow/uie', [uie1, uie2])
```
### 启动服务化
执行代码的即可启动服务
```
paddlenlp server server:app --host 0.0.0.0 --port 8189 --workers 1
```
服务化整体参数配置如下：
```text
--host: 启动服务化的IP地址，通常可以设置成 0.0.0.0
--port：启动服务化的网络端口
--workers: 接收服务化的进程数，默认为1
--log_level：服务化输出日志的级别，默认为 info 级别
--limit_concurrency：服务化能接受的并发数目，默认为None, 没有限制
--timeout_keep_alive：保持服务化连接的时间，默认为15s
--app_dir：服务化本地的路径，默认为服务化启动的位置
--reload: 当 app_dir的服务化相关配置和代码发生变化时，是否重启server，默认为False
```

### client 发送
```python
import requests
import json

url = "http://0.0.0.0:8189/taskflow/uie"
headers = {"Content-Type": "application/json"}
texts = ["城市内交通费7月5日金额114广州至佛山", "5月9日交通费29元从北苑到望京搜后"]
data = {
    "data": {
        "text": texts,
    }
}
r = requests.post(url=url, headers=headers, data=json.dumps(data))
datas = json.loads(r.text)
print(datas)
```
通过上述代码配置即可发送 POST 请求，同时注意在`data`这个 key 填入相关请求

同时可以支持定义 `schema` 传入到 client 请求中，可以快速切换 `schema`

```python
import requests
import json

url = "http://0.0.0.0:8189/taskflow/uie"
headers = {"Content-Type": "application/json"}
texts = ["城市内交通费7月5日金额114广州至佛山", "5月9日交通费29元从北苑到望京搜后"]
data = {
    "data": {
        "text": texts,
    },
    "parameters": {
        "schema": [] # 自定义schema
    }
}
r = requests.post(url=url, headers=headers, data=json.dumps(data))
datas = json.loads(r.text)
print(datas)
```

## 预训练模型部署
PaddleNLP SimpleServing 除了能支持 Taskflow 的服务化部署，也能支持预训练模型的部署，通过简单的配置即可加载预训练模型来进行服务化，同时在接口层面也能支持服务化的扩展，支持模型前后处理的定制化需求。

## server 搭建

下面是预训练模型的搭建的简易代码
```python
from paddlenlp import SimpleServer
from paddlenlp.server import CustomModelHandler, MultiClassificationPostHandler

app = SimpleServer()
app.register('cls_multi_class',
             model_path="./export",
             tokenizer_name='ernie-3.0-medium-zh',
             model_handler=CustomModelHandler,
             post_handler=MultiClassificationPostHandler)
```

这里主要是使用 `SimpleServer` 服务类来注册 Transformers Server，下面我们具体介绍一下 `register` 相关参数

```text
def register(task_name,
             model_path,
             tokenizer_name,
             model_handler,
             post_handler,
             precision='fp32',
             device_id=0)
task_name(str)：
      服务化的名称，最终的服务化的URL: https://host:port/{task_name}
model_path(str):
      需要部署的模型路径，这里的路径必须是动转静后的模型路径
model_handler(paddlenlp.server.BaseModelHandler):
      模型前置处理以及模型预测的Handler类别名字，这里可以继承 BaseModelHandler 自定义处理逻辑
 post_handler(paddlenlp.server.BasePostHandler):
      模型后置处理的Handler类别名字，这里可以继承 BasePostHandler 自定义处理逻辑
precision(str):
      模型的预测精度，默认为fp32；可选fp16，fp16的支持需要以下条件 1) **硬件**： V100、T4、A10、A100/GA100、Jetson AGX Xavier 、3080、3080、2080、2090 等显卡 2）**CUDA环境**：确保 CUDA >= 11.2，cuDNN >= 8.1.1 3) **安装依赖**：安装 onnx、 onnxruntime-gpu
device_id(int, list(int)):
       GPU设备，device_id默认为0，同时如果有多张显卡，可以设置成list,例如[0, 1]就可以支持多卡服务化；CPU设备，不用设置。
```
- BaseModelHandler 继承类：主要是 `CustomModelHandler`，该类的实现可以参考[链接](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/server/handlers/custom_model_handler.py), 绝大多数语义理解模型均可使用该继承类
- BasePostHandler 继承类：主要是文本分类 `MultiClassificationPostHandler`、`MultiLabelClassificationPostHandler` 来支持多分类、多标签分类，实现代码部分可以参考[链接](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/server/handlers/cls_post_handler.py)；`TokenClsModelHandler` 支持 序列标注任务，实现代码部分可以参考[链接](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/server/handlers/token_model_handler.py)

### 启动服务化
执行代码的即可启动服务
```
paddlenlp server server:app --host 0.0.0.0 --port 8189 --workers 1
```
服务化整体参数配置如下：
```text
--host: 启动服务化的IP地址，通常可以设置成 0.0.0.0
--port：启动服务化的网络端口，注意和已有网络端口冲突
--workers: 接收服务化的进程数，默认为1
--log_level：服务化输出日志的级别，默认为 info 级别
--limit_concurrency：服务化能接受的并发数目，默认为None, 没有限制
--timeout_keep_alive：保持服务化连接的时间，默认为15s
--app_dir：服务化本地的路径，默认为服务化启动的位置
--reload: 当 app_dir的服务化相关配置和代码发生变化时，是否重启server，默认为False
```

### 多卡服务化(可选)
在机器环境里面如果有多卡，通过简单设置 device_id 即可实现多卡服务化，保证机器设备利用率充分利用，下面是具体的使用例子
```python
from paddlenlp import SimpleServer
from paddlenlp.server import CustomModelHandler, MultiClassificationPostHandler

app = SimpleServer()
app.register('models/cls_multi_class',
             model_path="../../export",
             tokenizer_name='ernie-3.0-medium-zh',
             model_handler=CustomModelHandler,
             post_handler=MultiClassificationPostHandler，
             device_id=[0,1]) # device_id是0,1 两张卡
```
### client 发送
```python

import requests
import json

texts = [
        '黑苦荞茶的功效与作用及食用方法', '交界痣会凸起吗', '检查是否能怀孕挂什么科', '鱼油怎么吃咬破吃还是直接咽下去',
        '幼儿挑食的生理原因是'
    ]
    data = {
        'data': {
            'text': texts,
        },
        'parameters': {
            'max_seq_len': 128,
            'batch_size': 2
        }
    }
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    result_json = json.loads(r.text)
    print(result_json)
```
在 Client 发送请求的过程中可以一些参数来控制服务化处理逻辑，例如上面的 `max_seq_len`和 `batch_size` 均可以控制服务化处理时的序列长度和处理 batch_size 。

## 参考示例
- [UIE 服务化部署](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/uie/deploy/serving/simple_serving)
- [文本分类服务化部署](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_class/deploy/simple_serving)
- [预训练模型定制化 post_handler](https://github.com/PaddlePaddle/PaddleNLP/blob/release/2.8/model_zoo/ernie-health/cblue/deploy/serving/simple_serving)
