# PaddleNLP SimpleServing

PaddleNLP SimpleServing is a model deployment service tool based on unicorn, characterized by its flexibility and ease of use. It allows for the straightforward deployment of pre-trained models and the pre-trained model tool Taskflow. PaddleNLP SimpleServing has the following features:
  - Ease of Use: Deploy pre-trained models and the Taskflow tool with a single line of code.
  - Flexibility: The Handler mechanism allows for rapid customization of service deployment methods.

## Taskflow Deployment

Taskflow is a pre-trained model tool in PaddleNLP, offering out-of-the-box functionality. It also supports loading fine-tuned models. The service-oriented approach based on Taskflow can further reduce the deployment difficulty for users. PaddleNLP SimpleServing is designed to meet these needs, providing a rapid deployment method based on Taskflow. Below, we detail the usage from server setup to client request sending.

### Server Setup

Below is a simple code example for setting up a Taskflow service:

```python
schema = ['出发地', '目的地', '费用', '时间']
uie = Taskflow("information_extraction", schema=schema)
app = SimpleServer()
app.register_taskflow('taskflow/uie', uie)
```

Here, the `SimpleServer` service class is primarily used to register the Taskflow Server. We will now introduce the parameters related to `register_taskflow`:

```text
def register_taskflow(
    task_name,
    task,
    taskflow_handler=None)

task_name(str):
      The name of the service, with the final service URL being: https://host:port/{task_name}
task(paddlenlp.Taskflow or list(paddlenlp.Taskflow)):
      The Taskflow instance object. Register the Taskflow tasks you want to register, which can be multiple Taskflow instances to support multi-card service.
taskflow_handler(paddlenlp.server.BaseTaskflowHandler, optional):
      The Taskflow handler processing class. You can customize the processing class to tailor the Taskflow service. The default is None, which uses the default TaskflowHandler.
```

### Multi-Card Service (Optional)

If the machine environment has multiple cards, you can register multiple Taskflow instances when registering the taskflow service. During the service processing of requests, load balancing is performed to ensure full utilization of machine resources. Below is a specific usage example:
```python
schema = ['出发地', '目的地', '费用', '时间']
uie1 = Taskflow("information_extraction", schema=schema, device_id=0)
uie2 = Taskflow("information_extraction", schema=schema, device_id=1)
app = SimpleServer()
app.register_taskflow('taskflow/uie', [uie1, uie2])
```
### Start the Service
Execute the code to start the service
```
paddlenlp server server:app --host 0.0.0.0 --port 8189 --workers 1
```
The overall parameter configuration for the service is as follows:
```text
--host: The IP address to start the service, usually set to 0.0.0.0
--port: The network port to start the service
--workers: The number of processes to receive the service, default is 1
--log_level: The level of logs output by the service, default is info level
--limit_concurrency: The number of concurrent requests the service can accept, default is None, no limit
--timeout_keep_alive: The time to keep the service connection alive, default is 15s
--app_dir: The local path of the service, default is the location where the service is started
--reload: Whether to restart the server when the service-related configuration and code in app_dir change, default is False
```

### Client Sending
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
By configuring the above code, you can send a POST request. Make sure to fill in the relevant request under the `data` key.

You can also define and pass a `schema` in the client request to quickly switch the `schema`.
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
        "schema": [] # Custom schema
    }
}
r = requests.post(url=url, headers=headers, data=json.dumps(data))
datas = json.loads(r.text)
print(datas)
```

## Pre-trained Model Deployment
PaddleNLP SimpleServing not only supports the service deployment of Taskflow but also supports the deployment of pre-trained models. By simple configuration, pre-trained models can be loaded for service deployment. Additionally, at the interface level, it supports service extension and customization needs for model pre- and post-processing.

## Server Setup

Below is the simplified code for setting up a pre-trained model
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

Here, the `SimpleServer` service class is primarily used to register the Transformers Server. Below, we specifically introduce the parameters related to `register`.
```text
def register(task_name,
             model_path,
             tokenizer_name,
             model_handler,
             post_handler,
             precision='fp32',
             device_id=0)
task_name(str):
      The name for the service, which will be used in the final service URL: https://host:port/{task_name}
model_path(str):
      The path to the model that needs to be deployed. This path must be the model path after dynamic to static conversion.
model_handler(paddlenlp.server.BaseModelHandler):
      The class name of the handler for model preprocessing and prediction. You can inherit from BaseModelHandler to customize processing logic.
post_handler(paddlenlp.server.BasePostHandler):
      The class name of the handler for model post-processing. You can inherit from BasePostHandler to customize processing logic.
precision(str):
      The precision for model prediction, default is fp32; options include fp16. Support for fp16 requires the following conditions: 1) **Hardware**: Graphics cards such as V100, T4, A10, A100/GA100, Jetson AGX Xavier, 3080, 2080, 2090, etc. 2) **CUDA Environment**: Ensure CUDA >= 11.2, cuDNN >= 8.1.1 3) **Dependencies**: Install onnx, onnxruntime-gpu
device_id(int, list(int)):
       GPU device, default device_id is 0. If there are multiple GPUs, you can set it as a list, e.g., [0, 1] to support multi-GPU service deployment; for CPU devices, no setting is needed.
```
- BaseModelHandler Inherited Class: Mainly `CustomModelHandler`. The implementation of this class can be referenced [here](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/server/handlers/custom_model_handler.py). Most semantic understanding models can use this inherited class.
- BasePostHandler Inherited Class: Mainly `MultiClassificationPostHandler` and `MultiLabelClassificationPostHandler` for supporting multi-class and multi-label classification. The implementation can be referenced [here](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/server/handlers/cls_post_handler.py); `TokenClsModelHandler` supports sequence labeling tasks, with implementation details available [here](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/server/handlers/token_model_handler.py).

### Start the Service
Execute the code to start the service.
```
```
paddlenlp server server:app --host 0.0.0.0 --port 8189 --workers 1
```
The overall parameter configuration for service deployment is as follows:
```text
--host: The IP address for starting the service, usually set to 0.0.0.0
--port: The network port for starting the service, ensure it does not conflict with existing network ports
--workers: The number of processes to handle the service, default is 1
--log_level: The log level for service output, default is info level
--limit_concurrency: The number of concurrent requests the service can handle, default is None, meaning no limit
--timeout_keep_alive: The time to keep the service connection alive, default is 15s
--app_dir: The local path for the service, default is the location where the service is started
--reload: Whether to restart the server when the service-related configuration and code in app_dir change, default is False
```

### Multi-GPU Service Deployment (Optional)
If multiple GPUs are available in the machine environment, multi-GPU service deployment can be achieved by simply setting the `device_id`, ensuring full utilization of machine resources. Below is a specific usage example:
```python
from paddlenlp import SimpleServer
from paddlenlp.server import CustomModelHandler, MultiClassificationPostHandler

app = SimpleServer()
app.register('models/cls_multi_class',
             model_path="../../export",
             tokenizer_name='ernie-3.0-medium-zh',
             model_handler=CustomModelHandler,
             post_handler=MultiClassificationPostHandler,
             device_id=[0,1]) # device_id is 0,1 for two GPUs
```
### Client Request
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
During the client request process, certain parameters can be used to control the service processing logic, such as `max_seq_len` and `batch_size` in the example above.
Both the sequence length and the batch_size can be controlled during service processing.

## Reference Examples
- [UIE Service Deployment](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/uie/deploy/serving/simple_serving)
- [Text Classification Service Deployment](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/applications/text_classification/multi_class/deploy/simple_serving)
- [Pre-trained Model Custom Post_handler](https://github.com/PaddlePaddle/PaddleNLP/blob/release/2.8/model_zoo/ernie-health/cblue/deploy/serving/simple_serving)
