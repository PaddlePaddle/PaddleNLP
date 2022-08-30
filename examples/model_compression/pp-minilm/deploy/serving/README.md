# PP-MiniLM 使用 Paddle Serving 进行服务化部署

Paddle Serving 可以实现在服务器端部署推理模型，客户端远程通过 RPC/HTTP 方式发送数据进行推理，实现模型推理的服务化，下面以RPC方式为例进行说明。

## 前提条件
准备好 Inference 模型，需要2个文件：
| 文件                          | 说明                                   |
|-------------------------------|----------------------------------------|
| ppminilm.pdiparams      | 模型权重文件，供推理时加载使用            |
| ppminilm.pdmodel        | 模型结构文件，供推理时加载使用            |

假设这 2 个文件已生成，并放在在目录 `$MODEL_DIR` 下。

## 环境要求

使用 Paddle Serving 需要在服务器端安装相关模块，需要 v0.8.0 之后的版本：
```shell
pip install paddle-serving-app paddle-serving-client paddle-serving-server
```

如果服务器端可以使用GPU进行推理，则安装 server 的 gpu 版本，安装时要注意参考服务器当前 CUDA、TensorRT 的版本来安装对应的版本：[Serving readme](https://github.com/PaddlePaddle/Serving/tree/v0.8.0)

```shell
pip install paddle-serving-app paddle-serving-client paddle-serving-server-gpu
```

还需要在客户端安装相关模块，也需要 v0.8.0 之后的版本：
```shell
pip install paddle-serving-app paddle-serving-client
```

## 从 Inference 模型生成 Serving 模型和配置

以前提条件中准备好的 Inference 模型 `ppminilm.pdmodel`、`ppminilm.pdiparams` 为例：

```shell
python export_to_serving.py \
    --dirname  ${MODEL_DIR} \
    --model_filename ppminilm.pdmodel \
    --params_filename ppminilm.pdiparams \
    --server_path serving_server \
    --client_path serving_client \
    --fetch_alias_names logits \
```

其中参数释义如下：
- `dirname` : 表示 Inference 推理模型所在目录，这里是位于 `${MODEL_DIR}`。
- `model_filename` : 表示推理需要加载的模型结构文件。例如前提中得到的 `ppminilm.pdmodel`。如果设置为 `None` ，则使用 `__model__` 作为默认的文件名。
- `params_filename` : 表示推理需要加载的模型权重文件。例如前提中得到的 `ppminilm.pdiparams`。
- `server_path`: 转换后的模型文件和配置文件的存储路径。默认值为 serving_server。
- `client_path`: 转换后的客户端配置文件存储路径。默认值为 serving_client。
- `fetch_alias_names`: 模型输出的别名设置，比如输入的 input_ids 等，都可以指定成其他名字，默认不指定。
- `feed_alias_names`: 模型输入的别名设置，比如输出 pooled_out 等，都可以重新指定成其他名字，默认不指定。

执行命令后，会在当前目录下生成 2 个目录：serving_server 和 serving_client。serving_server 目录包含服务器端所需的模型和配置，需将其拷贝到服务器端；serving_client 目录包含客户端所需的配置，需将其拷贝到客户端。


## 配置 config 文件

在启动预测之前，需要按照自己的情况修改 config 文件中的配置，主要需要修改的配置释义如下：

- `rpc_port` : rpc端口。
- `device_type` : 0 代表 CPU, 1 代表 GPU, 2 代表 TensorRT, 3 代表 Arm CPU, 4 代表 Kunlun XPU。
- `devices` : 计算硬件 ID，当 devices 为 "" 或不写时，为 CPU 预测；当 devices 为"0"、 "0,1,2" 时为 GPU 预测。
- `fetch_list` : fetch 结果列表，以 client_config 中 fetch_var 的 alias_name 为准, 如果没有设置则全部返回。
- `model_config` : 模型路径。

## 启动 server

在服务器端容器中，使用上一步得到的 serving_server 目录启动 server：

```shell
python web_service.py

```

## 启动 client 发起推理请求
在客户端容器中，使用前面得到的 serving_client 目录启动 client 发起 RPC 推理请求。从命令行读取输入数据发起推理请求：

```shell
python rpc_client.py
```
