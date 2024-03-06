简体中文

# m3e 服务化部署示例

在服务化部署前，需确认

- 1. 服务化镜像的软硬件环境要求和镜像拉取命令请参考[FastDeploy服务化部署](https://github.com/PaddlePaddle/FastDeploy/tree/develop/serving)
- 2. FastDeploy Server镜像支持的推理后端有OpenVINO、TensorRT、Paddle Inference和ONNX Runtime，用户可以根据自己的软硬件条件进行选择。


## 准备模型

下载m3e-base模型(如果有已训练好的模型，跳过此步骤):
```bash
# 下载并解压m3e-base模型
wget https://paddlenlp.bj.bcebos.com/pipelines/m3e.tar.gz
tar -xzvf m3e.tar.gz
```

模型下载移动好之后，m3e的models目录结构如下:
```
models
├── m3e                      # pipeline
│   ├── 1
│   └── config.pbtxt         # 通过这个文件组合前后处理和模型推理
├── m3e_model                # 模型推理
│   ├── 1
│   │   ├── model.pdmodel
|   │   └── model.pdiparams
│   └── config.pbtxt
├── m3e_postprocess          # 后处理
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
└── m3e_tokenizer            # 预处理分词
    ├── 1
    │   └── model.py
    └── config.pbtxt
```
*注意*:如果使用TensorRT引擎，用 pipelines/examples/chatbot/config.pbtxt 替换 models/m3e_model/config.pbtxt
*BAAI/bge-small-zh-v1.5配置链接https://paddlenlp.bj.bcebos.com/pipelines/bge-small-zh.tar.gz
## 拉取并运行镜像

以GPU镜像部署为例：

```bash
# GPU镜像
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:1.0.7-gpu-cuda11.4-trt8.5-21.10

# 运行
nvidia-docker run  -it --net=host --name fastdeploy_server --shm-size="1g" -v /path/m3e/models:/models registry.baidubce.com/paddlepaddle/fastdeploy:1.0.7-gpu-cuda11.4-trt8.5-21.10 bash
```

## 部署模型

serving目录包含启动pipeline服务的配置和发送预测请求的代码，包括：

```
models                    # 服务化启动需要的模型仓库，包含模型和服务配置文件
```

*注意*:启动服务时，Server的每个python后端进程默认申请`64M`内存，默认启动的docker无法启动多个python后端节点。有两个解决方案：
- 1.启动容器时设置`shm-size`参数, 比如:`docker run  -it --net=host --name fastdeploy_server --shm-size="1g" -v /path/m3e/models:/models registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10 bash`
- 2.启动服务时设置python后端的`shm-default-byte-size`参数, 设置python后端的默认内存为10M： `tritonserver --model-repository=/models --backend-config=python,shm-default-byte-size=10485760`

### m3e_embedding任务
在容器内执行下面命令启动服务:
```
# 可通过参数只启动m3e任务
 fastdeployserver --model-repository=/models --model-control-mode=explicit --load-model=m3e --http-port=8082
```
输出打印如下:
```
...
I0804 04:14:54.478441 16055 server.cc:592]
+-------------------+---------+--------+
| Model             | Version | Status |
+-------------------+---------+--------+
| m3e               | 1       | READY  |
| m3e_model         | 1       | READY  |
| me3_postprocess   | 1       | READY  |
| m3e_tokenizer     | 1       | READY  |
+-------------------+---------+--------+
...

```
## 客户端请求
客户端请求可以在本地执行脚本请求；也可以在容器中执行。

本地执行脚本需要先安装依赖:
```
pip install grpcio
pip install tritonclient[all]

# 如果bash无法识别括号，可以使用如下指令安装:
pip install tritonclient\[all\]
```

### 示例m3eRetriever
注意执行客户端请求时关闭代理，并根据实际情况修改m3eRetriever中的url地址(启动服务所在的机器)
```
python examples/chatbot/ParallelRetriever_example.py \
                                            --file_paths ... \
                                            --api_key ... \
                                            --secret_key ... \
```
参数含义说明
* `file_paths`: 文件的路径
* `api_key`: 文心一言的apk key
* `secret_key`: 文心一言的secret key

## 配置修改

当前m3e默认配置在GPU上运行Paddle引擎; 3个GPU各部署一个实例，假设需要修改配置，详情请参考[配置文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/model_configuration.md#cpugpu%E5%92%8C%E5%AE%9E%E4%BE%8B%E4%B8%AA%E6%95%B0%E9%85%8D%E7%BD%AE)
