# **ELECTRA 使用Paddle Serving API进行推理**
Paddle Serving 可以实现在服务器端部署推理模型，客户端远程通过RPC/HTTP方式发送数据进行推理，实现模型推理的服务化，下面以RPC方式为例进行说明。

## 前提条件
准备好Inference所需模型，需要2个文件：
| 文件                          | 说明                                   |
|-------------------------------|----------------------------------------|
| electra-deploy.pdiparams      | 模型权重文件，供推理时加载使用            |
| electra-deploy.pdmodel        | 模型结构文件，供推理时加载使用            |

如何获得Inference模型？[可参考文档“导出推理模型”一节](../../README.md)，下面假设这2个文件已生成，并放在在当前目录下

## 在服务器端和客户端启动Serving的docker容器
建议在docker容器中运行服务器端和客户端以避免一些系统依赖库问题，启动docker镜像的命令参考：[Serving readme](https://github.com/PaddlePaddle/Serving/tree/v0.5.0)

## 在服务器端安装相关模块
```shell
pip install paddle-serving-app paddle-serving-client paddle-serving-server paddlepaddle
```
如果服务器端可以使用GPU进行推理，则安装server的gpu版本，安装时要注意参考服务器当前CUDA、TensorRT的版本来安装对应的版本：[Serving readme](https://github.com/PaddlePaddle/Serving/tree/v0.5.0)
```shell
pip install paddle-serving-app paddle-serving-client paddle-serving-server-gpu paddlepaddle-gpu
```

## 在客户端安装相关模块
```shell
pip install paddle-serving-app paddle-serving-client
```

## 从Inference模型生成Serving的模型和配置
以前提条件中准备好的Inference模型 electra-deploy.pdmodel/electra-deploy.pdiparams 为例：
```shell
python -u ./covert_inference_model_to_serving.py \
    --inference_model_dir ./ \
    --model_file ./electra-deploy.pdmodel \
    --params_file ./electra-deploy.pdiparams
```
其中参数释义如下：
- `inference_model_dir` 表示Inference推理模型所在目录，这里假设为当前目录。
- `model_file` 表示推理需要加载的模型结构文件。例如前提中得到的electra-deploy.pdmodel。
- `params_file` 表示推理需要加载的模型权重文件。例如前提中得到的electra-deploy.pdiparams。

执行命令后，会在当前目录下生成2个目录：serving_server 和 serving_client。serving_server目录包含服务器端所需的模型和配置，需将其cp到服务器端容器中；serving_client目录包含客户端所需的配置，需将其cp到客户端容器中

## 启动server
在服务器端容器中，使用上一步得到的serving_server目录启动server
```shell
python -m paddle_serving_server_gpu.serve \
    --model ./serving_server \
    --port 8383
```
其中参数释义如下：
- `model` 表示server加载的模型和配置所在目录。
- `port` 表示server开启的服务端口8383。

如果服务器端可以使用GPU进行推理计算，则启动服务器时可以配置server使用的GPU id
```shell
python -m paddle_serving_server_gpu.serve \
    --model ./serving_server \
    --port 8383 \
    --gpu_id 0
```
- `gpu_id` 表示server使用0号GPU。

## 启动client进行推理
在客户端容器中，使用前面得到的serving_client目录启动client发起RPC推理请求。和使用Paddle Inference API进行推理一样，有如下两种方法:
### 从命令行读取输入数据发起推理请求
```shell
python -u ./client.py \
    --client_config_file ./serving_client/serving_client_conf.prototxt \
    --server_ip_port 127.0.0.1:8383 \
    --predict_sentences "uneasy mishmash of styles and genres ." "director rob marshall went out gunning to make a great one ." \
    --batch_size 2 \
    --max_seq_length 128 \
    --model_name electra-small
```
其中参数释义如下：
- `client_config_file` 表示客户端需要加载的配置文件。
- `server_ip_port` 表示服务器端的ip和port。默认为127.0.0.1:8383。
- `predict_sentences` 表示用于推理的（句子）数据，可以配置1条或多条。如果此项配置，则predict_file不用配置。
- `batch_size` 表示每次推理的样本数目。
- `max_seq_length` 表示输入的最大句子长度，超过该长度将被截断。
- `model_name` 表示推理模型的类型，当前支持electra-small（约1400万参数）、electra-base（约1.1亿参数）、electra-large（约3.35亿参数）。

### 从文件读取输入数据发起推理请求
```shell
python -u ./client.py \
    --client_config_file ./serving_client/serving_client_conf.prototxt \
    --server_ip_port 127.0.0.1:8383 \
    --predict_file "./sst-2.test.tsv.1" "./sst-2.test.tsv.2" \
    --batch_size 2 \
    --max_seq_length 128 \
    --model_name electra-small
```
其中绝大部分和从命令行读取输入数据一样，这里描述不一样的参数：
- `predict_file` 表示用于推理的文件数据，可以配置1个或多个文件，每个文件和预训练数据格式一样，为utf-8编码的文本数据，每行1句文本。如果此项配置，则predict_sentences不用配置。

使用Paddle Serving API进行推理的结果和使用Inference API的结果是一样的：
```shell
===== batch 0 =====
Input sentence is : [CLS] uneasy mishmash of styles and genres . [SEP]
Output data is : 0
Input sentence is : [CLS] director rob marshall went out gunning to make a great one . [SEP]
Output data is : 1
inference total 2 sentences done, total time : 4.729415416717529 s
```
此推理结果表示：第1句话是负向情感，第2句话是正向情感。
