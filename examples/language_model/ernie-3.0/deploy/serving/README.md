# ERNIE-3.0服务化部署

## 环境准备

### 安装Paddle Serving，
安装指令如下，更多wheel包请参考[serving官网文档](https://github.com/PaddlePaddle/Serving/blob/develop/doc/Latest_Packages_CN.md)
```
pip install paddle_serving_app paddle_serving_clinet

#CPU server
pip install paddle_serving_server

#GPU server, 需要确认环境再选择执行哪一条:
# CUDA10.2 + Cudnn7 + TensorRT6
pip install paddle-serving-server-gpu==0.8.3.post102 -i https://pypi.tuna.tsinghua.edu.cn/simple 
# CUDA10.1 + TensorRT6
pip install paddle-serving-server-gpu==0.8.3.post101 -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA11.2 + TensorRT8
pip install paddle-serving-server-gpu==0.8.3.post112 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

默认开启国内清华镜像源来加速下载，如果您使用 HTTP 代理可以关闭(-i https://pypi.tuna.tsinghua.edu.cn/simple)


### 安装其他依赖
当您使用`paddle_serving_client.convert`命令或者`Python Pipeline`框架时需要安装`Paddle`库,更多CUDA版本的库可参考[Paddle官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)
```
# CPU 环境请执行
pip3 install paddlepaddle

# GPU CUDA 环境(默认CUDA10.2)
pip3 install paddlepaddle-gpu
```


安装paddlenlp
```
pip install paddlenlp
```


下载[faster_tokenizer](http://10.12.121.132:8000/gitlab/tokenizers/build/dist/faster_tokenizers-0.0.1-py3-none-any.whl)并安装, 也可先手动下载wheel包再执行pip指令安装
```
wget http://10.12.121.132:8000/gitlab/tokenizers/build/dist/faster_tokenizers-0.0.1-py3-none-any.whl
pip install faster_tokenizers-0.0.1-py3-none-any.whl
```


## 服务化部署模型
### 准备模型和数据
下载[Erine-3.0模型](TODO)

### 修改配置文件
目录中的`config.yml`文件解释了每一个参数的含义，可以根据实际需要修改其中的配置。比如：
```
#修改模型目录为下载的模型目录或自己的模型目录:
model_config: no_task_emb/serving_server =>  model_config: erine-3.0-tiny

#修改rpc端口号为9998
rpc_port: 9998   =>   rpc_port: 9998

#修改处理客户端请求的最大并发进程数为6:
worker_num: 4    =>   worker_num: 6
```

### 启动服务
#TODO： 提供参数启动不同任务的服务，当前默认分类任务
```
python web_service.py
```

### 启动client测试
```
python rpc_client.py
```
