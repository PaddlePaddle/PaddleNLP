# Docker 启动 pipelines 服务

## 1. Docker 一键启动

### 1.1 Docker 本地构建镜像启动

为了满足用户多种多样的需求，我们提供了 Dockerfile 来构建一个镜像(以语义索引为例)，其中`Docker`的安装请参考[官方文档](https://docs.docker.com/desktop/),安装完以后，修改服务端运行脚本`run_server.sh`，客户端界面脚本`run_client.sh`，然后执行下面的命令：

```
cd docker
# CPU
docker build --tag=pipeline_cpu_server . -f Dockerfile
# GPU
docker build --tag=pipeline_server . -f Dockerfile-GPU
```
构建完以后就可以运行，先启动`elastic search`，然后再启动刚刚打包好的镜像：

```
docker network create elastic
docker run \
      -d \
      --name es02 \
      --net elastic \
      -p 9200:9200 \
      -e discovery.type=single-node \
      -e ES_JAVA_OPTS="-Xms256m -Xmx256m"\
      -e xpack.security.enabled=false \
      -e cluster.routing.allocation.disk.threshold_enabled=false \
      -it \
      docker.elastic.co/elasticsearch/elasticsearch:8.3.3
# cpu
docker run -d --name paddlenlp_pipelines --net host -it pipeline_cpu_server
# gpu
nvidia-docker run -d --name paddlenlp_pipelines --net host -it pipeline_server
```

cpu 版本大概等待3分钟左右，gpu 版本大概1分钟左右，到这里您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验语义检索系统服务了。

## 2. Docker-Compose 一键启动

为了方便用户很方便的切换，并部署`Pipelines`的语义索引和问答应用，我们特地提供了`Docker-Compose`的方式，`Docker Compose`的安装请参考[Docker Compose](https://docs.docker.com/compose/)，用户只需要修改`docker-compose.yml`的配置，然后一键启动即可。

下面以语义检索的例子进行展示：

```
cd docker
# 启动cpu容器
docker-compose up -d
# 关闭cpu容器
docker-compose stop
# 查看容器运行的日志 cpu
docker logs pip01

# 启动 gpu 容器
docker-compose -f docker-compose-gpu.yml up -d
# 关闭 gpu 容器
docker-compose -f docker-compose-gpu.yml stop
# 查看容器运行的日志 gpu
docker logs pip02
```
构建过程一般会持续3分钟左右，然后 cpu 版本启动等待1分钟左右，然后您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验语义检索系统服务了。

## 3. Docker 编译一个定制化 CUDA 版本的 Pipelines 的镜像

Docker 编译一个定制化 CUDA 版本的 Pipelines 的镜像流程分2步，第一步是利用 Paddle 镜像构建 Pipelines 基础镜像，第二步是构建一键启动镜像。第一步构建的镜像是一个可用的状态，但是启动后，需要进入容器，手工启动服务，第二步是需要把运行命令打包到镜像中，使得 Docker 启动的时候能够自动启动 Pipelines 的服务。

### 3.1 基础镜像

以 CUDA 11.2环境为例，编译一个 Pipelines 基础镜像流程如下：

```
nvidia-docker run --name pipelines --net host --shm-size 4g -it registry.baidubce.com/paddlepaddle/paddle:2.3.2-gpu-cuda11.2-cudnn8 /bin/bash
cd /root
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/pipelines/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py install
apt-get install lsof
```
镜像构建完成可以使用`Ctrl+P+Q`组合键跳出容器。

在第一步构建镜像的过程中，如果是 CUDA 的其他版本，则需要在[Paddle 官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)上查找是否有对应的 CUDA 版本的 Paddle 镜像，如果没有，则需要自己手工构建一个该 CUDA 版本的 Docker，然后安装对应 CUDA 版本的 PaddlePaddle，然后继续执行上面的流程。

### 3.2 一键启动镜像

到了上一步就构建了一个可用的 Pipelines 镜像了，但是这个镜像还没有一键启动功能，即需要进入容器手动启动后台和前端。这里进一步打包镜像，把启动运行的命令也打包到镜像中，执行过程如下：

```
docker commit pipelines pipelines:1.0-gpu-cuda11.2-cudnn8
docker tag pipelines:1.0-gpu-cuda11.2-cudnn8  paddlepaddle/paddlenlp:pipelines-1.0-gpu-cuda11.2-cudnn8
# 在容器外下载一份PaddleNLP代码
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/pipelines/docker
```
修改`Dockerfile-GPU`文件，更换基础镜像，并添加一键运行命令：

```
FROM paddlepaddle/paddlenlp:pipelines-1.0-gpu-cuda11.2-cudnn8
# 使得Docker容器启动start.sh，并且保持运行
ENTRYPOINT /root/start.sh && tail -f /dev/null
```
然后执行：

```
# Dockerfile-GPU 包含一键启动的命令
docker build --tag=paddlepaddle/paddlenlp:2.4.0-gpu-cuda11.2-cudnn8 . -f Dockerfile-GPU
```

这样就构建了一键启动的 Docker 镜像。

### 3.3 启动镜像

一键启动的 Docker 构建完成以后就可以使用下面的命令启动：

```
nvidia-docker run -d --name paddlenlp_pipelines_gpu --net host -ti paddlepaddle/paddlenlp:2.4.0-gpu-cuda11.2-cudnn8
# 查看运行日志
sudo docker logs paddlenlp_pipelines_gpu
# 进入容器命令
sudo docker exec -it paddlenlp_pipelines_gpu bash
# 查看后台端口状态
lsof -i:8891
# 查看前端端口状态
lsof -i:8502
```
