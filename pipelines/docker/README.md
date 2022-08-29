# Docker 启动 pipelines服务

## 1. Docker 一键启动

### 1.1 Docker本地构建镜像启动

为了满足用户多种多样的需求，我们提供了Dockerfile来构建一个镜像(以语义索引为例)，其中`Docker`的安装请参考[官方文档](https://docs.docker.com/desktop/),安装完以后，修改服务端运行脚本`run_server.sh`，客户端界面脚本`run_client.sh`，然后执行下面的命令：

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

cpu版本大概等待3分钟左右，gpu版本大概1分钟左右，到这里您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验语义检索系统服务了。

## 2. Docker-Compose 一键启动

为了方便用户很方便的切换，并部署`Pipelines`的语义索引和问答应用，我们特地提供了`Docker-Compose`的方式，`Docker Compose`的安装请参考[Docker Compose](https://docs.docker.com/compose/)，用户只需要修改`docker-compose.yml`的配置，然后一键启动即可。

下面以语义检索的的例子进行展示：

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
构建过程一般会持续3分钟左右，然后cpu版本启动等待1分钟左右，然后您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验语义检索系统服务了。
