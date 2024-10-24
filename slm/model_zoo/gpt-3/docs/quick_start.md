
# 快速开始

## 1. 环境准备

这里介绍使用裸机或者 Docker 环境使用 PaddleFleetX 的方法，用户根据具体情况选择一种安装部署方式即可。
使用多机训练时，需要在每台机器上都部署相应的环境。

### 1.1 Docker 环境部署

推荐使用 Docker 安装部署 PaddleFleetX 进行大模型训练，Docker 环境的安装可以参考[文档](docker_install.md)。

请根据本地 CUDA 版本（使用 `nvidia-smi`命令查看）使用以下命令拉取对应或兼容的镜像，

```
docker pull registry.baidubce.com/ppfleetx/fleetx-cuda11.2-cudnn8:dev
```

如本地环境cuda版本较低可以参考 Dockerfile 根据需要定制镜像。

大模型训练需要使用GPU，如已安装 nvida-container-runtime 可以使用以下命令运行镜像，

```
docker run -it --name=paddle --net=host -v /dev/shm:/dev/shm --shm-size=32G -v $PWD:/paddle --runtime=nvidia registry.baidubce.com/ppfleetx/ppfleetx-cuda11.2-cudnn8:v0.1.0 bash
```

未安装 nvida-container-runtime 或启动后无法执行 `nvidia-smi` 查看GPU信息时可以尝试通过如下脚本启动运行，

```shell
export CUDA_SO="$(\ls /usr/lib64/libcuda* | grep -v : | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | grep -v : | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(find /dev/nvidia* -maxdepth 1 -not -type d | xargs -I{} echo '--device {}:{}')

nvsmi=`which nvidia-smi`

docker run \
${CUDA_SO} ${DEVICES} \
-v /dev/shm:/dev/shm \
-v $PWD:/paddle \
--name paddle \
--net=host \
--shm-size=32G \
-v $nvsmi:$nvsmi \
-it \
registry.baidubce.com/ppfleetx/ppfleetx-cuda11.2-cudnn8:v0.1.0 \
bash
```

以上命令 `-v $PWD:/paddle` 将当前目录映射到 /paddle 目录，在 docker 环境内部对该目录的更改将会持久化。

> 为保证通信效率和通信正常，添加参数 --net=host 使用主机网络，更多 docker run 参数说明请参考 [docker 文档](https://docs.docker.com/engine/reference/commandline/run/)。

### 1.2 裸机部署

**安装 PaddlePaddle**

首先根据环境在
[安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html) 选择对应的版本使用 pip install 执行对应命令安装 PaddlePaddle.
**请务必按照文档安装 GPU 版本且验证安装成功**。

例如使用如下命令将会安装基于 CUDA 11.2 最新版本的 PaddlePaddle.

```shell
python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

安装遇到问题以及环境验证的方法也可以参考[文档](deployment_faq.md#1-单机环境验证)。

**安装依赖**

使用以下命令安装 PaddleFleetX 运行所需依赖。

```shell
python -m pip install -r https://raw.githubusercontent.com/PaddlePaddle/PaddleFleetX/develop/requirements.txt -i https://mirror.baidu.com/pypi/simple
```

## 2. 模型训练

进入环境后首先使用以下命令拉取最新代码

```shell
git clone https://github.com/PaddlePaddle/PaddleFleetX.git
```

然后根据需求选择对应的训练方式。

### 2.1. 单卡训练

切换工作目录并下载demo数据，
```
mkdir data
wget -O data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget -O data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
```

然后使用以下命令运行程序，

```shell
python ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml
```

若要在显存容量更小的16G V100环境下进行GPT模型单卡训练，可将对应yaml文件中的Model-hidden size值改为原来的1/2即可。

**运行日志**

```
[2022-09-21 05:42:26,980] [    INFO] - [train] epoch: 0, batch: 0, loss: 10.999595642, avg_batch_cost: 2.73014 sec, speed: 0.37 step/s, ips_total: 3001 tokens/s, ips: 3001 tokens/s, learning rate: 2.77778e-08
[2022-09-21 05:42:27,492] [    INFO] - [train] epoch: 0, batch: 1, loss: 10.997043610, avg_batch_cost: 0.51164 sec, speed: 1.95 step/s, ips_total: 16011 tokens/s, ips: 16011 tokens/s, learning rate: 4.16667e-08
[2022-09-21 05:42:27,997] [    INFO] - [train] epoch: 0, batch: 2, loss: 10.994422913, avg_batch_cost: 0.50457 sec, speed: 1.98 step/s, ips_total: 16236 tokens/s, ips: 16236 tokens/s, learning rate: 5.55556e-08
[2022-09-21 05:42:28,503] [    INFO] - [train] epoch: 0, batch: 3, loss: 11.005314827, avg_batch_cost: 0.50497 sec, speed: 1.98 step/s, ips_total: 16223 tokens/s, ips: 16223 tokens/s, learning rate: 6.94444e-08
[2022-09-21 05:42:29,009] [    INFO] - [train] epoch: 0, batch: 4, loss: 10.988020897, avg_batch_cost: 0.50480 sec, speed: 1.98 step/s, ips_total: 16228 tokens/s, ips: 16228 tokens/s, learning rate: 8.33333e-08
[2022-09-21 05:42:29,513] [    INFO] - [train] epoch: 0, batch: 5, loss: 10.983006477, avg_batch_cost: 0.50393 sec, speed: 1.98 step/s, ips_total: 16256 tokens/s, ips: 16256 tokens/s, learning rate: 9.72222e-08
[2022-09-21 05:42:30,018] [    INFO] - [train] epoch: 0, batch: 6, loss: 10.988539696, avg_batch_cost: 0.50427 sec, speed: 1.98 step/s, ips_total: 16245 tokens/s, ips: 16245 tokens/s, learning rate: 1.11111e-07
```



### 2.2. 单机多卡训练

切换工作目录并下载demo数据，

```shell
mkdir data
wget -O data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget -O data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
```

然后使用以下命令运行单机多卡程序，

```
python -m paddle.distributed.launch \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp8.yaml
```

若要在显存容量更小的环境例如 16G 显存下进行GPT模型单机训练，可通过减小`Model.hidden_size`调整模型规模至合适大小再启动训练，命令如下：

```
python -m paddle.distributed.launch \
    ./tools/train.py -c \
    ./ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp8.yaml -o Model.hidden_size=1024
```

> 更多 launch 启动参数和用法请参考 [API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/launch_cn.html)。

成功则开始训练过程，
```
LAUNCH INFO 2022-08-15 07:37:38,946 -----------  Configuration  ----------------------
LAUNCH INFO 2022-08-15 07:37:38,946 devices: None
LAUNCH INFO 2022-08-15 07:37:38,947 elastic_level: -1
LAUNCH INFO 2022-08-15 07:37:38,947 elastic_timeout: 30
LAUNCH INFO 2022-08-15 07:37:38,947 gloo_port: 6767
LAUNCH INFO 2022-08-15 07:37:38,947 host: None
LAUNCH INFO 2022-08-15 07:37:38,947 ips: None
LAUNCH INFO 2022-08-15 07:37:38,947 job_id: default
LAUNCH INFO 2022-08-15 07:37:38,947 legacy: False
LAUNCH INFO 2022-08-15 07:37:38,947 log_dir: log
LAUNCH INFO 2022-08-15 07:37:38,947 log_level: INFO
LAUNCH INFO 2022-08-15 07:37:38,947 master: None
LAUNCH INFO 2022-08-15 07:37:38,947 max_restart: 3
LAUNCH INFO 2022-08-15 07:37:38,947 nnodes: 1
LAUNCH INFO 2022-08-15 07:37:38,947 nproc_per_node: None
LAUNCH INFO 2022-08-15 07:37:38,947 rank: -1
LAUNCH INFO 2022-08-15 07:37:38,947 run_mode: collective
LAUNCH INFO 2022-08-15 07:37:38,947 server_num: None
LAUNCH INFO 2022-08-15 07:37:38,947 servers:
LAUNCH INFO 2022-08-15 07:37:38,947 start_port: 6070
LAUNCH INFO 2022-08-15 07:37:38,947 trainer_num: None
LAUNCH INFO 2022-08-15 07:37:38,947 trainers:
LAUNCH INFO 2022-08-15 07:37:38,947 training_script: run_pretrain.py
LAUNCH INFO 2022-08-15 07:37:38,947 training_script_args: ['-c', './configs_1.3B_dp8.yaml']
LAUNCH INFO 2022-08-15 07:37:38,947 with_gloo: 1
LAUNCH INFO 2022-08-15 07:37:38,947 --------------------------------------------------
LAUNCH INFO 2022-08-15 07:37:38,948 Job: default, mode collective, replicas 1[1:1], elastic False
LAUNCH INFO 2022-08-15 07:37:38,949 Run Pod: vqhbut, replicas 8, status ready
LAUNCH INFO 2022-08-15 07:37:39,063 Watching Pod: vqhbut, replicas 8, status running
## 启动配置
[2022-08-15 07:41:23,063] [    INFO] - [train] epoch: 0, batch: 0, loss: 11.255846024, avg_batch_cost: 7.06713 sec, speed: 0.14 step/s, ips_total: 9273 tokens/s, ips: 1159 tokens/s, learning rate: 2.77778e-08
## 更多训练日志
```

如有启动异常请根据[文档](deployment_faq.md#1-单机环境验证)进行工作环境验证，其他问题可参考[FAQ](deployment_faq.md#3-faq)解决。

## 2.3. 多机多卡训练

使用以下命令进行多机分布式训练，其中 --nnodes 参数为分布式训练机器数量，--master 为训练机器中其中一台机器的IP，运行时需要将命令中示例IP替换为真实的机器IP和任意可用端口，然后在**每个节点**上都运行以下命令，
如果不知道机器IP可以不设置--master参数先在一台机器上启动，然后根据提示复制命令在其他机器上启动即可。

```
python -m paddle.distributed.launch --master=10.10.10.1:8099 --nnodes=2 \
    ./tools/train.py -c \
    ./ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml
```

> 该示例为16卡任务，需要满足总卡数为16的要求。

> 注意这里需要使用单机多卡训练部分的代码和数据。


成功则开始多机训练过程，日志和单机多卡类似，日志异常时请按照[文档](deployment_faq.md#2-分布式环境验证)进行环境验证和问题排查。

若要在显存容量更小的环境例如 16G 显存下进行GPT模型单机训练，可通过减小`Model.hidden_size`调整模型规模至合适大小再启动训练，命令如下：

```
python -m paddle.distributed.launch --master=10.10.10.1:8099 --nnodes=2 \
    ./tools/train.py -c \
    ./ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml -o Model.hidden_size=2048
```

更多大模型多机训练内容可见[文档](../projects/gpt/docs/README.md)。
