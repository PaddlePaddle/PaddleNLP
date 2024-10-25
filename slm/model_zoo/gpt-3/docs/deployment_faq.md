## 环境验证和常见问题

本文为环境问题排查指引，包括环境正确性验证的方法和常见的一些问题解决方法。

### 1. 单机环境验证

以下验证不区分本机环境和 Docker 环境。

**GPU验证**

当使用 GPU 时，使用 `nvidia-smi` 命令查看环境中 GPU 状态，预期输出如下

```shell
Thu Jul 21 19:32:03 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:3F:00.0 Off |                    0 |
| N/A   33C    P0    40W / 300W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  On   | 00000000:40:00.0 Off |                    0 |
| N/A   34C    P0    41W / 300W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  On   | 00000000:41:00.0 Off |                    0 |
| N/A   35C    P0    41W / 300W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  On   | 00000000:42:00.0 Off |                    0 |
| N/A   38C    P0    42W / 300W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   4  Tesla V100-SXM2...  On   | 00000000:62:00.0 Off |                    0 |
| N/A   34C    P0    39W / 300W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   5  Tesla V100-SXM2...  On   | 00000000:63:00.0 Off |                    0 |
| N/A   36C    P0    40W / 300W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   6  Tesla V100-SXM2...  On   | 00000000:64:00.0 Off |                    0 |
| N/A   37C    P0    41W / 300W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   7  Tesla V100-SXM2...  On   | 00000000:65:00.0 Off |                    0 |
| N/A   36C    P0    39W / 300W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

结果中可以看出

* CUDA Version栏显示的是当前环境中的CUDA版本号，此处为11.2。开始使用飞桨前，请先保证此处CUDA Version显示正常。如果CUDA Version栏不显示版本号，则需要添加CUDA相关库的路径到环境变量`LD_LIBRARY_PATH`中，例如执行命令添加 `export LD_LIBRARY_PATH=/usr/lib64/:/usr/local/lib/:/usr/local/cuda-11.2/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}` 。具体请参考[文档](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)。
* Memory-Usage 列显示的是当前的显存占用值，此处为0MiB，表示当前设备的显存未被占用；GPU-Util 列显示的是当前的GPU利用率，此处为0%，表示当前设备空闲，可以使用。开始使用飞桨前，请保证当前设备显存充足，且利用率处于空闲状态。
* 最后的 Processes 信息表示正在使用设备的进程，Docker 内可能存在不准确的情况，不影响使用。

**PaddlePaddle 安装验证**

首先运行如下命令确保 PaddlePaddle 正确安装

```shell
python -c "import paddle; paddle.utils.run_check()"
```

预期会有如下输出

```shell
Running verify PaddlePaddle program ...
W0720 09:29:22.035640 12791 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0720 09:29:22.040702 12791 gpu_resources.cc:91] device: 0, cuDNN Version: 8.1.
PaddlePaddle works well on 1 GPU.
W0720 09:29:36.763486 12791 fuse_all_reduce_op_pass.cc:79] Find all_reduce operators: 2. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 2.
PaddlePaddle works well on 8 GPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

表示 PaddlePaddle 已经正确安装。

如果出现以下错误信息请确保 CUDA 安装正确且已根据 CUDA 安装路径正确配置的 LD_LIBRARY_PATH。
例如执行命令添加 `export LD_LIBRARY_PATH=/usr/lib64/:/usr/local/lib/:/usr/local/cuda-11.2/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}` 。
具体请参考[文档](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)。

```
You are using GPU version Paddle, but your CUDA device is not set properly.
```

### 2. 分布式环境验证

如果单机运行正常，但多机分布式运行异常请先根据 [网络问题排查](#31-网络问题排查) 部分排查网络问题再进行以下排查。

请先确保**各个机器**的 PaddlePaddle 环境已经正确安装，然后在等待验证的其中一个节点上运行如下命令

```shell
python -m paddle.distributed.launch run_check
```

> 默认验证 2 机分布式环境，如果需要验证更多机器（例如4个）环境下飞桨分布式是否运行正常，请添加节点数参数 --nnodes，具体命令如下：
>
> `python -m paddle.distributed.launch --nnodes=4 run_check`

预期输出如下

```shell
LAUNCH INFO 2022-07-20 09:38:33,349 PaddlePaddle Distributed Check begin...
LAUNCH INFO 2022-07-20 09:38:33,358 -----------  Configuration  ----------------------
LAUNCH INFO 2022-07-20 09:38:33,358 devices: None
LAUNCH INFO 2022-07-20 09:38:33,358 elastic_level: -1
LAUNCH INFO 2022-07-20 09:38:33,358 elastic_timeout: 30
LAUNCH INFO 2022-07-20 09:38:33,358 gloo_port: 6767
LAUNCH INFO 2022-07-20 09:38:33,358 host: None
LAUNCH INFO 2022-07-20 09:38:33,358 job_id: default
LAUNCH INFO 2022-07-20 09:38:33,358 legacy: False
LAUNCH INFO 2022-07-20 09:38:33,358 log_dir: log
LAUNCH INFO 2022-07-20 09:38:33,358 log_level: ERROR
LAUNCH INFO 2022-07-20 09:38:33,358 master: None
LAUNCH INFO 2022-07-20 09:38:33,358 max_restart: 3
LAUNCH INFO 2022-07-20 09:38:33,358 nnodes: 2
LAUNCH INFO 2022-07-20 09:38:33,358 nproc_per_node: None
LAUNCH INFO 2022-07-20 09:38:33,358 rank: -1
LAUNCH INFO 2022-07-20 09:38:33,358 run_mode: collective
LAUNCH INFO 2022-07-20 09:38:33,359 server_num: None
LAUNCH INFO 2022-07-20 09:38:33,359 servers:
LAUNCH INFO 2022-07-20 09:38:33,359 trainer_num: None
LAUNCH INFO 2022-07-20 09:38:33,359 trainers:
LAUNCH INFO 2022-07-20 09:38:33,359 training_script: /usr/local/lib/python3.7/dist-packages/paddle/distributed/launch/plugins/test.py
LAUNCH INFO 2022-07-20 09:38:33,359 training_script_args: []
LAUNCH INFO 2022-07-20 09:38:33,359 with_gloo: 1
LAUNCH INFO 2022-07-20 09:38:33,359 --------------------------------------------------
LAUNCH INFO 2022-07-20 09:38:33,360 Job: default, mode collective, replicas 2[2:2], elastic False
LAUNCH INFO 2022-07-20 09:38:33,367 Waiting peer start...
Copy the following command to other nodes to run.
--------------------------------------------------------------------------------
python -m paddle.distributed.launch --master 10.10.1.1:49178 run_check
--------------------------------------------------------------------------------
```

> 如果当前安装的 PaddlePaddle 中未包含该工具，请根据上节提示安装 develop 版本进行测试。

根据提示，复制最后的命令（复制机器上个命令的执行结果，以下命令为示例），在其他节点上粘贴执行

```shell
python -m paddle.distributed.launch --master 10.10.1.1:49178 run_check
```

执行后，如果配置正常则每个节点都会有后续输出

```shell
LAUNCH INFO 2022-07-20 09:46:41,571 Run Pod: xqqbsr, replicas 2, status ready
LAUNCH INFO 2022-07-20 09:46:41,601 Watching Pod: xqqbsr, replicas 2, status running
Prepare distributed training with 2 nodes 2 cards
I0720 09:46:43.583846 13375 tcp_utils.cc:181] The server starts to listen on IP_ANY:14863
I0720 09:46:43.584153 13375 tcp_utils.cc:130] Successfully connected to 10.10.10.1:14863
W0720 09:46:47.089151 13375 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0720 09:46:47.098454 13375 gpu_resources.cc:91] device: 0, cuDNN Version: 8.1.
2022-07-20 09:46:51,333-INFO: [topology.py:187:__init__] HybridParallelInfo: rank_id: 0, mp_degree: 1, sharding_degree: 1, pp_degree: 1, dp_degree: 4, mp_group: [0],  sharding_group: [0], pp_group: [0], dp_group: [0, 1, 2, 3], check/clip group: [0]
Distributed training start...
[Epoch 0, batch 0] loss: 5.10316, acc1: 0.03125, acc5: 0.06250
Distributed training completed
I0720 09:46:54.828758 13432 tcp_store.cc:257] receive shutdown event and so quit from MasterDaemon run loop
LAUNCH INFO 2022-07-20 09:46:56,617 Pod completed
LAUNCH INFO 2022-07-20 09:46:57,085 Exit code 0
```

则表示分布式环境配置正常，多机分布式训练可以成功运行。

> 如果其他节点执行命令后各个节点没有后续输出或输出不符合预期请参考 [FAQ](#3-faq) 部分解决。

**实际分布式训练任务验证**

在启动分布式任务前需要确保各个节点上安装好 PaddlePaddle 环境，同步好数据和代码。

例如准备好训练代码 `train.py`，同步至每个训练节点的工作目录。

```python
import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock
from paddle.io import Dataset, BatchSampler, DataLoader

base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4

epoch = 10
batch_num = 3
batch_size = 32
class_dim = 102

class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([3, 224, 224]).astype('float32')
        label = np.random.randint(0, class_dim - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

def optimizer_setting(parameter_list=None):
    optimizer = paddle.optimizer.Momentum(
        learning_rate=base_lr,
        momentum=momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(l2_decay),
        parameters=parameter_list)
    return optimizer


def train_resnet():
    fleet.init(is_collective=True)

    resnet = ResNet(BottleneckBlock, 18, num_classes=class_dim)
    optimizer = optimizer_setting(parameter_list=resnet.parameters())
    optimizer = fleet.distributed_optimizer(optimizer)
    resnet = fleet.distributed_model(resnet)

    dataset = RandomDataset(batch_num * batch_size)
    train_loader = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=2)

    for eop in range(epoch):
        resnet.train()

        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True

            out = resnet(img)
            loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            avg_loss = paddle.mean(x=loss)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

            avg_loss.backward()
            optimizer.step()
            resnet.clear_gradients()

            print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (eop, batch_id, avg_loss, acc_top1, acc_top5))

if __name__ == '__main__':
    train_resnet()
```

启动分布式训练的命令如下，
这个命令需要在每个参与训练的节点上执行（每个节点上的 `--master`都设置为同一个），如节点较多可以考虑使用 `ssh` 脚本或 `mpirun` 进行跨节点命令分发。

```python
python -m paddle.distributed.launch --master=10.10.1.1:49178 --nnodes=2 train.py
```

这里用到了分布式启动最重要的两个参数

- `--nnodes` 为分布式任务的节点个数（一般为参与任务的机器数量），默认为 1 即启动单机任务，也可使用环境变量 PADDLE_NNODES 设置。

- `--master` 为分布式信息同步的主节点地址，ip:port 格式，可以由第一个启动的节点自动打印或者直接由用户设置为参与任务的任意节点 ip 和任意可用端口，也可使用环境变量 PADDLE_MASTER 设置。

> master 支持使用 etcd 服务，当使用 etcd 服务时，需要同时指定任务 id 以避免任务间冲突。具体地，可以通过 --job_id 参数或者设置环境变量 PADDLE_JOB_ID 指定任务id。


启动后，将看到如下日志，首先是配置部分

```shell
LAUNCH INFO 2022-07-20 12:10:15,863 -----------  Configuration  ----------------------
LAUNCH INFO 2022-07-20 12:10:15,863 devices: None
LAUNCH INFO 2022-07-20 12:10:15,863 elastic_level: -1
LAUNCH INFO 2022-07-20 12:10:15,863 elastic_timeout: 30
LAUNCH INFO 2022-07-20 12:10:15,863 gloo_port: 6767
LAUNCH INFO 2022-07-20 12:10:15,863 host: None
LAUNCH INFO 2022-07-20 12:10:15,863 job_id: default
LAUNCH INFO 2022-07-20 12:10:15,863 legacy: False
LAUNCH INFO 2022-07-20 12:10:15,863 log_dir: log
LAUNCH INFO 2022-07-20 12:10:15,863 log_level: INFO
LAUNCH INFO 2022-07-20 12:10:15,863 master: 127.0.0.1:8890
LAUNCH INFO 2022-07-20 12:10:15,863 max_restart: 3
LAUNCH INFO 2022-07-20 12:10:15,863 nnodes: 2
LAUNCH INFO 2022-07-20 12:10:15,863 nproc_per_node: None
LAUNCH INFO 2022-07-20 12:10:15,863 rank: -1
LAUNCH INFO 2022-07-20 12:10:15,863 run_mode: collective
LAUNCH INFO 2022-07-20 12:10:15,863 server_num: None
LAUNCH INFO 2022-07-20 12:10:15,863 servers:
LAUNCH INFO 2022-07-20 12:10:15,863 trainer_num: None
LAUNCH INFO 2022-07-20 12:10:15,863 trainers:
LAUNCH INFO 2022-07-20 12:10:15,863 training_script: train.py
LAUNCH INFO 2022-07-20 12:10:15,863 training_script_args: []
LAUNCH INFO 2022-07-20 12:10:15,864 with_gloo: 1
LAUNCH INFO 2022-07-20 12:10:15,864 --------------------------------------------------
```

这里打印分布式启动时的配置信息， 更多 launch 启动参数和用法请参考 [API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/launch_cn.html) 或通过以下命令获得。

```shell
python -m paddle.distributed.launch --help
```

然后打印的是任务启动相关的信息：

```shell
LAUNCH INFO 2022-07-20 12:10:15,864 Job: default, mode collective, replicas 2[2:2], elastic False
LAUNCH INFO 2022-07-20 12:10:15,870 Waiting peer start...
LAUNCH INFO 2022-07-20 12:10:25,860 Run Pod: bpdjev, replicas 2, status ready
LAUNCH INFO 2022-07-20 12:10:25,883 Watching Pod: bpdjev, replicas 2, status running
```

其中，每行对应的具体含义解释如下：

* 因为未设置 job_id，使用默认名称 default，启动的是 collective 模式，总共 2 个节点的分布式任务，不支持弹性（即节点数不可变）。
* 节点短暂处于等待其他节点启动的状态，如果其他节点已启动但日志长期处于等待状态，请根据 [FAQ](#31-网络问题排查) 进行排查。
* 任务准备启动，当前节点名为 bpdjev（该名称为随机生成）处于 ready 状态，当前节点包含 2 个进程（1 个进程对应 1 个 GPU）。
* 节点已启动，正在监控进程健康状态。

至此分布式启动成功，接下来打印业务日志（即用户代码相关输出日志）

```shell
I0720 12:10:27.763713 14071 tcp_utils.cc:181] The server starts to listen on IP_ANY:11061
I0720 12:10:27.763914 14071 tcp_utils.cc:130] Successfully connected to 10.10.10.1:11061
W0720 12:10:30.666985 14071 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0720 12:10:30.675815 14071 gpu_resources.cc:91] device: 0, cuDNN Version: 8.1.
2022-07-20 12:10:36,377-INFO: [topology.py:187:**init**] HybridParallelInfo: rank_id: 0, mp_degree: 1, sharding_degree: 1, pp_degree: 1, dp_degree: 4, mp_group: [0], sharding_group: [0], pp_group: [0], dp_group: [0, 1, 2, 3], check/clip group: [0]
/usr/local/lib/python3.7/dist-packages/paddle/nn/layer/norm.py:668: UserWarning: When training, we now always track global mean and variance.
 "When training, we now always track global mean and variance.")
[Epoch 0, batch 0] loss: 5.42939, acc1: 0.00000, acc5: 0.00000
[Epoch 0, batch 1] loss: 6.13338, acc1: 0.00000, acc5: 0.03125
[Epoch 0, batch 2] loss: 7.25566, acc1: 0.03125, acc5: 0.06250
// 此处省略多行类似日志
[Epoch 9, batch 0] loss: 7.23511, acc1: 0.00000, acc5: 0.00000
[Epoch 9, batch 1] loss: 4.69053, acc1: 0.03125, acc5: 0.06250
[Epoch 9, batch 2] loss: 5.08652, acc1: 0.00000, acc5: 0.03125
I0720 12:10:53.647085 14112 tcp_store.cc:257] receive shutdown event and so quit from MasterDaemon run loop
```

至此，训练结束，业务代码结束，最后打印退出日志

```shell
LAUNCH INFO 2022-07-20 12:10:56,915 Pod completed
LAUNCH INFO 2022-07-20 12:10:57,388 Exit code 0
```

更多日志请在 log 目录下查看，日志文件命名为` {job_id}.{节点名}.{卡号}.log` , 例如如下两个文件为本例子中 2 张卡分别对应的日志。

```shell
-rw-r--r--  1 root   root 2.9K Jul 20 12:10 default.bpdjev.0.log
-rw-r--r--  1 root   root 2.7K Jul 20 12:10 default.bpdjev.1.log
```

当有错误发生时，比如 GPU 卡被占用发生冲突时，会有如下输出

```shell
LAUNCH INFO 2022-07-21 11:58:59,451 Pod failed
LAUNCH ERROR 2022-07-21 11:58:59,452 Container failed !!!
Container rank 6 status failed cmd ['/usr/bin/python', '-u', 'train.py'] code 1 log log/default.fxemxd.6.log
env {'GREP_COLOR': '1;31', 'CUDNN_VERSION': '8.1.1.33', 'LC_ALL': 'en_US.UTF-8', 'LD_LIBRARY_PATH': '/usr/local/lib/python3.7/dist-packages/cv2/../../lib64:/usr/local/cuda-11.2/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64', 'LANG': 'en_US.UTF-8', 'HOSTNAME': 'xxxxx', 'OLDPWD': '/home/userhome', 'WITH_GPU': 'ON', 'NVIDIA_VISIBLE_DEVICES': 'all', 'NCCL_VERSION': '2.8.4', 'GOPATH': '/root/gopath', 'PWD': '/home/userhome/workspace/Paddle', 'HOME': '/home/userhome', 'GOROOT': '/usr/local/go', 'CLICOLOR': '1', 'DEBIAN_FRONTEND': 'noninteractive', 'LIBRARY_PATH': '/usr/local/cuda/lib64/stubs', 'TERM': 'xterm', 'WITH_AVX': 'ON', 'CUDA_VERSION': '11.2.1', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'SHLVL': '1', 'LANGUAGE': 'en_US.UTF-8', 'NVIDIA_REQUIRE_CUDA': 'cuda>=11.2 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450,driver<451', 'PATH': '/home/cmake-3.16.0-Linux-x86_64/bin:/usr/local/gcc-8.2/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin:/root/gopath/bin:/home/userhome/.fzf/bin', 'PS1': '\\[\\033[1;33m\\]kui \\[\\033[1;37m\\]\\h \\[\\033[1;32m\\]\\w\\[\\033[1;33m\\]$(__git_ps1 " \\[\\033[35m\\]{\\[\\033[36m\\]%s\\[\\033[35m\\]}")\\[\\033[0m\\] ', '_': '/usr/bin/python', 'CUSTOM_DEVICE_ROOT': '', 'OMP_NUM_THREADS': '1', 'QT_QPA_PLATFORM_PLUGIN_PATH': '/usr/local/lib/python3.7/dist-packages/cv2/qt/plugins', 'QT_QPA_FONTDIR': '/usr/local/lib/python3.7/dist-packages/cv2/qt/fonts', 'runtime_include_dir': '/usr/local/lib/python3.7/dist-packages/paddle/libs', 'POD_NAME': 'fxemxd', 'PADDLE_MASTER': '10.10.10.1:60216', 'PADDLE_GLOBAL_SIZE': '10', 'PADDLE_LOCAL_SIZE': '8', 'PADDLE_GLOBAL_RANK': '8', 'PADDLE_LOCAL_RANK': '6', 'PADDLE_NNODES': '2', 'PADDLE_TRAINER_ENDPOINTS': '10.10.10.1:49825,10.10.10.1:18781,10.10.10.1:53546,10.10.10.1:30837,10.10.10.1:11249,10.10.10.1:13092,10.10.10.1:11398,10.10.10.1:21309,10.10.10.1:47065,10.10.10.1:14834', 'PADDLE_CURRENT_ENDPOINT': '10.10.10.1:47065', 'PADDLE_TRAINER_ID': '8', 'PADDLE_TRAINERS_NUM': '10', 'PADDLE_RANK_IN_NODE': '6', 'FLAGS_selected_gpus': '6'}
I0721 11:58:51.079766 29676 tcp_utils.cc:130] Successfully connected to 10.10.10.1:60216
W0721 11:58:54.582710 29676 gpu_resources.cc:61] Please NOTE: device: 6, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0721 11:58:54.590724 29676 gpu_resources.cc:91] device: 6, cuDNN Version: 8.1.
Traceback (most recent call last):
  File "train.py", line 75, in <module>
    train_resnet()
  File "train.py", line 39, in train_resnet
    fleet.init(is_collective=True)
  File "/usr/local/lib/python3.7/dist-packages/paddle/distributed/fleet/base/fleet_base.py", line 319, in init
    paddle.distributed.init_parallel_env()
  File "/usr/local/lib/python3.7/dist-packages/paddle/distributed/parallel.py", line 264, in init_parallel_env
    paddle.distributed.barrier(group=group)
  File "/usr/local/lib/python3.7/dist-packages/paddle/distributed/collective.py", line 334, in barrier
    task = group.process_group.barrier()
OSError: (External) NCCL error(5), invalid usage.
  [Hint: 'ncclInvalidUsage'. The call to NCCL is incorrect. This is usually reflecting a programming error.] (at /paddle/Paddle/paddle/fluid/distributed/collective/ProcessGroupNCCL.cc:214)

LAUNCH INFO 2022-07-21 11:59:00,655 Exit code -15
```

这当中主要包含以下信息：

* 发生错误的提示 Pod failed 和 Container failed !!!.
* 错误的卡号（Container rank 6），错误命令和错误环境的环境变量。
* 具体的错误信息 trace，该部分取决于业务代码错误内容。
* 最后打印错误退出码 Exit code -15.

请根据报错信息进行排查，部分错误请参考 [FAQ](#3-faq)。

### 3. FAQ

#### 3.1 网络问题排查

请按照以下步骤排查网络问题

**获取节点IP**

使用命令 `hostname -i` 查看机器 ip，多网卡环境使用 `ifconfig` 命令查看(见上节)，获得 IP。

```shell
$ hostname -i
10.10.10.1
```

如果这里得到的IP非预期使用的IP或者和日志中打印的IP不相符时，请根据后面小节排查是否是多网卡环境导致使用的网卡不一致。


**确认节点间是否能通过ping连接**

这里举例获得 ip 为 10.10.10.1，在其他节点上使用 `ping 10.10.10.1` 测试机器间是否能连接，有如下输出即为连接成功

```shell
$ ping 10.10.10.1
PING 10.10.10.1 (10.10.10.1) 56(84) bytes of data.
64 bytes from 10.10.10.1: icmp_seq=1 ttl=61 time=0.089 ms
64 bytes from 10.10.10.1: icmp_seq=2 ttl=61 time=0.057 ms
64 bytes from 10.10.10.1: icmp_seq=3 ttl=61 time=0.059 ms
64 bytes from 10.10.10.1: icmp_seq=4 ttl=61 time=0.078 ms
64 bytes from 10.10.10.1: icmp_seq=5 ttl=61 time=0.055 ms
^C
--- 10.10.10.1 ping statistics ---
5 packets transmitted, 5 received, 0% packet loss, time 4053ms
rtt min/avg/max/mdev = 0.055/0.067/0.089/0.016 ms
```

长时间无输出或其他输出即无法连接，请联系机器网络管理员处理。

**确认节点间是否能通过HTTP/TCP连接**

在机器 `10.10.10.1`上运行命令 `python -m http.server 8090` 启动 http 服务，

```shell
$ python -m http.server 8090
Serving HTTP on 0.0.0.0 port 8090 (http://0.0.0.0:8090/) ...
```

如果提示端口被占用请使用其他可用端口启动服务，然后在其他的机器上运行命令
`curl 10.10.10.1:8090`

```shell
$ curl 10.10.10.1:8090
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<title>Directory listing for /</title>
</head>
<body>
<h1>Directory listing for /</h1>
<hr>
<li><a href="train.py">train.py</a></li>
</ul>
<hr>
</body>
</html>
```

有类似以上输出则说明连接成功，否则两台机器间网络可能存在问题，尝试其他端口仍有问题需要联系网络管理员处理。

**确认NCCL是否运行正常**

首先，设置环境变量NCCL_DEBUG，查看NCCL版本和当前使用的IP

```shell
export NCCL_DEBUG=INFO

python -m paddle.distributed.launch train.py
```

在输出日志中找到 NCCL 版本信息

```shell
NCCL version 2.8.4+cuda11.2
```

确认各个节点的 NCCL 版本相同且高于 2.8。

以及在输出的信息中查找如下信息

```shell
[0] NCCL INFO NET/Socket : Using [0]eth0:10.10.10.1<0> [1]
```

表示 nccl 使用了名为 `eth0` ip 为 10.10.10.1 的网卡，如果需要使用其他网卡，需要在运行命令前添加环境变量

```shell
export NCCL_SOCKET_IFNAME=eth1
```

注意这里添加的时网卡名不是 ip，对应关系参照 `ifconfig` 的输出。

上述测试均正常但是无法跑通分布式环境测试时
请使用 [nccl-test](https://github.com/NVIDIA/nccl-tests)  测试 GPU 通信是否正常。

#### 3.2 多Python环境问题

当工作环境中存在多个版本的 python 时可能存在不一致导致问题。

检查 python 版本

```shell
$ python --version
Python 3.7.12
```

检查 python 安装目录

```shell
$ which python
/usr/bin/python
```

直接调用绝对路径验证版本

```shell
$ /usr/bin/python --version
Python 3.7.12
```

如果两次打印的版本不匹配，可以通过使用绝对路径的方式解决。
获取绝对路径需要知道需要安装目录，默认环境中可以通过以下命令查看安装的版本。

```shell
$ ls /usr/bin/python*
/usr/bin/python   /usr/bin/python2.7  /usr/bin/python3.6   /usr/bin/python3.7
```

即当使用 python 时，使用绝对路径 `/usr/bin/python3.7` 替换。

#### 3.3 自动获取 IP 错误（多网卡环境问题）

使用 paddle.distributed.launch 会自动识别使用的 IP，在多网卡配置的环境中自动识别的网卡可能不是预期使用的网卡。

首先可以通过 `ifconfig` 命令查看机器的网卡配置情况，例如

```shell
docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.3.1  netmask 255.255.255.0  broadcast 0.0.0.0
        inet6 fe80::7050:1cff:fea2:14f3  prefixlen 64  scopeid 0x20<link>
        ether 1e:a6:0d:0d:3b:1e  txqueuelen 1000  (Ethernet)
        RX packets 27201548  bytes 12176726229 (11.3 GiB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 26762571  bytes 48666409371 (45.3 GiB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 1321339447  bytes 1047567817083 (975.6 GiB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 1321339447  bytes 1047567817083 (975.6 GiB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.10.10.1  netmask 255.255.255.192  broadcast 10.127.4.191
        inet6 f080::5200:4bff:f030:2090  prefixlen 64  scopeid 0x20<link>
        ether 50:6b:4b:31:2a:90  txqueuelen 1000  (Ethernet)
        RX packets 32040749852  bytes 43394575453133 (39.4 TiB)
        RX errors 0  dropped 391107  overruns 0  frame 0
        TX packets 24330967394  bytes 30441950099144 (27.6 TiB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```

结果中虽然有3项甚至更多但这里只有一张 ip 为 `10.10.10.1` 网卡（inet值），docker0 为 Docker 虚拟网卡， lo 为本地回路，都不需要关注。

当启动分布式训练命令时，如果飞桨自动识别出的网卡IP不正确时，可以使用--host参数手动配置IP，如

```python
python -m paddle.distributed.launch --master=10.10.10.1:49178 --nnodes=2 --host=10.10.10.1 train.py
```

> 当 --master 地址识别错误时，也需要手动替换。

#### 3.4 机器端口有限制，需要使用固定端口

当集群环境限制通信网卡时需要手动配置所有 ip 和 port 以启动分布式，以机器 `10.10.10.1` 和机器 `10.10.10.2` 必须使用端口 8000-8999 的情况为例，
假设每台机器有两个卡，使用如下脚本设置每个卡对应进程的环境变量，依次启动进程。

```shell
# 所有卡 ip port 列表， ip1:port1,ip2:port2
export PADDLE_TRAINER_ENDPOINTS=10.10.10.1:8000,10.10.10.1:8001,10.10.10.2:8000,10.10.10.2:8001
# 所有卡数
export PADDLE_TRAINERS_NUM=4
# 当前卡 ip:port
export PADDLE_CURRENT_ENDPOINT=10.10.10.1:8000
# 当前卡序号
export PADDLE_TRAINER_ID=0
# 当前卡在节点内序号
export PADDLE_RANK_IN_NODE=0
# 当前卡使用的 GPU 卡号
export FLAGS_selected_gpus=0

# 注意，这里不再使用 launch 启动，但本脚本需要运行多次
python train.py
```

注意在执行时，需要依次替换后面4个环境变量为对应值启动。

#### 3.5 常用的通信问题排查

GPU/NCCL 问题请先核对**版本是否匹配**，通过 `nvidia-smi` 查看是否有进程正在占用，仍有问题需要通过 [nccl-test](https://github.com/NVIDIA/nccl-tests)  测试。常见运行时错误和解决方法如下，

**NCCL error(5)**

```shell
OSError: (External) NCCL error(5), invalid usage.
  [Hint: 'ncclInvalidUsage'. The call to NCCL is incorrect. This is usually reflecting a programming error.]
```

原因和解决方法：该错误多为同一张 GPU 卡被多个进程同时使用导致冲突，请检查正在使用 GPU 的进程。如果需要在同一台机器上启动多个逻辑节点，可以使用 `CUDA_VISIBLE_DEVICES` 环境变量控制设备可见性。

**NCCL error(2)**

```shell
ExternalError: Nccl error(2), unhandled system error
```

原因和解决方法：该错误一般为 shm 设置太小，如果使用 Docker 环境需要在启动 Docker 时做映射和设置如 `--shm-size 32G`.
