
## 集群部署

本文档介绍在集群上使用分布式进行大模型训练的方法，包括在 Kubernetes 上使用 PaddlePaddle 分布式和在云上使用的方法。

### 1. Kubernetes部署

在 Kubernetes 上部署分布式任务需要安装 [paddle-operator](https://github.com/PaddleFlow/paddle-operator) 。

paddle-operator 通过添加自定义资源类型 (paddlejob) 以及部署 controller 和一系列 Kubernetes 原生组件的方式实现简单定义即可运行 PaddlePaddle 任务的需求。

目前支持运行 ParameterServer (PS) 和 Collective 两种分布式任务，当然也支持运行单节点任务。

**paddle-operator 安装**

安装 paddle-operator 需要有已经安装的 Kubernetes (v1.16+) 集群和 [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) (v1.16+) 工具。

本节所需配置文件和示例可以在 [这里](https://github.com/PaddleFlow/paddle-operator/tree/main/deploy) 找到，
可以通过 *git clone* 或者复制文件内容保存。

```yaml
deploy
|-- examples
|   |-- resnet.yaml
|   |-- wide_and_deep.yaml
|   |-- wide_and_deep_podip.yaml
|   |-- wide_and_deep_service.yaml
|   `-- wide_and_deep_volcano.yaml
|-- v1
|   |-- crd.yaml
|   `-- operator.yaml
```

执行以下命令，

```shell
kubectl create -f https://raw.githubusercontent.com/PaddleFlow/paddle-operator/dev/deploy/v1/crd.yaml
```

或者

```shell
kubectl create -f deploy/v1/crd.yaml
```

通过以下命令查看是否成功，

```shell
kubectl get crd
NAME                                    CREATED AT
paddlejobs.batch.paddlepaddle.org       2021-02-08T07:43:24Z
```

执行以下部署命令，

```shell
kubectl create -f https://raw.githubusercontent.com/PaddleFlow/paddle-operator/dev/deploy/v1/operator.yaml
```

或者

```shell
kubectl create -f deploy/v1/operator.yaml
```

通过以下命令查看部署结果和运行状态，

```shell
kubectl -n paddle-system get pods
NAME                                         READY   STATUS    RESTARTS   AGE
paddle-controller-manager-698dd7b855-n65jr   1/1     Running   0          1m
```

通过查看 controller 日志以确保运行正常，

```shell
kubectl -n paddle-system logs paddle-controller-manager-698dd7b855-n65jr
```

提交 demo 任务查看效果，

```shell
kubectl -n paddle-system create -f deploy/examples/wide_and_deep.yaml
```

查看 paddlejob 任务状态, pdj 为 paddlejob 的缩写，

```shell
kubectl -n paddle-system get pdj
NAME                     STATUS      MODE   AGE
wide-ande-deep-service   Completed   PS     4m4s
```

以上信息可以看出：训练任务已经正确完成，该任务为 ps 模式。
可通过 cleanPodPolicy 配置任务完成/失败后的 pod 删除策略，详见任务配置。

训练期间可以通过如下命令查看 pod 状态，

```shell
kubectl -n paddle-system get pods
```

**paddlejob 任务提交**

本resnet示例为 Collective 模式，使用 GPU 进行训练，只需要配置 worker，worker 配置中需要声明使用的 GPU 信息。

准备配置文件，

```yaml
apiVersion: batch.paddlepaddle.org/v1
kind: PaddleJob
metadata:
  name: resnet
spec:
  cleanPodPolicy: Never
  worker:
    replicas: 2
    template:
      spec:
        containers:
          - name: paddle
            image: registry.baidubce.com/paddle-operator/demo-resnet:v1
            command:
            - python
            args:
            - "-m"
            - "paddle.distributed.launch"
            - "train_fleet.py"
            volumeMounts:
            - mountPath: /dev/shm
              name: dshm
            resources:
              limits:
                nvidia.com/gpu: 1
        volumes:
        - name: dshm
          emptyDir:
            medium: Memory
```

注意：

* 这里需要添加 shared memory 挂载以防止缓存出错。
* 本示例采用内置 flower 数据集，程序启动后会进行下载，根据网络环境可能等待较长时间。

提交任务: 使用 kubectl 提交 yaml 配置文件以创建任务，

```shell
kubectl -n paddle-system create -f resnet.yaml
```

**卸载**

通过以下命令卸载部署的组件，

```shell
kubectl delete -f deploy/v1/crd.yaml -f deploy/v1/operator.yaml
```

*注意：重新安装时，建议先卸载再安装*

### 2. 公有云和私有云部署

在公有云上运行 PaddlePaddle 分布式建议通过选购容器引擎服务的方式，各大云厂商都推出了基于标准 Kubernetes 的云产品，然后根据上节中的教程安装使用即可。

| 云厂商 | 容器引擎 | 链接                                           |
| --- | ---- | -------------------------------------------- |
| 百度云 | CCE  | https://cloud.baidu.com/product/cce.html     |
| 阿里云 | ACK  | https://help.aliyun.com/product/85222.html   |
| 华为云 | CCE  | https://www.huaweicloud.com/product/cce.html |

更为方便的是使用百度提供的全功能AI开发平台 [BML](https://cloud.baidu.com/product/bml) 来使用，详细的使用方式请参考 [BML文档](https://ai.baidu.com/ai-doc/BML/pkhxhgo5v)。
