
## Docker 环境安装

使用 Docker 首先需要安装 Docker  环境，安装的完整流程请参考[文档](https://docs.docker.com/engine/install/)，基础安装流程如下所述。
另外在 Docker 中使用 GPU 还需要安装 [nvida-container-runtime](https://github.com/NVIDIA/nvidia-container-runtime)。

**Ubuntu**

添加 apt 源。
```
sudo curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
```

软件源升级， 安装docker

```
sudo apt-get update

sudo apt-get docker-ce docker-ce-cli containerd.io
```

使用 `docker version` 查看 docker 版本信息无错误信息即说明安装运行正常。

安装 nvida-container-runtime

```
sudo apt-get install nvidia-container-runtimeb
```

**CentOS**

添加yum源。

```
sudo wget -O /etc/yum.repos.d/docker-ce.repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
```

安装组件。
```
sudo yum install docker-ce docker-ce-cli containerd.io
```

启动Docker。
```
sudo systemctl start docker
```

查看Docker状态。
```
sudo systemctl status docker
```

如日志状态为 active (running) 则表示docker启动正常。
```
● docker.service - LSB: start and stop docker
   Loaded: loaded (/etc/rc.d/init.d/docker; bad; vendor preset: disabled)
   Active: active (running) since Thu 2022-08-11 20:11:19 CST; 3 days ago
     Docs: man:systemd-sysv-generator(8)
  Process: 29766 ExecStop=/etc/rc.d/init.d/docker stop (code=exited, status=0/SUCCESS)
  Process: 33215 ExecStart=/etc/rc.d/init.d/docker start (code=exited, status=0/SUCCESS)
```

安装 nvida-container-runtime。

```
sudo yum install nvidia-container-runtime
```
