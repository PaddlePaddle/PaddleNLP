## 1. 环境准备

本教程适用于 test_tipc 目录下基础功能测试的运行环境搭建。

推荐环境：
- CUDA 10.1/10.2
- CUDNN 7.6/cudnn8.1
- TensorRT 6.1.0.5 / 7.1 / 7.2

环境配置可以选择 docker 镜像安装，或者在本地环境 Python 搭建环境。推荐使用 docker 镜像安装，避免不必要的环境配置。

## 2. Docker 镜像安装

推荐 docker 镜像安装，按照如下命令创建镜像，当前目录映射到镜像中的`/paddle`目录下
```
nvidia-docker run --name paddle -it -v $PWD:/paddle paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82 /bin/bash
cd /paddle

# 安装带TRT的paddle
pip3.7 install https://paddle-wheel.bj.bcebos.com/with-trt/2.1.3/linux-gpu-cuda10.1-cudnn7-mkl-gcc8.2-trt6-avx/paddlepaddle_gpu-2.1.3.post101-cp37-cp37m-linux_x86_64.whl
```

## 3 Python 环境构建

非 docker 环境下，环境配置比较灵活，推荐环境组合配置：
- CUDA10.1 + CUDNN7.6 + TensorRT 6
- CUDA10.2 + CUDNN8.1 + TensorRT 7
- CUDA11.1 + CUDNN8.1 + TensorRT 7

下面以 CUDA10.2 + CUDNN8.1 + TensorRT 7 配置为例，介绍环境配置的流程。

### 3.1 安装 CUDNN

如果当前环境满足 CUDNN 版本的要求，可以跳过此步骤。

以 CUDNN8.1 安装安装为例，安装步骤如下，首先下载 CUDNN，从[Nvidia 官网](https://developer.nvidia.com/rdp/cudnn-archive)下载 CUDNN8.1版本，下载符合当前系统版本的三个 deb 文件，分别是：
- cuDNN Runtime Library ，如：libcudnn8_8.1.0.77-1+cuda10.2_amd64.deb
- cuDNN Developer Library ，如：libcudnn8-dev_8.1.0.77-1+cuda10.2_amd64.deb
- cuDNN Code Samples，如：libcudnn8-samples_8.1.0.77-1+cuda10.2_amd64.deb

deb 安装可以参考[官方文档](https://docs.nvidia.com/deeplearning/cudnn/latest/)，安装方式如下
```
# x.x.x表示下载的版本号
# $HOME为工作目录
sudo dpkg -i libcudnn8_x.x.x-1+cudax.x_arm64.deb
sudo dpkg -i libcudnn8-dev_8.x.x.x-1+cudax.x_arm64.deb
sudo dpkg -i libcudnn8-samples_8.x.x.x-1+cudax.x_arm64.deb

# 验证是否正确安装
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN

# 编译
make clean && make
./mnistCUDNN
```
如果运行 mnistCUDNN 完后提示运行成功，则表示安装成功。如果运行后出现 freeimage 相关的报错，需要按照提示安装 freeimage 库:
```
sudo apt-get install libfreeimage-dev
sudo apt-get install libfreeimage
```

### 3.2 安装 TensorRT

首先，从[Nvidia 官网 TensorRT 板块](https://developer.nvidia.com/tensorrt-getting-started)下载 TensorRT，这里选择7.1.3.4版本的 TensorRT，注意选择适合自己系统版本和 CUDA 版本的 TensorRT，另外建议下载 TAR package 的安装包。

以 Ubuntu16.04+CUDA10.2为例，下载并解压后可以参考[官方文档](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/install-guide/index.html#installing-tar)的安装步骤，按照如下步骤安装:
```
# 以下安装命令中 '${version}' 为下载的TensorRT版本，如7.1.3.4
# 设置环境变量，<TensorRT-${version}/lib> 为解压后的TensorRT的lib目录
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>

# 安装TensorRT
cd TensorRT-${version}/python
pip3.7 install tensorrt-*-cp3x-none-linux_x86_64.whl

# 安装graphsurgeon
cd TensorRT-${version}/graphsurgeon
```


### 3.3 安装 PaddlePaddle

下载支持 TensorRT 版本的 Paddle 安装包，注意安装包的 TensorRT 版本需要与本地 TensorRT 一致，下载[链接](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#python)
选择下载 linux-cuda10.2-trt7-gcc8.2 Python3.7版本的 Paddle：
```
# 从下载链接中可以看到是paddle2.1.1-cuda10.2-cudnn8.1版本
wget  https://paddle-wheel.bj.bcebos.com/with-trt/2.1.1-gpu-cuda10.2-cudnn8.1-mkl-gcc8.2/paddlepaddle_gpu-2.1.1-cp37-cp37m-linux_x86_64.whl
pip3.7 install -U paddlepaddle_gpu-2.1.1-cp37-cp37m-linux_x86_64.whl
```

## 4. 安装 PaddleNLP 依赖
```
# 安装AutoLog
git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog
pip3.7 install -r requirements.txt
python3.7 setup.py bdist_wheel
pip3.7 install ./dist/auto_log-1.0.0-py3-none-any.whl

# 下载PaddleNLP代码
cd ../
git clone https://github.com/PaddlePaddle/PaddleNLP

```

安装 PaddleNLP 依赖：
```
cd PaddleNLP
pip3.7 install ./
```

## FAQ :
Q. You are using Paddle compiled with TensorRT, but TensorRT dynamic library is not found. Ignore this if TensorRT is not needed.

A. 问题一般是当前安装 paddle 版本带 TRT，但是本地环境找不到 TensorRT 的预测库，需要下载 TensorRT 库，解压后设置环境变量 LD_LIBRARY_PATH;
如：
```
export LD_LIBRARY_PATH=/usr/local/python3.7.0/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/paddle/package/TensorRT-6.0.1.5/lib
```
或者问题是下载的 TensorRT 版本和当前 paddle 中编译的 TRT 版本不匹配，需要下载版本相符的 TensorRT 重新安装。
