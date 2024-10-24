# 多硬件自然语言理解模型列表
本文档主要针对昇腾 NPU、寒武纪 MLU、昆仑 XPU 硬件平台，介绍 PaddleNLP 支持的自然语言理解模型及使用方法。
## 1.模型列表
| 模型名称/硬件支持 | NPU | XPU | MLU |
| - | - | - | - |
| [BERT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/bert) | ✅ | ✅ | ✅ |
| [ERNIE-3.0](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/ernie-3.0) | ✅ | ❌ | ❌ |
| [UIE](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/uie) | ✅ | ❌ | ❌ |
| [UTC](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/applications/zero_shot_text_classification) | ✅ | ❌ | ❌ |
| [RoBERTa](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/roberta) | ✅ | ❌ | ❌ |

## 2.各硬件使用指南
首先在硬件平台上安装飞桨环境，然后参照模型文档中的使用方法，只需将 device 参数改为对应的硬件平台即可。
### 2.1 昇腾 NPU
昇腾 910 系列是华为昇腾（Ascend）推出的一款高能效、灵活可编程的人工智能处理器。采用自研华为达芬奇架构，集成丰富的计算单元, 提高 AI 计算完备度和效率，进而扩展该芯片的适用性。
#### 2.1.1 环境准备
当前 PaddleNLP 支持昇腾 910B 芯片（更多型号还在支持中，如果您有其他型号的相关需求，请提交 issue 告知我们），昇腾驱动版本为 23.0.3。考虑到环境差异性，我们推荐使用飞桨官方提供的标准镜像完成环境准备。
- 1. 拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包，镜像中已经默认安装了昇腾算子库 CANN-8.0.T13。

```
# 适用于 X86 架构
docker pull registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-x86_64-gcc84-py39
# 适用于 Aarch64 架构
docker pull registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-aarch64-gcc84-py39
```

- 2. 参考如下命令启动容器，ASCEND_RT_VISIBLE_DEVICES 指定可见的 NPU 卡号
```
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-$(uname -m)-gcc84-py39 /bin/bash
```
#### 2.1.2 安装 paddle 包
当前提供 Python3.9 的 wheel 安装包。如有其他 Python 版本需求，可以参考[飞桨官方文档](https://www.paddlepaddle.org.cn/install/quick)自行编译安装。

- 1. 下载安装 Python3.9 的 wheel 安装包

```
# 注意需要先安装飞桨 cpu 版本
python3.9 -m pip install paddlepaddle==3.0.0.dev20240520 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python3.9 -m pip install paddle_custom_npu==3.0.0.dev20240719 -i https://www.paddlepaddle.org.cn/packages/nightly/npu/
```
- 2. 验证安装包
安装完成之后，运行如下命令。
```
python -c "import paddle; paddle.utils.run_check()"
```
预期得到如下输出结果
```
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 npu.
PaddlePaddle works well on 8 npus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
### 2.2 昆仑 XPU
百度昆仑芯 AI 计算处理器（Baidu KUNLUN AI Computing Processor）是百度集十年 AI 产业技术实践于 2019 年推出的全功能 AI 芯片。基于自主研发的先进 XPU 架构，为云端和边缘端的人工智能业务而设计。 百度昆仑芯与飞桨及其他国产软硬件强强组合，打造一个全面领先的国产化 AI 技术生态，部署和应用于诸多 “人工智能+“的行业领域，包括智能云和高性能计算，智慧制造、智慧城市和安防等。
#### 2.2.1 环境准备
当前 PaddleNLP 支持昆仑 R200/R300 等芯片。考虑到环境差异性，我们推荐使用飞桨官方发布的昆仑 XPU 开发镜像，该镜像预装有昆仑基础运行环境库（XRE）。
- 1. 拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
```
docker pull registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 # X86 架构
docker pull registry.baidubce.com/device/paddle-xpu:kylinv10-aarch64-gcc82-py310 # ARM 架构
```
- 2. 参考如下命令启动容器
```
docker run -it --name=xxx -m 81920M --memory-swap=81920M \
    --shm-size=128G --privileged --net=host \
    -v $(pwd):/workspace -w /workspace \
    registry.baidubce.com/device/paddle-xpu:$(uname -m)-py310 bash
```
#### 2.2.2 安装 paddle 包
当前提供 Python3.10 的 wheel 安装包。有其他 Python 版本需求，可以参考[飞桨官方文档](https://www.paddlepaddle.org.cn/install/quick)自行编译安装。

- 1. 安装 Python3.10 的 wheel 安装包
```
pip install https://paddle-whl.bj.bcebos.com/paddlex/xpu/paddlepaddle_xpu-2.6.1-cp310-cp310-linux_x86_64.whl # X86 架构
pip install https://paddle-whl.bj.bcebos.com/paddlex/xpu/paddlepaddle_xpu-2.6.1-cp310-cp310-linux_aarch64.whl # ARM 架构
```
- 2. 验证安装包
安装完成之后，运行如下命令
```
python -c "import paddle; paddle.utils.run_check()"
```
预期得到如下输出结果
```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
### 2.3 寒武纪 MLU
思元 370 基于寒武纪芯片架构 MLUarch03 设计，是寒武纪（Cambricon）推出的人工智能领域高能效的通用智能芯片，支持 MLU-Link™多芯互联技术，可高效执行多芯多卡训练和分布式推理任务。
#### 2.3.1 环境准备
当前 PaddleNLP 支持寒武纪 MLU370X8 芯片。考虑到环境差异性，我们推荐使用飞桨官方提供的标准镜像完成环境准备。
- 1. 拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
```
# 适用于 X86 架构，暂时不提供 Arch64 架构镜像
docker pull registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-gcc84-py310
```
- 2. 参考如下命令启动容器
```
docker run -it --name paddle-mlu-dev -v $(pwd):/work \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-gcc84-py310 /bin/bash
```
#### 2.3.2 安装 paddle 包
当前提供 Python3.10 的 wheel 安装包。有其他 Python 版本需求，可以参考[飞桨官方文档](https://www.paddlepaddle.org.cn/install/quick)自行编译安装。

- 1. 下载安装 Python3.10 的 wheel 安装包。
```
# 注意需要先安装飞桨 cpu 版本
python -m pip install paddlepaddle==3.0.0.dev20240624 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -m pip install paddle-custom-mlu==3.0.0.dev20240806 -i https://www.paddlepaddle.org.cn/packages/nightly/mlu/
```
- 2. 验证安装包
安装完成之后，运行如下命令。
```
python -c "import paddle; paddle.utils.run_check()"
```
预期得到如下输出结果
```
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 mlu.
PaddlePaddle works well on 16 mlus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
