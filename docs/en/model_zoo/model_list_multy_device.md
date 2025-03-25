# Multi-Hardware Natural Language Understanding Model List

This document focuses on the Ascend NPU, Cambricon MLU, and Kunlun XPU hardware platforms, introducing the natural language understanding models supported by PaddleNLP and their usage methods.

## 1. Model List

| Model Name/Hardware Support | NPU | XPU | MLU |
| - | - | - | - |
| [BERT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/bert) | ✅ | ✅ | ✅ |
| [ERNIE-3.0](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/ernie-3.0) | ✅ | ❌ | ❌ |
| [UIE](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/uie) | ✅ | ❌ | ❌ |
| [UTC](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/applications/zero_shot_text_classification) | ✅ | ❌ | ❌ |
| [RoBERTa](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/slm/model_zoo/roberta) | ✅ | ❌ | ❌ |

## 2. Hardware Usage Guides

First install the PaddlePaddle environment on the hardware platform, then refer to the usage methods in the model documentation. Simply change the device parameter to the corresponding hardware platform.

### 2.1 Ascend NPU

The Ascend 910 series is a high-efficiency, flexible and programmable AI processor from Huawei Ascend. It adopts the self-developed Huawei Da Vinci architecture, integrates rich computing units, improves AI computing completeness and efficiency, thereby expanding the chip's applicability.

#### 2.1.1 Environment Preparation

Currently, PaddleNLP supports Ascend 910B chips (more models are being supported; if you have requirements for other models, please submit an issue). The Ascend driver version is 23.0.3. Considering environmental differences, we recommend using the standard image provided by PaddlePaddle for environment preparation.

1. Pull the image (this image is for development environment only; it does not contain precompiled PaddlePaddle packages, but includes the Ascend operator library CANN-8.0.T13 by default):

```
# For X86 architecture
docker pull registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-x86_64-gcc84-py39
# For Aarch64 architecture
docker pull registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-aarch64-gcc84-py39
```

2. Start the container with the following command, where ASCEND_RT_VISIBLE_DEVICES specifies visible NPU card numbers:
```
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-$(uname -m)-gcc84-py39 /bin/bash
```

#### 2.1.2 Install Paddle Packages
Currently providing Python3.9 wheel packages. For other Python version requirements, please refer to the [PaddlePaddle Official Documentation](https://www.paddlepaddle.org.cn/install/quick) for self-compilation and installation.

- 1. Download and install Python3.9 wheel packages

```
# Note: Need to install CPU version of PaddlePaddle first
python3.9 -m pip install paddlepaddle==3.0.0.dev20240520 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python3.9 -m pip install paddle_custom_npu==3.0.0.dev20240719 -i https://www.paddlepaddle.org.cn/packages/nightly/npu/
```

- 2. Verify installation
After installation, run the following command.
```
python -c "import paddle; paddle.utils.run_check()"
```
Expected output:
```
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 npu.
PaddlePaddle works well on 8 npus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
### 2.2 Kunlun XPU
Baidu Kunlun AI Computing Processor is a full-featured AI chip launched by Baidu in 2019, integrating a decade of AI industry technology practices. Designed for cloud and edge AI applications, it is based on Baidu's self-developed advanced XPU architecture. Kunlun XPU, combined with PaddlePaddle and other domestic software/hardware ecosystems, builds a comprehensive leading localized AI technology ecosystem. It has been deployed in various "AI+" industry fields including intelligent cloud, high-performance computing, smart manufacturing, smart cities, and security.

#### 2.2.1 Environment Preparation
Currently PaddleNLP supports Kunlun R200/R300 chips. Considering environmental differences, we recommend using the official PaddlePaddle Kunlun XPU development image, which comes pre-installed with the Kunlun base runtime library (XRE).

- 1. Pull the development image (note: this image does not contain pre-compiled Paddle packages)
```
docker pull registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 # For x86 architecture
docker pull registry.baidubce.com/device/paddle-xpu:kylinv10-aarch64-gcc82-py310 # For ARM architecture
```

- 2. Start container with the following command
```
docker run -it --name=xxx -m 81920M --memory-swap=81920M \
    --shm-size=128G --privileged --net=host \
    -v $(pwd):/workspace -w /workspace \
    registry.baidubce.com/device/paddle-xpu:$(uname -m)-py310 bash
```

#### 2.2.2 Install Paddle Package
We provide Python 3.10 wheel packages. For other Python versions, please refer to [PaddlePaddle Official Documentation](https://www.paddlepaddle.org.cn/install/quick) for self-compilation.

- 1. Install Python 3.10 wheel package
```
pip install https://paddle-whl.bj.bcebos.com/paddlex/xpu/paddlepaddle_xpu-2.6.1-cp310-cp310-linux_x86_64.whl # For x86 architecture
pip install https://paddle-whl.bj.bcebos.com/paddlex/xpu/paddlepaddle_xpu-2.6.1-cp310-cp310-linux_aarch64.whl # For ARM architecture
```

- 2. Verify installation
After installation, run:
```
python -c "import paddle; paddle.utils.run_check()"
```
Expected output:
```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
### 2.3 Cambricon MLU

The Siyuan 370 is a high-performance general-purpose AI chip developed by Cambricon, based on the MLUarch03 architecture. It supports MLU-Link™ multi-core interconnect technology, enabling efficient multi-chip and multi-card training and distributed inference tasks.

#### 2.3.1 Environment Preparation

PaddleNLP currently supports Cambricon MLU370X8 chips. Considering environmental differences, we recommend using the official PaddlePaddle standard images for environment setup.

- 1. Pull the development environment image (this image doesn't include precompiled PaddlePaddle packages):
```
# For X86 architecture (ARM64 architecture not currently supported)
docker pull registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-gcc84-py310
```

- 2. Start the container with the following command:
```bash
docker run -it --name paddle-mlu-dev -v $(pwd):/work \
  -w=/work --shm-size=128G --network=host --privileged \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-gcc84-py310 /bin/bash
```

#### 2.3.2 Install Paddle Packages

We currently provide Python 3.10 wheel packages. For other Python versions, please refer to the [PaddlePaddle official documentation](https://www.paddlepaddle.org.cn/install/quick) for compilation and installation.

- 1. Download and install Python 3.10 wheel packages:
```
# First install CPU version of PaddlePaddle
python -m pip install paddlepaddle==3.0.0.dev20240624 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -m pip install paddle-custom-mlu==3.0.0.dev20240806 -i https://www.paddlepaddle.org.cn/packages/nightly/mlu/
```

- 2. Verify installation:
After installation, run the following command:
```bash
python -c "import paddle; paddle.utils.run_check()"
```

Expected output:
```
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 mlu.
PaddlePaddle works well on 16 mlus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
