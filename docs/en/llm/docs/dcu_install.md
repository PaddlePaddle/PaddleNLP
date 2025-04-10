## 🚣‍♂️ Running llama2-7b Model on DCU(K100_AI) with PaddleNLP 🚣

PaddleNLP has implemented deep adaptation and optimization for the llama series models on Hygon DCU-K100AI chips. This document explains the process of performing high-performance inference for llama series models using PaddleNLP on DCU-K100_AI.

### Hardware Verification:

| Chip Type | Driver Version |
| --- | --- |
| K100_AI | 6.2.17a |

**Note: To verify if your machine uses Hygon K100-AI chip, execute the following command in system environment:**
```
lspci | grep -i -E "disp|co-pro"

# Expected output:
37:00.0 Co-processor: Chengdu Haiguang IC Design Co., Ltd. Device 6210 (rev 01)
3a:00.0 Co-processor: Chengdu Haiguang IC Design Co., Ltd. Device 6210 (rev 01)
```

### Environment Setup:
Note: K100_AI chip requires DTK 24.04 or higher. Follow these steps:

1. Pull the Docker image
```
# Note: This image is for development environment only, PaddlePaddle package not included
docker pull registry.baidubce.com/device/paddle-dcu:dtk24.04.1-kylinv10-gcc73-py310
```

2. Start container with following command
```
docker run -it --name paddle-dcu-dev -v `pwd`:/work \
  -w=/work --shm-size=128G --network=host --privilegged \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  registry.baidubce.com/device/paddle-dcu:dtk24.04.1-kylinv10-gcc73-py310 /bin/bash
```

3. Install PaddlePaddle
(Continue with remaining content as needed)
```
# PaddlePaddle Deep Learning Framework, providing fundamental computing capabilities
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle && mkdir build && cd build

cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_FLAGS="-Wno-error -w" \
  -DPY_VERSION=3.10 -DPYTHON_EXECUTABLE=`which python3` -DWITH_CUSTOM_DEVICE=OFF \
  -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_DISTRIBUTE=ON -DWITH_MKL=ON \
  -DWITH_ROCM=ON -DWITH_RCCL=ON

make -j128
pip install -U python/dist/paddlepaddle_rocm-0.0.0-cp310-cp310-linux_x86_64.whl

# Verify installation
python -c "import paddle; paddle.version.show()"
python -c "import paddle; paddle.utils.run_check()"

```
4. Clone PaddleNLP repository and install dependencies
```
# PaddleNLP is a natural language processing and large language model (LLM) development library based on PaddlePaddle, containing various large models implemented with PaddlePaddle, including llama series models. To facilitate your better use of PaddleNLP, you need to clone the entire repository.
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```
5. Install paddlenlp_ops
```
# The PaddleNLP repository includes dedicated fused operators to enable users to enjoy extremely optimized inference costs
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/csrc/
python setup_hip.py install
cd -
```

### High-performance Inference:
The inference commands for Hygon DCU are consistent with GPU inference commands. Please refer to the [Large Model Inference Tutorial](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/inference.md).
