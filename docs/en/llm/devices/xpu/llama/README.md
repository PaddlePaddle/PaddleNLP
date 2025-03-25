# Running llama2-7b Model on XPU with PaddleNLP
PaddleNLP has been deeply adapted and optimized for llama2-7B model on Kunlun XPU ([Learn about Kunlun](https://www.kunlunxin.com/)). Below are detailed installation steps.

## 🚀 Quick Start 🚀

### (0) Before starting, you need a Kunlun XPU machine with the following system requirements:

| Chip Type | Card Model | Driver Version |
| --- | --- | --- |
| Kunlun R480 | R300 | 4.31.0 |

#### Environment Dependencies
- **Machine:** Kunlun R480 32G, requires approximately 17.5G (bs=1)
- **Image:** registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310
- **GCC Path:** /usr/bin/gcc (8.4)
- **Python Version:** 3.10

**Note: This example uses 8-card machine. To verify if your machine has Kunlun chips, run:**

```
lspci | grep 1d22
# Example: $ lspci | grep 1d22, output shows:
53:00.0 Communication controller: Device 1d22:3684
56:00.0 Communication controller: Device 1d22:3684
6d:00.0 Communication controller: Device 1d22:3684
70:00.0 Communication controller: Device 1d22:3684
b9:00.0 Communication controller: Device 1d22:3684
bc:00.0 Communication controller: Device 1d22:3684
d2:00.0 Communication controller: Device 1d22:3684
d5:00.0 Communication controller: Device 1d22:3684
```

### (1) Environment Preparation: (This will take 5-15 minutes)

1. Pull the image
```
# Note: This image is only for development environment and does not contain precompiled PaddlePaddle packages
docker pull registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310
```

2. Start container with reference commands
```bash
docker run -it --privileged=true  --net host --device=/dev/xpu0:/dev/xpu0 --device=/dev/xpu1:/dev/xpu1 --device=/dev/xpu2:/dev/xpu2 --device=/dev/xpu3:/dev/xpu3 --device=/dev/xpu4:/dev/xpu4 --device=/dev/xpu5:/dev/xpu5 --device=/dev/xpu6:/dev/xpu6 --device=/dev/xpu7:/dev/xpu7 --device=/dev/xpuctrl:/dev/xpuctrl --name paddle-xpu-dev -v $(pwd):/work -w=/work -v xxx registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 /bin/bash
```

3. Install paddlepaddle-xpu
```bash
# PaddlePaddle『Paddle』Deep Learning Framework, providing fundamental computing capabilities
wget https://paddle-whl.bj.bcebos.com/nightly/xpu/paddlepaddle-xpu/paddlepaddle_xpu-3.0.0.dev20240612-cp310-cp310-linux_x86_64.whl
python -m pip install paddlepaddle_xpu-3.0.0.dev20240612-cp310-cp310-linux_x86_64.whl

Nightly build links:
https://www.paddlepaddle.org.cn/packages/nightly/xpu/paddlepaddle-xpu/
```

4. Clone PaddleNLP repository and install dependencies
# PaddleNLP is a natural language processing and large language model (LLM) development library based on PaddlePaddle. It contains various large models implemented using the Paddle framework, including the llama2-7B model. To help you better utilize PaddleNLP, you need to clone the entire repository.

# Clone PaddleNLP
git clone https://github.com/PaddlePaddle/PaddleNLP
cd PaddleNLP
# Switch to the specified commit with corresponding dependencies
git checkout 0844a5b730c636ad77975fd30a485ad5dc217eac
# Install dependencies
pip install -r requirements.txt
python -m pip install -e .

# Download XPU custom operators
cd csrc/xpu/src
# Download XDNN, XRE and XTDK with one click after setting paths
wget https://baidu-kunlun-product.su.bcebos.com/KL-SDK/klsdk-dev/release_paddle/20240429/xdnn-ubuntu_x86_64.tar.gz
wget https://baidu-kunlun-product.su.bcebos.com/KL-SDK/klsdk-dev/release_paddle/20240429/xre-ubuntu_x86_64.tar.gz
wget https://klx-sdk-release-public.su.bcebos.com/xtdk_llvm15/release_paddle/2.7.98.2/xtdk-llvm15-ubuntu1604_x86_64.tar.gz

# Extract to current directory
tar -xf xdnn-ubuntu_x86_64.tar.gz
tar -xf xre-ubuntu_x86_64.tar.gz
tar -xf xtdk-llvm15-ubuntu1604_x86_64.tar.gz

# Set environment variables
export PWD=$(pwd)
export XDNN_PATH=${PWD}/xdnn-ubuntu_x86_64/
export XRE_PATH=${PWD}/xre-ubuntu_x86_64/
export CLANG_PATH=${PWD}/xtdk-llvm15-ubuntu1604_x86_64/

# Install custom operators for XPU devices
bash ./cmake_build.sh
cd -

### (2) Data Preparation (This will take 2-5 minutes):
For fine-tuning: To facilitate testing, we also provide a ready-to-use dataset:
```
# Enter llm directory
cd llm
# Download dataset
wget https://baidu-kunlun-customer.su.bcebos.com/paddle-llm/infernce.tar.gz
# Extract
tar -zxvf infernce.tar.gz
```

### (3) Inference (This will take 10-15 minutes):
```
# Specify visible Kunlun chips by setting FLAGS_selected_xpus
export FLAGS_selected_xpus=0
# Set environment variables
export PYTHONPATH=$PYTHONPATH:../../../PaddleNLP/
```

High-performance dynamic graph inference command reference:
```python predictor.py --model_name_or_path ./inference --dtype float16 --src_length 2048 --max_length 2048 --mode "static" --batch_size 1 --inference_model --block_attn --device xpu
```

Expected result:
```
[[2024-08-22 13:23:34,969] [    INFO] - preprocess spend 0.012732744216918945
[2024-08-22 13:23:34,994] [    INFO] - We are using <class 'paddlenlp.transformers.llama.tokenizer.LlamaTokenizer'> to load './inference'.
[2024-08-22 13:23:35,014] [    INFO] - Start read result message
[2024-08-22 13:23:35,014] [    INFO] - Current path is /home/workspace/wangy_test/PaddleNLP/llm
[2024-08-22 13:23:53,313] [    INFO] - running spend 18.322898864746094
[2024-08-22 13:23:53,326] [    INFO] - Finish read result message
[2024-08-22 13:23:53,327] [    INFO] - End predict
***********Source**********
Explain "Reviewing the past to learn the new"
***********Target**********

***********Output**********
"Reviewing the past to learn the new" (wēn gù ér zhī xīn) is a Chinese proverb that emphasizes the importance of understanding historical knowledge to better comprehend new concepts. The phrase consists of two parts: "温故" (wēn gù) meaning "to review the old", and "知新" (zhī xīn) meaning "to learn the new". This proverb suggests that by re-examining and consolidating previous knowledge, one can gain deeper insights and make more informed judgments when encountering new information or situations.

In practical terms, this means:
1. Foundational knowledge serves as a basis for understanding new developments
2. Historical context helps prevent hasty dismissal of traditional wisdom
3. A balanced approach allows for integration of old and new knowledge

For example:
- In language learning, studying etymologies and historical grammar can enhance understanding of modern usage
- In technical fields, mastering fundamental principles is crucial before exploring advanced innovations

This concept aligns with modern educational philosophies that emphasize building upon prior knowledge and the spiral learning approach. It also reflects the importance of critical thinking in distinguishing between outdated and still-relevant traditional knowledge when absorbing new information.
[2024-08-22 13:23:53,328] [    INFO] - Start predict
[2024-08-22 13:23:53,335] [    INFO] - preprocess spend 0.007447242736816406
[2024-08-22 13:23:53,357] [    INFO] - We are using <class 'paddlenlp.transformers.llama.tokenizer.LlamaTokenizer'> to load './inference'.
[2024-08-22 13:23:53,386] [    INFO] - Start read result message
[2024-08-22 13:23:53,386] [    INFO] - Current path is /home/workspace/wangy_test/PaddleNLP/llm
[2024-08-22 13:23:57,859] [    INFO] - running spend 4.506801605224609
[2024-08-22 13:23:57,863] [    INFO] - Finish read result message
[2024-08-22 13:23:57,864] [    INFO] - End predict
***********Source**********
Hello, may I ask who you are?
***********Target**********

***********Output**********
Greetings! I'm an AI assistant developed using PaddleNLP framework. My purpose is to provide helpful and accurate information while maintaining safety and ethical standards. I don't possess personal identity or consciousness, but I'm designed to understand and process natural language effectively. How may I assist you today?
```
Certainly! Please provide the Chinese text you need translated, and I'll ensure the translation adheres to all your specified requirements while maintaining technical accuracy and formatting.
