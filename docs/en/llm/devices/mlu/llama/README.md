# Run llama-13b Model on MLU with PaddleNLP
PaddleNLP has conducted deep adaptation and optimization for the llama-13B model on Cambricon MLU ([Learn about Cambricon](https://www.cambricon.com/)). This toolkit achieves basic unification of training and inference entry points between Cambricon MLU and GPU, realizing the effect of "seamless switching".

## 🚀 Quick Start 🚀

### (0) Before starting, you need to have a Cambricon MLU machine. The system requirements for this machine are as follows:

| Chip Type | Driver Version | CNtoolKit Version |
| --- | --- | --- |
| MLU | 5.10.31 | 3.10.2 |

**Note: This example uses an 8-card machine and demonstrates the workflow through fine-tuning training + inference**
**Note: To verify whether your machine has Cambricon chips, simply enter the following command in the system environment and check the output:**
```
cnmon

# Example: $ cnmon, the output is as follows
Thu Dec 19 22:05:42 2024
+------------------------------------------------------------------------------+
| CNMON v5.10.31                                               Driver v5.10.31 |
+-------------------------------+----------------------+-----------------------+
| Card  VF  Name       Firmware |               Bus-Id | Util        Ecc-Error |
| Fan   Temp      Pwr:Usage/Cap |         Memory-Usage | Mode     Compute-Mode |
|===============================+======================+=======================|
| 0     /   MLUXXX-XX    v1.5.0 |         0000:4F:00.0 | 0%                  0 |
|  0%   35C        105 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 1     /   MLUXXX-XX    v1.5.0 |         0000:53:00.0 | 0%                  0 |
|  0%   34C        100 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 2     /   MLUXXX-XX    v1.5.0 |         0000:6F:00.0 | 0%                  0 |
|  0%   35C        100 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 3     /   MLUXXX-XX    v1.5.0 |         0000:73:00.0 | 0%                  0 |
|  0%   34C        109 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 4     /   MLUXXX-XX    v1.5.0 |         0000:AF:00.0 | 0%                  0 |
|  0%   34C        107 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 5     /   MLUXXX-XX    v1.5.0 |         0000:B3:00.0 | 0%                  0 |
|  0%   33C        105 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 6     /   MLUXXX-XX    v1.5.0 |         0000:CF:00.0 | 0%                  0 |
|  0%   36C        102 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 7     /   MLUXXX-XX    v1.5.0 |         0000:D3:00.0 | 0%                  0 |
|  0%   33C        105 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+

+------------------------------------------------------------------------------+
| Processes:                                                                   |
|  Card  MI  PID     Command Line                             MLU Memory Usage |
|==============================================================================|
|  No running processes found                                                  |
+------------------------------------------------------------------------------+
```
### (1) Environment Preparation: (This will take you 5~15 minutes)
1. Pull the Docker image
```
# Note: This image is for development environment only, the image does not contain precompiled PaddlePaddle packages
docker pull registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-x86_64-gcc84-py310
```
2. Start the container with the following command
```
docker run -it --name paddle-mlu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    --device /dev/cambricon_dev0 \
    --pid=host --ipc=host -it --privileged \
    -v -v /usr/bin/cnmon/:/usr/bin/cnmon/ \
    -v /usr/local/dcmi:/usr/local/dcmi \
    registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-x86_64-gcc84-py310 /bin/bash
```
3. Install PaddlePaddle
```
# paddlepaddle, the PaddlePaddle deep learning framework, provides foundational computing capabilities
pip install paddlepaddle==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```
4. Install paddleCustomDevice
```
# paddleCustomDevice is the custom hardware implementation for PaddlePaddle deep learning framework, providing MLU operator implementations.
pip install https://paddle-device.bj.bcebos.com/2.6.1/mlu/paddle_custom_mlu-2.6.1-cp310-cp310-linux_x86_64.whl
# For source compilation, please refer to https://github.com/PaddlePaddle/PaddleCustomDevice/blob/release/2.6/backends/mlu/README_cn.md
```
5. Clone PaddleNLP repository and install dependencies
```
# PaddleNLP is a natural language processing and large language model (LLM) development library based on PaddlePaddle, containing various large models including llama2-13B. To better utilize PaddleNLP, you need to clone the entire repository.
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
git checkout 1fc942924df46c8e149ac7ce8cbc42d884fbb823
python -m pip install -r requirements.txt
python -m pip install -e .
```

### (2) Pretraining Phase Data Preparation: (This will take you 8~9 minutes)
```
# Download OpenWebtext2 Dataset
mkdir openwebtext2 && cd openwebtext2
wget https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/llama/mmap/llama_mmap.bin
wget https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/llama/mmap/llama_mmap.idx
```

### (3) Model Download: (This will take 6-7 minutes)
```
# Randomly initialize model using this initialization (__internal_testing__/sci-benchmark-llama-13b-init0501) for training

python download_init0501_model.py
```

### (4) Model Pretraining: (This will take approximately 5 days)
Current configuration is for 4 machines. Users need to adjust based on their own machine setup, including machine IP and batch size.
```
# Machine 1
bash run_train.sh

# Machine 2
ssh notebook-devenviron-1104-202919-b065xu-worker-0
bash run_train.sh

# Machine 3
ssh notebook-devenviron-1104-202919-b065xu-worker-1
bash run_train.sh

# Machine 4
ssh notebook-devenviron-1104-202919-b065xu-worker-2
bash run_train.sh
```

### (5) Distributed Training Parameter Merging: (This will take 1-2 minutes)
```
# Merge distributed training parameters. After execution, 25G model_state.pdparams will be generated in ./checkpoints/llama_pretrain_ckpts/checkpoint-5000/
bash run_merge.sh
```

### (6) Post-Pretraining Model Accuracy Validation: (This will take 14-15 minutes)
Use provided benchmark script to test on given validation set.
```
bash run_eval.sh
```

### (7) Pretrained Model Evaluation: (This will take 15-16 minutes)
Use provided benchmark script to test on LAMBADA test dataset.
```
# Dataset preparation
mkdir wiki_lambada && cd wiki_lambada
wget https://paddlenlp.bj.bcebos.com/data/benchmark/lambada_test.jsonl
cd -

bash run_acc.sh
```

### (8) Fine-tuned Model Evaluation (SFT+LORA): (This will take approximately 5 days)
Download datasets meta-math/MetaMathQA, sahil2801/CodeAlpaca-20k, and Open-Orca/SlimOrca. Place these 3 datasets into specified directories: ./data_math, ./data_code, ./data_slim respectively.
Dataset download link: https://pan.baidu.com/s/1tbGYBqdmlrBq3vP_-WAIQA Password: a5eu
```
#1. meta-math/MetaMathQA Task
bash run_math_lora.sh
bash run_math_sft.sh

#2. sahil2801/CodeAlpaca-20k Task
bash run_code_lora.sh
bash run_code_sft.sh

#3. Open-Orca/SlimOrca Task
bash run_slim_lora.sh
bash run_slim_sft.sh
```
