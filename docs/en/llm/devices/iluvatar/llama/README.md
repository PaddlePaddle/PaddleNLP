# Run llama-13b Model Pre-training on Tiangai-150 Using PaddleNLP

Tiangai-150 accelerator card ([Learn about Iluvatar](https://www.iluvatar.com/)) is a training-inference integrated accelerator card based on Iluvatar's self-developed general-purpose GPU. It features broad applicability, strong flexibility, and high cost-performance advantages. Supporting mainstream ecosystems, it can be widely used in mainstream large model pre-training, fine-tuning, and inference tasks, as well as general computing and new algorithm research scenarios, empowering AI-driven society.

PaddleNLP has conducted deep adaptation and optimization for llama-13B model on Tiangai-150, essentially achieving unified training-inference interfaces between Tiangai-150 and GPU. This enables users to seamlessly migrate applications between Tiangai-150 and GPU platforms.

## 🚀 Quick Start 🚀

### (0) Before starting, you need four machines equipped with Tiangai-150 accelerator cards. The system requirements for these machines are as follows:

| Chip Type | Driver Version | SDK Version   |
| --------- | -------------- | ------------- |
| Tiangai 150 | ≥ 4.1.0        | ≥ 4.1.0       |

**Note: To verify if your machine is equipped with Tiangai-150, simply enter the following command in the system environment and check the output:**

```bash
lspci | grep -i iluvatar
```
```bash
ixsmi

# Output as follows

Timestamp    Fri Dec 20 11:14:29 2024
+-----------------------------------------------------------------------------+
|  IX-ML: 4.1.0       Driver Version: 4.1.0       CUDA Version: 10.2          |
|-------------------------------+----------------------+----------------------|
| GPU  Name                     | Bus-Id               | Clock-SM  Clock-Mem  |
| Fan  Temp  Perf  Pwr:Usage/Cap|      Memory-Usage    | GPU-Util  Compute M. |
|===============================+======================+======================|
| 0    Iluvatar BI-V150         | 00000000:13:00.0     | 1500MHz   1600MHz    |
| 0%   32C   P0    N/A / N/A    | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 1    Iluvatar BI-V150         | 00000000:16:00.0     | 1500MHz   1600MHz    |
| 0%   30C   P0    93W / 350W   | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 2    Iluvatar BI-V150         | 00000000:1C:00.0     | 1500MHz   1600MHz    |
| 0%   31C   P0    N/A / N/A    | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 3    Iluvatar BI-V150         | 00000000:1F:00.0     | 1500MHz   1600MHz    |
| 0%   31C   P0    94W / 350W   | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 4    Iluvatar BI-V150         | 00000000:27:00.0     | 1500MHz   1600MHz    |
| 0%   30C   P0    N/A / N/A    | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 5    Iluvatar BI-V150         | 00000000:2A:00.0     | 1500MHz   1600MHz    |
| 0%   31C   P0    98W / 350W   | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 6    Iluvatar BI-V150         | 00000000:34:00.0     | 1500MHz   1600MHz    |
| 0%   31C   P0    N/A / N/A    | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 7    Iluvatar BI-V150         | 00000000:37:00.0     | 1500MHz   1600MHz    |
| 0%   31C   P0    95W / 350W   | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 8    Iluvatar BI-V150         | 00000000:3D:00.0     | 1500MHz   1600MHz    |
| 0%   32C   P0    N/A / N/A    | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 9    Iluvatar BI-V150         | 00000000:40:00.0     | 1500MHz   1600MHz    |
| 0%   32C   P0    95W / 350W   | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 10   Iluvatar BI-V150         | 00000000:48:00.0     | 1500MHz   1600MHz    |
| 0%   31C   P0    N/A / N/A    | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 11   Iluvatar BI-V150         | 00000000:4B:00.0     | 1500MHz   1600MHz    |
| 0%   31C   P0    94W / 350W   | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 12   Iluvatar BI-V150         | 00000000:54:00.0     | 1500MHz   1600MHz    |
| 0%   30C   P0    N/A / N/A    | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 13   Iluvatar BI-V150         | 00000000:57:00.0     | 1500MHz   1600MHz    |
| 0%   32C   P0    93W / 350W   | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 14   Iluvatar BI-V150         | 00000000:64:00.0     | 1500MHz   1600MHz    |
| 0%   30C   P0    N/A / N/A    | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 15   Iluvatar BI-V150         | 00000000:67:00.0     | 1500MHz   1600MHz    |
| 0%   30C   P0    94W / 350W   | 116MiB / 32768MiB    | 0%        Default    |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU        PID      Process name                                Usage(MiB) |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
### (1) Environment Preparation: (This will take 5~55 minutes)
1. Pull the Docker image
```bash
# Please contact Iluvatar CoreX customer support (services@iluvatar.com) to obtain the SDK image

docker pull 10.150.9.98:80/sw_test/sw_home:4.1.0.20240528.110-x86_64-py3.10-bi150
```

2. Start the container with the following command

```bash
docker run -e USER=`id -u -n` -e USER_ID=`id -u` --name paddle-corex-dev -it --privileged --cap-add=ALL --pid=host --network=host -v /data1:/data1 --mount type=volume,dst=/home/`id -u -n`/.local bdkw:paddle-corex /bin/bash
```

3. Install PaddlePaddle

① If you have already obtained the PaddlePaddle installation package through Iluvatar CoreX (services@iluvatar.com), you can install it directly:

```bash
pip3 install paddlepaddle-2.5.2+corex*.whl
```
② You can also compile the PaddlePaddle installation package from source. Please ensure you have correctly installed the Iluvatar CoreX software stack.

```bash
# 1. Clone the Paddle4CoreX GitHub repository and switch to the BDKW/2.5.2_corex branch.
git clone --recurse-submodules -b BDKW/2.5.2_corex https://github.com/PaddlePaddle/Paddle4CoreX.git
# 2. Execute the compilation script
bash build_paddle.sh
# 3. Run the installation script after compilation completes
bash install_paddle.sh
```

4. Clone the PaddleNLP repository and install dependencies

```bash
# PaddleNLP is a natural language processing and large language model (LLM) development library based on PaddlePaddle. It contains various large models implemented using PaddlePaddle, including the llama-13B model. To facilitate your usage, you need to clone the entire repository.
git clone -b sci/benchmark_iluvatar https://github.com/tianyuzhou668/PaddleNLP.git
# Compile custom operators (optional)
cd PaddleNLP
cd ./model_zoo/gpt-3/external_ops/ && python3 setup.py install && cd -
# Install PaddleNLP
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

### (2) Data Preparation: (This will take 2~5 minutes)
To facilitate users in running tests for this model, we provide a pre-processed 100k doc training sample:

```bash
wget https://paddlenlp.bj.bcebos.com/data/gpt_en_dataset_300m_ids.npy
wget https://paddlenlp.bj.bcebos.com/data/gpt_en_dataset_300m_idx.npz
```
```bash
# Downloading Llama Model Data
cd ./llm
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx
```

You can also download the complete dataset:
```bash
cd ./llm
wget https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/llama/mmap/llama_mmap.bin
wget https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/llama/mmap/llama_mmap.idx
```

Place all preprocessed files into a single directory for training:
```bash
mkdir data
mv llama_openwebtext_100k_ids.npy ./data
mv llama_openwebtext_100k_idx.npz ./data
```

### (3) Pre-training:
We provide a corresponding 4-node pre-training script in this directory, which has been optimized with parallel strategies and configurations for 32 BI150 chip resources. The detailed steps to start pre-training are as follows:
```bash
# You need to replace the node IP addresses in the script with your actual node IPs
bash run_node4.sh
```

By default, we will train for 5000 steps. After completing the 5000-step training, the model will perform evaluation and generate the final checkpoint. You can use this checkpoint for fine-tuning tasks or directly proceed with inference tasks.
