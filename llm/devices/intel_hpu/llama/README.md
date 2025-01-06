## 🚣‍♂️ 使用 PaddleNLP 在 Intel HPU 下跑通 llama2-7b 模型 🚣
PaddleNLP 在 Intel® Gaudi®2D([了解 Gaudi](https://docs.habana.ai/en/latest/index.html))上对 llama2-7B 模型进行了深度适配和优化，下面给出详细安装步骤。

##  🚀 快速开始 🚀

### （0）在开始之前，您需要有一台 Intel Gaudi 机器，对此机器的系统要求如下：

 | 芯片类型 | 卡型号 | 驱动版本 |
 | --- | --- | --- |
 | Gaudi | 225D | 1.17.0 |


### （1）环境准备：(这将花费您5～15min 时间)
1. 拉取镜像
```
# 注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
docker pull vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest
```
2. 参考如下命令启动容器
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest
```
3. 安装 paddle
```
# paddlepaddle『飞桨』深度学习框架，提供运算基础能力
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
```
4. 安装 paddleCustomDevice
```
# paddleCustomDevice是paddlepaddle『飞桨』深度学习框架的自定义硬件接入实现，提供Intel HPU的算子实现。
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
git submodule sync
git submodule update --remote --init --recursive
cd backends/intel_hpu/
mkdir build && cd build
cmake ..
make -j8
pip install dist/paddle_intel_hpu-0.0.1-cp310-cp310-linux_x86_64.whl
```
5. 克隆 PaddleNLP 仓库代码，并安装依赖
```
# PaddleNLP是基于paddlepaddle『飞桨』的自然语言处理和大语言模型(LLM)开发库，存放了基于『飞桨』框架实现的各种大模型，llama2-7B模型也包含其中。为了便于您更好地使用PaddleNLP，您需要clone整个仓库。
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
python -m pip install -r requirements.txt
python -m pip install -e .
```

### （2）推理：(这将花费您10~15min 时间)
1. 单卡推理

执行如下命令进行推理：
```bash
python inference_hpu.py
```

成功运行后，可以查看到推理结果的生成，样例如下：
```
[2024-10-25 02:42:42,220] [    INFO] - We are using <class 'paddlenlp.transformers.llama.tokenizer.LlamaTokenizer'> to load 'meta-llama/Llama-2-7b-chat'.
[2024-10-25 02:42:42,427] [    INFO] - We are using <class 'paddlenlp.transformers.llama.modeling.LlamaForCausalLM'> to load 'meta-llama/Llama-2-7b-chat'.
[2024-10-25 02:42:42,427] [    INFO] - Loading configuration file /root/.paddlenlp/models/meta-llama/Llama-2-7b-chat/config.json
[2024-10-25 02:42:42,428] [    INFO] - Loading weights file from cache at /root/.paddlenlp/models/meta-llama/Llama-2-7b-chat/model_state.pdparams
[2024-10-25 02:43:32,871] [    INFO] - Loaded weights file from disk, setting weights to model.
[2024-10-25 02:44:15,226] [    INFO] - All model checkpoint weights were used when initializing LlamaForCausalLM.

[2024-10-25 02:44:15,226] [    INFO] - All the weights of LlamaForCausalLM were initialized from the model checkpoint at meta-llama/Llama-2-7b-chat.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
[2024-10-25 02:44:15,229] [    INFO] - Loading configuration file /root/.paddlenlp/models/meta-llama/Llama-2-7b-chat/generation_config.json

['myself. I am a 35 year old woman from the United States. I am a writer and artist, and I have been living in Japan for the past 5 years. I am originally from the Midwest, but I have lived in several different places around the world, including California, New York, and now Japan.\nI am passionate about many things, including art, writing, music, and travel. I love to explore new places and cultures, and I am always looking for new inspiration for my art and writing. I am also a big fan of Japanese culture, and I try to learn as much']
```
2. 多卡推理

执行如下命令进行推理：
```bash
bash test_llama_2x.sh
```
成功运行后，可以查看到推理结果的生成，样例如下：
```bash
[2024-10-29 11:24:39,468] [    INFO] - We are using <class 'paddlenlp.transformers.llama.tokenizer.LlamaTokenizer'> to load 'meta-llama/Llama-2-7b-chat'.
[2024-10-29 11:24:40,705] [    INFO] distributed_strategy.py:214 - distributed strategy initialized
I1029 11:24:40.706755 14711 tcp_utils.cc:181] The server starts to listen on IP_ANY:59129
I1029 11:24:40.706897 14711 tcp_utils.cc:130] Successfully connected to 127.0.0.1:59129
[2024-10-29 11:24:42,740] [    INFO] topology.py:357 - Total 2 pipe comm group(s) create successfully!
[2024-10-29 11:24:52,064] [    INFO] topology.py:357 - Total 2 data comm group(s) create successfully!
[2024-10-29 11:24:52,064] [    INFO] topology.py:357 - Total 1 model comm group(s) create successfully!
[2024-10-29 11:24:52,065] [    INFO] topology.py:357 - Total 2 sharding comm group(s) create successfully!
[2024-10-29 11:24:52,065] [    INFO] topology.py:279 - HybridParallelInfo: rank_id: 0, mp_degree: 2, sharding_degree: 1, pp_degree: 1, dp_degree: 1, sep_degree: 1, mp_group: [0, 1],  sharding_group: [0], pp_group: [0], dp_group: [0], sep:group: None, check/clip group: [0, 1]
[2024-10-29 11:24:52,067] [    INFO] - We are using <class 'paddlenlp.transformers.llama.modeling.LlamaForCausalLM'> to load 'meta-llama/Llama-2-7b-chat'.
[2024-10-29 11:24:52,067] [    INFO] - Loading configuration file /root/.paddlenlp/models/meta-llama/Llama-2-7b-chat/config.json
[2024-10-29 11:24:52,068] [    INFO] - Loading weights file from cache at /root/.paddlenlp/models/meta-llama/Llama-2-7b-chat/model_state.pdparams
[2024-10-29 11:25:43,202] [    INFO] - Starting to convert orignal state_dict to tensor parallel state_dict.
[2024-10-29 11:25:45,125] [    INFO] - Loaded weights file from disk, setting weights to model.
[2024-10-29 11:26:04,008] [    INFO] - All model checkpoint weights were used when initializing LlamaForCausalLM.
[2024-10-29 11:26:04,008] [    INFO] - All the weights of LlamaForCausalLM were initialized from the model checkpoint at meta-llama/Llama-2-7b-chat.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
[2024-10-29 11:26:04,010] [    INFO] - Loading configuration file /root/.paddlenlp/models/meta-llama/Llama-2-7b-chat/generation_config.json

['myself\nHello everyone my name is [Your Name], and I am a new member of this community']
I1029 11:26:16.184163 14767 tcp_store.cc:293] receive shutdown event and so quit from MasterDaemon run loop
LAUNCH INFO 2024-10-29 11:26:17,186 Pod completed
LAUNCH INFO 2024-10-29 11:26:17,186 Exit code 0
```
