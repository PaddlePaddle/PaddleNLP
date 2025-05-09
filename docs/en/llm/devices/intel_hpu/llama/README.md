# Run llama2-7b Model on Intel HPU with PaddleNLP
PaddleNLP has deeply adapted and optimized the llama2-7B model on Intel® Gaudi®2D ([Learn about Gaudi](https://docs.habana.ai/en/latest/index.html)). The detailed installation steps are provided below.

## 🚀 Quick Start 🚀

### (0) Before starting, you need an Intel Gaudi machine with the following system requirements:

| Chip Type | Card Model | Driver Version |
| --- | --- | --- |
| Gaudi | 225D | 1.17.0 |

### (1) Environment Preparation: (This will take 5-15 minutes)
1. Pull the image
```
# Note: This image is only a development environment and does not contain precompiled PaddlePaddle packages
docker pull vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest
```

2. Start the container with the following command
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest
```

3. Install PaddlePaddle
```
# PaddlePaddle, the deep learning framework, provides fundamental computing capabilities
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
```

4. Install paddleCustomDevice
```
# paddleCustomDevice is the custom hardware backend implementation for PaddlePaddle, providing operator implementations for Intel HPU.
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

5. Clone PaddleNLP repository and install dependencies
# PaddleNLP is a natural language processing and large language model (LLM) development library based on PaddlePaddle. It contains various large models implemented with PaddlePaddle, including the llama2-7B model. To help you better utilize PaddleNLP, you need to clone the entire repository.

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
python -m pip install -r requirements.txt
python -m pip install -e .
```

### (2) Inference: (This will take 10-15 minutes)
1. Single-GPU Inference

Execute the following command for inference:
```bash
python inference_hpu.py
```

After successful execution, you can view the generated inference results. A sample output is shown below:
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
2. Multi-GPU Inference

Execute the following command to perform inference:
```bash
bash test_llama_2x.sh
```
After a successful run, you can view the generated inference results. A sample is shown below:
```bash
[2024-10-29 11:24:39,468] [    INFO] - We are using <class 'paddlenlp.transformers.llama.tokenizer.LlamaTokenizer'> to load 'meta-llama/Llama-2-7b-chat'.
[2024-10-29 11:24:40,705] [    INFO] distributed_strategy.py:214 - Distributed strategy initialized
I1029 11:24:40.706755 14711 tcp_utils.cc:181] The server starts to listen on IP_ANY:59129
I1029 11:24:40.706897 14711 tcp_utils.cc:130] Successfully connected to 127.0.0.1:59129
[2024-10-29 11:24:42,740] [    INFO] topology.py:357 - Total 2 pipe comm group(s) created successfully!
[2024-10-29 11:24:52,064] [    INFO] topology.py:357 - Total 2 data comm group(s) created successfully!
[2024-10-29 11:24:52,064] [    INFO] topology.py:357 - Total 1 model comm group(s) created successfully!
[2024-10-29 11:24:52,065] [    INFO] topology.py:357 - Total 2 sharding comm group(s) created successfully!
[2024-10-29 11:24:52,065] [    INFO] topology.py:279 - HybridParallelInfo: rank_id: 0, mp_degree: 2, sharding_degree: 1, pp_degree: 1, dp_degree: 1, sep_degree: 1, mp_group: [0, 1], sharding_group: [0], pp_group: [0], dp_group: [0], sep_group: None, check/clip group: [0, 1]
[2024-10-29 11:24:52,067] [    INFO] - We are using <class 'paddlenlp.transformers.llama.modeling.LlamaForCausalLM'> to load 'meta-llama/Llama-2-7b-chat'.
[2024-10-29 11:24:52,067] [    INFO] - Loading configuration file /root/.paddlenlp/models/meta-llama/Llama-2-7b-chat/config.json
[2024-10-29 11:24:52,068] [    INFO] - Loading weights file from cache at /root/.paddlenlp/models/meta-llama/Llama-2-7b-chat/model_state.pdparams
[2024-10-29 11:25:43,202] [    INFO] - Starting to convert original state_dict to tensor parallel state_dict.
[2024-10-29 11:25:45,125] [    INFO] - Loaded weights file from disk, setting weights to model.
[2024-10-29 11:26:04,008] [    INFO] - All model checkpoint weights were used when initializing LlamaForCausalLM.
[2024-10-29 11:26:04,008] [    INFO] - All the weights of LlamaForCausalLM were initialized from the model checkpoint at meta-llama/Llama-2-7b-chat.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
[2024-10-29 11:26:04,010] [    INFO] - Loading configuration file /root/.paddlenlp/models/meta-llama/Llama-2-7b-chat/generation_config.json

['myself\nHello everyone my name is [Your Name], and I am a new member of this community']
I1029 11:26:16.184163 14767 tcp_store.cc:293] Receive shutdown event and so quit from MasterDaemon run loop
LAUNCH INFO 2024-10-29 11:26:17,186 Pod completed
LAUNCH INFO 2024-10-29 11:26:17,186 Exit code 0
```
Understood. I will strictly follow your instructions to perform the translation while maintaining all formatting, technical terms, code/math blocks, and link anchors. The output will be a professional technical translation with proper academic grammar.
