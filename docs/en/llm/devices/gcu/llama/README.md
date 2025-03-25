# Running llama2-13b Model on Enflame S60 with PaddleNLP

Enflame S60 ([Learn about Enflame](https://www.enflame-tech.com/)) is a next-generation AI inference acceleration card for large-scale deployment in data centers, meeting the requirements of large language models, search & recommendation systems, and traditional models. It features broad model coverage, strong usability, easy migration and deployment, and can be widely applied to mainstream inference scenarios such as image/text generation, search & recommendation, and text/image/speech recognition.

PaddleNLP has deeply adapted and optimized the llama2-13B model on Enflame S60, achieving basic unification of the GCU inference interface with GPUs. Only device modification is required to complete inference task migration.

## 🚀 Quick Start 🚀

### 0. Machine Preparation. Before getting started, you need to prepare a machine with Enflame S60 acceleration cards, with the following requirements:

| Chip Type | Driver Version | TopsPlatform Version |
| :---: | :---: | :---: |
| Enflame S60 | 1.0.5.1 | TopsPlatform_1.0.5.1-2c3111 |

**Note: To verify if your machine has Enflame S60 acceleration cards, simply run the following command in the system environment and check the output:**
```bash
lspci | grep S60

# Example: lspci | grep S60, output:
01:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
09:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
```

### 1. Environment Setup: (This will take 10-20 minutes)

1. Initialize environment and install drivers<br/>
  **Note: You can contact Enflame (Email: developer-enflame@enflame-tech.com) to obtain the driver package and other assistance**
```bash
# Assuming the installation package is located at: /home/paddle_user/deps/, named: TopsPlatform.tar.gz
cd /home/paddle_user/deps/ && tar -zxf TopsPlatform.tar.gz
cd TopsPlatform
./TopsPlatform_1.0.5.1-2c3111_deb_amd64.run --no-auto-load --driver -y
```

2. Pull Docker image
```bash
# Note: This image is only for the Paddle development environment. The image does not contain precompiled Paddle installation packages, TopsPlatform installation packages, etc.
docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev
```

3. Start container with reference commands
```bash
docker run --name paddle-gcu-test -v /home:/home --network=host --ipc=host -it --privileged registry.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
```

4. Install compilation toolkit
```bash
# Install cmake for source compilation
cd /root
wget https://github.com/Kitware/CMake/releases/download/v3.23.4/cmake-3.23.4-linux-x86_64.tar.gz
tar -zxf ./cmake-3.23.4-linux-x86_64.tar.gz
ln -sf /root/cmake-3.23.4-linux-x86_64/bin/cmake /usr/bin/cmake && ln -sf /root/cmake-3.23.4-linux-x86_64/bin/ctest /usr/bin/ctest
```

5. Install TopsPlatform software stack
```bash
# Install TopsPlatform software stack in paddle docker, compilation and execution will depend on sdk, runtime, eccl, aten, topstx (for profiler)
cd /home/paddle_user/deps/TopsPlatform
./TopsPlatform_1.0.5.1-2c3111_deb_amd64.run --no-auto-load -y
dpkg -i topsfactor_*.deb tops-sdk_*.deb eccl_*.deb topsaten_*.deb
```

6. Install PaddlePaddle
```bash
# PaddlePaddle『FeiJiang』Deep Learning Framework, providing fundamental computing capabilities
python -m pip install paddlepaddle==3.0.0b0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

7. Compile and install PaddleCustomDevice<br/>
  PaddleCustomDevice is the custom hardware access implementation for PaddlePaddle『FeiJiang』Deep Learning Framework, providing GCU device management and operator implementation.<br/>
  **Note: Currently PaddleCustomDevice still needs to be compiled from source. The precompiled version of paddle-custom-gcu is pending release.**
```bash
# Download source code
mkdir -p /home/paddle_user/workspace && cd /home/paddle_user/workspace
git clone https://github.com/PaddlePaddle/PaddleCustomDevice.git
cd PaddleCustomDevice
# Switch to v3.0.0-beta1 version
git checkout -b v3.0-beta v3.0.0-beta1
# Dependent operator library
cp /home/paddle_user/deps/TopsPlatform/libtopsop.a ./backends/gcu/kernels/topsflame/
# Start compilation. Dependent third-party libraries will be downloaded on demand during first compilation. Downloading from GitHub may be slow
cd backends/gcu/ && mkdir -p build && cd build
export PADDLE_CUSTOM_PATH=`python -c "import re, paddle; print(re.compile('/__init__.py.*').sub('',paddle.__file__))"`
cmake .. -DWITH_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPY_VERSION=3.9
make -j64
# The compilation output is in build/dist. Install using pip
python -m pip install --force-reinstall -U dist/paddle_custom_gcu*.whl
```
8. Download PaddleNLP repository code and install dependencies
```bash
# PaddleNLP is a natural language processing and large language model (LLM) development library based on PaddlePaddle, containing various large models implemented based on the PaddlePaddle framework, including the llama2-13B model. To facilitate your better use of PaddleNLP, you need to clone the entire repository.
cd /home/paddle_user/workspace
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
# Switch to v3.0.0-beta0 version
git checkout -b v3.0-beta v3.0.0-beta0
# Install dependency libraries
python -m pip install -r requirements.txt
# Compile and install paddlenlp v3.0.0-beta0 from source
python setup.py bdist_wheel && python -m pip uninstall paddlenlp -y && python -m pip install dist/paddlenlp*
```
### 2. Data Preparation: (This will take you 2-5 minutes)
Evaluate on wikitext-103 using trained model
```bash
cd llm/devices/gcu/llama
wget https://paddlenlp.bj.bcebos.com/data/benchmark/wikitext-103.tar.gz
tar -zxf wikitext-103.tar.gz
```
### 3. Inference: (This will take you 15-30 minutes)
Execute the following command for inference:
```bash
bash predict_llama_gcu.sh
```
The first inference will automatically download the weights and configuration, located in the ```/root/.paddlenlp/models/__internal_testing__/sci-benchmark-llama-13b-5k/``` directory.<br/>
**It is recommended to modify the inference configuration file after the initial weight download to achieve greater performance improvements.**<br/>
Change the ```/root/.paddlenlp/models/__internal_testing__/sci-benchmark-llama-13b-5k/config.json``` to the following content:

```json
{
  "alibi": false,
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 1,
  "dtype": "float16",
  "eos_token_id": 2,
  "hidden_dropout_prob": 0.1,
  "hidden_size": 5120,
  "initializer_range": 0.002,
  "intermediate_size": 13824,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 40,
  "num_hidden_layers": 40,
  "num_key_value_heads": 40,
  "pad_token_id": 0,
  "paddlenlp_version": null,
  "rms_norm_eps": 1e-06,
  "rope_scaling_factor": 1.0,
  "rope_scaling_type": null,
  "tie_word_embeddings": false,
  "use_recompute": false,
  "virtual_pp_degree": 1,
  "vocab_size": 32000,
  "use_fused_rope": true,
  "use_fused_rms_norm": true,
  "use_flash_attention": true,
  "fuse_attention_qkv": true,
  "fuse_attention_ffn": true
}
```

After successful execution, you can view the perplexity metric (ppl) of the inference results, with the final evaluation result ppl: 12.785.
```bash
[2024-08-16 01:55:24,753] [    INFO] - step 2000, batch: 2000, loss: 2.323283, speed: 1.40 step/s
[2024-08-16 01:55:31,813] [    INFO] - step 2010, batch: 2010, loss: 2.341318, speed: 1.42 step/s
[2024-08-16 01:55:38,859] [    INFO] - step 2020, batch: 2020, loss: 2.357684, speed: 1.42 step/s
[2024-08-16 01:55:45,897] [    INFO] - step 2030, batch: 2030, loss: 2.371745, speed: 1.42 step/s
[2024-08-16 01:55:52,942] [    INFO] - step 2040, batch: 2040, loss: 2.386801, speed: 1.42 step/s
[2024-08-16 01:55:59,991] [    INFO] - step 2050, batch: 2050, loss: 2.399686, speed: 1.42 step/s
[2024-08-16 01:56:07,037] [    INFO] - step 2060, batch: 2060, loss: 2.410638, speed: 1.42 step/s
[2024-08-16 01:56:14,080] [    INFO] - step 2070, batch: 2070, loss: 2.421459, speed: 1.42 step/s
[2024-08-16 01:56:21,141] [    INFO] - step 2080, batch: 2080, loss: 2.431433, speed: 1.42 step/s
[2024-08-16 01:56:28,170] [    INFO] - step 2090, batch: 2090, loss: 2.443705, speed: 1.42 step/s
[2024-08-16 01:56:35,238] [    INFO] - step 2100, batch: 2100, loss: 2.454847, speed: 1.41 step/s
[2024-08-16 01:56:42,275] [    INFO] - step 2110, batch: 2110, loss: 2.464446, speed: 1.42 step/s
[2024-08-16 01:56:49,323] [    INFO] - step 2120, batch: 2120, loss: 2.475107, speed: 1.42 step/s
[2024-08-16 01:56:56,348] [    INFO] - step 2130, batch: 2130, loss: 2.487760, speed: 1.42 step/s
[2024-08-16 01:57:03,372] [    INFO] - step 2140, batch: 2140, loss: 2.501706, speed: 1.42 step/s
[2024-08-16 01:57:10,395] [    INFO] - step 2150, batch: 2150, loss: 2.513665, speed: 1.42 step/s
[2024-08-16 01:57:17,411] [    INFO] - step 2160, batch: 2160, loss: 2.524555, speed: 1.43 step/s
[2024-08-16 01:57:24,437] [    INFO] - step 2170, batch: 2170, loss: 2.536793, speed: 1.42 step/s
[2024-08-16 01:57:31,461] [    INFO] - step 2180, batch: 2180, loss: 2.547897, speed: 1.42 step/s
[2024-08-16 01:57:34,378] [    INFO] -  validation results on ./wikitext-103/wiki.valid.tokens | avg loss: 2.5483E+00 | ppl: 1.2785E+01 | adjusted ppl: 2.6434E+01 | token ratio: 1.285056584007609 |
'Original Tokens: 279682, Detokenized tokens: 217642'
'Original Tokens: 279682, Detokenized tokens: 217642'
I0816 01:57:34.386860 10925 runtime.cc:130] Backend GCU finalize device:0
I0816 01:57:34.386868 10925 runtime.cc:98] Backend GCU Finalize
```
I will strictly follow your requirements to perform the translation. Please provide the Chinese content you need translated, and I will deliver the English version adhering to all specified guidelines.
