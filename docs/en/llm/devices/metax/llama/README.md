# Run llama2-13b Model on MX C550 with PaddleNLP

PaddleNLP has deeply adapted and optimized the llama2-13B model on XiYun® C550 ([Learn about Metax](https://www.metax-tech.com/)). This toolkit achieves complete unification of training and inference interfaces between XiYun C550 and GPUs, realizing the effect of 'seamless switching'.

The XiYun C500 series GPUs are Metax's flagship products based on proprietary GPU IP, featuring powerful multi-precision computing capabilities, 64GB large-capacity high-bandwidth memory, and advanced multi-GPU interconnect MetaLink technology. Equipped with MXMACA® software stack, it is fully compatible with mainstream GPU ecosystems, enabling zero-cost application migration. It can conveniently support AI computing, general-purpose computing, and data processing scenarios.

## 🚀 Quick Start 🚀

### (0) Before starting, you need a machine with XiYun C550. The system requirements for this machine are as follows:

| Chip Type | vbios Version | MXMACA Version      |
| --------- | ------------- | ------------------- |
| XiYun C550 | ≥ 1.13        | ≥ 2.23.0.1018      |

**Note: To verify if your machine has XiYun C550 GPU, simply enter the following command in the system environment and check for output:**
```
mx-smi

# Output as follows
mx-smi  version: 2.1.6

=================== MetaX System Management Interface Log ===================
Timestamp                                         : Mon Sep 23 06:24:52 2024

Attached GPUs                                     : 8
+---------------------------------------------------------------------------------+
| MX-SMI 2.1.6                        Kernel Mode Driver Version: 2.5.014         |
| MACA Version: 2.23.0.1018           BIOS Version: 1.13.4.0                      |
|------------------------------------+---------------------+----------------------+
| GPU         NAME                   | Bus-id              | GPU-Util             |
| Temp        Power                  | Memory-Usage        |                      |
|====================================+=====================+======================|
| 0           MXC550                 | 0000:2a:00.0        | 0%                   |
| 31C         44W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 1           MXC550                 | 0000:3a:00.0        | 0%                   |
| 31C         46W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 2           MXC550                 | 0000:4c:00.0        | 0%                   |
| 31C         47W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 3           MXC550                 | 0000:5c:00.0        | 0%                   |
| 31C         46W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 4           MXC550                 | 0000:aa:00.0        | 0%                   |
| 30C         46W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 5           MXC550                 | 0000:ba:00.0        | 0%                   |
| 31C         47W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 6           MXC550                 | 0000:ca:00.0        | 0%                   |
| 30C         46W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 7           MXC550                 | 0000:da:00.0        | 0%                   |
| 30C         47W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+

+---------------------------------------------------------------------------------+
| Process:                                                                        |
|  GPU                    PID         Process Name                 GPU Memory     |
|                                                                  Usage(MiB)     |
|=================================================================================|
|  no process found                                                               |
+---------------------------------------------------------------------------------+
```
### (1) Environment Preparation: (This will take 5~55 minutes)

1. Build runtime environment using container (optional)

```
# You can use --device=/dev/dri/card0 to make only GPU 0 visible inside the container (other cards follow the same logic), --device=/dev/dri makes all GPUs visible
docker run -it --rm --device=/dev/dri
    --device=/dev/mxcd --group-add video -network=host --uts=host --ipc=host --privileged=true --shm-size 128g registry.baidubce.com/paddlepaddle/paddle:2.6.1-gpu-cuda11.7-cudnn8.4-trt8.4
```

2. Install MXMACA software stack

   > You may contact fae_support@metax-tech.com to obtain MXMACA installation package and technical support. Authorized users can access [Metax Software Center](https://sw-download.metax-tech.com/login) to download related packages.
   >

```
# Assuming you have downloaded and extracted MXMACA driver
sudo bash /path/to/maca_package/mxmaca-sdk-install.sh
```

3. Install PaddlePaddle

① If you have already obtained PaddlePaddle installation package through Metax, you can directly install:

`pip install paddlepaddle_gpu-2.6.0+mc*.whl`

② You can also compile PaddlePaddle installation package from source code. Please ensure you have properly installed MXMACA software stack. The compilation process uses cu-bridge compilation tool based on MXMACA. You may refer to [documentation](https://gitee.com/p4ul/cu-bridge/tree/master/docs/02_User_Manual) for more information.
# 1. Access PaddlePaddle github repository, clone code and switch to mxmaca branch
git clone https://github.com/PaddlePaddle/Paddle.git
git checkout release-mxmaca/2.6
# 2. Pull third-party dependencies
git submodule update --init
# 3. Configure environment variables
export MACA_PATH=/real/maca/install/path
export CUDA_PATH=/real/cuda/install/path
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export PATH=${CUDA_PATH}/bin:${CUCC_PATH}/bin:${CUCC_PATH}/tools:${MACA_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
# 4. Verify configuration
cucc --version
# 5. Execute compilation
makdir -p build && cd build
cmake_maca .. -DPY_VERSION=3.8 -DWITH_GPU=ON -DWITH_DISTRIBUTE=ON -DWITH_NCCL=ON
make_maca -j64
# 6. Install whl package after compilation
pip install python/dist/paddlepaddle_gpu*.whl

4. Clone PaddleNLP repository and install dependencies

```
# PaddleNLP is a natural language processing and large language model (LLM) development library based on PaddlePaddle. It contains various large models implemented with PaddlePaddle, including the llama2-13B model. To facilitate your use of PaddleNLP, you need to clone the entire repository.
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
git checkout origin/release/3.0-beta1
python -m pip install -r requirements.txt
python -m pip install -e .
```

### (2) Inference: (This will take 5~10 minutes)

1. Try running inference demo

```
cd llm/predict
python predictor.py --model_name_or_path meta-llama/Llama-2-13b-chat --dtype bfloat16 --output_file "infer.json" --batch_size 1 --decode_strategy "greedy_search"
```

After successful execution, you can view generated inference results. A sample output is as follows:
You can also try to refer to the instructions in the [document](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/slm/examples/benchmark/wiki_lambada/README.md) to validate the inference accuracy using the wikitext dataset.
