## 🚣‍♂️ Run llama2-7b Model on CPU (with AVX Support) Using PaddleNLP 🚣
PaddleNLP has deeply adapted and optimized the llama series models for CPUs supporting AVX instructions. This document explains the process of performing high-performance inference for llama series models on AVX-enabled CPUs using PaddleNLP.

### Check Hardware:

| Chip Type | GCC Version | cmake Version |
| --- | --- | --- |
| Intel(R) Xeon(R) Platinum 8463B | 9.4.0 | >=3.18 |

**Note: To verify if your machine supports AVX instructions, execute the following command in the system environment and check for output:**
```
lscpu | grep -o -P '(?<!\w)(avx\w*)'

# Expected output -
avx
avx2
**avx512f**
avx512dq
avx512ifma
avx512cd
**avx512bw**
avx512vl
avx_vnni
**avx512_bf16**
avx512vbmi
avx512_vbmi2
avx512_vnni
avx512_bitalg
avx512_vpopcntdq
**avx512_fp16**
```

### Environment Setup:
#### 1 Install numactl
```
apt-get update
apt-get install numactl
```
#### 2 Install paddle
##### 2.1 Source Installation:
```shell
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle && mkdir build && cd build

cmake .. -DPY_VERSION=3.8 -DWITH_GPU=OFF

make -j128
pip install -U python/dist/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl
```
##### 2.2 pip Installation:
```shell
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
```
##### 2.3 Verify Installation:
```shell
python -c "import paddle; paddle.version.show()"
python -c "import paddle; paddle.utils.run_check()"
```
#### 3 Clone PaddleNLP Repository and Install Dependencies
```shell
# PaddleNLP is a natural language processing and large language model (LLM) development library based on PaddlePaddle, containing various large models implemented with PaddlePaddle, including the llama series models. To use PaddleNLP more effectively, you need to clone the entire repository.
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```
#### 4 Install Third-Party Libraries and paddlenlp_ops
```shell
cd ./paddlenlp_ops/
pip install -r requirements.txt
cd ../
```
```shell
# The PaddleNLP repository has built-in dedicated fusion operators for users to enjoy ultimate compression of inference costs
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/csrc/cpu
sh setup.sh
```
#### 5 Third-party library installation failures
```shell
# If oneccl installation fails, it is recommended to reinstall with gcc versions between 8.2-9.4
cd csrc/cpu/xFasterTransformer/3rdparty/
sh prepare_oneccl.sh

# If xFasterTransformer installation fails, it is recommended to reinstall with gcc 9.2 or higher
cd csrc/cpu/xFasterTransformer/build/
make -j24

# More commands and environment variables can be found in csrc/cpu/setup.sh
```
### CPU High-Performance Inference
PaddleNLP also provides CPU high-performance inference based on intel/xFasterTransformer, currently supporting FP16, BF16, and INT8 precision inference, as well as hybrid FP16-based Prefill and INT8-based Decode inference.

#### High-performance inference reference for non-HBM machines:
##### 1 Determine OMP_NUM_THREADS
```shell
OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}')
```
##### 2 Dynamic graph inference
```shell
cd ../../llm/
# 2. Dynamic graph inference: High-performance AVX dynamic graph model inference command reference
OMP_NUM_THREADS=$(lscku | grep "Core(s) per socket" | awk -F ':' '{print $2}') numactl -N 0 -m 0 python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float32 --avx_mode --avx_type "fp16_int8" --device "cpu"
```
##### 3 Static graph inference
```shell
# Step1: Static graph export
python ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --output_path ./inference --dtype float32 --avx_mode --avx_type "fp16_int8" --device "cpu"
# Step2: Static graph inference
OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}') numactl -N 0 -m 0 python ./predict/predictor.py --model_name_or_path ./inference --inference_model --dtype "float32" --mode "static" --device "cpu" --avx_mode
```
#### HBM Machine High-Performance Inference Reference:
##### 1 Hardware and OMP_NUM_THREADS Verification
```shell
# Theoretically, HBM machines achieve 1.3x-1.9x acceleration for next token latency compared to non-HBM machines
# Verify machine has hbm
lscpu
# Nodes like node2, node3 indicate HBM support
$NUMA node0 CPU(s):                  0-31,64-95
$NUMA node1 CPU(s):                  32-63,96-127
$NUMA node2 CPU(s):
$NUMA node3 CPU(s):

# Determine OMP_NUM_THREADS
lscpu | grep "Socket(s)" | awk -F ':' '{print $2}'
OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}')
```

##### 2 Dynamic Graph Inference
```shell
cd ../../llm/
# High-performance AVX dynamic graph inference command reference
FIRST_TOKEN_WEIGHT_LOCATION=0 NEXT_TOKEN_WEIGHT_LOCATION=2 OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}') numactl -N 0  -m 0 python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float32 --avx_mode --avx_type "fp16_int8" --device "cpu"
Note: FIRST_TOKEN_WEIGHT_LOCATION and NEXT_TOKEN_WEIGHT_LOCATION indicate placing first_token weights on numa0, next_token weights on numa2 (HBM cache node).
```

##### 3 Static Graph Inference
```shell
# High-performance static graph inference command reference
# Step1: Static graph export
python ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --output_path ./inference --dtype float32 --avx_mode --avx_type "fp16_int8" --device "cpu"
# Step2: Static graph inference
FIRST_TOKEN_WEIGHT_LOCATION=0 NEXT_TOKEN_WEIGHT_LOCATION=2 OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}') numactl -N 0  -m 0 python ./predict/predictor.py --model_name_or_path ./inference --inference_model --dtype "float32" --mode "static" --device "cpu" --avx_mode
```
