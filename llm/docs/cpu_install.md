## 🚣‍♂️ 使用 PaddleNLP 在 CPU(支持 AVX 指令)下跑通 llama2-7b 模型 🚣
PaddleNLP 在支持 AVX 指令的 CPU 上对 llama 系列模型进行了深度适配和优化，此文档用于说明在支持 AVX 指令的 CPU 上使用 PaddleNLP 进行 llama 系列模型进行高性能推理的流程。

### 检查硬件：

 | 芯片类型 | GCC 版本 |cmake 版本 |
 | --- | --- | --- |
 | Intel(R) Xeon(R) Platinum 8463B | 9.4.0| >=3.18 |

**注：如果要验证您的机器是否支持 AVX 指令，只需系统环境下输入命令，看是否有输出：**
```
lscpu | grep -o -P '(?<!\w)(avx\w*)'

# 显示如下结果 -
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

### 环境准备：
#### 1 安装 numactl
```
apt-get update
apt-get install numactl
```
#### 2 安装 paddle
##### 2.1 源码安装：
```shell
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle && mkdir build && cd build

cmake .. -DPY_VERSION=3.8 -DWITH_GPU=OFF

make -j128
pip install -U python/dist/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl
```
##### 2.2 pip 安装:
```shell
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
```
##### 2.3 检查是否安装正常:
```shell
python -c "import paddle; paddle.version.show()"
python -c "import paddle; paddle.utils.run_check()"

```
#### 3 克隆 PaddleNLP 仓库代码，并安装依赖
```shell
# PaddleNLP是基于paddlepaddle『飞桨』的自然语言处理和大语言模型(LLM)开发库，存放了基于『飞桨』框架实现的各种大模型，llama系列模型也包含其中。为了便于您更好地使用PaddleNLP，您需要clone整个仓库。
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```
#### 4 安装第三方库和 paddlenlp_ops
```shell
# PaddleNLP仓库内置了专用的融合算子，以便用户享受到极致压缩的推理成本
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/csrc/cpu
sh setup.sh
```
#### 5 第三方库安装失败
```shell
#如果oneccl安装失败 建议在gcc 8.2-9.4之间重新安装
cd csrc/cpu/xFasterTransformer/3rdparty/
sh prepare_oneccl.sh

#如果xFasterTransformer 安装失败,建议在gcc 9.2以上重新安装
cd csrc/cpu/xFasterTransformer/build/
make -j24

#更多命令和环境变量可参考csrc/cpu/setup.sh
```
### Cpu 高性能推理
PaddleNLP 还提供了基于 intel/xFasterTransformer 的 CPU 高性能推理，目前支持 FP16、BF16、INT8多种精度推理，以及 Prefill 基于 FP16,Decode 基于 INT8混合方式推理。

#### 非 HBM 机器高性能推理参考：
##### 1 确定 OMP_NUM_THREADS
```shell
OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}')
```
##### 2 动态图推理
```shell
cd ../../llm/
#2.动态图推理 高性能 AVX 动态图模型推理命令参考
OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}') numactl -N 0  -m 0 python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float32 --avx_mode --avx_type "fp16_int8" --device "cpu"
```
##### 3 静态图推理
```shell
#step1 : 静态图导出
python ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --output_path ./inference --dtype float32 --avx_mode --avx_type "fp16_int8" --device "cpu"
#step2: 静态图推理
OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}') numactl -N 0  -m 0 python ./predict/predictor.py --model_name_or_path ./inference --inference_model --dtype "float32" --mode "static" --device "cpu" --avx_mode
```

#### HBM 机器高性能推理参考：
##### 1 硬件和 OMP_NUM_THREADS 确认
```shell
#理论上HBM机器比非HBM机器nexttoken时延具有1.3倍-1.9倍的加速
#确认机器具有 hbm
lscpu
#如 node2、node3表示支持 hbm
$NUMA node0 CPU(s):                  0-31,64-95
$NUMA node1 CPU(s):                  32-63,96-127
$NUMA node2 CPU(s):
$NUMA node3 CPU(s):

#确定OMP_NUM_THREADS
lscpu | grep "Socket(s)" | awk -F ':' '{print $2}'
OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}')
```
##### 2 动态图推理
```shell
cd ../../llm/
# 高性能 AVX 动态图模型推理命令参考
FIRST_TOKEN_WEIGHT_LOCATION=0 NEXT_TOKEN_WEIGHT_LOCATION=2 OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}') numactl -N 0  -m 0 python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float32 --avx_mode --avx_type "fp16_int8" --device "cpu"
注:FIRST_TOKEN_WEIGHT_LOCATION和NEXT_TOKEN_WEIGHT_LOCATION表示first_token权重放在numa0,next_token权重放在numa2(hbm缓存节点)。
```
##### 3 静态图推理
```shell
# 高性能静态图模型推理命令参考
# step1 : 静态图导出
python ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --output_path ./inference --dtype float32 --avx_mode --avx_type "fp16_int8" --device "cpu"
# step2: 静态图推理
FIRST_TOKEN_WEIGHT_LOCATION=0 NEXT_TOKEN_WEIGHT_LOCATION=2 OMP_NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}') numactl -N 0  -m 0 python ./predict/predictor.py --model_name_or_path ./inference --inference_model --dtype "float32" --mode "static" --device "cpu" --avx_mode
```
