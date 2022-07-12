# 性能测试

本目录提供了 GPU 和 CPU 下性能测试的脚本。

脚本默认不使用 FasterTokenizer 对推理加速，可通过传入参数`--faster_tokenizer`来使用 FasterTokenizer 加速推理，当使用 FasterTokenizer 时，需要先安装 faster_tokenizer 包，并通过设置环境变量 `OMP_NUM_THREADS` 来设置切词的线程数，例如：

```shell
pip install faster_tokenizer
export OMP_NUM_THREADS=4
```

性能测试需要静态图模型 ${MODEL_PATH}，包含 `*.pdmodel` 和 `*.pdiparams` 文件。动态图转静态图可使用预训练模型导出脚本 export_model.py，传入动态图模型的路径 ${DYGRAPH_PATH}，以及导出的静态图路径 ${MODEL_PATH}，后者格式为 dirname/filename_prefix。

```shell
python export_model.py \
    --model_path ${DYGRAPH_PATH} \
    --output_path ${MODEL_PATH} \
```

这里以 CLUE 中的 IFLYTEK 数据集为例，因为 IFLYTEK 是长文本数据集（平均长度是 289.17），能够适应性能测试（最大序列长度为 128）的配置。

## GPU



### 环境依赖

为了在 GPU 上获得最佳的推理性能和稳定性，请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保CUDA >= 11.2，CuDNN >= 8.2，并使用以下命令安装所需依赖

```shell
pip install -r requirements_gpu.txt
```
如需使用量化（INT8）部署，请确保 GPU 设备的 CUDA 计算能力 (CUDA Compute Capability) 大于 7.0，典型的设备包括 V100、T4、A10、A100、GTX 20 系列和 30 系列显卡等。需要安装 TensorRT 以及 包含 TensorRT 预测库的 PaddlePaddle 。
更多关于 CUDA Compute Capability 和精度支持情况请参考 NVIDIA 文档：[GPU 硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)
1. TensorRT 安装请参考：[TensorRT 安装说明](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/install-guide/index.html#overview)，Linux 端简要步骤如下：
    (1)下载 TensorRT8.2，文件名 TensorRT-XXX.tar.gz，[下载链接](https://developer.nvidia.com/tensorrt)
    (2)解压得到 TensorRT-XXX 文件夹
    (3)通过 `export LD_LIBRARY_PATH=TensorRT-XXX/lib:$LD_LIBRARY_PATH` 将 lib 路径加入到`LD_LIBRARY_PATH` 中
    (4)使用 `pip install` 安装 `TensorRT-XXX/python` 中对应的 `TensorRT` 安装包
2. 含有预测库的 PaddlePaddle 的安装请参考：[Paddle Inference 的安装文档](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/source_compile.html)，Linux 端简要步骤如下：
    (1)根据 CUDA 环境和 Python 版本下载对应的支持 TensorRT 的 Paddle Inference 预测库，如 linux-cuda11.2-cudnn8.2-trt8-gcc8.2。[Paddle Inference 下载路径](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html#python)
    (2)使用 pip install 安装下载好的 Paddle Inference 安装包


### 启动测试

推理精度默认 PF32，针对 FP16 或者 INT8 的推理需要传入对应参数 `precision_mode` 为 `fp16` 或者 `int8`。还可以传入 `batch_size` 等参数。如：

```shell
MODEL_NAME_OR_PATH=ernie-3.0-medium-zh
bs=32
# Step1: collect shape info
python perf_iflytek.py \
    --device gpu \
    --collect_shape \
    --batch_size $bs \
    --faster_tokenizer \
    --model_path ${MODEL_PATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH}

# Step2(Option1): precision FP32
python perf_iflytek.py \
    --device gpu \
    --batch_size $bs \
    --faster_tokenizer \
    --model_path ${MODEL_PATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH}

# Step2(Option2):precision FP16
python perf_iflytek.py \
    --device gpu  \
    --precision_mode fp16  \
    --batch_size $bs \
    --faster_tokenizer \
    --model_path ${MODEL_PATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH}
```

## CPU

### 环境依赖

CPU端的部署请使用如下命令安装所需依赖

```shell
pip install -r requirements_cpu.txt
```
### 启动测试

推理精度默认 PF32，针对 FP16 或者 INT8 的推理需要传入对应参数 `precision_mode` 为 `fp16` 或者 `int8`。CPU 下的性能测试需要指定参数线程数`--num_threads`、`batch_size`等参数。

```shell
MODEL_NAME_OR_PATH=ernie-3.0-medium-zh
bs=32
# Option1: thread num: 16
python perf_iflytek.py \
    --device cpu \
    --num_threads 1 \
    --batch_size $bs \
    --faster_tokenizer \
    --model_path ${MODEL_PATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH}

# Option2: thread num: 8
python perf_iflytek.py \
    --device cpu \
    --num_threads 8 \
    --batch_size $bs \
    --faster_tokenizer \
    --model_path ${MODEL_PATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH}
```
