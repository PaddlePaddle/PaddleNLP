# ERNIE 3.0 Python部署指南

## 安装
ERNIE 3.0的部署分为CPU和GPU两种情况，请根据你的部署环境安装对应的依赖。
### CPU端
CPU端的部署请使用如下指令安装所需依赖
```
pip install -r requirements_cpu.txt
```
### GPU端
在进行GPU部署之前请先确保机器已经安装好CUDA >= 11.2，CuDNN >= 8.2，然后请使用如下指令安装所需依赖
```
pip install -r requirements_gpu.txt
```
在计算能力（Compute Capability）大于7.0的GPU硬件上，比如T4，如需FP16或者Int8量化推理加速，还需安装TensorRT和Paddle Inference，计算能力（Compute Capability）和精度支持情况请参考：[GPU算力和支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)  
1. TensorRT安装请参考：[TensorRT安装说明](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/install-guide/index.html#overview)，Linux端简要步骤如下：  
    (1)下载TensorRT8.2版本,文件名TensorRT-XXX.tar.gz，[下载链接](https://developer.nvidia.com/tensorrt)  
    (2)解压得到TensorRT-XXX文件夹  
    (3)通过export LD_LIBRARY_PATH=TensorRT-XXX/lib:$LD_LIBRARY_PATH将lib路径加入到LD_LIBRARY_PATH中  
    (4)使用pip install安装TensorRT-XXX/python中对应的tensorrt安装包
2. Paddle Inference的安装请参考：[Paddle Inference的安装文档](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/source_compile.html)，Linux端简要步骤如下：  
    (1)根据CUDA环境和Python版本下载对应的Paddle Inference预测库，注意须下载支持TensorRT的预测包，如linux-cuda11.2-cudnn8.2-trt8-gcc8.2。[Paddle Inference下载路径](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html#python)  
    (2)使用pip install安装下载好的Paddle Inference安装包


## 使用说明
如果使用CPU，请运行infer_cpu.py进行部署，如果使用gpu，请运行infer_gpu.py进行部署，请根据你的部署设备选择相应的部署脚本
### CPU端
在CPU端，请使用如下指令进行部署
```
python infer_cpu.py --task_name token_cls --model_path ./ner_model/infer
```
输出打印如下:
```
input data: 古老的文明，使我们引以为豪，彼此钦佩。
The model detects all entities:
-----------------------------
input data: 原产玛雅故国的玉米，早已成为华夏大地主要粮食作物之一。
The model detects all entities:
entity: 玛雅   label: LOC   pos: [2, 3]
entity: 华夏   label: LOC   pos: [14, 15]
-----------------------------
```
参数说明：
| 参数 |参数说明 |
|----------|--------------|
|--task_name | 配置任务名称，可选seq_cls和token_cls，默认为seq_cls|
|--model_name_or_path | 模型的路径或者名字，默认为ernie-3.0-medium-zh|
|--model_path | 用于推理的Paddle模型的路径|
|--max_seq_length |最大序列长度，默认为128|
|--enable_quantize | 是否使用动态量化进行加速，默认关闭 |
|--num_threads | 配置cpu的线程数，默认为cpu的最大线程数 |

**Note**：在支持avx512_vnni指令集或Intel® DL Boost的CPU设备上，可开启enable_quantize开关对FP32模型进行动态量化以获得更高的推理性能。  
batch size为32，max_seq_length为128时，推理加速情况如下表所示：  
当线程数为1时  
| 数据集 | FP32模型推理时间 | INT8量化推理时间 | 加速比 |
|----------|--------------|-------------|-------------|
| tnews |1026.41|425.89|2.41|
| msra_ner |3218.55|1442.10|2.23|
| cmrc2018 |7833.57|3680.89|2.12|

当线程为10时  
| 数据集 | FP32模型推理时间 | INT8量化推理时间 | 加速比 |
|----------|--------------|-------------|-------------|
| tnews |105.97|68.48|1.73|
| msra_ner |392.42|233.42|1.68|
| cmrc2018 |999.76|599.84|1.67|


### GPU端
在GPU端，请使用如下指令进行部署
```
python infer_gpu.py --task_name token_cls --model_path ./ner_model/infer
```
输出打印如下:
```
input data: 古老的文明，使我们引以为豪，彼此钦佩。
The model detects all entities:
-----------------------------
input data: 原产玛雅故国的玉米，早已成为华夏大地主要粮食作物之一。
The model detects all entities:
entity: 玛雅   label: LOC   pos: [2, 3]
entity: 华夏   label: LOC   pos: [14, 15]
-----------------------------
```
如果需要FP16进行加速，可以开启use_fp16开关，具体指令为
```
# 第一步，打开set_dynamic_shape开关，自动配置动态shape
python infer_gpu.py --task_name token_cls --model_path ./ner_model/infer --use_fp16 --set_dynamic_shape
# 第二步，开启预测
python infer_gpu.py --task_name token_cls --model_path ./ner_model/infer --use_fp16
```
如果需要进行int8量化加速，还需要使用量化脚本对训练的FP32模型进行量化，然后使用量化后的模型进行部署，模型的量化请参考：[模型量化脚本使用说明](./../../README.md)，量化模型的部署指令为  
```
# 第一步，打开set_dynamic_shape开关，自动配置动态shape
python infer_gpu.py --task_name token_cls --model_path ./ner_quant_model/int --set_dynamic_shape
# 第二步，开启预测
python infer_gpu.py --task_name token_cls --model_path ./ner_quant_model/int8
```
参数说明：
| 参数 |参数说明 |
|----------|--------------|
|--task_name | 配置任务名称，可选seq_cls和token_cls，默认为seq_cls|
|--model_name_or_path | 模型的路径或者名字，默认为ernie-3.0-medium-zh|
|--model_path | 用于推理的Paddle模型的路径|
|--batch_size |最大可测的batch size，默认为32|
|--max_seq_length |最大序列长度，默认为128|
|--use_fp16 | 是否使用FP16进行加速，默认关闭 |
|--set_dynamic_shape | 配置是否自动配置TensorRT的dynamic shape，开启use_fp16或者进行int8量化推理时需要先开启此选项进行dynamic shape配置，生成shape_info.txt后再关闭，默认关闭 |
