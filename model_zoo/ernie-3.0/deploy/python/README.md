# ERNIE 3.0 Python部署指南
本文介绍ERNIE 3.0 Python端的部署，包括部署环境的准备，序列标注和分类两大场景下的使用示例。
- [ERNIE 3.0 Python部署指南](#ERNIE3.0Python部署指南)
  - [1. 环境准备](#1-环境准备)
    - [1.1 CPU端](#11-CPU端)
    - [1.2 GPU端](#12-GPU端)
  - [2. 序列标注模型推理](#2-序列标注模型推理)
    - [2.1 模型获取](#21-模型获取)
    - [2.2 CPU端推理样例](#22-CPU端推理样例)
    - [2.3 GPU端推理样例](#23-GPU端推理样例)
  - [3. 分类模型推理](#3-分类模型推理)
    - [3.1 模型获取](#31-模型获取)
    - [3.2 CPU端推理样例](#32-CPU端推理样例)
    - [3.3 GPU端推理样例](#33-GPU端推理样例)
## 1. 环境准备
ERNIE 3.0的部署分为CPU和GPU两种情况，请根据你的部署环境安装对应的依赖。
### 1.1 CPU端
CPU端的部署请使用如下命令安装所需依赖
```
pip install -r requirements_cpu.txt
```
### 1.2 GPU端
为了在GPU上获得最佳的推理性能和稳定性，请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保CUDA >= 11.2，CuDNN >= 8.2，并使用以下命令安装所需依赖
```
pip install -r requirements_gpu.txt
```
如需使用半精度（FP16）或量化（INT8）部署，请确保GPU设备的CUDA计算能力 (CUDA Compute Capability) 大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。同时需额外安装TensorRT和Paddle Inference。  
更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)
1. TensorRT安装请参考：[TensorRT安装说明](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/install-guide/index.html#overview)，Linux端简要步骤如下：  
    (1)下载TensorRT8.2版本,文件名TensorRT-XXX.tar.gz，[下载链接](https://developer.nvidia.com/tensorrt)  
    (2)解压得到TensorRT-XXX文件夹  
    (3)通过export LD_LIBRARY_PATH=TensorRT-XXX/lib:$LD_LIBRARY_PATH将lib路径加入到LD_LIBRARY_PATH中  
    (4)使用pip install安装TensorRT-XXX/python中对应的tensorrt安装包
2. Paddle Inference的安装请参考：[Paddle Inference的安装文档](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/source_compile.html)，Linux端简要步骤如下：  
    (1)根据CUDA环境和Python版本下载对应的Paddle Inference预测库，注意须下载支持TensorRT的预测包，如linux-cuda11.2-cudnn8.2-trt8-gcc8.2。[Paddle Inference下载路径](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html#python)  
    (2)使用pip install安装下载好的Paddle Inference安装包


## 2. 序列标注模型推理
### 2.1 模型获取
用户可使用自己训练的模型进行推理，具体训练调优方法可参考[模型训练调优](./../../README.md#微调)，也可以使用我们提供的msra_ner数据集训练的ERNIE 3.0模型，请执行如下命令获取模型：
```
# 获取序列标注FP32模型
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_3.0/msra_ner_pruned_infer_model.zip
unzip msra_ner_pruned_infer_model.zip
```
### 2.2 CPU端推理样例
在CPU端，请使用如下命令进行部署
```
python infer_cpu.py --task_name token_cls --model_path ./msra_ner_pruned_infer_model/float32
```
输出打印如下:
```
input data: 北京的涮肉，重庆的火锅，成都的小吃都是极具特色的美食。
The model detects all entities:
entity: 北京   label: LOC   pos: [0, 1]
entity: 重庆   label: LOC   pos: [6, 7]
entity: 成都   label: LOC   pos: [12, 13]
-----------------------------
input data: 乔丹、科比、詹姆斯和姚明都是篮球界的标志性人物。
The model detects all entities:
entity: 乔丹   label: PER   pos: [0, 1]
entity: 科比   label: PER   pos: [3, 4]
entity: 詹姆斯   label: PER   pos: [6, 8]
entity: 姚明   label: PER   pos: [10, 11]
-----------------------------
```
infer_cpu.py脚本中的参数说明：
| 参数 |参数说明 |
|----------|--------------|
|--task_name | 配置任务名称，可选seq_cls和token_cls，默认为seq_cls|
|--model_name_or_path | 模型的路径或者名字，默认为ernie-3.0-medium-zh|
|--model_path | 用于推理的Paddle模型的路径|
|--max_seq_length |最大序列长度，默认为128|
|--use_quantize | 是否使用动态量化进行加速，默认关闭 |
|--num_threads | 配置cpu的线程数，默认为cpu的最大线程数 |

**Note**：在支持avx512_vnni指令集或Intel® DL Boost的CPU设备上，可开启use_quantize开关对FP32模型进行动态量化以获得更高的推理性能，具体性能提升情况请查阅[量化性能提升情况](../../README.md#压缩效果)。  
CPU端，开启动态量化的命令如下：
```
python infer_cpu.py --task_name token_cls --model_path ./msra_ner_pruned_infer_model/float32 --use_quantize
```
INT8的输出打印和FP32的输出打印一致。

### 2.3 GPU端推理样例
在GPU端，请使用如下命令进行部署
```
python infer_gpu.py --task_name token_cls --model_path ./msra_ner_pruned_infer_model/float32
```
输出打印如下:
```
input data: 北京的涮肉，重庆的火锅，成都的小吃都是极具特色的美食。
The model detects all entities:
entity: 北京   label: LOC   pos: [0, 1]
entity: 重庆   label: LOC   pos: [6, 7]
entity: 成都   label: LOC   pos: [12, 13]
-----------------------------
input data: 乔丹、科比、詹姆斯和姚明都是篮球界的标志性人物。
The model detects all entities:
entity: 乔丹   label: PER   pos: [0, 1]
entity: 科比   label: PER   pos: [3, 4]
entity: 詹姆斯   label: PER   pos: [6, 8]
entity: 姚明   label: PER   pos: [10, 11]
-----------------------------
```
如果需要FP16进行加速，可以开启use_fp16开关，具体命令为
```
# 第一步，打开set_dynamic_shape开关，自动配置动态shape，在当前目录下生成dynamic_shape_info.txt文件
python infer_gpu.py --task_name token_cls --model_path ./msra_ner_pruned_infer_model/float32 --use_fp16 --shape_info_file dynamic_shape_info.txt --set_dynamic_shape
# 第二步，读取上一步中生成的dynamic_shape_info.txt文件，开启预测
python infer_gpu.py --task_name token_cls --model_path ./msra_ner_pruned_infer_model/float32 --use_fp16 --shape_info_file dynamic_shape_info.txt
```
如果需要进行INT8量化加速，还需要使用量化脚本对训练好的FP32模型进行量化，然后使用量化后的模型进行部署，模型的量化请参考：[模型量化脚本使用说明](./../../README.md#模型压缩)，也可下载我们量化后的INT8模型进行部署，请执行如下命令获取模型：  
```
# 获取序列标注INT8量化模型
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_3.0/msra_ner_quant_infer_model.zip
unzip msra_ner_quant_infer_model.zip
```
量化模型的部署命令为：  
```
# 第一步，打开set_dynamic_shape开关，自动配置动态shape，在当前目录下生成dynamic_shape_info.txt文件
python infer_gpu.py --task_name token_cls --model_path ./msra_ner_quant_infer_model/int8 --shape_info_file dynamic_shape_info.txt --set_dynamic_shape
# 第二步，读取上一步中生成的dynamic_shape_info.txt文件，开启预测
python infer_gpu.py --task_name token_cls --model_path ./msra_ner_quant_infer_model/int8 --shape_info_file dynamic_shape_info.txt
```
FP16和INT8推理的运行结果和FP32的运行结果一致。  
infer_gpu.py脚本中的参数说明：
| 参数 |参数说明 |
|----------|--------------|
|--task_name | 配置任务名称，可选seq_cls和token_cls，默认为seq_cls|
|--model_name_or_path | 模型的路径或者名字，默认为ernie-3.0-medium-zh|
|--model_path | 用于推理的Paddle模型的路径|
|--batch_size |最大可测的batch size，默认为32|
|--max_seq_length |最大序列长度，默认为128|
|--shape_info_file | 指定dynamic shape info的存储文件名，默认为shape_info.txt |
|--set_dynamic_shape | 配置是否自动配置TensorRT的dynamic shape，开启use_fp16或者进行INT8量化推理时需要先开启此选项进行dynamic shape配置，生成shape_info.txt后再关闭，默认关闭 |
|--use_fp16 | 是否使用FP16进行加速，默认关闭 |

## 3. 分类模型推理
### 3.1 模型获取
用户可使用自己训练的模型进行推理，具体训练调优方法可参考[模型训练调优](./../../README.md#微调)，也可以使用我们提供的tnews数据集训练的ERNIE 3.0模型，请执行如下命令获取模型：
```
# 分类模型模型：
wget  https://paddlenlp.bj.bcebos.com/models/transformers/ernie_3.0/tnews_pruned_infer_model.zip
unzip tnews_pruned_infer_model.zip
```
### 3.2 CPU端推理样例
在CPU端，请使用如下命令进行部署
```
python infer_cpu.py --task_name seq_cls --model_path ./tnews_pruned_infer_model/float32
```
输出打印如下:
```
input data: 未来自动驾驶真的会让酒驾和疲劳驾驶成历史吗？
seq cls result:
label: news_car   confidence: 0.5543532371520996
-----------------------------
input data: 黄磊接受华少快问快答，不光智商逆天，情商也不逊黄渤
seq cls result:
label: news_entertainment   confidence: 0.9495906829833984
-----------------------------
```
和序列标注模型推理类似，开启动态量化的命令如下：
```
python infer_cpu.py --task_name seq_cls --model_path ./tnews_pruned_infer_model/float32 --use_quantize
```
输出打印如下:
```
input data: 未来自动驾驶真的会让酒驾和疲劳驾驶成历史吗？
seq cls result:
label: news_car   confidence: 0.5778735876083374
-----------------------------
input data: 黄磊接受华少快问快答，不光智商逆天，情商也不逊黄渤
seq cls result:
label: news_entertainment   confidence: 0.9206441044807434
-----------------------------
```
### 3.3 GPU端推理样例
在GPU端，请使用如下命令进行部署
```
python infer_gpu.py --task_name seq_cls --model_path ./tnews_pruned_infer_model/float32
```
输出打印如下:
```
input data: 未来自动驾驶真的会让酒驾和疲劳驾驶成历史吗？
seq cls result:
label: news_car   confidence: 0.5543532371520996
-----------------------------
input data: 黄磊接受华少快问快答，不光智商逆天，情商也不逊黄渤
seq cls result:
label: news_entertainment   confidence: 0.9495906829833984
-----------------------------
```
如果需要FP16进行加速，可以开启use_fp16开关，具体命令为
```
# 第一步，打开set_dynamic_shape开关，自动配置动态shape，在当前目录下生成dynamic_shape_info.txt文件
python infer_gpu.py --task_name seq_cls --model_path ./tnews_pruned_infer_model/float32 --use_fp16 --shape_info_file dynamic_shape_info.txt --set_dynamic_shape
# 第二步，读取上一步中生成的dynamic_shape_info.txt文件，开启预测
python infer_gpu.py --task_name seq_cls --model_path ./tnews_pruned_infer_model/float32 --use_fp16 --shape_info_file dynamic_shape_info.txt
```
输出打印如下:
```
input data: 未来自动驾驶真的会让酒驾和疲劳驾驶成历史吗？
seq cls result:
label: news_car   confidence: 0.5536671876907349
-----------------------------
input data: 黄磊接受华少快问快答，不光智商逆天，情商也不逊黄渤
seq cls result:
label: news_entertainment   confidence: 0.9494127035140991
-----------------------------
```
如果需要进行INT8量化加速，还需要使用量化脚本对训练好的FP32模型进行量化，然后使用量化后的模型进行部署，模型的量化请参考：[模型量化脚本使用说明](./../../README.md#模型压缩)，也可下载我们量化后的INT8模型进行部署，请执行如下命令获取模型：  
```
# 获取序列标注INT8量化模型
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_3.0/tnews_quant_infer_model.zip
unzip tnews_quant_infer_model.zip
```
量化模型的部署命令为：  
```
# 第一步，打开set_dynamic_shape开关，自动配置动态shape，在当前目录下生成dynamic_shape_info.txt文件
python infer_gpu.py --task_name seq_cls --model_path ./tnews_quant_infer_model/int8 --shape_info_file dynamic_shape_info.txt --set_dynamic_shape
# 第二步，读取上一步中生成的dynamic_shape_info.txt文件，开启预测
python infer_gpu.py --task_name seq_cls --model_path ./tnews_quant_infer_model/int8 --shape_info_file dynamic_shape_info.txt
```
输出打印如下:
```
input data: 未来自动驾驶真的会让酒驾和疲劳驾驶成历史吗？
seq cls result:
label: news_car   confidence: 0.5510320067405701
-----------------------------
input data: 黄磊接受华少快问快答，不光智商逆天，情商也不逊黄渤
seq cls result:
label: news_entertainment   confidence: 0.9432708024978638
-----------------------------
```
