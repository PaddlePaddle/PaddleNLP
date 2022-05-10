# Ernie-3.0 Python部署指南

## 安装
Ernie-3.0的部署分为cpu和gpu两种情况，请根据你的部署环境安装对应的依赖。
### CPU端
CPU端的部署请使用如下指令安装所需依赖
```
pip install -r requirement_cpu.txt
```
### GPU端
在进行GPU部署之前请先确保机器已经安装好CUDA11.04和CUDNN8.2+，然后请使用如下指令安装所需依赖
```
pip install -r requirement_gpu.txt
```
在计算能力大于7.0的GPU硬件上，比如T4，如果需要FP16或者Int8量化推理加速，还需安装TensorRT和PaddleInference，具体硬件和精度支持情况请参考：[GPU算力和支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)  
1. TensorRT安装请参考：[TensorRT安装说明](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/install-guide/index.html#overview)，简要步骤如下：  
    (1)下载TensorRT8.4版本,文件名TensorRT-XXX.tar.gz，[下载链接](https://developer.nvidia.com/tensorrt)  
    (2)解压得到TensorRT-XXX文件夹  
    (3)通过export LD_LIBRARY_PATH=TensorRT-XXX/lib:$LD_LIBRARY_PATH将lib路径加入到LD_LIBRARY_PATH中  
    (4)使用pip install安装TensorRT-XXX/python中对应的tensorrt安装包
2. PaddleInference-TRT安装步骤如下：  
    (1)下载对应版本的PaddleInference-TRT，[PaddleInference-TRT下载路径](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html#python)  
    (2)使用pip install安装下载好的PaddleInference-TRT安装包

## 使用说明
如果使用CPU，请运行infer_cpu.py进行部署，如果使用gpu，请运行infer_gpu.py进行部署，请根据你的部署设备选择相应的部署脚本
### CPU端
在CPU端，请使用如下指令进行部署
```
python infer_cpu.py --task_name tnews --model_path ./model/infer
```
如果在支持AVX512VNNI的CPU机器上，比如Intel(R) Xeon(R) Gold 6271C或十代i9等cascade lake及以上机器，可开启enable_quantize开关，无需数据便可对ONNX FP32模型进行量化，获得2到3倍的加速效果，具体部署指令如下，如无法确认是否支持avx512-vnni，可使用lscpu命令查看cpu信息，并在Flags中查找是否有avx512-vnni支持
```
python infer_cpu.py --task_name tnews --model_path ./model/infer -enable_quantize
```
参数说明：
| 参数 |参数说明 |
|----------|--------------|
|--task_name | 配置任务名称，默认tnews|
|--model_name_or_path | 模型的路径或者名字|
|--model_path | 用于推理的Paddle模型的路径|
|--batch_size |测试的batch size大小，默认为32|
|--perf | 是否测试性能 |
|--enable_quantize | 是否启动ONNX FP32的动态量化进行加速 |
|--num_threads | 配置cpu的线程数，默认为10 |
### GPU端
在GPU端，请使用如下指令进行部署
```
python infer_gpu.py --task_name tnews --model_path ./model/infer
```
如果需要FP16进行加速，可以开启enable_fp16开关，具体指令为
```
python infer_gpu.py --task_name tnews --model_path ./model/infer --enable_fp16
```
如果需要进行int8量化加速，还需要使用量化脚本对训练的FP32模型进行量化，然后使用量化后的模型进行部署，模型的量化具体请参考：[模型量化脚本使用说明]()，量化模型的部署指令为  
```
python infer_gpu.py --task_name tnews --model_path ./model/infer
```
参数说明：
| 参数 |参数说明 |
|----------|--------------|
|--task_name | 配置任务名称，默认tnews|
|--model_path | 配置包含Paddle模型的目录路径|
|--device | 配置部署设备，可选‘cpu’或者‘gpu’|
|--batch_size |测试的batch size大小|
|--enable_fp16 | 是否使用FP16进行加速 |
|--perf | 是否测试性能 |
|--collect_shape | 配置是否自动配置TensorRT的dynamic shape，开启enable_fp16或者进行int8量化推理时需要先开启此选项进行dynamic shape配置，生成shapeinfo.txt后再关闭 |
|--num_threads | 配置cpu的线程数 |
