# Ernie-3.0 Python部署说明

## 环境依赖
- python >= 3.6
- onnxruntime-gpu >= 1.10.0
- paddleinference-trt
- onnx >= 1.10.0
- paddle2onnx develop版本

## 部署说明
### 参数选项
| 参数 |参数说明 |
|----------|--------------|
|--model_path | 配置包含Paddle模型的目录路径|
|--device | 配置部署设备，可选‘cpu’或者‘gpu’|
|--batch_size |测试的batch size大小|
|--int8 | 配置是否进行量化推理 |
|--perf | 是否测试性能 |
|--collect_shape | 配置是否自动配置TensorRT的dynamic shape，当device为gpu，int8为True时需要先开启此选项进行dynamic shape配置，生成shapeinfo.txt后再关闭 |
|--num_threads | 配置cpu的线程数 |

### 运行指令
1. CPU非量化模型
```
python infer.py --model_path tnews/pruned_fp32/float32 --device ‘cpu’
```
2. CPU量化模型
```
python infer.py --model_path tnews/pruned_fp32/float32 --device ‘cpu’ --int8
```
3. GPU非量化模型
```
python infer.py --model_path tnews/pruned_fp32/float32 --device ‘gpu’
```
4. GPU量化模型
```
python infer.py --model_path tnews/pruned_quant/int8 --device ‘gpu’ --int8
```
