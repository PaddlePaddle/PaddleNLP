# FastDeploy ERNIE 3.0 模型C++部署示例

在部署前，参考[FastDeploy SDK安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)安装FastDeploy C++ SDK。

本目录下分别提供`seq_cls_infer.cc`以及`token_cls_infer.cc`快速完成在CPU/GPU的文本分类任务以及序列标注任务的C++部署示例。


## 文本分类任务

### 快速开始

以下示例展示如何基于FastDeploy库完成ERNIE 3.0 Medium模型在CLUE Benchmark 的[TNEWS数据集](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)上进行文本分类任务的C++预测部署，可通过命令行参数`--device`以及`--backend`指定运行在不同的硬件以及推理引擎后端。

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy SDK安装文档`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载TNEWS数据集的微调后的ERNIE 3.0模型以及词表
wget https://bj.bcebos.com/fastdeploy/models/ernie-3.0/ernie-3.0-medium-zh-tnews.tgz
tar xvfz ernie-3.0-medium-zh-tnews.tgz

# CPU 推理
./seq_cls_infer_demo --model_dir ernie-3.0-medium-zh-tnews --device cpu --backend paddle

# GPU 推理
./seq_cls_infer_demo --model_dir ernie-3.0-medium-zh-tnews --device gpu --backend paddle

```

运行完成后返回的结果如下：
```bash
[INFO] /paddle/PaddleNLP/model_zoo/ernie-3.0/fastdeploy/cpp/seq_cls_infer.cc(103)::CreateRuntimeOption    model_path = ernie-3.0-medium-zh-tnews/infer.pdmodel, param_path = ernie-3.0-medium-zh-tnews/infer.pdiparams
[INFO] fastdeploy/runtime.cc(500)::Init    Runtime initialized with Backend::PDINFER in Device::CPU.
input data: 未来自动驾驶真的会让酒驾和疲劳驾驶成历史吗？
seq cls result:
label: news_car confidence:0.596849
-----------------------------
input data: 黄磊接受华少快问快答，不光智商逆天，情商也不逊黄渤
seq cls result:
label: news_entertainment confidence:0.9522
-----------------------------
```

### 参数说明

`seq_cls_infer_demo` 除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定部署模型的目录， |
|--batch_size |最大可测的 batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--device | 运行的设备，可选范围: ['cpu', 'gpu']，默认为'cpu' |
|--backend | 支持的推理后端，可选范围: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，默认为'onnx_runtime' |
|--use_fp16 | 是否使用FP16模式进行推理。使用tensorrt和paddle_tensorrt后端时可开启，默认为False |

## 序列标注任务

### 快速开始

以下示例展示如何基于FastDeploy库完成ERNIE 3.0 Medium模型在CLUE Benchmark 的[MSRA_NER数据集](https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra)上进行序列标注任务的C++预测部署，可通过命令行参数`--device`以及`--backend`指定运行在不同的硬件以及推理引擎后端。

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载AFQMC数据集的微调后的ERNIE 3.0模型以及词表
wget https://bj.bcebos.com/fastdeploy/models/ernie-3.0/ernie-3.0-medium-zh-msra.tgz
tar xvfz ernie-3.0-medium-zh-msra.tgz

# CPU 推理
./token_cls_infer_demo --model_dir ernie-3.0-medium-zh-msra --device cpu --backend paddle

# GPU 推理
./token_cls_infer_demo --model_dir ernie-3.0-medium-zh-msra --device gpu --backend paddle

```

运行完成后返回的结果如下：

```bash

[INFO] /paddle/PaddleNLP/model_zoo/ernie-3.0/fastdeploy/cpp/token_cls_infer.cc(103)::CreateRuntimeOption    model_path = ernie-3.0-medium-zh-msra/infer.pdmodel, param_path = ernie-3.0-medium-zh-msra/infer.pdiparams
[INFO] fastdeploy/runtime.cc(500)::Init    Runtime initialized with Backend::PDINFER in Device::CPU.
input data: 北京的涮肉，重庆的火锅，成都的小吃都是极具特色的美食。
The model detects all entities:
entity: 北京, label: LOC, pos: [0, 1]
entity: 重庆, label: LOC, pos: [6, 7]
entity: 成都, label: LOC, pos: [12, 13]
-----------------------------
input data: 乔丹、科比、詹姆斯和姚明都是篮球界的标志性人物。
The model detects all entities:
entity: 乔丹, label: PER, pos: [0, 1]
entity: 科比, label: PER, pos: [3, 4]
entity: 詹姆斯, label: PER, pos: [6, 8]
entity: 姚明, label: PER, pos: [10, 11]
-----------------------------

```

### 参数说明

`token_cls_infer_demo` 除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定部署模型的目录， |
|--batch_size |最大可测的 batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--device | 运行的设备，可选范围: ['cpu', 'gpu']，默认为'cpu' |
|--backend | 支持的推理后端，可选范围: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，默认为'onnx_runtime' |
|--use_fp16 | 是否使用FP16模式进行推理。使用tensorrt和paddle_tensorrt后端时可开启，默认为False |

## 相关文档

[ERNIE 3.0模型详细介绍](../../README.md)

[ERNIE 3.0模型导出方法](../../README.md#模型导出)

[ERNIE 3.0模型Python部署方法](../python/README.md)
