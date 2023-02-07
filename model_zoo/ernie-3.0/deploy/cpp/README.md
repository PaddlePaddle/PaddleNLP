# FastDeploy ERNIE 3.0 模型 C++ 部署示例

在部署前，参考 [FastDeploy SDK 安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)安装 FastDeploy C++ SDK。

本目录下分别提供 `seq_cls_infer.cc` 以及 `token_cls_infer.cc` 快速完成在 CPU/GPU 的文本分类任务以及序列标注任务的 C++ 部署示例。


## 文本分类任务

### 快速开始

以下示例展示如何基于 FastDeploy 库完成 ERNIE 3.0 Medium 模型在 CLUE Benchmark 的 [AFQMC 数据集](https://github.com/CLUEbenchmark/CLUE)上进行文本分类任务的 C++ 预测部署，可通过命令行参数`--device`以及`--backend`指定运行在不同的硬件以及推理引擎后端。示例中的模型是 ERNIE 3.0 在 `AFQMC 数据集`微调后导出得到的部署模型。

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy SDK安装文档`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# CPU 推理
./seq_cls_infer_demo --model_dir ../../../best_models/afqmc/export/ --device cpu --backend paddle

# GPU 推理
./seq_cls_infer_demo --model_dir ../../../best_models/afqmc/export/ --device gpu --backend paddle

```

运行完成后返回的结果如下：
```bash
[INFO] /paddle/PaddleNLP/model_zoo/ernie-3.0/fastdeploy/cpp/seq_cls_infer.cc(103)::CreateRuntimeOption    model_path = ../../../best_models/afqmc/export/model.pdmodel, param_path = ../../../best_models/afqmc/export/model.pdiparams
[INFO] fastdeploy/runtime.cc(500)::Init    Runtime initialized with Backend::PDINFER in Device::CPU.
input data: 花呗收款额度限制, 收钱码，对花呗支付的金额有限制吗
seq cls result:
label: Similar confidence: 0.509855
-----------------------------
input data: 花呗支持高铁票支付吗, 为什么友付宝不支持花呗付款
seq cls result:
label: Similar confidence: 0.986198
-----------------------------
```

### 量化模型部署

该示例支持部署 Paddle INT8 新格式量化模型，仅需在`--model_dir`参数传入量化模型路径，并且在对应硬件上选择可用的推理引擎后端，即可完成量化模型部署。在 GPU 上部署量化模型时，可选后端为`paddle_tensorrt`、`tensorrt`；在CPU上部署量化模型时，可选后端为`paddle`、`onnx_runtime`。下面将展示如何使用该示例完成量化模型部署，示例中的模型是按照 [ERNIE 3.0 训练文档](../../README.md) 压缩量化后导出得到的量化模型。

```bash

# 在 GPU 上使用 tensorrt 后端运行量化模型，模型目录可按照实际模型路径设置
./seq_cls_infer_demo --model_dir ../../../best_models/afqmc/width_mult_0.75/mse16_1/ --device gpu --backend tensorrt --model_prefix int8

# 在 CPU 上使用paddle_inference后端，模型目录可按照实际模型路径设置
./seq_cls_infer_demo --model_dir ../../../best_models/afqmc/width_mult_0.75/mse16_1/ --device cpu --backend paddle --model_prefix int8

```

运行完成后返回的结果如下：

```bash
[INFO] /paddle/PaddleNLP/model_zoo/ernie-3.0/fastdeploy/cpp/seq_cls_infer.cc(67)::CreateRuntimeOption    model_path = ../../../best_models/afqmc/width_mult_0.75/mse16_1/int8.pdmodel, param_path = ../../../best_models/afqmc/width_mult_0.75/mse16_1/int8.pdmodel
[INFO] fastdeploy/runtime.cc(596)::Init    Runtime initialized with Backend::TRT in Device::GPU.
input data: 花呗收款额度限制, 收钱码，对花呗支付的金额有限制吗
seq cls result:
label: Similar confidence: 0.5259
-----------------------------
input data: 花呗支持高铁票支付吗, 为什么友付宝不支持花呗付款
seq cls result:
label: Similar confidence: 0.985738
-----------------------------
```

### 参数说明

`seq_cls_infer_demo` 除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定部署模型的目录 |
|--vocab_path| 指定的模型词表路径 |
|--batch_size |最大可测的 batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--device | 运行的设备，可选范围: ['cpu', 'kunlunxin', 'gpu']，默认为'cpu' |
|--backend | 支持的推理后端，可选范围: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，默认为'paddle' |
|--use_fp16 | 是否使用FP16模式进行推理。使用tensorrt和paddle_tensorrt后端时可开启，默认为False |
|--model_prefix| 模型文件前缀。前缀会分别与'.pdmodel'和'.pdiparams'拼接得到模型文件名和参数文件名。默认为 'model'|

## 序列标注任务

### 快速开始

以下示例展示如何基于 FastDeploy 库完成 ERNIE 3.0 Medium 模型在 CLUE Benchmark 的 [MSRA_NER 数据集](https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra)上进行序列标注任务的 C++ 预测部署，可通过命令行参数`--device`以及`--backend`指定运行在不同的硬件以及推理引擎后端。

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# CPU 推理
./token_cls_infer_demo --model_dir ../../../best_models/msra/export --device cpu --backend paddle

# GPU 推理
./token_cls_infer_demo --model_dir ../../../best_models/msra/export --device gpu --backend paddle

```

运行完成后返回的结果如下：

```bash

[INFO] /paddle/PaddleNLP/model_zoo/ernie-3.0/fastdeploy/cpp/token_cls_infer.cc(103)::CreateRuntimeOption    model_path = ../../../best_models/msra/export/model.pdmodel, param_path = ../../../best_models/msra/export/model.pdiparams
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

### 量化模型部署

该示例支持部署 Paddle INT8 新格式量化模型，仅需在`--model_dir`参数传入量化模型路径，并且在对应硬件上选择可用的推理引擎后端，即可完成量化模型部署。在 GPU 上部署量化模型时，可选后端为`paddle_tensorrt`、`tensorrt`；在CPU上部署量化模型时，可选后端为`paddle`、`onnx_runtime`。下面将展示如何使用该示例完成量化模型部署，示例中的模型是按照 [ERNIE 3.0 训练文档](../../README.md) 压缩量化后导出得到的量化模型。

```bash

# 在 GPU 上使用 tensorrt 后端运行量化模型，模型目录可按照实际模型路径设置
./token_cls_infer_demo --model_dir ../../../best_models/msra/width_mult_0.75/mse16_1/ --device gpu --backend tensorrt --model_prefix int8

# 在 CPU 上使用paddle_inference后端，模型目录可按照实际模型路径设置
./token_cls_infer_demo --model_dir ../../../best_models/msra/width_mult_0.75/mse16_1/ --device cpu --backend paddle --model_prefix int8

```

运行完成后返回的结果如下：

```bash
[INFO] /paddle/PaddleNLP/model_zoo/ernie-3.0/fastdeploy/cpp/token_cls_infer.cc(103)::CreateRuntimeOption    model_path = ../../../best_models/msra/export/model.pdmodel, param_path = ../../../best_models/msra/export/model.pdiparams
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
|--backend | 支持的推理后端，可选范围: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，默认为'paddle' |
|--use_fp16 | 是否使用FP16模式进行推理。使用tensorrt和paddle_tensorrt后端时可开启，默认为False |

## FastDeploy 高阶用法

FastDeploy 在 C++ 端上，提供 `fastdeploy::RuntimeOption::UseXXX()` 以及 `fastdeploy::RuntimeOption::UseXXXBackend()` 接口支持开发者选择不同的硬件、不同的推理引擎进行部署。在不同的硬件上部署 ERNIE 3.0 模型，需要选择硬件所支持的推理引擎进行部署，下表展示如何在不同的硬件上选择可用的推理引擎部署 ERNIE 3.0 模型。

符号说明: (1) ✅: 已经支持; (2) ❔: 正在进行中; (3) N/A: 暂不支持;

<table>
    <tr>
        <td align=center> 硬件</td>
        <td align=center> 硬件对应的接口</td>
        <td align=center> 可用的推理引擎  </td>
        <td align=center> 推理引擎对应的接口 </td>
        <td align=center> 是否支持 Paddle 新格式量化模型 </td>
        <td align=center> 是否支持 FP16 模式 </td>
    </tr>
    <tr>
        <td rowspan=3 align=center> CPU </td>
        <td rowspan=3 align=center> UseCpu() </td>
        <td align=center> Paddle Inference </td>
        <td align=center> UsePaddleInferBackend() </td>
        <td align=center>  ✅ </td>
        <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> ONNX Runtime </td>
      <td align=center> UseOrtBackend() </td>
      <td align=center>  ✅ </td>
      <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> OpenVINO </td>
      <td align=center> UseOpenvinoBackend() </td>
      <td align=center> ❔ </td>
      <td align=center>  N/A </td>
    </tr>
    <tr>
        <td rowspan=4 align=center> GPU </td>
        <td rowspan=4 align=center> UseGpu() </td>
        <td align=center> Paddle Inference </td>
        <td align=center> UsePaddleInferBackend() </td>
        <td align=center>  ✅ </td>
        <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> ONNX Runtime </td>
      <td align=center> UseOrtBackend() </td>
      <td align=center>  ✅ </td>
      <td align=center>  ❔ </td>
    </tr>
    <tr>
      <td align=center> Paddle TensorRT </td>
      <td align=center> UseTrtBackend() + EnablePaddleToTrt() </td>
      <td align=center> ✅ </td>
      <td align=center> ✅ </td>
    </tr>
    <tr>
      <td align=center> TensorRT </td>
      <td align=center> UseTrtBackend() </td>
      <td align=center> ✅ </td>
      <td align=center> ✅ </td>
    </tr>
    <tr>
        <td align=center> 昆仑芯 XPU </td>
        <td align=center> UseKunlunXin() </td>
        <td align=center> Paddle Lite </td>
        <td align=center> UsePaddleLiteBackend() </td>
        <td align=center>  N/A </td>
        <td align=center>  ✅ </td>
    </tr>
    <tr>
        <td align=center> 华为 昇腾 </td>
        <td align=center> UseAscend() </td>
        <td align=center> Paddle Lite </td>
        <td align=center> UsePaddleLiteBackend() </td>
        <td align=center> ❔ </td>
        <td align=center> ✅ </td>
    </tr>
    <tr>
        <td align=center> Graphcore IPU </td>
        <td align=center> UseIpu() </td>
        <td align=center> Paddle Inference </td>
        <td align=center> UsePaddleInferBackend() </td>
        <td align=center> ❔ </td>
        <td align=center> N/A </td>
    </tr>
</table>

## 相关文档

[ERNIE 3.0模型详细介绍](../../README.md)

[ERNIE 3.0模型Python部署方法](../python/README.md)
