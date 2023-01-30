# FastDeploy ERNIE 3.0 Tiny 模型 C++ 部署示例

在部署前，参考 [FastDeploy SDK安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md) 安装 FastDeploy C++ SDK。

本目录下分别提供 `infer_demo.cc` 快速完成在 CPU/GPU 的车载语音场景下的口语理解（Spoken Language Understanding，SLU）任务的 C++ 部署示例。

## 依赖安装

下载 FastDeploy 预编译库，用户可在上文提到的 [FastDeploy SDK安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md) 中自行选择合适的版本使用（例如1.0.2）。在 Linux 64位系统下，可以执行以下命令完成安装。

```bash

# 安装linux x64平台GPU版本的FastDeploy SDK
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-x.x.x.tgz
tar xvf fastdeploy-linux-x64-gpu-x.x.x.tgz

```

## 快速开始

以下示例可通过命令行参数`--device`以及`--backend`指定运行在不同的硬件以及推理引擎后端，并使用`--model_dir`参数指定运行的模型，具体参数设置可查看下面[参数说明](#参数说明)。示例中的模型是按照 [ERNIE 3.0 Tiny 训练文档](../../README.md) 导出得到的部署模型，其模型目录为`model_zoo/ernie-tiny/output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1/`（用户可按实际情况设置）。

```bash

mkdir build
cd build

# 指定解压后fastdeploy sdk目录进行编译
cmake .. -DFASTDEPLOY_INSTALL_DIR=/path/to/fastdeploy-linux-x64-gpu-x.x.x
make -j

# 在GPU上使用paddle_inference后端，模型目录可按照实际模型路径设置
./infer_demo --device gpu --backend paddle --model_dir ../../../output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1 --slot_label_path ../../../data/slot_label.txt --intent_label_path ../../../data/intent_label.txt

# 在CPU上使用paddle_inference后端，模型目录可按照实际模型路径设置
./infer_demo --device cpu --backend paddle --model_dir ../../../output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1 --slot_label_path /path/to/slot_label.txt --intent_label_path ../../../data/intent_label.txt

```

运行完成后返回的结果如下：

```bash

[INFO] /paddle/PaddleNLP/model_zoo/ernie-tiny/deploy/cpp/infer_demo.cc(108)::CreateRuntimeOption    model_path = ../../ernie-tiny/output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1/infer_model.pdmodel, param_path = ../../../output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1/infer_model.pdiparams
[INFO] fastdeploy/runtime.cc(596)::Init    Runtime initialized with Backend::PDINFER in Device::GPU.
No.0 text = 来一首周华健的花心
intent result: label = music.play, confidence = 0.99834
slot result:
slot = singer, entity = '周华健', pos = [3, 5]
slot = song, entity = '花心', pos = [7, 8]

No.1 text = 播放我们都一样
intent result: label = music.play, confidence = 0.998516
slot result:
slot = song, entity = '我们都一样', pos = [2, 6]

No.2 text = 到信阳市汽车配件城
intent result: label = navigation.navigation, confidence = 0.998626
slot result:
slot = destination, entity = '信阳市汽车配件城', pos = [1, 8]

```

### 量化模型部署

该示例支持部署 Paddle INT8 新格式量化模型，仅需在`--model_dir`参数传入量化模型路径，并且在对应硬件上选择可用的推理引擎后端，即可完成量化模型部署。在 GPU 上部署量化模型时，可选后端为`paddle_tensorrt`、`tensorrt`；在CPU上部署量化模型时，可选后端为`paddle`、`onnx_runtime`。下面将展示如何使用该示例完成量化模型部署，示例中的模型是按照 [ERNIE 3.0 Tiny训练文档](../../README.md) 压缩量化后导出得到的量化模型。

```bash

# 在 GPU 上使用 tensorrt 后端运行量化模型，模型目录可按照实际模型路径设置
./infer_demo --device gpu --backend tensorrt --model_prefix int8 --model_dir ../../../output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1_quant --slot_label_path ../../../data/slot_label.txt --intent_label_path ../../../data/intent_label.txt

# 在 CPU 上使用 paddle_inference 后端，模型目录可按照实际模型路径设置
./infer_demo --device cpu --backend paddle --model_prefix int8 --model_dir ../../../output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1_quant --slot_label_path /path/to/slot_label.txt --intent_label_path ../../../data/intent_label.txt

```

运行完成后返回的结果如下：

```bash
[INFO] fastdeploy/runtime.cc(596)::Init    Runtime initialized with Backend::TRT in Device::GPU.
No.0 text = 来一首周华健的花心
intent result: label = music.play, confidence = 0.997125
slot result:
slot = singer, entity = '周华健', pos = [3, 5]
slot = song, entity = '花心', pos = [7, 8]

No.1 text = 播放我们都一样
intent result: label = music.play, confidence = 0.997346
slot result:
slot = song, entity = '我们都一样', pos = [2, 6]

No.2 text = 到信阳市汽车配件城
intent result: label = navigation.navigation, confidence = 0.998141
slot result:
slot = destination, entity = '信阳市汽车配件城', pos = [1, 8]

```

## 参数说明

除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|--device | 运行的设备，可选范围: ['cpu', 'gpu']，默认为'cpu' |
|--backend | 支持的推理后端，可选范围: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，默认为'paddle' |
|--model_dir | 指定部署模型的目录。支持传入 Paddle INT8 新格式量化模型。 |
|--vocab_path| 指定的模型词表路径，默认为`model_dir`目录下的vocab.txt文件 |
|--added_tokens_path| 指定的模型词表路径，默认为`model_dir`目录下的added_tokens.json文件 |
|--slot_label_path| 指定的 slot label 文件路径 |
|--intent_label_path| 指定的 intent label 文件路径 |
|--batch_size |最大可测的 batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--use_trt_fp16 | 是否使用 FP16 模式进行推理。使用 TensorRT 和 Paddle TensorRT 后端时可开启，默认为False |
|--model_prefix| 模型文件前缀。前缀会分别与'.pdmodel'和'.pdiparams'拼接得到模型文件名和参数文件名。默认为 'infer_model'|

## FastDeploy 高阶用法

FastDeploy 在 C++ 端上，提供 `fastdeploy::RuntimeOption::UseXXX()` 以及 `fastdeploy::RuntimeOption::UseXXXBackend()` 接口支持开发者选择不同的硬件、不同的推理引擎进行部署。在不同的硬件上部署 ERNIE 3.0 Tiny 模型，需要选择硬件所支持的推理引擎进行部署，下表展示如何在不同的硬件上选择可用的推理引擎部署 ERNIE 3.0 Tiny 模型。

符号说明: (1) ✅: 已经支持; (2) ❔: 正在进行中; (3) N/A: 暂不支持;

<table>
    <tr>
        <td align=center> 硬件</td>
        <td align=center> 硬件对应的接口</td>
        <td align=center> 可用的推理引擎  </td>
        <td align=center> 推理引擎对应的接口 </td>
        <td align=center> 是否支持 ERNIE 3.0 Tiny 模型 </td>
        <td align=center> 是否支持 Paddle 新格式量化模型 </td>
        <td align=center> 是否支持 FP16 模式 </td>
    </tr>
    <tr>
        <td rowspan=3 align=center> CPU </td>
        <td rowspan=3 align=center> UseCpu() </td>
        <td align=center> Paddle Inference </td>
        <td align=center> UsePaddleInferBackend() </td>
        <td align=center>  ✅ </td>
        <td align=center>  ✅ </td>
        <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> ONNX Runtime </td>
      <td align=center> UseOrtBackend() </td>
      <td align=center> ✅ </td>
      <td align=center>  ✅ </td>
      <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> OpenVINO </td>
      <td align=center> UseOpenvinoBackend() </td>
      <td align=center> ✅ </td>
      <td align=center> ❔ </td>
      <td align=center>  N/A </td>
    </tr>
    <tr>
        <td rowspan=4 align=center> GPU </td>
        <td rowspan=4 align=center> UseGpu() </td>
        <td align=center> Paddle Inference </td>
        <td align=center> UsePaddleInferBackend() </td>
        <td align=center>  ✅ </td>
        <td align=center>  ✅ </td>
        <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> ONNX Runtime </td>
      <td align=center> UseOrtBackend() </td>
      <td align=center> ✅ </td>
      <td align=center>  ✅ </td>
      <td align=center>  ❔ </td>
    </tr>
    <tr>
      <td align=center> Paddle TensorRT </td>
      <td align=center> UseTrtBackend() + EnablePaddleToTrt() </td>
      <td align=center> ✅  </td>
      <td align=center> ✅ </td>
      <td align=center> ✅ </td>
    </tr>
    <tr>
      <td align=center> TensorRT </td>
      <td align=center> UseTrtBackend() </td>
      <td align=center> ✅  </td>
      <td align=center> ✅ </td>
      <td align=center> ✅ </td>
    </tr>
    <tr>
        <td align=center> 昆仑芯 XPU </td>
        <td align=center> UseKunlunXin() </td>
        <td align=center> Paddle Lite </td>
        <td align=center> UsePaddleLiteBackend() </td>
        <td align=center>  ✅ </td>
        <td align=center>  N/A </td>
        <td align=center>  ✅ </td>
    </tr>
    <tr>
        <td align=center> 华为 昇腾 </td>
        <td align=center> UseAscend() </td>
        <td align=center> Paddle Lite </td>
        <td align=center> UsePaddleLiteBackend() </td>
        <td align=center> ✅ </td>
        <td align=center> ❔ </td>
        <td align=center> ✅ </td>
    </tr>
    <tr>
        <td align=center> Graphcore IPU </td>
        <td align=center> UseIpu() </td>
        <td align=center> Paddle Inference </td>
        <td align=center> UsePaddleInferBackend() </td>
        <td align=center> ❔ </td>
        <td align=center> ❔ </td>
        <td align=center> N/A </td>
    </tr>
</table>

## 相关文档

[ERNIE 3.0 Tiny模型详细介绍](../../README.md)

[ERNIE 3.0 Tiny模型Python部署方法](../python/README.md)

[ERNIE 3.0 Tiny模型Android部署方法](../android/README.md)

[FastDeploy SDK安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)
