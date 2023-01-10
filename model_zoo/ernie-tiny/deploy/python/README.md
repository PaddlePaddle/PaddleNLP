# FastDeploy ERNIE 3.0 Tiny 模型Python部署示例

在部署前，参考[FastDeploy SDK安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)安装FastDeploy Python SDK。

本目录下分别提供`infer_demo.py`快速完成在CPU/GPU的车载语音场景下的口语理解（Spoken Language Understanding，SLU）任务的Python部署示例。


## 依赖安装

直接执行以下命令安装部署示例的依赖。

```bash

# 安装fast_tokenizer以及GPU版本fastdeploy
pip install fast-tokenizer-python fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html

```

## 快速开始

以下示例可通过命令行参数`--device`以及`--backend`指定运行在不同的硬件以及推理引擎后端，并使用`--model_dir`参数指定运行的模型，具体参数设置可查看下面[参数说明](#参数说明)。示例中的模型是按照[ERNIE 3.0 Tiny训练文档](../../README.md)导出得到的部署模型，其模型目录为`model_zoo/ernie-tiny/output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1/`（用户可按实际情况设置）。

```bash

# 在GPU上使用paddle_inference后端，模型目录可按照实际模型路径设置
python infer_demo.py --device gpu --backend paddle --model_dir ../../output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1 --slot_label_path ../../data/slots_label.txt --intent_label_path ../../data/intent_label.txt

# 在CPU上使用paddle_inference后端，模型目录可按照实际模型路径设置
python infer_demo.py --device cpu --backend paddle --model_dir ../../output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1 --slot_label_path ../../data/slots_label.txt --intent_label_path ../../data/intent_label.txt

```

运行完成后返回的结果如下：

```bash

[INFO] fastdeploy/runtime.cc(596)::Init    Runtime initialized with Backend::PDINFER in Device::GPU.
No. 0 text = 来一首周华健的花心
{'intent': 'music.play', 'confidence': 0.99833965, 'slot': [{'slot': 'singer', 'entity': '周华健', 'pos': [3, 5]}, {'slot': 'song', 'entity': '花心', 'pos': [7, 8]}]}
No. 1 text = 播放我们都一样
{'intent': 'music.play', 'confidence': 0.9985164, 'slot': [{'slot': 'song', 'entity': '我们都一样', 'pos': [2, 6]}]}
No. 2 text = 到信阳市汽车配件城
{'intent': 'navigation.navigation', 'confidence': 0.998626, 'slot': [{'slot': 'destination', 'entity': '信阳市汽车配件城', 'pos': [1, 8]}]}

```

### 量化模型部署

该示例支持部署Paddle INT8新格式量化模型，仅需在`--model_dir`参数传入量化模型路径，并且在对应硬件上选择可用的推理引擎后端，即可完成量化模型部署。在GPU上部署量化模型时，可选后端为`paddle_tensorrt`、`tensorrt`；在CPU上部署量化模型时，可选后端为`paddle`、`onnx_runtime`。下面将展示如何使用该示例完成量化模型部署，示例中的模型是按照[ERNIE 3.0 Tiny训练文档](../../README.md)压缩量化后导出得到的量化模型。

```bash

# 在GPU上使用paddle_tensorrt后端，模型目录可按照实际模型路径设置
python infer_demo.py --device gpu --backend paddle --model_prefix int8 --model_dir ../../output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1 --slot_label_path ../../data/slots_label.txt --intent_label_path ../../data/intent_label.txt

# 在CPU上使用paddle_inference后端，模型目录可按照实际模型路径设置
python infer_demo.py --device cpu --backend paddle --model_prefix int8 --model_dir ../../output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1_quant --slot_label_path ../../data/slots_label.txt --intent_label_path ../../data/intent_label.txt

```

运行完成后返回的结果如下：

```bash

[INFO] fastdeploy/runtime.cc(517)::Init	Runtime initialized with Backend::PDINFER in Device::GPU.
No. 0 text = 来一首周华健的花心
{'intent': 'music.play', 'confidence': 0.99706995, 'slot': [{'slot': 'singer', 'entity': '周华健', 'pos': [3, 5]}, {'slot': 'song', 'entity': '花心', 'pos': [7, 8]}]}
No. 1 text = 播放我们都一样
{'intent': 'music.play', 'confidence': 0.9973666, 'slot': [{'slot': 'song', 'entity': '我们都一样', 'pos': [2, 6]}]}
No. 2 text = 到信阳市汽车配件城
{'intent': 'navigation.navigation', 'confidence': 0.99799216, 'slot': [{'slot': 'destination', 'entity': '信阳市汽车配件城', 'pos': [1, 8]}]}

```

## 参数说明

除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|--device | 运行的设备，可选范围: ['cpu', 'gpu']，默认为'cpu' |
|--backend | 支持的推理后端，可选范围: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，默认为'paddle' |
|--model_dir | 指定部署模型的目录。支持传入Paddle INT8新格式量化模型。 |
|--slot_label_path| 指定的slot label文件路径 |
|--intent_label_path| 指定的intent label文件路径 |
|--batch_size |最大可测的 batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--use_trt_fp16 | 是否使用FP16模式进行推理。使用tensorrt和paddle_tensorrt后端时可开启，默认为False |
|--use_fast| 是否使用FastTokenizer加速分词阶段。默认为True|
|--model_prefix| 模型文件前缀。前缀会分别与'.pdmodel'和'.pdiparams'拼接得到模型文件名和参数文件名。默认为 'infer_model'|

## FastDeploy 高阶用法

FastDeploy在Python端上，提供`fastdeploy.RuntimeOption.use_xxx()`以及`fastdeploy.RuntimeOption.use_xxx_backend()`接口支持开发者选择不同的硬件、不同的推理引擎进行部署。在不同的硬件上部署ERNIE 3.0 Tiny模型，需要选择硬件所支持的推理引擎进行部署，下表展示如何在不同的硬件上选择可用的推理引擎部署ERNIE 3.0 Tiny模型。

符号说明: (1) ✅: 已经支持; (2) ❔: 正在进行中; (3) N/A: 暂不支持;

<table>
    <tr>
        <td align=center> 硬件</td>
        <td align=center> 硬件对应的接口</td>
        <td align=center> 可用的推理引擎  </td>
        <td align=center> 推理引擎对应的接口 </td>
        <td align=center> 是否支持ERNIE 3.0 Tiny模型 </td>
        <td align=center> 是否支持Paddle新格式量化模型 </td>
        <td align=center> 是否支持FP16模式 </td>
    </tr>
    <tr>
        <td rowspan=3 align=center> CPU </td>
        <td rowspan=3 align=center> use_cpu() </td>
        <td align=center> Paddle Inference </td>
        <td align=center> use_paddle_infer_backend() </td>
        <td align=center>  ✅ </td>
        <td align=center>  ✅ </td>
        <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> ONNX Runtime </td>
      <td align=center> use_ort_backend() </td>
      <td align=center> ✅ </td>
      <td align=center>  ✅ </td>
      <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> OpenVINO </td>
      <td align=center> use_openvino_backend() </td>
      <td align=center> ✅ </td>
      <td align=center> ❔ </td>
      <td align=center>  N/A </td>
    </tr>
    <tr>
        <td rowspan=4 align=center> GPU </td>
        <td rowspan=4 align=center> use_gpu() </td>
        <td align=center> Paddle Inference </td>
        <td align=center> use_paddle_infer_backend() </td>
        <td align=center>  ✅ </td>
        <td align=center>  ✅ </td>
        <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> ONNX Runtime </td>
      <td align=center> use_ort_backend() </td>
      <td align=center> ✅ </td>
      <td align=center>  ✅ </td>
      <td align=center>  ❔ </td>
    </tr>
    <tr>
      <td align=center> Paddle TensorRT </td>
      <td align=center> use_trt_backend() + enable_paddle_to_trt() </td>
      <td align=center> ✅  </td>
      <td align=center> ✅ </td>
      <td align=center> ✅ </td>
    </tr>
    <tr>
      <td align=center> TensorRT </td>
      <td align=center> use_trt_backend() </td>
      <td align=center> ✅  </td>
      <td align=center> ✅ </td>
      <td align=center> ✅ </td>
    </tr>
    <tr>
        <td align=center> 昆仑芯 XPU </td>
        <td align=center> use_kunlunxin() </td>
        <td align=center> Paddle Lite </td>
        <td align=center> use_paddle_lite_backend() </td>
        <td align=center>  ✅ </td>
        <td align=center>  N/A </td>
        <td align=center>  N/A </td>
    </tr>
    <tr>
        <td align=center> 华为 昇腾 </td>
        <td align=center> use_ascend() </td>
        <td align=center> Paddle Lite </td>
        <td align=center> use_paddle_lite_backend() </td>
        <td align=center> ✅ </td>
        <td align=center> ❔ </td>
        <td align=center> ✅ </td>
    </tr>
    <tr>
        <td align=center> Graphcore IPU </td>
        <td align=center> use_ipu() </td>
        <td align=center> Paddle Inference </td>
        <td align=center> use_paddle_infer_backend() </td>
        <td align=center> ❔ </td>
        <td align=center> ❔ </td>
        <td align=center> N/A </td>
    </tr>
</table>

## 相关文档

[ERNIE 3.0 Tiny模型详细介绍](../../README.md)

[ERNIE 3.0 Tiny模型C++部署方法](../cpp/README.md)

[ERNIE 3.0 Tiny模型Android部署方法](../android/README.md)

[FastDeploy SDK安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)
