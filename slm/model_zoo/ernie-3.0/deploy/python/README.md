# FastDeploy ERNIE 3.0 模型 Python 部署示例

在部署前，参考 [FastDeploy SDK 安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)安装 FastDeploy Python SDK。

本目录下分别提供 `seq_cls_infer.py` 以及 `token_cls_infer.py` 快速完成在 CPU/GPU 的文本分类任务以及序列标注任务的 Python 部署示例。

## 文本分类任务

### 快速开始

以下示例展示如何基于 FastDeploy 库完成 ERNIE 3.0 Medium 模型在 CLUE Benchmark 的 [AFQMC 数据集](https://github.com/CLUEbenchmark/CLUE)上进行文本分类任务的 Python 预测部署，可通过命令行参数`--device`以及`--backend`指定运行在不同的硬件以及推理引擎后端，并使用`--model_dir`参数指定运行的模型，具体参数设置可查看下面[参数说明](#参数说明)。示例中的模型是按照 [ERNIE 3.0 训练文档](../../README.md)导出得到的部署模型，其模型目录为`model_zoo/ernie-3.0/best_models/afqmc/export`（用户可按实际情况设置）。

```bash

# CPU 推理
python seq_cls_infer.py --model_dir ../../best_models/afqmc/export --device cpu --backend paddle

# GPU 推理
python seq_cls_infer.py --model_dir ../../best_models/afqmc/export --device gpu --backend paddle

```

运行完成后返回的结果如下：

```bash

[INFO] fastdeploy/runtime.cc(596)::Init    Runtime initialized with Backend::PDINFER in Device::CPU.
Batch id:0, example id:0, sentence1:花呗收款额度限制, sentence2:收钱码，对花呗支付的金额有限制吗, label:0, similarity:0.5099
Batch id:1, example id:0, sentence1:花呗支持高铁票支付吗, sentence2:为什么友付宝不支持花呗付款, label:0, similarity:0.9862

```

### 量化模型部署

该示例支持部署 Paddle INT8 新格式量化模型，仅需在`--model_dir`参数传入量化模型路径，并且在对应硬件上选择可用的推理引擎后端，即可完成量化模型部署。在 GPU 上部署量化模型时，可选后端为`paddle_tensorrt`、`tensorrt`；在 CPU 上部署量化模型时，可选后端为`paddle`、`onnx_runtime`。下面将展示如何使用该示例完成量化模型部署，示例中的模型是按照 [ERNIE 3.0 训练文档](../../README.md) 压缩量化后导出得到的量化模型。

```bash

# 在GPU上使用 tensorrt 后端，模型目录可按照实际模型路径设置
python seq_cls_infer.py --model_dir ../../best_models/afqmc/width_mult_0.75/mse16_1/ --device gpu --backend tensorrt --model_prefix int8

# 在CPU上使用paddle_inference后端，模型目录可按照实际模型路径设置
python seq_cls_infer.py --model_dir ../../best_models/afqmc/width_mult_0.75/mse16_1/ --device cpu --backend paddle --model_prefix int8

```

运行完成后返回的结果如下：

```bash
[INFO] fastdeploy/runtime/runtime.cc(101)::Init    Runtime initialized with Backend::PDINFER in Device::GPU.
Batch id:0, example id:0, sentence1:花呗收款额度限制, sentence2:收钱码，对花呗支付的金额有限制吗, label:0, similarity:0.5224
Batch id:1, example id:0, sentence1:花呗支持高铁票支付吗, sentence2:为什么友付宝不支持花呗付款, label:0, similarity:0.9856
```


### 参数说明

`seq_cls_infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定部署模型的目录， |
|--batch_size |输入的 batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--device | 运行的设备，可选范围: ['cpu', 'gpu']，默认为'cpu' |
|--backend | 支持的推理后端，可选范围: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，默认为'paddle' |
|--use_fp16 | 是否使用 FP16模式进行推理。使用 tensorrt 和 paddle_tensorrt 后端时可开启，默认为 False |

## 序列标注任务

### 快速开始

以下示例展示如何基于 FastDeploy 库完成 ERNIE 3.0 Medium 模型在 CLUE Benchmark 的[ MSRA_NER 数据集](https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra)上进行序列标注任务的 Python 预测部署，可通过命令行参数`--device`以及`--backend`指定运行在不同的硬件以及推理引擎后端，并使用`--model_dir`参数指定运行的模型，具体参数设置可查看下面[参数说明](#参数说明)。示例中的模型是按照 [ERNIE 3.0 训练文档](../../README.md)导出得到的部署模型，其模型目录为`model_zoo/ernie-3.0/best_models/msra_ner/export`（用户可按实际情况设置）。


```bash

# CPU 推理
python token_cls_infer.py --model_dir ../../best_models/msra_ner/export/ --device cpu --backend paddle

# GPU 推理
python token_cls_infer.py --model_dir ../../best_models/msra_ner/export/ --device gpu --backend paddle

```

运行完成后返回的结果如下：

```bash

[INFO] fastdeploy/runtime.cc(500)::Init    Runtime initialized with Backend::PDINFER in Device::CPU.
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

### 量化模型部署

该示例支持部署 Paddle INT8 新格式量化模型，仅需在`--model_dir`参数传入量化模型路径，并且在对应硬件上选择可用的推理引擎后端，即可完成量化模型部署。在 GPU 上部署量化模型时，可选后端为`paddle_tensorrt`、`tensorrt`；在 CPU 上部署量化模型时，可选后端为`paddle`、`onnx_runtime`。下面将展示如何使用该示例完成量化模型部署，示例中的模型是按照 [ERNIE 3.0 训练文档](../../README.md) 压缩量化后导出得到的量化模型。

```bash

# 在GPU上使用 tensorrt 后端，模型目录可按照实际模型路径设置
python token_cls_infer.py --model_dir ../../best_models/msra_ner/width_mult_0.75/mse16_1/ --device gpu --backend tensorrt --model_prefix int8

# 在CPU上使用paddle_inference后端，模型目录可按照实际模型路径设置
python token_cls_infer.py --model_dir ../../best_models/msra_ner/width_mult_0.75/mse16_1/ --device cpu --backend paddle --model_prefix int8

```

运行完成后返回的结果如下：

```bash

[INFO] fastdeploy/runtime.cc(500)::Init    Runtime initialized with Backend::PDINFER in Device::CPU.
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

### 参数说明

`token_cls_infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定部署模型的目录， |
|--batch_size |输入的 batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--device | 运行的设备，可选范围: ['cpu', 'gpu']，默认为'cpu' |
|--backend | 支持的推理后端，可选范围: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，默认为'paddle' |
|--use_fp16 | 是否使用 FP16模式进行推理。使用 tensorrt 和 paddle_tensorrt 后端时可开启，默认为 False |
|--model_prefix| 模型文件前缀。前缀会分别与'.pdmodel'和'.pdiparams'拼接得到模型文件名和参数文件名。默认为 'model'|


## FastDeploy 高阶用法

FastDeploy 在 Python 端上，提供 `fastdeploy.RuntimeOption.use_xxx()` 以及 `fastdeploy.RuntimeOption.use_xxx_backend()` 接口支持开发者选择不同的硬件、不同的推理引擎进行部署。在不同的硬件上部署 ERNIE 3.0 模型，需要选择硬件所支持的推理引擎进行部署，下表展示如何在不同的硬件上选择可用的推理引擎部署 ERNIE 3.0 模型。

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
        <td rowspan=3 align=center> use_cpu() </td>
        <td align=center> Paddle Inference </td>
        <td align=center> use_paddle_infer_backend() </td>
        <td align=center>  ✅ </td>
        <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> ONNX Runtime </td>
      <td align=center> use_ort_backend() </td>
      <td align=center>  ✅ </td>
      <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> OpenVINO </td>
      <td align=center> use_openvino_backend() </td>
      <td align=center> ❔ </td>
      <td align=center>  N/A </td>
    </tr>
    <tr>
        <td rowspan=4 align=center> GPU </td>
        <td rowspan=4 align=center> use_gpu() </td>
        <td align=center> Paddle Inference </td>
        <td align=center> use_paddle_infer_backend() </td>
        <td align=center>  ✅ </td>
        <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> ONNX Runtime </td>
      <td align=center> use_ort_backend() </td>
      <td align=center>  ✅ </td>
      <td align=center>  ❔ </td>
    </tr>
    <tr>
      <td align=center> Paddle TensorRT </td>
      <td align=center> use_trt_backend() + enable_paddle_to_trt() </td>
      <td align=center> ✅ </td>
      <td align=center> ✅ </td>
    </tr>
    <tr>
      <td align=center> TensorRT </td>
      <td align=center> use_trt_backend() </td>
      <td align=center> ✅ </td>
      <td align=center> ✅ </td>
    </tr>
    <tr>
        <td align=center> 昆仑芯 XPU </td>
        <td align=center> use_kunlunxin() </td>
        <td align=center> Paddle Lite </td>
        <td align=center> use_paddle_lite_backend() </td>
        <td align=center>  N/A </td>
        <td align=center>  ✅  </td>
    </tr>
    <tr>
        <td align=center> 华为 昇腾 </td>
        <td align=center> use_ascend() </td>
        <td align=center> Paddle Lite </td>
        <td align=center> use_paddle_lite_backend() </td>
        <td align=center> ❔ </td>
        <td align=center> ✅ </td>
    </tr>
    <tr>
        <td align=center> Graphcore IPU </td>
        <td align=center> use_ipu() </td>
        <td align=center> Paddle Inference </td>
        <td align=center> use_paddle_infer_backend() </td>
        <td align=center> ❔ </td>
        <td align=center> N/A </td>
    </tr>
</table>

## 相关文档

[ERNIE 3.0模型详细介绍](../../README.md)
