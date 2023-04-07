# FastDeploy UTC 模型 Python 部署示例

在部署前，参考 [FastDeploy SDK 安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)安装 FastDeploy Python SDK。

本目录下提供 `infer.py` 快速完成在 CPU/GPU 的通用文本分类任务的 Python 部署示例。

## 依赖安装

直接执行以下命令安装部署示例的依赖。

```bash
# 安装 fast_tokenizer 以及 GPU 版本 fastdeploy
pip install fast-tokenizer-python fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

## 快速开始

以下示例展示如何基于 FastDeploy 库完成 UTC 模型进行文本分类任务的 Python 预测部署，可通过命令行参数`--device`以及`--backend`指定运行在不同的硬件以及推理引擎后端，并使用`--model_dir`参数指定运行的模型，具体参数设置可查看下面[参数说明](#参数说明)。示例中的模型是按照 [UTC 训练文档](../../README.md)导出得到的部署模型，其模型目录为 `application/zero_shot_text_classification/checkpoint/model_best`（用户可按实际情况设置）。


```bash
# CPU 推理
python infer.py --model_dir ../../checkpoint/model_best --device cpu
# GPU 推理
python infer.py --model_dir ../../checkpoint/model_best --device gpu
```

运行完成后返回的结果如下：

```bash
[2023-03-02 06:32:47,528] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer'> to load './checkpoint/model_best'.
[INFO] fastdeploy/runtime/runtime.cc(266)::CreatePaddleBackend    Runtime initialized with Backend::PDINFER in Device::GPU.
[2023-03-02 06:33:18,120] [    INFO] - Assigning ['[O-MASK]'] to the additional_special_tokens key of the tokenizer
[{'predictions': [{'label': '这是一条好评', 'score': 0.9073}], 'text_a': '房间干净明亮，非常不错'}]
```

## 参数说明

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定部署模型的目录， |
|--batch_size |输入的batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--num_omask_tokens | 最大标签数量，默认为64|
|--device | 运行的设备，可选范围: ['cpu', 'gpu']，默认为'cpu' |
|--device_id | 运行设备的id。默认为0。 |
|--cpu_threads | 当使用cpu推理时，指定推理的cpu线程数，默认为1。|
|--backend | 支持的推理后端，可选范围: ['onnx_runtime', 'paddle', 'tensorrt', 'paddle_tensorrt']，默认为'paddle' |
|--use_fp16 | 是否使用FP16模式进行推理。使用tensorrt和paddle_tensorrt后端时可开启，默认为False |

## FastDeploy 高阶用法

FastDeploy 在 Python 端上，提供 `fastdeploy.RuntimeOption.use_xxx()` 以及 `fastdeploy.RuntimeOption.use_xxx_backend()` 接口支持开发者选择不同的硬件、不同的推理引擎进行部署。在不同的硬件上部署 UTC 模型，需要选择硬件所支持的推理引擎进行部署，下表展示如何在不同的硬件上选择可用的推理引擎部署 UTC 模型。

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
        <td rowspan=2 align=center> CPU </td>
        <td rowspan=2 align=center> use_cpu() </td>
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
      <td align=center> use_paddle_infer_backend() + paddle_infer_option.enable_trt = True </td>
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
