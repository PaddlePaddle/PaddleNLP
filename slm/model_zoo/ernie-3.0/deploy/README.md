# FastDeploy ERNIE 3.0 模型高性能部署

**⚡️FastDeploy** 是一款**全场景**、**易用灵活**、**极致高效**的 AI 推理部署工具，满足开发者**多硬件、多平台**的产业部署需求。开发者可以基于 FastDeploy 将训练好的预测模型在不同的硬件、不同的推理引擎后端上进行部署。目前 FastDeploy 提供多种编程语言的 SDK，包括 C++、Python 以及 Java SDK。

在部署 ERNIE 3.0 模型前，需要安装 FastDeploy SDK，可参考 [FastDeploy SDK 安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)确认部署环境是否满足 FastDeploy 环境要求，并按照介绍安装相应的 SDK。

目前，ERNIE 3.0 模型支持在如下的硬件以及推理引擎进行部署。

符号说明: (1) ✅: 已经支持; (2) ❔: 正在进行中; (3) N/A: 暂不支持;

<table>
    <tr>
        <td align=center> 硬件</td>
        <td align=center> 可用的推理引擎  </td>
        <td align=center> 是否支持 Paddle 新格式量化模型 </td>
        <td align=center> 是否支持 FP16 模式 </td>
    </tr>
    <tr>
        <td rowspan=3 align=center> CPU </td>
        <td align=center> Paddle Inference </td>
        <td align=center>  ✅ </td>
        <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> ONNX Runtime </td>
      <td align=center>  ✅ </td>
      <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> OpenVINO </td>
      <td align=center> ✅ </td>
      <td align=center>  N/A </td>
    </tr>
    <tr>
        <td rowspan=4 align=center> GPU </td>
        <td align=center> Paddle Inference </td>
        <td align=center>  ✅ </td>
        <td align=center>  N/A </td>
    </tr>
    <tr>
      <td align=center> ONNX Runtime </td>
      <td align=center>  ✅ </td>
      <td align=center>  ❔ </td>
    </tr>
    <tr>
      <td align=center> Paddle TensorRT </td>
      <td align=center> ✅ </td>
      <td align=center> ✅ </td>
    </tr>
    <tr>
      <td align=center> TensorRT </td>
      <td align=center> ✅ </td>
      <td align=center> ✅ </td>
    </tr>
    <tr>
        <td align=center> 昆仑芯 XPU </td>
        <td align=center> Paddle Lite </td>
        <td align=center>  N/A </td>
        <td align=center>  ✅ </td>
    </tr>
    <tr>
        <td align=center> 华为 昇腾 </td>
        <td align=center> Paddle Lite </td>
        <td align=center> ❔ </td>
        <td align=center> ✅ </td>
    </tr>
    <tr>
        <td align=center> Graphcore IPU </td>
        <td align=center> Paddle Inference </td>
        <td align=center> ❔ </td>
        <td align=center> N/A </td>
    </tr>
</table>

## 支持的 NLP 任务列表

符号说明: (1) ✅: 已经支持; (2) ❔: 正在进行中; (3) N/A: 暂不支持;

<table>
    <tr>
        <td align=center> 任务 Task</td>
        <td align=center> 部署方式  </td>
        <td align=center> 是否支持</td>
    </tr>
    <tr>
        <td rowspan=3 align=center> 文本分类 </td>
        <td align=center> Python </td>
        <td align=center>  ✅ </td>
    </tr>
    <tr>
      <td align=center> C++ </td>
      <td align=center>  ✅ </td>
    </tr>
    <tr>
      <td align=center> Serving </td>
      <td align=center> ✅ </td>
    </tr>
    <tr>
        <td rowspan=4 align=center> 序列标注 </td>
        <td align=center> Python </td>
        <td align=center>  ✅ </td>
    </tr>
    <tr>
      <td align=center> C++ </td>
      <td align=center>  ✅ </td>
    </tr>
    <tr>
      <td align=center> Serving </td>
      <td align=center> ✅ </td>
    </tr>
</table>

## 详细部署文档

ERNIE 3.0 模型支持 Python、C++ 部署以及 Serving 服务化部署。以下是详细文档。

- [Python 部署](python/README.md)
- [Serving 部署](serving/README.md)
