# FastDeploy ERNIE Tiny 模型Python部署示例

在部署前，参考[FastDeploy SDK安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)安装FastDeploy Python SDK。

本目录下分别提供`infer_demo.py`快速完成在CPU/GPU的车载语音场景下的口语理解（Spoken Language Understanding，SLU）任务的Python部署示例。


## 依赖安装

直接执行以下命令安装部署示例的依赖。

```bash

# 安装fast_tokenizer以及GPU版本fastdeploy
pip install fast-tokenizer-python fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html

```

## 快速开始

以下示例可通过命令行参数`--device`以及`--backend`指定运行在不同的硬件以及推理引擎后端。示例中的模型是按照[ERNIE Tiny训练文档](../../README.md)导出得到的部署模型。

```bash

# 在GPU上使用paddle_inference后端
python infer_demo.py --device gpu --backend paddle --model_dir ../../output/BS64_LR5e-5_20EPOCHS_WD0.01_WR0.1 --slot_label_path ../../data/slots_label.txt --intent_label_path ../../data/intent_label.txt

# 在CPU上使用paddle_inference后端
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


## 参数说明

除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定部署模型的目录 |
|--slot_label_path| 指定的slot label文件路径 |
|--intent_label_path| 指定的intent label文件路径 |
|--test_data_path| 指定的测试集路径，默认为空。 |
|--batch_size |最大可测的 batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--device | 运行的设备，可选范围: ['cpu', 'kunlunxin', 'gpu']，默认为'cpu' |
|--backend | 支持的推理后端，可选范围: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，默认为'paddle' |
|--use_fp16 | 是否使用FP16模式进行推理。使用tensorrt和paddle_tensorrt后端时可开启，默认为False |
|--use_fast| 是否使用FastTokenizer加速分词阶段。默认为True|

## 相关文档

[ERNIE Tiny模型详细介绍](../../README.md)

[ERNIE Tiny模型C++部署方法](../cpp/README.md)

[ERNIE Tiny模型Android部署方法](../android/README.md)

[FastDeploy SDK安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)
