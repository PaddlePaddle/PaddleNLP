# ERNIE 3.0 Tiny 模型 Python 推理示例

本目录下分别提供 `infer_demo.py` 快速完成在 CPU/GPU 的车载语音场景下的口语理解（Spoken Language Understanding，SLU）任务的 Python 部署示例，并展示端到端预测性能的 Benchmark。


## 快速开始

以下示例可通过命令行参数`--device`指定运行在不同的硬件，并使用`--model_dir`参数指定运行的模型，具体参数设置可查看下面[参数说明](#参数说明)。示例中的模型是按照[ERNIE 3.0 Tiny 训练文档](../../README.md)导出得到的部署模型，其模型目录为`model_zoo/ernie-tiny/output/BS64_LR5e-5_EPOCHS30/`（用户可按实际情况设置）。

```bash

# 在GPU上使用paddle_inference后端，模型目录可按照实际模型路径设置
python infer_demo.py --device gpu --model_dir ../../output/BS64_LR5e-5_EPOCHS30 --slot_label_path ../../data/slot_label.txt --intent_label_path ../../data/intent_label.txt

# 在CPU上使用paddle_inference后端，模型目录可按照实际模型路径设置
python infer_demo.py --device cpu --backend paddle --model_dir ../../output/BS64_LR5e-5_EPOCHS30 --slot_label_path ../../data/slot_label.txt --intent_label_path ../../data/intent_label.txt

```

运行完成后返回的结果如下：

```bash
......
--- Running PIR pass [inplace_pass]
I0423 14:02:46.963447  2082 print_statistics.cc:50] --- detected [2] subgraphs!
I0423 14:02:46.963521  2082 analysis_predictor.cc:1186] ======= pir optimization completed =======
I0423 14:02:46.971112  2082 pir_interpreter.cc:1640] pir interpreter is running by trace mode ...
No. 0 text = 来一首周华健的花心
{'intent': 'music.play', 'confidence': 0.9986396431922913, 'slot': [{'slot': 'song', 'entity': '来'}, {'slot': 'singer', 'entity': '华健的'}, {'slot': 'song', 'entity': '心'}]}
No. 1 text = 播放我们都一样
{'intent': 'music.play', 'confidence': 0.9983224272727966, 'slot': [{'slot': 'song', 'entity': '们都一样'}]}
No. 2 text = 到信阳市汽车配件城
{'intent': 'navigation.navigation', 'confidence': 0.9985769987106323, 'slot': [{'slot': 'destination', 'entity': '到'}, {'slot': 'destination', 'entity': '阳市汽车配件城'}]}
```

运行完成后返回的结果如下：

```bash

[INFO] fastdeploy/runtime.cc(517)::Init    Runtime initialized with Backend::TRT in Device::GPU.
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
|--model_dir | 指定部署模型的目录。支持传入 Paddle INT8 新格式量化模型。 |
|--slot_label_path| 指定的 slot label 文件路径 |
|--intent_label_path| 指定的 intent label 文件路径 |
|--batch_size |最大可测的 batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--model_prefix| 模型文件前缀。前缀会分别与'.pdmodel'和'.pdiparams'拼接得到模型文件名和参数文件名。默认为 'infer_model'|

## 相关文档

[ERNIE 3.0 Tiny 模型详细介绍](../../README.md)
