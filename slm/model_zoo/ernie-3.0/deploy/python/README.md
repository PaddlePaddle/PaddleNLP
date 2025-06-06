# FastDeploy ERNIE 3.0 模型 Python 部署示例

在部署前，参考 [FastDeploy SDK 安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)安装 FastDeploy Python SDK。

本目录下分别提供 `seq_cls_infer.py` 以及 `token_cls_infer.py` 快速完成在 CPU/GPU 的文本分类任务以及序列标注任务的 Python 部署示例。

## 文本分类任务

### 快速开始

以下示例展示如何完成 ERNIE 3.0 Medium 模型在 CLUE Benchmark 的 [AFQMC 数据集](https://github.com/CLUEbenchmark/CLUE)上进行文本分类任务的 Python 预测部署，可通过命令行参数`--device`指定运行在不同的硬件，并使用`--model_dir`参数指定运行的模型，具体参数设置可查看下面[参数说明](#参数说明)。示例中的模型是按照 [ERNIE 3.0 训练文档](../../README.md)导出得到的部署模型，其模型目录为`model_zoo/ernie-3.0/best_models/afqmc/export`（用户可按实际情况设置）。

```bash

# CPU 推理
python seq_cls_infer.py --model_dir ../../best_models/afqmc/export --device cpu

# GPU 推理
python seq_cls_infer.py --model_dir ../../best_models/afqmc/export --device gpu

```

运行完成后返回的结果如下：

```bash
I0423 05:00:21.622229  8408 print_statistics.cc:44] --- detected [85, 273] subgraphs!
--- Running PIR pass [dead_code_elimination_pass]
I0423 05:00:21.622710  8408 print_statistics.cc:50] --- detected [113] subgraphs!
--- Running PIR pass [replace_fetch_with_shadow_output_pass]
I0423 05:00:21.622859  8408 print_statistics.cc:50] --- detected [1] subgraphs!
--- Running PIR pass [remove_shadow_feed_pass]
I0423 05:00:21.626749  8408 print_statistics.cc:50] --- detected [2] subgraphs!
--- Running PIR pass [inplace_pass]
I0423 05:00:21.631474  8408 print_statistics.cc:50] --- detected [2] subgraphs!
I0423 05:00:21.631560  8408 analysis_predictor.cc:1186] ======= pir optimization completed =======
I0423 05:00:21.641817  8408 pir_interpreter.cc:1640] pir interpreter is running by trace mode ...
Batch 0, example 0 | s1: 花呗收款额度限制 | s2: 收钱码，对花呗支付的金额有限制吗 | label: 1 | score: 0.5175
Batch 1, example 0 | s1: 花呗支持高铁票支付吗 | s2: 为什么友付宝不支持花呗付款 | label: 0 | score: 0.9873

```

### 参数说明

`seq_cls_infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定部署模型的目录， |
|--batch_size |输入的 batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--device | 运行的设备，可选范围: ['cpu', 'gpu']，默认为'cpu' |

## 序列标注任务

### 快速开始

以下示例展示如何完成 ERNIE 3.0 Medium 模型在 CLUE Benchmark 的[ MSRA_NER 数据集](https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra)上进行序列标注任务的 Python 预测部署，可通过命令行参数`--device`指定运行在不同的硬件，并使用`--model_dir`参数指定运行的模型，具体参数设置可查看下面[参数说明](#参数说明)。示例中的模型是按照 [ERNIE 3.0 训练文档](../../README.md)导出得到的部署模型，其模型目录为`model_zoo/ernie-3.0/best_models/msra_ner/export`（用户可按实际情况设置）。


```bash

# CPU 推理
python token_cls_infer.py --model_dir ../../best_models/msra_ner/export/ --device cpu

# GPU 推理
python token_cls_infer.py --model_dir ../../best_models/msra_ner/export/ --device gpu

```

运行完成后返回的结果如下：

```bash
......
--- Running PIR pass [inplace_pass]
I0423 09:51:42.250245  4644 print_statistics.cc:50] --- detected [1] subgraphs!
I0423 09:51:42.250334  4644 analysis_predictor.cc:1186] ======= pir optimization completed =======
I0423 09:51:42.261358  4644 pir_interpreter.cc:1640] pir interpreter is running by trace mode ...
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
|--model_prefix| 模型文件前缀。前缀会分别与'PADDLE_INFERENCE_MODEL_SUFFIX'和'PADDLE_INFERENCE_WEIGHTS_SUFFIX'拼接得到模型文件名和参数文件名。默认为 'model'|

## 相关文档

[ERNIE 3.0模型详细介绍](../../README.md)
