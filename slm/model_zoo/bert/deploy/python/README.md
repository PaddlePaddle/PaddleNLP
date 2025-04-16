# BERT 模型 Python 推理示例
本目录下提供 `seq_cls_infer.py` 快速完成在 CPU/GPU 的 GLUE 文本分类任务的 Python 示例。

## 快速开始

可通过命令行参数`--device`指定运行在不同的硬件，并使用`--model_dir`参数指定运行的模型，具体参数设置可查看下面[参数说明](#参数说明)。示例中的模型是按照 [BERT 训练文档](../../README.md)导出得到的部署模型，其模型目录为`model_zoo/bert/infer_model`（用户可按实际情况设置）。


```bash
# CPU 推理
python infer.py --model_dir ../../infer_model/ --device cpu
# GPU 推理
python infer.py --model_dir ../../infer_model/ --device gpu
```

运行完成后返回的结果如下：

```bash
Batch id: 0, example id: 0, sentence: against shimmering cinematography that lends the setting the ethereal beauty of an asian landscape painting, label: positive, negative prob: 0.4623, positive prob: 0.5377.
Batch id: 0, example id: 1, sentence: the situation in a well-balanced fashion, label: positive, negative prob: 0.3500, positive prob: 0.6500.
Batch id: 1, example id: 0, sentence: at achieving the modest , crowd-pleasing goals it sets for itself, label: positive, negative prob: 0.4530, positive prob: 0.5470.
Batch id: 1, example id: 1, sentence: so pat it makes your teeth hurt, label: positive, negative prob: 0.3816, positive prob: 0.6184.
Batch id: 2, example id: 0, sentence: this new jangle of noise , mayhem and stupidity must be a serious contender for the title ., label: positive, negative prob: 0.3650, positive prob: 0.6350.
```

## 参数说明

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定部署模型的目录， |
|--batch_size |输入的 batch size，默认为 2|
|--max_length |最大序列长度，默认为 128|
|--device | 运行的设备，可选范围: ['cpu', 'gpu']，默认为'cpu' |
|--device_id | 运行设备的 id。默认为0。 |
|--cpu_threads | 当使用 cpu 推理时，指定推理的 cpu 线程数，默认为4。|
