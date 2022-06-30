# Ernie_doc 在iflytek数据集上的使用

## 简介

本示例将使用ERNIE-DOC模型，演示如何在长文本数据集上（e.g. iflytek）完成分类任务的训练，预测以及动转静过程。以下是本例的简要目录结构及说明:

```shell
.
├── LICENSE
├── README.md             #文档
├── data.py               #数据处理
├── export_model.py       #将动态图参数导出成静态图参数
├── metrics.py            #ERNIE-Doc下游任务指标
├── modeling.py           #ERNIE-Doc模型实现（针对实现静态图修改）
├── predict.py            #分类任务预测脚本（包括动态图预测和动转静）
└── train.py              #分类任务训练脚本（包括数据下载，模型导出和测试集结果导出）
```

## 快速开始

### 通用参数释义

除[ERNIE_DOC](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/language_model/ernie-doc/run_classifier.py)
展示的通用参数之外，本例还有如下参数：

- `static_mode` 在 `predict.py` 表示是否使用静态图进行预测。
- `test_results_file` 在`train.py`和`predict.py`中表示测试集预测结果所存储的地址，默认为`./test_restuls.json`。
- `static_path` 在`export_model.py`和`predict.py`中表示要将转化完成的静态图存储的地址，如果改地址已经有静态图模型参数，`predict.py`
  会直接读取该模型参数，而`export_model.py`会覆盖掉该模型参数。默认路径为`{HOME}/.paddlenlp/static/inference`。

### 分类任务训练

iflytek的数据示例如下：

```shell
{"label": "110", "label_des": "社区超市", "sentence": "朴朴快送超市创立于2016年，专注于打造移动端30分钟即时配送一站式购物平台，商品品类包含水果、蔬菜、肉禽蛋奶、海鲜水产、粮油调味、酒水饮料、休闲食品、日用品、外卖等。朴朴公司希望能以全新的商业模式，更高效快捷的仓储配送模式，致力于成为更快、更好、更多、更省的在线零售平台，带给消费者更好的消费体验，同时推动中国食品安全进程，成为一家让社会尊敬的互联网公司。,朴朴一下，又好又快,1.配送时间提示更加清晰友好2.保障用户隐私的一些优化3.其他提高使用体验的调整4.修复了一些已知bug"}
```

该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别。 使用训练脚本

```shell
python train.py --batch_size 4 \
                --model_name_or_path ernie-doc-base-zh \
                --epoch 5 \
                --output_dir ./checkpoints/
```

根据通用参数释义可自行更改训练超参数和模型保存地址。

### 模型导出和预测

可以使用模型导出脚本将动态图模型转化成静态图：

```shell
python export_model.py --batch_size 16 \
                       --model_name_or_path finetuned_model \
                       --max_seq_lenght 512 \
                       --memory_length 128 \
                       --static_path ./my_static_model/
```

也可以直接使用预测脚本将`static_mode`设为True （设置成False则使用动态图预测），直接完成转化静态图和使用静态图预测的步骤：

```shell
python predict.py --static_mode True \
        --dataset iflytek \
        --batch_size 16 \
        --model_name_or_path finetuned_model \
        --max_seq_lenght 512 \
        --memory_length 128 \
        --static_path ./my_static_model/ \
        --test_results_file ./test_results.json
```

模型输出的`test_results_file`示例：

```shell
{"id": "2590", "label": "70"}
{"id": "2591", "label": "91"}
{"id": "2592", "label": "20"}
{"id": "2593", "label": "28"}
{"id": "2594", "label": "95"}
{"id": "2595", "label": "116"}
{"id": "2596", "label": "59"}
{"id": "2597", "label": "22"}
```
