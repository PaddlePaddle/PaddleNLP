# 阅读理解 SQuAD

## 简介

### 任务说明
本文主要介绍基于Bert预训练模型的SQuAD（Stanford Question Answering Dataset）数据集的阅读理解任务，给定一篇文章和一个问题，计算答案在文章中的起始位置和结束位置。对于SQuAD2.0数据集，还可以返回答案在文章中不存在的概率。

### 数据集

此任务的数据集包括以下数据集：

SQuAD v1.1
- [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

SQuAD v2.0
- [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
- [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)


## 快速开始

### 数据准备

为了方便开发者进行测试，我们使用了HuggingFace的数据集，用户可以通过命令行传入`--version_2_with_negative`控制所需要的SQuAD数据集版本。

### Fine-tune

对于 SQuAD v1.1,按如下方式启动 Fine-tuning:

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --device gpu \
    --do_train \
    --do_predict
 ```

* `model_type`: 预训练模型的种类。如bert，ernie，roberta等。
* `model_name_or_path`: 预训练模型的具体名称。如bert-base-uncased，bert-large-cased等。或者是模型文件的本地路径。
* `output_dir`: 保存模型checkpoint的路径。
* `do_train`: 是否进行训练。
* `do_predict`: 是否进行预测。

训练结束后模型会自动对结果进行评估，得到类似如下的输出：

```text
{
  "exact": 81.18259224219489,
  "f1": 88.68817481234801,
  "total": 10570,
  "HasAns_exact": 81.18259224219489,
  "HasAns_f1": 88.68817481234801,
  "HasAns_total": 10570
}
```

对于 SQuAD v2.0,按如下方式启动 Fine-tuning:

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --device gpu \
    --do_train \
    --do_predict \
    --version_2_with_negative
 ```

* `version_2_with_negative`: 使用squad2.0数据集和评价指标的标志。

训练结束后会在模型会自动对结果进行评估，得到类似如下的输出：

```text
{
  "exact": 73.25865408910974,
  "f1": 76.63096554166046,
  "total": 11873,
  "HasAns_exact": 73.22874493927125,
  "HasAns_f1": 79.98303877802545,
  "HasAns_total": 5928,
  "NoAns_exact": 73.28847771236333,
  "NoAns_f1": 73.28847771236333,
  "NoAns_total": 5945,
  "best_exact": 74.31988545439232,
  "best_exact_thresh": -2.5820093154907227,
  "best_f1": 77.20521797731851,
  "best_f1_thresh": -1.559523582458496
}
```

其中会输出 `best_f1_thresh` 是最佳阈值，可以使用这个阈值重新训练，或者从 `all_nbest_json`变量中获取最终 `prediction`。
训练方法与前面大体相同，只需要设定 `--null_score_diff_threshold` 参数的值为测评时输出的 `best_f1_thresh` ，通常这个值在 -1.0 到 -5.0 之间。

**NOTE:** 如需恢复模型训练，则model_name_or_path只需指定到文件夹名即可。如`--model_name_or_path=./tmp/squad/model_19000/`，程序会自动加载模型参数`/model_state.pdparams`，也会自动加载词表，模型config和tokenizer的config。

### 预测

如需使用训练好的模型预测并输出结果，需将自己的数据集改成SQuAD格式（以下示例为SQuAD2.0）。

```text
{"data": [{'title': 'Beyoncé',
 'paragraphs': [
                 {'qas': [{'question': 'When did Beyonce start becoming popular?',
                         'id': '56be85543aeaaa14008c9063',
                         'answers': [],
                      'is_impossible': False}]],
                             'context':'Beyoncé Giselle Knowles-Carter(biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she.'}
     }]
```

并参考[以内置数据集格式读取本地数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_load.html#id4)中的方法创建自己的数据集并修改`run_squad.py`中对应的数据集读取代码。再运行以下脚本：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_squad.py \
    --model_type bert \
    --model_name_or_path your-best-model \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --device gpu \
    --do_predict \
    --version_2_with_negative
 ```

即可完成预测，预测的答案保存在`prediction.json`中。数据格式如下所示，左边的id与输入中的id对应。

```text
{
    "56be85543aeaaa14008c9063": "in the late 1990s",
    ...
}
```

### 静态图预测

在Fine-tune完成后，我们可以使用如下方式导出希望用来预测的模型：

```shell
python -u ./export_model.py \
    --model_type bert \
    --model_path bert-base-uncased \
    --output_path ./infer_model/model
```

其中参数释义如下：
- `model_type` 指示了模型类型，使用BERT模型时设置为bert即可。
- `model_path` 表示训练模型的保存路径，与训练时的`output_dir`一致。
- `output_path` 表示导出预测模型文件的前缀。保存时会添加后缀（`pdiparams`，`pdiparams.info`，`pdmodel`）；除此之外，还会在`output_path`包含的目录下保存tokenizer相关内容。

然后按照如下的方式对阅读理解任务进行预测：

```shell
python -u deploy/python/predict.py \
    --model_type bert \
    --model_name_or_path ./infer_model/model \
    --batch_size 4 \
    --max_seq_length 384
```

其中参数释义如下：
- `model_type` 指示了模型类型，使用BERT模型时设置为bert即可。
- `model_name_or_path` 表示预测模型文件的前缀，和上一步导出预测模型中的`output_path`一致。
- `batch_size` 表示每个预测批次的样本数目。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断，和训练时一致。

以上命令将在SQuAD v1.1的验证集上进行预测。此外，同训练时一样，用户可以通过命令行传入`--version_2_with_negative`控制所需要的SQuAD数据集版本。
