# Text information extraction

**Table of contents**
- [1. Text Information Extraction Application](#1)
- [2. Quick Start](#2)
   - [2.1 Code Structure](#21)
   - [2.2 Data Annotation](#22)
   - [2.3 Model Fine-tuning](#23)
   - [2.4 Model Evaluation](#24)
   - [2.5 Model Prediction](#25)
   - [2.6 Experiments](#26)
   - [2.7 Closed Domain Distillation](#27)

<a name="1"></a>

## 1. Text Information Extraction Application

This project provides an end-to-end application solution for plain text extraction based on UIE fine-tuning, and through the whole process of **data labeling-model training-model tuning-prediction deployment**, it can quickly realize the landing of document information extraction products.

In layman's terms, information extraction is the process of extracting structured information from given input data such as text/pictures. In the process of implementing information extraction, it usually faces many challenges such as changing fields, diverse tasks, and scarce data. Aiming at the difficulties and pain points in the field of information extraction, PaddleNLP information extraction applies the idea of UIE unified modeling, provides an industrial-level application solution for document information extraction, and supports entities, relationships, events, and opinions in document/picture/table and plain text scenarios and other task information extraction**. The application **does not limit the industry field and extraction target**, and can realize the seamless connection from the product prototype development, business POC stage to business landing and iteration stages, helping developers to achieve rapid adaptation and landing of extraction scenarios in specific fields.

**Application Highlights of Text Information Extraction:**

- **Comprehensive coverage of scenarios🎓:** Covers all kinds of mainstream tasks of text information extraction, supports multiple languages, and meets the needs of developers for various information extraction.
- **Leading effect🏃:** Using the UIE series models with outstanding effects in plain text as the training base, it provides pre-trained models of various sizes to meet different needs, and has extensive and mature practical applicability.
- **Easy to use⚡:** Implementing three lines of code through Taskflow can realize quick calls without labeled data, and one line of commands can start information extraction training, easily complete deployment and go online, and lower the threshold for information extraction technology.
- **Efficient Tuning✊:** Developers can easily get started with the data labeling and model training process without any background knowledge of machine learning.

<a name="2"></a>

## 2. Quick start

For simple extraction targets, you can directly use ```paddlenlp.Taskflow``` to achieve zero-shot extraction. For subdivision scenarios, we recommend using custom functions (labeling a small amount of data for model fine-tuning) to further improve the effect.

<a name="21"></a>

### 2.1 Code structure

```shell
.
├── utils.py # data processing tools
├── finetune.py # model fine-tuning, compression script
├── evaluate.py # model evaluation script
└── README.md
```

<a name="22"></a>

### 2.2 Data labeling

We recommend using [Label Studio](https://labelstud.io/) for text information extraction and data labeling. This project has opened up the channel from data labeling to training, that is, Label Studio can export data through [label_studio.py]( ../label_studio.py) script to easily convert the data into the form required for input into the model, achieving a seamless transition. For a detailed introduction to labeling methods, please refer to [Label Studio Data Labeling Guide](../label_studio_text.md).

Here we provide a pre-labeled `Military Relationship Extraction Dataset` file, you can run the following command line to download the dataset, we will show how to use the data conversion script to generate training/validation/test set files, and use the UIE model for fine-tuning .

Download the military relationship extraction dataset:

```shell
wget https://bj.bcebos.com/paddlenlp/datasets/military.tar.gz
tar -xvf military.tar.gz
mv military data
rm military.tar.gz
```

Generate training/validation set files:
```shell
python ../label_studio.py \
     --label_studio_file ./data/label_studio.json \
     --save_dir ./data \
     --splits 0.76 0.24 0 \
     --negative_ratio 3 \
     --task_type ext
```

For more labeling rules and parameter descriptions for different types of tasks (including entity extraction, relationship extraction, document classification, etc.), please refer to [Label Studio Data Labeling Guide](../label_studio.md).


<a name="23"></a>

### 2.3 Model fine-tuning

It is recommended to use [Trainer API ](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md) to fine-tune the model. Just input the model, data set, etc., and you can use the Trainer API to efficiently and quickly perform tasks such as pre-training, fine-tuning, and model compression. You can start multi-card training, mixed-precision training, gradient accumulation, breakpoint restart, log display, and other functions with one click. , the Trainer API also encapsulates the general training configuration of the training process, such as: optimizer, learning rate scheduling, etc.

Use the following command to fine-tune the model using `uie-base` as the pre-trained model, and save the fine-tuned model to `$finetuned_model`:

Single GPU:

```shell
python finetune.py  \
    --device gpu \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 1000 \
    --model_name_or_path uie-base \
    --output_dir ./checkpoint/model_best \
    --train_path data/train.txt \
    --dev_path data/dev.txt  \
    --max_seq_len 512  \
    --per_device_train_batch_size  16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir ./checkpoint/model_best \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1
```

Multiple GPUs：

```shell
python -u -m paddle.distributed.launch --gpus "0,1" finetune.py \
    --device gpu \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 1000 \
    --model_name_or_path uie-base \
    --output_dir ./checkpoint/model_best \
    --train_path data/train.txt \
    --dev_path data/dev.txt  \
    --max_seq_len 512  \
    --per_device_train_batch_size  8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir ./checkpoint/model_best \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1
```

Parameters:

* `device`: Training device, one of 'cpu' and 'gpu' can be selected; the default is GPU training.
* `logging_steps`: The interval steps of log printing during training, the default is 10.
* `save_steps`: The number of interval steps to save the model checkpoint during training, the default is 100.
* `eval_steps`: The number of interval steps to save the model checkpoint during training, the default is 100.
* `seed`: global random seed, default is 42.
* `model_name_or_path`: The pre-trained model used for few shot training. Defaults to "uie-x-base".
* `output_dir`: required, the model directory saved after model training or compression; the default is `None`.
* `train_path`: training set path; defaults to `None`.
* `dev_path`: Development set path; defaults to `None`.
* `max_seq_len`: The maximum segmentation length of the text. When the input exceeds the maximum length, the input text will be automatically segmented. The default is 512.
* `per_device_train_batch_size`: The batch size of each GPU core/CPU used for training, the default is 8.
* `per_device_eval_batch_size`: Batch size per GPU core/CPU for evaluation, default is 8.
* `num_train_epochs`: Training rounds, 100 can be selected when using early stopping method; the default is 10.
* `learning_rate`: The maximum learning rate for training, UIE-X recommends setting it to 1e-5; the default value is 3e-5.
* `label_names`: the name of the training data label label, UIE-X is set to 'start_positions' 'end_positions'; the default value is None.
* `do_train`: Whether to perform fine-tuning training, setting this parameter means to perform fine-tuning training, and it is not set by default.
* `do_eval`: Whether to evaluate, setting this parameter means to evaluate, the default is not set.
* `do_export`: Whether to export, setting this parameter means to export static images, and it is not set by default.
* `export_model_dir`: Static map export address, the default is None.
* `overwrite_output_dir`: If `True`, overwrite the contents of the output directory. If `output_dir` points to a checkpoint directory, use it to continue training.
* `disable_tqdm`: Whether to use tqdm progress bar.
* `metric_for_best_model`: Optimal model metric, UIE-X recommends setting it to `eval_f1`, the default is None.
* `load_best_model_at_end`: Whether to load the best model after training, usually used in conjunction with `metric_for_best_model`, the default is False.
* `save_total_limit`: If this parameter is set, the total number of checkpoints will be limited. Remove old checkpoints `output directory`, defaults to None.

<a name="24"></a>

### 2.4 Model Evaluation

Model evaluation:

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --batch_size 16 \
    --max_seq_len 512
```

Model evaluation for UIE-M：

```
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --batch_size 16 \
    --max_seq_len 512 \
    --multilingual
```

Description of the evaluation method: The single-stage evaluation method is adopted, that is, tasks that require staged prediction such as relationship extraction and event extraction are evaluated separately for the prediction results of each stage. By default, the validation/test set uses all labels at the same level to construct all negative examples.

The `debug` mode can be turned on to evaluate each positive category separately. This mode is only used for model debugging:

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --debug
```

Output print example:

```text
[2022-11-21 12:48:41,794] [    INFO] - -----------------------------
[2022-11-21 12:48:41,795] [    INFO] - Class Name: 武器名称
[2022-11-21 12:48:41,795] [    INFO] - Evaluation Precision: 0.96667 | Recall: 0.96667 | F1: 0.96667
[2022-11-21 12:48:44,093] [    INFO] - -----------------------------
[2022-11-21 12:48:44,094] [    INFO] - Class Name: X的产国
[2022-11-21 12:48:44,094] [    INFO] - Evaluation Precision: 1.00000 | Recall: 0.99275 | F1: 0.99636
[2022-11-21 12:48:46,474] [    INFO] - -----------------------------
[2022-11-21 12:48:46,475] [    INFO] - Class Name: X的研发单位
[2022-11-21 12:48:46,475] [    INFO] - Evaluation Precision: 0.77519 | Recall: 0.64935 | F1: 0.70671
[2022-11-21 12:48:48,800] [    INFO] - -----------------------------
[2022-11-21 12:48:48,801] [    INFO] - Class Name: X的类型
[2022-11-21 12:48:48,801] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
```

Parameters:

- `model_path`: The path of the model folder for evaluation, which must contain the model weight file `model_state.pdparams` and the configuration file `model_config.json`.
- `test_path`: The test set file for evaluation.
- `batch_size`: batch size, please adjust according to the machine situation, the default is 16.
- `max_seq_len`: The maximum segmentation length of the text. When the input exceeds the maximum length, the input text will be automatically segmented. The default is 512.
- `debug`: Whether to enable the debug mode to evaluate each positive category separately. This mode is only used for model debugging and is disabled by default.
- `multilingual`: Whether it is a multilingual model, it is turned off by default.
- `schema_lang`: select the language of the schema, optional `ch` and `en`. The default is `ch`, please select `en` for the English dataset.

<a name="25"></a>

### 2.5 Model Prediction

`paddlenlp.Taskflow` loads the custom model, and specifies the path of the model weight file through `task_path`, which must contain the trained model weight file `model_state.pdparams`.

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow

>>> schema = {"武器名称": ["产国", "类型", "研发单位"]}
# Set the extraction target and the fine-tuned model path
>>> my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best')
>>> pprint(my_ie("威尔哥（Virgo）减速炸弹是由瑞典FFV军械公司专门为瑞典皇家空军的攻击机实施低空高速轰炸而研制，1956年开始研制，1963年进入服役，装备于A32“矛盾”、A35“龙”、和AJ134“雷”攻击机，主要用于攻击登陆艇、停放的飞机、高炮、野战火炮、轻型防护装甲车辆以及有生力量。"))
[{'武器名称': [{'end': 14,
            'probability': 0.9998632702221926,
            'relations': {'产国': [{'end': 18,
                                  'probability': 0.9998815094394331,
                                  'start': 16,
                                  'text': '瑞典'}],
                          '研发单位': [{'end': 25,
                                    'probability': 0.9995875123178521,
                                    'start': 18,
                                    'text': 'FFV军械公司'}],
                          '类型': [{'end': 14,
                                  'probability': 0.999877336059086,
                                  'start': 12,
                                  'text': '炸弹'}]},
            'start': 0,
            'text': '威尔哥（Virgo）减速炸弹'}]}]
```

<a name="26"></a>

### 2.6 Experiments

  |  |  Precision  | Recall | F1 Score |
  | :---: | :--------: | :--------: | :--------: |
  | 0-shot | 0.64634| 0.53535 | 0.58564 |
  | 5-shot | 0.89474 | 0.85000 | 0.87179 |
  | 10-shot | 0.92793 | 0.85833 | 0.89177 |
  | full-set | 0.93103 | 0.90000 | 0.91525 |


<a name="27"></a>

### 2.7 Closed Domain Distillation

In some industrial application scenarios, the performance requirements are high, and the model cannot be practically applied if it cannot be effectively compressed. Therefore, we built the UIE Slim data distillation system based on the data distillation technology. The principle is to use the data as a bridge to transfer the knowledge of the UIE model to the small closed-domain information extraction model, so as to achieve the effect of greatly improving the prediction speed with a small loss of accuracy. For a detailed introduction, please refer to [UIE Slim Data Distillation](./data_distill/README.md)
