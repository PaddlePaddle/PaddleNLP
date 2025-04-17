# Text information extraction

**Table of contents**
- [1. Text Information Extraction Application](#1)
- [2. Quick Start](#2)
   - [2.1 Code Structure](#21)
   - [2.2 Data Annotation](#22)
   - [2.3 Finetuning](#23)
   - [2.4 Evaluation](#24)
   - [2.5 Inference](#25)
   - [2.6 Experiments](#26)
   - [2.7 Closed Domain Distillation](#27)

<a name="1"></a>

## 1. Text Information Extraction Application

This project provides an end-to-end application solution for plain text extraction based on UIE fine-tuning and goes through the full lifecycle of **data labeling, model training and model deployment**. We hope this guide can help you apply Information Extraction techniques in your own products or models.a

Information Extraction (IE) is the process of extracting structured information from given input data such as text, pictures or scanned document. While IE brings immense value, applying IE techniques is never easy with challenges such as domain adaptation, heterogeneous structures, lack of labeled data, etc. This PaddleNLP Information Extraction Guide builds on the foundation of our work in [Universal Information Extraction](https://arxiv.org/abs/2203.12277) and provides an industrial-level solution that not only supports **extracting entities, relations, events and opinions from plain text**, but also supports **cross-modal extraction out of documents, tables and pictures.** Our method features a flexible prompt, which allows you to specify extraction targets with simple natural language. We also provide a few different domain-adapated models specialized for different industry sectors.

**Highlights:**
- **Comprehensive Coverage🎓:** Covers various mainstream tasks of information extraction for plain text and document scenarios, supports multiple languages
- **State-of-the-Art Performance🏃:** Strong performance from the UIE model series models in plain text and multimodal datasets. We also provide pretrained models of various sizes to meet different needs
- **Easy to use⚡:** three lines of code to use our `Taskflow` for out-of-box Information Extraction capabilities. One line of command to model training and model deployment
- **Efficient Tuning✊:** Developers can easily get started with the data labeling and model training process without a background in Machine Learning.

<a name="2"></a>

## 2. Quick start

For quick start, you can directly use ```paddlenlp.Taskflow``` out-of-the-box, leveraging the zero-shot performance. For production use cases, we recommend labeling a small amount of data for model fine-tuning to further improve the performance.

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

We recommend using [Label Studio](https://labelstud.io/) for data labeling. We provide an end-to-end pipeline for the labeling -> training process. You can export the labeled data in Label Studio through [label_studio.py](../label_studio.py) script to export and convert the data into the required input form for the model. For a detailed introduction to labeling methods, please refer to [Label Studio Data Labeling Guide](../label_studio_text_en.md).

Here we provide a pre-labeled example dataset `Military Relationship Extraction Dataset`, which you can download with the following command. We will show how to use the data conversion script to generate training/validation/test set files for fine-tuning .

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

For more labeling rules and parameter descriptions for different types of tasks (including entity extraction, relationship extraction, document classification, etc.), please refer to [Label Studio Data Labeling Guide](../label_studio_text_en.md).


<a name="23"></a>

### 2.3 Finetuning

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

* `device`: Training device, one of 'cpu', 'gpu' and 'npu' can be selected; the default is GPU training.
* `logging_steps`: The interval steps of log printing during training, the default is 10.
* `save_steps`: The number of interval steps to save the model checkpoint during training, the default is 100.
* `eval_steps`: The number of interval steps to save the model checkpoint during training, the default is 100.
* `seed`: global random seed, default is 42.
* `model_name_or_path`: The pre-trained model used for few shot training. Defaults to "uie-x-base".
* `output_dir`: required, the model directory saved after model training or compression; the default is `None`.
* `train_path`: training set path; defaults to `None`.
* `dev_path`: Development set path; defaults to `None`.
* `max_seq_len`: The maximum segmentation length of the text. When the input exceeds the maximum length, the input text will be automatically segmented. The default is 512.
* `per_device_train_batch_size`: The batch size of each GPU core//NPU core/CPU used for training, the default is 8.
* `per_device_eval_batch_size`: Batch size per GPU core/NPU core/CPU for evaluation, default is 8.
* `num_train_epochs`: Training rounds, 100 can be selected when using early stopping method; the default is 10.
* `learning_rate`: The maximum learning rate for training, UIE-X recommends setting it to 1e-5; the default value is 3e-5.
* `label_names`: the name of the training data label, UIE-X is set to 'start_positions' 'end_positions'; the default value is None.
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

### 2.4 Evaluation

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

We adopt the single-stage method for evaluation, which means tasks that require multiple stages (e.g. relation extraction, event extraction) are evaluated separately for each stage. By default, the validation/test set uses all labels at the same level to construct the negative examples.

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

- `device`: Evaluation device, one of 'cpu', 'gpu' and 'npu' can be selected; the default is GPU evaluation.
- `model_path`: The path of the model folder for evaluation, which must contain the model weight file `model_state.pdparams` and the configuration file `model_config.json`.
- `test_path`: The test set file for evaluation.
- `batch_size`: batch size, please adjust according to the machine situation, the default is 16.
- `max_seq_len`: The maximum segmentation length of the text. When the input exceeds the maximum length, the input text will be automatically segmented. The default is 512.
- `debug`: Whether to enable the debug mode to evaluate each positive category separately. This mode is only used for model debugging and is disabled by default.
- `multilingual`: Whether it is a multilingual model, it is turned off by default.
- `schema_lang`: select the language of the schema, optional `ch` and `en`. The default is `ch`, please select `en` for the English dataset.

<a name="25"></a>

### 2.5 Inference
Same with the pretrained models, you can use `paddlenlp.Taskflow` to load your custom model by specifying the path of the model weight file through `task_path`

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

Some industrial application scenarios have high inference performance requirements and the model cannot go into production without being effectively compressed. We built the [UIE Slim Data Distillation](./data_distill/README_en.md) with knowledge distillation techniques. The principle is to use the data as a bridge to transfer the knowledge of the UIE model to the smaller closed-domain information extraction model in order to achieve speedup inference significantly with minimal loss to accuracy.
