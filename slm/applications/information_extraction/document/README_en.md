# document information extraction

**Table of contents**
- [1. Introduction](#1)
- [2. Quick Start](#2)
   - [2.1 Code Structure](#21)
   - [2.2 Data Annotation](#22)
   - [2.3 Finetuning](#23)
   - [2.4 Evaluation](#24)
   - [2.5 Inference](#25)
   - [2.6 Experiments](#26)

<a name="1"></a>

## 1. Introduction

This Information Extraction (IE) guide introduces our open-source industry-grade solution that covers the most widely-used application scenarios of Information Extraction. It features **multi-domain, multi-task, and cross-modal capabilities** and goes through the full lifecycle of **data labeling, model training and model deployment**. We hope this guide can help you apply Information Extraction techniques in your own products or models.

Information Extraction (IE) is the process of extracting structured information from given input data such as text, pictures or scanned document. While IE brings immense value, applying IE techniques is never easy with challenges such as domain adaptation, heterogeneous structures, lack of labeled data, etc. This PaddleNLP Information Extraction Guide builds on the foundation of our work in [Universal Information Extraction](https://arxiv.org/abs/2203.12277) and provides an industrial-level solution that not only supports **extracting entities, relations, events and opinions from plain text**, but also supports **cross-modal extraction out of documents, tables and pictures.** Our method features a flexible prompt, which allows you to specify extraction targets with simple natural language. We also provide a few different domain-adapted models specialized for different industry sectors.

**Highlights:**

- **Comprehensive Coverage🎓:** Covers various mainstream tasks of information extraction for plain text and document scenarios, supports multiple languages
- **State-of-the-Art Performance🏃:** Strong performance from the UIE model series models in plain text and multimodal datasets. We also provide pretrained models of various sizes to meet different needs
- **Easy to use⚡:** three lines of code to use our `Taskflow` for out-of-box Information Extraction capabilities. One line of command to model training and model deployment
- **Efficient Tuning✊:** Developers can easily get started with the data labeling and model training process without a background in Machine Learning.

<a name="2"></a>

## 2. Quick Start

For quick start, you can directly use ```paddlenlp.Taskflow``` out-of-the-box, leveraging the zero-shot capability. For production use cases, we recommend labeling a small amount of data for model fine-tuning to further improve the performance.

<a name="21"></a>

### 2.1 Code Structure

```shell
.
├── utils.py # data processing tools
├── finetune.py # model fine-tuning, compression script
├── evaluate.py # model evaluation script
└── README.md
```

<a name="22"></a>

### 2.2 Data Annotation

We recommend using [Label Studio](https://labelstud.io/) for data labeling. We provide an end-to-end pipeline for the labeling -> training process. You can export the labeled data in Label Studio through [label_studio.py](../label_studio.py) script to export and convert the data into the required input form for the model. For a detailed introduction to labeling methods, please refer to [Label Studio Data Labeling Guide](../label_studio_doc_en.md).

Here we provide the pre-labeled example dataset `VAT invoice dataset`, which you can download by running the following command. We will demonstrate how to use the data conversion script to generate training/validation/test set files for finetuning.

Download the VAT invoice dataset:
```shell
wget https://paddlenlp.bj.bcebos.com/datasets/tax.tar.gz
tar -zxvf tax.tar.gz
mv tax data
rm tax.tar.gz
```

Generate training/validation data files:

```shell
python ../label_studio.py \
     --label_studio_file ./data/label_studio.json \
     --save_dir ./data \
     --splits 0.8 0.2 0 \
     --task_type ext
```

Generate training/validation set files, you can use PP-Structure's layout analysis to optimize the sorting of OCR results:

```shell
python ../label_studio.py \
     --label_studio_file ./data/label_studio.json \
     --save_dir ./data \
     --splits 0.8 0.2 0\
     --task_type ext\
     --layout_analysis True
```

For more labeling rules and parameter descriptions for different types of tasks (including entity extraction, relationship extraction, document classification, etc.), please refer to [Label Studio Data Labeling Guide](../label_studio_doc_en.md).

<a name="23"></a>

### 2.3 Finetuning

Use the following command to fine-tune the model using `uie-x-base` as the pre-trained model, and save the fine-tuned model to `./checkpoint/model_best`:

Single GPU:

```shell
python finetune.py\
     --device gpu \
     --logging_steps 5 \
     --save_steps 25 \
     --eval_steps 25 \
     --seed 42 \
     --model_name_or_path uie-x-base \
     --output_dir ./checkpoint/model_best\
     --train_path data/train.txt \
     --dev_path data/dev.txt \
     --max_seq_len 512 \
     --per_device_train_batch_size 8 \
     --per_device_eval_batch_size 8 \
     --num_train_epochs 10 \
     --learning_rate 1e-5 \
     --do_train \
     --do_eval \
     --do_export \
     --export_model_dir ./checkpoint/model_best\
     --overwrite_output_dir \
     --disable_tqdm True \
     --metric_for_best_model eval_f1 \
     --load_best_model_at_end True \
     --save_total_limit 1
```

Multiple GPUs:

```shell
python -u -m paddle.distributed.launch --gpus "0" finetune.py \
     --device gpu \
     --logging_steps 5 \
     --save_steps 25 \
     --eval_steps 25 \
     --seed 42 \
     --model_name_or_path uie-x-base \
     --output_dir ./checkpoint/model_best\
     --train_path data/train.txt \
     --dev_path data/dev.txt \
     --max_seq_len 512 \
     --per_device_train_batch_size 8 \
     --per_device_eval_batch_size 8 \
     --num_train_epochs 10 \
     --learning_rate 1e-5 \
     --do_train \
     --do_eval \
     --do_export \
     --export_model_dir ./checkpoint/model_best\
     --overwrite_output_dir \
     --disable_tqdm True \
     --metric_for_best_model eval_f1 \
     --load_best_model_at_end True \
     --save_total_limit 1
```

Since the parameter `--do_eval` is set in the sample code, it will be automatically evaluated after training.

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
* `per_device_train_batch_size`: The batch size of each GPU core/NPU core/CPU used for training, the default is 8.
* `per_device_eval_batch_size`: Batch size per GPU core/NPU core/CPU for evaluation, default is 8.
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

### 2.4 Evaluation

```shell
python evaluate.py \
    --device "gpu" \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --output_dir ./checkpoint/model_best \
    --label_names 'start_positions' 'end_positions'\
    --max_seq_len 512 \
    --per_device_eval_batch_size 16
```
We adopt the single-stage method for evaluation, which means tasks that require multiple stages (e.g. relation extraction, event extraction) are evaluated separately for each stage. By default, the validation/test set uses all labels at the same level to construct the negative examples.
The `debug` mode can be turned on to evaluate each positive category separately. This mode is only used for model debugging:

```shell
python evaluate.py \
    --device "gpu" \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --output_dir ./checkpoint/model_best \
    --label_names 'start_positions' 'end_positions' \
    --max_seq_len 512 \
    --per_device_eval_batch_size 16 \
    --debug True
```

Output result:

```text
[2022-11-14 09:41:18,424] [    INFO] - ***** Running Evaluation *****
[2022-11-14 09:41:18,424] [    INFO] -   Num examples = 160
[2022-11-14 09:41:18,424] [    INFO] -   Pre device batch size = 4
[2022-11-14 09:41:18,424] [    INFO] -   Total Batch size = 4
[2022-11-14 09:41:18,424] [    INFO] -   Total prediction steps = 40
[2022-11-14 09:41:26,451] [    INFO] - -----Evaluate model-------
[2022-11-14 09:41:26,451] [    INFO] - Class Name: ALL CLASSES
[2022-11-14 09:41:26,451] [    INFO] - Evaluation Precision: 0.94521 | Recall: 0.88462 | F1: 0.91391
[2022-11-14 09:41:26,451] [    INFO] - -----------------------------
[2022-11-14 09:41:26,452] [    INFO] - ***** Running Evaluation *****
[2022-11-14 09:41:26,452] [    INFO] -   Num examples = 8
[2022-11-14 09:41:26,452] [    INFO] -   Pre device batch size = 4
[2022-11-14 09:41:26,452] [    INFO] -   Total Batch size = 4
[2022-11-14 09:41:26,452] [    INFO] -   Total prediction steps = 2
[2022-11-14 09:41:26,692] [    INFO] - Class Name: 开票日期
[2022-11-14 09:41:26,692] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-11-14 09:41:26,692] [    INFO] - -----------------------------
[2022-11-14 09:41:26,693] [    INFO] - ***** Running Evaluation *****
[2022-11-14 09:41:26,693] [    INFO] -   Num examples = 8
[2022-11-14 09:41:26,693] [    INFO] -   Pre device batch size = 4
[2022-11-14 09:41:26,693] [    INFO] -   Total Batch size = 4
[2022-11-14 09:41:26,693] [    INFO] -   Total prediction steps = 2
[2022-11-14 09:41:26,952] [    INFO] - Class Name: 名称
[2022-11-14 09:41:26,952] [    INFO] - Evaluation Precision: 0.87500 | Recall: 0.87500 | F1: 0.87500
[2022-11-14 09:41:26,952] [    INFO] - -----------------------------
...
```

Parameters:

* `device`: Evaluation device, one of 'cpu', 'gpu' and 'npu' can be selected; the default is GPU evaluation.
* `model_path`: The path of the model folder for evaluation, which must contain the model weight file `model_state.pdparams` and the configuration file `model_config.json`.
* `test_path`: The test set file for evaluation.
* `label_names`: the name of the training data label, UIE-X is set to 'start_positions' 'end_positions'; the default value is None.
* `batch_size`: batch size, please adjust according to the machine situation, the default is 16.
* `max_seq_len`: The maximum segmentation length of the text. When the input exceeds the maximum length, the input text will be automatically segmented. The default is 512.
* `per_device_eval_batch_size`: Batch size per GPU core/NPU core/CPU for evaluation, default is 8.
* `debug`: Whether to enable the debug mode to evaluate each positive category separately. This mode is only used for model debugging and is disabled by default.
* `schema_lang`: Select the language of the schema, optional `ch` and `en`. The default is `ch`, please select `en` for the English dataset.

<a name="25"></a>

### 2.5 Inference

Same with the pretrained models, you can use `paddlenlp.Taskflow` to load your custom model by specifying the path of the model weight file through `task_path`

```python
from pprint import pprint
from paddlenlp import Taskflow
from paddlenlp.utils.doc_parser import DocParser

schema = ['开票日期', '名称', '纳税人识别号', '开户行及账号', '金额', '价税合计', 'No', '税率', '地址、电话', '税额']
my_ie = Taskflow("information_extraction", model="uie-x-base", schema=schema, task_path='./checkpoint/model_best', precision='fp16')
```

We specify the extraction targets by setting `schema` and visualize the information of the specified `doc_path` document:

```python
doc_path = "./data/images/b199.jpg"
results = my_ie({"doc": doc_path})
pprint(results)

# Result visualization
DocParser.write_image_with_results(
    doc_path,
    result=results[0],
    save_path="./image_show.png")
```

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/206084942-44ba477c-9244-4ce2-bbb5-ba430c9b926e.png height=550 width=700 />
</div>

<a name="26"></a>

### 2.6 Experiments

  |  |  Precision  | Recall | F1 Score |
  | :---: | :--------: | :--------: | :--------: |
  | 0-shot| 0.44898 | 0.56410  | 0.50000 |
  | 5-shot| 0.9000 | 0.9231 | 0.9114 |
  | 10-shot| 0.9125 | 0.93590 |  0.9241 |
  | 20-shot| 0.9737 | 0.9487 | 0.9610 |
  | 30-shot|  0.9744  | 0.9744  | 0.9744 |
  | 30-shot+PP-Structure| 1.0  | 0.9625 |  0.9809 |


n-shot means that the training set contains n labeled image data for model fine-tuning. Experiments show that UIE-X can further improve the results through a small amount of data (few-shot) and PP-Structure layout analysis.
