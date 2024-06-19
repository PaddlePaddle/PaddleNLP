English | [简体中文](README_ch.md)

# ERNIE-Layout Python Deploy Guide

- [1. Quick Start](#1)
- [2. Key Information Extraction Deploy](#2)
- [3. Document Question Answering Deploy](#3)
- [4. Document Image Classification Deploy](#4)
- [5. Parameter Description](#5)

<a name="1"></a>

## 1. Quick Start

#### Environment

- Dependency Installation

```
pip install -r requirements.txt
```

#### Data Preparation

- Dowload the sample images and put in ``./images``

```shell
wget https://bj.bcebos.com/paddlenlp/datasets/document_intelligence/images.zip && unzip images.zip
```

<a name="2"></a>

## 2. Key Information Extraction Deploy

- Run

```shell
python infer.py \
    --model_path_prefix ../../ner_export/inference \
    --task_type ner \
    --lang "en" \
    --batch_size 8
```

- Output sample

```
[{'doc': './images/ner_sample.jpg',
  'result': [{'text': 'ATT . GEN . ADMIN . OFFICE',
              'label': 'QUESTION',
              'start': 0,
              'end': 12,
              'probability': 0.8961102192651806},
             {'text': 'Fax :',
              'label': 'QUESTION',
              'start': 13,
              'end': 14,
              'probability': 0.8005126895801068},
             {'text': '614',
              'label': 'ANSWER',
              'start': 15,
              'end': 16,
              'probability': 0.5063673730110718},
             {'text': 'Dec 10',
              'label': 'ANSWER',
              'start': 23,
              'end': 24,
              'probability': 0.6265156606943465},

            ......

             {'text': 'NOTE',
              'label': 'QUESTION',
              'start': 179,
              'end': 179,
              'probability': 0.9810855421041412}]}]
```

<a name="3"></a>

## 3. Document Question Answering Deploy

- Run

```shell
python infer.py \
    --model_path_prefix ../../mrc_export/inference \
    --task_type mrc \
    --lang "ch" \
    --batch_size 8
```

- Output sample

```
[{'doc': './images/mrc_sample.jpg',
  'result': [{'question': '杨小峰是什么身份？', 'answer': ['法定代表人']},
             {'question': '花了多少钱进行注册的这个公司？', 'answer': ['壹仟壹佰万元']},
             {'question': '公司的类型属于什么？', 'answer': ['有限责任公司']},
             {'question': '杨小峰的住所是在哪里？',
              'answer': ['成都市武侯区佳灵路20号九峰国际1栋16楼62号']},
             {'question': '这个公司的法定代表人叫什么？', 'answer': ['杨小峰']},
             {'question': '91510107749745776R代表的是什么？', 'answer': ['统一社会信用代码']},
             {'question': '公司在什么时候成立的？',
              'answer': ['2003年7月22日营业期限2003年7月22日']}]}]
```

<a name="4"></a>

## 4. Document Image Classification Deploy

- Run

```shell
python infer.py \
    --model_path_prefix ../../cls_export/inference \
    --lang "en" \
    --task_type cls \
    --batch_size 8
```

- Output sample

```
[{'doc': './images/cls_sample.jpg', 'result': 'email'}]
```

<a name="5"></a>

## 5. Parameter Description

- `model_path_prefix`: The file path of the Paddle model for inference, with the file prefix name。For example, the inference model file path is `./export/inference.pdiparams`, then pass `./export/inference`。
- `batch_size`: number of input of each batch, default to 1.
- `max_seq_length`: If the OCR result exceeds the set maximum length, the OCR result will be sliced. The default is 512.
- `task_type`: choose the task type，the options are `ner`, `cls` and `mrc`。
- `lang`: select the task language，the options are `en` and `ch`。
- `device`: choose the device，the options are `cpu` and `gpu`。
