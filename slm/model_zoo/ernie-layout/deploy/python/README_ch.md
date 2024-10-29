[English](README.md) | 简体中文

# ERNIE-Layout Python 部署指南

本文介绍 ERNIE-Layout Python 部署指南，包括部署环境的准备，文档信息抽取、文档视觉问答和文档图像分类三大场景下的使用示例。

- [1. 开始运行](#1-开始运行)
- [2. 文档信息抽取模型推理](#2-文档信息抽取模型推理)
- [3. 文档视觉问答模型推理](#3-文档视觉问答模型推理)
- [4. 文档图像分类模型推理](#4-文档图像分类模型推理)
- [5. 更多配置](#5-更多配置)

## 1. 开始运行

#### 环境要求

- 请执行以下命令进行安装项目依赖

```
pip install -r requirements.txt
```

#### 数据准备

- 提供了少量图片数据，可用于后续章节的部署测试，下载后放在``./images``目录。

```shell
wget https://bj.bcebos.com/paddlenlp/datasets/document_intelligence/images.zip && unzip images.zip
```

## 2. 文档信息抽取模型推理

- 使用如下命令进行英文文档信息抽取部署

```shell
python infer.py \
    --model_path_prefix ../../ner_export/inference \
    --task_type ner \
    --lang "en" \
    --batch_size 8
```

- 输出样例

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

## 3. 文档视觉问答模型推理

- 使用如下命令进行中文文档视觉问答部署

```shell
python infer.py \
    --model_path_prefix ../../mrc_export/inference \
    --task_type mrc \
    --lang "ch" \
    --batch_size 8
```

- 输出样例

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

## 4. 文档图像分类模型推理

- 使用如下命令进行英文文档图像分类部署

```shell
python infer.py \
    --model_path_prefix ../../cls_export/inference \
    --lang "en" \
    --task_type cls \
    --batch_size 8
```

- 输出样例

```
[{'doc': './images/cls_sample.jpg', 'result': 'email'}]
```

## 5. 更多配置

- `model_path_prefix`: 用于推理的 Paddle 模型文件路径，需加上文件前缀名称。例如模型文件路径为`./export/inference.pdiparams`，则传入`./export/inference`。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_length`: 如果 OCR 的结果超过设定的最大长度则对 OCR 结果进行自动切分，默认为512。
- `task_type`: 选择任务类型，可选有`ner`, `cls`和`mrc`。
- `lang`: 选择任务的语言类型，可选有`en`, `ch`。
- `device`: 选用什么设备进行训练，可选`cpu`或`gpu`。
