[English](README.md) | 简体中文

# ERNIE-Layout Python部署指南

本文介绍 ERNIE-Layout Python部署指南，包括部署环境的准备，文档信息抽取、文档视觉问答和文档图像分类三大场景下的使用示例。

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
python infer.py --model_dir ../../ner_export --task_type ner --device gpu --lang "en" --batch_size 8
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
python infer.py --model_dir ../../mrc_export/ --task_type mrc --device gpu  --lang "ch" --batch_size 8
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
python infer.py --model_dir ../../cls_export/ --task_type cls --lang "en" --batch_size 8
```

- 输出样例

```
[{'doc': './images/cls_sample.jpg', 'result': 'email'}]
```

## 5. 更多配置

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定部署模型的目录 |
|--batch_size |输入的batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--task_type| 选择任务类型，可选有`ner`, `cls`和`mrc`。|
|--lang| 选择任务的语言类型，可选有`en`, `ch`。|
|--device | 运行的设备，可选范围: ['cpu', 'gpu']，默认为'cpu' |
|--device_id | 运行设备的id。默认为0。 |
|--cpu_threads | 当使用cpu推理时，指定推理的cpu线程数，默认为1。|
|--backend | 支持的推理后端，可选范围: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，默认为'paddle' |
|--use_fp16 | 是否使用FP16模式进行推理。使用tensorrt和paddle_tensorrt后端时可开启，默认为False |
