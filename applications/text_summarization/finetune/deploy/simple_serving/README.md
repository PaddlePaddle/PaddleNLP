# SimpleServing服务化部署

本文档将介绍如何使用[PaddleNLP SimpleServing](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/server.md)工具部署自动文本摘要在线服务。

## 目录
- [SimpleServing服务化部署](#SimpleServing服务化部署)
  - [目录](#目录)
  - [背景介绍](#背景介绍)
  - [环境准备](#环境准备)
  - [启动服务](#启动服务)
  - [发送请求](#发送请求)
  - [服务化自定义参数](#服务化自定义参数)
    - [server参数](#server参数)
      - [模型路径](#模型路径)
      - [多卡服务化预测](#多卡服务化预测)
      - [Taskflow加速](#Taskflow加速)
    - [client参数](#client参数)



## 背景介绍
PaddleNLP SimpleServing 是基于 unicorn 封装的模型部署服务化工具，该服务化工具具备灵活、易用的特性，可以简易部署预训练模型和预训练模型工具Taskflow，PaddleNLP SimpleServing 具备以下两个特性：

- 易用：一行代码即可部署预训练模型和预训练工具Taskflow
- 灵活：Handler机制可以快速定制化服务化部署方式

PaddleNLP SimpleServing Python端预测部署主要包含以下步骤：
- 环境准备
- 启动服务
- 发送请求

## 环境准备
下载安装包含SimpleServing功能的PaddleNLP版本：
```shell
pip install paddlenlp
```

## 启动服务
```shell
paddlenlp server server:app --workers 1 --host 0.0.0.0 --port 8189
```

## 发送请求
```shell
python client.py
```

## 服务化自定义参数

### server参数

#### 模型路径

默认使用的模型为 `IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese` , 用户也可以通过修改`task_path`参数使用其他模型或自己的模型：

```shell
ts = Taskflow("text_summarization", task_path='../../checkpoint/model_best/')
```
可选模型有 `PaddlePaddle/Randeng-Pegasus-238M-Summary-Chinese-SSTIA`， `PaddlePaddle/Randeng-Pegasus-523M-Summary-Chinese-SSTIA`， `unimo-text-1.0-summary`， `IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese`， `IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese`

#### 多卡服务化预测
PaddleNLP SimpleServing 支持多卡负载均衡预测，主要在服务化注册的时候，注册两个Taskflow的task即可，下面是示例代码：

```shell
ts1 = Taskflow('text_summarization', device_id=0)
ts2 = Taskflow('text_summarization', device_id=1)
service.register_taskflow("taskflow/text_summarization", [ts1, ts2])
```

#### Taskflow加速
PaddleNLP SimpleServing 支持在线服务加速，需要在注册Taskflow时设置参数`use_faster`：

```shell
ts = Taskflow("text_summarization", use_faster=True)
```

### client参数
用户修改`client.py`中的texts变量以对任意文本进行摘要。
