# 使用TextCNN模型完成中文对话情绪识别任务

情感分析旨在自动识别和提取文本中的倾向、立场、评价、观点等主观信息。情感分析其中的一个任务就是对话情绪识别，针对智能对话中的用户文本，自动判断该文本的情绪类别并给出相应的置信度，情绪类型分为积极（positive）、消极（negative）和中性（neutral）。

本示例展示了如何用TextCNN预训练模型在机器人聊天数据集上进行Finetune完成中文对话情绪识别任务。

## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
textcnn/
├── deploy # 部署
│   └── python
│       └── predict.py # python预测部署示例
├── data.py # 数据处理脚本
├── export_model.py # 动态图参数导出静态图参数脚本
├── model.py # 模型组网脚本
├── predict.py # 模型预测脚本
├── README.md # 文档说明
└── train.py # 对话情绪识别任务训练脚本
```

### 数据准备

这里我们提供一份已标注的机器人聊天数据集，包括训练集（train.tsv），开发集（dev.tsv）和测试集（test.tsv）。
完整数据集可以通过以下命令下载并解压：

```shell
wget https://bj.bcebos.com/paddlenlp/datasets/RobotChat.tar.gz
tar xvf RobotChat.tar.gz
```

### 词表下载

在模型训练之前，需要先下载词汇表文件word_dict.txt，用于构造词-id映射关系。

```shell
wget https://bj.bcebos.com/paddlenlp/robot_chat_word_dict.txt
```

**NOTE:** 词表的选择和实际应用数据相关，需根据实际数据选择词表。

### 预训练模型下载

这里我们提供了一个百度基于海量数据训练好的TextCNN模型，用户通过以下方式下载预训练模型。

```shell
wget https://bj.bcebos.com/paddlenlp/models/textcnn.pdparams
```

### 模型训练

在下载好词表和预训练模型后就可以在机器人聊天数据集上进行finetune，通过运行以下命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证，这里通过`--init_from_ckpt=./textcnn.pdparams`指定TextCNN预训练模型。

CPU 启动：

```shell
python train.py --vocab_path=./robot_chat_word_dict.txt \
    --init_from_ckpt=./textcnn.pdparams \
    --device=cpu \
    --lr=5e-5 \
    --batch_size=64 \
    --epochs=10 \
    --save_dir=./checkpoints \
    --data_path=./RobotChat
```

GPU 启动：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py \
    --vocab_path=./robot_chat_word_dict.txt \
    --init_from_ckpt=./textcnn.pdparams \
    --device=gpu \
    --lr=5e-5 \
    --batch_size=64 \
    --epochs=10 \
    --save_dir=./checkpoints \
    --data_path=./RobotChat
```

XPU启动：

```shell
python train.py --vocab_path=./robot_chat_word_dict.txt \
    --init_from_ckpt=./textcnn.pdparams \
    --device=xpu \
    --lr=5e-5 \
    --batch_size=64 \
    --epochs=10 \
    --save_dir=./checkpoints \
    --data_path=./RobotChat
```

以上参数表示：

* `vocab_path`: 词汇表文件路径。
* `init_from_ckpt`: 恢复模型训练的断点路径。
* `device`: 选用什么设备进行训练，可选cpu、gpu或xpu。如使用gpu训练则参数gpus指定GPU卡号。
* `lr`: 学习率， 默认为5e-5。
* `batch_size`: 运行一个batch大小，默认为64。
* `epochs`: 训练轮次，默认为10。
* `save_dir`: 训练保存模型的文件路径。
* `data_path`: 数据集文件路径。


程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── 0.pdopt
├── 0.pdparams
├── 1.pdopt
├── 1.pdparams
├── ...
└── final.pdparams
```

**NOTE:**

* 如需恢复模型训练，则init_from_ckpt只需指定到文件名即可，不需要添加文件尾缀。如`--init_from_ckpt=checkpoints/0`即可，程序会自动加载模型参数`checkpoints/0.pdparams`，也会自动加载优化器状态`checkpoints/0.pdopt`。
* 使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，具体代码见export_model.py。静态图参数保存在`output_path`指定路径中。
  运行方式：

```shell
python export_model.py --vocab_path=./robot_chat_word_dict.txt --params_path=./checkpoints/final.pdparams --output_path=./static_graph_params
```

其中`params_path`是指动态图训练保存的参数路径，`output_path`是指静态图参数导出路径。

导出模型之后，可以用于部署，deploy/python/predict.py文件提供了python部署预测示例。运行方式：

```shell
python deploy/python/predict.py --model_file=static_graph_params.pdmodel --params_file=static_graph_params.pdiparams
```

### 模型预测

启动预测：

CPU启动：

```shell
python predict.py --vocab_path=./robot_chat_word_dict.txt \
    --device=cpu \
    --params_path=./checkpoints/final.pdparams
```

GPU启动：

```shell
export CUDA_VISIBLE_DEVICES=0
python predict.py --vocab_path=./robot_chat_word_dict.txt \
    --device=gpu \
    --params_path=./checkpoints/final.pdparams
```

XPU启动：

```shell
python predict.py --vocab_path=./robot_chat_word_dict.txt \
    --device=xpu \
    --params_path=./checkpoints/final.pdparams
```

待预测数据如以下示例：

```text
你再骂我我真的不跟你聊了
你看看我附近有什么好吃的
我喜欢画画也喜欢唱歌
```

经过`preprocess_prediction_data`函数处理后，调用`predict`函数即可输出预测结果。

如

```text
Data: 你再骂我我真的不跟你聊了    Label: negative
Data: 你看看我附近有什么好吃的   Label: neutral
Data: 我喜欢画画也喜欢唱歌       Label: positive
```

## Reference

TextCNN参考论文：

- [EMNLP2014-Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181.pdf)
