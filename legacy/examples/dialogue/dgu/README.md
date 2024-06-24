# 对话通用理解模型 (DGU, Dialogue General Understanding)

## 模型简介

对话系统 (Dialogue System) 常常需要根据应用场景的变化去解决多种多样的任务。任务的多样性（意图识别、槽填充、行为识别、状态追踪等等），以及领域训练数据的稀少，给Dialogue System的研究和应用带来了巨大的困难和挑战，要使得Dialogue System得到更好的发展，需要开发一个通用的对话理解模型。为此，我们给出了基于BERT的对话通用理解模型 (DGU: Dialogue General Understanding)，通过实验表明，使用base-model (BERT)并结合常见的学习范式，就可以在几乎全部对话理解任务上取得比肩甚至超越各个领域业内最好的模型的效果，展现了学习一个通用对话理解模型的巨大潜力。

DGU模型内共包含6个任务，全部基于公开数据集在Paddle2.0上完成训练及评估，详细说明如下：

```
udc: 使用UDC (Ubuntu Corpus V1) 数据集完成对话匹配 (Dialogue Response Selection) 任务;
dstc2: 使用DSTC2 (Dialog State Tracking Challenge 2) 数据集完成对话状态追踪 (Dialogue State Tracking) 任务;
atis_slot: 使用ATIS (Airline Travel Information System) 数据集完成对话槽填充 (Dialogue Slot Filling) 任务；
atis_intent: 使用ATIS (Airline Travel Information System) 数据集完成对话意图识别 (Dialogue Intent Detection) 任务；
mrda: 使用MRDAC (Meeting Recorder Dialogue Act Corpus) 数据集完成对话行为识别 (Dialogue Act Detection) 任务；
swda: 使用SwDAC (Switchboard Dialogue Act Corpus) 数据集完成对话行为识别 (Dialogue Act Detection) 任务;
```

## 模型效果

DGU模型中的6个任务，分别采用不同的评估指标在test集上进行评估，结果如下：

<table>
    <tr><th style="text-align:center">任务</th><th style="text-align:center">评估指标</th><th style="text-align:center">DGU</th></tr>
    <tr align="center"><td rowspan="3" style="vertical-align:middle;">udc</td><td>R1@10</td><td>81.04%</td></tr>
    <tr align="center"><td>R2@10</td><td>89.85%</td></tr>
    <tr align="center"><td>R5@10</td><td>97.59%</td></tr>
    <tr align="center"><td>dstc2</td><td>Joint_Acc</td><td>90.43%</td></tr>
    <tr align="center"><td>atis_slot</td><td>F1_Micro</td><td>97.98%</td></tr>
    <tr align="center"><td>atis_intent</td><td>Acc</td><td>97.42%</td></tr>
    <tr align="center"><td>mrda</td><td>Acc</td><td>90.94%</td></tr>
    <tr align="center"><td>swda</td><td>Acc</td><td>80.61%</td></tr>
</table>

**NOTE:** 以上结果均是采用默认配置在GPU单卡上训练和评估得到的，用户如需复现效果，可采用默认配置在单卡上进行训练评估。

## 快速开始

### 数据准备

下载数据集压缩包并解压后，DGU_datasets目录下共存在6个目录，分别对应每个任务的训练集train.txt、评估集dev.txt和测试集test.txt。

```shell
wget https://bj.bcebos.com/paddlenlp/datasets/DGU_datasets.tar.gz
tar -zxf DGU_datasets.tar.gz
```

DGU_datasets目录结构：

```text
DGU_datasets/
├── atis_intent
│   ├── dev.txt
│   ├── map_tag_intent_id.txt
│   ├── test.txt
│   └── train.txt
├── udc
│   ├── dev.txt
│   ├── dev.txt-small
│   ├── test.txt
│   └── train.txt
├── atis_slot
│   ├── dev.txt
│   ├── map_tag_slot_id.txt
│   ├── test.txt
│   └── train.txt
├── dstc2
│   ├── dev.txt
│   ├── map_tag_id.txt
│   ├── test.txt
│   └── train.txt
├── mrda
│   ├── dev.txt
│   ├── map_tag_id.txt
│   ├── test.txt
│   └── train.txt
└── swda
    ├── dev.txt
    ├── map_tag_id.txt
    ├── test.txt
    └── train.txt
```

数据的每一行由多列组成，都以"\t"作为分割符，详细数据格式说明如下：

```
udc：由label、多轮对话conv和回应response组成
格式：label \t conv1 \t conv2 \t conv3 \t ... \t response

dstc2：由多轮对话id、当前轮QA对(使用\1拼接)和对话状态序列state_list(state_list中每个state由空格分割)组成
格式：conversation_id \t question \1 answer \t state1 state2 state3 ...

atis_slot：由对话内容conversation_content和标签序列label_list (label_list中每个label由空格分割) 组成, 其中标签序列和对话内容中word为一一对应关系
格式：conversation_content \t label1 label2 label3 ...

atis_intent：由标签label和对话内容conversation_content组成
格式： label \t conversation_content

mrda：由多轮对话id、标签label、发言人caller、对话内容conversation_content组成
格式：conversation_id \t label \t caller \t conversation_content

swda：由多轮对话id、标签label、发言人caller、对话内容conversation_content组成
格式：conversation_id \t label \t caller \t conversation_content
```

**NOTE:** 上述数据集来自于 [Paddle1.8静态图版本](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/dialogue_system/dialogue_general_understanding)，是由相应的开源数据集经过数据格式转换而得来的，本项目中暂未包含数据格式转换脚本，细节请参考 [Paddle1.8静态图版本](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/dialogue_system/dialogue_general_understanding)。

### 模型训练

运行如下命令即可在训练集 (train.tsv) 上进行模型训练，并在开发集 (dev.tsv) 验证，训练结束后会在测试集 (test.txt) 上进行模型评估

```shell
# GPU启动，gpus指定训练所用的GPU卡号，可以是单卡，也可以多卡。默认会进行训练、验证和评估
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" --log_dir ./log main.py --task_name=udc --data_dir=./DGU_datasets/udc --output_dir=./checkpoints/udc --device=gpu
# 若只需进行评估，do_train设为False，并且必须指定init_from_ckpt
# python -m paddle.distributed.launch --gpus "0" --log_dir ./log main.py --task_name=udc --data_dir=./DGU_datasets/udc --do_train=False --init_from_ckpt=./checkpoints/udc/best --device=gpu
```

以上参数表示：

* `task_name`：任务名称，可以为udc、dstc2、atis_slot、atis_intent、mrda或swda。
* `data_dir`：训练数据路径。
* `output_dir`：训练保存模型的文件路径。
* `do_train：是否进行训练，默认为`True`。
* `init_from_ckpt`：恢复模型参数的路径。
* `device`：表示训练使用的设备。

其他可选参数和参数的默认值请参考`args.py`。

程序运行时将会自动进行训练，验证和评估。同时训练过程中会自动保存模型在指定的`output_dir`中。
如：
```text
checkpoints/
├── 1000.pdopt
├── 1000.pdparams
├── 2000.pdopt
├── 2000.pdparams
├── ...
├── best.pdopt
└── best.pdparams
```

**NOTE:** 如需恢复模型训练，则init_from_ckpt只需指定到文件名即可，不需要添加文件尾缀。如`--init_from_ckpt=checkpoints/1000`即可，程序会自动加载模型参数`checkpoints/1000.pdparams`，也会自动加载优化器状态`checkpoints/1000.pdopt`。
