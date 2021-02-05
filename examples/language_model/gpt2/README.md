# GPT-2

## 模型介绍
[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)(Language Models are Unsupervised Multitask Learners) 以[Transformer](https://arxiv.org/abs/1706.03762) 解码器为网络基本组件，使用自回归的方式在大规模无标注文本语料上进行预训练得到的语言生成模型。

本项目是语言模型 GPT-2 的 PaddlePaddle 实现， 包含模型训练，预测等内容。下是本例的简要目录结构及说明：

```text
.
├── data.py                 # 数据处理
├── decompress.sh           # 数据集解压脚本
├── generate_sample.py      # inference demo
├── lr.py                   # 学习率控制
├── process_data.py         # 数据预处理脚本
├── README.md               # 文档
├── run_pretrain.py         # 预训练入口
└── scripts                 # 训练脚本
```

## 快速开始

### 安装说明

1. paddle安装

    本项目依赖于 PaddlePaddle 2.0及以上版本或适当的develop版本，请参考 [安装指南](https://www.paddlepaddle.org.cn/install/quick) 进行安装

2. 下载代码

    克隆代码库到本地

3. 环境依赖

    该模型使用PaddlePaddle，关于环境依赖部分，请先参考PaddlePaddle[安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)关于环境依赖部分的内容。


### 数据准备

#### 原始数据获取

[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/)是一个开源的英文网页文本数据集，数据来源于Reddit，经过去重、清洗、提取，最终包含800多万个文档。

下载以后通过以下命令解压：

```shell
xz -d openwebtext.tar.xz
tar xf openwebtext.tar
mkdir raw_data
bash decompress.sh  
```

解压以后得到的raw_data目录大小约为54GB。

#### 数据预处理

为了提升训练速度，我们在训练前将文本数据转成相应的id，并保存为npz格式：

```shell
python process_data.py --input_path raw_data \
 --model_name gpt2-medium-en \
 --append_eod \
 --workers 8
```

运行命令后，产出`raw_data_ids.npz`文件。为了方便用户运行测试本模型，本项目提供了处理好的300M的训练样本：

```shell
wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt2/train.data.json_ids.npz
```

将所有预处理得到的npz文件统一放入一个文件夹中，以备训练使用：

```
mkdir data
mv train.data.json_ids.npz data
```


#### 单卡训练

```shell
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --model_name_or_path gpt2-small-en \
    --input_dir "./data"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --warmup_rate 0.01\
    --batch_size 8\
    --device gpu
```

其中参数释义如下：
- `model_name_or_path` 要训练的模型或者之前训练的checkpoint。
- `input_dir` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件。
- `output_dir` 指定输出文件。
- `weight_decay` 权重衰减参数。
- `grad_clip` 梯度裁剪范围。
- `max_steps` 最大训练步数
- `save_steps` 保存模型间隔
- `batch_size` 训练的batch大小
- `device` 训练设备

用户也可以使用提供的shell脚本直接训练`sh scripts/run.sh`.

### 单机多卡

同样，可以执行如下命令实现八卡训练：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py --model_name_or_path gpt2-small-en \
    --input_dir "./data"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --warmup_rate 0.01\
    --batch_size 8\
    --device gpu

```

用户也可以使用提供的shell脚本直接训练`sh scripts/run_multi.sh`.

#### 文本生成

本项目提供了简单的文本生成的demo，供用户测试文本生成效果。

```shell
python generate_sample.py
```

生成效果展示:
```text
问题：中国的首都是哪里？答案：北京。

默写古诗: 大漠孤烟直，长河落日圆。
举杯邀明月，

对影成三人。
```

## 参考文献
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413)
