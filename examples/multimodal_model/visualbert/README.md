# VisualBERT: A Simple and Performant Baseline for Vision and Language


[VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/pdf/1908.03557.pdf)

多模态预训练 + 子任务微调

## 快速开始

### （一）数据准备

`coco2014`图像 + `VQA2`对应问答标注数据，`NLVR2` 图像+文字数据

coco数据读取需要安装 `pycocotools`

```bash
pip install pycocotools
```

### （二）模型预训练

#### 2.1 基于coco captions数据预训练 visualbert-vqa-coco-pre

使用COCO captions数据预训练

两个优化任务：(1) Masked language Model 30w+分类伪标签 (2) 随机采样两个句子拼接，看是否是对同一个图片的描述，2分类伪标签

> 运行 `sh vqa-coco-pretrain.sh`，载入预训练模型`visualbert-vqa-coco-pre.pdparams`
VQA2数据集对应 `coco_detectron_fix_100` 图像特征, 需要`./X_COCO/data/detectron_fix_100`

```bash
export DATA_DIR=./X_COCO/
export LOG_DIR=./logs/vqa-coco-pre
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "1" --log_dir $LOG_DIR run_pretrain.py \
    --input_dir $DATA_DIR \
    --output_dir $LOG_DIR \
    --dataset coco_captions \
    --model_type visualbert \
    --model_name_or_path visualbert-vqa-coco-pre \
    --image_feature_type coco_detectron_fix_100 \
    --train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 3
```

`input_dir`: 数据根目录
`output_dir`: 运行日志，checkpoint存放位置
`dataset`: 预训练使用模型名称，在`paddlenlp.datasets`中定义
`model_type`: 训练模型名称，在`paddlenlp.transformers`中定义
`model_name_or_path`: 训练参数名称或 `pdparams` 地址
`image_feature_type`: `visualbert`采用的图像特征的名称
`train_batch_size`: 训练 `batchsize`
`learning_rate`: 学习率
`num_train_epochs`: 训练轮次

#### 2.2 基于vqa2数据预训练 visualbert-vqa-pre

使用VQA2数据预训练，VQA数据集对COCO的图像添加了问答形式的标注，图像特征仍来自于COCO数据集

一个优化任务：(1) Masked language Model 30w+分类伪标签

> 运行 `sh vqa-pretrain.sh`，载入预训练模型`visualbert-vqa-pre.pdparams`, 训练日志见 `logs/vqa-pre`
VQA2数据集对应 `coco_detectron_fix_100` 图像特征, 需要`./X_COCO/data/detectron_fix_100`

```bash
export DATA_DIR=./X_COCO/
export LOG_DIR=./logs/vqa-pre
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "3" --log_dir $LOG_DIR run_pretrain.py \
    --input_dir $DATA_DIR \
    --output_dir $LOG_DIR \
    --dataset vqa2 \
    --model_type visualbert \
    --model_name_or_path visualbert-vqa-pre \
    --image_feature_type coco_detectron_fix_100 \
    --train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 3
```

`input_dir`: 数据根目录
`output_dir`: 运行日志，checkpoint存放位置
`dataset`: 预训练使用模型名称，在`paddlenlp.datasets`中定义
`model_type`: 训练模型名称，在`paddlenlp.transformers`中定义
`model_name_or_path`: 训练参数名称或 `pdparams` 地址
`image_feature_type`: `visualbert`采用的图像特征的名称
`train_batch_size`: 训练 `batchsize`
`learning_rate`: 学习率
`num_train_epochs`: 训练轮次

---

#### 2.3 基于coco captions数据预训练 nlvr2-coco-pre

使用COCO captions数据预训练
两个优化任务：(1) Masked language Model 30w+分类伪标签 (2) 随机采样两个句子拼接，看是否是对同一个图片的描述，2分类伪标签

> 运行 `sh nlvr2-coco-pretrain.sh`，载入预训练模型`visualbert-nlvr2-coco-pre.pdparams`, 训练日志见 `logs/nlvr2-coco-pre`
NLVR2数据集对应 采用 coco_detectron_fix_144 图像特征, 需要`./X_COCO/data/detectron_fix_144`

```bash
export DATA_DIR=./X_COCO/
export LOG_DIR=./logs/nlvr2-coco-pre
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "2" --log_dir $LOG_DIR run_pretrain.py \
    --input_dir $DATA_DIR \
    --output_dir $LOG_DIR \
    --dataset coco_captions \
    --model_type visualbert \
    --model_name_or_path visualbert-nlvr2-coco-pre \
    --image_feature_type coco_detectron_fix_144 \
    --train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 3
```

`input_dir`: 数据根目录
`output_dir`: 运行日志，checkpoint存放位置
`dataset`: 预训练使用模型名称，在`paddlenlp.datasets`中定义
`model_type`: 训练模型名称，在`paddlenlp.transformers`中定义
`model_name_or_path`: 训练参数名称或 `pdparams` 地址
`image_feature_type`: `visualbert`采用的图像特征的名称
`train_batch_size`: 训练 `batchsize`
`learning_rate`: 学习率
`num_train_epochs`: 训练轮次

#### 2.4 基于nlvr2数据预训练 nlvr2-pre

NLVR2数据集对多对网络图片进行了 Visual Reasoning 形式的标注
两张图组成一个样本
一个优化任务：(1) Masked language Model 30w+分类伪标签

> 运行 `sh nlvr2-pretrain.sh`，载入预训练模型`visualbert-nlvr2-pre.pdparams`, 训练日志见 `logs/nlvr2-pre`
NLVR2数据集对应 `nlvr2_detectron_fix_144` 图像特征, 需要`./X_NLVR/data/detectron_fix_144`

```bash
export DATA_DIR=./X_NLVR/
export LOG_DIR=./logs/nlvr2-pre
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "4" --log_dir $LOG_DIR run_pretrain.py \
    --input_dir $DATA_DIR \
    --output_dir $LOG_DIR \
    --dataset nlvr2 \
    --model_type visualbert \
    --model_name_or_path visualbert-nlvr2-pre \
    --image_feature_type nlvr2_detectron_fix_144 \
    --train_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 3
```

`input_dir`: 数据根目录
`output_dir`: 运行日志，checkpoint存放位置
`dataset`: 预训练使用模型名称，在`paddlenlp.datasets`中定义
`model_type`: 训练模型名称，在`paddlenlp.transformers`中定义
`model_name_or_path`: 训练参数名称或 `pdparams` 地址
`image_feature_type`: `visualbert`采用的图像特征的名称
`train_batch_size`: 训练 `batchsize`
`learning_rate`: 学习率
`num_train_epochs`: 训练轮次


### （三）下游任务微调

#### 3.1. VQA2

> 运行 `sh vqa-finetune.sh`，载入预训练模型`visualbert-vqa-pre/model_state.pdparams`, 训练日志见 `logs/vqa`
VQA2数据集对应 `coco_detectron_fix_100` 图像特征, 需要`./X_COCO/data/detectron_fix_100`

一个优化任务：(1) 3129 个 answer 分类(有监督)

```bash
export DATA_DIR=./X_COCO/
export LOG_DIR=./logs/vqa
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "0" --log_dir $LOG_DIR VQA2/run_vqa2.py \
    --input_dir $DATA_DIR \
    --output_dir $LOG_DIR \
    --task_name vqa2 \
    --model_type visualbert \
    --model_name_or_path visualbert-vqa-pre \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --save_steps 10000 \
    --num_train_epochs 10
```

`input_dir`: 数据根目录
`output_dir`: 运行日志，checkpoint存放位置
`task_name`: 下游任务名称，通过名称选择下游任务的评估Metric
`model_type`: 训练模型名称，在`paddlenlp.transformers`中定义
`model_name_or_path`: 训练参数名称或 `pdparams` 地址
`batch_size`: 训练和评估 `batchsize`
`learning_rate`: 学习率
`num_train_epochs`: 训练轮次

#### 3.2. NLVR2
> 运行 `sh nlvr2-finetune.sh`，载入预训练模型`visualbert-nlvr2-pre/model_state.pdparams`, 训练日志见 `logs/nlvr2`
NLVR2数据集对应 `nlvr2_detectron_fix_144` 图像特征, 需要`./X_NLVR/data/detectron_fix_144`

一个优化任务：(1) 2 个 answer 分类(有监督)

```bash
export DATA_DIR=./X_NLVR/
export LOG_DIR=./logs/nlvr2
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "1,2" --log_dir $LOG_DIR NLVR2/run_nlvr2.py \
    --input_dir $DATA_DIR \
    --output_dir $LOG_DIR \
    --task_name nlvr2 \
    --model_type visualbert \
    --model_name_or_path visualbert-nlvr2-pre \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --save_steps 5000 \
    --num_train_epochs 10
```

`input_dir`: 数据根目录
`output_dir`: 运行日志，checkpoint存放位置
`task_name`: 下游任务名称，通过名称选择下游任务的评估Metric
`model_type`: 训练模型名称，在`paddlenlp.transformers`中定义
`model_name_or_path`: 训练参数名称或 `pdparams` 地址
`batch_size`: 训练和评估 `batchsize`
`learning_rate`: 学习率
`num_train_epochs`: 训练轮次

#### 路径
.
├── NLVR2 # NLVR2 数据集 训练和推理程序
│   ├── run_nlvr2.py # 训练程序
│   └── run_predict.py # 推理程序，结果与原论文对比
├── VQA2 # VQA2 数据集 训练和推理程序
│   ├── result.json # 推理结果，提交到竞赛网站，获取测试集得分
│   ├── run_predict.py # 推理程序，结果与原论文对比
│   └── run_vqa2.py # 训练程序
├── X_COCO # COCO数据
│   └── data
│       ├── detectron_fix_100 -> /mnt/ssd/X_COCO/data/detectron_fix_100
│       └── detectron_fix_144 -> /mnt/ssd/X_COCO/data/detectron_fix_144
├── X_NLVR # NLVR数据
|   └── data
|       └── detectron_fix_144 -> /mnt/ssd/X_NLVR/data/detectron_fix_144
├── run_pretrain.py # 模型预训练程序
├── vqa-coco-pretrain.sh # coco captions 预训练脚本
├── nlvr2-coco-pretrain.sh
├── vqa-pretrain.sh # vqa2 预训练脚本
├── nlvr2-pretrain.sh # nlvr2 预训练脚本
├── vqa-finetune.sh # vqa2 微调脚本
├── nlvr2-finetune.sh # nlvr2 微调脚本
└── README.md

图像特征使用 detetron2 预提取，例子如下：
[Generate Embeddings for VisualBERT (Colab Notebook) ](https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing): This notebook contains an example on how to generate visual embeddings.

**（1）COCO 预提取图像特征**

`X_COCO` : 下载链接
> 链接：https://pan.baidu.com/s/1xRZBrxRst3oqtaXdjP5lrA
提取码：7s29
--来自百度网盘超级会员V5的分享

`coco_detectron_fix_100`: `X_COCO/data/detectron_fix_100` 经过分卷压缩，包含 `.part00`-`.part32` 33 个压缩文件，每个2G，是`pddlenlp.dataset.coco_captions`和`pddlenlp.dataset.vqa2`数据集对应的图像特征
`coco_detectron_fix_144`: `X_COCO/data/detectron_fix_144` 经过分卷压缩，包含 `.part00`-`.part14` 15 个压缩文件，每个2G，是`pddlenlp.dataset.coco_captions`和`pddlenlp.dataset.nlvr2`数据集对应的图像特征

**（2）NLVR 预提取图像特征**
`X_NLVR` : 下载链接
> 链接：https://pan.baidu.com/s/1vdR6OcCxo6aEJLS4Wl4PPQ
提取码：13pc
--来自百度网盘超级会员V5的分享

`nlvr_detectron_fix_144`: `X_NLVR/data/detectron_fix_144` 包含 `.part00`~`.part14` 15 个压缩文件，每个2G，是`pddlenlp.dataset.nlvr2`数据集对应的图像特征


---
调用 `load_dataset(dataset_name)` 会将数据集对应 标注下载到 `.paddlenlp` 路径下
本样例中 `dataset_name` 可选为: `coco_captions`, `vqa2`, `nlvr2`

以下是`$HOME/.paddlenlp`目录, 存放`paddlenlp.dataset`中文本类型的数据集标注文件和模型文件

```bash
.
├── datasets
│   ├── ChnSentiCorp
│   ├── COCOCaptions
│   │   ├── annotations # 标注文件：文字+对应图片id
│   │   └── annotations_trainval2014.zip # 解压此文件，得到`annotations`文件夹
│   ├── NLVR2
│   │   └── annotations # 标注文件：文字+对应图片id，load_dataset 时会自动下载解压至此，并检查md5，`dev.json`, `test1.json` `train.json`
│   ├── VQA2
│   │   ├── annotations # 标注文件：文字+对应图片id, load_dataset 时会自动下载解压至此，并检查md5，imdb_test2015.npy，imdb_train2014.npy，imdb_val2014.npy
│   │   ├── extras # 子文件夹 vocabs 内包含 `answers_vqa.txt`  `vocabulary_100k.txt`  `vocabulary_vqa.txt` 三个文件
└── models # 存放`.pdparams`文件，from_pretrain 时下载至此
    ├── bert-base-cased
    ├── bert-base-uncased
    ├── ernie-1.0
    ├── visualbert-nlvr2
    ├── visualbert-nlvr2-coco-pre
    ├── visualbert-nlvr2-pre
    ├── visualbert-nvlr2
    ├── visualbert-vcr
    ├── visualbert-vcr-coco-pre
    ├── visualbert-vcr-pre
    ├── visualbert-vqa
    ├── visualbert-vqa-coco-pre
    └── visualbert-vqa-pre
```
