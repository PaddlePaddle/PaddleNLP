# ERNIE-ViL 2.0 基于多视角对比学习的跨模态预训练模型

 **目录**
   * [ERNIE-ViL 2.0  介绍](#模型介绍)
   * [预训练模型效果](#模型效果)
   * [代码结构](#代码结构)
   * [开始运行](#开始运行)
       * [任务介绍](#任务介绍)
       * [环境要求](#环境要求)
       * [数据准备](#数据准备)
   * [模型训练](#模型训练)
   * [模型评估](#模型评估)
   * [模型预测](#模型预测)
   * [参考文献](#参考文献)

本项目开源了 **ERNIE-ViL 2.0** 预训练模型及微调方案。


<a name="模型介绍"></a>

## ERNIE-ViL 2.0 介绍

近年来，基于大规模数据预训练的跨模态模型取得了令人瞩目的成绩。基于对比学习的双塔预训练框架能够利用大规模的噪声图文数据，在跨模态检索等任务上展现出较大的效果提升，同时具备计算效率高等优势，受到了广泛的关注（如CLIP，ALIGN等）。然而，已有的视觉-语言预训练技术基于单视角的对比学习，无法同时学习多种模态间和模态内的关联性。
ERNIE-ViL 2.0提出了一种基于多视角对比学习的预训练框架，通过构建丰富的视觉/文本视角，能够同时学习模态间和模态内的多种关联性，从而学习到更鲁棒的跨模态对齐，在跨模态检索等任务上取得了业界领先水平。

![framework](https://user-images.githubusercontent.com/12107462/212857637-c26882ab-c164-403c-b310-12282955dbc0.png)

使用 PaddleNLP 只需要一行代码就可以下载并获取 ERNIE-ViL 2.0 预训练模型，之后可以用自己的下游数据下进行微调。

```python

import requests
import paddle.nn.functional as F
from PIL import Image
from paddlenlp.transformers import ErnieViLModel, ErnieViLProcessor

processor = ErnieViLProcessor.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")
model = ErnieViLModel.from_pretrained("PaddlePaddle/ernie_vil-2.0-base-zh")
model.eval()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["一只猫的照片", "一条狗的照片"],
                images=image,
                padding=True,
                return_tensors="pd")

outputs = model(**inputs)

logits_per_image = outputs[0]
probs = F.softmax(logits_per_image, axis=1)
print(probs)


```

<a name="模型效果"></a>

## 预训练模型效果

### 跨模态检索效果
以下为以中、英文模型在Flickr30K、COCO-CN的zero-shot结果，其他详见论文。
* **ERNIE-ViL 2.0 英文 on Flickr30k**:

| Name       |   R@1 |   R@5 |   R@10 |
|------------|-------|-------|--------|
| Text2Image | 85.0 | 97.0 |  98.3 |
| Image2Text | 96.1 | 99.9 |  100.0 |

* **ERNIE-ViL 2.0 中文 COCO-CN**:

| Name       |   R@1 |   R@5 |   R@10 |
|------------|-------|-------|--------|
| Text2Image | 69.6 | 91.2 |  96.9 |
| Image2Text | 69.1 | 92.9 |  97.1 |

* 这里结果均为论文最好结果


<a name="代码结构"></a>

## 代码结构

以下是本项目代码结构

```text
├── data
│   └── datasets
│       └── Flickr30k-CN
│           ├── lmdb # 文本图像数据
│           │   ├── test
│           │   │   ├── imgs
│           │   │   │   ├── data.mdb
│           │   │   │   └── lock.mdb
│           │   │   └── pairs
│           │   │       ├── data.mdb
│           │   │       └── lock.mdb
│           │   ├── train
│           │   │   ├── imgs
│           │   │   │   ├── data.mdb
│           │   │   │   └── lock.mdb
│           │   │   └── pairs
│           │   │       ├── data.mdb
│           │   │       └── lock.mdb
│           │   └── valid
│           │       ├── imgs
│           │       │   ├── data.mdb
│           │       │   └── lock.mdb
│           │       └── pairs
│           │           ├── data.mdb
│           │           └── lock.mdb
│           ├── test_imgs.tsv # 图像测试集， 图片id & 图片内容
│           ├── test_texts.jsonl # 文本测试数据，文本id & 文本内容，连同匹配的图片id列表
│           ├── train_imgs.tsv # 图像训练集
│           ├── train_texts.jsonl # 文本训练集
│           ├── valid_imgs.tsv # 图像验证集
│           ├── valid_texts.jsonl # 文本验证集，文到图
│           └── valid_texts.tr.jsonl # 文本验证集，图到文
├── data.py  # 训练的预处理操作
├── extract_features.py # 提取图片和文本特征
├── extract_features.sh # 提取特征的bash脚本
├── README.md # README文档
├── recall_image_to_text.sh # 以图搜文的bash脚本
├── recall_text_to_image.sh # 以文搜图的bash脚本
├── run_train.sh # 微调的bash脚本
├── test_example.py # 预测的示例
├── trainer_finetune.py # trainer实现微调
├── trainer_util.py # 微调的工具代码
└── utils
    ├── evaluation.py # 评估以文搜图的召回脚本
    ├── evaluation_tr.py # 评估以图搜文的召回脚本
    ├── make_topk_predictions.py # 以文搜图的ann检索
    ├── make_topk_predictions_tr.py # 以图搜文的ann检索
    └── transform_ir_annotation_to_tr.py # 将图文对标注的jsonl文件由文到图的格式转为图到文
```

<a name="开始运行"></a>

## 开始运行

<a name="任务介绍"></a>

### 任务介绍

本项目是使用 ERNIE-ViL 2.0 的跨模态检索方案，任务背景搜索场景下图文互搜的任务。本项目包括微调流程。


### 环境要求
- python >= 3.7
- paddlepaddle >= 2.4.1
- paddlenlp >= 2.5
- lmdb==1.3.0

### 数据准备

本项目使用了 [Flickr30k-CN](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/Flickr30k-CN.zip) 中文场景下的图文数据集。

为了训练的时候方便随机读取，我们将tsv和jsonl文件一起序列化，转换成内存索引的LMDB数据库文件。

```shell
python preprocess/create_lmdb_dataset.py \
    --data_dir data/datasets/Flickr30k-CN \
    --splits train,valid,test

```
执行完后，data 目录应是如下结构：

```text
├── data
    └── datasets
        └── Flickr30k-CN
            ├── lmdb # 文本图像数据
            │   ├── test
            │   │   ├── imgs
            │   │   │   ├── data.mdb
            │   │   │   └── lock.mdb
            │   │   └── pairs
            │   │       ├── data.mdb
            │   │       └── lock.mdb
            │   ├── train
            │   │   ├── imgs
            │   │   │   ├── data.mdb
            │   │   │   └── lock.mdb
            │   │   └── pairs
            │   │       ├── data.mdb
            │   │       └── lock.mdb
            │   └── valid
            │       ├── imgs
            │       │   ├── data.mdb
            │       │   └── lock.mdb
            │       └── pairs
            │           ├── data.mdb
            │           └── lock.mdb
            ├── test_imgs.tsv # 图像测试集， 图片id & 图片内容
            ├── test_texts.jsonl # 文本测试数据，文本id & 文本内容，连同匹配的图片id列表
            ├── train_imgs.tsv # 图像训练集
            ├── train_texts.jsonl # 文本训练集
            ├── valid_imgs.tsv # 图像验证集
            ├── valid_texts.jsonl # 文本验证集，文到图
            └── valid_texts.tr.jsonl # 文本验证集，图到文
```


<a name="模型训练"></a>

## 模型训练


运行下面的脚本，使用 Trainer API 启动训练：

```shell
DATAPATH=./data

# data options
train_data=${DATAPATH}/datasets/Flickr30k-CN/lmdb/train
val_data=${DATAPATH}/datasets/Flickr30k-CN/lmdb/valid

# 启动方式
# --test_only \
log_dir=train_log
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
                --log_dir ${log_dir}  \
                trainer_finetune.py --output_dir output_pd \
                --train_data=${train_data} \
                --val_data=${val_data} \
                --do_train \
                --learning_rate 5e-5 \
                --warmup_steps 100 \
                --logging_steps 50 \
                --per_device_train_batch_size 128 \
                --dataloader_num_workers 8 \
                --save_steps 50 \
                --num_train_epochs 5 \
                --weight_decay 0.001 \
                --save_total_limit 50 \
                --seed 1 \
                --label_names index \
                --data_root ./data \
                --lr_scheduler_type cosine \
                --recompute
```

可配置参数说明：
* `do_train` 是否进行微调训练，设置该参数表示进行微调训练。
* `train_data` 必须，训练集路径
* `val_data` 必须，验证集路径
* `learning_rate` 训练的学习率
* `warmup_steps` warmup的step数
* `logging_steps` 训练过程中日志打印的间隔 steps 数
* `per_device_train_batch_size` 训练集训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为128。
* `dataloader_num_workers` Dataloader的num_worker的数目
* `save_steps` 训练过程中保存模型 checkpoint 的间隔 steps 数，默认50。
* `num_train_epochs` 训练的epoch数目。
* `weight_decay` 除了所有 bias 和 LayerNorm 权重之外，应用于所有层的权重衰减数值。可选；默认为 0.0；
* `save_total_limit` 保存checkpoints的数目，默认-1，表示不设限制。
* `seed` 随机种子，用于固定模型训练的随机因素。
* `label_names`训练集中标签对应的的 key 名称。如果不传入，在训练时 Trainer 可能由于无法区分输入数据和标签造成错误。
* `data_root` 数据集的根目录路径。
* `lr_scheduler_type` 学习率变化的类型，支持linear,cosine,constant等。
* `recompute` 节省缓存的策略，是一种以时间换空间的技术。

<a name="模型评估"></a>

## 模型评估

### 提取特征

```
DATAPATH=./data

split=valid # 指定计算valid或test集特征
python -u extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/Flickr30k-CN/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/Flickr30k-CN/${split}_texts.jsonl" \
    --resume output_pd/checkpoint-600 \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52
```
可配置参数说明：
* `extract-image-feats` 是否进行图像特征提取。
* `extract-image-feats` 是否进行文本特征提取。
* `image-data` 图像数据的地址
* `text-data` 文本数据的地址
* `resume` checkpoints的加载地址
* `img-batch-size` 图像特征提取的batch size
* `text-batch-size` 文本特征提取的batch size
* `context-length` 文本序列的最大长度

### 以文搜图评估


```shell
DATAPATH=./data
dataset_name=Flickr30k-CN
split=valid # 指定计算valid或test集特征

python -u utils/make_topk_predictions.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl"

python utils/evaluation.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl \
    output.json
cat output.json

```
运行结束后会有如下的输出：

```
{"success": true, "score": 86.64, "scoreJson": {"score": 86.64, "mean_recall": 86.64, "r1": 72.42, "r5": 91.74, "r10": 95.76}}
```

### 以图搜文评估

```
DATAPATH=./data
dataset_name=Flickr30k-CN

split=valid # 指定计算valid或test集特征
python -u utils/make_topk_predictions_tr.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl"

python utils/transform_ir_annotation_to_tr.py \
    --input ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl

split=valid # 指定计算valid或test集特征
python utils/evaluation_tr.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.tr.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl \
    output.json
cat output.json
```
运行结束后会有如下的输出：

```
{"success": true, "score": 95.36666666666666, "scoreJson": {"score": 95.36666666666666, "mean_recall": 95.36666666666666, "r1": 88.8, "r5": 97.89999999999999, "r10": 99.4}}
```



<a name="模型预测"></a>

## 模型预测

给定一张图：

![000000039769](https://user-images.githubusercontent.com/12107462/212855663-c0a54707-e14c-4450-b45d-0162ae76aeb8.jpeg)

把图像下载下来放到 `examples`目录。然后给定文本：

```
["猫的照片", "狗的照片"]
```

运行如下的命令，计算图像和文本的相似度：

```
python test_example.py --resume output_pd/checkpoint-600/
```
运行结束以后会有如下的输出：

```
......
         -0.16923490,  0.51063180,  0.50393420, -0.67453980,  0.26532388,
         -0.09257773, -1.26810551, -0.09288968,  2.01344800, -0.94694304,
          0.47568727, -0.59764022, -0.54964906,  0.25573757,  0.74198186,
         -0.50854760, -1.16371655,  0.86338192, -0.25785092, -0.20684876,
         -0.37839428,  0.04234251,  0.63441759]])
Text features
Tensor(shape=[2, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[ 0.06673798, -0.40317190,  0.28922316, ...,  0.26859814,
          0.32744539,  0.19407970],
        [ 0.11071929, -0.38126343,  0.17159876, ...,  0.21692061,
          0.33863425,  0.32423186]])
Label probs: Tensor(shape=[1, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[0.99927372, 0.00072624]])
```
可以看到`猫的照片`的相似度更高，结果符合预期。

<a name="参考文献"></a>

## 参考文献
* Bin Shan, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang: ERNIE-ViL 2.0: Multi-view Contrastive Learning for Image-Text Pre-training. CoRR abs/2209.15270 (2022)
