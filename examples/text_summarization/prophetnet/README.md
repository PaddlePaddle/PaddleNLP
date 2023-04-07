# Prophetnet

## 模型简介

ProphetNet（先知网络）是一种新型的 seq2seq 预训练模型。在训练时，Prophetnet 每一时刻将会学习同时预测未来的 N 个字符，这种自监督学习目标可以使得模型考虑未来更远的字符，防止模型对强局部相关（strong
local correlation）过拟合。

本项目是 Prophetnet 在 PaddlePaddle 2.4 上开源实现的文本摘要的例子，包含了在 CNN/DailyMail 数据集，Gigaword 数据集上微调和生成的代码。

### 项目依赖

```
pip install -r requirements.txt
python -m pip install paddlepaddle-gpu==2.4.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install paddlenlp==2.5.2
```

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
├── train_prophetnet.py # 模型finetune主程序入口
├── generate.py # 模型生成主程序入口
├── eval.py # 生成结果评估入口
├── uncase_tokenize_data.py # 数据预处理
├── uncompress_data.sh # 数据解压脚本
├── run_train.sh # 模型训练脚本
├── run_eval.sh # 模型评估脚本
├── requirements.txt # 环境依赖文件
└── README.md # 文档说明
```

### 数据准备

GLGE 数据集下载：[链接](https://drive.google.com/file/d/1F4zppa9Gqrh6iNyVsZJkxfbm5waalqEA/view)

GLGE 测试集下载：[链接](https://drive.google.com/file/d/11lDXIG87dChIfukq3x2Wx4r5_duCRm_J/view)

将glge_public.tar与glge_hidden_v1.1.tar.gz放入到项目根目录下。

```
bash uncompress_data.sh
```

### 数据预处理

```
python uncase_tokenize_data.py --dataset <DATASET>
```

说明：

- `<DATASET>`可选`cnndm`, `gigaword`.

### 模型训练

```
bash run_train.sh <DATASET>
```

或直接运行finetune程序

- cnndm:

```
python -m paddle.distributed.launch --gpus 0 python train_prophetnet.py \
    --dataset=cnndm \
    --model_name_or_path=prophetnet-large-uncased \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=8 \
    --num_train_epochs=4 \
    --learning_rate=0.0001 \
    --warmup_init_lr=1e-07 \
    --warmup_steps=1000 \
    --max_grad_norm=0.1 \
    --dataloader_num_workers=4 \
    --logging_steps 10 \
    --save_steps 100 \
    --do_train \
    --do_eval \
    --output_dir=./ckpt/cnndm
```

- gigaword:

```
python -m paddle.distributed.launch --gpus 0 python train_prophetnet.py \
    --dataset=gigaword \
    --model_name_or_path=prophetnet-large-uncased \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=32 \
    --num_train_epochs=6 \
    --learning_rate=0.0001 \
    --warmup_init_lr=1e-07 \
    --warmup_steps=1000 \
    --max_grad_norm=0.1 \
    --dataloader_num_workers=8 \
    --logging_steps 10 \
    --save_steps 100 \
    --do_train \
    --do_eval \
    --output_dir=./ckpt/gigaword
```

其中参数释义如下：

- `dataset` 指定数据集，可选cnndm和gigaword

- `model_name_or_path` 预训练模型名称或本地预训练模型初始化权重文件路径

- `per_device_train_batch_size` 表示单卡训练样本批大小

- `per_device_eval_batch_size` 表示单卡验证样本批大小

- `num_train_epochs` 表示训练轮数

- `learning_rate` 表示学习率

- `warmup_init_lr` 表示预热学习率

- `warmup_steps` 表示预热学习步数

- `max_grad_norm` 表示梯度裁剪

- `dataloader_num_workers` 指定数据加载规模

- `logging_steps` 表示打印结果间隔

- `save_steps`表示验证间隔

- `do_train` 表示是否训练

- `do_eval` 表示是否验证

- `output_idr` 指定微调结果权重存放路径

已经finetune好的模型权重：

- cnndm : [链接](https://pan.baidu.com/s/1cemrUDxkqEW9raoasJ_VKw), 提取码：1egi

- gigaword : [链接](https://pan.baidu.com/s/1qRH2FStT3vNQtDjZLkYJBQ), 提取码：on5v

### 模型评估

使用prophetNet源码的[评估脚本](https://pan.baidu.com/s/1FOnd01rNvDJoONYegacq1Q), 此脚本依赖于pyrouge，需要提前安装rouge。

```
pip install git+https://github.com/pltrdy/pyrouge
```

```
bash run_eval.sh <DATASET>
```

或直接运行模型生成程序

- cnndm:

```
python generate.py \
    --dataset=cnndm \
    --model_name_or_path=prophetnet-large-uncased \
    --output_path=./generate/cnndm/generate.txt \
    --min_target_length=45 \
    --max_target_length=110 \
    --decode_strategy=beam_search \
    --num_beams=4 \
    --length_penalty=1.2 \
    --batch_size=16 \
    --ignore_pad_token_for_loss=True \
    --early_stopping=True \
    --logging_steps=100 \
    --device=gpu

python eval.py --dataset cnndm --generated ./generate/cnndm/generate.txt
```

- gigaword:

```
python generate.py \
    --dataset=gigaword \
    --model_name_or_path=prophetnet-large-uncased \
    --output_path=./generate/gigaword/generate.txt \
    --min_target_length=1 \
    --max_target_length=200 \
    --decode_strategy=beam_search \
    --num_beams=4 \
    --length_penalty=1.6 \
    --batch_size=16 \
    --ignore_pad_token_for_loss=True \
    --early_stopping=True \
    --logging_steps=100 \
    --device=gpu

python eval.py --dataset gigaword --generated ./generate/gigaword/generate.txt
```

其中参数释义如下：

- `dataset` 指定数据集，可选cnndm和gigaword

- `vocab_file` 指定词表文件

- `output_path` 指定生成结果存放路径

- `min_target_length` 指定解码最短长度

- `max_target_length` 指定解码最大长度

- `decode_strategy` 指定解码策略

- `num_beams` 指定beam_search解码宽度

- `length_penalty` 指定beam_search解码的长度指数惩罚

- `batch_size` 指定评估样本批大小

- `ignore_pad_token_for_loss` 表示计算loss时忽略padding

- `early_stopping` 指定生成结束符是否停止预测

- `logging_steps` 指定日志打印间隔

- `device` 指定使用设备

### 微调测试精度

> #### 在CNN/DM数据集的测试效果如下表。

|网络 |opt|batch_size|数据集|ROUGE_1|ROUGE_2|ROUGE_L|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|prophetnet-large-uncased|Adam|4|CNN/DM|44.17|21.24|41.36|

> #### 在gigaword数据集的测试效果如下表。

|网络 |opt|batch_size|数据集|ROUGE_1|ROUGE_2|ROUGE_L|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|prophetnet-large-uncased|Adam|16|gigaword|38.92|19.81|36.06|

### 实验环境

- GPU RTX3090 * 1, CPU Intel i7-11700k
- Ubuntu 18.04

### 参考文献

1. Qi W, Yan Y, Gong Y, et al. Prophetnet: Predicting future n-gram for sequence-to-sequence pre-training[J]. arXiv
   preprint arXiv:2001.04063, 2020.
