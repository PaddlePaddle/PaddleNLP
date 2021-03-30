# ERNIE-CTM

## 简介

ERNIE for **C**hinese **T**ext **M**ining，适用于中文挖掘任务的预训练语言模型，包含全中文词表、中英文标点、单位符号、汉语拼音等token，大幅减少tokenize中UNK的情况，对中文标注、挖掘类任务有较大增益。

## 快速开始

### 模型训练

模型训练支持 CPU 和 GPU，使用 GPU 之前应指定使用的显卡卡号：

```bash
export CUDA_VISIBLE_DEVICES=0 # 支持多卡训练，如使用双卡，可以设置为0,1
```

训练启动方式如下：

```bash
python -u ./train.py \
    --max_seq_len 128 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 1 \
    --output_dir ./tmp/ \
    --n_gpu 1 \
```

### 模型评估

通过加载训练保存的模型，可以对测试集数据进行验证，启动方式如下：

```bash
python -u ./eval.py \
    --max_seq_len 128 \
    --batch_size 32   \
    --model_dir ./tmp/ernie_ctm_ft_model_1.pdparams \
    --n_gpu 1 \
```

其中 model_dir 是模型加载路径。

### 模型预测

对无标签数据可以启动模型预测：

```bash
python -u ./predict.py \
    --max_seq_len 128 \
    --batch_size 32   \
    --model_dir ./tmp/ernie_ctm_ft_model_1.pdparams \
    --n_gpu 1 \
```
