# 向量检索模型训练

## 安装

推荐安装gpu版本的[PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)，以cuda11.7的paddle为例，安装命令如下：

```
python -m pip install paddlepaddle-gpu==2.6.0.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```
安装其他依赖：
```
pip install -r requirements.txt
```

下载DuReader-Retrieval中文数据集：

```
cd data
wget https://paddlenlp.bj.bcebos.com/datasets/dureader_dual.train.jsonl
```

## 运行

### 单卡训练

```
export CUDA_VISIBLE_DEVICES=0
python train.py --do_train \
              --model_name_or_path rocketqa-zh-base-query-encoder \
              --output_dir ./checkpoints \
              --train_data ./data/dureader_dual.train.jsonl \
              --overwrite_output_dir \
              --fine_tune_type sft \
              --sentence_pooling_method cls \
              --num_train_epochs 3 \
              --per_device_train_batch_size 64 \
              --learning_rate 3e-5 \
              --train_group_size 4 \
              --recompute \
              --passage_max_len 512 \
              --use_matryoshka
```

- `model_name_or_path`: 选择预训练模型，可选rocketqa-zh-base-query-encoder
- `output_dir`: 模型保存路径
- `train_data`: 训练数据集路径，这里使用的是dureader中文数据集
- `overwrite_output_dir`: 是否覆盖模型保存路径，默认为False
- `fine_tune_type`: 训练模式，可选sft和lora, bitfit等策略
- `sentence_pooling_method`: 句子池化方法，可选cls和mean, cls为CLS层，mean为平均池化
- `num_train_epochs`: 训练轮数
- `per_device_train_batch_size`: 单卡训练batch大小
- `learning_rate`: 学习率
- `train_group_size`: 每个训练集正负样本的数据，默认为8，例如train_group_size=4，则每个训练集包含1个正样本和3个负样本
- `max_example_num_per_dataset`: 每个训练集的最大样本数，默认为100000000
- `recompute`: 是否重新计算，默认为False
- `query_max_len`: query的最大长度，默认为32
- `query_instruction_for_retrieval`: query的检索指令，默认为None
- `passage_instruction_for_retrieval`: passage的检索指令，默认为None
- `passage_max_len`: passage的最大长度，默认为512
- `use_matryoshka`: 是否使用俄罗斯套娃策略（matryoshka），默认为False
- `matryoshka_dims`: 俄罗斯套娃策略的维度，默认为[64, 128, 256, 512, 768]
- `matryoshka_loss_weights`: 俄罗斯套娃策略的损失权重，默认为[1, 1, 1, 1, 1]
- `use_inbatch_neg`: 是否使用in batch negatives策略，默认为False
- `use_flash_attention`: 是否使用flash attention，默认为False
- `temperature`: in batch negatives策略的temperature参数，默认为0.02
- `negatives_cross_device`: 跨设备in batch negatives策略，默认为False
- `margin`: in batch negatives策略的margin参数，默认为0.2

### 多卡训练

单卡训练效率过低，batch_size较小，建议使用多卡训练，对于对比学习训练推荐使用大batch_size，多卡训练，示例命令如下：

```
python -m paddle.distributed.launch --gpus "1,2,3,4" train.py --do_train \
              --model_name_or_path rocketqa-zh-base-query-encoder \
              --output_dir ./checkpoints \
              --train_data ./data/dual.train.json \
              --overwrite_output_dir \
              --fine_tune_type sft \
              --sentence_pooling_method cls \
              --num_train_epochs 3 \
              --per_device_train_batch_size 32 \
              --learning_rate 3e-5 \
              --train_group_size 8 \
              --recompute \
              --passage_max_len 512 \
              --use_matryoshka
```

## 评估

评估脚本：

```
export CUDA_VISIBLE_DEVICES=0
python evaluation/benchmarks.py --model_type bert \
                              --query_model checkpoints/checkpoint-1500 \
                              --passage_model checkpoints/checkpoint-1500 \
                              --query_max_length 64 \
                              --passage_max_length 512 \
                              --evaluate_all
```
- `model_type`: 模型的类似，可选bert或roberta等等
- `query_model`: query向量模型的路径
- `passage_model`: passage向量模型的路径
- `query_max_length`: query的最大长度
- `passage_max_length`: passage的最大长度
- `evaluate_all`: 是否评估所有的checkpoint，默认为False，即只评估指定的checkpoint

## Reference

[1] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham M. Kakade, Prateek Jain, Ali Farhadi: Matryoshka Representation Learning. NeurIPS 2022
