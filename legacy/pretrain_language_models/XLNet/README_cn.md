[ENGLISH](README.md)

该项目是 [XLNet](https://github.com/zihangdai/xlnet) 基于 Paddle Fluid 的实现，目前支持该项目支持所有下游任务的 fine-tuning, 包括自然语言推断任务和阅读理解任务 （SQuAD2.0）等。

XLNet 与 [BERT](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/PaddleLARK/BERT) 有着许多的不同，XLNet 利用一个全新的模型 [Transformer-XL](https://arxiv.org/abs/1901.02860) 作为语义表示的骨架， 将置换语言模型的建模作为优化目标，同时在预训练阶段也利用了更多的数据。 最终，XLNet 在多个 NLP 任务上达到了 SOTA 的效果。

更多的细节，请参考学术论文

[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)

## 目录结构

```
├── model/                        # 模型结构定义目录
│   ├── classifier.py             # 回归/分类模型结构
│   ├── xlnet.py                  # XLNet 模型结构
├── reader/                       # 数据读取 reader 定义目录
│   ├── cls.py                    # 分类任务数据读取
│   ├── squad.py                  # squad 数据读取
├── utils/                        # 辅助文件目录
│── modeling.py                   # 网络定义模块
│── optimization.py               # 优化方法
│── run_classifier.py             # 运行回归/分类任务的脚本
│── run_squad.py                  # 运行 squad 任务的脚本
```

## 安装

该项目要求 Paddle Fluid 1.6.0 及以上版本，请参考 [安装指南](https://www.paddlepaddle.org.cn/start) 进行安装。

## 预训练模型

这里提供了从官方开源模型转换而来的两个预训练模型供下载

| Model | Layers | Hidden size | Heads |
| :------| :------: | :------: |:------: |
| [XLNet-Large, Cased](https://xlnet.bj.bcebos.com/xlnet_cased_L-24_H-1024_A-16.tgz)| 24 | 1024 | 16 |
| [XLNet-Base, Cased](https://xlnet.bj.bcebos.com/xlnet_cased_L-12_H-768_A-12.tgz)| 12 | 768 | 12 |

每个压缩包都包含了一个子文件夹和两个文件:

- `params`: 由参数构成的文件夹, 每个模型文件包含一个参数
- `spiece.model`: [Sentence Piece](https://github.com/google/sentencepiece) 模型，用于文本的（反）tokenization
- `xlnet_config.json`: 配置文件，指定了模型的超参数


## 利用 XLNet 进行 Fine-tuning

我们提供了利用 XLNet 在多卡 GPU 上为自然语言处理任务进行 fine-tuning 的脚本。通过基于 V100 GPU 进行实验，达到官方报告的效果 （主要是基于 TPU），这些脚本的正确性已得到过验证。在下面的陈述中，我们假定以上两个预训练已下载和解压好。

### 文本回归/分类任务

文本回归和分类任务的 fine-tuning 可以通过运行脚本 `run_classifier.py` 来进行，其中包含了单文本分类、单文本回归、文本对分类等示例。下面的两个例子，一个用于示例回归任务，另一个用于分类任务，可以按以下的方式进行 fine-tuning。

#### (1) STS-B: 句子对相关性回归

-  通过运行 [脚本](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) 下载 [GLUE 数据集](https://gluebenchmark.com/tasks)， 并解压到某个文件夹 $GLUE_DIR。

  - **请注意**: 在 Python 2.x 环境下运行这个脚本，可能会遇到报错 `ImportError: No module named request` , 这是因为模块 `urllib` 不包含子模块 `request`. 这个问题可以通过将脚本中的代码 `urllib.request` 全部替换为 `urllib`，或者在 Python 3.x 环境下运行予以解决。

- 使用 XLNet-Large 在 4 卡 V100 GPU 上进行 fine-tuning

```
export GLUE_DIR=glue_data
export LARGE_DIR=xlnet_cased_L-24_H-1024_A-16

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier.py \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --task_name=sts-b \
  --data_dir=${GLUE_DIR}/STS-B \
  --checkpoints=exp/sts-b \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=${LARGE_DIR}/xlnet_config.json \
  --init_pretraining_params=${LARGE_DIR}/params \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --learning_rate=5e-5 \
  --predict_dir=exp/sts-b-pred \
  --skip_steps=10 \
  --train_steps=1200 \
  --warmup_steps=120 \
  --save_steps=600 \
  --is_regression=True
```

该配置不需要特别大的 GPU 显存，16GB 的 4 卡 V100 （或其它 GPU）即可运行。

在 fine-tuning 结束后，会得到在 dev 数据集上的评估结果，包括平均误差和皮尔逊相关系数

```
[dev evaluation] ave loss: 0.383523, eval_pearsonr: 0.916912, elapsed time: 21.804057 s
```

按官方实现的说法，预期的 `eval_pearsonr` 是 `91.3+`，该实验应该能复现这个结果。

#### (2) IMDB: 电影评论情感分类

- 下载和解压 IMDB 数据集

```shell
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar zxvf aclImdb_v1.tar.gz
```

- 使用 XLNet-Large 在 8 卡 V100 GPU (32GB) 上进行 fine-tuning

```shell
export IMDB_DIR=aclImdb
export LARGE_DIR=xlnet_cased_L-24_H-1024_A-16

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_classifier.py \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --task_name=imdb \
  --checkpoints=exp/imdb \
  --init_pretraining_params=${LARGE_DIR}/params \
  --data_dir=${IMDB_DIR} \
  --predict_dir=predict_imdb_1028 \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=${LARGE_DIR}/xlnet_config.json \
  --max_seq_length=512 \
  --train_batch_size=4 \
  --eval_batch_size=8 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
```

期望的准确率是 `96.2+`， 以下是评估结果的一个样例

```
[dev evaluation] ave loss: 0.220047, eval_accuracy: 0.963480, elapsed time: 2799.974465 s
```

其它 NLP 回归/分类任务的 fine-tuning 可以通过同样的方式进行。

### SQuAD 2.0

- 下载 SQuAD 2.0 数据集并将其放入 `data/squad2.0` 目录中

```
mkdir -p data/squad2.0
wget -P data/squad2.0 https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -P data/squad2.0 https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```

- 在 6 卡 V100 GPU (32GB) 上运行脚本 `run_squad.py`

```
SQUAD_DIR=data/squad2.0
INIT_CKPT_DIR=xlnet_cased_L-24_H-1024_A-16
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python run_squad.py \
  --model_config_path=${INIT_CKPT_DIR}/xlnet_config.json \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --init_checkpoint=${INIT_CKPT_DIR}/params \
  --train_file=${SQUAD_DIR}/train-v2.0.json \
  --predict_file=${SQUAD_DIR}/dev-v2.0.json \
  --uncased=False \
  --checkpoints squad_2.0_0828 \
  --max_seq_length=512 \
  --do_train=True \
  --do_predict=True \
  --skip_steps=100 \
  --save_steps=10000 \
  --epoch 200 \
  --dropout=0.1 \
  --dropatt=0.1 \
  --train_batch_size=4 \
  --predict_batch_size=3 \
  --learning_rate=2e-5 \
  --save_steps=1000 \
  --train_steps=12000 \
  --warmup_steps=1000 \
  --verbose=True\
```

运行结束后的评测结果如下所示

```
================================================================================
Result | best_f1 88.0893932758 | best_exact_thresh -2.07637166977 | best_exact 85.5049271456 | has_ans_f1 0.940979062625 | has_ans_exact 0.880566801619 | best_f1_thresh -2.07337403297 |
================================================================================
```

### 使用自定义数据

如需使用自定义数据进行 fine-tuning，请参考 GLUE/SQuAD 的数据格式说明。

## 致谢

我们向 XLNet 的作者们所做的杰出工作致以谢意！
