[中文版](README_cn.md)

This project is the implementation of [XLNet](https://github.com/zihangdai/xlnet) on Paddle Fluid, currently supporting the fine-tuning on all downstream tasks, including natural language inference, question answering (SQuAD) etc.

There are a lot differences between XLNet and [BERT](../BERT). XLNet takes adavangtage of a new novel model [Transformer-XL](https://arxiv.org/abs/1901.02860) as the backbone of language representation, and the permutation language modeling as the optimizing objective. Also XLNet involed much more data in the pre-training stage. Finally, XLNet achieved SOTA results on several NLP tasks.

For more details, please refer to the research paper

[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)

## Directory structure

```
├── model/                        # directory for model structure definition
│   ├── classifier.py             # model for regression/classification
│   ├── xlnet.py                  # model for XLNet
├── reader/                       # directory for data reader
│   ├── cls.py                    # data reader for regression/classification
│   ├── squad.py                  # data reader for squad
├── utils/                        # directory for utility files
│── modeling.py                   # network modules
│── optimization.py               # optimization method
│── run_classifier.py             # script for running regression/classification task
│── run_squad.py                  # script for running squad
```

## Installation

This project requires Paddle Fluid version 1.6.0 and later, please follow the [installation guide](https://www.paddlepaddle.org.cn/start) to install.  

## Pre-trained models

Two pre-trained models converted from the official release are available

| Model | Layers | Hidden size | Heads |
| :------| :------: | :------: |:------: |
| [XLNet-Large, Cased](https://xlnet.bj.bcebos.com/xlnet_cased_L-24_H-1024_A-16.tgz)| 24 | 1024 | 16 |
| [XLNet-Base, Cased](https://xlnet.bj.bcebos.com/xlnet_cased_L-12_H-768_A-12.tgz)| 12 | 768 | 12 |

Each compressed package contains one subdirectory and two files:

- `params`: a directory consisting of all converted parameters, one file for a parameter.
- `spiece.model`: a [Sentence Piece](https://github.com/google/sentencepiece) model used for (de)tokenization.
- `xlnet_config.json`: a config file which specifies the hyperparameters of the model.

## Fine-tuning with XLNet

We provide the scripts for fine-tuning on NLP tasks with XLNet on multi-card GPUs. And their correctness has been verified that all experiments on V100 GPUs can achieve the same performance as the officially reported (mainly on TPU). In the following statements, we assume that the two pre-trained models have been downloaded and extracted.

### Text regression/classification

The fine-tuning of regression/classification can be preformed via the script `run_classifier.py` , which contains examples for standard one-document classification, one-document regression, and document pair classification. The two examples, one for regression and another one for classification can go on in the following way.

#### (1) STS-B: sentence pair relevance regression

- Download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory $GLUE_DIR

  - **Note**: You may meet the error `ImportError: No module named request` when running the script under Python 2.x, this is because the module `urllib` doesn't have submodule `request`. It can be resolved by replacing all the code `urllib.request` with `urllib` or changing to a Python 3.x environment.

- Perform fine-tuning on 4 V100 GPUs with XLNet-Large

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
This configuration doesn't require that large GPU memory, and 4 V100 (or other) GPUs with 16GB should be enough.

As the fine-tuning finished, the evaluation result on dev dataset, including the average loss and pearson correlation coefficient, will yield

```
[dev evaluation] ave loss: 0.383523, eval_pearsonr: 0.916912, elapsed time: 21.804057 s
```

The expected `eval_pearsonr` is `91.3+`, quoted from the official repository, and the experiment can reproduce this performance.

#### (2) IMDB: movie review sentiment classification

- Download and unpack the IMDB dataset by running

```shell
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar zxvf aclImdb_v1.tar.gz
```

- Perform fine-tuning with XLNet-Large on 8 V100 GPUs (32GB) by running

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

The expected accuracy is `96.2+`, and here is an example of evaluation result

```
[dev evaluation] ave loss: 0.220047, eval_accuracy: 0.963480, elapsed time: 2799.974465 s
```

Other NLP regression/classification tasks' fine-tuning can be carried out in the similar way.

### SQuAD 2.0

- Download SQuAD2.0 data and put it in the `data/squad2.0` directory

```
mkdir -p data/squad2.0
wget -P data/squad2.0 https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -P data/squad2.0 https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```

- Perform fine-tuning running the script `run_squad.py` on V100 GPUs (32GB)

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

And the final evaluation result after fine-tuning should looks like

```
================================================================================
Result | best_f1 88.0893932758 | best_exact_thresh -2.07637166977 | best_exact 85.5049271456 | has_ans_f1 0.940979062625 | has_ans_exact 0.880566801619 | best_f1_thresh -2.07337403297 |
================================================================================
```

### Use your own data

Please refer to the data-format guidelines of GLUE/SQuAD if you want to use your own data for fine-tuning.


## Acknowledgement

We thank the distiguished work done by the authors of XLNet!
