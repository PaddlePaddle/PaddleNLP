运行本目录下的范例模型需要安装PaddlePaddle Fluid 1.6版。如果您的 PaddlePaddle 安装版本低于此要求，请按照[安装文档](https://www.paddlepaddle.org.cn/#quick-start)中的说明更新 PaddlePaddle 安装版本。

# Variational Autoencoder (VAE) for Text Generation

以下是本范例模型的简要目录结构及说明：

```text
.
├── README.md         # 文档，本文件
├── args.py           # 训练、预测以及模型参数配置程序
├── reader.py         # 数据读入程序
├── download.py       # 数据下载程序
├── train.py          # 训练主程序
├── infer.py          # 预测主程序
├── run.sh            # 默认配置的启动脚本
├── infer.sh          # 默认配置的解码脚本
└── model.py                    # VAE模型配置程序

```

## 简介
本目录下此范例模型的实现，旨在展示如何用Paddle Fluid的 **<font color='red'>新Seq2Seq API</font>** 构建用于文本生成的VAE示例，其中LSTM作为编码器和解码器。 分别对官方PTB数据和SWDA数据进行培训。

关于VAE的详细介绍参照： [(Bowman et al., 2015) Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf)

## 数据介绍

本教程使用了两个文本数据集：

PTB dataset，原始下载地址为: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz。

SWDA dataset，来源于[Knowledge-Guided CVAE for dialog generation](https://arxiv.org/pdf/1703.10960.pdf)，原始数据集下载地址为：https://github.com/snakeztc/NeuralDialog-CVAE ，感谢作者@[snakeztc](https://github.com/snakeztc)。我们过滤了数据集中长度小于5的短句子。

### 数据获取

```
python download.py --task ptb/swda
```

## 模型训练

`run.sh`包含训练程序的主函数，要使用默认参数开始训练，只需要简单地执行：

```
sh run.sh ptb/swda
```

如果需要修改模型的参数设置，也可以通过下面命令配置：

```
python train.py \
        --vocab_size 10003 \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset_prefix data/${dataset}/${dataset} \
        --model_path ${dataset}_model\
        --use_gpu True \
        --max_epoch 50 \
```

训练程序采用了 Early Stopping，会在每个epoch根据ppl的表现来决定是否保存模型。

## 模型预测

当模型训练完成之后， 可以利用infer.sh的脚本进行预测，选择加载模型保存目录下的第 k 个epoch的模型进行预测，生成batch_size条短文本。

```
sh infer.sh ptb/swda k
```

如果需要修改模型预测输出的参数设置，也可以通过下面命令配置：

```
python infer.py \
        --vocab_size 10003 \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset_prefix data/${dataset}/${dataset} \
        --use_gpu True \
        --reload_model ${dataset}_model/epoch_${k} \
```

## 效果评价

```sh
PTB数据集:
Test PPL: 102.24
Test NLL: 108.22

SWDA数据集:
Test PPL: 64.21
Test NLL: 81.92
```

## 生成样例

the movie are discovered in the u.s. industry that on aircraft variations for a <unk> aircraft that was repaired

the percentage of treasury bonds rose to N N at N N up N N from the two days N N and a premium over N

he could n't plunge as a factor that attention now has n't picked up for the state according to mexico

take the remark we need to do then support for the market to tell it

i think that it believes the core of the company in its first quarter of heavy mid-october to fuel earnings after prices on friday
