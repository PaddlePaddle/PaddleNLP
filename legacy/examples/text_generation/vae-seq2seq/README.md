# Variational Autoencoder (VAE) for Text Generation
以下是本范例模型的简要目录结构及说明：

```text
.
├── README.md         # 文档
├── args.py           # 训练、预测以及模型参数配置程序
├── data.py           # 数据读入程序
├── train.py          # 训练主程序
├── predict.py        # 预测主程序
└── model.py          # VAE模型组网部分，以及Metric等
```

## 简介

本目录下此范例模型的实现，旨在展示如何用Paddle构建用于文本生成的VAE示例，其中LSTM作为编码器和解码器。分别对PTB数据集和Yahoo Answer（采样100k）数据集进行训练。

关于VAE的详细介绍参照： [(Bowman et al., 2015) Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf)

## 数据介绍

本教程使用了两个文本数据集：

PTB数据集由华尔街日报的文章组成，包含929k个训练tokens，词汇量为10k。下载地址为: [PTB](https://dataset.bj.bcebos.com/imikolov%2Fsimple-examples.tgz)。

Yahoo数据集来自[(Yang et al., 2017) Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](https://arxiv.org/pdf/1702.08139.pdf)，该数据集从原始Yahoo Answer数据中采样100k个文档，数据集的平均文档长度为78，词汇量为200k。下载地址为：[YahooAnswer100k](https://bj.bcebos.com/paddlenlp/datasets/yahoo-answer-100k.tar.gz)，运行本例程序后，数据集会自动下载到`~/.paddlenlp/datasets/YahooAnswer100k`目录下。


## 模型训练

如果使用ptb数据集训练，可以通过下面命令配置：

```
export CUDA_VISIBLE_DEVICES=0
python train.py \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset ptb \
        --model_path ptb_model\
        --device gpu \
        --max_epoch 50 \

```

如果需要多卡运行，可以运行如下命令：

```
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0,1,2,3" train.py \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset ptb \
        --model_path ptb_model \
        --device gpu \
        --max_epoch 50 \

```

如果需要使用yahoo数据集进行多卡运行，可以将参数配置如下：

```
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0,1,2,3" train.py \
        --batch_size 32 \
        --embed_dim 512 \
        --hidden_size 550 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset yahoo \
        --model_path yahoo_model \
        --device gpu \
        --max_epoch 50 \

```

**NOTE:** 如需恢复模型训练，则`init_from_ckpt`只需指定到文件名即可，不需要添加文件尾缀。如`--init_from_ckpt ptb_model/49`即可，程序会自动加载模型参数`ptb_model/49.pdparams`，也会自动加载优化器状态`ptb_model/49.pdopt`。


## 模型预测

当模型训练完成之后，可以选择加载模型保存目录下的第 50 个epoch的模型进行预测，生成batch_size条短文本。生成的文本位于参数`infer_output_file`指定的路径下。如果使用ptb数据集，可以通过下面命令配置：

```
export CUDA_VISIBLE_DEVICES=0
python predict.py \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset ptb \
        --device gpu \
        --infer_output_file infer_output.txt \
        --init_from_ckpt ptb_model/49 \

```

使用yahoo数据集，需要配置embed_dim和hidden_size：

```
python predict.py \
        --batch_size 32 \
        --init_scale 0.1 \
        --embed_dim 512 \
        --hidden_size 550 \
        --max_grad_norm 5.0 \
        --dataset yahoo \
        --device gpu \
        --infer_output_file infer_output.txt \
        --init_from_ckpt yahoo_model/49 \

```

## 效果评价

||Test PPL|Test NLL|
|:-|:-:|:-:|
|ptb dataset|108.71|102.76|
|yahoo dataset|78.38|349.48|


## 生成样例

shareholders were spent about N shares to spend $ N million to ual sell this trust stock last week

new york stock exchange composite trading trading outnumbered closed at $ N a share down N cents

the company cited pressure to pursue up existing facilities in the third quarter was for <unk> and four N million briefly stocks for so-called unusual liability

people had <unk> down out the kind of and much why your relationship are anyway

there are a historic investment giant chips which ran the <unk> benefit the attempting to original maker
