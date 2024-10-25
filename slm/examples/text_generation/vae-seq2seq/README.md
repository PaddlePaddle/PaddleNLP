# Variational Autoencoder (VAE) for Text Generation

## 简介

本目录下此范例模型的实现，旨在展示如何用 Paddle 构建用于文本生成的 VAE 示例，其中 LSTM 作为编码器和解码器。分别对 PTB 数据集和 Yahoo Answer（采样100k）数据集进行训练。

关于 VAE 的详细介绍参照： [(Bowman et al., 2015) Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf)

## 数据介绍

本教程使用了两个文本数据集：

PTB 数据集由华尔街日报的文章组成，包含929k 个训练 tokens，词汇量为10k。下载地址为: [PTB](https://dataset.bj.bcebos.com/imikolov%2Fsimple-examples.tgz)。

Yahoo 数据集来自[(Yang et al., 2017) Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](https://arxiv.org/pdf/1702.08139.pdf)，该数据集从原始 Yahoo Answer 数据中采样100k 个文档，数据集的平均文档长度为78，词汇量为200k。下载地址为：[YahooAnswer100k](https://bj.bcebos.com/paddlenlp/datasets/yahoo-answer-100k.tar.gz)，运行本例程序后，数据集会自动下载到`~/.paddlenlp/datasets/YahooAnswer100k`目录下。


## 生成样例

shareholders were spent about N shares to spend $ N million to ual sell this trust stock last week

new york stock exchange composite trading trading outnumbered closed at $ N a share down N cents

the company cited pressure to pursue up existing facilities in the third quarter was for <unk> and four N million briefly stocks for so-called unusual liability

people had <unk> down out the kind of and much why your relationship are anyway

there are a historic investment giant chips which ran the <unk> benefit the attempting to original maker

使用请[参考](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/examples/text_generation/vae-seq2seq)
