# 使用Seq2Seq模型完成自动对联

以下是本范例模型的简要目录结构及说明：

```
.
├── README.md              # 文档，本文件
├── args.py                # 训练、预测以及模型参数配置程序
├── data.py                # 数据读入程序
├── train.py               # 训练主程序
├── predict.py             # 预测主程序
└── model.py               # 带注意力机制的对联生成程序
```

## 简介

Sequence to Sequence (Seq2Seq)，使用编码器-解码器（Encoder-Decoder）结构，用编码器将源序列编码成vector，再用解码器将该vector解码为目标序列。Seq2Seq 广泛应用于机器翻译，自动对话机器人，文档摘要自动生成，图片描述自动生成等任务中。

本目录包含Seq2Seq的一个经典样例：自动对联生成，带attention机制的文本生成模型。

运行本目录下的范例模型需要安装PaddlePaddle 2.0-rc1及以上版本。如果您的 PaddlePaddle 安装版本低于此要求，请按照[安装文档](https://www.paddlepaddle.org.cn/#quick-start)中的说明更新 PaddlePaddle 安装版本。


## 模型概览

本模型中，在编码器方面，我们采用了基于LSTM的多层的RNN encoder；在解码器方面，我们使用了带注意力（Attention）机制的RNN decoder，在预测时我们使用柱搜索（beam search）算法来生对联的下联。


## 数据介绍

本教程使用[couplet数据集](https://paddlenlp.bj.bcebos.com/datasets/couplet.tar.gz)数据集作为训练语料，train_src.tsv及train_tgt.tsv为训练集，dev_src.tsv及dev_tgt.tsv为开发集，test_src.tsv及test_tgt.tsv为测试集。

数据集会在`CoupletDataset`初始化时自动下载，如果用户在初始化数据集时没有提供路径，在linux系统下，数据集会自动下载到`~/.paddlenlp/datasets/machine_translation/CoupletDataset/`目录下


## 模型训练

执行以下命令即可训练带有注意力机制的Seq2Seq模型：

```sh
python train.py \
    --num_layers 2 \
    --hidden_size 512 \
    --batch_size 128 \
    --use_gpu True \
    --model_path ./couplet_models \
    --max_epoch 20
```

各参数的具体说明请参阅 `args.py` 。训练程序会在每个epoch训练结束之后，保存一次模型。

**NOTE:** 如需恢复模型训练，则`init_from_ckpt`只需指定到文件名即可，不需要添加文件尾缀。如`--init_from_ckpt=couplet_models/19`即可，程序会自动加载模型参数`couplet_models/19.pdparams`，也会自动加载优化器状态`couplet_models/19.pdopt`。

## 模型预测

训练完成之后，可以使用保存的模型（由 `--init_from_ckpt` 指定）对测试集进行beam search解码，命令如下：

```sh
python predict.py \
    --num_layers 2 \
    --hidden_size 512 \
    --batch_size 128 \
    --init_from_ckpt couplet_models/19 \
    --infer_output_file infer_output.txt \
    --beam_size 10 \
    --use_gpu True
```

各参数的具体说明请参阅 `args.py` ，注意预测时所用模型超参数需和训练时一致。

## 生成对联样例

上联：崖悬风雨骤                下联：月落水云寒

上联：约春章柳下                下联：邀月醉花间

上联：箬笠红尘外                下联：扁舟明月中

上联：书香醉倒窗前月        下联：烛影摇红梦里人

上联：踏雪寻梅求雅趣        下联：临风把酒觅知音

上联：未出南阳天下论        下联：先登北斗汉中书

上联：朱联妙语千秋颂        下联：赤胆忠心万代传

上联：月半举杯圆月下        下联：花间对酒醉花间

上联：挥笔如剑倚麓山豪气干云揽月去       下联：落笔似龙飞沧海龙吟破浪乘风来
