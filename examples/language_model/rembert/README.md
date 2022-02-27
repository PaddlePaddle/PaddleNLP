# RemBert with PaddleNLP

[RemBERT: Rethinking embedding coupling in pre-trained language models](https://arxiv.org/pdf/2010.12821v1.pdf)

**模型简介：**
作者发现，分离词嵌入为建模语言模型提供更好的灵活性，使我们能够显著提高多语言模型输入词嵌入中参数分
配的效率。通过在transformers层中重新分配输入词嵌入参数，在微调过程中，相比于具有相同数量参数量的
自然语言模型在自然语言理解任务上获得了更好的性能。作者还发现，增大输出词嵌入维度可以提升模型的性能，
即使在预训练结束后，输出词嵌入被丢弃，该模型仍能在微调阶段保持不变。作者分析表明，增大输出词嵌入维度
可以防止模型在预训练数据集上过拟合，并让模型在其他NLP数据集上有更强的泛化能力。利用这些发现，我们能够
训练性能更强大的模型，而无需在微调阶段增加参数。

## 快速开始

### 下游任务微调

####数据集
下载XTREME-XNLI数据集:
训练集:[下载地址](https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip)
测试集:[下载地址](https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip)
其中训练集为位于`XNLI-MT-1.0/multinli/multinli.train.en.tsv`, 测试集位于`XNLI-1.0/xnli.test.tsv`

下载XTREME-PAWS-X数据集：
[下载地址](https://storage.googleapis.com/paws/pawsx/x-final.tar.gz)
每个训练集、验证集和测试集分别为`train`、`dev`和`test`开头的`tsv`文件, 将所有语言的数据集解压后，请合并所有语言测试集到一个文件(此任务需要在多语言进行测试)

#### 1、XTREME-XNLI
XTREME-XNLI数据集为例：
运行以下两个命令即可训练并评估RemBert在XTREME-XNLI数据集的精度

```shell
python -m paddle.distributed.launch examples/language_model/rembert/main.py \
    --model_type rembert \
    --data_dir data/
    --output_dir output/ \
    --device gpu
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --train_batch_size 16 \
    --do_train \
    --do_eval \
    --task xnli \
    --eval_step 500
```
其中参数释义如下：
- `model_type` 指示了模型类型，当前支持`rembert`
- `data_dir` 数据集路径。
- `train_batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `output_dir` 表示模型保存路径。
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用多GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。
- `num_train_epochs` 表示需要训练的epoch数量
- `do_train` 表示是否开启训练
- `do_eval` 表示是否开启评估
- `task` 表示训练的任务
- `eval_step` 表示训练多少步评估一次模型

训练结束后模型会对模型进行评估，训练完成后你将看到如下结果:
```bash
Accuracy 0.8089
```

#### 2、XTREME-PAWS-X
在此数据集训练使用如下命令：

```shell
python -m paddle.distributed.launch examples/language_model/rembert/main.py \
    --model_type rembert \
    --data_dir data/
    --output_dir output/ \
    --device gpu
    --learning_rate 8e-6 \
    --num_train_epochs 3 \
    --train_batch_size 16 \
    --do_train \
    --do_eval \
    --task paws \
    --eval_step 500
```
训练结束后模型会对模型进行评估，其评估在测试集上完成, 训练完成后你将看到如下结果:
```bash
Accuracy 0.8778
```


# Reference

```bibtex
@article{chung2020rethinking,
  title={Rethinking embedding coupling in pre-trained language models},
  author={Chung, Hyung Won and Fevry, Thibault and Tsai, Henry and Johnson, Melvin and Ruder, Sebastian},
  journal={arXiv preprint arXiv:2010.12821},
  year={2020}
}
```
