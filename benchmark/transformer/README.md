# Transformer Benchmark with Fleet API

## Transformer


## 模型简介

机器翻译（machine translation, MT）是利用计算机将一种自然语言(源语言)转换为另一种自然语言(目标语言)的过程，输入为源语言句子，输出为相应的目标语言的句子。

本项目是机器翻译领域主流模型 Transformer 的 PaddlePaddle 实现， 包含模型训练，预测以及使用自定义数据等内容。用户可以基于发布的内容搭建自己的翻译模型。


## 快速开始

### 安装说明

1. paddle安装

    本项目依赖于 PaddlePaddle 2.0 及以上版本或适当的develop版本，请参考 [安装指南](https://www.paddlepaddle.org.cn/install/quick) 进行安装

2. 下载代码

    克隆代码库到本地

3. 环境依赖

    该模型使用PaddlePaddle，关于环境依赖部分，请先参考PaddlePaddle[安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)关于环境依赖部分的内容。
    此外，需要另外涉及：
      * attrdict
      * pyyaml



### 数据准备

公开数据集：WMT 翻译大赛是机器翻译领域最具权威的国际评测大赛，其中英德翻译任务提供了一个中等规模的数据集，这个数据集是较多论文中使用的数据集，也是 Transformer 论文中用到的一个数据集。我们也将[WMT'14 EN-DE 数据集](http://www.statmt.org/wmt14/translation-task.html)作为示例提供。

同时，我们提供了一份已经处理好的数据集，可以编写如下代码，对应的数据集将会自动下载并且解压到 `~/.paddlenlp/datasets/machine_translation/WMT14ende/`。这部分已经在 reader.py 中有写明，若无自行修改可以无需编写相应代码。

``` python
datasets = load_dataset('wmt14ende', data_files=data_files, splits=('train', 'dev'))
```

### 单机训练

### 单机单卡

以提供的英德翻译数据为例，可以执行以下命令进行模型训练：

#### 静态图
如果是需要单机单卡训练，则使用下面的命令进行训练：
``` shell
cd static/
export CUDA_VISIBLE_DEVICES=0
python3 train.py --config ../configs/transformer.base.yaml
```

需要注意的是，单卡下的超参设置与多卡下的超参设置有些不同，单卡执行需要修改 `configs/transformer.big.yaml` 或是 `configs/transformer.base.yaml` 中：
* `warmup_steps` 参数为 `16000`。
* `is_distributed` 参数为 `False`。

#### 动态图
如果使用单机单卡进行训练可以使用如下命令：
``` shell
cd dygraph/
export CUDA_VISIBLE_DEVICES=0
python3 train.py --config ../configs/transformer.base.yaml
```

需要注意的是，单卡下的超参设置与多卡下的超参设置有些不同，单卡执行需要修改 `configs/transformer.big.yaml` 或是 `configs/transformer.base.yaml` 中：
* `warmup_steps` 参数为 `16000`。
* `is_distributed` 参数为 `False`。

### 单机多卡

同样，可以执行如下命令实现八卡训练：

#### 静态图

如果是需要单机多卡训练，则使用下面的命令进行训练：
##### PE 的方式启动单机多卡：
``` shell
cd static/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 train.py --config ../configs/transformer.base.yaml
```

使用 PE 的方式启动单机多卡需要设置 `configs/transformer.big.yaml` 或是 `configs/transformer.base.yaml` 中 `is_distributed` 参数为 `False`。

##### fleet 的方式启动单机多卡：
``` shell
cd static/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" train.py --config ../configs/transformer.base.yaml
```

使用 fleet 的方式启动单机多卡需要设置 `configs/transformer.big.yaml` 或是 `configs/transformer.base.yaml` 中 `is_distributed` 参数为 `True`。

#### 动态图
如果使用单机多卡进行训练可以使用如下命令：
``` shell
cd dygraph/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train.py --config ../configs/transformer.base.yaml
```

### 模型推断

以英德翻译数据为例，模型训练完成后可以执行以下命令对指定文件中的文本进行翻译：

#### 静态图
``` sh
# setting visible devices for prediction
cd static/
export CUDA_VISIBLE_DEVICES=0
python3 predict.py --config ../configs/transformer.base.yaml
```

 由 `predict_file` 指定的文件中文本的翻译结果会输出到 `output_file` 指定的文件。执行预测时需要设置 `init_from_params` 来给出模型所在目录，更多参数的使用可以在 `configs/transformer.big.yaml` 和 `configs/transformer.base.yaml` 文件中查阅注释说明并进行更改设置。如果执行不提供 `--config` 选项，程序将默认使用 big model 的配置。

 需要注意的是，目前预测仅实现了单卡的预测，原因在于，翻译后面需要的模型评估依赖于预测结果写入文件顺序，多卡情况下，目前暂未支持将结果按照指定顺序写入文件。

#### 动态图
``` sh
# setting visible devices for prediction
cd dygraph/
export CUDA_VISIBLE_DEVICES=0
python3 predict.py --config ../configs/transformer.base.yaml
```

 由 `predict_file` 指定的文件中文本的翻译结果会输出到 `output_file` 指定的文件。执行预测时需要设置 `init_from_params` 来给出模型所在目录，更多参数的使用可以在 `configs/transformer.big.yaml` 和 `configs/transformer.base.yaml` 文件中查阅注释说明并进行更改设置。如果执行不提供 `--config` 选项，程序将默认使用 big model 的配置。

 需要注意的是，目前预测仅实现了单卡的预测，原因在于，翻译后面需要的模型评估依赖于预测结果写入文件顺序，多卡情况下，目前暂未支持将结果按照指定顺序写入文件。


### 模型评估

预测结果中每行输出是对应行输入的得分最高的翻译，对于使用 BPE 的数据，预测出的翻译结果也将是 BPE 表示的数据，要还原成原始的数据（这里指 tokenize 后的数据）才能进行正确的评估。评估过程具体如下（BLEU 是翻译任务常用的自动评估方法指标）：

``` sh
# 还原 predict.txt 中的预测结果为 tokenize 后的数据
sed -r 's/(@@ )|(@@ ?$)//g' predict.txt > predict.tok.txt
# 若无 BLEU 评估工具，需先进行下载
git clone https://github.com/moses-smt/mosesdecoder.git
# 以英德翻译 newstest2014 测试数据为例
perl mosesdecoder/scripts/generic/multi-bleu.perl ~/.paddlenlp/datasets/machine_translation/WMT14ende/WMT14.en-de/wmt14_ende_data/newstest2014.tok.de < predict.tok.txt
```

执行上述操作之后，可以看到类似如下的结果，此处结果是 big model 在 newstest2014 上的 BLEU 结果：
```
BLEU = 27.48, 58.6/33.2/21.1/13.9 (BP=1.000, ratio=1.012, hyp_len=65312, ref_len=64506)
```

## FAQ

**Q:** 预测结果中样本数少于输入的样本数是什么原因  
**A:** 若样本中最大长度超过 `transformer.yaml` 中 `max_length` 的默认设置，请注意运行时增大 `--max_length` 的设置，否则超长样本将被过滤。

**Q:** 预测时最大长度超过了训练时的最大长度怎么办  
**A:** 由于训练时 `max_length` 的设置决定了保存模型 position encoding 的大小，若预测时长度超过 `max_length`，请调大该值，会重新生成更大的 position encoding 表。


## 参考文献
1. Vaswani A, Shazeer N, Parmar N, et al. [Attention is all you need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)[C]//Advances in Neural Information Processing Systems. 2017: 6000-6010.
2. Devlin J, Chang M W, Lee K, et al. [Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805)[J]. arXiv preprint arXiv:1810.04805, 2018.
3. He K, Zhang X, Ren S, et al. [Deep residual learning for image recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
4. Ba J L, Kiros J R, Hinton G E. [Layer normalization](https://arxiv.org/pdf/1607.06450.pdf)[J]. arXiv preprint arXiv:1607.06450, 2016.
5. Sennrich R, Haddow B, Birch A. [Neural machine translation of rare words with subword units](https://arxiv.org/pdf/1508.07909)[J]. arXiv preprint arXiv:1508.07909, 2015.
