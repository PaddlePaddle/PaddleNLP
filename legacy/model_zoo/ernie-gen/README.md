# ERNIE-Gen: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation

## 1. 简介

ERNIE-GEN 是面向生成任务的预训练-微调框架，首次在预训练阶段加入**span-by-span 生成任务**，让模型每次能够生成一个语义完整的片段。在预训练和微调中通过**填充式生成机制**和**噪声感知机制**来缓解曝光偏差问题。此外, ERNIE-GEN 采样**多片段-多粒度目标文本采样策略**, 增强源文本和目标文本的关联性，加强了编码器和解码器的交互。

![multi-flow-attention](https://github.com/PaddlePaddle/ERNIE/raw/repro/ernie-gen/.meta/multi-flow-attention.png)

## 快速开始

### 环境依赖

- tqdm

安装方式：`pip install tqdm`

### 数据准备

在本例中，我们提供了古诗词数据集，示例数据如下：

```text
画\002精\002禅\002室\002冷\002，\002方\002暑\002久\002徘\002徊\002。	不\002尽\002林\002端\002雪\002，\002长\002青\002石\002上\002苔\002。\002心\002闲\002对\002岩\002岫\002，\002目\002浄\002失\002尘\002埃\002。\002坐\002久\002清\002风\002至\002，\002疑\002从\002翠\002涧\002来\002。
```

每行数据都是由两列组成，以制表符分隔。第一列是输入的诗句前文，第二列是输出的诗句后文，所有文字都以 `\002` 分隔。

完整数据集可以通过以下命令下载并解压：

```bash
wget --no-check-certificate https://bj.bcebos.com/paddlenlp/datasets/poetry.tar.gz
tar xvf poetry.tar.gz
```

### 模型微调

#### 单卡训练

训练启动方式如下：

```bash
python -u ./train.py \
    --model_name_or_path ernie-1.0 \
    --max_encode_len 24 \
    --max_decode_len 72 \
    --batch_size 48  \
    --learning_rate 2e-5 \
    --num_epochs 12 \
    --logging_steps 1 \
    --save_steps 1000 \
    --output_dir ./tmp/ \
    --device gpu \
    # --init_checkpoint ./tmp/model_10000/model_state.pdparams
```

参数释义如下：
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer，支持[PaddleNLP Transformer类预训练模型](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer) 中的所有模型，但只有`ernie-gen-base-en, ernie-gen-large-en, ernie-gen-large-en-430g`三种模型会加载最后输出层的参数，其余模型只会加载transformer参数作热启动。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `max_encode_len` 表示最大输入句子长度，超过该长度将被截断。
- `max_decode_len` 表示最大输出句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device`: 训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `init_checkpoint` 表示模型加载路径，通过设置此参数可以开启增量训练。

训练会持续很长的时间，为此我们提供了[微调后的模型](https://bj.bcebos.com/paddlenlp/models/transformers/ernie_gen_finetuned/ernie_1.0_poetry.pdparams)。您可以下载该模型并通过`init_checkpoint`加载其参数进行增量训练、评估或预测。

#### 多卡训练

训练启动方式如下：

```bash
python -m paddle.distributed.launch --gpus "0,1" ./train.py \
    --model_name_or_path ernie-1.0 \
    --max_encode_len 24 \
    --max_decode_len 72 \
    --batch_size 48  \
    --learning_rate 2e-5 \
    --num_epochs 12 \
    --logging_steps 1 \
    --save_steps 1000 \
    --output_dir ./tmp/ \
    --device gpu \
    # --init_checkpoint ./tmp/model_10000/model_state.pdparams
```

### 模型评估

通过加载训练保存的模型，可以对验证集数据进行验证，启动方式如下：

```bash
python -u ./eval.py \
    --model_name_or_path ernie-1.0 \
    --max_encode_len 24 \
    --max_decode_len 72 \
    --batch_size 48   \
    --init_checkpoint ./tmp/model_10000/model_state.pdparams \
    --device gpu
```

参数释义如下：
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer，支持[PaddleNLP Transformer类预训练模型](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer) 中的所有模型，但只有`ernie-gen-base-en, ernie-gen-large-en, ernie-gen-large-en-430g`三种模型会加载最后输出层的参数，其余模型只会加载transformer参数作热启动。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `max_encode_len` 表示最大输入句子长度，超过该长度将被截断。
- `max_decode_len` 表示最大输出句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `init_checkpoint` 表示模型加载路径。
- `use_gpu` 表示使用GPU。

### 模型预测

对无标签数据可以启动模型预测：

```bash
python -u ./predict.py \
    --model_name_or_path ernie-1.0 \
    --max_encode_len 24 \
    --max_decode_len 72 \
    --batch_size 48   \
    --init_checkpoint ./tmp/model_10000/model_state.pdparams \
    --device gpu
```


## Citation

您可以按下面的格式引用ERNIE-Gen论文:

```
@article{xiao2020ernie-gen,
  title={ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation},
  author={Xiao, Dongling and Zhang, Han and Li, Yukun and Sun, Yu and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2001.11314},
  year={2020}
}
```

## 线上教程体验

我们为诗歌文本生成提供了线上教程，欢迎体验：

* [使用PaddleNLP预训练模型ERNIE-GEN生成诗歌](https://aistudio.baidu.com/aistudio/projectdetail/1339888)


## Acknowledgement

- 感谢 [chinese-poetry数据集](https://github.com/chinese-poetry/chinese-poetry) 开放的诗歌数据集
