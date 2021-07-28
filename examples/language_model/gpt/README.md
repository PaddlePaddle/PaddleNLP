# GPT

## 模型介绍
GPT-[2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)/[3](https://arxiv.org/pdf/2005.14165.pdf) 是以[Transformer](https://arxiv.org/abs/1706.03762) 解码器为网络基本组件，使用自回归的方式在大规模无标注文本语料上进行预训练得到的语言生成模型。

本项目是语言模型 GPT 的 PaddlePaddle 实现， 包含模型训练，预测等内容。下是本例的简要目录结构及说明：

```text
.
├── args.py                 # 训练参数配置
├── create_pretraining_data.py         # 数据预处理脚本
├── dataset.py              # 数据处理
├── decompress.sh           # 数据集解压脚本
├── deploy/                 # 模型部署的inference脚本
├── export_model.py         # 导出预测部署的模型脚本
├── lr.py                   # 学习率控制
├── predict.py              # 生成文本示例demo
├── README.md               # 文档
├── run_eval.py             # 评估入口
├── run_pretrain.py         # 预训练入口
├── run_pretrain_static.py  # 混合并行，预训练脚本
└── scripts/                # 训练脚本
```

## 快速开始

### 环境依赖
- regex
- sentencepiece
- tqdm
- visualdl
安装命令 `pip install regex sentencepiece tqdm visualdl`

### 数据准备

#### 原始数据获取

[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/)是一个开源的英文网页文本数据集，数据来源于Reddit，经过去重、清洗、提取，最终包含800多万个文档。

下载以后通过以下命令解压：

```shell
xz -d openwebtext.tar.xz
tar xf openwebtext.tar
mkdir raw_data
bash decompress.sh
```

解压以后得到的`raw_data`目录大小约为54GB。

#### 数据预处理

为了提升训练速度，我们在训练前将文本数据转成相应的id，并保存为npz格式：

```shell
python create_pretraining_data.py --input_path raw_data \
 --model_name gpt2-en \
 --append_eod \
 --workers 8
```

运行命令后，产出`raw_data_ids.npz`文件。为了方便用户运行测试本模型，本项目提供了处理好的300M的训练样本：

```shell
wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt/train.data.json_ids.npz
```

将所有预处理得到的npz文件统一放入一个文件夹中，以备训练使用：

```
mkdir data
mv train.data.json_ids.npz data
```

### 模型训练

#### 单卡训练

```shell
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-en \
    --input_dir "./data"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --micro_batch_size 4\
    --device gpu
```

其中参数释义如下：
- `model_name_or_path` 要训练的模型或者之前训练的checkpoint。
- `input_dir` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件。
- `output_dir` 指定输出文件。
- `weight_decay` 权重衰减参数。
- `grad_clip` 梯度裁剪范围。
- `max_steps` 最大训练步数
- `save_steps` 保存模型间隔
- `mirco_batch_size` 训练的batch大小
- `device` 训练设备

用户也可以使用提供的shell脚本直接训练`sh scripts/run.sh`.

#### 单机多卡

同样，可以执行如下命令实现八卡训练：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-en \
    --input_dir "./data"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --micro_batch_size 4\
    --device gpu
```

用户也可以使用提供的shell脚本直接训练`sh scripts/run_multi.sh`.

### 模型评估

我们提供了对[WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)、[LAMBADA](https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl)两种数据集的评估脚本, 使用如下命令启动评估：

1. WikiText数据集评估
```bash
python run_eval.py --model_name gpt2-en \
    --eval_path ./wikitext-103/wiki.valid.tokens \
    --overlapping_eval 32 \
    --init_checkpoint_path ./output/model_100000/model_state.pdparams \
    --batch_size 8 \
    --device gpu
```

2. LAMBADA数据集评估
```bash
python run_eval.py --model_name gpt2-en \
    --eval_path ./lambada_test.jsonl \
    --cloze_eval \
    --init_checkpoint_path ./output/model_100000/model_state.pdparams \
    --batch_size 8 \
    --device gpu
```
其中参数释义如下：
`model_name` 使用的模型名称，如gpt2-en、gpt2-medium-en等。
`eval_path` 数据集地址。
`init_checkpoint_path` 模型参数地址。
`batch_size` batch size大小。
`device` 运行设备，cpu，gpu，xpu可选。
`overlapping_eval` wikitext数据集参数。
`cloze_eval` lambada数据参数，作为完型填空任务。

其中数据集WikiText采用的是PPL(perplexity)评估指标，LAMBADA采用的是ACC(accuracy)指标。

注：不设置`init_checkpoint_path` 参数时，可以评估默认预训练好的模型参数。


### 文本生成

本项目提供了简单的文本生成的demo，供用户测试文本生成效果。

```shell
# 中文示例
python predict.py gpt-cn
# 英文示例
python predict.py
```

生成效果展示:
```text
问题：中国的首都是哪里？答案：北京。
问题：苹果的CEO是谁? 答案：乔布斯。

默写古诗: 大漠孤烟直，长河落日圆。
举杯邀明月，对影成三人。

Question: Who is the CEO of Apple?
Answer: Tim Cook.
```

## 模型导出预测

下面提供了简单的示例，帮助用户将预训练模型导出成预测部署的参数。

导出中文模型
```"shell
python export_model.py --model_type=gpt-cn \
    --model_path=gpt-cpm-large-cn \
    --output_path=./infer_model/model
```
用户在`infer_model`中可以看到导出的文件。

对于导出的模型，我们提供了Python的infer脚本，调用预测库对简单的例子进行预测。
```shell
python deploy/python/inference.py --model_type gpt-cn \
    --model_path ./infer_model/model
```


导出英文模型
```"shell
python export_model.py --model_type=gpt \
    --model_path=gpt2-medium-en \
    --output_path=./infer_model/model

python deploy/python/inference.py --model_type gpt \
    --model_path ./infer_model/model
```

用户可以看到屏幕输出预测结果。

## 飞桨4D混合并行训练
飞桨4D混合并行，使用sharding、模型并行、流水线并行和数据并行策略，使得训练千亿参数规模的模型成为可能。在本示例中，我们提供了基于飞桨最新混合并行策略的GPT预训练模型。运行下面脚本，即可进行模型预训练：
```shell
sh scripts/run_static.sh
```
用户可以根据自己的机器资源，灵活调整并行策略，选择最合适的策略来训练模型。更多关于混合并行策略的的例子详见[飞桨4D混合并行训练使用指南](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/collective/collective_mp/hybrid_parallelism.html)

## 参考文献
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
- [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)
