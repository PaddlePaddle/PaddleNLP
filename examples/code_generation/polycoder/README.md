# Code Generation

## 模型介绍
GPT-[2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)/[3](https://arxiv.org/pdf/2005.14165.pdf) 是以[Transformer](https://arxiv.org/abs/1706.03762) 解码器为网络基本组件，使用自回归的方式在大规模无标注文本语料上进行预训练得到的语言生成模型。

本项目是语言模型 GPT 的 PaddlePaddle 实现， 包含模型训练，预测等内容。下是本例的简要目录结构及说明：

```text
.
├── args.py                 # 训练参数配置
├── dataset.py              # 数据处理
├── lr.py                   # 学习率控制
├── README.md               # 文档
├── run_generation.py       # 生成入口
├── run_pretrain.py         # 预训练入口
└── scripts/                # 数据处理/训练/生成脚本
```

## 快速开始

### 环境依赖

- regex
- sentencepiece >= 0.1.94
- tqdm
- visualdl
- paddlepaddle-gpu >= 2.2
- pybind11
- lac (可选)
- zstandard (可选)

安装命令 `pip install regex sentencepiece tqdm visualdl pybind11 lac zstandard`。
注：需要PaddlePaddle版本大于等于2.2rc，或者使用最新develop版本，安装方法请参见Paddle[官网](https://www.paddlepaddle.org.cn)。

### 数据准备

#### 数据获取与制作

C++ 一共 3309507个文件，共44G  
Python 一共714758个文件，共7.6G
一共是4024265个文件，共计51.6G
链接: https://pan.baidu.com/s/1TEWUZ9rT3rJa3OyIs1Qa-A?pwd=sjet 提取码: sjet

下载以后通过以下命令解压：

```shell
tar -zxvf Code_python_cpp.tar.gz -C  /path/to/code_dir/
```

然后使用 `scripts/process_data.sh` 的脚本进行数据集制作：
```shell
bash scripts/process_data.sh
```
处理完成后就可以在 `data_tools` 目录下看到我们需要的 `code_python_ids.npy` ,  `code_python_idx.npz` 数据集文件。


### 模型训练

#### 单卡训练

```shell
set -x
export CUDA_VISIBLE_DEVICES=0

python -u run_pretrain.py \
    --model_type "gpt"\
    --model_name_or_path "gpt2-en"\
    --input_dir "./data_tools"\
    --output_dir "output"\
    --max_seq_len 1024 \
    --micro_batch_size 4 \
    --max_lr 0.00015\
    --min_lr 0.00001\
    --max_steps 50000 \
    --save_steps 100000\
    --decay_steps 320000\
    --weight_decay 0.01\
    --warmup_rate 0.01\
    --grad_clip 1.0\
    --logging_freq 1\
    --eval_freq 1000 \
    --device "gpu" \
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

用户也可以使用提供的shell脚本直接训练`bash scripts/run.sh`.

#### 单机多卡

同样，可以执行如下命令实现八卡训练：

```shell
set -x

task_name="gpt-dygraph"
rm -rf output/$task_name/log

export CUDA_VISIBLE_DEVICES=2,3
python -m paddle.distributed.launch \
    --gpus "2,3" \
    --log_dir "output/$task_name/log"  run_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-en"\
    --input_dir "./data_tools"\
    --output_dir "output/$task_name"\
    --max_seq_len 1024 \
    --micro_batch_size 4 \
    --max_lr 0.00015\
    --min_lr 0.00001\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --weight_decay 0.01\
    --warmup_rate 0.01\
    --grad_clip 1.0\
    --logging_freq 1\
    --eval_freq 1000\
    --device "gpu"
```

用户也可以使用提供的shell脚本直接训练`bash scripts/run_multi.sh`.

### 文本生成

同样，可以执行如下命令实现生成：

```shell
set -x
export CUDA_VISIBLE_DEVICES=0

python -u run_generation.py \
    --model_type "gpt2-en"\
    --model_name_or_path "/path/to/trained_model"\
    --decode_strategy sampling \
    --top_k 10 \
    --temperature 0.5 \
    --num_return_sequences 10 \
    --max_dec_len 512 \
    --device "gpu" \
```

用户也可以使用提供的shell脚本直接训练`bash scripts/run_generate.sh`.
