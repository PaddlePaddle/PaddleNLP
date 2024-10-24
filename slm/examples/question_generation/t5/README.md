# 问题生成(Question Generation)

## 简介

Question Generation（QG），即问题生成，指的是给定一段上下文（passage 或 sentence），自动生成一个流畅且符合上下文主题的问句。问题生成通常可以分为两个分支，即无答案问题生成（answer-agnostic question generation）和有答案问题生成（answer-aware question generation）。

本项目是 T5在 PaddlePaddle 上开源实现的有答案问题生成的例子，包含了在 SQuAD 数据集上微调和生成的代码。

## 快速开始

### 环境依赖

- nltk
- evaluate


安装方式：`pip install -r requirements.txt`

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
.
├── finetune.py # 模型微调主程序入口
├── generate.py # 模型生成主程序入口
├── utils.py # 定义参数及一些工具函数
├── requirements.txt # 环境依赖文件
└── README.md # 文档说明
```

### 数据准备

#### 数据加载
**SQuAD**（Stanford Question Answering Dataset）数据集是一个英文问答数据集，现有的问题生成研究主要在该数据集上进行评价。**SQuAD**中的数据由段落、问题、答案3个主要部分组成，其中段落从维基百科中获取，问题和答案通过众包的方式由人工标注。

为了方便用户快速测试，PaddleNLP Dataset API 内置了 Squad 数据集，一键即可完成数据集加载，示例代码如下：

```python
from paddlenlp.datasets import load_dataset
train_set, dev_set, test_set = load_dataset("squad",  splits=["train_v1", "dev_v1"])
```

#### 数据处理
针对**SQuAD**数据集，我们需要将 QA 任务格式的数据进行转换从而得到 text2text 形式的数据，默认构造方式如下，其他形式输入数据用户可以在 convert_example 函数中自行定义
```text
answer: {answer_text} context: {context_text}
question: {question_text}
```
具体案例如下，
```text
answer: the Miller–Rabin primality test context: The property of being prime (or not) is called primality. A simple but slow method of verifying the primality of a given number n is known as trial division. It consists of testing whether n is a multiple of any integer between 2 and . Algorithms much more efficient than trial division have been devised to test the primality of large numbers. These include the Miller–Rabin primality test, which is fast but has a small probability of error, and the AKS primality test, which always produces the correct answer in polynomial time but is too slow to be practical. Particularly fast methods are available for numbers of special forms, such as Mersenne numbers. As of January 2016[update], the largest known prime number has 22,338,618 decimal digits.

question: What is the name of the process which confirms the primality of a number n?
```

### 模型训练

运行如下命令即可在训练集上进行 finetune，并在验证集上进行验证

```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
# 例如使用1号和2号卡，则：`--gpu 1,2`
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus 1,2 train.py \
    --model_name_or_path=t5-base \
    --dataset_name=squad \
    --output_dir=output \
    --max_source_length=1024 \
    --max_target_length=142 \
    --learning_rate=1e-4 \
    --num_train_epochs=6 \
    --logging_steps=100 \
    --save_steps=1000 \
    --seed=42 \
    --train_batch_size=4 \
    --eval_batch_size=64 \
    --warmup_proportion=0.1 \
    --ignore_pad_token_for_loss \
    --device=gpu
```

其中参数释义如下：
- `gpus` 指示了训练所用的 GPU

- `model_name_or_path` 指示了 finetune 使用的预训练模型，可以是 PaddleNLP 提供的预训练模型，或者是本地的模型。如果使用本地的模型，则配置为本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含 paddle 模型参数 model_state.pdparams。如果使用 PaddleNLP 提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP 提供的预训练模型        |
   |---------------------------------|
   | t5-base |
   | t5-large |

- `dataset_name` 表示训练的数据集。

- `output_dir` 表示模型的保存路径。

- `max_source_length` 表示输入序列的长度，超过该长度将被截断。

- `max_target_length` 表示输出的最大长度。

- `learning_rate` 表示基础学习率大小，将与 learning rate scheduler 产生的值相乘作为当前学习率。

- `num_train_epochs` 表示训练轮数。

- `epochs` 表示训练轮数。

- `logging_steps` 表示日志打印间隔。

- `save_steps` 表示模型保存及评估间隔。

- `seed` 表示随机数生成器的种子。

- `train_batch_size` 表示训练每张卡上的样本数目。

- `eval_batch_size` 表示预测单卡上的样本数目。

- `warmup_proportion` 表示 warmup_steps 所占总步数的比例。学习率逐渐升高到基础学习率（即上面配置的 learning_rate）所需要的迭代数。

- `device` 表示使用的设备。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`output_dir`中。如：

```text
./output/
├── t5_model_1000
│   ├── model_config.json
│   ├── model_state.pdparams
│   ├── special_tokens_map.json
│   ├── spiece.model
│   └── tokenizer_config.json
└── ...
```

**NOTE:** 如需恢复模型训练，只需指定`model_name_or_path`为本地微调模型的路径即可。

### 模型预测

运行如下命令即可在验证集上进行测试

```shell
# GPU启动，预测仅支持单卡
export CUDA_VISIBLE_DEVICES=0
python predict.py \
    --model_name_or_path=./checkpoints/model_xx/ \
    --dataset_name=squad \
    --output_path=generate.txt \
    --max_source_length=1024 \
    --max_target_length=142 \
    --decode_strategy=greedy_search \
    --top_k=2 \
    --top_p=1.0 \
    --num_beams=1 \
    --length_penalty=0.0 \
    --batch_size=64 \
    --seed=42 \
    --ignore_pad_token_for_loss \
    --logging_steps=20 \
    --device=gpu
```

其中参数释义如下：
- `model_name_or_path` 指示了预测使用的模型，可以是 PaddleNLP 提供的预训练模型，或者是本地的模型。如果使用本地的模型，则配置为本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含 paddle 模型参数 model_state.pdparams。如果使用 PaddleNLP 提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP 提供的预训练模型        |
   |---------------------------------|
   | t5-base |
   | t5-large |
   | mrm8488/t5-base-finetuned-question-generation-ap |

- `dataset_name` 表示预测的数据集。

- `output_path` 表示预测结果的保存路径。

- `max_source_length` 表示输入序列的长度，超过该长度将被截断。

- `max_target_length` 表示输出的最大长度。

- `decode_strategy` 表示预测解码时采取的策略，可选"sampling"、"greedy_search"和"beam_search"之一。

- `top_k` 表示采用"sampling"解码策略时，token 的概率按从大到小排序，生成的 token 只从前`top_k`个中进行采样。

- `top_p` 表示采用"sampling"解码策略时，从词表中采样并选择概率之和大于给定阈值`top_p`的 token。

- `num_beams` 表示 besm search 的 beam size。

- `length_penalty` 表示 besm search 生成长度的指数惩罚。

- `batch_size` 表示每次迭代**单卡**上的样本数目。

- `seed` 表示随机数生成器的种子。

- `logging_steps` 表示日志打印间隔。

- `device` 表示使用的设备。

程序运行结束后会将预测生成的问题保存在`output_path`中。同时终端中会输出评估结果。

采用社区微调模型 mrm8488/t5-base-finetuned-question-generation-ap 在验证集上有如下结果：

|   model_name_or_path    |     BLEU-1     |     BLEU-2     |    BLEU-3    |    BLEU-4    |
| :----------------------: | :-------------: | :-------------: |:-------------: |:-------------: |
|        [mrm8488/t5-base-finetuned-question-generation-ap](https://huggingface.co/mrm8488/t5-base-finetuned-question-generation-ap )      | 50.11 | 35.83 | 27.68 |  22.03 |




## 参考文献
1. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W. and Liu, P.J., 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(140), pp.1-67.
