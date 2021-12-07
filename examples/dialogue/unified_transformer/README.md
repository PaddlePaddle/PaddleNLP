# UnifiedTransformer

## 模型简介

近年来，人机对话系统受到了学术界和产业界的广泛关注并取得了不错的发展。开放域对话系统旨在建立一个开放域的多轮对话系统，使得机器可以流畅自然地与人进行语言交互，既可以进行日常问候类的闲聊，又可以完成特定功能，以使得开放域对话系统具有实际应用价值。具体的说，开放域对话可以继续拆分为支持不同功能的对话形式，例如对话式推荐，知识对话技术等，如何解决并有效融合以上多个技能面临诸多挑战。

[UnifiedTransformer](https://arxiv.org/abs/2006.16779)以[Transformer](https://arxiv.org/abs/1706.03762) 编码器为网络基本组件，采用灵活的注意力机制，十分适合对话生成任务。

本项目是UnifiedTransformer在 Paddle 2.0上的开源实现，介绍了如何使用UnifiedTransformer在DuConv任务型对话数据集上进行微调，并给出了一个搭建简单中文聊天机器人的例子。

## 快速开始

### 环境依赖

- sentencepiece
- termcolor

安装方式：`pip install sentencepiece termcolor`

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
.
├── finetune.py # 模型finetune主程序入口
├── infer.py # 模型预测主程序入口
├── utils.py # 定义参数及一些工具函数
└── README.md # 文档说明
```

### 数据准备

**DuConv**是百度发布的基于知识图谱的主动聊天任务数据集，让机器根据构建的知识图谱进行主动聊天，使机器具备模拟人类用语言进行信息传递的能力。数据集的创新性是：强调了bot的主动性，并且在闲聊对话中引入了明确的对话目标，即将对话引导到特定实体上。数据中的知识信息来源于电影和娱乐人物领域有聊天价值的知识信息，如票房、导演、评价等，以三元组SPO的形式组织，对话目标中的话题为电影或娱乐人物实体。数据集中共有3万session，约12万轮对话，划分为训练集、开发集、测试集1和测试集2，其中测试集1中包含对话的response，而测试集2中只有对话历史。

为了方便用户快速测试，PaddleNLP Dataset API内置了DuConv数据集，一键即可完成数据集加载，示例代码如下：

```python
from paddlenlp.datasets import load_dataset
train_ds, dev_ds, test1_ds, test2_ds = load_dataset('duconv', splits=('train', 'dev', 'test_1', 'test_2'))
```

### 预训练模型

以下是PaddleNLP支持的对话类预训练模型：

|模型名称| 模型参数 | 模型特点 |
|:-----:|:------:|:-------:|
|unified_transformer-12L-cn| 12-layers, 12-heads, 768-hidden| 在千万级别的中文会话数据上进行预训练。|
|unified_transformer-12L-cn-luge| 12-layers, 12-heads, 768-hidden|由unified_transformer-12L-cn预训练模型在千言对话数据集上进行微调。并且模型输入中加入了标识不同对话技能的special token，使得模型能同时支持闲聊对话、推荐对话和知识对话。|
|plato-mini| 6-layers, 12-heads, 768-hidden|在十亿级别的中文对话数据上进行预训练。参数量更小，但效果更好。只支持闲聊型对话。|

### 模型训练

运行如下命令即可在训练集上进行finetune，并在验证集上进行验证

```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
# 例如使用1号和2号卡，则：`--gpu '1,2'`
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus '0' --log_dir ./log finetune.py \
    --model_name_or_path=unified_transformer-12L-cn-luge \
    --save_dir=./checkpoints \
    --logging_steps=100 \
    --save_steps=1000 \
    --seed=2021 \
    --epochs=3 \
    --batch_size=16 \
    --lr=5e-5 \
    --weight_decay=0.01 \
    --warmup_steps=2500 \
    --max_grad_norm=0.1 \
    --max_seq_len=512 \
    --max_response_len=128 \
    --max_knowledge_len=256 \
    --device=gpu
```

其中参数释义如下：
- `gpus` 指示了训练所用的GPU
- `log_dir` 指示了日志保存目录
- `model_name_or_path` 指示了finetune使用的预训练模型，可以是PaddleNLP提供的预训练模型，或者是本地的模型。如果使用本地的模型，则配置为本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle模型参数model_state.pdparams。如果使用PaddleNLP提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP提供的预训练模型        |
   |---------------------------------|
   | unified_transformer-12L-cn      |
   | unified_transformer-12L-cn-luge |

- `save_dir` 表示模型的保存路径。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `seed` 表示随机数生成器的种子。
- `epochs` 表示训练轮数。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `lr` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示AdamW优化器中使用的weight_decay的系数。
- `warmup_steps` 表示学习率逐渐升高到基础学习率（即上面配置的lr）所需要的迭代数，最早的使用可以参考[这篇论文](https://arxiv.org/pdf/1706.02677.pdf)。
- `max_grad_norm` 表示梯度裁剪允许的最大梯度值。
- `max_seq_len` 表示输入序列的最大长度。
- `max_response_len` 表示输入response的最大长度。
- `max_knowledge_len` 表示输入knowledge序列的最大长度。
- `device` 表示使用的设备。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`save_dir`中，其中loss最小的模型会被保存在`save_dir/model_best`中。如：

```text
./checkpoints/
├── model_1000
│   ├── model_config.json
│   ├── model_state.pdparams
│   ├── spm.model
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:** 如需恢复模型训练，只需指定`model_name_or_path`为本地微调模型的路径即可。

### 模型预测

运行如下命令即可在测试集上进行测试。

```shell
# GPU启动，预测仅支持单卡
export CUDA_VISIBLE_DEVICES=0
python infer.py \
    --model_name_or_path=./checkpoints/model_best \
    --output_path=./predict.txt \
    --logging_steps=10 \
    --seed=2021 \
    --max_seq_len=512 \
    --max_knowledge_len=256 \
    --batch_size=4 \
    --min_dec_len=1 \
    --max_dec_len=64 \
    --num_return_sequences=20 \
    --decode_strategy=sampling \
    --top_k=5 \
    --device=gpu
```

其中参数释义如下：
- `model_name_or_path` 指示了预测使用的模型，可以是PaddleNLP提供的预训练模型，或者是本地的模型。如果使用本地的模型，则配置为本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle模型参数model_state.pdparams。如果使用PaddleNLP提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP提供的预训练模型        |
   |---------------------------------|
   | unified_transformer-12L-cn      |
   | unified_transformer-12L-cn-luge |

- `output_path` 表示预测结果的保存路径。
- `logging_steps` 表示日志打印间隔。
- `seed` 表示随机数生成器的种子。
- `max_seq_len` 表示输入序列的最大长度。
- `max_knowledge_len` 表示输入knowledge序列的最大长度。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `min_dec_len` 表示预测生成的句子的最小长度。
- `max_dec_len` 表示预测生成的句子的最大长度。
- `num_return_sequences` 表示每条样本生成的句子的数量。对于每条样本，模型会生成`num_return_sequences`个句子，根据每个句子的概率得分进行排序，得分最高的句子作为最终的生成结果。
- `decode_strategy` 表示预测解码时采取的策略，可选"sampling"、"greedy_search"和"beam_search"之一。
- `top_k` 表示采用"sampling"解码策略时，token的概率按从大到小排序，生成的token只从前`top_k`个中进行采样。
- `device` 表示使用的设备。

同时，我们提供了基于 FasterTransformer 的高性能预测的选项，可以选择性开启是否需要采用高性能预测，PaddleNLP 提供了 JIT 的方式，可以自动完成对所需的自定义 op 的动态库编译：
- `faster` 表示是否开启高性能预测。设置 `--faster` 即表示开启。
- `use_fp16_decoding` 表示在开启高性能预测的时候，是否使用 fp16 来完成预测过程。设置 `--use_fp16_decoding` 即表示使用 fp16 进行预测，否则使用 fp32。

程序运行结束后会将预测生成的response保存在`output_path`中。同时终端中会输出评估结果。

采用预训练模型及微调模型在测试集上有如下结果：

|       model_name_or_path        |  BLEU1 / BLEU2  | DISTINCT1 / DISTINCT2 |
| :-----------------------------: | :-------------: | :-------------------: |
| unified_transformer-12L-cn-luge | 0.2606 / 0.1576 |    0.1168 / 0.2977    |
|    ./checkpoints/model_best     | 0.2808 / 0.1744 |    0.1124 / 0.2899    |

**NOTE:** `./checkpoints/model_best`是按本项目中的超参在单卡上finetune得到的结果。

### 人机交互

运行如下命令即可开始与聊天机器人用中文进行简单的对话。

```shell
# GPU启动，仅支持单卡
export CUDA_VISIBLE_DEVICES=0
python interaction.py \
    --model_name_or_path=plato-mini \
    --min_dec_len=0 \
    --max_dec_len=64 \
    --num_return_sequences=20 \
    --decode_strategy=sampling \
    --top_k=5 \
    --device=gpu
```

其中参数释义如下：
- `model_name_or_path` 指示了预测使用的模型，可以是PaddleNLP提供的预训练模型，或者是本地的模型。如果使用本地的模型，则配置为本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle模型参数model_state.pdparams。如果使用PaddleNLP提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP提供的预训练模型        |
   |---------------------------------|
   | unified_transformer-12L-cn      |
   | unified_transformer-12L-cn-luge |
   | plato-mini                      |

- `min_dec_len` 表示预测生成的句子的最小长度。
- `max_dec_len` 表示预测生成的句子的最大长度。
- `num_return_sequences` 表示每条样本生成的句子的数量。对于每条样本，模型会生成`num_return_sequences`个句子，根据每个句子的概率得分进行排序，得分最高的句子作为最终的生成结果。
- `decode_strategy` 表示预测解码时采取的策略，可选"sampling"、"greedy_search"和"beam_search"之一。
- `top_k` 表示采用"sampling"解码策略时，token的概率按从大到小排序，生成的token只从前`top_k`个中进行采样。
- `device` 表示使用的设备。

**NOTE:** 输入"[EXIT]"退出交互程序，输入"[NEXT]"开启下一轮新的对话。需要注意使用退格会导致错误。

## Reference

- [UnifiedTransformer](https://arxiv.org/abs/2006.16779)
- [Knover/luge-dialogue](https://github.com/PaddlePaddle/Knover/tree/luge-dialogue/luge-dialogue)
- [DuConv](https://www.aclweb.org/anthology/P19-1369/)
