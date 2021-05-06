# LIC 2021对话比赛baseline

## 模型简介

近年来，人机对话系统受到了学术界和产业界的广泛关注并取得了不错的发展。开放域对话系统旨在建立一个开放域的多轮对话系统，使得机器可以流畅自然地与人进行语言交互，既可以进行日常问候类的闲聊，又可以完成特定功能，以使得开放域对话系统具有实际应用价值，例如进行对话式推荐，或围绕一个主题进行深入的知识对话等。具体的说，开放域对话可以继续拆分为支持不同功能的对话形式，例如对话式推荐，知识对话技术等，如何解决并有效融合以上多个技能面临诸多挑战。

LIC 2021对话比赛收集了一系列公开的开放域对话数据并提供了统一的评测方式，旨在为研究人员和开发者提供学术和技术交流的平台，进一步提升开放域对话的研究水平，推动自然语言理解和人工智能领域技术的应用和发展。

为了方便参赛者快速了解LIC 2021对话比赛的流程，并快速地参与到比赛中，本项目基于UnifiedTransformer模型提供了一个基础baseline，利用小规模样例数据在预训练模型上完成了微调及预测。参赛者可以针对赛题进行其他改进，例如修改数据预处理方法、修改网络结构、修改训练方式、修改预测的解码方式或对结果的后处理策略等方式提升模型效果。

UnifiedTransformer模型的细节可以[参阅论文](https://arxiv.org/abs/2006.16779)。

## 快速开始

### 环境依赖

- sentencepiece

安装方式：`pip install sentencepiece`

### 数据准备

由于样例数据涉及LIC 2021对话比赛，暂不开放。
关于数据集及数据集的预处理过程，详见[2021语言与智能技术竞赛：多技能对话](https://aistudio.baidu.com/aistudio/competition/detail/67)及官方提供的基线系统Baselines。

模型的输入由3部分组成：词向量token_ids，句向量token_type_ids和位置向量position_ids。本项目的数据集是样例文本经过数据预处理脚本得到的id化的数据集。数据的每一行由3列组成，以";"作为分割符，格式：token_ids;token_type_ids;position_ids。具体细节请参考`data.py`。

### 模型训练

运行如下命令即可在样例训练集上进行finetune，并在样例验证集上进行验证

```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" --log_dir ./log finetune.py \
    --model_name_or_path=unified_transformer-12L-cn \
    --train_data_path=./datasets/train.txt \
    --valid_data_path=./datasets/valid.txt \
    --save_dir=./checkpoints \
    --logging_steps=500 \
    --save_steps=8000 \
    --seed=2021 \
    --epochs=10 \
    --batch_size=8192 \
    --lr=1e-5 \
    --weight_decay=0.01 \
    --warmup_steps=4000 \
    --max_grad_norm=0.1 \
    --sort_pool_size=65536 \
    --device=gpu
```

其中参数释义如下：
- `gpus` 指示了训练所用的GPU卡号。
- `model_name_or_path` 指示了finetune使用的具体预训练模型，可以是PaddleNLP提供的预训练模型，或者是本地的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle预训练模型model_state.pdparams。如果使用PaddleNLP提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP提供的预训练模型        |
   |---------------------------------|
   | unified_transformer-12L-cn      |
   | unified_transformer-12L-cn-luge |

- `train_data_path` 表示训练集文件路径。
- `valid_data_path` 表示验证集文件路径。
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
- `sort_pool_size` 表示在构建batch数据时，用来排序的pool size。
- `device` 表示训练使用的设备。

参数详情和参数的默认值请参考`args.py`。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
./checkpoints/
├── model_8000
│   ├── model_config.json
│   ├── model_state.pdparams
│   ├── spm.model
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:** 如需恢复模型训练，`model_name_or_path`配置本地模型的目录地址即可。

### 模型预测

运行如下命令即可在样例测试集上进行测试

```shell
export CUDA_VISIBLE_DEVICES=0
# GPU启动，预测仅支持单卡
python infer.py \
    --model_name_or_path=./checkpoints/model_80000 \
    --test_data_path=./datasets/test.txt \
    --output_path=./predict.txt \
    --logging_steps=500 \
    --seed=2021 \
    --batch_size=4 \
    --min_dec_len=1 \
    --max_dec_len=64 \
    --num_samples=20 \
    --decode_strategy=sampling \
    --top_k=5 \
    --device=gpu
```

其中参数释义如下：
- `model_name_or_path` 指示了finetune使用的具体预训练模型，可以是PaddleNLP提供的预训练模型，或者是本地的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle预训练模型model_state.pdparams。如果使用PaddleNLP提供的预训练模型，可以选择下面其中之一。

   | PaddleNLP提供的预训练模型        |
   |---------------------------------|
   | unified_transformer-12L-cn      |
   | unified_transformer-12L-cn-luge |

- `test_data_path` 表示预测集文件路径。
- `output_path` 表示预测结果的保存路径。
- `logging_steps` 表示日志打印间隔。
- `seed` 表示随机数生成器的种子。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `min_dec_len` 表示预测生成的句子的最小长度。
- `max_dec_len` 表示预测生成的句子的最大长度。
- `num_samples` 表示每条样本生成的句子的数量。对于每条样本，模型会生成`num_samples`个句子，根据每个句子的概率得分进行排序，得分最高的句子作为最终的生成结果。
- `decode_strategy` 表示预测解码时采取的策略，可选"sampling"、"greedy_search"和"beam_search"之一。
- `top_k` 表示采用"sampling"解码策略时，token的概率按从大到小排序，生成的token只从前`top_k`个中进行采样。
- `device` 表示训练使用的设备。

参数详情和参数的默认值请参考`args.py`。

程序运行结束后会将预测结果保存在`output_path`中。将预测结果准备成比赛官网要求的格式，提交评估即可得评估结果。

采用不同的模型在样例测试集上有如下结果：

|       model_name_or_path        |  F1   | BLEU1 / BLEU2 | DISTINCT1 / DISTINCT2 |
| :-----------------------------: | :---: | :-----------: | :-------------------: |
|   unified_transformer-12L-cn    | 10.62 | 0.070 / 0.022 |     0.065 / 0.304     |
| unified_transformer-12L-cn-luge | 33.11 | 0.245 / 0.157 |     0.074 / 0.238     |
|    ./checkpoints/model_80000    | 32.38 | 0.239 / 0.150 |     0.070 / 0.219     |
