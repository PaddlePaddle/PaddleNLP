# DDParser

* [模型简介](#模型简介)
* [快速开始](#快速开始)
    * [环境依赖](#环境依赖)
    * [句法分析任务](#句法分析任务)
* [致谢](#致谢)
* [参考论文](#参考论文)

## 模型简介

依存句法分析任务通过分析句子中词语之间的依存关系来确定句子的句法结构，DDParser是一款依存句法分析工具，
该用例是基于Paddle v2.1的[baidu/ddparser](https://github.com/baidu/DDParser)实现。

以下是本项目主要代码结构及说明：

```text
ddparser/
├── model # 部署
│   ├── dropouts.py # dropout
│   ├── encoder.py # 编码器
│   ├── metric.py # 指标计算
│   ├── model.py # 模型网络
│   └── model_utils.py # 模型网络工具函数
├── README.md # 使用说明
├── data.py # 数据结构
├── env.py # 环境配置工具
├── run.py # 主入口，包含训练、评估和预测任务
└── utils.py # 任务工具函数
```

## 快速开始

### 环境依赖
* `python`: >=3.6.0
* `paddlepaddle`: >=2.1
* `LAC`: >=2.1
* `dill`

### 句法分析任务

#### 模型训练

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run.py \
                                                --mode train \
                                                --device=gpu \
                                                --batch_size=4000 \
                                                --encoding_model=ernie-gram-zh \
                                                --train_data_path=data/train.txt \
                                                --dev_data_path=data/dev.txt 
```

#### 模型评估
```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" run.py \
                                                --mode evaluate \
                                                --device=gpu \
                                                --model_file_path=checkpoint \
                                                --tree
```

#### 模型预测
```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" run.py \
                                                --mode predict \
                                                --device=gpu \
                                                --model_file_path=checkpoint \
                                                --infer_result_path=infer_result \
                                                --tree
```

#### 可配置参数说明

* `mode`: 任务模式，可选为train、evaluate和predict。
* `device`: 选用什么设备进行训练，可选cpu、gpu或xpu。如使用gpu训练则参数gpus指定GPU卡号。
* `encoding_model`: 选择模型编码网络，可选lstm、lstm-pe、ernie-1.0、ernie-tiny和ernie-gram-zh。
* `preprocess`: 训练模式下的使用参数，设置表示会基于训练数据进行词统计等操作，不设置则使用已统计好的信息；针对统一训练数据，多次训练可不设置该参数; 默认为True。
* `epochs`: 训练轮数。
* `save_dir`: 保存训练模型的路径；默认将当前在验证集上效果最好的模型保存在目录model_file文件夹下。
* `train_data_path`: 训练集文件路径。
* `dev_data_path`: 开发集文件路径。
* `batch_size`: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数，默认为1000。
* `init_from_params`: 模型参数路径，热启动模型训练；默认为None。
* `clip`: 梯度裁剪阈值，将梯度限制在阈值范围内。
* `lstm_lr`: 模型编码网络为lstm或lstm-pe时的学习率，默认为0.002。
* `ernie_lr`: 模型编码网络为ernie-1.0、ernie-tiny、ernie-gram-zh时的学习率，默认为5e-5。
* `seed`: 随机种子，默认为1000。
* `test_data_path`: 测试集文件路径。
* `model_file_path`: 评估和预测模式下的使用参数，设置后会从该路径加载已训练保存的模型文件进行模型评估或预测，默认为model_file文件夹。
* `infer_result_path`: 预测结果保存路径，默认保存在当前目录infer_result文件夹下。
* `min_freq`: 训练模式下的使用参数，基于训练数据生成的词表的最小词频，默认为2。
* `n_buckets`: 训练模式下的使用参数，选择数据分桶数，对训练数据按照长度进行分桶。
* `tree`: 确保输出结果是正确的依存句法树，默认为True。
* `feat`: 模型编码网络为lstm时的使用参数，选择输入的特征，可选char（句子的char级表示）和pos（词性标签）；ernie类别的模型只能为None。
* `warmup_proportion`: 学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
* `weight_decay`: 控制正则项力度的参数，用于防止过拟合，默认为0.0。

## 致谢

* 感谢[百度NLP](https://github.com/baidu/DDParser)提供ddparser的开源代码实现。

## 参考论文

- [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)