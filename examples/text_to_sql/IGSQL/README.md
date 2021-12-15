# IGSQL: Database Schema Interaction Graph Based Neural Model for Context-Dependent Text-to-SQL Generation

## 上下文相关的 Text2SQL 任务

语义解析是一种交互式分析技术，其将用户输入的自然语言表述转成可操作执行的语义表示形式，如逻辑表达式（如一阶逻辑表示，lambda表示等）、编程语言（如SQL、python等）、数学公式等。

Text2SQL 是语义解析技术中的一类任务，让机器自动将用户输入的自然语言问题转成可与数据库交互的 SQL 查询语言，实现基于数据库的自动问答能力。

上下文相关的 Text2SQL 则指在多轮问答、对话等场景中，对问题的解析除了依赖当前轮次的输入语句，往往同时依赖于上文中的用户语句和系统答复等，即要求模型具备上下文的感知（建模）能力，才可以更好地完成 SQL 生成的任务。这种多轮交互的方式更符合人类的行为习惯，所以上下文相关的 Text2SQL 解析技术也日益受到重视，成为学术界、工业界的研究重点和应用方向。

## 数据集

当前学术界主流的上下文相关的 Text2SQL 数据集包括[SParC](https://yale-lily.github.io/sparc)、[CoSQL](https://yale-lily.github.io/cosql) 等，详细说明可参见上述链接页面及相应的论文。

## 基线系统
本系统基于 PaddlePaddle 动态图复现了 [IGSQL](https://github.com/headacheboy/IGSQL) 模型，其核心是基于预训练模型（ERNIE、BERT等）和LSTM的基础 Encoder，以及针对多轮场景的交互 Schema Encoder 和上下文句子 Encoder，而解码端则是在 EditSQL 基础上扩展而来的、基于门控机制和拷贝机制的 SQL 序列生成 Decoder。

# 环境准备
代码运行需要 Linux 主机，Python 3.7 和 PaddlePaddle 2.1 以上版本。

## 推荐的环境

* 操作系统 CentOS 7.5
* Python 3.7.9
* PaddlePaddle develop

除此之外，强烈建议使用支持 GPU 的硬件环境。

## PaddlePaddle

可根据机器情况和个人需求在 PaddlePaddle 和 PaddlePaddle-GPU 中二选一安装。
如果机器支持GPU，则建议安装GPU版本。

关于 PaddlePaddle 的安装教程、使用方法等请参考[官方文档](https://www.paddlepaddle.org.cn/#quick-start).

## 第三方 Python 库
除 PaddlePaddle 及其依赖之外，还依赖其它第三方 Python 库，位于代码根目录的 requirements.txt 文件中。

可使用 pip 一键安装

```pip install -r requirements.txt```

# 数据准备

```bash
# 下载模型训练、测试数据
# 得到的sparc，cosql 两个数据集
wget https://bj.bcebos.com/paddlenlp/paddlenlp/resource/igsql_data.tar.gz
tar xzvf igsql_data.tar.gz
# 下载glove词向量
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

# 数据预处理

对原始数据进行数据预处理，以适配模型的输入，以sparc为例：

```bash
python preprocess.py --dataset=sparc --remove_from
```

## 训练

以训练sparc模型为例:

```bash
python run.py --raw_train_filename="data/sparc_data_removefrom/train.pkl" \
          --raw_validation_filename="data/sparc_data_removefrom/dev.pkl" \
          --database_schema_filename="data/sparc_data_removefrom/tables.json" \
          --embedding_filename="glove.840B.300d.txt" \
          --data_directory="processed_data_sparc_removefrom" \
          --logdir="logs_sparc" \
          --train=True \
          --evaluate=True
```

参数说明：
* raw_train_filename, raw_validation_filename, database_schema_filename: 数据集文件路径。
* embedding_filename: GLOVE 词向量文件路径。
* data_directory: 预处理得到的文件夹路径。
* logdir: 输出日志文件夹路径。
* train，evaluate: 是否执行trian，evaluate。


### 训练阶段的输出日志
训练过程会输出loss、acc相关日志，内容类似：
```
total_gold_tokens:13, step:5981================================= ]  99% ETA:  0:00:03
LOSS:0.4242228865623474
train     [==================================] 100% Time: 1:20:22
Predicting with file logs_sparc/train-eval_predictions.json
logs_sparc/train-eval[==================================] 100% Time: 0:01:30
Predicting with file logs_sparc/valid-eval_predictions.json
logs_sparc/valid-eval[==================================]100% Time: 0:04:53
```

## 预测

以预测sparc数据集为例:

```bash
python run.py --raw_train_filename="data/sparc_data_removefrom/train.pkl" \
          --raw_validation_filename="data/sparc_data_removefrom/dev.pkl" \
          --database_schema_filename="data/sparc_data_removefrom/tables.json" \
          --embedding_filename="glove.840B.300d.txt" \
          --data_directory="processed_data_sparc_removefrom" \
          --logdir="logs_sparc_eval" \
          --evaluate=True \
          --save_file="logs_sparc/best_model"
```

参数说明：
* save_file: 加载的模型路径，请修改为真实的模型加载路径。

执行完上述命令后，预测结果保存在 "logs_sparc_eval/valid_use_gold_queries_predictions.json"。

# 评估

执行以下命令获得评估结果：

```bash
python postprocess_eval.py --dataset=sparc --split=dev --pred_file logs_sparc_eval/valid_use_gold_queries_predictions.json --remove_from
```

其中的 --pred_file 参数请修改为真实的模型预测输出路径，评估结果保存在 "logs_sparc_eval/valid_use_gold_queries_predictions.json.eval"。
