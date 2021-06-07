# Enhanced RAT-SQL

## Text2SQL 任务

语义解析是一种交互式分析技术，其将用户输入的自然语言表述转成一种指定的语义表示形式，如图表示（AMR等）、逻辑表达式（一阶逻辑表示，lambda表示等）、编程语言（SQL、python等）、数学公式等。

Text2SQL 是语义解析技术中的一类任务，基于给定的数据库，其将用户输入的自然语言问题转成可与数据库交互的 SQL 查询语句，实现基于数据库的自动问答能力。


## 数据集

数据集是推动自然语言处理技术进步的基石。为了处理不同场景、不同领域的应用需求，学术界及工业界陆续开放了一些相关数据集。千言项目为了验证模型的鲁棒性、泛化性等，针对每个自然语言处理问题，均收集和整理多个开源数据集，进行统一的处理并提供统一的测评方式。

作为千言项目的重要任务之一，语义解析方向收集和整理了NL2SQL、CSpider和DuSQL数据集，详情可参见千言官网的语义解析任务页面。


## 基线系统

本基线系统基于PaddlePaddle2.0动态图实现了复杂数据集上的SOTA模型[RAT-SQL](https://arxiv.org/abs/1911.04942)，其核心是基于encoder-decoder框架的序列生成模型。本系统编码端使用了 ERNIE + Relation-aware Transformer对问题和数据库schema 进行编码， 解码端实现了基于语法指导的解码算法，具体算法思想请见[TRANX](https://www.aclweb.org/anthology/D18-2002.pdf)。

同时，为了兼容上述提及的三个数据集，我们基于RAT-SQL模型进行扩展以丰富其问题解决能力，主要包括：1）多语言处理能力；2）value识别能力。

该基线系统除了提供模型的训练、预测外，还提供了评估及数据处理脚本。参赛选手及相关研究者可基于此系统进行更深层的效果优化。

# 环境准备
代码运行需要 Linux 主机，Python 3.7 和 PaddlePaddle 2.0 以上版本。

## 推荐的环境

* 操作系统 CentOS 7.5
* Python 3.7.9
* PaddlePaddle 2.0.0

除此之外，强烈建议使用支持 GPU 的硬件环境。

## PaddlePaddle

可根据机器情况和个人需求在 PaddlePaddle 和 PaddlePaddle-GPU 中二选一安装。
如果机器支持GPU，则建议安装GPU版本。

```
# CPU 版本
pip3 install paddlepaddle
# GPU 版本
pip3 install paddlepaddle-gpu
```

更多关于 PaddlePaddle 的安装教程、使用方法等请参考[官方文档](https://www.paddlepaddle.org.cn/#quick-start).

## 第三方 Python 库
除 PaddlePaddle 及其依赖之外，还依赖其它第三方 Python 库，位于代码根目录的 requirements.txt 文件中。

可使用 pip 一键安装

```pip3 install -r requirements.txt```

# 数据准备
运行前需要自行下载训练、测试数据。

```
# 下载模型训练、测试数据
# 得到的数据包括 DuSQL, NL2SQL 和 CSpider 三个数据集（同[千言-语义解析](https://aistudio.baidu.com/aistudio/competition/detail/47)任务的三个数据集）
bash data/download_model_data.sh

# 下载训练好的 Text2SQL 模型
# 得到的数据包括：
#   data
#   ├── trained_model
#   │   ├── dusql.pdparams
#   │   ├── nl2sql.pdparams
#   │   ├── cspider.pdparams
bash data/download_trained_model.sh

```

# 数据预处理

对原始数据进行格式转换、依赖信息补充等，以适配模型的输入。下面以DuSQL数据集为例进行说明。

## 获取 Schema Linking 结果
将 schema linking 独立出来，以便于针对这一步进行特定优化，可有效提升模型最终的效果。

```
# 训练集
./run.sh ./script/schema_linking.py \
        -s data/DuSQL/db_schema.json \
        -c data/DuSQL/db_content.json \
        -o data/DuSQL/match_values_train.json \
        data/DuSQL/train.json --is-train

# 开发集和测试集
./run.sh ./script/schema_linking.py \
        -s data/DuSQL/db_schema.json \
        -c data/DuSQL/db_content.json \
        -o data/DuSQL/match_values_dev.json \
        data/DuSQL/dev.json
./run.sh ./script/schema_linking.py \
        -s data/DuSQL/db_schema.json \
        -c data/DuSQL/db_content.json \
        -o data/DuSQL/match_values_test.json \
        data/DuSQL/test.json

```

需要注意的是，对于 NL2SQL 数据集，需要额外指定参数 `--sql-format nl2sql`，以便适配其简化的 SQL Json 格式。
此参数默认取值为 'dusql'，可同时兼容 DuSQL 和 CSpider 数据集。

## 获得模型输入

对 DuSQL 原始数据和Schema Linking的结果做处理，得到模型的输入，位于 data/DuSQL/preproc 目录下：
```
./run.sh ./script/text2sql_main.py \
        --mode preproc \
        --config conf/text2sql_dusql.jsonnet \
        --data-root data/DuSQL/ \
        --is-cached false \
        --output data/DuSQL/preproc
```

# 运行模型

## 模型配置文件

模型运行必需的配置位于conf下，默认提供的配置包括：text2sql_dusql.jsonnet, text2sql_nl2sql.jsonnet 和 text2sql_cspider.jsonnet， 分别用于 DuSQL, NL2SQL 和 CSpider 三个数据集的训练、预测等任务。 下文中如无特殊说明，则上述配置统称为 config.jsonnet。

## 训练

以训练DuSQL 模型为例

```
bash ./train.sh 1 output/dusql_baseline --config conf/text2sql_dusql.jsonnet --data-root data/DuSQL/preproc
```

参数说明：
* 1 表示并发数，代码会自动选取剩余显存最多的卡使用，当前仅支持 1 卡训练；也可手动指定使用哪张卡，比如使用卡2，则此参数写为 cuda:2
* output/dusql_experiment 表示训练过程保存的模型、预测开发集的结果等保存的目录，按需设置即可
* --config conf/text2sql_dusql.jsonnet 指定本次任务的核心配置。注意 text2sql_dusql.jsonnet 需要替换为特定的配置文件
* --data-root: 指定数据集的根目录。也可通过 --train-set/--dev-set/--test-set/--db 分别指定不同文件的路径

全部的参数可通过 `bash ./run.sh ./script/text2sql_main.py -h` 查看。其中常用参数：
* --pretrain-model: 指定 ERNIE 预训练模型目录
* --batch-size: batch size 大小
* --epochs: 总的训练轮数
* --init-model-params: 热启模型的文件路径

命令行参数的优先级高于配置文件，即如果在命令行指定了config文件包含的参数，则以命令行的设置为准。

### 训练阶段的输出日志
训练过程会输出loss、acc相关日志，日志会同时输出到屏幕和 --output 参数指定目录下的 train.log 文件中。
内容类似：
```
[train] epoch 1, batch 600. loss is 34.1222593689. cost 442.40s
[train] epoch 1, batch 700. loss is 33.3783876610. cost 424.55s
[train] epoch 1/30 loss is 34.777802, cost 2826.10s.
[eval] dev loss 0.000000, acc 1.0000. got best and saved. cost [27.94s]
```
其中，间隔多少steps输出一次日志在conf中设置(train.log_steps)，也可通过命令行参数指定（--log-steps）。
为了提升训练速度，并非每个 epoch 结束都会执行评估，所以 eval 一行的 acc 实际中使用了 epoch 代替。

## 预测

以预测 DuSQL 开发集为例，结果保存到 output/dusql_dev_infer_result.json。

```
bash ./run.sh ./script/text2sql_main.py --mode infer \
         --config conf/text2sql_dusql.jsonnet \
         --data-root data/DuSQL/preproc \
         --test-set data/DuSQL/preproc/dev.pkl \
         --init-model-param output/dusql_baseline/....../model.pdparams \
         --output output/dusql_dev_infer_result.sql
```
其中的 --init-model-param 参数请修改为真实的模型路径。

## 评估

同样以 DuSQL 开发集的预测结果为例。

```
python ./evaluation/text2sql_evaluation.py \
        -g data/DuSQL/gold_dev.sql \
        -t data/DuSQL/db_schema.json \
        -d DuSQL \
        -p output/dusql_dev_infer_result.sql
```

注意，其中的 `data/DuSQL/gold_dev.sql` 需要开发者从 dev.json 中提取得到，格式为“question_id \t sql_query \t db_id”。

# 基线效果

评价指标：Exact Match Accuracy，即预测的SQL query与标准SQL query相等的问题占比，这里“相等”判断时忽略成分的顺序影响，即SELECT A,B和SELECT B,A是相等的。
评估数据集：各数据集的开发集合。
评估设置：基线系统默认代码和配置。

| 数据集  | 准确率(%) |
|-------- | ---    |
| DuSQL   | 64.3   |
| NL2SQL  | 73.0   |
| CSpider | 33.6   |
