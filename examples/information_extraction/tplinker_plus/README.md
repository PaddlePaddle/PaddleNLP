# TPLinkerPlus

**目录**

- [1.模型简介](#模型简介)
- [2.代码结构](#代码结构)
- [3.数据格式](#数据格式)
  - [3.1 实体抽取数据示例](#实体抽取数据示例)
  - [3.2 关系抽取数据示例](#关系抽取数据示例)
  - [3.3 事件抽取数据示例](#事件抽取数据示例)
  - [3.4 评论观点抽取数据示例](#评论观点抽取数据示例)
- [4.模型训练](#模型训练)
  - [4.1 实体抽取](#实体抽取)
  - [4.2 关系抽取](#关系抽取)
  - [4.3 事件抽取](#事件抽取)
  - [4.4 评论观点抽取](#评论观点抽取)
  - [4.5 更多配置说明](#更多配置说明)
- [5.模型部署](#模型部署)

<a name="模型简介"></a>

## 1. 模型简介

TPLinker提出了一种一阶段联合抽取模型，它能够解决实体、关系重叠的问题，同时不受暴露偏差的影响。TPlinkerPlus在原论文的基础上对任务层设计和损失函数进行了优化，本项目是TPlinkerPlus在PaddlePaddle上的开源实现。

<a name="代码结构"></a>

## 2. 代码结构

```text
├── deploy
│   └── python
│       └── predict.py # python端部署脚本
├── train.py # 训练脚本
├── convert.py # 数据转换脚本
├── evaluate.py # 评估脚本
├── metric.py # 指标计算
├── export_model.py # 动态图参数导出静态图参数脚本
├── utils.py # 工具函数脚本
├── criterion.py # 损失函数
├── model.py # 模型组网
├── components.py # 模型组件
└── README.md # 使用说明
```


<a name="数据格式"></a>

## 3. 数据格式

本章节展示了不同抽取任务数据格式示例，如果是自己的数据根据任务类型转成对应格式即可用于模型训练及评估。

<a name="实体抽取数据示例"></a>

#### 3.1 实体抽取数据示例

```text
{
    "text": "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！", // 文本
    "entity_list": [
        {
            "text": "2月8日上午", // 实体词
            "type": "时间", // 实体类别
            "start_index": 0 // 实体词起始位置索引
        },
        {
            "text": "北京冬奥会自由式滑雪女子大跳台决赛",
            "type": "赛事名称",
            "start_index": 6
        },
        {
            "text": "谷爱凌",
            "type": "选手",
            "start_index": 28
        }
    ] // 实体列表
}
```

<a name="关系抽取数据示例"></a>

#### 3.2 关系抽取数据示例

```text
{
    "text": "《离开》是由张宇谱曲，演唱", // 文本
    "entity_list": [
        {
            "text": "离开", // 实体词
            "type": "DEFAULT", // 实体类别，设置为`DEFAULT`时表示不关注实体类别
            "start_index": 1 // 实体词起始位置索引
        },
        {
            "text": "张宇",
            "type": "DEFAULT",
            "start_index": 6
        }
    ], // 实体列表
    "spo_list": [
        {
            "subject": "离开", // 主体
            "predicate": "作曲", // 谓语
            "object": "张宇", // 客体
            "subject_start_index": 1, // 主体起始位置索引
            "object_start_index": 6 // 客体起始位置索引
        },
        {
            "subject": "离开",
            "predicate": "歌手",
            "object": "张宇",
            "subject_start_index": 1,
            "object_start_index": 6
        }
    ] // 关系列表
}
```

<a name="事件抽取数据示例"></a>

#### 3.3 事件抽取数据示例

```text
{
    "text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", // 文本
    "event_list": [
        {
            "event_type": "组织关系-裁员", // 事件类型
            "trigger": "裁员", // 事件触发词
            "trigger_start_index": 15, // 触发词起始位置索引
            "arguments": [
                {
                    "argument_start_index": 17, // 论元的起始位置索引
                    "role": "裁员人数", // 论元角色
                    "argument": "900余人", // 论元
                },
                {
                    "argument_start_index": 10, // 论元的起始位置索引
                    "role": "时间", // 论元角色
                    "argument": "5月份", // 论元
                }
            ] // 论元列表
        }
    ] // 事件列表
}
```

<a name="评论观点抽取数据示例"></a>

#### 3.4 评论观点抽取数据示例

```text
{
    "text": "位置不错，环境也还可以，早餐不行，", // 文本
    "entity_list": [
        {
            "text": "早餐", // 评价维度
            "type": "评价维度",
            "start_index": 12 // 评价维度起始位置索引
        },
        {
            "text": "位置",
            "type": "评价维度",
            "start_index": 0
        },
        {
            "text": "环境",
            "type": "评价维度",
            "start_index": 5
        },
        {
            "text": "不行", // 观点词
            "type": "观点词",
            "start_index": 14 // 观点词起始位置索引
        },
        {
            "text": "不错",
            "type": "观点词",
            "start_index": 2
        },
        {
            "text": "还可以",
            "type": "观点词",
            "start_index": 8
        }
    ], // 评价维度及观点词列表
    "aso_list": [
        {
            "aspect": "早餐", // 评价维度
            "sentiment": "negative", // 评价维度级情感倾向
            "opinion": "不行", // 观点词
            "aspect_start_index": 12, // 评价维度起始位置索引
            "opinion_start_index": 14 // 观点词起始位置索引
        },
        {
            "aspect": "位置",
            "sentiment": "positive",
            "opinion": "不错",
            "aspect_start_index": 0,
            "opinion_start_index": 2
        },
        {
            "aspect": "环境",
            "sentiment": "positive",
            "opinion": "还可以",
            "aspect_start_index": 5,
            "opinion_start_index": 8
        }
    ] // (Aspect, Sentiment, Opinion) 三元组列表
}
```

<a name="模型训练"></a>

## 4 模型训练

本章节提供了不同类型抽取任务的训练示例。

<a name="实体抽取"></a>

#### 4.1 实体抽取

- 数据下载

    下载[CLUENER细粒度命名实体识别数据集](https://www.cluebenchmarks.com/introduce.html)，解压后存放于ner_data/目录下

    ```text
    .ner_data
    ├── train.json
    └── dev.json
    ```

- 格式转换

    使用`convert.py`转换为训练需要的数据格式并生成标签文件`label_dict.json`。

    ```shell
    python convert.py --data_dir ./ner_data --dataset_name CLUENER
    ```

- 启动训练

    ```shell
    python train.py \
        --task_type entity_extraction \
        --train_path ./ner_data/train_data.json \
        --dev_path ./ner_data/dev_data.json \
        --label_dict_path ./ner_data/label_dict.json \
        --save_dir ./checkpoint \
        --learning_rate 3e-5 \
        --batch_size 16 \
        --max_seq_len 128 \
        --num_epochs 50 \
        --seed 1000 \
        --logging_steps 10 \
        --valid_steps 500 \
        --device gpu
    ```

- 模型评估

    ```shell
    python evaluate.py \
        --task_type entity_extraction \
        --model_path ./checkpoint/model_best \
        --test_path ./ner_data/dev_data.json \
        --batch_size 8
    ```


<a name="关系抽取"></a>

#### 4.2 关系抽取

- 数据下载

    从千言下载数据[DuIE2.0中文关系抽取数据集](https://www.luge.ai/#/luge/dataDetail?id=5)，解压存放于re_data/目录下

    ```text
    .re_data
    ├── duie_dev.json
    ├── duie_schema.json
    └── duie_train.json
    ```

- 格式转换

    使用`convert.py`转换为训练需要的数据格式并生成标签文件`label_dict.json`。

    ```shell
    python convert.py --data_dir ./re_data --dataset_name DuIE2.0
    ```

- 启动训练

    ```shell
    python train.py \
        --task_type relation_extraction \
        --train_path ./re_data/train_data.json \
        --dev_path ./re_data/dev_data.json \
        --label_dict_path ./re_data/label_dict.json \
        --save_dir ./checkpoint \
        --learning_rate 3e-5 \
        --batch_size 16 \
        --max_seq_len 256 \
        --num_epochs 20 \
        --seed 1000 \
        --logging_steps 10 \
        --valid_steps 1000 \
        --device gpu
    ```

- 模型评估

    ```shell
    python evaluate.py \
        --task_type relation_extraction \
        --model_path ./checkpoint/model_best \
        --test_path ./re_data/dev_data.json \
        --batch_size 8
    ```

<a name="事件抽取"></a>

#### 4.3 事件抽取

- 数据下载

    从千言下载数据[DuEE1.0中文事件抽取数据集](https://www.luge.ai/#/luge/dataDetail?id=6)，解压存放于ee_data/目录下

    ```text
    .ee_data
    ├── duee_dev.json
    ├── duee_event_schema.json
    └── duie_train.json
    ```

- 格式转换

    使用`convert.py`转换为训练需要的数据格式并生成标签文件`label_dict.json`。

    ```shell
    python convert.py --data_dir ./ee_data --dataset_name DuEE1.0
    ```

- 启动训练

    ```shell
    python train.py \
        --task_type event_extraction \
        --train_path ./ee_data/train_data.json \
        --dev_path ./ee_data/dev_data.json \
        --label_dict_path ./ee_data/label_dict.json \
        --save_dir ./checkpoint \
        --learning_rate 3e-5 \
        --batch_size 16 \
        --max_seq_len 256 \
        --num_epochs 50 \
        --seed 1000 \
        --logging_steps 10 \
        --valid_steps 500 \
        --device gpu
    ```

<a name="评论观点抽取"></a>

#### 4.4 评论观点抽取

- 数据下载

    下载demo数据[观点抽取示例数据集]()，解压存放于ote_data/目录下（数据集已默认转好格式）

    ```text
    .ote_data
    ├── dev_data.json
    ├── label_dict.json
    └── train_data.json
    ```

- 启动训练

    ```shell
    python train.py \
        --task_type opinion_extraction \
        --train_path ./ote_data/train_data.json \
        --dev_path ./ote_data/dev_data.json \
        --label_dict_path ./ote_data/label_dict.json \
        --save_dir ./checkpoint \
        --learning_rate 3e-5 \
        --batch_size 16 \
        --max_seq_len 128 \
        --num_epochs 50 \
        --seed 1000 \
        --logging_steps 10 \
        --valid_steps 500 \
        --device gpu
    ```

<a name="更多配置说明"></a>

#### 4.5 更多配置说明

- `train.py`脚本配置说明：

    * `train_path`: 训练集文件路径。
    * `dev_path`: 验证集文件路径。
    * `label_dict_path`: 标签文件路径。
    * `task_type`: 选择抽取任务类型，可选有`entity_extraction`, `relation_extraction`, `event_extraction`和`opinion_extraction`。
    * `batch_size`: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为16。
    * `learning_rate`: 学习率，默认为3e-5。
    * `save_dir`: 模型存储路径，默认为`./checkpoint`。
    * `max_seq_len`：最大序列长度，若出现显存不足，请适当调低这一参数；默认为128。
    * `num_epochs`: 训练轮数，默认为20。
    * `seed`: 随机种子，默认为1000.
    * `logging_steps`: 日志打印的间隔steps数，默认10。
    * `valid_steps`: evaluate的间隔steps数，默认100。
    * `device`: 选用什么设备进行训练，可选cpu或gpu。
    * `init_from_ckpt`: 模型参数路径，热启动模型训练；默认为None。
    * `warmup_proportion`：学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
    * `weight_decay`：控制正则项力度的参数，用于防止过拟合，默认为0.0。

- `evaluate.py`脚本配置说明：

    * `task_type`: 选择抽取任务类型，可选有`entity_extraction`, `relation_extraction`, `event_extraction`和`opinion_extraction`。
    * `label_dict_path`: 标签文件地址。
    * `model_path`: 进行评估的模型文件夹路径，路径下需包含模型权重文件`model_state.pdparams`。
    * `test_path`: 进行评估的测试集文件。
    * `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
    * `max_seq_len`: 文本最大长度，输入超过最大长度时会进行截断处理，默认为128。

<a name="模型部署"></a>

## 5. 模型部署

- 模型导出

    ```shell
    python export_model.py \
        --model_path ./checkpoint/model_best \
        --output_path ./export \
        --task_type entity_extraction \
        --label_dict_path ./ner_data/label_dict.json
    ```

    可配置参数说明：

    * `model_path`: 动态图训练保存的参数路径，路径下包含模型参数文件`model_state.pdparams`和模型配置文件`model_config.json`。
    * `output_path`: 静态图参数导出路径，默认导出路径为`./export`。
    * `label_dict_path`: 标签文件路径。
    * `task_type`: 选择抽取任务类型，可选有`entity_extraction`, `relation_extraction`, `event_extraction`和`opinion_extraction`。


- 推理

    ```shell
    python deploy/python/predict.py \
        --model_path_prefix ./export/inference \
        --task_type entity_extraction \
        --label_dict_path ./ner_data/label_dict.json
    ```

    可配置参数说明：

    * `model_path_prefix`: 用于推理的Paddle模型文件路径，需加上文件前缀名称。例如模型文件路径为`./export/inference.pdiparams`，则传入`./export/inference`。
    * `label_dict_path`: 标签文件路径。
    * `task_type`: 选择抽取任务类型，可选有`entity_extraction`, `relation_extraction`, `event_extraction`和`opinion_extraction`。
    * `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
    * `max_seq_len`: 文本最大长度，输入超过最大长度时会进行截断处理，默认为128。
    * `device`: 选择执行预测的设备，可选cpu或gpu。

# References

- **[131250208/TPlinker-joint-extraction](https://github.com/131250208/TPlinker-joint-extraction)**

- **[JunnYu/GPLinker_pytorch](https://github.com/JunnYu/GPLinker_pytorch)**

- **[TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking](https://www.aclweb.org/anthology/2020.coling-main.138.pdf)**
