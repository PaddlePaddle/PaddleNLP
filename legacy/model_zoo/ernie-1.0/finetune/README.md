# 预训练下游任务 finetune 示例
使用训练中产出的checkpoint，或者paddlenlp内置的模型权重，使用本脚本，用户可以快速对当前模型效果进行评估。

## 运行示例
本文档适配了三大主流下游任务，用户可以根据自己的需求，评估自己所需的数据集。

运行脚本示例如下:

1. 序列分类
```shell
dataset="chnsenticorp_v2"
python run_seq_cls.py \
    --do_train \
    --do_eval \
    --do_predict \
    --model_name_or_path ernie-1.0 \
    --dataset $dataset \
    --output_dir ./tmp/$dataset
```

2. Token分类
```shell
dataset="peoples_daily_ner"
python run_ner.py \
    --do_train \
    --do_eval \
    --do_predict \
    --model_name_or_path ernie-1.0 \
    --dataset $dataset \
    --output_dir ./tmp/$dataset
```

3. 阅读理解
```shell
dataset="cmrc2018"
python run_qa.py \
    --do_train \
    --do_eval \
    --model_name_or_path ernie-1.0 \
    --dataset $dataset \
    --output_dir ./tmp/$dataset
```

## 参数说明

### 传入参数
必须参数
- do_train、do_eval、do_predict分别表示运行训练、评估、测试数据集合。
- do_export 导出为inference预测模型
- model_name_or_path 表示模型权重名称，或者训练中保存的checkpoint地址
- dataset 表示数据集名称
- output_dir 表示运行中，一些checkpoint等参数的输出目录

其他可配置参数：
- per_device_train_batch_size 训练时batch大小
- per_device_eval_batch_size 评估时batch大小
- num_train_epochs 训练epoch数目
- learning_rate 学习率
- max_seq_length 最大序列长度
- weight_decay 训练时优化器对参数衰减系数
- logging_steps 打印日志间隔步数
- eval_steps 评估效果间隔步数
- max_steps 最大训练步数（可覆盖num_train_epochs）


### yaml文件参数
本示例也支持用户在yaml文件中配置参数。用户可以自行修改`config.yaml`文件。

注意：
- 这些参数会重写传入的默认参数，以yaml文件参数为准。
- yaml文件中的batch_size同时等价于per_device_train_batch_size，per_device_eval_batch_size
