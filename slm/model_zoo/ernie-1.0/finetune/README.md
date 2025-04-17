# 预训练下游任务 finetune 示例
使用训练中产出的 checkpoint，或者 paddlenlp 内置的模型权重，使用本脚本，用户可以快速对当前模型效果进行评估。

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

2. Token 分类
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
- do_train、do_eval、do_predict 分别表示运行训练、评估、测试数据集合。
- do_export 导出为 inference 预测模型
- model_name_or_path 表示模型权重名称，或者训练中保存的 checkpoint 地址
- dataset 表示数据集名称
- output_dir 表示运行中，一些 checkpoint 等参数的输出目录

其他可配置参数：
- per_device_train_batch_size 训练时 batch 大小
- per_device_eval_batch_size 评估时 batch 大小
- num_train_epochs 训练 epoch 数目
- learning_rate 学习率
- max_seq_length 最大序列长度
- weight_decay 训练时优化器对参数衰减系数
- logging_steps 打印日志间隔步数
- eval_steps 评估效果间隔步数
- max_steps 最大训练步数（可覆盖 num_train_epochs）


### yaml 文件参数
本示例也支持用户在 yaml 文件中配置参数。用户可以自行修改`config.yaml`文件。

注意：
- 这些参数会重写传入的默认参数，以 yaml 文件参数为准。
- yaml 文件中的 batch_size 同时等价于 per_device_train_batch_size，per_device_eval_batch_size
