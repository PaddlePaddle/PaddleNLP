# P-Tuning

[GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)

## 算法简介

P-tuning 引入可学习的连续型提示向量 prompt embeddings 参数, 让模型自己去学习最优的 prompt embedding, 而不再依赖人工去设置自然语言形式的提示（Prompt）信息。P-Tuning 算法的数据和模型定义如下图所示，对应于数据预处理模块 `SoftTemplate` 和标签词映射模块 `MaskedLMVerbalizer`，详细介绍及定义方法参见 [Prompt API 文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/advanced_guide/prompt.md)。

![p-tuning](https://user-images.githubusercontent.com/25607475/204214359-3036c6c6-f101-4a5f-958c-abe0e40c243a.png)


## 快速开始

CLUE（Chinese Language Understanding Evaluation）作为中文语言理解权威测评榜单，在学术界和工业界都有着广泛影响。FewCLUE 是其设立的中文小样本学习测评子榜，旨在探索小样本学习最佳模型和中文实践。PaddleNLP 内置了 FewCLUE 数据集，可以直接用来进行 PET 策略训练、评估、预测，并生成 FewCLUE 榜单的提交结果，参与 FewCLUE 竞赛。
PaddleNLP 内置了 FewCLUE 数据集，可以直接用来进行 P-tuning 策略训练、评估、预测，并生成 FewCLUE 榜单的提交结果，参与 FewCLUE 竞赛。

### 代码结构及说明
```
├── run_train.py # P-Tuning 算法提示学习脚本
├── data.py      # 数据集构造、数据增强
├── utils.py     # FewCLUE 提交结果保存等工具函数
└── prompt/      # FewCLUE 各数据集的 prompt 定义文件
```

###  数据准备

读取 FewCLUE 数据集只需要 1 行代码，这部分代码在 `data.py` 脚本中。以情感分类数据集 `eprstmt` 为例：
```
from paddlenlp.datasets import load_dataset

# 通过指定 "fewclue" 和数据集名字 name="eprstmt" 即可一键加载 FewCLUE 中的 eprstmt 数据集
train_ds, dev_ds, public_test_ds = load_dataset("fewclue", name="eprstmt", splits=("train_0", "dev_0", "test_public"))
```

### 模型训练、评估、预测

通过如下命令，指定 GPU 0 卡， 使用一个连续型提示向量在 FewCLUE 的 `eprstmt` 数据集上进行训练和评估。如果要使用多个可学习连续型提示向量，可修改 `./prompt/` 目录下相应的文件，修改 `soft` 的长度属性 `length` 即可。
```
python -u -m paddle.distributed.launch --gpus "0" run_train.py \
    --output_dir checkpoint_eprstmt \
    --task_name eprstmt \
    --split_id few_all \
    --prompt_path prompt/eprstmt.json \
    --prompt_index 0 \
    --do_train \
    --do_eval \
    --do_test \
    --do_predict \
    --do_label \
    --max_steps 1000 \
    --learning_rate 3e-5 \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 5  \
    --per_device_train_batch_size 16 \
    --max_seq_length 128 \
    --load_best_model_at_end \
    --metric_for_best_model accuracy \
    --save_total_limit 1
```

参数含义说明
- `task_name`: FewCLUE 中的数据集名字
- `split_id`: 数据集编号，包括0, 1, 2, 3, 4 和 few_all
- `prompt_path`: prompt 定义文件名
- `prompt_index`: 使用定义文件中第 `prompt_index` 个 prompt
- `augment_type`: 数据增强策略，可选 swap, delete, insert, substitute
- `num_augment`: 数据增强策略为每个样本生成的样本数量
- `word_augment_percent`: 每个序列中数据增强词所占的比例
- `pseudo_data_path`: 使用模型标注的伪标签数据文件路径
- `do_label`: 是否使用训练后的模型给无标签数据标注伪标签
- `do_test`: 是否在公开测试集上评估模型效果
- `model_name_or_path`: 预训练模型名，默认为 `ernie-1.0-large-zh-cw`
- `use_rdrop`: 是否使用对比学习策略 R-Drop
- `alpha_rdrop`: R-Drop 损失值权重，默认为 0.5
- `dropout`: 预训练模型的 dropout 参数值，用于 R-Drop 策略中参数配置
- `export_type`: 模型导出格式，默认为 `paddle`，动态图转静态图
- 更多配置参考 [Trainer 参数文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/trainer.md#trainingarguments-%E5%8F%82%E6%95%B0%E4%BB%8B%E7%BB%8D) 和 [PromptTrainer 参数文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/advanced_guide/prompt.md#prompttrainer%E5%8F%82%E6%95%B0%E5%88%97%E8%A1%A8)

### 模型部署

Coming soon...

## References
[1]X. Liu et al., “GPT Understands, Too,” arXiv:2103.10385 [cs], Mar. 2021, Accessed: Mar. 22, 2021. [Online]. Available: http://arxiv.org/abs/2103.10385
