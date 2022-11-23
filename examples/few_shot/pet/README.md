# PET


[Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676)


## 摘要

自然语言处理任务可以通过给预训练模型提供“任务描述”等方式来进行无监督学习，但效果一般低于有监督训练。而 Pattern-Exploiting Training (PET) 是一种半监督方法，通过将输入转换为完形填空形式的短语来帮助语言模型理解任务。然后用这些短语来给无标注数据打软标签。最后在得到的标注数据集上用有监督方法进行训练。在小样本设置下，PET 在部分任务上远超有监督学习和强半监督学习方法。


## 快速开始


## 代码结构说明
```
|—— train.py # PET 算法训练、评估主程序入口
|—— utils.py # 数据集构造、数据增强等工具函数
|—— prompt/  # FewCLUE 各数据集的 prompt 定义文件
```

## 基于 FewCLUE 进行 PET 实验
PaddleNLP 内置了 FewCLUE 数据集，可以直接用来进行 PET 策略训练、评估、预测，并生成 FewCLUE 榜单的提交结果，参与 FewCLUE 竞赛。

###  数据准备
基于 FewCLUE 数据集进行实验只需要 1 行代码，这部分代码在 `utils.py` 脚本中

```
from paddlenlp.datasets import load_dataset

# 通过指定 "fewclue" 和数据集名字 name="tnews" 即可一键加载 FewCLUE 中的 tnews 数据集
train_ds, dev_ds, public_test_ds = load_dataset("fewclue", name="tnews", splits=("train_0", "dev_0", "test_public"))
```

### 模型训练、评估、预测

通过如下命令，指定 GPU 0 卡,  在 FewCLUE 的 `tnews` 数据集上进行训练&评估
```
python -u -m paddle.distributed.launch --gpus "0" train.py \
    --output_dir checkpoint_tnews \
    --task_name tnews \
    --split_id few_all \
    --prompt_path prompt/tnews.json \
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


## References
[1] Schick, Timo, and Hinrich Schütze. “Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference.” ArXiv:2001.07676 [Cs], January 25, 2021. http://arxiv.org/abs/2001.07676.
