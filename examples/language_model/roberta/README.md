# RoBERTa预训练（Masked Language Modeling）
本项目是RoBERTa模型在 Paddle 2.0上的开源实现，包含了数据tokenization和预训练代码。本项目旨在用简练清晰的代码完成基本预训练任务（仅Masked Language Model）。该代码易于理解，便于修改和定制。
## 快速开始
本目录下包含:

collator.py: 数据采样class

create_data.py: tokenize数据（使用HF datasets导入和预处理wikipedia数据）

run_pretrain.py: 预训练代码

运行如下：

```
python -m paddle.distributed.launch --gpus "0,1" run_pretrain.py \
--model_name_or_path roberta-en-base \
--batch_size 16 \
--learning_rate 1e-4 \
--weight_decay 1e-2 \
--warmup_steps 10000 \
--num_train_epochs 3 \
--input_file wiki \
--output_dir ckp/ \
--logging_steps 100 \
--save_steps 10000 \
--max_steps -1 \
--device gpu \
--max_seq_length 512 \
--amp True
```

其中参数释义如下：
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示AdamW优化器中使用的weight_decay的系数。
- `warmup_steps` 表示动态学习率热启的step数。
- `num_train_epochs` 表示训练轮数。
- `input_file` 表示输入数据的目录，由create_data.py创建
- `output_dir` 表示模型的保存目录。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `max_steps` 表示最大训练步数。若训练`num_train_epochs`轮包含的训练步数大于该值，则达到`max_steps`后就提前结束。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `max_seq_length` 训练数据最大长度。
- `amp` 指示是否启用自动混合精度训练。

### 数据准备
直接运行create_data.py即可，使用wikipedia corpus数据
其中--line_by_line默认False，将把所有数据group到max_seq_length长度。

## 注：
paddle.Dataloader需2.3rc版本才支持HF datasets类，现行版本可以直接在paddle库中的reader.py中注释掉：
```
assert isinstance(dataset, Dataset)
```
https://github.com/PaddlePaddle/Paddle/blob/0ee230a7d3177f791d2a5388ab4dffdccc03f4aa/python/paddle/fluid/reader.py#L335
