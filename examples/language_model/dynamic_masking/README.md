# 简化动态预训练代码（Masked Language Modeling）
本项目是ERNIE 2.0模型在 Paddle 2.0上的开源实现，包含了数据tokenization和预训练代码。本项目旨在用简练清晰的代码完成基本预训练任务（仅Masked Language Model）。该代码易于理解，便于修改和定制。
## 快速开始
本目录下包含：
collator.py: 数据采样class
modeling.py: MLM模型class
ru_pretrain.py: 预训练代码
运行如下：
```shell
python -m paddle.distributed.launch --gpus "0,1" run_pretrain.py \
--model_name_or_path ernie-2.0-en \
--batch_size 32 \
--learning_rate 1e-4 \
--weight_decay 1e-2 \
--warmup_steps 10000 \
--num_train_epochs 3 \
--input_file /work/test/data.txt \
--output_dir /work/test/model_save/ \
--logging_steps 100 \
--save_steps 1 \
--max_steps 100000 \
--device gpu:0 \
--max_seq_length 128 \
--amp True
```

其中参数释义如下：
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示AdamW优化器中使用的weight_decay的系数。
- `adam_epsilon` 表示AdamW优化器中使用的epsilon值。
- `warmup_steps` 表示动态学习率热启的step数。
- `num_train_epochs` 表示训练轮数。
- `input_file` 表示输入数据的目录，该目录下所有文件名中包含training的文件将被作为训练数据。
- `output_dir` 表示模型的保存目录。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `max_steps` 表示最大训练步数。若训练`num_train_epochs`轮包含的训练步数大于该值，则达到`max_steps`后就提前结束。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `amp` 指示是否启用自动混合精度训练。
### 数据准备
原始数据默认每一行为一条训练样本，数据读取函数为 read_data(fileName):
```shell
def read_data(fileName):
    # Read the raw txt file, each line represents one sample.
    # customize the read function if necessary
    with open(fileName, 'r') as f:
        for line in f:
            yield line
        f.close()
```
可以根据需求更改。
