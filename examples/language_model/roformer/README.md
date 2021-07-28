# RoFormer

## 模型简介

[RoFormer](https://arxiv.org/pdf/2104.09864.pdf) (RoFormer: Enhanced Transformer with Rotary Position Embedding)是一个带有旋转位置嵌入(RoPE)的MLM预训练语言模型。 RoPE是一种相对位置编码方法，具有良好的理论特性。其主要思想是根据绝对位置将上下文嵌入（transformer中的 q，k）乘以旋转矩阵。可以证明上下文嵌入的内积将仅取决于相对位置。
RoPE 是唯一可用于线性注意力的相对位置嵌入。更多详情请参考[论文](https://arxiv.org/pdf/2104.09864.pdf)或[原博客](https://kexue.fm/archives/8265)。EleutherAI还发布了一篇[博客](https://blog.eleuther.ai/rotary-embeddings/)，其中包含有关 RoPE 的直观解释和实验。

本项目是RoFormer在 Paddle 2.0上的开源实现，包含了`THUCNews分类任务`的微调代码。

## 快速开始
### THUCNews分类任务数据

THUCNews分类任务所含数据集已在paddlenlp中以API形式提供，无需预先准备，使用`run_thucnews.py`执行微调时将会自动下载。

### 执行Fine-tunning

启动Fine-tuning的方式如下：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_thucnews.py \
    --model_type roformer \
    --model_name_or_path roformer-chinese-base \
    --max_seq_length 256 \
    --batch_size 64   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/ \
    --device gpu \
    --use_amp False
```

其中参数释义如下：
- `model_type` 指示了模型类型，使用RoFormer模型时设置为roformer即可。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。注：`roformer-chinese-base`等对应使用的预训练模型转自[huggingface/transformers](https://github.com/huggingface/transformers)，具体可参考当前目录下converter中的内容。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `use_amp` 指示是否启用自动混合精度训练。

基于`roformer-chinese-base`在THUCNews分类任务上Fine-tuning后，在验证集上有如下结果：

| Task  | Metric                       | Result            |
|:-----:|:----------------------------:|:-----------------:|
| THUCNews | Accuracy                     |      0.98      |
