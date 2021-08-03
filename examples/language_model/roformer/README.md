# RoFormer

## 模型简介

[RoFormer](https://arxiv.org/pdf/2104.09864.pdf) (RoFormer: Enhanced Transformer with Rotary Position Embedding)是一个带有旋转位置嵌入(RoPE)的MLM预训练语言模型。 RoPE是一种相对位置编码方法，具有良好的理论特性。其主要思想是根据绝对位置将上下文嵌入（transformer中的 q，k）乘以旋转矩阵。可以证明上下文嵌入的内积将仅取决于相对位置。
RoPE 是唯一可用于线性注意力的相对位置嵌入。更多详情请参考[论文](https://arxiv.org/pdf/2104.09864.pdf)或[原博客](https://kexue.fm/archives/8265)。EleutherAI还发布了一篇[博客](https://blog.eleuther.ai/rotary-embeddings/)，其中包含有关 RoPE 的直观解释和实验。

本项目是RoFormer在 Paddle 2.x上的开源实现，包含了`THUCNews分类任务`和`Cail2019 Scm任务`的微调代码。

## 快速开始


### 预训练MLM测试
    ```bash
    python test_mlm.py --model_name roformer-chinese-base --text 今天[MASK]很好，我想去公园玩！
    # paddle: 今天[天气||天||阳光||太阳||空气]很好，我想去公园玩！
    python test_mlm.py --model_name roformer-chinese-base --text 北京是[MASK]的首都！
    # paddle: 北京是[中国||谁||中华人民共和国||我们||中华民族]的首都！
    python test_mlm.py --model_name roformer-chinese-char-base --text 今天[MASK]很好，我想去公园玩！
    # paddle: 今天[天||气||都||风||人]很好，我想去公园玩！
    python test_mlm.py --model_name roformer-chinese-char-base --text 北京是[MASK]的首都！
    # paddle: 北京是[谁||我||你||他||国]的首都！
    ```

### THUCNews分类任务数据

THUCNews分类任务所含数据集已在paddlenlp中以API形式提供，无需预先准备，使用`run_thucnews.py`执行微调时将会自动下载。

### 执行Fine-tunning

启动thucnews分类任务的Fine-tuning的方式如下：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" examples/language_model/roformer/run_thucnews.py \
    --model_type roformer \
    --model_name_or_path roformer-chinese-base \
    --max_seq_length 256 \
    --batch_size 64   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./thucnews/ \
    --device gpu \
    --use_amp False
```
其中参数释义如下：
- `model_type` 指示了模型类型，可以选择roformer。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录的地址。
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



### Cail2019_Scm任务数据

Cail2019_Scm分类任务所含数据集已在paddlenlp中以API形式提供，无需预先准备，使用`cail2019_scm.py`执行微调时将会自动下载。

### 执行Fine-tunning

启动cail2019_scm任务的Fine-tuning的方式如下：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" examples/language_model/roformer/run_cail2019_scm.py \
    --model_type roformer_mean_pooling \
    --model_name_or_path roformer-chinese-base \
    --max_seq_length 512 \
    --batch_size 16   \
    --learning_rate 6e-6 \
    --num_train_epochs 20 \
    --logging_steps 60 \
    --save_steps 600 \
    --output_dir ./cail2019_scm/ \
    --device gpu \
    --use_amp False
```

其中参数释义如下：
- `model_type` 指示了模型类型，可以选择roformer_cls_pooling和roformer_mean_pooling两种类型。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录的地址。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示学习率大小，本代码并未使用学习率衰减。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `use_amp` 指示是否启用自动混合精度训练。

基于`roformer-chinese-base`在Cail2019_Scm任务上Fine-tuning后，有如下结果：

|     Model     |    Dev Accuracy   |    Test Accuracy   |
|:-------------:|:-----------------:|:------------------:|
| RoFormer-512  |       0.6307      |        0.6947      |

注: `run_cail2019_scm.py`参考了[原论文微调的代码](https://github.com/ZhuiyiTechnology/roformer/blob/main/finetune_scm.py)，原代码未使用学习率衰减，而是使用了固定学习率6e-6。
