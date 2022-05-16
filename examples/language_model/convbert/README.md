# ConvBert with PaddleNLP

[ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496)

**摘要：**
像BERT及其变体这样的预训练语言模型最近在各种自然语言理解任务中取得了令人印象深刻的表现。然而，BERT严重依赖全局自注意力块，因此需要大量内存占用和计算成本。
虽然它的所有注意力头从全局角度查询整个输入序列以生成注意力图，但我们观察到一些头只需要学习局部依赖，这意味着存在计算冗余。
因此，我们提出了一种新颖的基于跨度的动态卷积来代替这些自注意力头，以直接对局部依赖性进行建模。新的卷积头与其余的自注意力头一起形成了一个新的混合注意力块，在全局和局部上下文学习中都更有效。
我们为 BERT 配备了这种混合注意力设计并构建了一个ConvBERT模型。实验表明，ConvBERT 在各种下游任务中明显优于BERT及其变体，具有更低的训练成本和更少的模型参数。
值得注意的是，ConvBERT-base 模型达到86.4GLUE分数，比ELECTRA-base高0.7，同时使用不到1/4的训练成本。

本项目是 ConvBert 在 Paddle 2.x上的开源实现。

## **数据准备**

### Fine-tuning数据
Fine-tuning 使用GLUE数据，这部分Paddle已提供，在执行Fine-tuning 命令时会自动下载并加载


## **模型预训练**
模型预训练过程可参考[Electra的README](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/electra/README.md)

## **Fine-tuning**

### 运行Fine-tuning

#### **使用Paddle提供的预训练模型运行 Fine-tuning**

以 GLUE/SST-2 任务为例，启动 Fine-tuning 的方式如下：
```shell
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=SST-2

python -u examples/language_model/convbert/run_glue.py \
    --model_type convbert \
    --model_name_or_path convbert-small \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 256   \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --logging_steps 100 \
    --save_steps 100 \
    --output_dir ./glue/$TASK_NAME/ \
    --device gpu
```
其中参数释义如下：
- `model_type` 指示了模型类型，当前支持BERT、ELECTRA、ERNIE、CONVBERT模型。
- `model_name_or_path` 模型名称或者路径，其中convbert模型当前仅支持convbert-small、convbert-medium-small、convbert-base几种规格。
- `task_name` 表示 Fine-tuning 的任务，当前支持CoLA、SST-2、MRPC、STS-B、QQP、MNLI、QNLI、RTE。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用多GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。

Fine-tuning过程将按照 `logging_steps` 和 `save_steps` 的设置打印如下格式的日志：

```
global step 100/792, epoch: 0, batch: 99, rank_id: 0, loss: 0.333723, lr: 0.0000970547, speed: 3.6162 step/s
eval loss: 0.295912, acc: 0.8623853211009175, eval done total : 0.5295147895812988 s
global step 200/792, epoch: 0, batch: 199, rank_id: 0, loss: 0.243273, lr: 0.0000830295, speed: 3.6822 step/s
eval loss: 0.249330, acc: 0.8899082568807339, eval done total : 0.508596658706665 s
global step 300/792, epoch: 1, batch: 35, rank_id: 0, loss: 0.166950, lr: 0.0000690042, speed: 3.7250 step/s
eval loss: 0.307219, acc: 0.8956422018348624, eval done total : 0.5816614627838135 s
global step 400/792, epoch: 1, batch: 135, rank_id: 0, loss: 0.185729, lr: 0.0000549790, speed: 3.6896 step/s
eval loss: 0.201950, acc: 0.9025229357798165, eval done total : 0.5364704132080078 s
global step 500/792, epoch: 1, batch: 235, rank_id: 0, loss: 0.132817, lr: 0.0000409537, speed: 3.7708 step/s
eval loss: 0.239518, acc: 0.9094036697247706, eval done total : 0.5128316879272461 s
global step 600/792, epoch: 2, batch: 71, rank_id: 0, loss: 0.163107, lr: 0.0000269285, speed: 3.7303 step/s
eval loss: 0.199408, acc: 0.9139908256880734, eval done total : 0.5226929187774658 s
global step 700/792, epoch: 2, batch: 171, rank_id: 0, loss: 0.082950, lr: 0.0000129032, speed: 3.7664 step/s
eval loss: 0.236055, acc: 0.9025229357798165, eval done total : 0.5140993595123291 s
global step 792/792, epoch: 2, batch: 263, rank_id: 0, loss: 0.025735, lr: 0.0000000000, speed: 4.1180 step/s
eval loss: 0.226449, acc: 0.9013761467889908, eval done total : 0.5103530883789062 s
```

使用convbert-small预训练模型进行单卡Fine-tuning ，在验证集上有如下结果（这里各类任务的结果是运行1次的结果）：

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| CoLA  | Matthews corr                | 56.22       |
| SST-2 | acc.                         | 91.39       |
| MRPC  | acc./F1                      | 87.70       |
| STS-B | Pearson/Spearman corr        | 86.34       |
| QQP   | acc./F1                      | 85.47       |
| MNLI  | matched acc./mismatched acc. | 81.87       |
| QNLI  | acc.                         | 87.71       |
| RTE   | acc.                         | 66.06       |

注：acc.是Accuracy的简称，表中Metric字段名词取自[GLUE论文](https://openreview.net/pdf?id=rJ4km2R5t7)



## Reference
[Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan. ConvBERT: Improving BERT with Span-based Dynamic Convolution. In NeurIPS 2020](https://arxiv.org/abs/2008.02496)
