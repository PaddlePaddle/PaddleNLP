# ELECTRA with PaddleNLP

[ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) 在[BERT](https://arxiv.org/abs/1810.04805)的基础上对其预训练过程进行了改进：预训练由两部分模型网络组成，称为Generator和Discriminator，各自包含1个BERT模型。Generator的预训练使用和BERT一样的Masked Language Model(MLM)任务，但Discriminator的预训练使用Replaced Token Detection(RTD)任务（主要改进点）。预训练完成后，使用Discriminator作为精调模型，后续的Fine-tuning不再使用Generator。根据论文中给出的实验结果，在和BERT具有相同的模型参数、预训练计算量一样的情况下，GLUE得分比BERT明显好，small模型为79.9：75.1，Base模型为85.1：82.2，Large模型为89.0：87.2。作者给出的原因是：
1. 相比MLM任务只着眼于输入中15%词的完形填空，ELECTRA的RTD任务着眼于整个输入内容，模型更具有全局观
2. 应用了GAN生成对抗的思想，替换词的时候使用Generator做逼真替换而不是随机替换，加快Discriminator收敛
3. 虽然预训练模型包括Generator和Discriminator，但也不是完全照搬GAN，和常规GAN不一样的地方：
   - 输入为真实文本，常规GAN输入为随机噪声
   - 生成器的输入输出都为句子，而句子中的字词都是离散的，因此判别器的梯度无法传给生成器，而常规GAN是可以传递的
   - 如果生成出原来的词，则为正例，而常规GAN生成的都是负例

本项目是 ELECTRA 在 Paddle 2.0上的开源实现。

## 发布要点

1. 动态图ELECTRA模型，支持 Fine-tuning，在 GLUE 所有任务上进行了验证。
2. 支持 ELECTRA Pre-training。

## NLP 任务的 Fine-tuning
使用../glue/run_glue.py运行，详细可参考../glue/README.md，有两种方式：
1. 使用已有的预训练模型运行 Fine-tuning。
2. 运行 ELECTRA 模型的预训练后，使用预训练模型运行 Fine-tuning（需要很多资源）。

下面的例子基于方式1进行介绍。

### 语句和句对分类任务

以 GLUE/SST-2 任务为例，启动 Fine-tuning 的方式如下（`paddlenlp` 要已经安装或能在 `PYTHONPATH` 中找到）：

```shell
export CUDA_VISIBLE_DEVICES=0,1
export TASK_NAME=SST-2

cd ../glue/ && python -u ./run_glue.py \
    --model_type electra \
    --model_name_or_path electra-small \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/$TASK_NAME/ \
    --n_gpu 1 \

```

其中参数释义如下：
- `model_type` 指示了模型类型，当前支持BERT、ELECTRA模型。
- `model_name_or_path` 指示了使用哪种预训练模型，对应有其预训练模型和预训练时使用的 tokenizer，当前支持electra-small、electra-base、electra-large。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `task_name` 表示 Fine-tuning 的任务，当前支持CoLA、SST-2、MRPC、STS-B、QQP、MNLI、QNLI、RTE。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `n_gpu` 表示使用的 GPU 卡数。若希望使用多卡训练，将其设置为指定数目即可；若为0，则使用CPU。

训练过程将按照 `logging_steps` 和 `save_steps` 的设置打印如下日志：

```
global step 6310/6315, epoch: 2, batch: 2099, rank_id: 0, loss: 0.035772, lr: 0.0000000880, speed: 3.1527 step/s
global step 6311/6315, epoch: 2, batch: 2100, rank_id: 0, loss: 0.056789, lr: 0.0000000704, speed: 3.4201 step/s
global step 6312/6315, epoch: 2, batch: 2101, rank_id: 0, loss: 0.096717, lr: 0.0000000528, speed: 3.4694 step/s
global step 6313/6315, epoch: 2, batch: 2102, rank_id: 0, loss: 0.044982, lr: 0.0000000352, speed: 3.4513 step/s
global step 6314/6315, epoch: 2, batch: 2103, rank_id: 0, loss: 0.139579, lr: 0.0000000176, speed: 3.4566 step/s
global step 6315/6315, epoch: 2, batch: 2104, rank_id: 0, loss: 0.046043, lr: 0.0000000000, speed: 3.4590 step/s
eval loss: 0.549763, acc: 0.9151376146788991, eval done total : 1.8206987380981445 s
```

使用electra-small预训练模型进行单卡 Fine-tuning ，在验证集上有如下结果：

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| CoLA  | Matthews corr                | 58.22       |
| SST-2 | acc.                         | 91.85       |
| MRPC  | acc./F1                      | 88.24       |
| STS-B | Pearson/Spearman corr        | 87.24       |
| QQP   | acc./F1                      | 88.83       |
| MNLI  | matched acc./mismatched acc. | 82.45       |
| QNLI  | acc.                         | 88.61       |
| RTE   | acc.                         | 66.78       |

注：acc.是Accuracy的简称，表中Metric字段名词取自[GLUE论文](https://openreview.net/pdf?id=rJ4km2R5t7)

## 预训练
预训练需要BookCorpus数据，当前BookCorpus数据已不再开源，可以使用其它数据替代，只要是纯文本数据即可。
例如[Gutenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html)
下面例子假设数据在./BookCorpus/，数据文件为纯文本train.data

```shell
export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR=./BookCorpus/

python -u ./run_pretrain.py \
    --model_type electra \
    --model_name_or_path electra-small \
    --train_batch_size 96 \
    --learning_rate 5e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 4 \
    --input_dir $DATA_DIR \
    --output_dir ./tmp2/ \
    --logging_steps 1 \
    --save_steps 20000 \
    --max_steps 1000000 \
    --n_gpu 2
```
