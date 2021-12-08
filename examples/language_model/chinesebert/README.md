# chineseBert with PaddleNLP

[ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information](https://arxiv.org/pdf/2106.16038.pdf)

**摘要：**
最近的汉语预训练模型忽略了汉语特有的两个重要方面：字形和拼音，它们对语言理解具有重要的语法和语义信息。在本研究中，我们提出了汉语预训练，它将汉字的字形和拼音信息纳入语言模型预训练中。字形嵌入是基于汉字的不同字体获得的，能够从视觉特征中捕捉汉字语义，拼音嵌入代表汉字的发音，处理汉语中高度流行的异义现象（同一汉字具有不同的发音和不同的含义）。在大规模的未标记中文语料库上进行预训练后，所提出的ChineseBERT模型在训练步骤较少的基线模型上产生了显著的性能提高。该模型在广泛的中国自然语言处理任务上实现了新的SOTA性能，包括机器阅读理解、自然语言推理、文本分类、句子对匹配和命名实体识别方面的竞争性能。

本项目是 ConvBert 在 Paddle 2.x上的开源实现。

## **数据准备**
数据在data目录下。


## **模型预训练**
模型预训练过程可参考[Electra的README](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/language_model/electra/README.md)

## **Fine-tuning**

### 运行Fine-tuning

#### **使用Paddle提供的预训练模型运行 Fine-tuning**

#### 1、ChnSentiCorp
以ChnSentiCorp数据集为例

#### （1）模型微调：
```shell
# 运行训练
python train_chn.py \
--data_path './data/ChnSentiCorp' \
--device 'gpu' \
--epochs 10 \
--max_seq_length 512 \
--batch_size 8 \
--learning_rate 2e-5 \
--weight_decay 0.0001 \
--warmup_proportion 0.1 \
--seed 2333 \
--save_dir 'outputs/chn' | tee outputs/train_chn.log
```
其中参数释义如下：
- `data_path` 表示微调数据路径
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用多GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。
- `epochs` 表示训练轮数。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示优化器中使用的weight_decay的系数。
- `warmup_steps` 表示动态学习率热启动的step数。
- `seed` 指定随机种子。
- `save_dir` 表示模型保存路径。

#### (2) 评估

在dev和test数据集上acc分别为95.8和96.08，达到论文精度要求。

#### 2、XNLI

#### （1）训练

```bash
python train_xnli.py \
--data_path './data/XNLI' \
--device 'gpu' \
--epochs 5 \
--max_seq_len 256 \
--batch_size 16 \
--learning_rate 1.3e-5 \
--weight_decay 0.001 \
--warmup_proportion 0.1 \
--seed 2333 \
--save_dir outputs/xnli | tee outputs/train_xnli.log
```
其中参数释义如下：
- `data_path` 表示微调数据路径
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用多GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。
- `epochs` 表示训练轮数。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示优化器中使用的weight_decay的系数。
- `warmup_steps` 表示动态学习率热启动的step数。
- `seed` 指定随机种子。
- `save_dir` 表示模型保存路径。

#### （2）评估

test数据集 acc最好结果为81.657,达到论文精度要求。

#### 3、cmrc2018

#### (1) 训练

```shell
# 开始训练
python train_cmrc2018.py \
    --model_type chinesebert \
    --data_dir "data/cmrc2018" \
    --model_name_or_path ChineseBERT-large \
    --max_seq_length 512 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --eval_batch_size 16 \
    --learning_rate 4e-5 \
    --max_grad_norm 1.0 \
    --num_train_epochs 3 \
    --logging_steps 2 \
    --save_steps 20 \
    --warmup_radio 0.1 \
    --weight_decay 0.01 \
    --output_dir outputs/cmrc2018 \
    --seed 1111 \
    --num_workers 0 \
    --use_amp
```
其中参数释义如下：
- `model_type` 指示了模型类型。
- `data_path` 表示微调数据路径
- `model_name_or_path` 模型名称或者路径，支持ChineseBERT-base、ChineseBERT-large两种种规格。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `train_batch_size` 表示训练过程中每次迭代**每张卡**上的样本数目。
- `gradient_accumulation_steps` 梯度累加步数。
- `eval_batch_size` 表示验证过程中每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `max_grad_norm` 梯度裁剪。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `warmup_radio` 表示动态学习率热启动的比例。
- `weight_decay` 表示优化器中使用的weight_decay的系数。
- `output_dir` 表示模型保存路径。
- `seed` 指定随机种子。
- `num_workers` 表示同时工作进程。
- `use_amp` 表示是否使用混合精度。

训练过程中模型会在dev数据集进行评估，其中最好的结果如下所示：

```python

{
    AVERAGE = 82.791
    F1 = 91.055
    EM = 74.526
    TOTAL = 3219
    SKIP = 0
}

```

#### （2）运行eval.py，生成test数据集预测答案

```bash
python eval.py --model_name_or_path outputs/step-340 --n_best_size 35 --max_answer_length 65
```

其中，model_name_or_path为模型路径

#### （3）提交CLUE

test数据集 EM为78.55，达到论文精度要求


### 训练日志

Training logs  can be find [HERE](logs)


## Reference

```bibtex
@article{sun2021chinesebert,
  title={ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information},
  author={Sun, Zijun and Li, Xiaoya and Sun, Xiaofei and Meng, Yuxian and Ao, Xiang and He, Qing and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:2106.16038},
  year={2021}
}

```

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

```bibtex
@article{sun2021chinesebert,
  title={ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information},
  author={Sun, Zijun and Li, Xiaoya and Sun, Xiaofei and Meng, Yuxian and Ao, Xiang and He, Qing and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:2106.16038},
  year={2021}
}

```
