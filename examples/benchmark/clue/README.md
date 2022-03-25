# CLUE Benchmark

[CLUE](https://www.cluebenchmarks.com/) 自成立以来发布了多项 NLP 评测基准，包括分类榜单，阅读理解榜单和自然语言推断榜单等，在学术界、工业界产生了深远影响。是目前应用最广泛的中文语言测评指标之一。详细可参考 [CLUE论文](https://arxiv.org/abs/2004.05986)。

本项目基于 PaddlePaddle 在 CLUE 数据集上对领先的开源预训练模型模型进行了充分评测，为开发者在预训练模型选择上提供参考，同时开发者基于本项目可以轻松一键复现模型效果，也可以参加 CLUE 竞赛取得好成绩。

## CLUE 评测结果

使用多种中文预训练模型微调在 CLUE 的各验证集上有如下结果：

| Model                 | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL   |
| --------------------- | ----- | ----- | ------- | ----- | ----- | ----------- | ----- |
| RoBERTa-wwm-ext-large | 76.20 | 59.50 | 62.10   | 84.02 | 79.15 | 90.79       | 82.03 |


其中 AFQMC、TNEWS、 IFLYTEK、CMNLI、OCNLI、CLUEWSC2020 和 CSL 任务使用的评估指标均是 Accuracy。

**NOTE：具体评测方式如下**
1. 以上所有任务均基于 Grid Search 方式进行超参寻优，训练每间隔 100 steps 评估验证集效果，取验证集最优效果作为表格中的汇报指标。

2. Grid Search 超参范围: batch_size: 16, 32, 64; learning rates: 1e-5, 2e-5, 3e-5, 5e-5;

3. 因为 CLUEWSC2020 数据集效果对 batch_size 较为敏感，对CLUEWSC2020 评测时额外增加了 batch_size = 8 的超参搜索。


## 一键复现模型效果

这一小节以 TNEWS 任务为例展示如何一键复现本文的评测结果。

### 启动 CLUE 任务
以 CLUE 的 TNEWS 任务为例，启动 CLUE 任务进行 Fine-tuning 的方式如下：

#### 单卡训练
```shell
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=TNEWS
export LR=3e-5
export BS=16
export EPOCH=6
export MAX_SEQ_LEN=128
export MODEL_PATH=roberta-wwm-ext-large

cd classification
python -u ./run_clue_classifier.py \
    --model_type roberta  \
    --model_name_or_path ${MODEL_PATH} \
    --task_name ${TASK_NAME} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --batch_size ${BS}   \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCH} \
    --logging_steps 100 \
    --seed 42  \
    --save_steps  100 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --output_dir ${MODEL_PATH}/models/${TASK_NAME}/${LR}_${BS}/ \
    --device gpu  \

```

另外，如需评估，传入参数 `--do_eval True` 即可，如果只对读入的 checkpoint 进行评估不训练，可以将 `--do_train` 设为 False。

#### 多卡训练

```shell

unset CUDA_VISIBLE_DEVICES
export TASK_NAME=TNEWS
export LR=3e-5
export BS=32
export EPOCH=6
export MAX_SEQ_LEN=128
export MODEL_PATH=roberta-wwm-ext-large

cd classification
python -m paddle.distributed.launch --gpus "0,1" run_clue_classifier.py \
    --model_type roberta  \
    --model_name_or_path ${MODEL_PATH} \
    --task_name ${TASK_NAME} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --batch_size ${BS}   \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCH} \
    --logging_steps 100 \
    --seed 42  \
    --save_steps  100 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --output_dir ${MODEL_PATH}/models/${TASK_NAME}/${LR}_${BS}/ \
    --device gpu  \

```
其中参数释义如下：
- `model_type` 指示了 Fine-tuning 使用的预训练模型类型，如：ernie、bert、roberta 等，因不同类型的预训练模型可能有不同的 Fine-tuning layer 和 tokenizer。
- `model_name_or_path` 指示了 Fine-tuning 使用的具体预训练模型，可以是 PaddleNLP 提供的预训练模型，可以选择 `model_type` 在[Transformer预训练模型汇总](../../../docs/model_zoo/transformers.rst)中相对应的中文预训练权重。注意这里选择的模型权重要和上面配置的模型类型匹配，例如 model_type 配置的是 ernie，则 model_name_or_path 只能选择 ernie 相关的模型。另，clue 任务应选择中文预训练权重。

- `task_name` 表示 Fine-tuning 的任务，当前支持 AFQMC、TNEWS、IFLYTEK、OCNLI、CMNLI、CSL、CLUEWSC2020。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu' 表示使用GPU, 'xpu' 表示使用百度昆仑卡, 'cpu' 表示使用 CPU。

Fine-tuning 过程将按照 `logging_steps` 和 `save_steps` 的设置打印出如下日志：

```
global step 100/20010, epoch: 0, batch: 99, rank_id: 0, loss: 2.734340, lr: 0.0000014993, speed: 8.7969 step/s
eval loss: 2.720359, acc: 0.0827, eval done total : 25.712125062942505 s
global step 200/20010, epoch: 0, batch: 199, rank_id: 0, loss: 2.608563, lr: 0.0000029985, speed: 2.5921 step/s
eval loss: 2.652753, acc: 0.0945, eval done total : 25.64827537536621 s
global step 300/20010, epoch: 0, batch: 299, rank_id: 0, loss: 2.555283, lr: 0.0000044978, speed: 2.6032 step/s
eval loss: 2.572999, acc: 0.112, eval done total : 25.67190170288086 s
global step 400/20010, epoch: 0, batch: 399, rank_id: 0, loss: 2.631579, lr: 0.0000059970, speed: 2.6238 step/s
eval loss: 2.476962, acc: 0.1697, eval done total : 25.794789791107178 s
```

## 参加 CLUE 竞赛

对于 CLUE 分类任务，可以直接使用本项目中提供的脚本 `classification/predict_clue_classifier.py` 对单个任务进行预测，并将分类结果输出到文件。

以 TNEWS 为例，假设 TNEWS 模型所在路径为 `${TNEWS_MODEL}`，可以运行如下脚本得到模型在测试集上的预测结果，并将预测结果写入地址 `${OUTPUT_DIR}/tnews_predict.json`：

```
cd classification
OUTPUT_DIR=results
mkdir ${OUTPUT_DIR}

python predict_clue_classifier.py \
    --model_type roberta \
    --task_name TNEWS \
    --model_name_or_path ${TNEWS_MODEL}  \
    --output_dir ${OUTPUT_DIR} \
```

对各个任务运行预测脚本，汇总多个结果文件压缩之后，即可提交至CLUE官网进行评测。
