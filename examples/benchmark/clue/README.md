# CLUE Benchmark

[CLUE](https://www.cluebenchmarks.com/)自成立以来发布了多项 NLP 评测基准，包括分类榜单，阅读理解榜单和自然语言推断榜单等，在学术界、工业界产生了深远影响。
是目前应用最广泛的中文语言测评指标之一，被包括阿里巴巴达摩院、腾讯 AI 实验室、华为诺亚方舟实验室在内的 20 多家国内语言实验室所采纳。学术引用 100+，github star 超 6000+。详细可参考 [CLUE论文](https://arxiv.org/abs/2004.05986)

本项目是 CLUE 评测任务 在 Paddle 2.0 上的开源实现。

## 快速开始

### 启动CLUE任务
以 CLUE/TNEWS 任务为例，启动 CLUE 任务进行 Fine-tuning 的方式如下：

#### 单卡训练
```shell
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=TNEWS
export LR=3e-5
export BS=32
export EPOCH=6
export MAX_SEQ_LEN=128
export MODEL_PATH=ernie-1.0

python -u ./run_clue.py \
    --model_type ernie-1.0  \
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
export MODEL_PATH=ernie-1.0

python -m paddle.distributed.launch --gpus "0,1" run_clue.py \
    --model_type ernie-1.0  \
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
- `model_type` 指示了 Fine-tuning 使用的预训练模型类型，如：ernie-1.0、ernie-tiny 等，因不同类型的预训练模型可能有不同的 Fine-tuning layer 和 tokenizer。
- `model_name_or_path` 指示了 Fine-tuning 使用的具体预训练模型，可以是 PaddleNLP 提供的预训练模型 或者 本地的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: /home/xx_model/，目录中需包含 paddle 预训练模型 model_state.pdparams。
如果使用PaddleNLP提供的预训练模型，可以选择 `model_type` 在[Transformer预训练模型汇总](../../../docs/model_zoo/transformers.rst)中相对应的英文预训练权重。注意这里选择的模型权重要和上面配置的模型类型匹配，例如 model_type 配置的是 bert，则 model_name_or_path 只能选择 bert 相关的模型。另，clue 任务应选择中文预训练权重。

- `task_name` 表示 Fine-tuning 的任务，当前支持 AFQMC、TNEWS、IFLYTEK、OCNLI、CMNLI、CSL。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于 learning rate scheduler 产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu' 表示使用GPU, 'xpu' 表示使用百度昆仑卡, 'cpu' 表示使用 CPU。

Fine-tuning过程将按照 `logging_steps` 和 `save_steps` 的设置打印如下日志：

```
global step 100/10008, epoch: 0, batch: 99, rank_id: 0, loss: 2.719922, lr: 0.0000030000, speed: 12.0090 step/s
eval loss: 2.762143, acc: 0.0532, eval done total : 9.460448503494263 s
global step 200/10008, epoch: 0, batch: 199, rank_id: 0, loss: 2.536534, lr: 0.0000060000, speed: 5.6834 step/s
eval loss: 2.326450, acc: 0.26, eval done total : 9.412081480026245 s
global step 300/10008, epoch: 0, batch: 299, rank_id: 0, loss: 1.847109, lr: 0.0000090000, speed: 5.2913 step/s
eval loss: 1.447455, acc: 0.471, eval done total : 9.519582033157349 s
```

使用各种预训练模型进行 Fine-tuning ，在 CLUE 验证集上有如下结果：

| Model CLUE Score      | AFMQC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL   | CMRC2018 | CHID | C3   |
| --------------------- | ----- | ----- | ------- | ----- | ----- | ----------- | ----- | -------- | ---- | ---- |
| ERNIE Base3.0         | 76.23 | 59.29 | 61.02   | 83.62 | 79.86 | 86.84       | 82.77 |          |      |      |
| RoBERTa-wwm-ext-large |       |       |         | 84.02 | 79.15 | 90.13       |       |          |      |      |
