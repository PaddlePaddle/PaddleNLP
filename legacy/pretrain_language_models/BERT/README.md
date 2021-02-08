# BERT on PaddlePaddle

[BERT](https://arxiv.org/abs/1810.04805) 是一个迁移能力很强的通用语义表示模型， 以 [Transformer](https://arxiv.org/abs/1706.03762) 为网络基本组件，以双向 `Masked Language Model`  
和 `Next Sentence Prediction` 为训练目标，通过预训练得到通用语义表示，再结合简单的输出层，应用到下游的 NLP 任务，在多个任务上取得了 SOTA 的结果。本项目是 BERT 在 Paddle Fluid 上的开源实现。

同时推荐用户参考[ IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/122282)

### 发布要点

1）完整支持 BERT 模型训练到部署, 包括:

- 支持 BERT GPU 单机、分布式预训练
- 支持 BERT GPU 多卡 Fine-tuning
- 支持 BERT XPU 单机 Fine-tuning
- 提供 BERT 预测接口 demo, 方便多硬件设备生产环境的部署

2）支持 FP16/FP32 混合精度训练和 Fine-tuning，节省显存开销、加速训练过程；

3）提供转换成 Paddle Fluid 参数格式的 [BERT 开源预训练模型](https://github.com/google-research/bert) 供下载，以进行下游任务的 Fine-tuning, 包括如下模型:


| Model | Layers | Hidden size | Heads |Parameters |
| :------| :------: | :------: |:------: |:------: |
| [BERT-Large, Uncased (Whole Word Masking)](https://bert-models.bj.bcebos.com/wwm_uncased_L-24_H-1024_A-16.tar.gz)| 24 | 1024 | 16 | 340M |
| [BERT-Large, Cased (Whole Word Masking)](https://bert-models.bj.bcebos.com/wwm_cased_L-24_H-1024_A-16.tar.gz)| 24 | 1024 | 16 | 340M |
| [RoBERTa-Base, Chinese](https://bert-models.bj.bcebos.com/chinese_roberta_wwm_ext_L-12_H-768_A-12.tar.gz) | 12 | 768 |12 |110M |
| [RoBERTa-Large, Chinese](https://bert-models.bj.bcebos.com/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.tar.gz) | 24 | 1024 |16 |340M |
| [BERT-Base, Uncased](https://bert-models.bj.bcebos.com/uncased_L-12_H-768_A-12.tar.gz) | 12 | 768 |12 |110M |
| [BERT-Large, Uncased](https://bert-models.bj.bcebos.com/uncased_L-24_H-1024_A-16.tar.gz) | 24 | 1024 |16 |340M |
|[BERT-Base, Cased](https://bert-models.bj.bcebos.com/cased_L-12_H-768_A-12.tar.gz)|12|768|12|110M|
|[BERT-Large, Cased](https://bert-models.bj.bcebos.com/cased_L-24_H-1024_A-16.tar.gz)|24|1024|16|340M|
|[BERT-Base, Multilingual Uncased](https://bert-models.bj.bcebos.com/multilingual_L-12_H-768_A-12.tar.gz)|12|768|12|110M|
|[BERT-Base, Multilingual Cased](https://bert-models.bj.bcebos.com/multi_cased_L-12_H-768_A-12.tar.gz)|12|768|12|110M|
|[BERT-Base, Chinese](https://bert-models.bj.bcebos.com/chinese_L-12_H-768_A-12.tar.gz)|12|768|12|110M|

每个压缩包都包含了模型配置文件 `bert_config.json`、参数文件夹 `params` 和词汇表 `vocab.txt`；

4）支持 BERT TensorFlow 模型到 Paddle Fluid 参数的转换。

## 内容速览
- [**安装**](#安装)
- [**预训练**: 预训练数据预处理和训练流程介绍](#预训练)
  - [数据预处理](#数据预处理)
  - [单机训练](#单机训练)
  - [分布式训练](#分布式训练)
- [**Fine-Tuning**: 预训练模型如何应用到特定 NLP 任务上](#nlp-任务的-fine-tuning)
  - [语句和句对分类任务](#语句和句对分类任务)
  - [阅读理解 SQuAD](#阅读理解-squad)
- [**动态混合精度训练**: 利用混合精度加速训练](#动态混合精度训练)
- [**模型转换**: 如何将 BERT TensorFlow 模型转换为 Paddle Fluid 模型](#模型转换)
- [**模型部署**: 多硬件环境模型部署支持](#模型部署)
  - [产出用于部署的 inference model](#保存-inference-model)
  - [inference 接口调用示例](#inference-接口调用示例)

## 目录结构
```text
.
├── data                     # 示例数据
├── inference                # 预测部署示例
├── model                    # 模型定义
├── reader                   # 数据读取
├── utils                    # 辅助文件
├── batching.py              # 构建 batch 脚本
├── convert_params.py        # 参数转换脚本
├── optimization.py          # 优化方法定义
├── predict_classifier.py    # 分类任务生成 inference model
|── run_classifier.py        # 分类任务的 fine tuning
|── run_squad.py             # 阅读理解任务 SQuAD 的 fine tuning
|── test_local_dist.sh       # 本地模拟分布式预训练
|── tokenization.py          # 原始文本的 token 化
|── train.py                 # 预训练过程的定义
|── train.sh                 # 预训练任务的启动脚本
```

## 安装
本项目依赖于 Paddle Fluid **1.7.1** 及以上版本，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装。如果需要进行 TensorFlow 模型到 Paddle Fluid 参数的转换，则需要同时安装 TensorFlow 1.12。

## 预训练

### 数据预处理

以中文模型的预训练为例，可基于中文维基百科数据构造具有上下文关系的句子对作为训练数据，用 [`tokenization.py`](tokenization.py) 中的 CharTokenizer 对构造出的句子对数据进行 token 化处理，得到 token 化的明文数据，然后将明文数据根据词典 [`vocab.txt`](data/demo_config/vocab.txt) 映射为 id 数据并作为训练数据，该示例词典和模型配置[`bert_config.json`](./data/demo_config/bert_config.json)均来自[BERT-Base, Chinese](https://bert-models.bj.bcebos.com/chinese_L-12_H-768_A-12.tar.gz)。

我们给出了 token 化后的示例明文数据: [`demo_wiki_tokens.txt`](./data/demo_wiki_tokens.txt)，其中每行数据为2个 tab 分隔的句子，示例如下:

```
1 . 雏 凤 鸣 剧 团    2 . 古 典 之 门 ： 帝 女 花 3 . 戏 曲 之 旅 ： 第 155 期 心 系 唐 氏 慈 善 戏 曲 晚 会 4 . 区 文 凤 , 郑 燕 虹 1999 编 ， 香 港 当 代 粤 剧 人 名 录 ， 中 大 音 乐 系 5 . 王 胜 泉 , 张 文 珊 2011 编 ， 香 港 当 代 粤 剧 人 名 录 ， 中 大 音 乐 系
```

同时我们也给出了 id 化后的部分训练数据：[`demo_wiki_train.gz`](./data/train/demo_wiki_train.gz)、和测试数据：[`demo_wiki_validation.gz`](./data/validation/demo_wiki_validation.gz)，每行数据为1个训练样本，示例如下:

```
1 7987 3736 8577 8020 2398 969 1399 8038 8021 3221 855 754 7270 7029 1344 7649 4506 2356 4638 676 6823 1298 928 5632 1220 6756 6887 722 769 3837 6887 511 2 4385 3198 6820 3313 1423 4500 511 2;0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1;0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41;1
```

每个样本由4个 '`;`' 分隔的字段组成，数据格式: `token_ids; sentence_type_ids; position_ids; next_sentence_label`；

### 单机训练

利用提供的示例训练数据和测试数据，我们来说明如何进行单机训练。关于预训练的启动方式，可以查看脚本 `train.sh` ，该脚本已经默认以示例数据作为输入，以 GPU 模式进行训练。在开始预训练之前，需要把 CUDA、cuDNN、NCCL2 等动态库路径加入到环境变量 `LD_LIBRARY_PATH` 之中，然后按如下方式即可开始单机多卡预训练

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./train.sh -local y
```

如果采用 CPU 多核的方式进行预训练，则需要通过环境设置所用 CPU 的核数，例如 `export CPU_NUM=5`，否则会占据所有的CPU。


这里需要特别说明的是，参数 `generate_neg_sample` 为 `True` 表示在预训练过程中，`Next Sentence Prediction` 任务的负样本是根据训练数据中的正样本动态生成的，我们给出的样例训练数据 [`demo_wiki_train.gz`](data/train/demo_wiki_train.gz) 只包含 `Next Sentence Prediction` 任务的正样本；如果已事先构造了 `Next Sentence Prediction` 任务的正负样本，则需要将 `generate_neg_sample` 置为 `False`。

预训练任务进行的过程中会输出当前学习率、训练数据所经过的轮数、当前迭代的总步数、训练误差、训练速度等信息，根据 `--validation_steps ${N}` 的配置，每间隔 `N` 步输出模型在验证集的各种指标:

```
current learning_rate:0.000002
epoch: 1, progress: 1/1, step: 60, loss: 10.487363, ppl: 17796.216797, next_sent_acc: 0.560417, speed: 1.060437 steps/s, file: demo_wiki_train.gz
current learning_rate:0.000002
epoch: 1, progress: 1/1, step: 80, loss: 10.262355, ppl: 14686.215820, next_sent_acc: 0.625947, speed: 1.065939 steps/s, file: demo_wiki_train.gz
current learning_rate:0.000003
epoch: 1, progress: 1/1, step: 100, loss: 10.132796, ppl: 12748.593750, next_sent_acc: 0.509135, speed: 1.070796 steps/s, file: demo_wiki_train.gz
[validation_set] epoch: 1, step: 100, loss: 10.036789, global ppl: 12706.841797, batch-averged ppl: 12706.841797, next_sent_acc: 1.000000, speed: 0.987255 steps/s
```

如果用自定义的真实数据进行训练，请参照该脚本对参数做相应的修改。


### 分布式训练
分布式训练可以部署在单机（多进程）或者多机上，我们使用 NCCL 进行节点间的通信。同时需要设置参与训练的所有节点 endpoints 和当前节点 endpoint.
这里我们提供了一个在本地可以模拟分布式训练的样例: `test_local_dist.sh`，这个脚本已经说明主要的分布式训练的启动逻辑。
例如, 如果我们需要在两台机器（192.168.0.16, 192.168.0.17）上进行分布式训练，我们可以在节点 0（192.168.0.16）上运行：

```shell
export worker_endpoints=192.168.0.16:9184,192.168.0.17:9185
export current_endpoint=192.168.0.16:9184
./train.sh -local n
```

节点 1（192.168.0.17）上运行:

```shell
export worker_endpoints=192.168.0.16:9184,192.168.0.17:9185
export current_endpoint=192.168.0.17:9185
./train.sh -local n
```

## NLP 任务的 Fine-tuning

在完成 BERT 模型的预训练后，即可利用预训练参数在特定的 NLP 任务上做 Fine-tuning。以下利用开源的预训练模型，示例如何进行分类任务和阅读理解任务的 Fine-tuning，如果要运行这些任务，请通过 [发布要点](#发布要点) 一节提供的链接预先下载好对应的预训练模型。

### 语句和句对分类任务

对于 [GLUE 数据](https://gluebenchmark.com/tasks)，请运行这个[脚本](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)予以下载; 对于 XNLI 任务，则需分别下载 [XNLI dev/test set](https://bert-data.bj.bcebos.com/XNLI-1.0.zip) 和 [XNLI machine-translated training set](https://bert-data.bj.bcebos.com/XNLI-MT-1.0.zip)，然后解压到同一个目录。以 XNLI 任务为例，启动 Fine-tuning 的方式如下：

```shell
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

BERT_BASE_PATH="chinese_L-12_H-768_A-12"
TASK_NAME='XNLI'
DATA_PATH=/path/to/xnli/data/
CKPT_PATH=/path/to/save/checkpoints/

python -u run_classifier.py --task_name ${TASK_NAME} \
                   --use_cuda true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 32 \
                   --in_tokens false \
                   --init_pretraining_params ${BERT_BASE_PATH}/params \
                   --data_dir ${DATA_PATH} \
                   --vocab_path ${BERT_BASE_PATH}/vocab.txt \
                   --checkpoints ${CKPT_PATH} \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.1 \
                   --validation_steps 100 \
                   --epoch 3 \
                   --max_seq_len 128 \
                   --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
                   --learning_rate 5e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 10 \
                   --verbose true
```

以 XNLI 任务为例，启动 XPU Fine-tuning 的方式如下：

```shell
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_selected_xpus=0
export XPUSIM_DEVICE_MODEL=KUNLUN1
export XPU_PADDLE_TRAIN_L3_SIZE=13631488
export XPU_PADDLE_MAIN_STREAM=0

BERT_BASE_PATH="chinese_L-12_H-768_A-12"
TASK_NAME='XNLI'
DATA_PATH=/path/to/xnli/data/
CKPT_PATH=/path/to/save/checkpoints/

python -u run_classifier.py --task_name ${TASK_NAME} \
                   --use_cuda false \
                   --use_xpu true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 16 \
                   --in_tokens false \
                   --init_pretraining_params ${BERT_BASE_PATH}/params \
                   --data_dir ${DATA_PATH} \
                   --vocab_path ${BERT_BASE_PATH}/vocab.txt \
                   --checkpoints ${CKPT_PATH} \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.1 \
                   --validation_steps 100 \
                   --epoch 3 \
                   --max_seq_len 128 \
                   --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
                   --learning_rate 5e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 10 \
                   --verbose true
```

这里的 `chinese_L-12_H-768_A-12` 即是转换后的中文预训练模型。需要注意的是，BERT on PaddlePaddle 支持按两种方式构建一个 batch 的数据，`in_tokens` 参数影响 `batch_size` 参数的意义，如果 `in_tokens` 为 `true` 则按照 token 个数构建 batch, 如不设定则按照 example 个数来构建 batch. 训练过程中会输出训练误差、训练速度等信息，训练结束后会输出如下所示的在验证集上的测试结果：

```
[dev evaluation] ave loss: 0.622958, ave acc: 0.770281, elapsed time: 8.946956 s
```

xpu训练结束后，验证集上的测试结果：

```
[dev evaluation] ave loss: 0.620479, ave acc: 0.762249, elapsed time: 70.831693 s
[test evaluation] ave loss: 0.616955, ave acc: 0.762275, elapsed time: 142.251840 s
```

### 阅读理解 SQuAD

下载 SQuAD 的数据以及测评脚本到 `$SQUAD_APTH` 目录。

SQuAD v1.1

 [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)

 [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

 [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

 SQuAD v2.0

 [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)

 [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)

 [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

 对于 SQuAD v1.1, 按如下方式启动 Fine-tuning:

 ```shell
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

BERT_BASE_PATH="uncased_L-12_H-768_A-12"
CHECKPOINT_PATH=/path/to/save/checkpoints/
SQUAD_PATH=/path/to/squad/data/

python -u run_squad.py --use_cuda true\
                    --batch_size 12 \
                    --in_tokens false\
                    --init_pretraining_params ${BERT_BASE_PATH}/params \
                    --checkpoints ${CHECKPOINT_PATH} \
                    --vocab_path ${BERT_BASE_PATH}/vocab.txt \
                    --do_train true \
                    --do_predict true \
                    --save_steps 100 \
                    --warmup_proportion 0.1 \
                    --weight_decay  0.01 \
                    --epoch 2 \
                    --max_seq_len 384 \
                    --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
                    --predict_file ${SQUAD_PATH}/dev-v1.1.json \
                    --do_lower_case true \
                    --doc_stride 128 \
                    --train_file ${SQUAD_PATH}/train-v1.1.json \
                    --learning_rate 3e-5 \
                    --lr_scheduler linear_warmup_decay \
                    --skip_steps 10 \
 ```

对预测结果进行测评

```shell
python ${SQUAD_PATH}/evaluate-v1.1.py ${SQUAD_PATH}/dev-v1.1.json ${CHECKPOINT_PATH}/predictions.json
```

会得到类似如下的输出

```text
{"f1": 88.41249612335034, "exact_match": 81.2488174077578}
```

 对于 SQuAD v2.0, 按如下方式启动 Fine-tuning:

```shell
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
BERT_BASE_PATH="uncased_L-12_H-768_A-12"
CHECKPOINT_PATH=/path/to/save/checkpoints/
SQUAD_PATH=/path/tp/squad/data/

python -u run_squad.py --use_cuda true \
                    --batch_size 12 \
                    --in_tokens false\
                    --init_pretraining_params ${BERT_BASE_PATH}/params \
                    --checkpoints ${CHECKPOINT_PATH} \
                    --vocab_path ${BERT_BASE_PATH}/vocab.txt \
                    --do_train true \
                    --do_predict true \
                    --save_steps 100 \
                    --warmup_proportion 0.1 \
                    --weight_decay  0.01 \
                    --epoch 2 \
                    --max_seq_len 384 \
                    --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
                    --predict_file ${SQUAD_PATH}/dev-v2.0.json \
                    --do_lower_case true \
                    --doc_stride 128 \
                    --train_file ${SQUAD_PATH}/train-v2.0.json \
                    --learning_rate 3e-5 \
                    --skip_steps 10 \
                    --lr_scheduler linear_warmup_decay \
                    --version_2_with_negative true
```

训练结束后会在 `${CHECKPOINT_PATH}` 文件夹生成最终预测结果 `prediction.json` ，以及至多 `n_best_size` 个最有可能预测结果 `nbest_predictions`，以及每个问题的最优的非空答案和空答案（亦即 “”）之间的得分差 `null_odds.json`。这个文件可以用来调节一个阈值。当空答案的得分与得分最高的空答案的得分之差大于阈值，则选择空答案。运行以下的脚本可以得到令 `F1` 最大的阈值。

```shell
python ${SQUAD_PATH}/evaluate-v2.0.py ${SQUAD_PATH}/dev-v2.0.json ${CHECKPOINT_PATH}/predictions.json --na-prob-file ${CHECKPOINT_PATH}/null_odds.json
```

会得到类似的输出

```text
{
  "exact": 71.33833066621747,
  "f1": 74.6240213073386,
  "total": 11873,
  "HasAns_exact": 71.00202429149797,
  "HasAns_f1": 77.58282810088231,
  "HasAns_total": 5928,
  "NoAns_exact": 71.67367535744323,
  "NoAns_f1": 71.67367535744323,
  "NoAns_total": 5945,
  "best_exact": 72.23953507959236,
  "best_exact_thresh": -1.5834908485412598,
  "best_f1": 75.12664062186217,
  "best_f1_thresh": -1.410658597946167
}
```

其中会输出 `best_f1_thresh` 是最佳阈值，可以使用这个阈值重新训练，或者从 `nbest_predictions.json` 中重新抽取最终 `prediction`。
训练方法与前面大体相同，只需要设定 `--null_score_diff_threshold` 参数的值为测评时输出的 `best_f1_thresh` ，通常这个值在 -1.0 到 -5.0 之间。

## 动态混合精度训练

预训练过程和 Fine-tuning 均支持 FP16/FP32 动态混合精度训练（Auto Mixed-Precision training, AMP）。在 V100/T4 等支持 tensorcore 的 GPU 设备上，AMP 能显著地加速训练过程。要使能 AMP，只需在前面所述的这些训练启动命令中加入参数

```
--use_fp16=true \
```

为了减少混合精度训练的精度损失，通常在训练过程中计算误差的反向传播时，会将损失函数乘上一个大于 1.0 的因子 `loss_scaling`。在动态混合精度训练过程中，`loss_scaling` 会动态调整，使得训练过程相对于 FP32 尽可能无精度损失。

实验表明，在 GPU V100 上 BERT BASE 的 AMP 训练相对于 FP32 训练有 1.7x 的加速比， BERT LARGE 有 2.0x 的加速比。

更多的细节，可参见[参考论文](https://arxiv.org/abs/1710.03740)。

## 模型转换

除了开放完成转换的 BERT 开源模型，我们还提供了脚本，支持将 BERT TensorFlow 实现所训练的任意预训练模型一键转换为 Paddle Fluid 模型参数。以中文模型的转换为例，只需指定 TensorFlow 模型 checkpoint 的路径和 Paddle Fluid 参数的保存位置，执行

```shell
python convert_params.py \
    --init_tf_checkpoint chinese_L-12_H-768_A-12/bert_model.ckpt \
    --fluid_params_dir params
```
即可完成模型转换。


## 模型部署

深度学习模型需要应用于实际情景，则需要进行模型的部署，把训练好的模型部署到不同的机器上去，这需要考虑不同的硬件环境，包括 GPU/CPU 的环境，单机/分布式集群，或者嵌入式设备；同时还要考虑软件环境，比如部署的机器上是否都安装了对应的深度学习框架；还要考虑运行性能等。但是要求部署环境都安装整个框架会给部署带来不便，为了解决深度学习模型的部署，一种可行的方案是使得模型可以脱离框架运行，Paddle Fluid 采用这种方法进行部署，编译 [Paddle Fluid inference](http://paddlepaddle.org/documentation/docs/zh/1.2/advanced_usage/deploy/inference/build_and_install_lib_cn.html) 库，并且编写加载模型的 `C++` inference 接口。预测的时候则只要加载保存的预测网络结构和模型参数，就可以对输入数据进行预测，不再需要整个框架而只需要 Paddle Fluid inference 库，这带来了模型部署的灵活性。

以语句和语句对分类任务为例子，下面讲述如何进行模型部署。首先需要进行模型的训练，其次是要保存用于部署的模型。最后编写 `C++` inference 程序加载模型和参数进行预测。

前面 [语句和句对分类任务](#语句和句对分类任务) 一节中讲到了如何训练 XNLI 任务的模型，并且保存了 checkpoints。但是值得注意的是这些 checkpoint 中只是包含了模型参数以及对于训练过程中必要的状态信息（参见 [params](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/io_cn.html#save-params) 和 [persistables](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/io_cn.html#save-persistables) ), 现在要生成预测用的 [inference model](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/io_cn.html#permalink-5-save_inference_model)，可以按照下面的步骤进行。

### 保存 inference model

```shell
BERT_BASE_PATH="chinese_L-12_H-768_A-12"
TASK_NAME="XNLI"
DATA_PATH=/path/to/xnli/data/
INIT_CKPT_PATH=/path/to/a/finetuned/checkpoint/

python -u predict_classifier.py --task_name ${TASK_NAME} \
       --use_cuda true \
       --batch_size 64 \
       --data_dir ${DATA_PATH} \
       --vocab_path ${BERT_BASE_PATH}/vocab.txt \
       --init_checkpoint ${INIT_CKPT_PATH} \
       --do_lower_case true \
       --max_seq_len 128 \
       --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
       --do_predict true \
       --save_inference_model_path ${INIT_CKPT_PATH}
```

以上的脚本完成可以两部分工作：

1. 从某一个 `init_checkpoint` 加载模型参数，此时如果设定参数 `--do_predict` 为 `true` 则在 `test` 数据集上进行测试，输出预测结果。
2. 生成对应于 `init_checkpoint` 的 inference model，这会被保存在 `${INIT_CKPT_PATH}/{CKPT_NAME}_inference_model` 目录。

### inference 接口调用示例

使用 `C++` 进行预测的过程需要使用 Paddle Fluid inference 库，具体的使用例子参考 [`inference`](./inference) 目录下的 `README.md`.

下面的代码演示了如何使用 `C++` 进行预测，更多细节请见 [`inference`](./inference) 目录下的例子，可以参考例子写 inference。

``` cpp
#include <paddle_inference_api.h>

// create and set configuration
paddle::NativeConfig config;
config.model_dir = "xxx";
config.use_gpu = false;

// create predictor
auto predictor = CreatePaddlePredictor(config);

// create input tensors
paddle::PaddleTensor src_id;
src.dtype = paddle::PaddleDType::INT64;
src.shape = ...;
src.data.Reset(...);

paddle::PaddleTensor pos_id;
paddle::PaddleTensor segmeng_id;
paddle::PaddleTensor input_mask;

// create iutput tensors and run prediction
std::vector<paddle::PaddleTensor> output;
predictor->Run({src_id, pos_id, segmeng_id, input_mask}, &output);

// XNLI task for example
std::cout << "example_id\tcontradiction\tentailment\tneutral";
for (size_t i = 0; i < output.front().data.length() / sizeof(float); i += 3) {
  std::cout << static_cast<float *>(output.front().data.data())[i] << "\t"
            << static_cast<float *>(output.front().data.data())[i + 1] << "\t"
            << static_cast<float *>(output.front().data.data())[i + 2] << std::endl;
}
```
