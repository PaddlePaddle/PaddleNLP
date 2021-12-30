# [EFL](https://arxiv.org/abs/2104.14690)

[EFL](https://arxiv.org/abs/2104.14690) (Entailment as Few-Shot Learner)  提出将 NLP Fine-tune 任务转换统一转换为 Entailment 2 分类任务，为小样本场景下的任务求解提供了新的视角。

## 代码结构及说明
```
|—— train.py # EFL 策略的训练、评估主脚本
|—— dataset.py # EFL 策略针对 FewCLUE 9 个数据集的 Entailment 任务转换逻辑
|—— task_label_description.py # 各个 task 的 label 文本描述
|—— evaluate.py # 针对 FewCLUE 9 个数据集的评估函数
|—— predict.py # 针对 FewCLUE 9 个数据集进行预测
|—— export_model.py # 动态图参数导出静态图参数脚本
|—— deploy # 部署
	|—— python # 部署
		|—— predict.py # python预测部署示例
```

## 基于 FewCLUE 进行 EFL 实验
PaddleNLP 内置了 FewCLUE 数据集，可以直接用来进行 EFL 策略训练、评估、预测，并生成 FewCLUE 榜单的提交结果，参与 FewCLUE 竞赛。

###  数据准备
基于 FewCLUE 数据集进行实验只需要  1 行代码，这部分代码在 `train.py` 脚本中

```
from paddlenlp.datasets import load_dataset

# 通过指定 "fewclue" 和数据集名字 name="tnews" 即可一键加载 FewCLUE 中的 tnews 数据集
train_ds, dev_ds, public_test_ds = load_dataset("fewclue", name="tnews", splits=("train_0", "dev_0", "test_public"))
````
### 模型训练&评估
通过如下命令，指定 GPU 0 卡,  在 FewCLUE 的 `tnews` 数据集上进行训练&评估
```
python -u -m paddle.distributed.launch --gpus "0" \
    train.py \
    --task_name "tnews" \
    --device gpu \
    --negative_num 1 \
    --save_dir "checkpoints" \
    --batch_size 32 \
    --learning_rate 5E-5 \
    --epochs 10 \
    --max_seq_length 512 \
    --rdrop_coef 0 \
```
参数含义说明
- `task_name`: FewCLUE 中的数据集名字
- `negative_num`:  负样本采样个数，对于多分类任务，负样本数量对效果影响很大。负样本数量参数取值范围为 [1, class_num - 1]
- `device`: 使用 cpu/gpu 进行训练
- `save_dir`: 模型存储路径
- `max_seq_length`: 文本的最大截断长度
- `rdrop_coef`: R-Drop 策略 Loss 的权重系数，默认为 0， 若为 0 则未使用 R-Drop 策略

模型每训练 1 个 epoch,  会在验证集上进行评估，并针对测试集进行预测存储到预测结果文件。

### 模型预测
#### 基于动态图的预测
通过如下命令，指定 GPU 0 卡， 在 `FewCLUE` 的 `iflytek` 数据集上进行预测
```
python -u -m paddle.distributed.launch --gpus "0" predict.py \
        --task_name "iflytek" \
        --device gpu \
        --init_from_ckpt "${model_params_file}" \
        --output_dir "./output" \
        --batch_size 32 \
        --max_seq_length 512
```

#### 基于静态图的预测

使用动态图训练结束之后，可以将动态图参数导出成静态图参数，从而获得最优的预测性能，执行如下命令完成动态图转换静态图的功能:
```
python export_model.py --params_path=./checkpoint/model_100/model_state.pdparams --output_path=./output

```

导出静态图模型之后，可以基于静态图做预测部署，`deploy/python/predict.py` 脚本提供了 python 静态图预测示例。以 tnews 数据集为例, 执行如下命令基于静态图预测：
```
python deploy/python/predict.py --model_dir=./output --task_name="tnews"

```

## References
[1] Wang, Sinong, Han Fang, Madian Khabsa, Hanzi Mao, and Hao Ma. “Entailment as Few-Shot Learner.” ArXiv:2104.14690 [Cs], April 29, 2021. http://arxiv.org/abs/2104.14690.
