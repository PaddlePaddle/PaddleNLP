# 千言-问题匹配鲁棒性评测基线

我们基于预训练模型 ERNIE-Gram 在[千言-问题匹配鲁棒性评测竞赛]() 建立了 Baseline 方案和评测结果.

### 评测效果
本项目分别基于ERNIE-1.0、Bert-base-chinese、ERNIE-Gram 3 个中文预训练模型训练了单塔 Point-wise 的匹配模型，基于 ERNIE-Gram 的模型效果显著优于其它 2 个预训练模型。

| 模型  | dev acc | test-A acc | test-B acc|
| ---- | -----|--------|------- |
| ernie-1.0-base | 86.96 |76.20 | 77.50|
| bert-base-chinese | 86.93| 76.90 |77.60 |
| ernie-gram-zh | 87.66 | **80.80** | **81.20** |


## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：
```
question_matching/
├── model.py # 匹配模型组网
├── data.py # 训练样本的数据读取、转换逻辑
├── predict.py # 模型预测脚本，输出测试集的预测结果: 0,1
└── train.py # 模型训练评估
```

### 模型训练
本项目使用竞赛提供的 LCQMC、BQ、OPPO 这 3 个数据集的训练集合集作为训练集， 这 3 个数据集的验证集合集作为验证集，可以运行如下命令，即可复现本项目基于 ERNIE-Gram 的基线模型:

```shell
$unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "0,1,2,3" train.py \
       --train_set ${train_set} \
       --dev_set ${dev_set} \
       --device gpu \
       --eval_step 100 \
       --save_dir ./checkpoints \
       --train_batch_size 32 \
       --learning_rate 2E-5
```

可支持配置的参数：
* `train_set`: 训练集的文件
* `dev_set`：验证集数据文件
* `train_batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.0
* `epochs`: 训练轮次，默认为3。
* `warmup_proption`：可选，学习率 warmup 策略的比例，如果 0.1，则学习率会在前 10% 训练 step 的过程中从 0 慢慢增长到 learning_rate, 而后再缓慢衰减，默认为 0.0。
* `init_from_ckpt`：可选，模型参数路径，热启动模型训练；默认为None。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── model_100
│   ├── model_state.pdparams
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:**
* 如需恢复模型训练，则可以设置`init_from_ckpt`， 如`init_from_ckpt=checkpoints/model_100/model_state.pdparams`。


### 开始预测
训练完成后，在指定的 checkpoints 路径下会自动存储在验证集评估指标最高的模型，运行如下命令开始生成预测结果:
```shell
$ unset CUDA_VISIBLE_DEVICES
python -u \
    predict.py \
    --device gpu \
    --params_path "./checkpoints/1000/model_state.pdparams" \
    --batch_size 128 \
    --input_file "${test_set}" \
    --result_file "predict_result"
```

输出预测结果示例如下:
```text
0
1
0
1
```
### 提交进行评测
提交预测结果进行评测
