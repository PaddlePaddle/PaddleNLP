# 无监督语义匹配模型 [SimCSE](https://arxiv.org/abs/2104.08821)

我们实现了 SimCSE 模型，并借鉴 ESimCSE 论文思想，通过 Word Repetition(WR) 策略进一步提升了 SimCSE 模型效果，在 4 个权威中文语义匹配数据集上做了充分效果评测。SimCSE 模型适合缺乏监督数据，但是又有大量无监督数据的匹配和检索场景。

## 效果评估
本项目分别使用 LCQMC、BQ_Corpus、STS-B、ATEC 这 4 个中文语义匹配数据集的训练集作为无监督训练集(仅使用文本信息，不使用 Label)，并且在各自数据集上的验证集上进行效果评估，评估指标采用 SimCSE 论文中采用的 Spearman 相关系数，Spearman 相关系数越高，表示模型效果越好。中文数据集的下载地址为：[下载地址](https://paddlenlp.bj.bcebos.com/datasets/senteval_cn.zip)

### 中文语义匹配数据集效果

| 模型| LCQMC | BQ_Corpus|STS-B|ATEC|
|-------|-------|-----|------|-----|
|SimCSE| 57.01 | **51.72** | 74.76 | 33.56 |
| SimCSE + WR| **58.97** | 51.58 | **78.32** | **33.73** |

SimCSE + WR 策略在中文数据集训练的超参数设置如下：

| 数据集|epoch | learning rate | dropout|batch size| dup rate|
|-------|-------|-----|------|-----|-----|
|LCQMC|1| 5E-5 | 0.3 |64| 0.32 |
|BQ_Corpus|1| 1E-5 | 0.3 |64|0.32 |
|STS-B|8| 5E-5 | 0.1 |64| 0.32 |
|ATEC|1| 5E-5 | 0.3 | 64| 0.32 |



## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：

```
simcse/
├── model.py # SimCSE 模型组网代码
├── data.py # 无监督语义匹配训练数据、测试数据的读取逻辑
├── predict.py # 基于训练好的无监督语义匹配模型计算文本 Pair 相似度
├── train.sh # 模型训练的脚本
└── train.py # SimCSE 模型训练、评估逻辑
```

### 模型训练
我们以中文文本匹配公开数据集 LCQMC 为示例数据集， 仅使用 LCQMC 的文本数据构造生成了无监督的训练数据。可以运行如下命令，开始模型训练并且在 LCQMC 的验证集上进行 Spearman 相关系数评估。

```shell
$ unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus '0' \
    train.py \
    --device gpu \
    --save_dir ./checkpoints/ \
    --batch_size 64 \
    --learning_rate 5E-5 \
    --epochs 1 \
    --save_steps 100 \
    --eval_steps 100 \
    --max_seq_length 64 \
    --dropout 0.3 \
    --train_set_file "./senteval_cn/LCQMC/train.txt" \
    --test_set_file "./senteval_cn/LCQMC/dev.tsv"
```

可支持配置的参数：

* `infer_with_fc_pooler`：可选，在预测阶段计算文本 embedding 表示的时候网络前向是否会过训练阶段最后一层的 fc;  建议关闭模型效果最好。
* `dup_rate`: 可选，word reptition 的比例，默认是0.32，根据论文 Word Repetition 比例采用 0.32 效果最佳。
* `scale`：可选，在计算 cross_entropy loss 之前对 cosine 相似度进行缩放的因子；默认为 20。
* `dropout`：可选，SimCSE 网络前向使用的 dropout 取值；默认 0.1。
* `save_dir`：可选，保存训练模型的目录；默认保存在当前目录checkpoints文件夹下。
* `max_seq_length`：可选，ERNIE-Gram 模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.0。
* `epochs`: 训练轮次，默认为1。
* `warmup_proption`：可选，学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
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

### 基于动态图模型预测

我们用 LCQMC 的测试集作为预测数据,  测试数据示例如下，：
```text
谁有狂三这张高清的  这张高清图，谁有
英雄联盟什么英雄最好    英雄联盟最好英雄是什么
这是什么意思，被蹭网吗  我也是醉了，这是什么意思
现在有什么动画片好看呢？    现在有什么好看的动画片吗？
请问晶达电子厂现在的工资待遇怎么样要求有哪些    三星电子厂工资待遇怎么样啊
```

执行如下命令开始预测：
```shell
python -u -m paddle.distributed.launch --gpus "0" \
        predict.py \
        --device gpu \
        --params_path "./checkpoints/model_4400/model_state.pdparams"\
        --batch_size 64 \
        --max_seq_length 64 \
        --text_pair_file 'test.tsv'
```

输出预测结果如下:
```text
0.7201147675514221
0.9010907411575317
0.5393891334533691
0.9698929786682129
0.6056119203567505
```

## Reference
[1] Gao, Tianyu, Xingcheng Yao, and Danqi Chen. “SimCSE: Simple Contrastive Learning of Sentence Embeddings.” ArXiv:2104.08821 [Cs], April 18, 2021. http://arxiv.org/abs/2104.08821.

[2] Wu, Xing, et al. "ESimCSE: Enhanced Sample Building Method for Contrastive Learning of Unsupervised Sentence Embedding." arXiv preprint arXiv:2109.04380 (2021). https://arxiv.org/abs/2109.04380.
