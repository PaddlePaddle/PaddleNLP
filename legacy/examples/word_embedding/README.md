# Word Embedding with PaddleNLP

## 简介

PaddleNLP已预置多个公开的预训练Embedding，用户可以通过使用`paddlenlp.embeddings.TokenEmbedding`接口加载预训练Embedding，从而提升训练效果。以下通过基于开源情感倾向分类数据集ChnSentiCorp的文本分类训练例子展示`paddlenlp.embeddings.TokenEmbedding`对训练提升的效果。更多的`paddlenlp.embeddings.TokenEmbedding`用法，请参考[TokenEmbedding 接口使用指南](../../docs/model_zoo/embeddings.md) 。


## 快速开始

### 环境依赖

- visualdl

安装命令：`pip install visualdl`


### 启动训练

我们以中文情感分类公开数据集ChnSentiCorp为示例数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在验证集（dev.tsv）验证。训练时会自动下载词表dict.txt，用于对数据集进行切分，构造数据样本。

启动训练：

```shell
# 使用paddlenlp.embeddings.TokenEmbedding
python train.py --device='gpu' \
                --lr=5e-4 \
                --batch_size=64 \
                --epochs=20 \
                --use_token_embedding=True \
                --vdl_dir='./vdl_dir'

# 使用paddle.nn.Embedding
python train.py --device='gpu' \
                --lr=1e-4 \
                --batch_size=64 \
                --epochs=20 \
                --use_token_embedding=False \
                --vdl_dir='./vdl_dir'
```

以上参数表示：

* `device`: 选择训练设备，目前可选'gpu', 'cpu', 'xpu'。 默认为`gpu`。
* `lr`: 学习率， 默认为5e-4。
* `batch_size`: 运行一个batch大小，默认为64。
* `epochs`: 训练轮次，默认为5。
* `use_token_embedding`: 是否使用`paddlenlp.embeddings.TokenEmbedding`，默认为True。
* `vdl_dir`: VisualDL日志目录。训练过程中的VisualDL信息会在该目录下保存。默认为`./vdl_dir`

该脚本还提供以下参数：

* `save_dir`: 模型保存目录。默认值为"./checkpoints/"。
* `init_from_ckpt`: 恢复模型训练的断点路径。默认值为None，表示不恢复训练。
* `embedding_name`: 预训练Embedding名称，默认为`w2v.baidu_encyclopedia.target.word-word.dim300`. 支持的预训练Embedding可参考[Embedding 模型汇总](../../docs/model_zoo/embeddings.md)。

**注意：**

程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存模型在指定的`save_dir`中。训练过程中会实时保存每个epoch的模型参数，并以当前epoch值命名。如第2个Epochs，模型参数会被保存为`./checkpoints/2.pdparams`，优化器状态保存为`./checkpoints/2.pdopt`。

如：
```text
checkpoints/
├── 0.pdopt
├── 0.pdparams
├── 1.pdopt
├── 1.pdparams
├── ...
└── final.pdparams
```

如需恢复模型训练，则init_from_ckpt只需指定到文件名即可，不需要添加文件尾缀。如果用户想热启第10个Epoch保存的模型，则设置 `--init_from_ckpt=./checkpoints/10`即可，程序会自动加载模型参数`./checkpoints/10.pdparams`，也会自动加载优化器状态`./checkpoints/10.pdopt`。


### 启动VisualDL

推荐使用VisualDL查看实验对比。以下为VisualDL的启动命令，其中logdir参数指定的目录需要与启动训练时指定的`vdl_dir`相同。（更多VisualDL的用法，可参考[VisualDL使用指南](https://github.com/PaddlePaddle/VisualDL#2-launch-panel)）

```
visualdl --logdir ./vdl_dir --port 8888 --host 0.0.0.0
```

### 训练效果对比

在Chrome浏览器输入 `ip:8888` (ip为启动VisualDL机器的IP)。

以下为示例实验效果对比图，蓝色是使用`paddlenlp.embeddings.TokenEmbedding`进行的实验，绿色是使用没有加载预训练模型的Embedding进行的实验。
可以看到，使用`paddlenlp.embeddings.TokenEmbedding`的训练，其验证acc变化趋势上升，并收敛于0.90左右，收敛后相对平稳，不容易过拟合。
而没有使用`paddlenlp.embeddings.TokenEmbedding`的训练，其验证acc变化趋势向下，并收敛于0.86左右。从示例实验可以观察到，使用`paddlenlp.embedding.TokenEmbedding`能提升训练效果。

Eval Acc：

![eval acc](https://user-images.githubusercontent.com/16698950/102076935-79ac5480-3e43-11eb-81f8-6e509c394fbf.png)

|                                     |    Best Acc    |
| ------------------------------------| -------------  |
| paddle.nn.Embedding                 |    0.8965      |
| paddelnlp.embeddings.TokenEmbedding |    0.9082      |

## 致谢
- 感谢 [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)提供Word2Vec中文Embedding预训练模型，[GloVe Project](https://nlp.stanford.edu/projects/glove)提供的GloVe英文Embedding预训练模型，[FastText Project](https://fasttext.cc/docs/en/english-vectors.html)提供的fasttext英文预训练模型。

## 参考论文
- Li, Shen, et al. "Analogical reasoning on chinese morphological and semantic relations." arXiv preprint arXiv:1805.06504 (2018).
- Qiu, Yuanyuan, et al. "Revisiting correlations between intrinsic and extrinsic evaluations of word embeddings." Chinese Computational Linguistics and Natural Language Processing Based on Naturally Annotated Big Data. Springer, Cham, 2018. 209-221.
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.
- T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, A. Joulin. Advances in Pre-Training Distributed Word Representations
