# ELMo

### 介绍

ELMo(Embeddings from Language Models) 是重要的通用语义表示模型之一，以双向 LSTM  为网路基本组件，以 Language Model 为训练目标，通过预训练得到通用的语义表示，将通用的语义表示作为 Feature 迁移到下游 NLP 任务中，会显著提升下游任务的模型性能。本项目是 ELMo 在 Paddle Fluid 上的开源实现, 基于百科类数据训练并发布了预训练模型。

同时推荐用户参考[ IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/124374)

### 发布要点:
1） 基于百科类数据训练的 [ELMo 中文预训练模型](https://dureader.gz.bcebos.com/elmo/baike_elmo_checkpoint.tar.gz);

2） 完整支持 ELMo 模型训练及表示迁移, 包括:

- 支持 ELMo 多卡训练，训练速度比主流实现快约1倍
- 以 LAC 任务为示例提供 ELMo 语义表示迁移到下游 NLP 任务的示例

3）我们在阅读理解任务和 LAC 任务上评估了 ELMo 预训练模型带给下游任务的性能提升:
- LAC 加入 ELMo 后 F1 可以提升 **1.1%**
- 阅读理解任务加入 ELMo 后 Rouge-L 提升 **1%**

| Task | 评估指标 | Baseline | +ELMo |
| :------| :------: | :------: |:------: |
| [LAC](https://github.com/baidu/lac) | F1 | 87.3% | **88.4%** |
| [阅读理解](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/unarchived/machine_reading_comprehension) | Rouge-L | 39.4% | **40.4%** |

**Note**:

- LAC 任务是基于 20w 训练数据训练的词模型
- 阅读理解任务是基于 [DuReader](https://github.com/baidu/DuReader) 数据集训练的词模型

### 安装

本项目依赖于 Paddle Fluid **1.4.0**，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装。
其他安装依赖：python2.7 或者 python3.7

提示：
(1)使用Windows GPU环境的用户，需要将示例代码中的[fluid.ParallelExecutor](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/fluid_cn.html#parallelexecutor)替换为[fluid.Executor](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/fluid_cn.html#executor)。
(2)ELMo代码库暂不支持Windows CPU环境。  

### 预训练

#### 数据预处理

将文档按照句号、问号、感叹切分成句子，然后对句子进行切词。预处理后的数据文件中每行为一个分词后的句子。我们给出了示例训练数据 [`data/train`](data/train) 和测试数据 [`data/dev`](data/dev)，数据示例如下：

```
本 书 介绍 了 中国 经济 发展 的 内外 平衡 问题 、 亚洲 金融 危机 十 周年 回顾 与 反思 、 实践 中 的 城乡 统筹 发展 、 未来 十 年 中国 需要 研究 的 重大 课题 、 科学 发展 与 新型 工业 化 等 方面 。
吴 敬 琏 曾经 提出 中国 股市 “ 赌场 论 ” ， 主张 维护 市场 规则 ， 保护 草根 阶层 生计 ， 被 誉 为 “ 中国 经济 学界 良心 ” ， 是 媒体 和 公众 眼中 的 学术 明星
```

#### 模型训练

利用提供的示例训练数据和测试数据，我们来说明如何进行单机多卡预训练。关于预训练的启动方式，可以查看脚本 `run.sh` ，该脚本已经默认以示例数据作为输入。在开始预训练之前，需要把 CUDA、cuDNN、NCCL2 等动态库路径加入到环境变量 `LD_LIBRARY_PATH` 之中，然后按如下方式即可开始单机多卡预训练

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sh run.sh
```

训练过程中，默认每间隔 10000 steps 将模型参数写入到 checkpoints 路径下，可以通过 `--save_interval ${N}` 自定义保存模型的间隔 steps。

### ELMo 预训练模型如何迁移到下游 NLP 任务

我们在 [bilm.py](./LAC_demo/bilm.py) 中提供了 `elmo_encoder` 接口获取 ELMo 预训练模型的语义表示, 便于用户将 ELMo 语义表示快速迁移到下游任务;以 [LAC](https://github.com/baidu/lac) 任务为示例, 将 ELMo 预训练模型的语义表示迁移到 LAC 任务的主要步骤如下：

1） 搭建 LAC 网络结构，并加载 ELMo 预训练模型参数; 我们在 [bilm.py](./LAC_demo/bilm.py) 中提供了加载预训练模型的接口函数 init_pretraining_params

```
#step1: create_lac_model()

#step2: load pretrained ELMo model
from bilm import init_pretraining_params
init_pretraining_params(exe, args.pretrain_elmo_model_path,
                        fluid.default_main_program())
```

2） 基于 [ELMo 字典](data/vocabulary_min5k.txt) 将输入数据转化为 word_ids

3）利用 elmo_encoder 接口获取 ELMo embedding

```
from bilm import elmo_encoder
elmo_embedding = elmo_encoder(word_ids)
```

4） ELMo embedding 与 LAC 原有 word_embedding 拼接得到最终的 embedding
```
word_embedding=fluid.layers.concat(input=[elmo_embedding, word_embedding], axis=1)
```

### 参考论文
[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
