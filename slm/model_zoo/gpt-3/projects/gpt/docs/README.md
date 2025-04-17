# GPT

## 模型介绍
GPT-[2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)/[3](https://arxiv.org/pdf/2005.14165.pdf) 是以[Transformer](https://arxiv.org/abs/1706.03762) 解码器为网络基本组件，使用自回归的方式在大规模无标注文本语料上进行预训练得到的语言生成模型。

本项目是语言模型 GPT 的 PaddlePaddle 大模型实现。目前，PaddleFleetX 提供了 [GPT-345M](https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz) 的预训练模型文件；分别基于 [LAMBADA](https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl) 和 [WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) 数据集，采用 ACC(accuracy) 和 PPL(perplexity) 指标后的评估结果如下：

| **模型文件** | **ACC** | **PPL** |
|--------------|---------|---------|
| GPT-345M     | 44.17%  | 18.01   |

下面是本例的简要目录结构及说明：

```text
.
├── auto_export_gpt_345M_mp2.sh            # 自动并行345M模型两卡张量并行导出入口
├── auto_gpt_345M_single_card.sh           # 自动并行345M模型单卡预训练入口
├── auto_gpt_1.3B_single_card.sh           # 自动并行1.3B模型单卡预训练入口
├── auto_gpt_1.3B_dp8.sh                   # 自动并行1.3B模型数据并行预训练入口
├── auto_gpt_6.7B_sharding16.sh            # 自动并行6.7B模型分组切片并行预训练入口
├── evaluate_gpt_345M_single_card.sh       # 单卡345M模型评估入口
├── export_gpt_345M_single_card.sh         # 单卡345M模型动转静导出入口
├── finetune_gpt_345M_single_card.sh       # 单卡345M模型finetune训练入口
├── inference_gpt_345M_single_card.sh      # 单卡345M模型推理入口
├── pretrain_gpt_345M_single_card.sh       # 单卡345M模型预训练入口
├── pretrain_gpt_1.3B_single_card.sh       # 单卡1.3B模型预训练入口
├── pretrain_gpt_1.3B_dp8.sh               # 8卡1.3B模型数据并行预训练入口
├── pretrain_gpt_6.7B_sharding16.sh        # 16卡6.7B模型分组切片并行预训练入口
├── pretrain_gpt_175B_mp8_pp16.sh          # 128卡175B模型混合并行预训练入口
├── qat_gpt_345M_single_card.sh            # 单卡345M模型量化训练入口
├── qat_gpt_345M_mp8.sh                    # 8卡345M模型模型并行量化训练入口
├── qat_gpt_6.7B_sharding16.sh             # 16卡6.7B模型分组切片并行量化训练入口
├── eval_qat_gpt_345M_single_card.sh       # 单卡345M量化模型验证入口
├── export_qat_gpt_345M_single_card.sh     # 单卡345M量化模型导出入口
```

## 快速开始

### 环境依赖

请确保已根据根目录 requirements.txt 安装所需依赖，或者通过以下命令快速安装

```shell
python -m pip install -r https://raw.githubusercontent.com/PaddlePaddle/PaddleFleetX/develop/requirements.txt -i https://mirror.baidu.com/pypi/simple
```

### 数据准备

数据获取和制作详见[GPT 模型预训练数据准备流程](https://github.com/PaddlePaddle/PaddleFleetX/tree/develop/ppfleetx/data/data_tools/gpt)

为了方便用户运行测试本模型，此处提供处理好的300M的训练样本，在单卡训练或混合并行训练前都需要通过以下命令获取数据。

**数据下载命令**
```shell
cd PaddleNLP/model_zoo/gpt-3 # 如果已在 PaddleNLP/model_zoo/gpt-3 目录下，则忽略

# 下载样例数据
mkdir data && cd data
wget -O gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget -O gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz

cd .. # 回到 PaddleFleetX 根目录下
```

### 模型训练

除了单卡训练，飞桨还支持数据并行、混合并行、自动并行、重计算等多种分布式策略，减少显存占用、加速训练，达到大模型可训练且训得快的效果。在模型训练前，需要根据模型规模选择合适的并行策略。下面分别从单卡训练、混合并行训练和自动并行训练三个方面来介绍GPT模型训练的配置文件和启动方式。


- [单卡训练](./single_card.md)

- [混合并行训练](./hybrid_parallel.md)

- [自动并行训练](./auto_parallel.md)

### 文本生成体验

- [单卡预训练模型文本生成](./single_card.md#GPT-Zero-shot-文本生成)

- [混合并行预训练模型文本生成](./hybrid_parallel.md#GPT-Zero-shot-文本生成)


### 模型压缩

- [量化训练](./quantization_aware_training.md)

### 推理部署

- [推理部署](inference.md)
### GLUE 下游任务微调

- [单卡微调](./single_finetune.md)


## 参数释义


### 全局信息
全局参数指定训练的batch size，以及设备、随机种子等信息。
```yaml
  Global:
    device: gpu
    seed: 1024

    global_batch_size:
    local_batch_size: 1
    micro_batch_size: 1
```

其中参数对应的释义如下：
| **参数名**        | **参数释义**                                         |
|-------------------|------------------------------------------------------|
| device            | 设备信息                                             |
| seed              | 随机数种子                                           |
| global_batch_size | 全局的batch size大小，即一次参数更新等效的batch size |
| local_batch_size  | 每个进程训练的batch size大小                         |
| micro_batch_size  | 每次前向计算的batch size大小                         |


### Engine训练控制

Engine训练设置完成模型训练/验证/推理等过程中的参数设置，是fleetX的EagerEngine的必要参数，所有使用该Engine都必须指定该配置。 其中包含的参数有：

```yaml
  Engine:
    max_steps: 500000
    num_train_epochs: 1
    accumulate_steps:
    logging_freq: 1
    eval_freq: 500
    eval_iters: 10
    test_iters:
    mix_precision:
      enable: True
      dtype: "float16"
      level: "O2"
      scale_loss: 32768.0
      custom_black_list: ["reduce_sum", "c_softmax_with_cross_entropy", "elementwise_div"]
      custom_white_list: ["lookup_table", "lookup_table_v2"]
    save_load:
      save_steps: 1000
      save_epoch: 1
      output_dir: ./output
      ckpt_dir:
```
其中参数对应的释义如下：

| **参数名**        | **参数释义**                                                                                                                                               |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| max_steps         | 最大训练步数                                                                                                                                               |
| num_train_epochs  | 训练的epoch数量                                                                                                                                            |
| accumulate_steps  | 梯度累加次数                                                                                                                                               |
| logging_freq      | 训练日志打印的频率                                                                                                                                         |
| eval_freq         | 模型评估间隔                                                                                                                                               |
| eval_iters        | 模型评估时训练评估测试集的轮数                                                                                                                             |
| test_iters        | 模型测试或推理时的轮数                                                                                                                                     |
| enable            | 是否使用混合精度策略进行训练                                                                                                                               |
| dtype             | 混合精度训练数据类型使用float16还是bfloat16，默认为float16类型                                                                                             |
| level             | 混合精度训练模式，默认``O2``模式                                                                                                                           |
| scale_loss        | 使用fp16混合精度策略下，loss的放缩比例                                                                                                                     |
| custom_black_list | 自定义算子黑名单。这个名单中的算子在支持混合精度计算时会被认为是数值危险的，它们的影响也可能会在下游操作中观察到。这些算子通常不会转为float16/bfloat16计算 |
| custom_white_list | 自定义算子白名单。这个名单中的算子在支持混合精度计算时会被认为是数值安全的，并且对性能至关重要。如果设置了白名单，该名单中的算子会使用float16/bfloat16计算 |
| save_steps        | 保存模型间隔step数                                                                                                                                         |
| save_epoch        | 保存模型间隔epoch数                                                                                                                                        |
| output_dir        | 指定输出文件                                                                                                                                               |
| ckpt_dir          | checkpoint的加载目录                                                                                                                                       |

### 模型网络

网络部分完成了网络的组网操作，GPT在[PaddleFleetX/ppfleetx/models/language_model/gpt/dygraph/single_model.py](https://github.com/PaddlePaddle/PaddleFleetX/blob/develop/ppfleetx/models/language_model/gpt/dygraph/single_model.py)下。
可以使用配置文件配置模型的规模，如：

```yaml
  Model:
    module: "GPTModule"
    name: "GPT"
    vocab_size: 50304
    hidden_size: 1024
    num_layers: 24
    num_attention_heads: 16
    ffn_hidden_size:
    hidden_dropout_prob: 0.1
    attention_probs_dropout_prob: 0.1
    max_position_embeddings: 1024
    type_vocab_size: 16
    initializer_range: 0.02
    use_recompute: True
    recompute_granularity:
    no_recompute_layers:
    fused_linear: True
    fuse_attn_qkv: True
    sequence_parallel: False
```

其中参数对应的释义如下：
| **参数名**                   | **参数释义**                                                                                                                                                                                                                                                                                                                         |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| module                       | 指定GPT模型的执行模块 ｜                                                                                                                                                                                                                                                                                                             |
| vocab_size                   | 训练词表大小                                                                                                                                                                                                                                                                                                                         |
| hidden_size                  | 隐藏层大小                                                                                                                                                                                                                                                                                                                           |
| num_layers                   | transformer层数                                                                                                                                                                                                                                                                                                                      |
| num_attention_heads          | attention head的数量                                                                                                                                                                                                                                                                                                                 |
| max_seq_len                  | 输入文本序列的长度                                                                                                                                                                                                                                                                                                                   |
| ffn_hidden_size              | ffn层大小，一般为隐藏层的四倍                                                                                                                                                                                                                                                                                                        |
| attention_probs_dropout_prob | attention中的dropout的失活率                                                                                                                                                                                                                                                                                                         |
| max_position_embeddings      | position embedding的长度                                                                                                                                                                                                                                                                                                             |
| type_vocab_size              | 词表类型                                                                                                                                                                                                                                                                                                                             |
| initializer_range            | 参数初始化的范围                                                                                                                                                                                                                                                                                                                     |
| use_recompute                | 是否使用recompute训练                                                                                                                                                                                                                                                                                                                |
| recompute_granularity        | recompute训练的粒度，可选 `full` `full_attn` `core_attn`，full即recompute全部transformer，full_attn表明只recompute所有self attention部分，core_attn表明只recompute `softmax(qkT)v` 部分。注：显存占用方面，`core_attn` > `full_attn` > `full`，若所选策略产生OOM错误，可以适当更改recompute_granularity                              |
| no_recompute_layers          | list of integer，标识哪些层的transformer不需要进行recompute。所有在该list中的值应该 >= 0 同时应该 < num_layers。向该参数中增加不进行recompute 的层数可以提升模型训练的整体吞吐，但是会适当的增加显存。若训练中发现有显存富裕，可以适当增加不进行recompute的层数。如果使用该参数后出现OOM错误，可以适当减小不进行recompute的层数。 ｜ |
| fused_linear                 | 是否使用fused_linear代替传统Linear加速训练。注：该功能需要cuda 11.6及以上编译的paddle支持。                                                                                                                                                                                                                                          |
| fuse_attn_qkv                | 是否对attention层中的qkv计算使用fuse策略以加速训练                                                                                                                                                                                                                                                                                   |
| sequence_parallel            | 是否使用序列并行策略以加速训练。注：只有混合并行的GPT才支持该功能，它与张量模型并行共用通信组，当mp_degree=1时，序列并行策略会被强制关闭。                                                                                                                                                                                           |
| virtual_pp_degree            | 虚拟流水线并行维度，该参数会减小流水线bubble的占比以提升流水线的吞吐。但是该参数会增加流水线间的通讯，所以该参数的推荐值为2。并且，只有 num_layers可以被 pp_degree * virtual_pp_degree 整除时，才可以使用虚拟流水线并行。                                                                                                            |
### 数据集

数据集参数分为“Train”、“Eval”和“Test”三部分，分别对应模型预训练、离线评估、推理等三个模块。

每个模型的配置参数都包含以下内容：

```yaml
  Data:
    Train:
      dataset:
        name: GPTDataset
        input_dir: ./data/
        split: [949, 50, 1]
        max_seq_len: 1024
      sampler:
        name: DistributedBatchSampler
        shuffle: False
        drop_last: True
      loader:
        num_workers: 1
        return_list: False
        collate_fn: gpt_collate_fn
```

其中参数对应的释义如下：
| **参数名**   | **参数释义**                                                 |
|--------------|--------------------------------------------------------------|
| dataset.name | 指定自定义数据集的名称                                       |
| input_dir    | 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件 |
| split        | 训练集，验证集和测试集的切分比例                             |
| max_seq_len  | 输入文本序列的长度                                           |
| sampler.name | 指定自定义采样器的名称                                       |
| shuffle      | 是否需要在生成样本下标时打乱顺序                             |
| drop_last    | 是否需要丢弃最后无法凑整一个mini-batch的样本                 |
| num_workers  | 用于加载数据的子进程个数                                     |
| return_list  | 每个设备上的数据是否以list形式返回                           |
| collate_fn   | 通过此参数指定如果将样本列表组合为mini-batch数据；支持自定义 |


### 优化器


GPT训练默认使用AdamW优化器以及cosine学习率衰减，这里通过配置文件配置优化器的参数，如：

```yaml
  Optimizer:
    name: AdamW
    weight_decay: 0.01
    beta1: 0.9
    beta2: 0.999
    epsilon: 1.0e-8
    lr:
      name: CosineAnnealingWithWarmupDecay
      decay_steps: 360000
      warmup_rate: 0.01
      max_lr: 5.0e-5
      min_lr: 1.0e-5
    grad_clip:
      name: "ClipGradByGlobalNorm"
      clip_norm: 1.0
    tensor_fusion: False
```

其中参数说明：

| **参数名**     | **参数释义**                       |
|----------------|------------------------------------|
| name           | 指定自定义优化器的名称             |
| weight_decay   | weight的衰减率                     |
| beta1          | 一阶矩估计的指数衰减率             |
| beta2          | 二阶矩估计的指数衰减率             |
| epsilon        | 指定优化器需要优化的参数           |
| lr.name        | 指定自定义学习率策略的名称         |
| decay_steps    | 衰减的步长                         |
| warmup_rate    | warmup 率                          |
| max_lr         | Adam 的初始最大学习率              |
| min_lr         | Adam 的初始最小学习率              |
| grad_clip.name | 指定自定义梯度裁剪策略的名称       |
| clip_norm      | 所允许的范数最大值                 |
| tensor_fusion  | 是否使用tensor_fustion功能加速训练 |

另外，[Profiler](./hybrid_profiler.md)中还介绍了在 GPT 中开启 Profiler 并分析调试分析结果的方法及相关的参数解释。

### 模型压缩
PaddleFleetX 集成了 PaddleSlim 中的常见的压缩方法：量化训练（Qutization Aware Training，QAT）、结构化稀疏（Structured Pruning，SP）和知识蒸馏（Knowledge Distillation，KD）。详细参数介绍见[模型压缩介绍](../../../docs/compression.md)。


## 参考文献
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
- [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413)
