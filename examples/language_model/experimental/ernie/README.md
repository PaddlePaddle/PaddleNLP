## ERNIE

注：PaddleNLP提供了最新版本的Ernie预训练代码，采用了全新的数据流设置。请使用[ernie-1.0](../../ernie-1.0)目录训练模型。不建议本示例进行模型训练任务。

ERNIE是百度开创性提出的基于知识增强的持续学习语义理解框架，它将大数据预训练与多源丰富知识相结合，通过持续学习技术，不断吸收海量文本数据中词汇、结构、语义等方面的知识，实现模型效果不断进化。

ERNIE在情感分析、文本匹配、自然语言推理、词法分析、阅读理解、智能问答等16个公开数据集上全面显著超越世界领先技术，在国际权威的通用语言理解评估基准GLUE上，得分首次突破90分，获得全球第一。
相关创新成果也被国际顶级学术会议AAAI、IJCAI收录。
同时，ERNIE在工业界得到了大规模应用，如搜索引擎、新闻推荐、广告系统、语音交互、智能客服等。

本示例简要开源了ERNIE的预训练代码。

### 环境依赖
- visualdl
安装命令 `pip install visualdl`

### 使用方法

用户需要下载样例数据和词表文件，即可运行预训练脚本，训练ERNIE Base模型。
```shell
# 下载样例数据集，并解压
wget https://bj.bcebos.com/paddlenlp/data/ernie_hybrid_parallelism_data.tar
tar -xvf ernie_hybrid_parallelism_data.tar

# 下载Vocab文件
wget https://bj.bcebos.com/paddlenlp/data/ernie_hybrid_parallelism-30k-clean.vocab.txt -O ./config/vocab.txt

# 运行Pretrain 脚本
sh pretrain.sh

# 或者直接运行python脚本进行预训练
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.fleet.launch \
    --gpus 0,1,2,3,4,5,6,7 \
    --log_dir ./output_dir/log \
    run_pretraining.py \
    --global_bsz 64 \
    --micro_bsz 1 \
    --max_seq_len 512 \
    --ernie_config_file config/ernie_base_config.json \
    --learning_rate 1e-4 \
    --log_steps 1 \
    --num_train_steps 1000000 \
    --save_steps 100000 \
    --output_dir ./output_dir \
    --use_recompute true \
    --use_sharding true \
    --use_sop false \
    --num_mp=2 \
    --num_sharding=2 \
    --num_pp=2 \
    --num_dp=1 \
```
下面对`run_pretrain.py`中的预训练参数进行简要说明：

- `global_bsz` 全局的batch size大小，即一次参数更新等效的batch size。
- `micro_bsz` 单卡单次的 batch size大小。即单张卡运行一次前向网络的 batch size大小。
- `max_seq_len` 输入数据的最大序列长度。
- `ernie_config_file` ernie的模型参数配置文件。
- `learning_rate` 学习率。
- `num_train_steps` 最大训练step数。
- `save_steps` 模型参数保存间隔。
- `output_dir` 模型参数保存的文件夹。
- `use_recompute` 使用重计算（可减少显存）。
- `use_sharding` 使用sharding策略（参数分组切片）。
- `use_sop` 为 ernie 模型配置sop策略。
- `num_mp` 模型并行划分的数（如 num_mp=2 表示将计算的Tensor划分到两个设备）。
- `num_sharding` 切参数切分的分组大小（如 num_sharding=4 表示参数分为4组，分别到4个设备）。
- `num_pp` 流水线并行参数，表示将网络划分成多少段。
- `num_dp` 数据并行参数。

注：一般而言，需要设置 `num_mp * num_sharding * num_pp * num_dp` = 训练机器的总卡数。

### Fleet 4D 混合并行助力ERNIE预训练

为了探索更优的语言模型，本示例采用了飞桨Fleet 4D混合并行策略，可以将模型扩展到128层transformer网络，千亿参数规模。
100多层transformer网络结构的模型，计算复杂，训练需占用T级显存资源。
飞桨的分布式混合并行能力，突破了模型参数的显存限制瓶颈，使得训练千亿参数模型成为可能。
同时飞桨还提供了一系列性能优化和显存优化措施，进一步减少用户训练的机器资源、加速训练过程。

#### 飞桨 4D 并行简介

飞桨的4D混合并行包括一下4个维度：

- 模型并行(Model Parallelism，通过将乘法张量切片)
- 参数分组切片的数据并行(Sharding)
- 流水线并行(Pipeline Parallelism)
- 纯数据并行(Data Parallelism)

除了上述混合并行策略外，飞桨还支持重计算、offload、混合精度等策略。下面，我们分别从性能优化和显存优化的角度，对飞桨的混合并行能力进行简要分析：

首先看如何性能优化。我们通过一个公式来看哪些因素可以影响训练速度，在固定的硬件环境下：

```
总训练速度 ∝ 单卡速度∗卡数∗多卡加速比
```

其中单卡速度由数据读取和计算速度决定；多卡加速比由计算/通信效率决定。显而易见，这三个是关键因素。
除了单卡可以使用的算子融合、混合精度之类的基础性能优化策略之外，分布式训练还引入一系列并行策略。
并行策略的核心思想是将数据和计算有关的图/算子切分到不同设备上，同时尽可能降低设备间通信所需的代价，合理使用多台设备资源，实现高效的并发调度训练，最大化提升训练速度。
常见并行策略有数据并行DP（Data Parallelism）、Layer间并行（流水线并行PP，Pipeline Parallelism）、Layer内并行（模型并行MP，Model Parallelism）。
我们从设备资源和计算/通信效率来分析三种策略的优缺点：

- 数据并行训练加速比最高，但要求每个设备上都备份一份模型，显存占用比较高。为此我们的改进方案是使用分组参数切片数据并行策略（Sharding），兼容了MP+DP的优势，但缺点是通信量大。

- 模型并行，通信量比较高，适合在机器内做模型并行。

- 流水线并行，训练设备容易出现空闲状态，加速效率没有DP高；但能减少通信边界支持更多的层数，适合在机器间使用。

<p align="center">
  <img src="https://p8.pstatp.com/origin/pgc-image/599bd8c2d5a14341a85c8c8d1150f5de.jpeg" width="500" height ="400"/>
  <br>飞桨 4D 混合并行策略示意图
</p>

其次看显存问题，上述的并行策略可以很好的应对不同来源的显存占用，
更多的层数可以通过流水线并行和分组参数切分策略来解决；
某层参数很大可以通过模型并行来解决；
其次飞桨还提供一些其它灵活的优化方式，例如每层输出占用的显存，可以通过重计算和offload来解决。

综上所述，针对性能优化和显存优化，几种并行策略都有用武之地，但是同时也有各自的局限性，所以如果想高效训练千亿模型，需要这几种策略相互组合，取长补短，发挥各自的优势。
用户可以根据自己的需求，尝试自己最优的混合并行策略。

以128层transformer，230B参数配置的模型为例，推荐机器配置为 32 * 8 V100 32G，即32台* 8 卡 32G V100机器。
整体运行脚本配置如下：
```shell
python -m paddle.distributed.fleet.launch \
    --log_dir output_dir/log \
    run_pretraining.py \
    --global_bsz 4096 \
    --micro_bsz 8 \
    --max_seq_len 512 \
    --ernie_config_file config/ernie_230b_config_proj.json \
    --learning_rate 1e-4 \
    --log_steps 1 \
    --num_train_steps 1000000 \
    --save_steps 100000 \
    --output_dir output_dir \
    --use_recompute true \
    --use_sharding true \
    --use_sop false \
    --num_mp=2 \
    --num_pp=32 \
    --num_sharding 4 \
```
建议配置Model Parallel并行数为2，流水线层数为32，sharding数据分片数为4。

### 参考文献
- [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223.pdf)
- [ERNIE 2.0: A Continual Pre-training Framework for Language Understanding](https://arxiv.org/pdf/1907.12412.pdf)
- [飞桨分布式训练又推新品，4D混合并行可训千亿级AI模型](https://baijiahao.baidu.com/s?id=1697085717806202673)
