# GPT-3 千亿参数模型训练

## 模型介绍
GPT-[3](https://arxiv.org/pdf/2005.14165.pdf) 是以[Transformer](https://arxiv.org/abs/1706.03762) 为基础的语言生成模型。GPT-3模型的最大参数量可以达到170B，如此大规模参数的模型对于训练使用的深度学习框架是一个巨大的挑战。

本示例主要提供了GPT-3的训练过程，数据准备、预测部署等内容请参见[GPT](../../../model_zoo/gpt) 目录。
本示例包含了GPT-3的[静态图](./static)和动态图的多级并行训练流程。
用户可以根据自己的需求，训练GPT-3模型，或者参考本示例，使用模型并行、流水线并行等策略，开发训练其他大模型。


## 使用方法

### 环境依赖

- regex
- sentencepiece >= 0.1.94
- tqdm
- visualdl
- paddlepaddle-gpu >= 2.2rc

安装命令 `pip install regex sentencepiece tqdm visualdl`。
注：需要PaddlePaddle版本大于等于2.2rc，或者使用最新develop版本，安装方法请参见Paddle[官网](https://www.paddlepaddle.org.cn)。


### 数据获取与制作

[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/)是一个开源的英文网页文本数据集，数据来源于Reddit，经过去重、清洗、提取，最终包含800多万个文档。
本示例采用EleutherAI清洗好的[OpenWebText2数据](https://openwebtext2.readthedocs.io/en/latest/index.html#download-plug-and-play-version)

下载以后通过以下命令解压：

```shell
wget https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar
tar -xvf openwebtext2.json.zst.tar -C  /path/to/openwebtext
```

然后使用[data_tools](./data_tools)工具下的`create_pretraining_data.py`脚本进行数据集制作：
```
python -u  create_pretraining_data.py \
    --model_name gpt2-en \
    --tokenizer_name GPTTokenizer \
    --data_format JSON \
    --input_path /path/to/openwebtext/ \
    --append_eos \
    --output_prefix gpt_openwebtext  \
    --workers 40 \
    --log_interval 10000
```
处理时间约一个小时左右，就可以得到我们需要的`gpt_openwebtext_ids.npy`, `gpt_openwebtext_idx.npz`数据集文件。

为了方便用户运行测试本模型，本项目提供了处理好的300M的训练样本：
```shell
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
```

将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：

```
mkdir data
mv gpt_en_dataset_300m_ids.npy ./data
mv gpt_en_dataset_300m_idx.npz ./data
```


```shell
cd static # 或者 cd dygraph
# 下载样例数据
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
cd ..
# 运行pretrian 脚本
sh run.sh
```
下面以静态图的运行脚本为例，说明训练参数的具体作用：
```shell
python -u  -m paddle.distributed.fleet.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name/log" run_pretrain_static.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-en" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_len 1024 \
    --micro_batch_size 8 \
    --global_batch_size 32 \
    --sharding_degree 4\
    --mp_degree 2 \
    --dp_degree 1 \
    --pp_degree 1 \
    --use_sharding true \
    --use_amp true \
    --use_recompute true \
    --max_lr 0.00015 \
    --min_lr 0.00001 \
    --max_steps 500000 \
    --save_steps 100000 \
    --decay_steps 320000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 1\
    --eval_freq 10000 \
    --device "gpu"
```
其中参数释义如下：
- `model_name_or_path` 要训练的模型或者之前训练的checkpoint。
- `input_dir` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件。
- `output_dir` 指定输出文件。
- `max_seq_len` 输入文本序列的长度。
- `micro_batch_size` 单卡单次的 batch size大小。即单张卡运行一次前向网络的 batch size大小。
- `global_batch_size` 全局的batch size大小，即一次参数更新等效的batch size。
- `mp_degree` 模型并行划分的数（如 mp_degree=2 表示将计算的Tensor划分到两个设备）。
- `sharding_degree` 切参数切分的分组大小（如 sharding_degree=4 表示参数分为4组，分别到4个设备）。
- `pp_degree` 流水线并行参数，表示将网络划分成多少段。
- `dp_degree` 数据并行参数。
- `use_sharding` 开启sharding策略
- `use_amp` 开启混合精度策略。
- `use_recompute` 开启重计算策略。
- `max_lr` 训练学习率。
- `min_lr` 学习率衰减的最小值。
- `max_steps` 最大训练步数。
- `save_steps` 保存模型间隔。
- `weight_decay` 权重衰减参数。
- `warmup_rate` 学习率warmup参数。
- `grad_clip` 梯度裁剪范围。
- `logging_freq` 日志输出间隔。
- `eval_freq` 模型评估间隔。
- `device` 训练设备。

注：
- 一般而言，需要设置 `mp_degree * sharding_degree * pp_degree * dp_degree` = 训练机器的总卡数。
- 一般而言， `global_batch_size = micro_batch_size * sharding_degree * dp_degree`。用户也可以使用梯度累积的方式增大`global_batch_size`。

### 飞桨 4D 并行简介

飞桨的4D混合并行包括一下4个维度：

- 模型并行(Model Parallelism，通过将乘法张量切片)
- 参数分组切片的数据并行(Sharding)
- 流水线并行(Pipeline Parallelism)
- 纯数据并行(Data Parallelism)

除了上述混合并行策略外，飞桨还支持重计算、offload、混合精度等策略，来减少显存占用、加速训练。更多具体内容可以参考稿件:[飞桨分布式训练又推新品，4D混合并行可训千亿级AI模型](https://baijiahao.baidu.com/s?id=1697085717806202673)。

### 飞桨超大模型部署

飞桨超大模型部署工具：

- Paddle Fleet: 飞桨训练自适应并行技术，同样适应于超大模型部署，针对推理硬件自适应切分
- Paddle Inference: 支持模型并行、流水线并行、混合并行策略，经过极致优化，性能领先
- Paddle Serving: 支持服务化部署，支持自动Batch、容错调度、服务监控、负载均衡
- Paddle Slim: 支持超大模型量化、稀疏压缩

具体部署示例参考[GPT-3超大模型部署教程](deploy)

### 参考文献
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
