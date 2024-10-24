# GPT 混合并行模型训练

当训练超大模型时，就必须借助混合并行策略，混合并行策略分别指数据并行、张量模型并行、流水线并行和分组切片并行。其中数据并行保存完整的模型参数并独立处理一份子数据集，以加速模型训练过程；张量模型并行将网络中的张量（Tensor）切分到不同的设备，从而降低单个设备的显存消耗；流水线并行将模型的不同层放置到不同的计算设备，降低单个计算设备的显存消耗；分组切片并行将参数和模型状态划分到不同卡上，每个GPU只保存部分副本，以减少显存占用。联合四种训练方式，可以实现更大模型、更快训练的效果。具体策略以及相关FleetAPI介绍可以参考以下教程：

- [数据并行](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/data_parallel/index_cn.html)

- [张量模型并行](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/model_parallel_cn.html
)
- [流水线并行](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/pipeline_parallel_cn.html)

- [分组切片并行](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/group_sharded_parallel_cn.html)


## 参数释义

### 并行维度

当前GPT模型已适配3D混合并行，并能够在训练超大模型，用户可以通过配置文件选择并行的维度。

```yaml
  Distributed:
    dp_degree: 2
    mp_degree: 2
    pp_degree: 2
    sharding:
      sharding_degree: 1
      sharding_stage: 1
      sharding_offload: False
      reduce_overlap: False
      broadcast_overlap: False
```

其中参数说明：

| **参数名**          | **参数释义**                             |
|------------------|--------------------------------------|
| dp_degree        | 数据并行维度                               |
| mp_degree        | 张量模型并行维度                             |
| pp_degree        | 流水线并行维度                              |
| sharding_degree  | 分组切分并行维度                             |
| sharding_stage   | 切分策略；1表示仅切分优化器状态，2表示再切分梯度，3表示再切分前向参数 |
| sharding_offload | CPU offload策略                        |
|reduce_overlap| 是否在sharding stage 2的模式下进行reduce通讯与反向计算的overlap，该策略暂时不支持sharding_offload|
|broadcast_overlap| 是否在sharding stage 2的模式下进行broadcast通讯与下一个batch的 前向计算的overlap，该策略暂时不支持sharding_offload。若使用该模型，在evaluation与save之前，必须调用 `paddle.device.cuda.synchronize()` 方法|

## 运行方式
本目录中按照345M、1.3B、6.7B和175B规模大小，给出32G V100环境下GPT模型混合并行训练的策略配置如下：

| 模型规模 | 训练策略                 | yaml文件                   |
|----------|---------------------------|------------------------------|
| 345M     | fp16+mp8+qat              | qat_gpt_345M_mp8.yaml    |
| 1.3B     | fp16+dp8+recompute        | pretrain_gpt_1.3B_dp8.yaml   |
| 6.7B     | fp16+sharding16+recompute | pretrain_gpt_6.7B_sharding16.yaml  |
| 175B     | fp16+mp8+pp16+recompute   | pretrain_gpt_175B_mp8_pp16.yaml   |

若要在显存容量更小的16G V100环境下进行GPT大模型训练，可将对应yaml文件中的`Model`-`hidden size`值改为原来的1/2即可。

### 策略支持

飞桨的混合并行技术包括4个维度：数据并行、张量模型并行、流水线并行和分组切片并行，此外还支持重计算、offload、混合精度、序列并行等策略，来减少显存占用、加速训练。

目前，GPT模型训练已支持前3个维度的任意策略组合，但分组切片并行stage2/3仅支持与数据并行策略组合使用；详见下表。

|                 | data parallel | tensor parallel | pipeline parallel | pure fp16 | recompute |
|-----------------|---------------|-----------------|-------------------|-----------|-----------|
| sharding stage1 | ✓             | ✓               | ✓                 | ✓         | ✓         |
| sharding stage2 | ✓             | ㄨ               | ㄨ                 | ✓         | ✓         |
| sharding stage3 | ✓             | ㄨ               | ㄨ                 | ✓         | ✓         |

### 单机训练

以单机1.3B模型数据并行训练为例，通过``paddle.distributed.launch``启动多进程训练，该gpt程序需要8卡32G V100以运行。

**启动命令**
```shell
cd PaddleNLP/model_zoo/gpt-3 # 如果已在 PaddleNLP/model_zoo/gpt-3 目录下，则忽略

log_dir=log_dp8
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp8.yaml
```

若要在显存容量更小的16G V100环境下进行GPT模型单机训练，可通过减小`Model.hidden_size`调整模型规模至合适大小再启动训练，命令如下：

**启动命令**
```shell
log_dir=log_dp8
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp8.yaml \
    -o Model.hidden_size=1024
```

每张GPU的运行日志`workerlog.x`可在launch命令中指定的`log_dir`路径下找到；若未指定，日志路径为`log/workerlog.x`。运行日志具体内容如下：

**运行日志**

```
[2022-09-21 05:43:58,797] [    INFO] - [train] epoch: 0, batch: 0, loss: 10.992407799, avg_batch_cost: 5.51734 sec, speed: 0.18 step/s, ips_total: 11878 tokens/s, ips: 1485 tokens/s, learning rate: 2.77778e-08
[2022-09-21 05:43:59,508] [    INFO] - [train] epoch: 0, batch: 1, loss: 11.000075340, avg_batch_cost: 0.71029 sec, speed: 1.41 step/s, ips_total: 92267 tokens/s, ips: 11533 tokens/s, learning rate: 4.16667e-08
[2022-09-21 05:44:00,242] [    INFO] - [train] epoch: 0, batch: 2, loss: 11.017463684, avg_batch_cost: 0.73301 sec, speed: 1.36 step/s, ips_total: 89406 tokens/s, ips: 11176 tokens/s, learning rate: 5.55556e-08
[2022-09-21 05:44:00,965] [    INFO] - [train] epoch: 0, batch: 3, loss: 10.983654976, avg_batch_cost: 0.72319 sec, speed: 1.38 step/s, ips_total: 90620 tokens/s, ips: 11328 tokens/s, learning rate: 6.94444e-08
[2022-09-21 05:44:01,678] [    INFO] - [train] epoch: 0, batch: 4, loss: 11.014451981, avg_batch_cost: 0.71223 sec, speed: 1.40 step/s, ips_total: 92016 tokens/s, ips: 11502 tokens/s, learning rate: 8.33333e-08
[2022-09-21 05:44:02,385] [    INFO] - [train] epoch: 0, batch: 5, loss: 11.005180359, avg_batch_cost: 0.70707 sec, speed: 1.41 step/s, ips_total: 92687 tokens/s, ips: 11586 tokens/s, learning rate: 9.72222e-08
[2022-09-21 05:44:03,100] [    INFO] - [train] epoch: 0, batch: 6, loss: 10.989698410, avg_batch_cost: 0.71402 sec, speed: 1.40 step/s, ips_total: 91785 tokens/s, ips: 11473 tokens/s, learning rate: 1.11111e-07
[2022-09-21 05:44:03,806] [    INFO] - [train] epoch: 0, batch: 7, loss: 10.992337227, avg_batch_cost: 0.70554 sec, speed: 1.42 step/s, ips_total: 92888 tokens/s, ips: 11611 tokens/s, learning rate: 1.25000e-07
[2022-09-21 05:44:04,516] [    INFO] - [train] epoch: 0, batch: 8, loss: 10.972790718, avg_batch_cost: 0.71011 sec, speed: 1.41 step/s, ips_total: 92290 tokens/s, ips: 11536 tokens/s, learning rate: 1.38889e-07
[2022-09-21 05:44:05,228] [    INFO] - [train] epoch: 0, batch: 9, loss: 10.983499527, avg_batch_cost: 0.71128 sec, speed: 1.41 step/s, ips_total: 92138 tokens/s, ips: 11517 tokens/s, learning rate: 1.52778e-07
```

### 多机训练

若需要在更多机器上进行大模型训练，则需要在每个参与训练的节点上设置master节点ip/port信息后执行启动命令（master节点ip为训练所用某一台机器的ip即可）。

以2机16卡32G V100上的6.7B模型分组切分并行训练为例，启动命令为：

```shell
master_ip=master节点ip
master_port=可用的空闲端口号

log_dir=log_sharding16
python -m paddle.distributed.launch --log_dir $log_dir \
    --master=$master_ip:$master_port --nnodes=2 --devices "0,1,2,3,4,5,6,7" \
    tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml
```

若要在显存容量更小的16G V100环境下进行GPT模型两机训练，也可通过减小`Model.hidden_size`调整模型规模至合适大小再启动训练，命令如下：

```shell
master_ip=master节点ip
master_port=可用的空闲端口号

log_dir=log_sharding16
python -m paddle.distributed.launch --log_dir $log_dir \
    --master=$master_ip:$master_port --nnodes=2 --devices "0,1,2,3,4,5,6,7" tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
    -o Model.hidden_size=2048
```

若要执行16机175B大模型混合并行训练，以运行启动命令为：

```shell
master_ip=master节点ip
master_port=可用的空闲端口号

log_dir=log_mp8_pp16
python -m paddle.distributed.launch --log_dir $log_dir \
    --master=$master_ip:$master_port --nnodes=16 --devices "0,1,2,3,4,5,6,7" tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml
```

当节点较多时，可以考虑使用 `ssh` 脚本或 `mpirun` 进行跨节点命令分发。

### 量化训练


若需要对模型进行量化训练，按照以上在配置文件中添加量化参数，可参考`qat_gpt_345M_mp8.yaml`，量化训练时可以可以适当减少训练轮数和学习率。以单机345M模型模型并行训练为例，通过``paddle.distributed.launch``启动多进程训练，该gpt程序需要8卡32G V100以运行，命令如下：

```shell
log_dir=log_mp8
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" tools/train.py \
    -c ppfleetx/configs/nlp/gpt/qat_gpt_345M_mp8.yaml
    -o Engine.max_steps=100000 \
    -o Optimizer.lr.decay_steps=72000 \
    -o Optimizer.lr.max_lr=5.0e-6 \
    -o Optimizer.lr.min_lr=1.0e-6
```


# GPT Zero-shot 文本生成

## 参数释义

```yaml
Generation:
  top_k: 50
  top_p: 0.75
  temperature: 1.0
  min_dec_len: 1
  max_dec_len: 200
  num_return_sequences: 1
  decode_strategy: "sampling"
```

其中参数说明：

| **参数名**      | **参数释义**                  |
|--------------|---------------------------|
| top_k | 每次为采样挑选保留分数最高的 k 个 token        |
| top_p   | 如果设置小于 1.0 的小数，则保留加起来为 top_p 或更高的最可能的概率的 token。默认值为 1.0        |
| temperature   |  调节下一个 token 的概率温度，logits = logits / temperature，默认值为 1.0           |
| min_dec_len | 最小生成 token 长度              |
| max_dec_len  | 最大生成 token 长度                     |
| num_return_sequences  | 每个输入生成的序列个数，默认值为 1                  |
| decode_strategy       | 解码策略，默认值为 "sampling"，目前只支持 "sampling"，未来会支持 "greedy_search"，"beam_search" |

## 文本生成

下载预训练好的模型，快速体验文本生成

```shell
cd PaddleNLP/model_zoo/gpt-3 # 如果已在 PaddleNLP/model_zoo/gpt-3 目录下，则忽略

mkdir -p ckpt
wget -O ckpt/GPT_345M.tar.gz https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
tar -xzf ckpt/GPT_345M.tar.gz -C ckpt/

# --devices 根据并行策略设置设备

python -m paddle.distributed.launch --devices "0" tasks/gpt/generation.py \
    -c ppfleetx/configs/nlp/gpt/generation_gpt_345M_dp8.yaml \
    -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/

# 生成的文本，由于 checkpoint 不同，超参不同，随机数不同，您执行可能会生成不一样的内容

Prompt: Hi, GPT2. Tell me who Jack Ma is.
Generation: Hi, GPT2. Tell me who Jack Ma is. I don’t want to hear that.”

For now, the only question the crowd is asking is whether or not Jack Ma will step down from the board of directors of Alibaba.

Jack Ma on why he never wanted to run for President in 2016:

There were two reasons. One is that I wanted to spend more time with my family. I thought it was better to spend more time with my family and spend more time with my children. So it was a very personal reason. But the second reason was that I thought it would be difficult to get elected, because there are a lot of political interests in this country. So I thought it was better to spend more time with my family.

On how Alibaba will evolve into a new player in China’s transportation and logistics sector:

I think that we are going to become a very important player in the logistics industry. So our strategy is to make it easy for people to travel.
```

### 剖析体验文本生成

#### GPT 文本生成模块初始化

```python
    module = build_module(cfg)
    module.model.eval()
```

#### 预训练模型加载

```python
    # 获取到预训练 checkpoint 的根目录
    ckpt_dir = cfg.Engine.save_load.ckpt_dir

    # 构造出具体路径
    model_path = os.path.join(ckpt_dir, "model.pdparams")

    # 加载模型参数
    model_dict = paddle.load(model_path)

    # FP16 模型参数转成 FP32 模型参数
    for key, value in model_dict.items():
        model_dict[key] = model_dict[key].astype(paddle.float32)

    # 设置模型参数为预训练参数
    module.model.set_state_dict(model_dict)
```

#### 文本生成与结果展示

```python
    input_text = "Historical Records: Tell us about the history of the Great Wall."
    result = module.generate(input_text)

    print(f'Prompt: {input_text}')
    print(f'Generation: {result[0]}')
```
