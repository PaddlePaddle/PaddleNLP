
 **目录**

* [背景介绍](#背景介绍)
* [ERNIE 1.0](#ERNIE1.0)
    * [1. 技术方案和评估指标](#技术方案)
    * [2. 环境依赖](#环境依赖)  
    * [3. 代码结构](#代码结构)
    * [4. 数据准备](#数据准备)
    * [5. 模型训练](#模型训练)
    * [6. 模型转换](#模型转换)

<a name="背景介绍"></a>

# 背景介绍


ERNIE是百度开创性提出的基于知识增强的持续学习语义理解框架，它将大数据预训练与多源丰富知识相结合，通过持续学习技术，不断吸收海量文本数据中词汇、结构、语义等方面的知识，实现模型效果不断进化。

ERNIE在情感分析、文本匹配、自然语言推理、词法分析、阅读理解、智能问答等16个公开数据集上全面显著超越世界领先技术，在国际权威的通用语言理解评估基准GLUE上，得分首次突破90分，获得全球第一。
相关创新成果也被国际顶级学术会议AAAI、IJCAI收录。
同时，ERNIE在工业界得到了大规模应用，如搜索引擎、新闻推荐、广告系统、语音交互、智能客服等。

本示例采用了全新数据流程，适配了ERNIE预训练任务，具有高效易用，方便快捷的特点。支持动态文本mask，自动断点训练重启等。
用户可以根据自己的需求，灵活修改mask方式。具体可以参考`./data_tools/dataset_utils.py`中`create_masked_lm_predictions`函数。
用户可以设置`checkpoint_steps`，间隔`checkpoint_steps`数，即保留最新的checkpoint到`model_last`文件夹。重启训练时，程序默认从最新checkpoint重启训练，学习率、数据集都可以恢复到checkpoint时候的状态。


<a name="ERNIE 1.0"></a>

# ERNIE 1.0


<a name="技术方案"></a>

## 1. 技术方案和评估指标

### 技术方案
采用ERNIE1.0预训练垂直领域的模型


<a name="环境依赖"></a>

## 2. 环境依赖和安装说明

**环境依赖**
* python >= 3.6
* paddlepaddle >= 2.1.3
* paddlenlp >= 2.2
* visualdl >=2.2.2
* pybind11

安装命令 `pip install visualdl pybind11`

<a name="代码结构"></a>

## 3. 代码结构

以下是本项目主要代码结构及说明：

```
ERNIE 1.0/
|—— scripts
    |—— run_pretrain_static.sh # 静态图与训练bash脚本
├── ernie_static_to_dynamic.py # 静态图转动态图
├── run_pretrain_static.py # ernie1.0静态图预训练
├── args.py # 预训练的参数配置文件
└── data_tools # 预训练数据处理文件目录
```

<a name="数据准备"></a>

## 4. 数据准备

数据准备部分请移步[data_tools](./data_tools/)目录，根据文档，创建训练数据。

## 5. 模型训练

**领域适应模型下载链接：**

|Model|训练参数配置|硬件|MD5|
| ------------ | ------------ | ------------ |-----------|
|[ERNIE 1.0](https://bj.bcebos.com/v1/paddlenlp/models/ernie_pretrain.zip)|<div style="width: 150pt">max_lr:0.0001 min_lr:0.00001  bs:512 max_len:512 </div>|<div style="width: 100pt">4卡 v100-32g</div>|91b0e30c444ab654dd99c0ee354f6a8f|

### 训练环境说明


- NVIDIA Driver Version: 440.64.00
- Ubuntu 16.04.6 LTS (Docker)
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz


### 单机单卡训练/单机多卡训练

这里采用单机多卡方式进行训练，通过如下命令，指定 GPU 0,1,2,3 卡, 基于SimCSE训练模型，数据量比较小，几分钟就可以完成。如果采用单机单卡训练，只需要把 `--gpus` 参数设置成单卡的卡号即可



### 模型训练


```
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3" \
    --log_dir "output/$task_name/log" \
    run_pretrain_static.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_len 512 \
    --micro_batch_size 32 \
    --global_batch_size 128 \
    --sharding_degree 1\
    --dp_degree 4 \
    --use_sharding false \
    --use_amp true \
    --use_recompute false \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 200000 \
    --save_steps 100000 \
    --checkpoint_steps 5000 \
    --decay_steps 1980000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --num_workers 2 \
    --logging_freq 20\
    --eval_freq 1000 \
    --device "gpu"
```
也可以直接运行脚本：

```
sh scripts/run_pretrain_static.sh
```

其中参数释义如下：
- `model_name_or_path` 要训练的模型或者之前训练的checkpoint。
- `input_dir` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件。
- `output_dir` 指定输出文件。
- `max_seq_len` 输入文本序列的长度。
- `micro_batch_size` 单卡单次的 batch size大小。即单张卡运行一次前向网络的 batch size大小。
- `global_batch_size` 全局的batch size大小，即一次参数更新等效的batch size。
- `sharding_degree` 切参数切分的分组大小（如 sharding_degree=4 表示参数分为4组，分别到4个设备）。
- `dp_degree` 数据并行参数。
- `use_sharding` 开启sharding策略，sharding_degree > 1时，请设置为True。
- `use_amp` 开启混合精度策略。
- `use_recompute` 开启重计算策略。暂时未支持，后续将支持。
- `max_lr` 训练学习率。
- `min_lr` 学习率衰减的最小值。
- `max_steps` 最大训练步数。
- `save_steps` 保存模型间隔。
- `checkpoint_steps` 模型checkpoint间隔，用于模型断点重启训练。
- `weight_decay` 权重衰减参数。
- `warmup_rate` 学习率warmup参数。
- `grad_clip` 梯度裁剪范围。
- `logging_freq` 日志输出间隔。
- `eval_freq` 模型评估间隔。
- `device` 训练设备。

注：
- 一般而言，需要设置 `mp_degree * sharding_degree` = 训练机器的总卡数。
- 一般而言， `global_batch_size = micro_batch_size * sharding_degree * dp_degree`。可以使用梯度累积的方式增大`global_batch_size`。设置`global_batch_size`为理论值的整数倍是，默认启用梯度累积。
- 训练断点重启，直接启动即可，程序会找到最新的checkpoint，开始重启训练。

<a name="模型转换"></a>

## 6. 模型转换

### 静态图转动态图

修改代码中的路径：

```
static_model_path="./output/ERNIE 1.0-dp8-gb1024/model_last/static_vars"
```
然后运行
```
python ernie_static_to_dynamic.py
```
运行结束后，动态图的模型就会保存到ernie_checkpoint文件夹里，也可以根据情况，修改代码，保存到自己的指定路径

### 参考文献

- [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223.pdf)
