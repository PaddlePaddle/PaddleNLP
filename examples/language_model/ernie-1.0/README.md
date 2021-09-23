# Ernie-1.0 预训练

ERNIE是百度开创性提出的基于知识增强的持续学习语义理解框架，它将大数据预训练与多源丰富知识相结合，通过持续学习技术，不断吸收海量文本数据中词汇、结构、语义等方面的知识，实现模型效果不断进化。

ERNIE在情感分析、文本匹配、自然语言推理、词法分析、阅读理解、智能问答等16个公开数据集上全面显著超越世界领先技术，在国际权威的通用语言理解评估基准GLUE上，得分首次突破90分，获得全球第一。
相关创新成果也被国际顶级学术会议AAAI、IJCAI收录。
同时，ERNIE在工业界得到了大规模应用，如搜索引擎、新闻推荐、广告系统、语音交互、智能客服等。

本示例采用了全新数据流程，适配了ERNIE预训练任务，具有高效易用，方便快捷的特点。支持动态文本mask，自动断点训练重启等。
用户可以根据自己的需求，灵活修改mask方式。具体可以参考`../data_tools/dataset_utils.py`中`create_masked_lm_predictions`函数。
用户可以设置`checkpoint_steps`，间隔`checkpoint_steps`数，即保留最新的checkpoint到`model_last`文件夹。重启训练时，程序默认从最新checkpoint重启训练，学习率、数据集都可以恢复到checkpoint时候的状态。

### 环境依赖

- visualdl
- pybind11

安装命令 `pip install visualdl pybind11`

### 数据准备
数据准备部分请移步[data_tools](../data_tools/)目录，根据文档，创建训练数据。

### 使用方法
```
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/ernie-1.0-dp8-gb512/log" \
    run_pretrain_static.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0" \
    --input_dir "./data" \
    --output_dir "output/ernie-1.0-dp8-gb512" \
    --max_seq_len 512 \
    --micro_batch_size 64 \
    --global_batch_size 512 \
    --sharding_degree 1\
    --dp_degree 8 \
    --use_sharding false \
    --use_amp true \
    --use_recompute false \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 4000000 \
    --save_steps 50000 \
    --checkpoint_steps 5000 \
    --decay_steps 3960000 \
    --weight_decay 0.01 \
    --warmup_rate 0.0025 \
    --grad_clip 1.0 \
    --logging_freq 20\
    --num_workers 2 \
    --eval_freq 1000 \
    --device "gpu"\
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


### 参考文献
- [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223.pdf)
