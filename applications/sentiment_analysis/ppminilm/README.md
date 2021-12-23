# 基于PP-MiniLM中文特色小模型进行细粒度情感分类

本项目中，无论是评论维度和观点抽取模型，还是细粒度情感分类模型，使用的均是Large版的SKEP模型，考虑到企业用户在线上部署时会考虑到模型预测效率，所以本项目提供了一套基于[PP-MiniLM](https://github.com/LiuChiachi/PaddleNLP/tree/add-ppminilm/examples/model_compression/PP-MiniLM)中文特色小模型的细粒度情感分类解决方案。

PP-MiniLM提供了一套完整的小模型优化方案：首先使用Task-agnostic的方式进行模型蒸馏、然后依托于[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) 进行模型裁剪、模型量化等模型压缩技术，有效减小了模型的规模，加快了模型运行速度。

本项目基于PP-MiniLM中文特色小模型进行fine-tune细粒度情感分类模型，然后使用PaddleSlim对训练好的模型进行量化操作。


## 1. 基于PP-MiniLM中文特色小模型训练细粒度情感分类模型

本项目基于PP-MiniLM中文特色小模型进行细粒度情感分类，具体的方案设计和基于SKEP的细粒度情感分类一样，有需要的同学请移步[这里](./../classification/README.md)，这里不再赘述。

### 1.1 项目结构说明
以下是本项目的简要目录结构及说明：

```shell
.
├── data                      # 数据目录
├── checkpoints               # 模型保存目录
│   └── static                # 静态图模型保存目录
├── data.py                   # 数据处理脚本
├── model.py                  # 模型组网脚本
├── train.py                  # 模型训练脚本
├── evaluate.py               # 模型评估脚本
├── quant_post.py             # 模型量化脚本
├── static_predict.py         # 静态图预测脚本
├── utils.py                  # 工具函数
├── run_train.sh              # 模型训练命令
├── run_evaluate.sh           # 模型评估命令
├── run_quant.sh              # 模型量化命令
├── run_static_predict.sh     # 静态图预测命令
└── README.md
```

### 1.2 数据说明

本模型将基于评论维度和观点进行细粒度的情感分析，因此数据集中需要包含3列数据：文本串和相应的序列标签数据，下面给出了一条样本，其中第1列是情感标签，第2列是评论维度和观点，第3列是原文。

> 1   口味清淡   口味很清淡，价格也比较公道

可点击[data_cls](https://bj.bcebos.com/v1/paddlenlp/data/data_ext.tar.gz)进行Demo数据下载，将数据解压之后放入本目录的`data`文件夹下。

### 1.3 模型效果展示

在分类模型训练过程中，总共训练了10轮，并选择了评估F1得分最高的best模型， 更加详细的训练参数设置如下表所示：
|Model|训练参数配置|硬件|MD5|
| ------------ | ------------ | ------------ |-----------|
|[PP-MiniLM_cls](https://bj.bcebos.com/paddlenlp/models/best_mini.pdparams)|<div style="width: 150pt"> learning_rate: 3e-5, batch_size: 16, max_seq_len:256, epochs：10 </div>|<div style="width: 100pt">Tesla V100-32g</div>|d04fc43efa61c77f47c23ef042dcb325|

我们基于训练过程中的best模型在验证集`dev_set`和测试集`test_set`上进行了评估测试，模型效果如下表所示:
|Model|数据集|precision|Recall|F1|
| ------------ | ------------ | ------------ |-----------|------------ |
|PP-MiniLM|dev_set|0.98624|0.99183|0.98903|
|PP-MiniLM|test_set|0.98379|0.98859|0.98618|

**备注**：以上数据是基于全量数据训练和测试结果，并非Demo数据集。

### 1.4 模型训练
通过运行以下命令进行分类小模型训练：
```shell
sh run_train.sh
```

### 1.5 模型测试
通过运行以下命令进行分类小模型测试：
```shell
sh run_evaluate.sh
```

## 2. 对PP-MiniLM小模型进行量化
本节将基于[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)，对训练好的PP-MiniLM小模型进行量化。具体来讲，本节采用的是静态离线量化方法，即在训练好的模型基础上，使用少量校准数据计算量化因子，便可快速得到量化模型。量化过程中，默认使用`avg`的量化策略，对`matmul/matmul_v2` 算子进行`channel_wise_abs_max`类型的量化。

首先，需要先将训练好的动态图模型，转为静态图模型：
```shell
cd ..
sh run_export.sh ppminilm
```

然后，使用如下命令进行量化生成的静态图模型：
```shell
sh run_quant.sh
```

最后，对量化后的小模型进行评估：
```shell
sh run_static_predict.sh
```

## 3. 对量化后的小模型进行性能测试

### 3.1 环境要求
本节需要使用安装有Inference预测库的[PaddlePaddle 2.2.1](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)进行预测，请根据合适的机器环境进行下载安装。若想要得到明显的加速效果，推荐在 NVIDA Tensor Core GPU（如 T4、A10、A100)上进行测试，若在V系列GPU卡上测试，由于其不支持 Int8 Tensor Core，将达不到预期的加速效果。

**备注**：本项目基于T4进行性能测试。

### 3.2 运行方式
本项目使用了动态 shape 功能(tuned_dynamic_shape)，因此需要设置获取 shape 的范围。Paddle Inference 提供了相应的接口，即首先通过离线输入数据来统计出所有临时 tensor 的 shape 范围，TensorRT 子图的 tensor 输入 shape 范围可直接根据上一步 tune 出来的结果来设置，即可完成自动 shape 范围设置。统计完成后，只需设置统计结果路径，即可启用 tuned_dynamic_shape 功能。

在本案例中，只需要先设置 `--collect_shape` 参数，运行 `static_predict.py`，然后再取消传入这个参数，再次运行`static_predict.py`。同时性能测试需要设置`--perf`，默认情况下会运行10轮，然后取平均时间作为最终结果，具体使用方式如下：

首先，设置`--collect_shape`参数，生成shape range info文件：
```shell
python  static_predict.py \
        --base_model_path "./checkpoints/ppminilm" \
        --model_path "./checkpoints/quant/infer" \
        --test_path "./data/test_cls.txt" \
        --label_path "./data/label_cls.dict" \
        --num_epochs 10 \
        --batch_size 16 \
        --max_seq_len 256 \
        --use_tensorrt \
        --int8 \
        --perf \
        --collect_shape
```
然后，基于shape range info文件进行预测：
```shell
python  static_predict.py \
        --base_model_path "./checkpoints/ppminilm" \
        --model_path "./checkpoints/quant/infer" \
        --test_path "./data/test_cls.txt" \
        --label_path "./data/label_cls.dict" \
        --num_epochs 10 \
        --batch_size 16 \
        --max_seq_len 256 \
        --use_tensorrt \
        --int8 \
        --perf
```


## 4. PP-MiniLM模型效果展示

关于SKEP-Large、PP-MiniLM，量化PP-MiniLM三个模型在性能和效果方面的对比如下表所示。可以看到，三者在本任务数据集上的评估指标几乎相等，但是PP-MiniLM小模型运行速度较SKEP-Large提高了4倍，量化后的PP-MiniLM运行速度较SKEP-Large提高了近8倍。

|Model|运行时间(s)|precision|Recall|F1|
| ------------ | ------------ | ------------ |-----------|------------ |
|SKEP-Large|1.00x|0.98497|0.99139|0.98817|
|PP-MiniLM|4.95x|0.98379|0.98859|0.98618|
|量化 PP-MiniLM|8.93x|0.98312|0.98953|0.98631|
