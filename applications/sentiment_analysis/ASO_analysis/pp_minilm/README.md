# 基于 PP-MiniLM 的小模型优化策略

本项目中，无论是评论观点抽取模型，还是属性级情感分类模型，使用的均是 Large 版的 SKEP 模型，考虑到企业用户在线上部署时会考虑到模型预测效率，所以本项目提供了开源小模型 [PP-MiniLM](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression/pp-minilm) 及量化加速方案，大幅提升预测性能。

在本项目中，我们基于 PP-MiniLM 中文特色小模型进行 fine-tune 属性级情感分类模型，然后使用 [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) 进行模型量化，减小模型规模，加快模型预测性能。

## 1. 基于 PP-MiniLM 训练属性级情感分类模型

本实验的方案设计和基于 SKEP 的细粒度情感分类一样，有需要的同学请移步[这里](./../classification/README.md)，这里不再赘述。

### 1.1 项目结构说明
以下是本项目运行的完整目录结构及说明：

```shell
.
├── data.py                   # 数据处理脚本
├── model.py                  # 模型组网脚本
├── train.py                  # 模型训练脚本
├── evaluate.py               # 模型评估脚本
├── quant_post.py             # 模型量化脚本
├── performance_test.py       # 静态图预测脚本
├── run_train.sh              # 模型训练命令
├── run_evaluate.sh           # 模型评估命令
├── run_quant.sh              # 模型量化命令
├── run_performance_test.sh   # 静态图预测命令
└── README.md
```

### 1.2 数据说明

本实验数据和基于SKEP的细粒度情感分类实验所用数据是同一份，如果已将数据下载，并放入父目录的`data/cls_data/`目录下，则无需重复下载操作。更多信息请参考[这里](../classification/README.md)。

### 1.3 模型效果展示

在分类模型训练过程中，总共训练了10轮，并选择了评估 F1 得分最高的 best 模型， 下表展示了训练过程中使用的训练参数。我们同时开源了相应的模型，可点击下表的 `PP-MiniLM_cls` 进行下载，下载后将模型重命名为 `best.pdparams`，然后放入父目录的 `checkpoints/pp_checkpoints` 中。
|Model|训练参数配置|MD5|
| ------------ | ------------ |-----------|
|[PP-MiniLM_cls](https://bj.bcebos.com/paddlenlp/models/best_mini.pdparams)|<div style="width: 150pt"> learning_rate: 3e-5, batch_size: 16, max_seq_len:256, epochs：10 </div>|643d358620e84879921b42d326f97aae|

我们基于训练过程中的 best 模型在 `cls_data` 验证集 `dev` 和测试集 `test` 上进行了评估测试，模型效果如下表所示:
|Model|数据集|precision|Recall|F1|
| ------------ | ------------ | ------------ |-----------|------------ |
|PP-MiniLM|dev_set|0.98668|0.99115|0.98891|
|PP-MiniLM|test_set|0.98263|0.98766|0.98514|

**备注**：以上数据是基于全量数据训练和测试结果，并非 Demo 数据集。

### 1.4 模型训练
通过运行以下命令进行分类小模型训练，模型训练后会默认保存到父目录的`checkpoints/pp_checkpoints/`文件夹下：
```shell
sh run_train.sh
```

### 1.5 模型测试
通过运行以下命令进行分类小模型测试：
```shell
sh run_evaluate.sh
```

## 2. 对 PP-MiniLM 小模型进行量化
本节将基于 [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) ，对训练好的 PP-MiniLM 小模型进行量化。具体来讲，本节采用的是静态离线量化方法，即在训练好的模型基础上，使用少量校准数据计算量化因子，便可快速得到量化模型。量化过程中，默认使用 `avg` 的量化策略，对 `matmul/matmul_v2` 算子进行 `channel_wise_abs_max` 类型的量化。

首先，需要先将训练好的动态图模型，转为静态图模型，注意这里需要跳到父目录进行操作：
```shell
cd ..
sh run_export_model.sh pp_minilm
```

然后，使用如下命令进行量化生成的静态图模型：
```shell
sh run_quant.sh
```
执行以上命令时，需要使用 `static_model_dir` 指定待量化的模型目录，量化后，模型将会被保存在 `quant_model_dir` 指定的目录中。

最后，对量化后的小模型可使用 `performance_test.py` 进行评估， 该脚本主要用于性能测试，如果需要做评估，需要设置 `--eval`，如下所示：
```shell
python  performance_test.py \
        --base_model_name "ppminilm-6l-768h" \
        --model_path "../checkpoints/pp_checkpoints/quant/infer" \
        --test_path "../data/cls_data/test.txt" \
        --label_path "../data/cls_data/label.dict" \
        --batch_size 16 \
        --max_seq_len 256 \
        --eval
```

## 3. 对量化后的小模型进行性能测试

### 3.1 环境要求
本节需要使用安装有 Paddle Inference 预测库的 [PaddlePaddle 2.2.1](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html) 进行预测，请根据合适的机器环境进行下载安装。若想要得到明显的加速效果，推荐在 NVIDA Tensor Core GPU（如 T4、A10、A100) 上进行测试，若在 V 系列 GPU 卡上测试，由于其不支持 Int8 Tensor Core，将达不到预期的加速效果。

**备注**：本项目基于T4进行性能测试。

### 3.2 运行方式
本项目使用了动态 shape 功能 (tuned_dynamic_shape)，因此需要设置获取 shape 的范围。Paddle Inference 提供了相应的接口，即首先通过离线输入数据来统计出所有临时 tensor 的 shape 范围，TensorRT 子图的 tensor 输入 shape 范围可直接根据上一步 tune 出来的结果来设置，即可完成自动 shape 范围设置。统计完成后，只需设置统计结果路径，即可启用 tuned_dynamic_shape 功能。

在本案例中，进行性能测试的脚本为 `performance_test.py`，需要先设置 `--collect_shape` 参数，然后再取消传入这个参数，再次运行 `performance_test.py`。可通过设置 `--num_epochs` 计算多轮运行时间，然后取平均时间作为最终结果，具体使用方式如下：

首先，设置 `--collect_shape` 参数，生成 shape range info 文件：
```shell
python  performance_test.py \
        --base_model_name "ppminilm-6l-768h" \
        --model_path "../checkpoints/pp_checkpoints/quant/infer" \
        --test_path "../data/cls_data/test.txt" \
        --label_path "../data/cls_data/label.dict" \
        --num_epochs 1 \
        --batch_size 16 \
        --max_seq_len 256 \
        --use_tensorrt \
        --int8 \
        --collect_shape
```
然后，开始进行性能测试：
```shell
python  performance_test.py \
        --base_model_name "ppminilm-6l-768h" \
        --model_path "../checkpoints/pp_checkpoints/quant/infer" \
        --test_path "../data/cls_data/test.txt" \
        --label_path "../data/cls_data/label.dict" \
        --num_epochs 10 \
        --batch_size 16 \
        --max_seq_len 256 \
        --use_tensorrt \
        --int8 \
```


## 4. PP-MiniLM 模型效果展示

关于 SKEP-Large、PP-MiniLM、量化PP-MiniLM 三个模型在性能和效果方面的对比如下表所示。可以看到，三者在本任务数据集上的评估指标几乎相等，但是 PP-MiniLM 小模型运行速度较 SKEP-Large 提高了4倍，量化后的 PP-MiniLM 运行速度较 SKEP-Large 提高了近8倍。

|Model|运行时间(s)|precision|Recall|F1|
| ------------ | ------------ | ------------ |-----------|------------ |
|SKEP-Large|1.00x|0.98497|0.99139|0.98817|
|PP-MiniLM|4.95x|0.98263|0.98766|0.98514|
|量化 PP-MiniLM|8.93x|0.97696|0.98720|0.98205|
