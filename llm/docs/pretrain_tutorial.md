# PaddleNLP 大模型新手指南-预训练
本教程将从新手视角出发，讲解如何使用 PaddleNLP 进行大语言模型（LLM）预训练。我们以 Qwen2.5-0.5B 模型为例，运行在百度星河平台（AI Studio）上，完整展示数据准备、模型构建、训练启动及调优建议。

我们在 Ai Studio 上同步公开了项目，也可以点击[链接](https://aistudio.baidu.com/projectdetail/9038113)在线体验大模型预训练。

目标：
- 了解预训练任务基本流程
- 能运行 PaddleNLP 提供的训练脚本
- 会在自己的数据上复现训练过程

## 1. 依赖安装
首先安装 PaddlePaddle 和 PaddleNLP 的[最新版本](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)：


```python
# 安装PaddlePaddle最新版本
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
```


```python
# Clone PaddleNLP仓库，训练/微调/对齐/量化的脚本都在仓库的llm/目录下
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

## 2. 数据准备
大模型的预训练任务目标是让模型学习语言的结构和语义，因此数据通常是大规模的自然语言文本，如：
* 新闻、小说、百科

* 网络论坛、问答内容

这里的训练数据与我们常见的<数据，标签>的监督学习所用的数据并不相同。我们使用[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/)作为本次预训练的数据。

以[PaddleNLP 预训练数据流程](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/tools/preprocess)中的样例分析：

```
飞桨是功能完备、开源开放的产业级深度学习平台。
飞桨拥有核心训练和推理框架、基础模型库。
PaddleNLP是自然语言处理领域的优秀工具。
```
大模型的预训练数据就是这样的自然语言文本。大模型的预训练是一种无监督训练，基本的思想是根据之前的词语来预测下一个词，以第一句举例：
```
(输入)飞桨是功能完备、开源开放的产业级深度学习 -> (输出)平台
```
不需要额外的标注数据。

我们通过下面的命令下载已经预处理过的数据。


```python
# llama 模型数据下载
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx
```


```python
mkdir data
mv llama_openwebtext_100k.bin ./data
mv llama_openwebtext_100k.idx ./data
```

### 数据处理
我们下载的数据是经过处理的数据，虽然我们本次不需要自己处理，但是可以简单了解一下数据的格式。
* 原始数据：用换行符隔开的句子。
* json/jsonl：两者区别是 jsonl 是每行一个句子，json 完整格式相对复杂一些。```{"text": "PaddleNLP是自然语言..."}```
* 分词（可选）：```PaddleNLP 是 自然语言处理领域 的 优秀工具。```
* 转换为 ID：每个词会转换为一个数字 ID，最终形成一个 mmap（memory-mapped file）文件。bin 的二进制文件里面是所有文本的数字 ID，idx 文件里面是每句话的起始位置。

详细数据处理可以参考[PaddleNLP 预训练数据流程](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/tools/preprocess)。

## 3. 模型训练
PaddleNLP 实现了一个对常用开源大模型便捷的预训练方式，本次我们使用 Qwen2.5-0.5B 进行实验。从 PaddleNLP 支持的[模型列表](https://paddlenlp.readthedocs.io/zh/latest/llm/docs/pretrain.html#model-weight)将制定模型配置文件作为参数输入训练脚本即可开展训练。


```python
# 16G 显存可训练
python -u ~/PaddleNLP/llm/run_pretrain.py ~/PaddleNLP/llm/config/qwen/pretrain_argument_0p5b.json
```

看到下面的提示说明已经开始进行训练了：
```
[    INFO] - loss: 12.0635252, learning_rate: 2e-06, global_step: 1, current_memory_allocated: 7.549170255661011, current_memory_reserved: 7.753237724304199, max_memory_allocated: 7.549170255661011, max_memory_reserved: 7.753237724304199, interval_runtime: 1.1518, interval_samples_per_second: 0.8682, interval_tokens_per_second_per_device: 889.0448, interval_hardware_tflops_per_device: 2.77, interval_steps_per_second: 0.8682, progress_or_epoch: 0.0
[    INFO] - loss: 12.05887604, learning_rate: 3e-06, global_step: 2, current_memory_allocated: 7.549170255661011, current_memory_reserved: 12.4834623336792, max_memory_allocated: 12.307440280914307, max_memory_reserved: 12.4834623336792, interval_runtime: 0.2556, interval_samples_per_second: 3.913, interval_tokens_per_second_per_device: 4006.9441, interval_hardware_tflops_per_device: 12.49, interval_steps_per_second: 3.913, progress_or_epoch: 0.0
```

### FAQ1：显存不足
如果在训练时候看到类似的错误信息：
```
RuntimeError: CUDA out of memory. ......
```

说明训练所需的显存超过了当前显卡提供的最大显存，说明此时我们无法按照默认设置进行单卡训练，解决方式有如下几种：
* 更换拥有更大显存的显卡
* 使用模型量化或一些机制来节省显存
* 多卡并行训练

我们重点介绍一下除了更换显卡之外的另外两种解决方案。

### 3.1 节省显存
#### 模型量化
模型量化（Quantization）是指将模型中的权重和激活值从高精度（如 FP32）压缩为低精度（如 INT8 或 FP16），以减小模型大小、加快推理速度、降低内存/显存占用。这部分将在模型量化的指南当中进行介绍。

#### 其他机制
在模型的 config 文件中，修改以下参数：
```
    "use_flash_attention": false,
    "use_fused_rms_norm": false,
    ......
    "recompute": false,
```

**注意：**
1. Flash attention 对于显卡的硬件架构有要求，需要在 V100、H100等显卡上面才能开启，建议使用 cuda11.8及以上环境；
2. use_fused_rms_norm 需要安装自定义算子。如果安装后仍然找不到算子，需要额外设置 PYTHONPATH。

### 3.2 高性能/多卡/多机训练
单张显卡的显存/性能不足可以通过多卡的并行来进行解决，我们常见的大模型也是在很多张显卡上面进行并行训练的。飞桨大模型套件支持4D 并行，在实际使用上也很便捷。相比于单卡训练，主要的区别在于输入了多张显卡的编号。


```python
# 编译自定义算子，可选
cd ../slm/model_zoo/gpt-3/external_ops/ && python3 setup.py install && cd -
# 多卡模型预训练参考:
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_pretrain.py ./config/llama/pretrain_argument.json
# 多机训练参考: 占用45G显存左右
python -u -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7"  --master=192.168.1.1:8090 --nnodes=2  run_pretrain.py ./config/llama/pretrain_argument.json

```

### 3.3 训练结果
当训练结束时，可以看到下面的输出：
```
[    INFO] - Saving model checkpoint to ./checkpoints/pretrain_ckpts
[    INFO] - tokenizer config file saved in ./checkpoints/pretrain_ckpts/tokenizer_config.json
[    INFO] - Special tokens file saved in ./checkpoints/pretrain_ckpts/special_tokens_map.json
[    INFO] - added tokens file saved in ./checkpoints/pretrain_ckpts/added_tokens.json
[ WARNING] - Asynchronous saving is not supported for single card environment currently.
[    INFO] - Configuration saved in ./checkpoints/pretrain_ckpts/config.json
[    INFO] - Configuration saved in ./checkpoints/pretrain_ckpts/generation_config.json
[    INFO] - ***** train metrics *****
[    INFO] -   progress_or_epoch        =     0.0868
[    INFO] -   train_loss               =     5.4334
[    INFO] -   train_runtime            = 0:35:59.57
[    INFO] -   train_samples_per_second =     4.6305
[    INFO] -   train_steps_per_second   =     4.6305
[    INFO] - ***** Running Prediction *****
[    INFO] -   Num examples = 258
[    INFO] -   Total prediction steps = 129
[    INFO] -   Pre device batch size = 2
[    INFO] -   Total Batch size = 2
[    INFO] - ***** test metrics *****
[    INFO] -   test_loss               =     4.8691
[    INFO] -   test_runtime            = 0:00:12.72
[    INFO] -   test_samples_per_second =    20.2781
[    INFO] -   test_steps_per_second   =    10.1391
Effective Tokens per second: 4741.68
ips: 4741.68 tokens/s
```
说明我们已经成功训练并且将训练后的模型参数进行了保存，在 ```checkpoints/pretrain_ckpts```目录下。我们可以简单浏览一下目录，看看预训练后的模型文件是什么样子。


```python
ls -l checkpoints/pretrain_ckpts/
```

| 分类          | 代表文件                                                | 作用                   |
| ----------- | --------------------------------------------------- | -------------------- |
| 模型结构        | `config.json`                                       | 定义模型维度、层数等超参数        |
| 模型权重        | `model-*.safetensors` + index 文件                    | 保存 Transformer 模型的参数 |
| 分词器         | `vocab.json`, `merges.txt`, `tokenizer_config.json` | 定义 tokenizer 行为和词表   |
| 特殊 token 信息 | `added_tokens.json`, `special_tokens_map.json`      | 管理新增或特殊 token        |
| 训练状态        | `trainer_state.json`, `training_args.bin`           | 记录训练进度和参数            |
| 评估结果        | `all_results.json`, `train_results.json`            | 保存训练评估指标结果           |
| TensorBoard | `runs/`                                             | 可视化训练曲线              |
| 检查点         | `checkpoint-*/`                                     | 每 N 步保存的快照           |
