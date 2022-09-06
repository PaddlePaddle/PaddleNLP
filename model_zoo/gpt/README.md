# GPT

## 模型介绍
GPT-[2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)/[3](https://arxiv.org/pdf/2005.14165.pdf) 是以[Transformer](https://arxiv.org/abs/1706.03762) 解码器为网络基本组件，使用自回归的方式在大规模无标注文本语料上进行预训练得到的语言生成模型。

本项目是语言模型 GPT 的 PaddlePaddle 实现， 包含模型训练，预测等内容。下是本例的简要目录结构及说明：

```text
.
├── args.py                 # 训练参数配置
├── converter.py            # 权重转化脚本
├── dataset.py              # 数据处理
├── deploy/                 # 模型部署的inference脚本
├── export_model.py         # 导出预测部署的模型脚本
├── faster_gpt/             # 使用 FasterGPT 高性能预测 sample
├── lr.py                   # 学习率控制
├── predict.py              # 生成文本示例demo
├── README.md               # 文档
├── run_eval.py             # 评估入口
├── run_pretrain.py         # 预训练入口
├── run_pretrain_static.py  # 混合并行，预训练脚本
└── scripts/                # 训练脚本
```

## 快速开始

### 环境依赖

- regex
- sentencepiece >= 0.1.94
- tqdm
- visualdl
- paddlepaddle-gpu >= 2.2rc
- pybind11
- lac (可选)
- zstandard (可选)

安装命令 `pip install regex sentencepiece tqdm visualdl pybind11 lac zstandard`。
注：需要PaddlePaddle版本大于等于2.2rc，或者使用最新develop版本，安装方法请参见Paddle[官网](https://www.paddlepaddle.org.cn)。

### 数据准备

#### 数据获取与制作

[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/)是一个开源的英文网页文本数据集，数据来源于Reddit，经过去重、清洗、提取，最终包含800多万个文档。
本示例采用EleutherAI清洗好的[OpenWebText2数据](https://openwebtext2.readthedocs.io/en/latest/index.html#download-plug-and-play-version)

下载以后通过以下命令解压：

```shell
wget https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar
tar -xvf openwebtext2.json.zst.tar -C  /path/to/openwebtext
```

然后使用[data_tools](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/gpt/../ernie-1.0/data_tools) 工具下的`create_pretraining_data.py`脚本进行数据集制作：
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

### 模型训练

#### 单卡训练

```shell
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-en \
    --input_dir "./data"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --micro_batch_size 4\
    --device gpu
```

其中参数释义如下：
- `model_name_or_path` 要训练的模型或者之前训练的checkpoint。
- `input_dir` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件。
- `output_dir` 指定输出文件。
- `weight_decay` 权重衰减参数。
- `grad_clip` 梯度裁剪范围。
- `max_steps` 最大训练步数
- `save_steps` 保存模型间隔
- `mirco_batch_size` 训练的batch大小
- `device` 训练设备

用户也可以使用提供的shell脚本直接训练`sh scripts/run.sh`.

#### 单机多卡

同样，可以执行如下命令实现八卡训练：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-en \
    --input_dir "./data"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --micro_batch_size 4\
    --device gpu
```

用户也可以使用提供的shell脚本直接训练`sh scripts/run_multi.sh`.

### 模型评估

我们提供了对[WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)、[LAMBADA](https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl)两种数据集的评估脚本, 使用如下命令启动评估：

1. WikiText数据集评估
```bash
python run_eval.py --model_name gpt2-en \
    --eval_path ./wikitext-103/wiki.valid.tokens \
    --overlapping_eval 32 \
    --init_checkpoint_path ./output/model_100000/model_state.pdparams \
    --batch_size 8 \
    --device gpu
```

2. LAMBADA数据集评估
```bash
python run_eval.py --model_name gpt2-en \
    --eval_path ./lambada_test.jsonl \
    --cloze_eval \
    --init_checkpoint_path ./output/model_100000/model_state.pdparams \
    --batch_size 8 \
    --device gpu
```
其中参数释义如下：
`model_name` 使用的模型名称，如gpt2-en、gpt2-medium-en等。
`eval_path` 数据集地址。
`init_checkpoint_path` 模型参数地址。
`batch_size` batch size大小。
`device` 运行设备，cpu，gpu，xpu可选。
`overlapping_eval` wikitext数据集参数。
`cloze_eval` lambada数据参数，作为完型填空任务。

其中数据集WikiText采用的是PPL(perplexity)评估指标，LAMBADA采用的是ACC(accuracy)指标。

注：不设置`init_checkpoint_path` 参数时，可以评估默认预训练好的模型参数。


### 文本生成

本项目提供了简单的文本生成的demo，供用户测试文本生成效果。

```shell
# 中文示例
python predict.py gpt-cn
# 英文示例
python predict.py
```

生成效果展示:
```text
问题：中国的首都是哪里？答案：北京。
问题：苹果的CEO是谁? 答案：乔布斯。

默写古诗: 大漠孤烟直，长河落日圆。
举杯邀明月，对影成三人。

Question: Who is the CEO of Apple?
Answer: Tim Cook.
```

## 模型导出预测

下面提供了简单的示例，帮助用户将预训练模型导出成预测部署的参数。

导出中文模型
```"shell
python export_model.py --model_type=gpt-cn \
    --model_path=gpt-cpm-large-cn \
    --output_path=./infer_model/model
```
用户在`infer_model`中可以看到导出的文件。

对于导出的模型，我们提供了Python的infer脚本，调用预测库对简单的例子进行预测。
```shell
python deploy/python/inference.py --model_type gpt-cn \
    --model_path ./infer_model/model
```


导出英文模型
```"shell
python export_model.py --model_type=gpt \
    --model_path=gpt2-medium-en \
    --output_path=./infer_model/model

python deploy/python/inference.py --model_type gpt \
    --model_path ./infer_model/model
```

用户可以看到屏幕输出预测结果。

## Taskflow一键预测
可以使用PaddleNLP提供的Taskflow工具来进行知识问答和写诗，具体使用方法如下:

```python

from paddlenlp import Taskflow

# 默认是知识问答任务
qa = Taskflow("question_answering")
qa("中国的国土面积有多大？")
'''
[{'text': '中国的国土面积有多大？', 'answer': '960万平方公里。'}]
'''

qa(["中国国土面积有多大？", "中国的首都在哪里？"])
'''
[{'text': '中国国土面积有多大？', 'answer': '960万平方公里。'}, {'text': '中国的首都在哪里？', 'answer': '北京。'}]
'''

# 使用写诗任务进行写诗

 poetry = Taskflow("poetry_generation")
 poetry("林密不见人")
 '''
 [{'text': '林密不见人', 'answer': ',但闻人语响。'}]
 '''

 poetry(["林密不见人", "举头邀明月"])
 '''
 [{'text': '林密不见人', 'answer': ',但闻人语响。'}, {'text': '举头邀明月', 'answer': ',低头思故乡。'}]
 '''
```

### 文本分类

以GLUE中的SST-2任务为例，启动Fine-tuning的方式如下：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type gpt \
    --model_name_or_path gpt2-medium-en \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/ \
    --device gpu \
    --use_amp False
```

其中参数释义如下：
- `model_type` 指示了模型类型。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `task_name` 表示Fine-tuning的任务。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `use_amp` 指示是否启用自动混合精度训练。

基于`gpt2-medium-en`在SST-2任务上Fine-tuning后，在验证集上有如下结果：

| Task  | Metric                       | Result            |
|:-----:|:----------------------------:|:-----------------:|
| SST-2 | Accuracy                     |      0.94495      |


### 序列标注

以MSRA命名实体识别任务为例，启动Fine-tuning的方式如下：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_msra_ner.py \
    --model_name_or_path gpt-cpm-small-cn-distill \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 25 \
    --save_steps 250 \
    --output_dir ./tmp/msra_ner/ \
    --device gpu
```

其中参数释义如下：
- `model_name_or_path`: 指示了某种特定配置的模型。
- `max_seq_length`: 表示最大句子长度，超过该长度将被截断。
- `batch_size`: 表示每次迭代**每张卡**上的样本数目。
- `learning_rate`: 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs`: 表示训练轮数。
- `logging_steps`: 表示日志打印间隔。
- `save_steps`: 表示模型保存及评估间隔。
- `output_dir`: 表示模型保存路径。
- `device`: 训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。

基于`gpt-cpm-small-cn-distill`在MSRA的NER任务上Fine-tuning后，在验证集上有如下结果：

 Metric                       | Result      |
------------------------------|-------------|
Precision                     | 0.484939    |
Recall                        | 0.634716    |
F1                            | 0.549810    |

## 其他

本项目提供了Huggingface的权重转化示例`converter.py`，`python converter.py xxx-gpt.bin`即可完成转换。用户可以参考转化脚本，转换自己需要的模型权重。

## 参考文献
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
- [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)
