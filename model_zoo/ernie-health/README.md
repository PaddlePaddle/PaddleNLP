# ERNIE-Health 中文医疗预训练模型

医疗领域存在大量的专业知识和医学术语，人类经过长时间的学习才能成为一名优秀的医生。那机器如何才能“读懂”医疗文献呢？尤其是面对电子病历、生物医疗文献中存在的大量非结构化、非标准化文本，计算机是无法直接使用、处理的。这就需要自然语言处理（Natural Language Processing，NLP）技术大展身手了。

## 模型介绍

本项目针对中文医疗语言理解任务，开源了中文医疗预训练模型 [ERNIE-Health](https://arxiv.org/pdf/2110.07244.pdf)（模型名称`ernie-health-chinese`）。

ERNIE-Health 依托百度文心 ERNIE 先进的知识增强预训练语言模型打造, 通过医疗知识增强技术进一步学习海量的医疗数据, 精准地掌握了专业的医学知识。ERNIE-Health 利用医疗实体掩码策略对专业术语等实体级知识学习, 学会了海量的医疗实体知识。同时，通过医疗问答匹配任务学习病患病状描述与医生专业治疗方案的对应关系，获得了医疗实体知识之间的内在联系。ERNIE-Health 共学习了 60 多万的医疗专业术语和 4000 多万的医疗专业问答数据，大幅提升了对医疗专业知识的理解和建模能力。此外，ERNIE-Health 还探索了多级语义判别预训练任务，提升了模型对医疗知识的学习效率。该模型的整体结构与 ELECTRA 相似，包括生成器和判别器两部分。

![Overview_of_EHealth](https://user-images.githubusercontent.com/25607475/163949632-8b34e23c-d0cd-49df-8d88-8549a253d221.png)

更多技术细节可参考论文
- [Building Chinese Biomedical Language Models via Multi-Level Text Discrimination, ACL 2022](https://arxiv.org/pdf/2110.07244.pdf)

## 模型效果

ERNIE-Health模型以超越人类医学专家水平的成绩登顶中文医疗信息处理权威榜单 [CBLUE](https://github.com/CBLUEbenchmark/CBLUE) 冠军, 验证了 ERNIE 在医疗行业应用的重要价值。

![CBLUERank](https://user-images.githubusercontent.com/25607475/160394225-04f75498-ce1a-4665-85f7-d495815eed51.png)

相应的开源模型参数 ``ernie-health-chinese`` 在 CBLUE **验证集** 上的评测指标如下表所示：

| Task      |  metric  | results | results (fp16) |
| --------- | :------: | :-----: | :------------: |
| CHIP-STS  | Macro-F1 | 0.88749 |    0.88555     |
| CHIP-CTC  | Macro-F1 | 0.84136 |    0.83514     |
| CHIP-CDN  |    F1    | 0.76979 |    0.76489     |
| KUAKE-QQR | Accuracy | 0.83865 |    0.84053     |
| KUAKE-QTR | Accuracy | 0.69722 |    0.69722     |
| KUAKE-QIC | Accuracy | 0.81483 |    0.82046     |
| CMeEE     | Micro-F1 | 0.66120 |    0.66026     |
| CMeIE     | Micro-F1 | 0.61385 |    0.60076     |

## 环境依赖

- paddlepaddle >= 2.2.0
- paddlenlp >= 2.3.4

## 模型预训练

PaddleNLP中提供了ERNIE-Health训练好的模型参数。``ernie-health-chinese``版本为160G医疗文本数据上的训练结果，数据包括脱敏医患对话语料、医疗健康科普文章、脱敏医院电子医疗病例档案以及医学和临床病理学教材。本节提供了预训练的整体流程，可用于自定义数据的学习。

#### 注意: 预训练资源要求

- 推荐使用至少4张16G以上显存的GPU进行预训练。
- 数据量应尽可能接近ERNIE-Health论文中训练数据的量级，以获得好的预训练模型效果。
- 若资源有限，可以直接使用开源的ERNIE-Health模型进行微调，具体实现可参考 [CBLUE样例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-health/cblue)。

#### 数据准备

- 数据编码：UTF-8
- 数据格式：预训练文本数据放在同个目录下，每个文件中每行一句中文文本。

- 数据预处理：首先对原始文本进行分词，分词结果中非首中文字符替换为``##``前缀的字符（例如，``医疗``处理后得到``[医, ##疗]``）。接着将token转换为对应的id。最后将目录下的全部数据合并存储，token ids拼接后存储至``.npy``文件，每条样本的长度存储在``.npz``文件。

```shell
python preprocess.py --input_path ./raw_data/ --output_file ./data/samples --tokenize_tool lac --num_worker 8
```
可配置参数包括
- ``input_path`` 为原始文本数据所在目录，该目录下包含至少一个中文文本文件，UTF-8编码。
- ``output_file`` 为预处理后数据的存储路径及文件名（不包含后缀）。
- ``tokenize_tool``表示分词工具，包括``lac``、``seg``和``jieba``，默认为``lac``。
- ``logging_steps`` 表示日志打印间隔，每处理``logging_steps``个句子打印一次日志。
- ``num_worker`` 表示使用的进程数，增加进程数可加速预处理。


#### 单机单卡

```
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
    --input_dir ./data \
    --output_dir ./output \
    --learning_rate 1e-7 \
    --batch_size 10 \
    --adam_epsilon 1e-8 \
    --weight_decay 1e-2 \
    --warmup_steps 10000 \
    --max_steps 1000000 \
    --save_steps 10000 \
    --logging_steps 1 \
    --seed 1000 \
    --use_amp
```

#### 单机多卡

```
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0,1,2,3" run_pretrain.py \
    --input_dir ./data \
    --output_dir ./output \
    --learning_rate 1e-7 \
    --batch_size 10 \
    --adam_epsilon 1e-8 \
    --weight_decay 1e-2 \
    --warmup_steps 10000 \
    --max_steps 1000000 \
    --save_steps 10000 \
    --logging_steps 1 \
    --seed 1000 \
    --use_amp
```

可配置参数包括
- ``model_name_or_path``表示内置模型参数名（目前支持``ernie-health-chinese``），或者模型参数配置路径（这时需配置 --init_from_ckpt 参数一起使用，一般用于断点恢复训练场景。）
- ``input_dir``表示训练数据所在目录，该目录下要有``.npy``和``.npz``两个文件，格式与```preprocess.py``预处理结果相同。
- ``output_dir``表示预训练模型参数和训练日志的保存目录。
- ``batch_size``表示每次迭代每张卡上的样本数量。当batch_size=4时，运行时单卡约需要12G显存。如果实际GPU显存小于12G或大大多于12G，可适当调小/调大此配置。
- ``learning_rate`` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- ``max_seq_length`` 表示最大句子长度，超过该长度将被截断。
- ``weight_decay`` 表示每次迭代中参数缩小的比例，该值乘以学习率为真正缩小的比例。
- ``adam_epsilon`` 表示adam优化器中的epsilon值。
- ``warmup_steps`` 表示学习率逐渐升高到基础学习率（即上面配置的learning_rate）所需要的迭代数，最早的使用可以参考[这篇论文](https://arxiv.org/pdf/1706.02677.pdf)。
- ``num_epochs`` 表示训练轮数。
- ``logging_steps`` 表示日志打印间隔。
- ``save_steps`` 表示模型保存间隔。
- ``max_steps`` 如果配置且大于0，表示预训练最多执行的迭代数量；如果不配置或配置小于0，则根据输入数据量、``batch_size``和``num_epochs``来确定预训练迭代数量。
- ``device`` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。
- ``use_amp`` 表示是否开启混合精度(float16)进行训练，默认不开启。如果在命令中加上了--use_amp，则会开启。
- ``init_from_ckpt`` 表示是否从某个checkpoint继续训练（断点恢复训练），默认不开启。如果在命令中加上了--init_from_ckpt，且 --model_name_or_path 配置的是路径，则会开启从某个checkpoint继续训练。

#### Trainer 训练版本
本样例同时提供了Trainer版本的预训练流程，预训练重启、可视化等流程较为完备。需要从源码安装paddlenlp使用。

```
unset CUDA_VISIBLE_DEVICES
task_name="eheath-pretraining"

python -u -m paddle.distributed.launch \
    --gpus 0,1,2,3,4,5,6,7  \
    run_pretrain_trainer.py \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_length 512 \
    --gradient_accumulation_steps 1\
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 0.001 \
    --max_steps 1000000 \
    --save_steps 50000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 20 \
    --dataloader_num_workers 2 \
    --device "gpu"\
    --fp16  \
    --fp16_opt_level "O1"  \
    --do_train \
    --disable_tqdm \
    --save_total_limit 10
```
大部分参数含义如上文所述，这里简要介绍一些新参数:

- ``per_device_train_batch_size`` 同上文batch_size。训练时，每次迭代每张卡上的样本数目。
- ``per_device_eval_batch_size`` 同上文batch_size。评估时，每次迭代每张卡上的样本数目。
- ``warmup_ratio`` 与warmup_steps类似，warmup步数占总步数的比例。
- ``fp16`` 与`use_amp`相同，表示使用混合精度
- ``fp16_opt_level`` 混合精度的策略。注：O2训练eHealth存在部分问题，暂时请勿使用。
- ``save_total_limit`` 保存的ckpt数量的最大限制

## 微调

模型预训练结束后，可以对判别器进行微调以完成下游医疗任务。不同任务的模型加载方式如下：

```
from paddlenlp.transformers import *

tokenizer = AutoTokenizer.from_pretrained('ernie-health-chinese')

# 分类任务
model = AutoModelForSequenceClassification.from_pretrained('ernie-health-chinese')
# 序列标注任务
model = AutoModelForTokenClassification.from_pretrained('ernie-health-chinese')
# 阅读理解任务
model = AutoModelForQuestionAnswering.from_pretrained('ernie-health-chinese')
```

本项目提供了在 CBLUE 数据集上的微调脚本，包括分类、实体识别和关系抽取三类任务，详细信息可参考 ``cblue``[目录](./cblue)。

## 部署

我们为ERNIE-Health微调后的模型提供了Python端部署方案，请根据实际情况进行实现。

详细部署流程请参考：[基于ONNXRuntime推理部署指南](./cblue/deploy/predictor/)


## Reference

Wang, Quan, et al. “Building Chinese Biomedical Language Models via Multi-Level Text Discrimination.” arXiv preprint arXiv:2110.07244 (2021). [pdf](https://arxiv.org/abs/2110.07244)
