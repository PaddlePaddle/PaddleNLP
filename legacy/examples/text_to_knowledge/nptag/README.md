# 解语：NPTag（名词短语标注工具）

NPTag（名词短语标注工具）是首个能够覆盖所有中文名词性词汇及短语的细粒度知识标注工具，旨在解决NLP中，名词性短语收录不足，导致的OOV（out-of-vocabulary，超出收录词表）问题。可直接应用构造知识特征，辅助NLP任务。

## NPTag特点

- 包含2000+细粒度类别，覆盖所有中文名词性短语的词类体系，更丰富的知识标注结果
    - NPTag试用的词类体系未覆盖所有中文名词性短语的词类体系，对所有类目做了更细类目的识别（如注射剂、鱼类、博物馆等），共包含2000+细粒度类别，且可以直接关联百科知识树。
- 可自由定制的分类框架
    - NPTag开源版标注使用的词类体系是我们在实践中对**百科词条**分类应用较好的一个版本，用户可以自由定制自己的词类体系和训练样本，构建自己的NPTag，以获得更好的适配效果。例如，可按照自定义的类别构造训练样本，使用小学习率、短训练周期微调NPTag模型，即可获得自己定制的NPTag工具。

## NPTag模型介绍

NPTag使用[ERNIE-CTM](../ernie-ctm)+prompt训练而成，使用启发式搜索解码，保证分类结果都在标签体系之内。

### finetune任务

在微调任务中提供了一个中文名词短语标注的任务，旨在对中文名词短语进行细粒度分类。

#### 代码结构说明

```text
nptag/
├── deploy # 部署
│   └── python
│       └── predict.py # python预测部署示例
├── data.py # 训练数据处理脚本
├── export_model.py # 模型导出脚本
├── metric.py # 模型效果验证指标脚本
├── predict.py # 预测脚本
├── README.md # 使用说明
├── train.py # 训练脚本
└── utils.py  # 工具函数
```

#### 数据准备

执行以下命令，下载并解压示例数据集：

```bash
wget https://bj.bcebos.com/paddlenlp/paddlenlp/datasets/nptag_dataset.tar.gz && tar -zxvf nptag_dataset.tar.gz
```

解压之后
```text
data/
├── name_category_map.json # NPTag标签文件
├── dev.txt # 验证集
└── train.txt  # 训练集
```

数据集`train.txt`和`dev.txt`格式示例(text VS label)
```
石竹  植物
杂链聚合物   化学物质
罗伯特·布雷森   人
```

标签文件`name_category_map.json`格式示例，其中key为细粒度标签，即NPTag的预测结果；value为粗粒度标签，示例中对应WordTag的标签集合，用户可以根据场景需要自定义修改该标签映射
```
{
    "植物": "生物类_植物",
    "化学物质": "物体类_化学物质",
    "人": "人物类_实体",
}
```

#### 模型训练
```bash
python -m paddle.distributed.launch --gpus "0" train.py \
    --batch_size 64 \
    --learning_rate 1e-6 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 100 \
    --output_dir ./output \
    --device "gpu"
```

可支持配置的参数：
- `data_dir`: 数据集文件路径，默认数据集存放在当前目录data文件夹下。
- `init_from_ckpt`: 模型参数路径，热启动模型训练，默认为None。
- `output_dir`: 模型保存路径，默认保存在当前目录的output文件夹下。
- `max_seq_len`: 模型使用的最大序列长度，默认为64。
- `learning_rate`: finetune的最大学习率；默认为1e-6。
- `num_train_epochs`: 表示训练轮数，默认为3。
- `logging_steps`: 日志打印步数间隔，默认为10。
- `save_steps`: 模型保存的步数间隔， 默认为100。
- `batch_size`: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为64。
- `weight_decay`: 控制正则项力度的参数，用于防止过拟合，默认为0.0。
- `warmup_proportion`: 学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
- `adam_epsilon`: Adam优化器的参数，避免分母为零，默认为1e-8。
- `seed`: 随机种子，默认为1000。
- `device`: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。

### 基于动态图的预测

```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" predict.py \
    --device=gpu \
    --params_path ./output/model_100/model_state.pdparams
```

### 基于静态图的预测部署

使用动态图训练结束之后，可以将动态图参数导出成静态图参数，从而获得最优的预测部署性能，执行如下命令完成动态图转换静态图的功能：
```shell
python export_model.py --params_path=./output/model_100/model_state.pdparams --output_path=./export
```

导出静态图模型之后，可以用于部署，`deploy/python/predict.py`脚本提供了python部署预测示例。运行方式：
```shell
python deploy/python/predict.py --model_dir=./export
```

## Taskflow一键预测

除了以上的finetune示例，Taskflow内置了一个百度基于大规模标注汉语短语数据集训练的名词短语标注工具`NPTag`。用户可以方便地使用该工具完成对中文名词短语的一键预测。

```python
from paddlenlp import Taskflow

nptag = Taskflow("knowledge_mining", model="nptag")
nptag("糖醋排骨")
'''
[{'text': '糖醋排骨', 'label': '菜品'}]
'''

nptag(["糖醋排骨", "红曲霉菌"])
'''
[{'text': '糖醋排骨', 'label': '菜品'}, {'text': '红曲霉菌', 'label': '微生物'}]
'''

# 输出粗粒度类别标签`category`，即WordTag的词汇标签。
nptag = Taskflow("knowledge_mining", model="nptag", linking=True)
nptag(["糖醋排骨", "红曲霉菌"])

'''
[{'text': '糖醋排骨', 'label': '菜品', 'category': '饮食类_菜品'}, {'text': '红曲霉菌', 'label': '微生物', 'category': '生物类_微生物'}]
'''
```

## 在论文中引用NPTag

如果您的工作成果中使用了NPTag，请增加下述引用。我们非常乐于看到解语对您的工作带来帮助。

```
@article{zhao2020TermTree,
    title={TermTree and Knowledge Annotation Framework for Chinese Language Understanding},
    author={Zhao, Min and Qin, Huapeng and Zhang, Guoxin and Lyu, Yajuan and Zhu, Yong},
    technical report={Baidu, Inc. TR:2020-KG-TermTree},
    year={2020}
}
```


## 问题与反馈

解语在持续优化中，如果您有任何建议或问题，欢迎提交issue到Github。
