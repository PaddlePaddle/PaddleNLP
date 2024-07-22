# UIE Slim 数据蒸馏

在UIE强大的抽取能力背后，同样需要较大的算力支持计算。在一些工业应用场景中对性能的要求较高，若不能有效压缩则无法实际应用。因此，我们基于数据蒸馏技术构建了UIE Slim数据蒸馏系统。其原理是通过数据作为桥梁，将UIE模型的知识迁移到封闭域信息抽取小模型，以达到精度损失较小的情况下却能达到大幅度预测速度提升的效果。

#### UIE数据蒸馏三步

- **Step 1**: 使用UIE模型对标注数据进行finetune，得到Teacher Model。

- **Step 2**: 用户提供大规模无标注数据，需与标注数据同源。使用Taskflow UIE对无监督数据进行预测。

- **Step 3**: 使用标注数据以及步骤2得到的合成数据训练出封闭域Student Model。

## UIE Finetune

参考[UIE关系抽取微调](../README.md)完成模型微调，得到``../checkpoint/model_best``。

## 离线蒸馏

#### 通过训练好的UIE定制模型预测无监督数据的标签

```shell
python data_distill.py \
    --data_path ../data \
    --save_dir student_data \
    --task_type relation_extraction \
    --synthetic_ratio 10 \
    --model_path ../checkpoint/model_best
```

**NOTE**：schema需要根据标注数据在`data_distill.py`中进行配置，且schema需要包含标注数据中的所有标签类型。

可配置参数说明：

- `data_path`: 标注数据（`doccano_ext.json`）及无监督文本（`unlabeled_data.txt`）路径。
- `model_path`: 训练好的UIE定制模型路径。
- `save_dir`: 学生模型训练数据保存路径。
- `synthetic_ratio`: 控制合成数据的比例。最大合成数据数量=synthetic_ratio*标注数据数量。
- `platform`: 标注数据的所使用的标注平台，可选有`doccano`，`label_studio`，默认为`label_studio`。
- `task_type`: 选择任务类型，可选有`entity_extraction`，`relation_extraction`，`event_extraction`和`opinion_extraction`。因为是封闭域抽取，不同任务的后处理逻辑不同，因此需指定任务类型。
- `seed`: 随机种子，默认为1000。

#### 老师模型评估

UIE微调阶段针对UIE训练格式数据评估模型效果（该评估方式非端到端评估，非关系抽取或事件抽取的标准评估方式），可通过以下评估脚本进行端到端评估。

```shell
python evaluate_teacher.py \
    --task_type relation_extraction \
    --test_path ./student_data/dev_data.json \
    --label_maps_path ./student_data/label_maps.json \
    --model_path ../checkpoint/model_best
```

可配置参数说明：

- `model_path`: 训练好的UIE定制模型路径。
- `test_path`: 测试数据集路径。
- `label_maps_path`: 学生模型标签字典。
- `batch_size`: 批处理大小，默认为8。
- `max_seq_len`: 最大文本长度，默认为256。
- `task_type`: 选择任务类型，可选有`entity_extraction`，`relation_extraction`，`event_extraction`和`opinion_extraction`。因为是封闭域信息抽取的评估，需指定任务类型。


#### 学生模型训练

```shell
python train.py \
    --task_type relation_extraction \
    --train_path student_data/train_data.json \
    --dev_path student_data/dev_data.json \
    --label_maps_path student_data/label_maps.json \
    --num_epochs 50 \
    --encoder ernie-3.0-mini-zh
```

可配置参数说明：

- `train_path`: 训练集文件路径。
- `dev_path`: 验证集文件路径。
- `batch_size`: 批处理大小，默认为16。
- `learning_rate`: 学习率，默认为3e-5。
- `save_dir`: 模型存储路径，默认为`./checkpoint`。
- `max_seq_len`: 最大文本长度，默认为256。
- `weight_decay`: 表示AdamW优化器中使用的 weight_decay 的系数。
- `warmup_proportion`: 学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
- `num_epochs`: 训练轮数，默认为100。
- `seed`: 随机种子，默认为1000。
- `encoder`: 选择学生模型的模型底座，默认为`ernie-3.0-mini-zh`。
- `task_type`: 选择任务类型，可选有`entity_extraction`，`relation_extraction`，`event_extraction`和`opinion_extraction`。因为是封闭域信息抽取，需指定任务类型。
- `logging_steps`: 日志打印的间隔steps数，默认10。
- `eval_steps`: evaluate的间隔steps数，默认200。
- `device`: 选用什么设备进行训练，可选cpu或gpu。
- `init_from_ckpt`: 可选，模型参数路径，热启动模型训练；默认为None。

#### 学生模型评估

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path student_data/dev_data.json \
    --task_type relation_extraction \
    --label_maps_path student_data/label_maps.json \
    --encoder ernie-3.0-mini-zh
```

可配置参数说明：

- `model_path`: 训练好的UIE定制模型路径。
- `test_path`: 测试数据集路径。
- `label_maps_path`: 学生模型标签字典。
- `batch_size`: 批处理大小，默认为8。
- `max_seq_len`: 最大文本长度，默认为256。
- `encoder`: 选择学生模型的模型底座，默认为`ernie-3.0-mini-zh`。
- `task_type`: 选择任务类型，可选有`entity_extraction`，`relation_extraction`，`event_extraction`和`opinion_extraction`。因为是封闭域信息抽取的评估，需指定任务类型。

## Taskflow部署学生模型

- 通过Taskflow一键部署封闭域信息抽取模型，`task_path`为学生模型路径。

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow

>>> my_ie = Taskflow("information_extraction", model="uie-data-distill-gp", task_path="checkpoint/model_best/") # Schema is fixed in closed-domain information extraction
>>> pprint(my_ie("威尔哥（Virgo）减速炸弹是由瑞典FFV军械公司专门为瑞典皇家空军的攻击机实施低空高速轰炸而研制，1956年开始研制，1963年进入服役，装备于A32“矛盾”、A35“龙”、和AJ134“雷”攻击机，主要用于攻击登陆艇、停放的飞机、高炮、野战火炮、轻型防护装甲车辆以及有生力量。"))
[{'武器名称': [{'end': 14,
            'probability': 0.9976037,
            'relations': {'产国': [{'end': 18,
                                  'probability': 0.9988706,
                                  'relations': {},
                                  'start': 16,
                                  'text': '瑞典'}],
                          '研发单位': [{'end': 25,
                                    'probability': 0.9978277,
                                    'relations': {},
                                    'start': 18,
                                    'text': 'FFV军械公司'}],
                          '类型': [{'end': 14,
                                  'probability': 0.99837446,
                                  'relations': {},
                                  'start': 12,
                                  'text': '炸弹'}]},
            'start': 0,
            'text': '威尔哥（Virgo）减速炸弹'}]}]
```


# References

- **[GlobalPointer](https://kexue.fm/search/globalpointer/)**

- **[GPLinker](https://kexue.fm/archives/8888)**

- **[JunnYu/GPLinker_pytorch](https://github.com/JunnYu/GPLinker_pytorch)**

- **[CBLUE](https://github.com/CBLUEbenchmark/CBLUE)**
