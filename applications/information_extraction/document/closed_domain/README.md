# 封闭域文档抽取

在UIE强大的抽取能力背后，同样需要较大的算力支持计算。在一些工业应用场景中对性能的要求较高，若不能有效压缩则无法实际应用。因此，我们基于数据蒸馏技术构建了UIE Slim数据蒸馏系统。其原理是通过数据作为桥梁，将UIE模型的知识迁移到封闭域信息抽取小模型，以达到精度损失较小的情况下却能达到大幅度预测速度提升的效果。

#### UIE数据蒸馏三步

- **Step 1**: 使用UIE模型对标注数据进行finetune，得到Teacher Model。

- **Step 2**: 用户提供大规模无标注数据，需与标注数据同源。使用Taskflow UIE对无监督数据进行预测。

- **Step 3**: 使用标注数据以及步骤2得到的合成数据训练出封闭域Student Model。

## UIE Finetune

参考[UIE-X文档抽取微调示例](../README.md)完成模型微调，得到``../checkpoint/model_best``。

## 离线蒸馏

#### 通过训练好的UIE定制模型预测无监督数据的标签

```shell
python data_distill.py \
    --data_path ../data \
    --save_dir data \
    --synthetic_ratio 10 \
    --model_path ../checkpoint/model_best
```

**NOTE**：schema需要根据标注数据在`data_distill.py`中进行配置，且schema需要包含标注数据中的所有标签类型。

可配置参数说明：

- `data_path`: 标注数据（`label_studio.json`）及无监督数据路径。
- `model_path`: 训练好的UIE定制模型路径。
- `save_dir`: 学生模型训练数据保存路径。
- `synthetic_ratio`: 控制合成数据的比例。最大合成数据数量=synthetic_ratio*标注数据数量。
- `seed`: 随机种子，默认为1000。


#### 学生模型训练

```shell
python train.py \
    --train_path data/train_data.json \
    --dev_path data/dev_data.json \
    --label_maps_path data/label_maps.json \
    --num_epochs 50 \
    --model_name_or_path ernie-layoutx-base-uncased
```

可配置参数说明：

- `train_path`: 训练集文件路径。
- `dev_path`: 验证集文件路径。
- `batch_size`: 批处理大小，默认为16。
- `learning_rate`: 学习率，默认为3e-5。
- `save_dir`: 模型存储路径，默认为`./checkpoint`。
- `max_seq_len`: 最大文本长度，默认为512。
- `weight_decay`: 表示AdamW优化器中使用的 weight_decay 的系数。
- `warmup_proportion`: 学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
- `num_epochs`: 训练轮数，默认为100。
- `seed`: 随机种子，默认为1000。
- `model_name_or_path`: 选择封闭域模型的编码器，默认为`ernie-layoutx-base-uncased`。
- `logging_steps`: 日志打印的间隔steps数，默认10。
- `eval_steps`: evaluate的间隔steps数，默认200。
- `device`: 选用什么设备进行训练，可选cpu或gpu。
- `init_from_ckpt`: 可选，模型参数路径，热启动模型训练；默认为None。

#### 封闭域模型评估

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path data/dev_data.json
```

可配置参数说明：

- `model_path`: 训练好的UIE定制模型路径。
- `test_path`: 测试数据集路径。
- `label_maps_path`: 学生模型标签字典。
- `batch_size`: 批处理大小，默认为8。
- `max_seq_len`: 最大文本长度，默认为512。
- `encoder`: 选择学生模型的模型底座，默认为`ernie-3.0-mini-zh`。

## Taskflow封闭域模型一键预测

`paddlenlp.Taskflow`装载定制模型，通过`task_path`指定模型权重文件的路径，路径下需要包含训练好的模型权重文件`model_state.pdparams`。

```python
from pprint import pprint
from paddlenlp import Taskflow
from paddlenlp.utils.doc_parser import DocParser

my_ie = Taskflow("information_extraction", model="global-pointer", task_path="checkpoint/model_best/") # Schema is fixed in closed-domain information extraction
```

- 对指定的doc_path文档进行信息抽取并进行可视化：

```python
doc_path = "../data/images/b199.jpg"
results = my_ie({"doc": doc_path})
pprint(results)

# 结果可视化
DocParser.write_image_with_results(
    doc_path,
    result=results[0],
    save_path="./image_show.png")
```

# References

- **[GlobalPointer](https://kexue.fm/search/globalpointer/)**

- **[GPLinker](https://kexue.fm/archives/8888)**

- **[JunnYu/GPLinker_pytorch](https://github.com/JunnYu/GPLinker_pytorch)**

- **[CBLUE](https://github.com/CBLUEbenchmark/CBLUE)**
