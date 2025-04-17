# 使用方法

## 数据集和数据加载对齐步骤

* 使用下面的命令，判断数据预处理以及数据集是否构建正确。

```shell
python test_data.py
```

显示出以下内容，Dataset 以及 Dataloader 的长度和内容 diff 均满足小于指定阈值，可以认为复现成功。

```
[2021/11/17 20:57:06] root INFO: length:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_0_input_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_0_token_type_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_0_labels:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_1_input_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_1_token_type_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_1_labels:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_2_input_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_2_token_type_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_2_labels:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_3_input_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_3_token_type_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_3_labels:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_4_input_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_4_token_type_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataset_4_labels:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_0_input_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_0_token_type_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_0_labels:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_1_input_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_1_token_type_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_1_labels:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_2_input_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_2_token_type_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_2_labels:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_3_input_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_3_token_type_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_3_labels:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_4_input_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_4_token_type_ids:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: dataloader_4_labels:
[2021/11/17 20:57:06] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 20:57:06] root INFO: diff check passed
```


## 数据评估对齐流程

### 评估代码和修改内容说明

Pytorch 准确率评估指标使用的是 huggingface 的 datasets 库。

```python
import torch
import numpy as np
from datasets import load_metric
hf_metric = load_metric("accuracy.py")
logits = np.random.normal(0, 1, size=(64, 2)).astype("float32")
labels = np.random.randint(0, 2, size=(64,)).astype("int64")
hf_metric.add_batch(predictions=torch.from_numpy(logits).argmax(dim=-1), references=torch.from_numpy(labels))
hf_accuracy = hf_metric.compute()["accuracy"]
print(hf_accuracy)
```

对应地，PaddlePaddle 评估指标代码如下

```python
import paddle
import numpy as np
from paddle.metric import Accuracy
pd_metric = Accuracy()
pd_metric.reset()
logits = np.random.normal(0, 1, size=(64, 2)).astype("float32")
labels = np.random.randint(0, 2, size=(64,)).astype("int64")
correct = pd_metric.compute(paddle.to_tensor(logits), paddle.to_tensor(labels))
pd_metric.update(correct)
pd_accuracy = pd_metric.accumulate()
print(pd_accuracy)
```

### 操作步骤

运行下面的命令，验证数据集评估是否正常。

```shell
# 生成paddle和pytorch指标
python test_metric.py
# 对比生成log
python check_step2.py
```

最终结果输出如下，accuracy 精度 diff 为0，小于阈值，结果前向验证，
```
[2021/11/17 21:15:05] root INFO: accuracy:
[2021/11/17 21:15:05] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:15:05] root INFO: diff check passed

```
