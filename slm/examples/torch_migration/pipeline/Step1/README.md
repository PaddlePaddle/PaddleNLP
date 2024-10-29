# 使用方法


本部分内容以前向对齐为例，介绍基于`repord_log`工具对齐的检查流程。其中与`reprod_log`工具有关的部分都是需要开发者需要添加的部分。


```shell
# 进入文件夹并生成torch的bert模型权重
cd pipeline/weights/ && python torch_bert_weights.py
# 进入文件夹并将torch的bert模型权重转换为paddle
cd pipeline/weights/ && python torch2paddle.py
# 进入文件夹并生成classifier权重
cd pipeline/classifier_weights/ && python generate_classifier_weights.py
# 进入Step1文件夹
cd pipeline/Step1/
# 生成paddle的前向数据
python pd_forward_bert.py
# 生成torch的前向数据
python pt_forward_bert.py
# 对比生成log
python check_step1.py
```

具体地，以 PaddlePaddle 为例，`pd_forward_bert.py`的具体代码如下所示。

```python
import numpy as np
import paddle
from reprod_log import ReprodLogger
import sys
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 1)[0]
sys.path.append(config_path)
from models.pd_bert import *

# 导入reprod_log中的ReprodLogger类
from reprod_log import ReprodLogger

reprod_logger = ReprodLogger()

# 组网初始化加载BertModel权重
paddle_dump_path = '../weights/paddle_weight.pdparams'
config = BertConfig()
model = BertForSequenceClassification(config)
checkpoint = paddle.load(paddle_dump_path)
model.bert.load_dict(checkpoint)

# 加载分类权重
classifier_weights = paddle.load(
        "../classifier_weights/paddle_classifier_weights.bin")
model.load_dict(classifier_weights)
model.eval()
# 读入fake data并转换为tensor，这里也可以固定seed在线生成fake data
fake_data = np.load("../fake_data/fake_data.npy")
fake_data = paddle.to_tensor(fake_data)
# 模型前向
out = model(fake_data)
# 保存前向结果，对于不同的任务，需要开发者添加。
reprod_logger.add("logits", out.cpu().detach().numpy())
reprod_logger.save("forward_paddle.npy")
```

diff 检查的代码可以参考：[check_step1.py](./check_step1.py)，具体代码如下所示。

```python
# https://github.com/littletomatodonkey/AlexNet-Prod/blob/master/pipeline/Step1/check_step1.py
# 使用reprod_log排查diff
from reprod_log import ReprodDiffHelper
if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./forward_torch.npy")
    paddle_info = diff_helper.load_info("./forward_paddle.npy")
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="forward_diff.log")
```

产出日志如下，同时会将 check 的结果保存在`forward_diff.log`文件中。

```
[2021/11/17 20:15:50] root INFO: logits:
[2021/11/17 20:15:50] root INFO:     mean diff: check passed: True, value: 1.30385160446167e-07
[2021/11/17 20:15:50] root INFO: diff check passed
```

平均绝对误差为1.3e-7，测试通过。
