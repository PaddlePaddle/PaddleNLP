# 使用方法

首先运行下面的 python 代码，生成`train_align_torch.npy`和`train_align_paddle.npy`文件。

```python
# 运行生成paddle结果
cd bert_paddle/
sh train.sh
# 运行生成torch结果
cd bert_torch/
sh train.sh
```

然后运行下面的代码，运行训练脚本；之后使用`check_step5.py`进行精度 diff 验证。

```shell
# 对比生成log
python check_step5.py
```

这里需要注意的是，由于是精度对齐，SST-2数据集的精度 diff 在0.15%以内时，可以认为对齐，因此将`diff_threshold`参数修改为了`0.0015`。

```
[2021/11/17 22:41:12] root INFO: acc:
[2021/11/17 22:41:12] root INFO:     mean diff: check passed: True, value: 0.0011467889908256534
[2021/11/17 22:41:12] root INFO: diff check passed
```

最终 diff 为`0.00114`，小于阈值标准，检查通过。
