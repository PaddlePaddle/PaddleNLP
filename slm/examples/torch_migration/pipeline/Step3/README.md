# 使用方法

## 代码解析

以 PaddlePaddle 为例，下面为定义模型、计算 loss 并保存的代码。

```python
# paddle_loss.py
if __name__ == "__main__":
    paddle.set_device("cpu")

    # def logger
    reprod_logger = ReprodLogger()

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_classes=2)
    classifier_weights = paddle.load("../classifier_weights/paddle_classifier_weights.bin")
    model.load_dict(classifier_weights)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # read or gen fake data
    fake_data = np.load("../fake_data/fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)

    fake_label = np.load("../fake_data/fake_label.npy")
    fake_label = paddle.to_tensor(fake_label)

    # forward
    out = model(fake_data)

    loss = criterion(out, fake_label)
    #
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_paddle.npy")

```

记录 loss 并保存在`loss_paddle.npy`文件中。


## 操作步骤

* 具体操作步骤如下所示。


```shell
# 生成paddle的前向loss结果
python paddle_loss.py

# 生成torch的前向loss结果
python torch_loss.py

# 对比生成log
python check_step3.py
```

`check_step3.py`的输出结果如下所示，同时也会保存在`loss_diff.log`文件中。

```
[2021/11/17 21:27:35] root INFO: loss:
[2021/11/17 21:27:35] root INFO:     mean diff: check passed: True, value: 5.960464477539063e-08
[2021/11/17 21:27:35] root INFO: diff check passed

```

diff 为5.96e-8，check 通过。
