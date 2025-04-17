# 使用方法

### 学习率对齐验证

运行下面的命令，检查学习率模块设置是否正确。

```shell
python test_lr_scheduler.py
```

最终输出内容如下。

```
[2021/11/17 21:44:19] root INFO: step_100_linear_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:44:19] root INFO: step_300_linear_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:44:19] root INFO: step_500_linear_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:44:19] root INFO: step_700_linear_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:44:19] root INFO: step_900_linear_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:44:19] root INFO: step_100_cosine_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:44:19] root INFO: step_300_cosine_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:44:19] root INFO: step_500_cosine_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: False, value: 9.35605818719964e-06
[2021/11/17 21:44:19] root INFO: step_700_cosine_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: False, value: 1.3681476625617212e-05
[2021/11/17 21:44:19] root INFO: step_900_cosine_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: False, value: 1.8924391285779562e-05
[2021/11/17 21:44:19] root INFO: step_100_polynomial_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:44:19] root INFO: step_300_polynomial_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:44:19] root INFO: step_500_polynomial_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:44:19] root INFO: step_700_polynomial_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:44:19] root INFO: step_900_polynomial_lr:
[2021/11/17 21:44:19] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 21:44:19] root INFO: diff check failed

```

linear 和 polynomial 方式衰减的学习率 diff 为0，check 通过，cosine 方式衰减学习率可能由于计算误差未通过。


### 反向对齐操作方法

#### 代码讲解

以 PaddlePaddle 为例，训练流程核心代码如下所示。每个 iter 中输入相同的 fake data 与 fake label，计算 loss，进行梯度反传与参数更新，将 loss 批量返回，用于后续的验证。

```python
def pd_train_some_iters(model,
                     criterion,
                     optimizer,
                     fake_data,
                     fake_label,
                     max_iter=2):
    paddle_dump_path = '../weights/paddle_weight.pdparams'
    config = PDBertConfig()
    model = PDBertForSequenceClassification(config)
    checkpoint = paddle.load(paddle_dump_path)
    model.bert.load_dict(checkpoint)
    classifier_weights = paddle.load(
        "../classifier_weights/paddle_classifier_weights.bin")
    model.load_dict(classifier_weights)
    model.eval()
    criterion = paddle.nn.CrossEntropy()
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(learning_rate=3e-5, parameters=model.parameters(),
        weight_decay=1e-2,
        epsilon=1e-6,
        apply_decay_param_fun=lambda x: x in decay_params)
    loss_list = []
    for idx in range(max_iter):
        input_ids = paddle.to_tensor(fake_data)
        labels = paddle.to_tensor(fake_label)

        output = model(input_ids)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        loss_list.append(loss)
    return loss_list
```


#### 操作方法

运行下面的命令，基于 fake data 与 fake label，依次生成若干轮 loss 数据并保存，使用`reprod_log`工具进行 diff 排查。

```shell
# 生成paddle和torch的前向数据
python test_bp.py

# 对比生成log
python check_step4.py
```

最终输出结果如下，同时会保存在文件`bp_align_diff.log`中。

```
[2021/11/17 22:08:30] root INFO: loss_0:
[2021/11/17 22:08:30] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 22:08:30] root INFO: loss_1:
[2021/11/17 22:08:30] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 22:08:30] root INFO: loss_2:
[2021/11/17 22:08:30] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 22:08:30] root INFO: loss_3:
[2021/11/17 22:08:30] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 22:08:30] root INFO: loss_4:
[2021/11/17 22:08:30] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 22:08:30] root INFO: loss_5:
[2021/11/17 22:08:30] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 22:08:30] root INFO: loss_6:
[2021/11/17 22:08:30] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 22:08:30] root INFO: loss_7:
[2021/11/17 22:08:30] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 22:08:30] root INFO: loss_8:
[2021/11/17 22:08:30] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 22:08:30] root INFO: loss_9:
[2021/11/17 22:08:30] root INFO:     mean diff: check passed: True, value: 0.0
[2021/11/17 22:08:30] root INFO: diff check passed

```

前面10轮的 loss diff 均等于0，check 通过。
