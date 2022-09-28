#### 基于静态图的预测
导出静态图模型之后，可以基于静态图做预测部署，`deploy/python/predict.py` 脚本提供了 python 静态图预测示例。以 tnews 数据集为例, 执行如下命令基于静态图预测：
```
python deploy/python/predict.py --model_dir=./export --task_name="tnews"

```
