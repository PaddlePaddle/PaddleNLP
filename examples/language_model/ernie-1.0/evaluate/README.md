# 预训练效果测试

为了方便用户使用，训练好权重在已有任务上测试效果。
本示例提供了以下

### 添加自定义权重
这里有两种方案:

1. 添加模型config。
2. 修改，已有的权重网址，快速验证
```
vim ../../../../paddlenlp/transformers/ernie/modeling.py +176
# https://bj.bcebos.com/paddlenlp/models/transformers/ernie/ernie_v1_chn_base.pdparams
# -> https://path/of/your/pdparams
```

### 对自定义权重进行评估
假如我们使用方案2进行权重地址替换，在本目录下直接运行`run.sh`, 即可对预训练参数进行评估：
```
bash -x run.sh
```

运行结束后，即可在当前目录下查看运行日志。如：
```
log_ernie
├── chnsenticorp
│   ├── endpoints.log
│   └── workerlog.0
├── cmrc
│   ├── endpoints.log
│   └── workerlog.0
├── peoples_daily_ner
│   ├── endpoints.log
│   └── workerlog.0
└── xnli
    ├── endpoints.log
    └── workerlog.0
```
查看相应的日志，即可获取finetune结果。
