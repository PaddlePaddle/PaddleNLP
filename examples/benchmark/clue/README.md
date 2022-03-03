# CLUE Benchmark

[CLUE](https://www.cluebenchmarks.com/)自成立以来发布了多项NLP评测基准，包括分类榜单，阅读理解榜单和自然语言推断榜单等，在学术界、工业界产生了深远影响。
是目前应用最广泛的中文语言测评指标之一，被包括阿里巴巴达摩院、腾讯 AI 实验室、华为诺亚方舟实验室在内的 20 多家国内语言实验室所采纳。学术引用100+，github star超6000+。

本项目是 CLUE评测任务 在 Paddle 2.0上的开源实现。

## 数据集

数据集下载地址为：[https://github.com/CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE) 

paddlenlp已经内置支持所有的分类任务数据集，阅读理解任务的数据集需要单独下载。

## 分类任务

修改 bash 脚本来指定任务，然后训练：

```
sh train.sh
```

预测：

```
sh predict.sh
```

## 阅读理解任务

修改 bash 脚本来指定任务，然后训练

```
sh train.sh
```

预测：

```
sh predict.sh
```