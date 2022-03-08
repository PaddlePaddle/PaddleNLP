# CLUE Benchmark

[CLUE](https://www.cluebenchmarks.com/)自成立以来发布了多项NLP评测基准，包括分类榜单，阅读理解榜单和自然语言推断榜单等，在学术界、工业界产生了深远影响。
是目前应用最广泛的中文语言测评指标之一，被包括阿里巴巴达摩院、腾讯 AI 实验室、华为诺亚方舟实验室在内的 20 多家国内语言实验室所采纳。学术引用100+，github star超6000+。

本项目是 CLUE评测任务 在 Paddle 2.0上的开源实现。

## 数据集

数据集下载地址为：[https://github.com/CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE)

paddlenlp已经内置支持所有的分类任务数据集，阅读理解任务的数据集需要单独下载。

## 测试结果

### 分类任务

| 模型   | Score  | AFQMC  | TNEWS'  | IFLYTEK'   | CMNLI   | CLUEWSC2020 | CSL  |
| :----:| :----: | :----: |:----: |:----: |:----: |:----: |:----: |
| BERT-base     | 68.77 |  73.70 | 56.58  | 60.29 | 79.69 |  62.0 | 80.36 |
| ERNIE-Gram     | 71.226|  73.45| 55.91  | 58.81 | 78.2 |  76.21 | 83.03 |

#### 阅读理解任务

| 模型 | Score  | CMRC2018 | CHID | C<sup>3</sup> |
| :----:| :----:  | :----: |:----: |:----: |
| BERT-base | 72.71 | 71.60 | 82.04 | 64.50 |
| ERNIE-Gram	| - | - | - | - |

## 实验

### 分类任务

修改 bash 脚本来指定任务，然后训练：

```
sh train.sh
```

预测：

```
sh predict.sh
```

### 阅读理解任务

修改 bash 脚本来指定任务，然后训练

```
sh train.sh
```

预测：

```
sh predict.sh
```

### 提交流程

提交流程只需要把 json 文件压缩程 zip 文件，上传到CLUE的网站就行了，这里以 chid10_predict.json 为例：

```
zip submit.zip cmrc2018_predict.json
```
然后把压缩后的 submit.zip 上传到 [CLUE网站](https://www.cluebenchmarks.com/index.html)
