# 使用TCN序列模型解决时间序列预测

## 简介

时间序列是指按照时间先后顺序排列而成的序列，例如每日发电量、每小时营业额等组成的序列。通过分析时间序列中的发展过程、方向和趋势，我们可以预测下一段时间可能出现的情况。在本例中，我们使用时间卷积网络TCN进行建模，将学习到的特征接入全连接层完成预测。TCN的网络如下所示：<br />

![TCN](http://paddlenlp.bj.bcebos.com/imgs/tcn.png)

图中是一个filters number=3, dilated rate=1的时间卷积网络，它能够学习前T个时序的数据特征。关于TCN更详细的资料请参考论文：[An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)。

## 快速开始

2019年末，新冠疫情席卷而来，影响了我们工作、生活中的方方面面。如今，疫情在国内逐渐得到控制，但在国际上依然呈现急剧扩增的趋势，预测今后的疫情形势对我们的规划实施具有重大的指导意义。在本例中，我们关注时下还在发展进行的新冠疫情，将病例数作为时序预测对象。

### 数据准备

数据集由约翰·霍普金斯大学系统科学与工程中心提供，每日最新数据可以从 [COVID-19](https://github.com/CSSEGISandData/COVID-19) 仓库中获取，我们在本例中提供了2020年11月24日下载的病例数据。如您需要使用最新数据，请运行：

```
wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv

### 模型训练

模型训练支持 CPU 和 GPU，使用 GPU 之前应指定使用的显卡卡号：

```bash
export CUDA_VISIBLE_DEVICES=0 # 只支持单卡训练
```

训练启动方式如下：

```bash
python train.py --data_path time_series_covid19_confirmed_global.csv \
                --epochs 10 \
                --batch_size 32 \
                --use_gpu
```

### 模型预测

预测启动方式如下：

```bash
python predict.py --data_path time_series_covid19_confirmed_global.csv \
                  --use_gpu
```


## 线上教程体验

我们为时间序列预测任务提供了线上教程，欢迎体验：

* [使用TCN网络完成新冠疫情病例数预测](https://aistudio.baidu.com/aistudio/projectdetail/1290873)
