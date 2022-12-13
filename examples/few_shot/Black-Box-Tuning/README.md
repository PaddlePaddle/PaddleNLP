# EFL


[Black-Box Tuning for Language-Model-as-a-Service](https://arxiv.org/abs/2201.03514)


## 算法简介

这是一种黑盒调优框架，通过无导数优化对输入文本前面增加的连续提示语进行优化，当标记样本数量较少时可以取得较好效果。

## 代码结构及说明
```
|—— train.py # 策略的训练、评估主脚本
|—— data.py # 加载本地数据集脚本，以tnews为例
|—— model.py # BBT 模型
|—— utils.py # 一些配置变量
|—— run.sh # 一键训练及评估脚本

```

## References
[1] Sun T ,  Shao Y ,  Qian H , et al. Black-Box Tuning for Language-Model-as-a-Service[J].  2022.
