# ERNIE 3.0 的 RocketQA 模型


## 模型介绍

本次开源的模型 ERNIE 3.0系列的模型的基础上，使用 RocketQA 的训练策略训练的`DualEncoder`和`CrossEncoder`模型，相比于原始的 RocektQA 模型，在中文上的效果达到最佳。

## 模型效果

### DualEncoder 模型效果

|                               模型                               |       模型规模       |   MRR@10   |  Recall@1  | Recall@50  |
|:----------------------------------------------------------------:|:--------------------:|:----------:|:----------:|:----------:|
|                        rocketqa-baselines                        | 12-layer, 768-hidden |   56.45%   |   45.05%   |   91.40%   |
|   rocketqa-zh-base-query-encoder&rocketqa-zh-base-para-encoder   | 12-layer, 768-hidden | **59.14%** | **48.00%** | **92.15%** |
| rocketqa-zh-medium-query-encoder&rocketqa-zh-medium-para-encoder | 6-layer, 768-hidden  |   53.92%   |   42.35%   |   89.75%   |
|   rocketqa-zh-mini-query-encoder&rocketqa-zh-mini-para-encoder   | 6-layer, 384-hidden  |   44.97%   |   34.25%   |   84.97%   |
|  rocketqa-zh-micro-query-encoder&rocketqa-zh-micro-para-encoder  | 4-layer, 384-hidden  |   40.22%   |   28.70%   |   83.40%   |
|   rocketqa-zh-nano-query-encoder&rocketqa-zh-nano-para-encoder   | 4-layer, 312-hidden  |   38.07%   |   27.35%   |   80.35%   |


### CrossEncoder 模型效果

|             模型              |       模型规模       |   MRR@10   |  Recall@1  | Recall@50  |
|:-----------------------------:|:--------------------:|:----------:|:----------:|:----------:|
|      rocketqa-baselines       | 12-layer, 768-hidden |   65.62%   |   55.50%   |   91.75%   |
|  rocketqa-base-cross-encoder  | 12-layer, 768-hidden | **76.64%** | **70.05%** | **91.75%** |
| rocketqa-medium-cross-encoder | 6-layer, 768-hidden  |   74.82%   |   67.30%   |   91.75%   |
|  rocketqa-mini-cross-encoder  | 6-layer, 384-hidden  |   70.25%   |   60.85%   |   91.75%   |
| rocketqa-micro-cross-encoder  | 4-layer, 384-hidden  |   68.80%   |   59.35%   |   91.75%   |
|  rocketqa-nano-cross-encoder  | 4-layer, 312-hidden  |   67.99%   |   58.25%   |   91.75%   |


## 模型性能

RocketQA 系列的模型在 GPU 上能够达到 ms 级别的速度，一个 query 大概20ms 左右，最快能够达到10ms，另外，为了验证在 CPU 上的速度，我们使用 RocketQA 系列的模型在 CPU 上测试了构建索引的时间，数据集是1398条，在 cpu 上的测试时间，以下 cpu 上的查询时间指的是在后台 query 发一次请求经过模型处理后得到结果的时间，包含召回和排序两部分，召回的文本数为30条。

|        模型        |       模型规模       | 构建索引时间 | 查询一条文本时间 |
|:------------------:|:--------------------:|:------------:|:----------------:|
| rocketqa-baselines | 12-layer, 768-hidden |   15min46s   |   10.04s/30条    |
|  rocketqa-zh-base  | 12-layer, 768-hidden |   15min54s   |   11.50s/30条    |
| rocketqa-zh-medium | 6-layer, 768-hidden  |   7min54s    |    5.39s/30条    |
|  rocketqa-zh-mini  | 6-layer, 384-hidden  |   4min10s    |    4.02s/30条    |
| rocketqa-zh-micro  | 4-layer, 384-hidden  |   2min49s    |    3.0s/30条     |
|  rocketqa-zh-nano  | 4-layer, 312-hidden  |   2min30s    |    2.84s/30条    |

注意在测试速度的时候会有一些浮动，有很多因素会影响测试速度，比如 `elastic search`本身的性能，检索的文本数目，但总体的结论不会变。

## 模型评测
- 数据集选择，使用 [T2Ranking](https://github.com/THUIR/T2Ranking/tree/main) 数据集，由于 T2Ranking 的数据集太大，评测起来的时间成本有些高，所以我们只选择了 T2Ranking 中的前 10000 篇文章
- 评测方式，使用 MTEB 的方式进行评测，报告 map@1, map@10, mrr@1, mrr@10, ndcg@1, ndcg@10
1. 安装依赖
```bash
pip install -r requirements.txt
```
2. 运行评测脚本
```bash
python retrieval_benchmarks.py
```
