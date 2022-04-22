# ERNIE 3.0 系列模型

## ERNIE 3.0 系列模型简介


## ERNIE 3.0 系列模型在 CLUE Benchark 上的精度

ERNIE 3.0 系列模型在[CLUE Benchmark](../../benchmark/clue/)上的效果如下所示：

| 参数量           | AVG      | AFQMC | TNEWS | IFLYTEK   | CMNLI     | OCNLI     | CLUEWSC2020 | CSL       | CMRC2018    | CHID      | C<sup>3</sup>   |
| ---------------- | -------- | ----- | ----- | --------- | --------- | --------- | ----------- | --------- | ----------- | --------- | --------------- | ------ | ----- |
| ERNIE 3.0-Large  | 24L1024H |       | 78.71 | **77.36** | **60.21** | **62.75** | **85.06**   | **82.14** | **91.12**   | **84.23** | **72.76/91.87** | **86.84** | **84.67** |
| ERNIE 3.0-Base   | 12L768H  |       |       | 76.53     | 58.73     | 60.72     | 80.31       | 83.65     | TODO | 83.3      | 70.52/90.72     | >84.08 | 78.07 |
| ERNIE 3.0-Medium | 6L768H   |       | 72.86 | 75.35     | 57.45     | 60.18     | 81.16       | 77.19     | 79.28       | 81.93     | 65.83/87.29     | 80.00  | 70.36 |


ERNIE 3.0 系列预训练模型 ERNIE 3.0-Large、ERNIE 3.0-Base、ERNIE 3.0-Medium 模型已在 [PaddleNLP](../../../paddlenlp/transformers/ernie/modeling.py) 开源，可参考下方示例代码调用，可以使用[CLUE Benchmark](../../benchmark/clue/)的代码复现上表精度，也可以将模型放到实际任务中去微调使用。


```python
from paddlenlp.transformers import AutoModelForSequenceClassification
from paddlenlp.transformers import AutoModelForQuestionAnswering
from paddlenlp.transformers import AutoModelForMultipleChoice

model_large = AutoModelForSequenceClassification.from_pretrained("ernie-3.0-large")
model_base = AutoModelForQuestionAnswering.from_pretrained("ernie-3.0-base")
model_medium = AutoModelForMultipleChoice.from_pretrained("ernie-3.0-tiny")
```

## 对 ERNIE 3.0-Medium 模型进一步压缩


## 压缩后的模型性能

GPU
CPU

# Reference
