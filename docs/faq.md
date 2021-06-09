## 常见问题

#### 如遇问题请首先查看PaddleNLP官方文档：[https://paddlenlp.readthedocs.io/zh/latest/index.html](https://paddlenlp.readthedocs.io/zh/latest/index.html) 

#### 1、如何自定义数据集

参照官网文档：[https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)

#### 2、如何加载训练好的模型

在训练中通过以下方式保存下来的模型

```
model.save_pretrained('./checkpoint')
tokenizer.save_pretrained('./checkpoint')
```

通过下面代码来加载已保存的模型

```
model.from_pretrained('./checkpoint')
tokenizer.from_pretrained('./checkpoint')
```

#### 3、如何保存各epoch指标

paddle.Model.fit在验证阶段会打印eval data的评价指标。同时也可使用paddle.Model.fit指定save_freq参数，间隔一定的epoch数保存模型参数。

具体可参考：[https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/hapi/model/Model_cn.html#model]( https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/hapi/model/Model_cn.html#model) 

#### 4、如何复现相同结果

在验证、测试过程中常常出现的结果不一致情况主要有以下可能因素：

+ 如果是在预训练模型的微调阶段首先查看是否导入fine-tune模型，导入参数后，线性层在预测时就不会随机初始化，预测结果就是唯一的。
+ 确保验证模式下排除一些随机性参数条件，例如dropout等随机因素。

#### 5、如何在conda11安装和使用PaddlNLP

针对在conda11上安装以下方案二选一即可，其他安装请参考:

[https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html)

+ 方案一

```
conda create -n pd python=3.7 -y
conda activate pd
conda install cudatoolkit=11.0 -y
python -m pip install paddlepaddle-gpu==2.0.2.post110 -f https://paddlepaddle.org.cn/whl/mkl/stable.html
pip install --upgrade paddlenlp -i https://pypi.org/simple
```

+ 方案二

  将代码保存为env.yaml, 注意把最后一行prefix改为自己的conda或miniconda中paddle的对应路径，然后运行 conda env create -f env.yaml

具体请参考：[https://github.com/PaddlePaddle/PaddleNLP/issues/348](https://github.com/PaddlePaddle/PaddleNLP/issues/348) 

#### 6、Paddle Serving示例

训练模型转成推理模型可以参考这里：

[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert#预测 ](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert#预测 )

Paddle Serving等更多预测相关内容可以先参考这里：

[https://github.com/PaddlePaddle/PaddleNLP/pull/164](https://github.com/PaddlePaddle/PaddleNLP/pull/164)

#### 7、Paddle Inference预测示例

Paddle Inference预测示例：[https://github.com/PaddlePaddle/PaddleNLP/pull/281](https://github.com/PaddlePaddle/PaddleNLP/pull/281) 

参考项目：[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification) 



