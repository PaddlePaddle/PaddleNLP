### PaddleNLP之FAQ文档

#### 1、自定义数据集

参照官网文档：[https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)

#### 2、如何加载训练好的模型

+ 通过Paddle原生的[save接口](#save)保存

  model.save_pretrained('./checkpoint')

  tokenizer.save_pretrained('./checkpoint')

+ 通过Paddle原生的[load接口](#load)来加载

#### 3、如何保存各epoch指标

paddle.Model.fit在eval阶段会打印eval data的评价指标。同时paddle.Model.fit可以指定save_freq参数，间隔epoch数之后保存模型参数。

可参考：[ [https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/hapi/model/Model_cn.html#model](#model)]( [https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/hapi/model/Model_cn.html#model](#model))

#### 4、PaddleServing示例

训练模型转成推理模型可以参考这里：

[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert#预测 ](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert#预测 )

Paddle Serving等更多预测相关内容可以先参考这里：

[https://github.com/PaddlePaddle/PaddleNLP/pull/164](https://github.com/PaddlePaddle/PaddleNLP/pull/164)

#### 5、Paddle Inference预测示例

Paddle Inference预测示例：[https://github.com/PaddlePaddle/PaddleNLP/pull/281](https://github.com/PaddlePaddle/PaddleNLP/pull/281) 

参考项目：[https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification) 

#### 6、如何复现相同结果

确保训练模式下排除一些随机性参数条件，例如dropout的mask位置

#### 7、在conda11安装和使用PaddlNLP

以下方案二选一即可，其他安装请参考:

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

  将代码保存为env.yaml, 注意把最后一行prefix改为你自己的conda或miniconda中paddle的对应路径，然后运行 conda env create -f env.yaml

具体请参考：[https://github.com/PaddlePaddle/PaddleNLP/issues/348](https://github.com/PaddlePaddle/PaddleNLP/issues/348) 