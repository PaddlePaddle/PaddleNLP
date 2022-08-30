## PaddleNLP常见问题汇总（持续更新）

+ [【精选】NLP精选5问](#NLP精选)

  + [Q1.1 如何加载自己的本地数据集，以便使用PaddleNLP的功能？](#1-1)
  + [Q1.2 PaddleNLP会将内置的数据集、模型下载到默认路径，如何修改路径？](#1-2)
  + [Q1.3 PaddleNLP中如何保存、加载训练好的模型？](#1-3)
  + [Q1.4 当训练样本较少时，有什么推荐的方法能提升模型效果吗？](#1-4)
  + [Q1.5 如何提升模型的性能，提升QPS？](#1-5)

+ [【理论篇】NLP通用问题](#NLP通用问题 )

  + [Q2.1 数据类别分布不均衡， 有哪些应对方法？](#2-2)
  + [Q2.2 如果使用预训练模型，一般需要多少条样本？](#2-3)

+ [【实战篇】PaddleNLP实战问题](#PaddleNLP实战问题)

  [数据集和数据处理](#数据问题)

  + [Q3.1 使用自己的数据集训练预训练模型时，如何引入额外的词表？](#3-1)

  [模型训练调优](#训练调优问题)

  + [Q3.2 如何加载自己的预训练模型，进而使用PaddleNLP的功能？](#4-1)
  + [Q3.3 如果训练中断，需要继续热启动训练，如何保证学习率和优化器能从中断地方继续迭代？](#4-2)
  + [Q3.4 如何冻结模型梯度？](#4-3)
  + [Q3.5 如何在eval阶段打印评价指标，在各epoch保存模型参数？](#4-4)
  + [Q3.6 训练过程中，训练程序意外退出或Hang住，应该如何排查？](#4-5)

  + [Q3.7 在模型验证和测试过程中，如何保证每一次的结果是相同的？](#4-6)
  + [Q3.8 ERNIE模型如何返回中间层的输出？](#4-7)

  [预测部署](#部署问题)

  + [Q3.9 PaddleNLP训练好的模型如何部署到服务器 ？](#5-1)
  + [Q3.10 静态图模型如何转换成动态图模型？](#5-2)

+ [特定模型和应用场景咨询](#NLP应用场景)
  + [Q4.1 【词法分析】LAC模型，如何自定义标签label，并继续训练？](#6-1)
  + [Q4.2 信息抽取任务中，是否推荐使用预训练模型+CRF，怎么实现呢？](#6-2)
  + [Q4.3 【阅读理解】`MapDatasets`的`map()`方法中对应的`batched=True`怎么理解，在阅读理解任务中为什么必须把参数`batched`设置为`True`？](#6-3)
  + [Q4.4 【语义匹配】语义索引和语义匹配有什么区别？](#6-4)
  + [Q4.5 【解语】wordtag模型如何自定义添加命名实体及对应词类?](#6-5)

+ [其他使用咨询](#使用咨询问题)
  + [Q5.1 在CUDA11使用PaddlNLP报错?](#7-1)
  + [Q5.2 如何设置parameter？](#7-2)
  + [Q5.3 GPU版的Paddle虽然能在CPU上运行，但是必须要有GPU设备吗？](#7-3)
  + [Q5.4  如何指定用CPU还是GPU训练模型？](#7-4)
  + [Q5.5 动态图模型和静态图模型的预测结果一致吗？](#7-5)
  + [Q5.6 如何可视化acc、loss曲线图、模型网络结构图等？](#7-6)

<a name="NLP精选"></a>

## ⭐️【精选】NLP精选5问

<a name="1-1"></a>

##### Q1.1 如何加载自己的本地数据集，以便使用PaddleNLP的功能？

**A:** 通过使用PaddleNLP提供的 `load_dataset`，  `MapDataset` 和 `IterDataset` ，可以方便的自定义属于自己的数据集哦，也欢迎您贡献数据集到PaddleNLP repo。

从本地文件创建数据集时，我们 **推荐** 根据本地数据集的格式给出读取function并传入 `load_dataset()` 中创建数据集。
以[waybill_ie](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/waybill_ie)快递单信息抽取任务中的数据为例：

```python
from paddlenlp.datasets import load_dataset

def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        # 跳过列名
        next(f)
        for line in f:
            words, labels = line.strip('\n').split('\t')
            words = words.split('\002')
            labels = labels.split('\002')
            yield {'tokens': words, 'labels': labels}

# data_path为read()方法的参数
map_ds = load_dataset(read, data_path='train.txt', lazy=False)
iter_ds = load_dataset(read, data_path='train.txt', lazy=True)
```

如果您习惯使用`paddle.io.Dataset/IterableDataset`来创建数据集也是支持的，您也可以从其他python对象如`List`对象创建数据集，详细内容可参照[官方文档-自定义数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)。

<a name="1-2"></a>

##### Q1.2 PaddleNLP会将内置的数据集、模型下载到默认路径，如何修改路径？

**A:** 内置的数据集、模型默认会下载到`$HOME/.paddlenlp/`下，通过配置环境变量可下载到指定路径：

（1）Linux下，设置 `export PPNLP_HOME="xxxx"`，注意不要设置带有中文字符的路径。

（2）Windows下，同样配置环境变量 PPNLP_HOME 到其他非中文字符路径，重启即可。

<a name="1-3"></a>

##### Q1.3 PaddleNLP中如何保存、加载训练好的模型？

**A：**（1）PaddleNLP预训练模型

​    保存：

```python
model.save_pretrained("./checkpoint')
tokenizer.save_pretrained("./checkpoint')
```

​    加载：

```python
model.from_pretrained("./checkpoint')
tokenizer.from_pretrained("./checkpoint')
```

（2）常规模型
    保存：

```python
emb = paddle.nn.Embedding(10, 10)
layer_state_dict = emb.state_dict()
paddle.save(layer_state_dict, "emb.pdparams") #保存模型参数
```

​    加载：
```python
emb = paddle.nn.Embedding(10, 10)
load_layer_state_dict = paddle.load("emb.pdparams") # 读取模型参数
emb.set_state_dict(load_layer_state_dict) # 加载模型参数
```

<a name="1-4"></a>

##### Q1.4 当训练样本较少时，有什么推荐的方法能提升模型效果吗？

**A:** 增加训练样本带来的效果是最直接的。此外，可以基于我们开源的[预训练模型](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers)进行热启，再用少量数据集fine-tune模型。此外，针对分类、匹配等场景，[小样本学习](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/few_shot)也能够带来不错的效果。

<a name="1-5"></a>

##### Q1.5 如何提升模型的性能，提升QPS？

**A:** 从工程角度，对于服务器端部署可以使用[Paddle Inference](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)高性能预测引擎进行预测部署。对于Transformer类模型的GPU预测还可以使用PaddleNLP中提供的[FasterTransformer](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/ops)功能来进行快速预测，其集成了[NV FasterTransformer](https://github.com/NVIDIA/FasterTransformer)并进行了功能增强。

从模型策略角度，可以使用一些模型小型化技术来进行模型压缩，如模型蒸馏和裁剪，通过小模型来实现加速。PaddleNLP中集成了ERNIE-Tiny这样一些通用小模型供下游任务微调使用。另外PaddleNLP提供了[模型压缩示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression)，实现了DynaBERT、TinyBERT、MiniLM等方法策略，可以参考对自己的模型进行蒸馏压缩。

<a name="NLP通用问题"></a>

## ⭐️【理论篇】NLP通用问题

<a name="2-2"></a>

##### Q2.1 数据类别分布不均衡， 有哪些应对方法？

**A:** 可以采用以下几种方法优化类别分布不均衡问题：

（1）欠采样：对样本量较多的类别进行欠采样，去除一些样本，使得各类别数目接近。

（2）过采样：对样本量较少的类别进行过采样，选择样本进行复制，使得各类别数目接近。

（3）修改分类阈值：直接使用类别分布不均衡的数据训练分类器，会使得模型在预测时更偏向于多数类，所以不再以0.5为分类阈值，而是针对少数类在模型仅有较小把握时就将样本归为少数类。

（4）代价敏感学习：比如LR算法中设置class_weight参数。

<a name="2-3"></a>

##### Q2.2 如果使用预训练模型，一般需要多少条样本？

**A:** 很难定义具体需要多少条样本，取决于具体的任务以及数据的质量。如果数据质量没问题的话，分类、文本匹配任务所需数据量级在百级别，翻译则需要百万级能够训练出一个比较鲁棒的模型。如果样本量较少，可以考虑数据增强，或小样本学习。


<a name="PaddleNLP实战问题"></a>

## ⭐️【实战篇】PaddleNLP实战问题

<a name="数据问题"></a>

### 数据集和数据处理

<a name="3-1"></a>

##### Q3.1 使用自己的数据集训练预训练模型时，如何引入额外的词表？

**A:** 预训练模型通常会有配套的tokenzier和词典，对于大多数中文预训练模型，如ERNIE-3.0，使用的都是字粒度的输入，tokenzier会将句子转换为字粒度的形式，模型无法收到词粒度的输入。如果希望引入额外的词典，需要修改预训练模型的tokenizer和词典，可以参考这里[blog](https://kexue.fm/archives/7758/comment-page-1#Tokenizer )，另外注意embedding矩阵也要加上这些新增词的embedding表示。

另外还有一种方式可以使用这些字典信息，可以将数据中在词典信息中的词进行整体mask进行一个mask language model的二次预训练，这样经过二次训练的模型就包含了对额外字典的表征。可参考 [PaddleNLP 预训练数据流程](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-1.0/data_tools)。


此外还有些词粒度及字词混合粒度的预训练模型，在这些词粒度的模型下引入额外的词表也会容易些，我们也将持续丰富PaddleNLP中的预训练模型。

<a name="训练调优问题"></a>

### 模型训练调优

<a name="4-1"></a>

##### Q3.2 如何加载自己的预训练模型，进而使用PaddleNLP的功能？

**A:** 以bert为例，如果是使用PaddleNLP训练，通过`save_pretrained()`接口保存的模型，可通过`from_pretrained()`来加载：

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
```

如果不是上述情况，可以使用如下方式加载模型，也欢迎您贡献模型到PaddleNLP repo中。

（1）加载`BertTokenizer`和`BertModel`

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
```

（2）调用`save_pretrained()`生成 `model_config.json`、 ``tokenizer_config.json``、`model_state.pdparams`、  `vocab.txt `文件，保存到`./checkpoint`：

```python
tokenizer.save_pretrained("./checkpoint")
model.save_pretrained("./checkpoint")
```

（3）修改`model_config.json`、 `tokenizer_config.json`这两个配置文件，指定为自己的模型，之后通过`from_pretrained()`加载模型。

```python
tokenizer = BertTokenizer.from_pretrained("./checkpoint")
model = BertModel.from_pretrained("./checkpoint")
```

<a name="4-2"></a>

##### Q3.3 如果训练中断，需要继续热启动训练，如何保证学习率和优化器能从中断地方继续迭代？

**A:**

 （1）完全恢复训练状态，可以先将`lr`、` optimizer`、`model`的参数保存下来：

```python
paddle.save(lr_scheduler.state_dict(), "xxx_lr")
paddle.save(optimizer.state_dict(), "xxx_opt")
paddle.save(model.state_dict(), "xxx_para")
```

（2）加载`lr`、` optimizer`、`model`参数即可恢复训练：

```python
lr_scheduler.set_state_dict(paddle.load("xxxx_lr"))
optimizer.set_state_dict(paddle.load("xxx_opt"))
model.set_state_dict(paddle.load("xxx_para"))
```

<a name="4-3"></a>

##### Q3.4 如何冻结模型梯度？

**A:**
有多种方法可以尝试：

（1）可以直接修改 PaddleNLP 内部代码实现，在需要冻结梯度的地方用 `paddle.no_grad()` 包裹一下

   `paddle.no_grad()` 的使用方式，以对 `forward()` 进行冻结为例：

``` python
   # Method 1
   class Model(nn.Layer):
      def __init__(self, ...):
         ...

      def forward(self, ...):
         with paddle.no_grad():
            ...


   # Method 2
   class Model(nn.Layer):
      def __init__(self, ...):
         ...

      @paddle.no_grad()
      def forward(self, ...):
         ...
```

   `paddle.no_grad()` 的使用也不局限于模型内部实现里面，也可以包裹外部的方法，比如：

``` python
   @paddle.no_grad()
   def evaluation(...):
      ...

      model = Model(...)
      model.eval()

      ...

```

（2）第二种方法：以ERNIE为例，将模型输出的 tensor 设置 `stop_gradient` 为 True。可以使用 `register_forward_post_hook` 按照如下的方式尝试：

``` python
   def forward_post_hook(layer, input, output):
      output.stop_gradient=True

   self.ernie.register_forward_post_hook(forward_post_hook)
```

（3）第三种方法：在 `optimizer` 上进行处理，`model.parameters` 是一个 `List`，可以通过 `name` 进行相应的过滤，更新/不更新某些参数，这种方法需要对网络结构的名字有整体了解，因为网络结构的实体名字决定了参数的名字，这个使用方法有一定的门槛：

```python
   [ p for p in model.parameters() if 'linear' not in p.name]  # 这里就可以过滤一下linear层，具体过滤策略可以根据需要来设定
```

<a name="4-4"></a>

##### Q3.5 如何在eval阶段打印评价指标，在各epoch保存模型参数？

**A:** 飞桨主框架提供了两种训练与预测的方法，一种是用 [paddle.Model()](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Model_cn.html)对模型进行封装，通过高层API如`Model.fit()`、`Model.evaluate()`、`Model.predict()`等完成模型的训练与预测；另一种就是基于基础API常规的训练方式。

（1）对于第一种方法：

- 我们可以设置 `paddle.Model.fit() ` API中的 *eval_data* 和 *eval_freq* 参数在训练过程中打印模型评价指标：*eval_data* 参数是一个可迭代的验证集数据源，*eval_freq* 参数是评估的频率；当*eval_data* 给定后，*eval_freq* 的默认值为1，即每一个epoch进行一次评估。注意：在训练前，我们需要在 `Model.prepare()` 接口传入metrics参数才能在eval时打印模型评价指标。

- 关于模型保存，我们可以设置 `paddle.Model.fit()` 中的 *save_freq* 参数控制模型保存的频率：*save_freq* 的默认值为1，即每一个epoch保存一次模型。

（2）对于第二种方法：

- 我们在PaddleNLP的examples目录下提供了常见任务的训练与预测脚本：如[GLUE](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/benchmark/glue) 和 [SQuAD](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_reading_comprehension/SQuAD)等

- 开发者可以参考上述脚本进行自定义训练与预测脚本的开发。

<a name="4-5"></a>

##### Q3.6 训练过程中，训练程序意外退出或Hang住，应该如何排查？

**A:**  一般先考虑内存、显存（使用GPU训练的话）是否不足，可将训练和评估的batch size调小一些。

需要注意，batch size调小时，学习率learning rate也要调小，一般可按等比例调整。

<a name="4-6"></a>

##### Q3.7 在模型验证和测试过程中，如何保证每一次的结果是相同的？

**A:** 在验证和测试过程中常常出现的结果不一致情况一般有以下几种解决方法：

（1）确保设置了eval模式，并保证数据相关的seed设置保证数据一致性。

（2）如果是下游任务模型，查看是否所有模型参数都被导入了，直接使用bert-base这种预训练模型是不包含任务相关参数的，要确认导入的是微调后的模型，否则任务相关参数会随机初始化导致出现随机性。

（3）部分算子使用CUDNN后端产生的不一致性可以通过环境变量的设置来避免。如果模型中使用了CNN相关算子，可以设置`FLAGS_cudnn_deterministic=True`。如果模型中使用了RNN相关算子，可以设置`CUBLAS_WORKSPACE_CONFIG=:16:8`或`CUBLAS_WORKSPACE_CONFIG=:4096:2`（CUDNN 10.2以上版本可用，参考[CUDNN 8 release note](https://docs.nvidia.com/deeplearning/sdk/cudnn-release-notes/rel_8.html)）。

<a name="4-7"></a>

##### Q3.8 ERNIE模型如何返回中间层的输出？

**A:** 目前的API设计不保留中间层输出，当然在PaddleNLP里可以很方便地修改源码。
此外，还可以在`ErnieModel`的`__init__`函数中通过`register_forward_post_hook()`为想要保留输出的Layer注册一个`forward_post_hook`函数，在`forward_post_hook`函数中把Layer的输出保存到一个全局的`List`里面。`forward_post_hook`函数将会在`forward`函数调用之后被调用，并保存Layer输出到全局的`List`。详情参考[`register_forward_post_hook()`](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#register_forward_post_hook)。

<a name="部署问题"></a>

### 预测部署

<a name="5-1"></a>

##### Q3.9 PaddleNLP训练好的模型如何部署到服务器 ？

**A:** 我们推荐在动态图模式下开发，静态图模式部署。

（1）动转静

   动转静，即将动态图的模型转为可用于部署的静态图模型。
   动态图接口更加易用，python 风格的交互式编程体验，对于模型开发更为友好，而静态图相比于动态图在性能方面有更绝对的优势。因此动转静提供了这样的桥梁，同时兼顾开发成本和性能。
   可以参考官方文档 [动态图转静态图文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/04_dygraph_to_static/index_cn.html)，使用 `paddle.jit.to_static` 完成动转静。
   另外，在 PaddleNLP 我们也提供了导出静态图模型的例子，可以参考 [waybill_ie 模型导出](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/waybill_ie/#%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA)。

（2）借助Paddle Inference部署

   动转静之后保存下来的模型可以借助Paddle Inference完成高性能推理部署。Paddle Inference内置高性能的CPU/GPU Kernel，结合细粒度OP横向纵向融合等策略，并集成 TensorRT 实现模型推理的性能提升。具体可以参考文档 [Paddle Inference 简介](https://paddleinference.paddlepaddle.org.cn/master/product_introduction/inference_intro.html)。
   为便于初次上手的用户更易理解 NLP 模型如何使用Paddle Inference，PaddleNLP 也提供了对应的例子以供参考，可以参考 [/PaddleNLP/examples](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/) 下的deploy目录，如[基于ERNIE的命名实体识别模型部署](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/waybill_ie/deploy/python)。

<a name="5-2"></a>

##### Q3.10 静态图模型如何转换成动态图模型？

**A:** 首先，需要将静态图参数保存成`ndarray`数据，然后将静态图参数名和对应动态图参数名对应，最后保存成动态图参数即可。详情可参考[参数转换脚本](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie/static_to_dygraph_params)。

<a name="NLP应用场景"></a>

### ⭐️特定模型和应用场景咨询

<a name="6-1"></a>

##### Q4.1 【词法分析】LAC模型，如何自定义标签label，并继续训练？

**A:** 更新label文件`tag.dict`，添加 修改下CRF的标签数即可。

可参考[自定义标签示例](https://github.com/PaddlePaddle/PaddleNLP/issues/662)，[增量训练自定义LABLE示例](https://github.com/PaddlePaddle/PaddleNLP/issues/657)。

<a name="6-2"></a>

##### Q4.2 信息抽取任务中，是否推荐使用预训练模型+CRF，怎么实现呢？

**A:** 预训练模型+CRF是一个通用的序列标注的方法，目前预训练模型对序列信息的表达也是非常强的，也可以尝试直接使用预训练模型对序列标注任务建模。

<a name="6-3"></a>

##### Q4.3.【阅读理解】`MapDatasets`的`map()`方法中对应的`batched=True`怎么理解，在阅读理解任务中为什么必须把参数`batched`设置为`True`？

**A:** `batched=True`就是对整个batch（这里不一定是训练中的batch，理解为一组数据就可以）的数据进行map，即map中的trans_func接受一组数据为输入，而非逐条进行map。在阅读理解任务中，根据使用的doc_stride不同，一条样本可能被转换成多条feature，对数据逐条map是行不通的，所以需要设置`batched=True`。

<a name="6-4"></a>

##### Q4.4 【语义匹配】语义索引和语义匹配有什么区别？

**A:** 语义索引要解决的核心问题是如何从海量 Doc 中通过 ANN 索引的方式快速、准确地找出与 query 相关的文档，语义匹配要解决的核心问题是对 query和文档更精细的语义匹配信息建模。换个角度理解， [语义索引](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/semantic_indexing)是要解决搜索、推荐场景下的召回问题，而[语义匹配](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_matching)是要解决排序问题，两者要解决的问题不同，所采用的方案也会有很大不同，但两者间存在一些共通的技术点，可以互相借鉴。

<a name="6-5"></a>

##### Q4.5 【解语】wordtag模型如何自定义添加命名实体及对应词类?

**A:** 其主要依赖于二次构造数据来进行finetune，同时要更新termtree信息。wordtag分为两个步骤：
（1）通过BIOES体系进行分词；
（2）将分词后的信息和TermTree进行匹配。
    因此我们需要：
（1）分词正确，这里可能依赖于wordtag的finetune数据，来让分词正确；
（2）wordtag里面也需要把分词正确后term打上相应的知识信息。wordtag自定义TermTree的方式将在后续版本提供出来。

可参考[issue](https://github.com/PaddlePaddle/PaddleNLP/issues/822)。

<a name="使用咨询问题"></a>

### ⭐️其他使用咨询

<a name="7-1"></a>

##### Q5.1 在CUDA11使用PaddlNLP报错?

**A:** 在CUDA11安装，可参考[issue](https://github.com/PaddlePaddle/PaddleNLP/issues/348)，其他CUDA版本安装可参考 [官方文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html)

<a name="7-2"></a>

##### Q5.2 如何设置parameter？

**A:** 有多种方法：
（1）可以通过`set_value()`来设置parameter，`set_value()`的参数可以是`numpy`或者`tensor`。

```python
   layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.ernie.config["initializer_range"],
                        shape=layer.weight.shape))
```
（2）通过`create_parameter()`设置参数。

``` python
    class MyLayer(paddle.nn.Layer):
        def __init__(self):
            super(MyLayer, self).__init__()
            self._linear = paddle.nn.Linear(1, 1)
            w_tmp = self.create_parameter([1,1])
            self.add_parameter("w_tmp", w_tmp)

        def forward(self, input):
            return self._linear(input)

    mylayer = MyLayer()
    for name, param in mylayer.named_parameters():
        print(name, param)
```

<a name="7-3"></a>

##### Q5.3 GPU版的Paddle虽然能在CPU上运行，但是必须要有GPU设备吗？

**A:** 不支持 GPU 的设备只能安装 CPU 版本的 PaddlePaddle。 GPU 版本的 PaddlePaddle 如果想只在 CPU 上运行，可以通过 `export CUDA_VISIBLE_DEVICES=-1` 来设置。

<a name="7-4"></a>

##### Q5.4  如何指定用CPU还是GPU训练模型？

**A:** 一般我们的训练脚本提供了 `--device` 选项，用户可以通过 `--device` 选择需要使用的设备。

具体而言，在Python文件中，我们可以通过·paddle.device.set_device()·，设置为gpu或者cpu，可参考 [set_device文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html#set-device)。

<a name="7-5"></a>

##### Q5.5 动态图模型和静态图模型的预测结果一致吗？

**A:** 正常情况下，预测结果应当是一致的。如果遇到不一致的情况，可以及时反馈给 PaddleNLP 的开发人员，我们进行处理。

<a name="7-6"></a>

##### Q5.6 如何可视化acc、loss曲线图、模型网络结构图等？

**A:** 可使用[VisualDL](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/03_VisualDL/index_cn.html)进行可视化。其中acc、loss曲线图的可视化可参考[Scalar——折线图组件](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/03_VisualDL/visualdl_usage_cn.html#scalar)使用指南，模型网络结构的可视化可参考[Graph——网络结构组件](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/03_VisualDL/visualdl_usage_cn.html#graph)使用指南。
