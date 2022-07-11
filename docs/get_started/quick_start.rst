========
10分钟完成高精度中文情感分析
========

1. 安装PaddleNLP
========

安装相关过程和问题可以参考PaddleNLP的 安装文档_。

.. _安装文档: https://paddlenlp.readthedocs.io/en/latest/gettingstarted/install.html


.. code-block::

    >>> pip install --upgrade paddlenlp -i https://pypi.org/simple

2. 一键加载预训练模型
========

情感分析本质是一个文本分类任务。PaddleNLP内置了ERNIE、BERT、RoBERTa、Electra等丰富的预训练模型，并且内置了各种预训练模型对于不同下游任务的Fine-tune网络。用户可以使用PaddleNLP提供的模型，完成问答、序列分类、token分类等任务。查阅 预训练模型_ 了解更多。这里以ERNIE模型为例，介绍如何将预训练模型Fine-tune完成文本分类任务。

.. _预训练模型: https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html

加载预训练模型ERNIE

.. code-block::

    >>> MODEL_NAME = "ernie-3.0-medium-zh"
    >>> ernie_model = paddlenlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)
    
加载预训练模型ERNIE用于文本分类任务的Fine-tune网络，只需指定想要使用的模型名称和文本分类的类别数即可完成网络定义。

.. code-block::

    >>> model = paddlenlp.transformers.ErnieForSequenceClassification.from_pretrained(
    ...     MODEL_NAME, num_classes=len(label_list))
    
3. 调用Tokenizer进行数据处理
========    

Tokenizer用于将原始输入文本转化成模型可以接受的输入数据形式。PaddleNLP对于各种预训练模型已经内置了相应的Tokenizer，指定想要使用的模型名字即可加载。

.. code-block::

    >>> tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)

Transformer类预训练模型所需的数据处理步骤通常包括将原始输入文本切分token；将token映射为对应的token id；拼接上预训练模型对应的特殊token ，如[CLS]、[SEP]；最后转化为框架所需的数据格式。为了方便使用，PaddleNLP提供了高阶API，一键即可返回模型所需数据格式。

一行代码完成切分token，映射token ID以及拼接特殊token:

.. code-block::

    >>> encoded_text = tokenizer(text="请输入测试样例")
    
转化成paddle框架数据格式:

.. code-block::

    >>> input_ids = paddle.to_tensor([encoded_text['input_ids']])
    >>> print("input_ids : {}".format(input_ids))
    >>> token_type_ids = paddle.to_tensor([encoded_text['token_type_ids']])
    >>> print("token_type_ids : {}".format(token_type_ids))
    input_ids : Tensor(shape=[1, 9], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
       [[1  , 647, 789, 109, 558, 525, 314, 656, 2  ]])
    token_type_ids : Tensor(shape=[1, 9], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
       [[0, 0, 0, 0, 0, 0, 0, 0, 0]])

input_ids: 表示输入文本的token ID。

token_type_ids: 表示对应的token属于输入的第一个句子还是第二个句子。（Transformer类预训练模型支持单句以及句对输入。）

此时即可输入ERNIE模型中得到相应输出。

.. code-block::

    >>> sequence_output, pooled_output = ernie_model(input_ids, token_type_ids)
    >>> print("Token wise output: {}, Pooled output: {}".format(
    ...     sequence_output.shape, pooled_output.shape))
    Token wise output: [1, 9, 768], Pooled output: [1, 768]

可以看出，ERNIE模型输出有2个tensor。

sequence_output是对应每个输入token的语义特征表示，shape为(1, num_tokens, hidden_size)。其一般用于序列标注、问答等任务。

pooled_output是对应整个句子的语义特征表示，shape为(1, hidden_size)。其一般用于文本分类、信息检索等任务。

4. 加载数据集
========  
PaddleNLP内置了适用于阅读理解、文本分类、序列标注、机器翻译等下游任务的多个数据集，这里我们使用公开中文情感分析数据集ChnSenticorp，包含7000多条正负向酒店评论数据。

一键加载PaddleNLP内置数据集：

.. code-block::

    >>> train_ds, dev_ds, test_ds = paddlenlp.datasets.load_dataset(
    ...     'chnsenticorp', splits=['train', 'dev', 'test'])

获取分类数据标签：

.. code-block::

    >>> label_list = train_ds.label_list
    >>> print(label_list)
    ['0', '1']

展示一些数据：

.. code-block::

    >>> for idx in range(5):
    ...     print(train_ds[idx])

    {'text': '选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。
    酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般', 'label': 1}
    {'text': '15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错', 'label': 1}
    {'text': '房间太小。其他的都一般。。。。。。。。。', 'label': 0}
    {'text': '1.接电源没有几分钟,电源适配器热的不行. 2.摄像头用不起来. 3.机盖的钢琴漆，手不能摸，一摸一个印. 4.硬盘分区不好办.', 'label': 0}
    {'text': '今天才知道这书还有第6卷,真有点郁闷:为什么同一套书有两种版本呢?当当网是不是该跟出版社商量商量,
    单独出个第6卷,让我们的孩子不会有所遗憾。', 'label': 1}

5. 模型训练与评估
========  
数据读入时使用 :func:`paddle.io.DataLoader` 接口多线程异步加载数据，然后设置适用于ERNIE这类Transformer模型的动态学习率和损失函数、优化算法、评价指标等。

模型训练的过程通常按照以下步骤：

#. 从dataloader中取出一个batch data。
#. 将batch data喂给model，做前向计算。
#. 将前向计算结果传给损失函数，计算loss。将前向计算结果传给评价方法，计算评价指标。
#. loss反向回传，更新梯度。重复以上步骤。
#. 每训练一个epoch时，程序将会评估一次，评估当前模型训练的效果。

本示例同步在AIStudio上，可直接 在线体验模型训练_。

.. _在线体验模型训练: https://aistudio.baidu.com/aistudio/projectdetail/1294333

最后，保存训练好的模型用于预测。

6. 模型预测
========  
保存训练模型，定义预测函数 :func:`predict` ，即可开始预测文本情感倾向。

以自定义预测数据和数据标签为示例：

.. code-block::

    >>> data = [
    ...     '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般',
    ...     '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片',
    ...     '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。',
    ... ]
    >>> label_map = {0: 'negative', 1: 'positive'}

得到预测结果：

.. code-block::

    >>> results = predict(
    ...     model, data, tokenizer, label_map, batch_size=batch_size)
    >>> for idx, text in enumerate(data):
    ...     print('Data: {} \t Label: {}'.format(text, results[idx]))
    Data: 这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般 	 Label: negative
    Data: 怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片 	 Label: negative
    Data: 作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。 	 Label: positive
