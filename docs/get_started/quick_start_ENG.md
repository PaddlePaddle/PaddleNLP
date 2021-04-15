 [**中文**](./quick_start.rst)

Finish the High-precision Sentiment Analysis within 10 Minutes
  ==============================

## 1\. Install PaddleNLP

If you want to know the installation, or meet with some problems, please refer to [the installation documentation](https://paddlenlp.readthedocs.io/en/latest/gettingstarted/install.html) of PaddleNLP.

``` {.}
>>> pip install --upgrade paddlenlp>=2.0.0rc -i https://pypi.org/simple
```

## 2\. Load Pre-trained Models with One Click

The essence of sentiment analysis is text classification. PaddleNLP has a lot of built-in pre-trained models like ERNIE, BERT, RoBERTa, Electra. There are also different finetuned nets of pre-trained models, aiming to process various downstream tasks. You can finish tasks like Q&A, sequence classification, token classification and so forth. If you want to know more details, please refer to [pre-trained models](https://paddlenlp.readthedocs.io/en/latest/modelzoo/transformer.html). We will take ERNIE as an example and introduce how to use the finetuned net of the pre-trained model to finish text classification tasks.

Load the pre-trained model–ERNIE

``` {.}
>>> MODEL_NAME = "ernie-1.0"
>>> ernie_model = paddlenlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)
```

Load the fine-tuned net of ERNIE that is specialized for text classification. As long as you specify model names and categories of text classification, you can define the net.

``` {.}
>>> model = paddlenlp.transformers.ErnieForSequenceClassification.from_pretrained(
...     MODEL_NAME, num_classes=len(label_list))
```

## 3\. Call Tokenizer to Process Data

We use Tokenizer to change original input texts into acceptable data format. PaddleNLP has built-in Tokenizers aiming to process different pre-trained models. Once specifying one model name, you can load it. 

``` {.}
>>> tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
```

The data processing of pre-trained models based on Transformer, usually includes text segmentation token; and then you need to map the token into its token id; it is also needed to be spliced with the specialized tokens, like [CLS] and [SEP], which match the pre-trained model; finally, you are required to change it into the data format required by the framework. For easiness of using, PaddleNLP provides you with high-level APIs. Only one click can make the format suitable. 

One line of codes can finish token segmenting, token ID mapping, and specialized token splicing.

``` {.}
>>> encoded_text = tokenizer(text="Please enter the test")
```

It is changed into the data format of PaddlePaddle.

``` {.}
>>> input_ids = paddle.to_tensor([encoded_text['input_ids']])
>>> print("input_ids : {}".format(input_ids))
>>> token_type_ids = paddle.to_tensor([encoded_text['token_type_ids']])
>>> print("token_type_ids : {}".format(token_type_ids))
input_ids : Tensor(shape=[1, 9], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
   [[1  , 647, 789, 109, 558, 525, 314, 656, 2  ]])
token_type_ids : Tensor(shape=[1, 9], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
   [[0, 0, 0, 0, 0, 0, 0, 0, 0]])
```

input_ids: It represents the token ID of input texts.

token_type_ids: It represents that whether the token belongs to the first input sentence or the second input sentence. (Pre-trained models based on Transformer support the inputs like singe sentence or sentence pairs)

At this time, you can take the outputs of ERNIE as the input.

``` {.}
>>> sequence_output, pooled_output = ernie_model(input_ids, token_type_ids)
>>> print("Token wise output: {}, Pooled output: {}".format(
...     sequence_output.shape, pooled_output.shape))
Token wise output: [1, 9, 768], Pooled output: [1, 768]
```

It can be inferred that there are two tensors in the outputs of ERNIE.

sequence_output is the semantic feature representation of every input token, and shape is (1, num_tokens, hidden_size). sequence_output is always used in the tasks like sequence tagging, Q&A and so forth.

pooled_output is the semantic feature representation of the whole sentence, and shape is (1, hidden_size). It is always used in the tasks like text classification, information retrieval and so forth.

## 4. Load Datasets

PaddleNLP has several built-in datasets, used for downstream tasks like reading comprehension, sequence tagging, machine translation and so forth. Here, we use ChnSenticorp, the public dataset of Chinese sentiment analysis. There are over 7000 positive or negative remarks of hotels.

Load PaddleNLP’s built-in datasets with one click:

``` {.}
>>> train_ds, dev_ds, test_ds = paddlenlp.datasets.load_dataset(
...     'chnsenticorp', splits=['train', 'dev', 'test'])
```

Acquire the labels of classification data:

``` {.}
>>> label_list = train_ds.label_list
>>> print(label_list)
['0', '1']
```

Display part of data: 

``` {.}
>>> for idx in range(5):
...     print(train_ds[idx])

{'text': '选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。
酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般', 'label': 1}
{'text': '15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错', 'label': 1}
{'text': '房间太小。其他的都一般。。。。。。。。。', 'label': 0}
{'text': '1.接电源没有几分钟,电源适配器热的不行. 2.摄像头用不起来. 3.机盖的钢琴漆，手不能摸，一摸一个印. 4.硬盘分区不好办.', 'label': 0}
{'text': '今天才知道这书还有第6卷,真有点郁闷:为什么同一套书有两种版本呢?当当网是不是该跟出版社商量商量,
单独出个第6卷,让我们的孩子不会有所遗憾。', 'label': 1}
```

## 5\. Model Training and Evaluation

When you access the data, please use the inference–`paddle.io.DataLoader`{.interpreted-text role="func"} to load data in the asynchronous multi-threaded way. And then you can set dynamic learning rate, loss function, optimization algorithm, evaluation index of ERNIE.

The process of training models is usually as following:
1.	Take out one batch data from dataloader.
2.	Use batch data to feed the model, and do some forward propagation

3.	Please pass the result of forward propagation to the loss function, and calculate the loss. And then pass the result of forward propagation to the evaluation index, and calculate the index.
4.	The backward return of loss will help users update the gradients. Please repeat the above steps.
5.	Once one epoch of training is finished, the program will evaluate the effects of the models trained at present.

This example will also be shown on AI Studio, where you can [experience models’ training online](https://aistudio.baidu.com/aistudio/projectdetail/1294333).

Finally, please save the trained models to infer.

## 6. Model Inference

Please save the trained models, and define the function– `predict`{.interpreted-text role="func"}. And then you can start to infer the sentiment tendency of texts.

Let’s take the self-defined data and data labels as an example:

``` {.}
>>> data = [
...     '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般',
...     '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片',
...     '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。',
... ]
>>> label_map = {0: 'negative', 1: 'positive'}
```

And you will get the inference result:

``` {.}
>>> results = predict(
...     model, data, tokenizer, label_map, batch_size=batch_size)
>>> for idx, text in enumerate(data):
...     print('Data: {} \t Label: {}'.format(text, results[idx]))
Data: 这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般     Label: negative
Data: 怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片    Label: negative
Data: 作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。   Label: positive
```

