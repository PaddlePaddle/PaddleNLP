 [**中文**](./quick_start.rst)

Finish the High-precision Sentiment Analysis within 10 Minutes
  ==============================

## 1\. Install PaddleNLP

If you want to know the installation, or meet with some problems, please refer to [the installation documentation](https://paddlenlp.readthedocs.io/en/latest/gettingstarted/install.html) of PaddleNLP.

``` {.}
>>> pip install --upgrade paddlenlp>=2.0.0rc -i https://pypi.org/simple
```

## 2\. Load Pre-trained Models with One Click

The essence of sentiment analysis is text classification. PaddleNLP has a lot of built-in pre-trained models like ERNIE, BERT, RoBERTa, Electra. There are also different finetuned nets of pre-trained models, aiming to process various downstream tasks. You can finish tasks like Q&A, sequence classification, token classification and so forth. If you want to know more details, please refer to [pre-trained models](https://paddlenlp.readthedocs.io/en/latest/modelzoo/transformer.html). We will take ERNIE as an example and introduce how to use the finetuned nets of the pre-trained models to finish text classification tasks.

Load the pre-trained model–ERNIE

``` {.}
>>> MODEL_NAME = "ernie-1.0"
>>> ernie_model = paddlenlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)
```

Load the fine-tuned net of ERNIE that is specialized for text classification. As long as you specify model names and categories of text classification, you can define the nets.

``` {.}
>>> model = paddlenlp.transformers.ErnieForSequenceClassification.from_pretrained(
...     MODEL_NAME, num_classes=len(label_list))
```

## 3\. Call Tokenizer to Process Data

We use Tokenizer to change original input texts into acceptable data format. PaddleNLP has built-in Tokenizers aiming to process different pre-trained models. Once specifying one model name, you can load it. 

``` {.}
>>> tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
```

The data processing of pre-trained models based on Transformer, usually includes text segmentation token; and then you need to map the token into its token id; it is also needed to be sliced with the specialized token, like [CLS] and [SEP]; finally, you are required to change it into the data format required by the framework. For easiness of using, PaddleNLP provides you with high-level APIs. Only one click can make the format suitable. 

Only one line of codes can finish token segmenting, token ID mapping, and specialized token slicing.

``` {.}
>>> encoded_text = tokenizer(text="Please enter the test sample")
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

token_type_ids: It represents that the relative token is the first input sentence or the second input sentence. (Pre-trained models based on Transformer support the inputs like singe sentence or sentence pairs)

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

{'text': 'The reason for choosing Zhujiang Garden was its convenience. There were escalators directly to the seaside, and there were restaurants, food galleries, shopping malls, supermarkets and stalls all around.
The decoration of the hotel was not bad, but it's clean and tidy. The pool was on the roof, so it's very small. But my daughter liked it. The breakfast was western-style and rich. Service? Not bad
', 'label': 1}
{'text': 'The keyboard of 15.4" laptop is really cool. It's just like the desktop. I really like the numeric keypad. It's very convenient for inputting numbers. It's also very beautiful and the workmanship is quite good', 'label': 1}
{'text': 'The room was so small, and others were not bad…', 'label': 0}
{'text': '1. Only a few minutes since plugged in, the power adapter became too hot. 2.Camera did not work. 3.The piano lacquer of the cover couldn’t be touched, or there would be some marks left. 4.It’s not easy to distinguish hard disks.', 'label': 0}
{'text': 'Until today, I know that there is the 6th volume of this book, which is a little depressing. Why does one set of books have two versions? Should Dangdang coordinate with the publisher to publish the 6th volume as a single book? Then our kids will not feel regretful.', 'label': 1}
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
...     'This hotel was a little old, and the special promotion room was not bad. Overall, it was just so so',
...     'I was excited to show the films. When the film was finished, there was one episode of the cartoon–Mickey Mouse',
...     'As the old four-star hotel, its rooms were tidy and clean, which was really good. The collection service from the airport was so nice that you could check-in on the bus and saved the time.',
... ]
>>> label_map = {0: 'negative', 1: 'positive'}
```

And you will get the inference result:

``` {.}
>>> results = predict(
...     model, data, tokenizer, label_map, batch_size=batch_size)
>>> for idx, text in enumerate(data):
...     print('Data: {} \t Label: {}'.format(text, results[idx]))
Data: This hotel was a little old, and the special promotion room was not bad. Overall, it was just so so      Label: negative
Data: I was excited to show the films. When the film was finished, there was one episode of the cartoon–Mickey Mouse    Label: negative
Data: As the old four-star hotel, its rooms were tidy and clean, which was really good. The collection service from the airport was so nice that you could check-in on the bus and saved the time.   Label: positive
```
