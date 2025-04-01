========
10-Minute Guide to High-Accuracy Chinese Sentiment Analysis
========

1. Install PaddleNLP
========

For installation procedures and troubleshooting, please refer to the [Installation Documentation](https://paddlenlp.readthedocs.io/en/latest/gettingstarted/install.html).

.. code-block::

    >>> pip install --upgrade paddlenlp -i https://pypi.org/simple

2. One-Click Loading of Pretrained Models
========

Sentiment analysis is essentially a text classification task. PaddleNLP provides various pretrained models including ERNIE, BERT, RoBERTa, and Electra, along with fine-tuning networks for different downstream tasks. Let's use ERNIE as an example.

Load ERNIE model:

.. code-block::

    >>> MODEL_NAME = "ernie-3.0-medium-zh"
    >>> ernie_model = paddlenlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)

Load text classification head:

.. code-block::

    >>> model = paddlenlp.transformers.ErnieForSequenceClassification.from_pretrained(
    ...     MODEL_NAME, num_classes=len(label_list))

3. Data Processing with Tokenizer
========

Load tokenizer:

.. code-block::

    >>> tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)

Text processing example:

.. code-block::

    >>> encoded_text = tokenizer(text="Please input test sample")

Convert to tensor:

.. code-block::

    >>> input_ids = paddle.to_tensor([encoded_text['input_ids']])
    >>> token_type_ids = paddle.to_tensor([encoded_text['token_type_ids']])

Model inference:

.. code-block::

    >>> sequence_output, pooled_output = ernie_model(input_ids, token_type_ids)
    >>> print(f"Token wise output: {sequence_output.shape}, Pooled output: {pooled_output.shape}")

4. Load Dataset
========

Load ChnSenticorp dataset:

.. code-block::

    >>> train_ds, dev_ds, test_ds = paddlenlp.datasets.load_dataset(
    ...     'chnsenticorp', splits=['train', 'dev', 'test'])

Get label list:

.. code-block::

    >>> label_list = train_ds.label_list
    >>> print(label_list)

Sample data:

.. code-block::

    >>> for idx in range(5):
    ...     print(train_ds[idx])

5. Model Training and Evaluation
========

(Note: The original content ends here, so the translation stops accordingly while maintaining consistency.)
The :func:`paddle.io.DataLoader` interface asynchronously loads data with multi-threading, while configuring dynamic learning rates, loss functions, optimization algorithms, and evaluation metrics suitable for Transformer models like ERNIE.

The model training process typically follows these steps:

#. Fetch a batch of data from the dataloader.
#. Feed the batch data to the model for forward computation.
#. Pass the forward computation results to the loss function to calculate loss, and to evaluation metrics to compute performance metrics.
#. Perform backpropagation with the loss to update gradients. Repeat the above steps.
#. After each epoch, the program evaluates the model's current performance.

This example is also available on AIStudio for _online model training experience_.

.. _online model training experience: https://aistudio.baidu.com/aistudio/projectdetail/1294333

Finally, save the trained model for prediction.

6. Model Prediction
==================
After saving the trained model, define the prediction function :func:`predict` to perform sentiment analysis.

Example with custom prediction data and labels:

.. code-block::

    >>> data = [
    ...     'This hotel is rather outdated, and the discounted rooms are mediocre. Overall average',
    ...     'Started watching with great excitement, but found a Mickey Mouse cartoon appearing after the main feature',
    ...     'As an established four-star hotel, the rooms remain well-kept and impressive. The airport shuttle service is excellent, allowing check-in during the ride to save time.',
    ... ]
    >>> label_map = {0: 'negative', 1: 'positive'}

Prediction results:

.. code-block::

    >>> results = predict(
    ...     model, data, tokenizer, label_map, batch_size=batch_size)
    >>> for idx, text in enumerate(data):
    ...     print('Data: {} \t Label: {}'.format(text, results[idx]))
    Data: This hotel is rather outdated, and the discounted rooms are mediocre. Overall average 	 Label: negative
    Data: Started watching with great excitement, but found a Mickey Mouse cartoon appearing after the main feature 	 Label: negative
    Data: As an established four-star hotel, the rooms remain well-kept and impressive. The airport shuttle service is excellent, allowing check-in during the ride to save time. 	 Label: positive