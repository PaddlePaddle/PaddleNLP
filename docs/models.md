# PaddleNLP Models API

该模块提供了百度自研的模型的高阶API，如文本分类模型Senta，文本匹配模型SimNet，通用预训练模型ERNIE等。

```python
class paddlenlp.models.Ernie(model_name, num_classes, task=None, **kwargs):
    """
    预训练模型ERNIE。
    更多信息参考：ERNIE: Enhanced Representation through Knowledge Integration(https://arxiv.org/abs/1904.09223)

    参数：
        `model_name (obj:`str`)`： 模型名称，如`ernie-1.0`，`ernie-tiny`，`ernie-2.0-en`， `ernie-2.0-large-en`。
        `num_classes (obj:`int`)`： 分类类别数。
        `task (obj:`str`)： 预训练模型ERNIE用于下游任务名称，可以为`seq-cls`，`token-cls`，`qa`. 默认为None

            - task='seq-cls'： ERNIE用于文本分类任务。其将从ERNIE模型中提取句子特征，用于最后一层全连接网络进行文本分类。
                详细信息参考：`paddlenlp.transformers.ErnieForSequenceClassification`。
            - task='token-cls'： ERNIE用于序列标注任务。其将从ERNIE模型中提取每一个token特征，用于最后一层全连接网络进行token分类。
                详细信息参考：`paddlenlp.transformers.ErnieForQuestionAnswering`。
            - task='qa'： ERNIE用于阅读理解任务。其将从ERNIE模型中提取每一个token特征，用于最后一层全连接网络进行答案位置在原文中位置的预测。
                详细信息参考：`paddlenlp.transformers.ErnieForTokenClassification`。
            - task='None'：预训练模型ERNIE。可将其作为backbone，用于提取句子特征pooled_output、token特征sequence_output。
                详细信息参考：`paddlenlp.transformers.ErnieModel`
    """

    def forward(input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        """
        参数：
            `input_ids (obj:`paddle.Tensor`)`：文本token id，shape为（batch_size, sequence_length）。
            `token_type_ids (obj:`paddle.Tensor`)`： 各token所在文本的标识（token属于文本1或者文本2），shape为（batch_size, sequence_length）。
                默认为None，表示所有token都属于文本1。
            `position_ids（obj:`paddle.Tensor`)`：各Token在输入序列中的位置，shape为（batch_size, sequence_length）。默认为None。
            `attention_mask`（obj:`paddle.Tensor`)`：为了避免在padding token上做attention操作，`attention_mask`表示token是否为padding token的标志矩阵，
                shape为（batch_size, sequence_length）。mask的值或为0或为1， 为1表示该token是padding token，为0表示该token为真实输入token id。默认为None。

        返回：
            - 当`task=None`时，返回相应下游任务的分类概率值`probs(obj:`paddle.Tensor`)`，shape为（batch_size，num_classes）。
            - 当`task=None`时，返回预训练模型ERNIE的句子特征pooled_output、token特征sequence_output。
              * pooled_output(obj:`paddle.Tensor`)：shape (batch_size，hidden_size)
              * sequence_output(obj:`paddle.Tensor`)：shape (batch_size，sequence_length, hidden_size)

        """

```


```python
class paddlenlp.models.Senta(network, vocab_size, num_classes, emb_dim=128, pad_token_id=0):
    """
    文本分类模型Senta

    参数：
        `network(obj:`str`)`： 网络名称，可选bow，bilstm，bilstm_attn，bigru，birnn，cnn，lstm，gru，rnn以及textcnn。

            - network='bow'，对输入word embedding相加作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.BoWEncoder`。
            - network=`bilstm`， 对输入word embedding进行双向lstm操作，取最后一个step的表示作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.LSTMEncoder`。
            - network=`bilstm_attn`， 对输入word embedding进行双向lstm和Attention操作，取最后一个step的表示作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.LSTMEncoder`。
            - network=`bigru`， 对输入word embedding进行双向gru操作，取最后一个step的表示作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.GRUEncoder`。
            - network=`birnn`， 对输入word embedding进行双向rnn操作，取最后一个step的表示作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.RNNEncoder`。
            - network='cnn'，对输入word embedding进行一次积操作后进行max-pooling，作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.CNNEncoder`。
            - network='lstm', 对输入word embedding进行lstm操作后进行max-pooling，作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.LSTMEncoder`。
            - network='gru', 对输入word embedding进行lstm操作后进行max-pooling，作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.GRUEncoder`。
            - network='rnn', 对输入word embedding进行lstm操作后进行max-pooling，作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.RNNEncoder`。
            - network='textcnn'，对输入word embedding进行多次卷积和max-pooling，作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.CNNEncoder`。

        `vocab_size(obj:`int`)`：词汇表大小。
        `num_classes(obj:`int`)`：分类类别数。
        `emb_dim(obj:`int`)`：word embedding维度，默认128.
        `pad_token_id(obj:`int`)`：padding token 在词汇表中index，默认0。

    """

    def forward(text, seq_len):
        """
        参数：
            `text(obj:`paddle.Tensor`)`: 文本token id，shape为（batch_size, sequence_length）。
            `seq_len(obj:`paddle.Tensor`): 文本序列长度, shape为（batch_size)。

        返回：
            `probs(obj:`paddle.Tensor`)`： 分类概率值，shape为（batch_size，num_classes）。

        """

```

```python
class paddlenlp.models.SimNet(nn.Layer):
    """
    文本匹配模型SimNet

    参数：
        `network(obj:`str`)`： 网络名称，可选bow，cnn，lstm，以及gru，rnn以及textcnn。

            - network='bow'，对输入word embedding相加作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.BoWEncoder`。
            - network='cnn'，对输入word embedding进行一次积操作后进行max-pooling，作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.CNNEncoder`。
            - network='lstm', 对输入word embedding进行lstm操作，取最后一个step的表示作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.LSTMEncoder`。
            - network='gru', 对输入word embedding进行lstm操作后进行max-pooling，取最后一个step的表示作为文本特征表示。
                详细信息参考：`paddlenlp.seq2vec.GRUEncoder`。

        `vocab_size(obj:`int`)`：词汇表大小。
        `num_classes(obj:`int`)`：分类类别数。
        `emb_dim(obj:`int`)`：word embedding维度，默认128。
        `pad_token_id(obj:`int`)`：padding token 在词汇表中index，默认0。

    """

    def forward(query, title, query_seq_len=None, title_seq_len=None):
        """
        参数：
            `query(obj:`paddle.Tensor`)`: query文本token id，shape为（batch_size, query_sequence_length）。
            `title(obj:`paddle.Tensor`)`: title文本token id，shape为（batch_size, title_sequence_length）。

            `query_seq_len(obj:`paddle.Tensor`): query文本序列长度，shape为（batch_size)。。

        返回：
            `probs(obj:`paddle.Tensor`)`： 分类概率值，shape为（batch_size，num_classes）。

        """

```
