由BERT到Bi-LSTM的知识蒸馏
============


整体原理介绍
------------

本例是将特定任务下BERT模型的知识蒸馏到基于Bi-LSTM的小模型中，主要参考论文 `Distilling Task-Specific Knowledge from BERT into Simple Neural Networks <https://arxiv.org/abs/1903.12136>`_ \
实现。整体原理如下：

1. 在本例中，较大的模型是BERT被称为教师模型，Bi-LSTM被称为学生模型。

2. 小模型学习大模型的知识，需要小模型学习蒸馏相关的损失函数。在本实验中，损失函数是均方误差损失函数，传入函数的两个参数分别是学生模型的输出和教师模型的输出。

3. 在论文的模型蒸馏阶段，作者为了能让教师模型表达出更多的“暗知识”(dark knowledge，通常指分类任务中低概率类别与高概率类别的关系)供学生模型学习，对训练数据进行了数据增强。通过数据增强，可以产生更多无标签的训练数据，在训练过程中，学生模型可借助教师模型的“暗知识”，在更大的数据集上进行训练，产生更好的蒸馏效果。本文的作者使用了三种数据增强方式，分别是：

 A. Masking，即以一定的概率将原数据中的word token替换成 ``[MASK]`` ；

 B. POS—guided word replacement，即以一定的概率将原数据中的词用与其有相同POS tag的词替换；

 C. n-gram sampling，即以一定的概率，从每条数据中采样n-gram，其中n的范围可通过人工设置。



模型蒸馏步骤介绍
------------

本实验分为三个训练过程：在特定任务上对BERT进行微调、在特定任务上对基于Bi-LSTM的小模型进行训练（用于评价蒸馏效果）、将BERT模型的知识蒸馏到基于Bi-LSTM的小模型上。

1. 基于bert-base-uncased预训练模型在特定任务上进行微调
^^^^^^^^^^^^

训练BERT的fine-tuning模型，可以去 `PaddleNLP <https:github.com/PaddlePaddle/PaddleNLP>`_ 中\
的 `glue <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/benchmark/glue>`_ 目录下对bert-base-uncased做微调。

以GLUE的SST-2任务为例，用bert-base-uncased做微调之后，可以得到一个在SST-2任务上的教师模型，可以把在dev上取得最好Accuracy的模型保存下来，用于第三步的蒸馏。


2. 训练基于Bi-LSTM的小模型
^^^^^^^^^^^^

在本示例中，小模型采取的是基于双向LSTM的分类模型，网络层分别是 ``Embedding`` 、``LSTM`` 、 带有 ``tanh`` 激活函数的 ``Linear`` 层，最后经过\
一个全连接的输出层得到logits。``LSTM`` 网络层定义如下：

.. code-block::

    self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, 
        'bidirectional', dropout=dropout_prob)

基于Bi-LSTM的小模型的 ``forward`` 函数定义如下：

.. code-block::

    def forward(self, x, seq_len):
        x_embed = self.embedder(x)
        lstm_out, (hidden, _) = self.lstm(
            x_embed, sequence_length=seq_len) # 双向LSTM
        out = paddle.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        out = paddle.tanh(self.fc(out))
        logits = self.output_layer(out)
        
        return logits


3.数据增强介绍
^^^^^^^^^^^^

接下来的蒸馏过程，蒸馏时使用的训练数据集并不只包含数据集中原有的数据，而是按照上文原理介绍中的A、C两种方法进行数据增强后的总数据。
在多数情况下，``alpha`` 会被设置为0，表示无视硬标签，学生模型只利用数据增强后的无标签数据进行训练。根据教师模型提供的软标签 ``teacher_logits`` \
，对比学生模型的 ``logits`` ，计算均方误差损失。由于数据增强过程产生了更多的数据，学生模型可以从教师模型中学到更多的暗知识。

数据增强的核心代码如下：

.. code-block::

    def ngram_sampling(words, words_2=None, p_ng=0.25, ngram_range=(2, 6)):
        if np.random.rand() < p_ng:
            ngram_len = np.random.randint(ngram_range[0], ngram_range[1] + 1)
            ngram_len = min(ngram_len, len(words))
            start = np.random.randint(0, len(words) - ngram_len + 1)
            words = words[start:start + ngram_len]
            if words_2:
                words_2 = words_2[start:start + ngram_len]
        return words if not words_2 else (words, words_2)

    def data_augmentation(data, whole_word_mask=whole_word_mask):
        # 1. Masking
        words = []
        if not whole_word_mask:
            tokenized_list = tokenizer.tokenize(data)
            words = [
                tokenizer.mask_token if np.random.rand() < p_mask else word
                for word in tokenized_list
            ]
        else:
            for word in data.split():
                words += [[tokenizer.mask_token]] if np.random.rand(
                ) < p_mask else [tokenizer.tokenize(word)]
        # 2. N-gram sampling
        words = ngram_sampling(words, p_ng=p_ng, ngram_range=ngram_range)
        words = flatten(words) if isinstance(words[0], list) else words
        new_text = " ".join(words)
        return words, new_text


4.蒸馏模型
^^^^^^^^^^^^

这一步是将教师模型BERT的知识蒸馏到基于Bi-LSTM的学生模型中，在本例中，主要是让学生模型（Bi-LSTM）去学习教师模型的输出logits。\
蒸馏时使用的训练数据集是由上一步数据增强后的数据，核心代码如下：

.. code-block::

    ce_loss = nn.CrossEntropyLoss() # 交叉熵损失函数
    mse_loss = nn.MSELoss() # 均方误差损失函数

    for epoch in range(args.max_epoch):
        for i, batch in enumerate(train_data_loader):
            bert_input_ids, bert_segment_ids, student_input_ids, seq_len, labels = batch

            # Calculate teacher model's forward.
            with paddle.no_grad():
                teacher_logits = teacher.model(bert_input_ids, bert_segment_ids)

            # Calculate student model's forward.
            logits = model(student_input_ids, seq_len)

            # Calculate the loss, usually args.alpha equals to 0.
            loss = args.alpha * ce_loss(logits, labels) + (
                1 - args.alpha) * mse_loss(logits, teacher_logits)

            loss.backward()
            optimizer.step()

