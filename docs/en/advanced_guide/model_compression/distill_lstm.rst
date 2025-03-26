# Knowledge Distillation from BERT to Bi-LSTM

## Overall Principle Introduction

This example demonstrates knowledge distillation from a BERT model to a smaller Bi-LSTM-based model for specific tasks, primarily referencing the paper [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136). The core principles are as follows:

1. In this case, the larger model (BERT) serves as the teacher model, while the Bi-LSTM model acts as the student model.

2. For knowledge transfer, the student model learns through a distillation-related loss function. In this experiment, we employ the Mean Squared Error (MSE) loss function, where the inputs are the outputs from both the student and teacher models.

3. During the model distillation phase, the authors apply data augmentation to enable the teacher model to express more "dark knowledge" (referring to relationships between low-probability and high-probability classes in classification tasks) for the student model to learn. Three data augmentation strategies are used:

   A. **Masking**: Randomly replaces word tokens with `[MASK]` at a specified probability.

   B. **POS-guided word replacement**: Substitutes words with others sharing the same POS tag at a defined probability.

   C. **n-gram sampling**: Samples n-grams from each data instance at a specified probability, with n ranging within manually set bounds.

## Model Distillation Steps Introduction

This experiment involves three training phases: 
1. Fine-tuning BERT on the specific task.
2. Training the Bi-LSTM-based small model on the same task (to evaluate distillation effectiveness).
3. Distilling knowledge from BERT into the Bi-LSTM-based model.

### 1. Fine-tuning BERT (bert-base-uncased) on the Specific Task

To train the fine-tuned BERT model, refer to the [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) repository, particularly the [GLUE example](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/benchmark/glue). 

Taking the SST-2 task from GLUE as an example, after fine-tuning bert-base-uncased, we obtain a teacher model. The checkpoint with the best Accuracy on the dev set should be saved for distillation in the third step.

### 2. Training the Bi-LSTM-based Small Model

In this example, the small model is a Bi-LSTM-based classification network with the following architecture: `Embedding` → `LSTM` → `Linear` layer with `tanh` activation → fully connected output layer for logits. The `LSTM` layer is defined as:

```python
self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, 
    'bidirectional', dropout=dropout_prob)
```
The ``forward`` function is defined as follows:

.. code-block::

    def forward(self, x, seq_len):
        x_embed = self.embedder(x)
        lstm_out, (hidden, _) = self.lstm(
            x_embed, sequence_length=seq_len) # Bidirectional LSTM
        out = paddle.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        out = paddle.tanh(self.fc(out))
        logits = self.output_layer(out)
        
        return logits


3. Data Augmentation Introduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the subsequent distillation process, the training dataset used for distillation doesn't only contain the original data from the dataset, but rather the total data augmented according to methods A and C described in the previous principles.
In most cases, ``alpha`` is set to 0, indicating that hard labels are ignored, and the student model is trained using only the augmented unlabeled data. Based on the soft labels ``teacher_logits`` provided by the teacher model, the student model's ``logits`` are compared.
Calculate the mean squared error loss. Since the data augmentation process generates more data, the student model can learn additional dark knowledge from the teacher model.

The core code for data augmentation is as follows:

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


4. Distillation Model
^^^^^^^^^^^^^^^^^^^^^

This step involves distilling the knowledge from teacher model BERT to the Bi-LSTM based student model. The key aspect is to make the student model (Bi-LSTM) learn the output logits from the teacher model. The training dataset used for distillation comes from the data augmented in the previous step. The core code is as follows:

.. code-block::

    ce_loss = nn.CrossEntropyLoss() # Cross-entropy loss
    mse_loss = nn.MSELoss() # Mean squared error loss

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