============
模型压缩
============


近些年，基于Transformer的语言模型在机器翻译、阅读理解、文本匹配、自然语言推理等自然语言处理任务上取得了实质的进展。然而，海量的参数和计算资源的大量耗费，使BERT及其变体在部署中困难重重。模型压缩的发展，使得这些问题得到了缓解。下面将会对 `PaddleNLP repo <https:github.com/PaddlePaddle/PaddleNLP>`_ 下模型压缩的示例进行介绍。

一、将Bert的知识蒸馏到Bi-LSTM
============


整体原理介绍
------------

本例是将特定任务下BERT模型的知识蒸馏到基于Bi-LSTM的小模型中，主要参考论文 `Distilling Task-Specific Knowledge from BERT into Simple Neural Networks <https://arxiv.org/abs/1903.12136>`_ 实现。
在模型蒸馏中，较大的模型（在本例中是BERT）通常被称为教师模型，较小的模型（在本例中是Bi-LSTM）通常被称为学生模型。知识的蒸馏通常是通过模型学习蒸馏相关的损失函数实现，在本实验中，损失函数是均方误差损失函数，传入函数的两个参数分别是学生模型的输出和教师模型的输出。
在 `论文 <https://arxiv.org/abs/1903.12136>`_ 的模型蒸馏阶段，作者为了能让教师模型表达出更多的暗知识供学生模型学习，对训练数据进行了数据增强。作者使用了三种数据增强方式，分别是：

1. Masking，即以一定的概率将原数据中的word token替换成 ``[MASK]`` ；

2. POS—guided word replacement，即以一定的概率将原数据中的词用与其有相同POS tag的词替换；

3. n-gram sampling，即以一定的概率，从每条数据中采样n-gram，其中n的范围可通过人工设置。

通过数据增强，可以产生更多无标签的训练数据，在训练过程中，学生模型可借助教师模型的“暗知识”，在更大的数据集上进行训练，产生更好的蒸馏效果。

本实验分为三个训练过程：在特定任务上对BERT进行微调、在特定任务上对基于Bi-LSTM的小模型进行训练（用于评价蒸馏效果）、将BERT模型的知识蒸馏到基于Bi-LSTM的小模型上。


模型蒸馏步骤介绍
------------

1. 基于Bert-base-uncased预训练模型在特定任务上进行微调
^^^^^^^^^^^^

训练BERT的fine-tuning模型，可以去 `PaddleNLP repo <https:github.com/PaddlePaddle/PaddleNLP>`_ 下的examples中找到glue目录

以GLUE的SST-2任务为例，用bert-base-uncased做微调之后，可以得到一个在SST-2任务上的小模型，可以把在dev上取得最好Accuracy的模型保存下来，用于第三步的蒸馏。


2. 训练基于Bi-LSTM的小模型
^^^^^^^^^^^^

在本示例中，小模型采取的是基于双向LSTM的分类模型，网络层分别是 ``Embedding`` 、``LSTM`` 、 带有 ``tanh`` 激活函数的 ``Linear`` 层，最后经过一个全连接的输出层得到logits。``LSTM`` 网络层定义如下：

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


3.蒸馏模型
^^^^^^^^^^^^

这一步是将教师模型BERT的知识蒸馏到基于Bi-LSTM的学生模型中，在本例中，主要是让学生模型（Bi-LSTM）去学习教师模型的输出logits。核心代码如下：

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



数据增强部分介绍
------------

上面蒸馏过程的第三步中，蒸馏时使用的 ``train_data_loader`` 并不只包含数据集中原有的数据，而是按照上文原理介绍中的第1、3种方法进行数据增强后的总数据。
在多数情况下，``alpha`` 会被设置为0，表示无视硬标签，学生模型只利用数据增强后的无标签数据进行训练。根据教师模型提供的软标签 ``teacher_logits`` ，对比学生模型的 ``logits`` ，计算均方误差损失。由于数据增强过程产生了更多的数据，学生模型可以从教师模型中学到更多的暗知识。

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
    

二、使用DynaBERT中的策略蒸馏BERT小模型
============

本例是对BERT模型进行压缩的原理介绍。并以 `PaddleNLP repo <https:github.com/PaddlePaddle/PaddleNLP>`_ 中BERT-base模型为例，说明如何快速把整体压缩流程迁移到其他NLP模型。

本教程使用的是 `DynaBERT-Dynamic BERT with Adaptive Width and Depth <https://arxiv.org/abs/2004.04037>`_ 中的训练策略。把原始模型作为超网络中最大的子模型，原始模型包括多个相同大小的Transformer Block。在每次训练前会选择当前轮次要训练的子模型，每个子模型包含多个相同大小的Sub Transformer Block，每个Sub Transformer Block是选择不同宽度的Transformer Block得到的，一个Transformer Block包含一个Multi-Head Attention和一个Feed-Forward Network，Sub Transformer Block获得方式为：

1. 一个 ``Multi-Head Attention`` 层中有多个Head，每次选择不同宽度的子模型时，会同时对Head数量进行等比例减少，例如：如果原始模型中有12个Head，本次训练选择的模型是宽度为原始宽度75%的子模型，则本次训练中所有Transformer Block的Head数量为9。

2. ``Feed-Forward Network`` 层中 ``Linear`` 的参数大小进行等比例减少，例如：如果原始模型中 ``FFN`` 层的特征维度为3072，本次训练选择的模型是宽度为原始宽度75%的子模型，则本次训练中所有Transformer Block中 ``FFN`` 层的特征维度为2304。


整体原理介绍
------------

1. 首先对预训练模型的参数和head根据其重要性进行重排序，把重要的参数和head排在参数的前侧，保证训练过程中的参数裁剪不会裁剪掉这些重要的参数。参数的重要性计算是先使用dev数据计算一遍每个参数的梯度，然后根据梯度和参数的整体大小来计算当前参数的重要性，head的的重要性计算是通过传入一个全1的对head的mask，并计算这个mask的梯度，根据mask的梯度来判断每个 ``Multi-Head Attention`` 层中每个Head的重要性。
2. 使用原本的预训练模型作为蒸馏过程中的教师网络。同时定义一个超网络，这个超网络中最大的子网络的结构和教师网络的结构相同其他小的子网络是对最大网络的进行不同的宽度选择来得到的，宽度选择具体指的是网络中的参数进行裁剪，所有子网络在整个训练过程中都是参数共享的。
3. 使用重排序之后的预训练模型参数初始化超网络，并把这个超网络作为学生网络。分别为 ``Embedding`` 层，每个transformer block层和最后的logit添加蒸馏损失。
4. 每个batch数据在训练前首先中会选择当前要训练的子网络配置（子网络配置目前仅包括对整个模型的宽度的选择），参数更新时仅会更新当前子网络计算中用到的那部分参数。
5. 通过以上的方式来优化整个超网络参数，训练完成后选择满足加速要求和精度要求的子模型。

.. image:: ../../examples/model_compression/ofa/imgs/ofa_bert.jpg

.. centered:: 整体流程


基于PaddleSlim进行模型压缩
------------

本教程基于PaddleSlim2.0或之后版本，可按如下命令进行安装：

.. code-block::

    pip install paddleslim==2.0.0 -i https://pypi.org/simple


在本例中，也需要训练基于特定任务的BERT模型，方法同上。下面介绍模型压缩的过程。

1. 定义初始网络
^^^^^^^^^^^^
定义原始 ``BERT-base`` 模型并定义一个字典保存原始模型参数。普通模型转换为超网络之后，由于其组网OP的改变导致原始模型加载的参数失效，所以需要定义一个字典保存原始模型的参数并用来初始化超网络。

.. code-block::

    model = BertForSequenceClassification.from_pretrained('bert', num_classes=2)
    origin_weights = {}
    for name, param in model.named_parameters():
        origin_weights[name] = param


2. 构建超网络
^^^^^^^^^^^^
定义搜索空间，并根据搜索空间把普通网络转换为超网络。

.. code-block::

    # 定义搜索空间
    sp_config = supernet(expand_ratio=[0.25, 0.5, 0.75, 1.0])
    # 转换模型为超网络
    model = Convert(sp_config).convert(model)
    paddleslim.nas.ofa.utils.set_state_dict(model, origin_weights)


3. 定义教师网络
^^^^^^^^^^^^
调用PaddleNLP的接口直接构造教师网络。

.. code-block::

    teacher_model = BertForSequenceClassification.from_pretrained('bert', num_classes=2)


4. 配置蒸馏相关参数
^^^^^^^^^^^^
需要配置的参数包括教师模型实例；需要添加蒸馏的层，在教师网络和学生网络的 ``Embedding`` 层和每一个 ``Tranformer Block`` 层之间添加蒸馏损失，中间层的蒸馏损失使用默认的MSE损失函数；配置'`lambda_distill'`参数表示整体蒸馏损失的缩放比例。

.. code-block::

    mapping_layers = ['bert.embeddings']
    for idx in range(model.bert.config['num_hidden_layers']):
        mapping_layers.append('bert.encoder.layers.{}'.format(idx))

    default_distill_config = {
        'lambda_distill': 0.1,
        'teacher_model': teacher_model,
        'mapping_layers': mapping_layers,
    }
    distill_config = DistillConfig(**default_distill_config)


5. 定义Once-For-All模型
^^^^^^^^^^^^
普通模型和蒸馏相关配置传给 ``OFA`` 接口，自动添加蒸馏过程并把超网络训练方式转为 ``OFA`` 训练方式。

.. code-block::

    ofa_model = paddleslim.nas.ofa.OFA(model, distill_config=distill_config)


6. 计算神经元和head的重要性并根据其重要性重排序参数
^^^^^^^^^^^^

.. code-block::

    head_importance, neuron_importance = utils.compute_neuron_head_importance(
        'sst-2',
        ofa_model.model,
        dev_data_loader,
        num_layers=model.bert.config['num_hidden_layers'],
        num_heads=model.bert.config['num_attention_heads'])
    reorder_neuron_head(ofa_model.model, head_importance, neuron_importance)


7. 传入当前OFA训练所处的阶段
^^^^^^^^^^^^

.. code-block::

    ofa_model.set_epoch(epoch)
    ofa_model.set_task('width')


8. 传入网络相关配置，开始训练
^^^^^^^^^^^^
本示例使用DynaBERT的方式进行超网络训练。

.. code-block::

    width_mult_list = [1.0, 0.75, 0.5, 0.25]
    lambda_logit = 0.1
    for width_mult in width_mult_list:
        net_config = paddleslim.nas.ofa.utils.dynabert_config(ofa_model, width_mult)
        ofa_model.set_net_config(net_config)
        logits, teacher_logits = ofa_model(input_ids, segment_ids, attention_mask=[None, None])
        rep_loss = ofa_model.calc_distill_loss()
        logit_loss = soft_cross_entropy(logits, teacher_logits.detach())
        loss = rep_loss + lambda_logit * logit_loss
        loss.backward()
    optimizer.step()
    lr_scheduler.step()
    ofa_model.model.clear_gradients()



**NOTE**

由于在计算head的重要性时会利用一个mask来收集梯度，所以需要通过monkey patch的方式重新实现一下 ``BERTModel`` 类的 ``forward`` 函数。示例如下:

.. code-block::

    from paddlenlp.transformers import BertModel
    def bert_forward(self,
                    input_ids,
                    token_type_ids=None,
                    position_ids=None,
                    attention_mask=[None, None]):
        wtype = self.pooler.dense.fn.weight.dtype if hasattr(
            self.pooler.dense, 'fn') else self.pooler.dense.weight.dtype
        if attention_mask[0] is None:
            attention_mask[0] = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(wtype) * -1e9, axis=[1, 2])
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


    BertModel.forward = bert_forward
