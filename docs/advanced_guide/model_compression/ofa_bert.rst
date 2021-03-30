使用DynaBERT中的策略对BERT进行压缩
============

本教程使用的是 `DynaBERT-Dynamic BERT with Adaptive Width and Depth <https://arxiv.org/abs/2004.04037>`_ 中的训练策略。\
把原始模型作为超网络中最大的子模型，这里超网络指的是包含所有搜索空间在内的一个网络。\
原始模型包括多个相同大小的Transformer Block。在每次训练前会选择当前轮次要训练的子模型，\
每个子模型包含多个相同大小的Sub Transformer Block，每个Sub Transformer Block是选择不同宽度的Transformer Block得到的，\
一个Transformer Block包含一个Multi-Head Attention和一个Feed-Forward Network，Sub Transformer Block获得方式为：

1. 一个 ``Multi-Head Attention`` 层中有多个Head，每次选择不同宽度的子模型时，会同时对Head数量进行等比例减少，\
例如：如果原始模型中有12个Head，本次训练选择的模型是宽度为原始宽度75%的子模型，则本次训练中所有Transformer Block的Head数量为9。

2. ``Feed-Forward Network`` 层中 ``Linear`` 的参数大小进行等比例减少，例如：如果原始模型中 ``FFN`` 层的特征维度为3072，\
本次训练选择的模型是宽度为原始宽度75%的子模型，则本次训练中所有Transformer Block中 ``FFN`` 层的特征维度为2304。


整体原理介绍
------------

1. 首先对预训练模型的参数和head根据其重要性进行重排序，把重要的参数和head排在参数的前侧，保证训练过程中的参数裁剪不会裁剪掉这些重要的参数。\
参数的重要性计算是先使用dev数据计算一遍每个参数的梯度，然后根据梯度和参数的整体大小来计算当前参数的重要性，head的的重要性计算是通过传入一个\
全1的对head的mask，并计算这个mask的梯度，根据mask的梯度来判断每个 ``Multi-Head Attention`` 层中每个Head的重要性。

2. 使用原本的预训练模型作为蒸馏过程中的教师网络。同时定义一个超网络，这个超网络中最大的子网络的结构和教师网络的结构相同其他小的子网络是对最大网络\
进行不同的宽度选择来得到的，宽度选择具体指对网络中的参数进行裁剪，所有子网络在整个训练过程中都是参数共享的。

3. 使用重排序之后的预训练模型参数初始化超网络，并把这个超网络作为学生网络。分别为 ``Embedding`` 层，为每个transformer block层和最后的logits添加蒸馏损失。

4. 每个batch数据在训练前首先会选择当前要训练的子网络配置（子网络配置目前仅包括对整个模型的宽度的选择），参数更新时仅会更新当前子网络计算中用到的那部分参数。

5. 通过以上的方式来优化整个超网络参数，训练完成后选择满足加速要求和精度要求的子模型。

.. image:: ../../../examples/model_compression/ofa/imgs/ofa_bert.jpg

.. centered:: 整体流程


基于PaddleSlim进行模型压缩
------------

在本例中，也需要训练基于特定任务的BERT模型，方法同上一篇教程《由BERT到Bi-LSTM的知识蒸馏》。下面重点介绍本例模型压缩的过程。

1. 定义初始网络
^^^^^^^^^^^^
定义原始BERT-base模型并定义一个字典保存原始模型参数。普通模型转换为超网络之后，由于其组网OP的改变导致原始模型加载的参数失效，所以需要定义一个字典保存原始模型的参数并用来初始化超网络。

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
构造教师网络。

.. code-block::

    teacher_model = BertForSequenceClassification.from_pretrained('bert', num_classes=2)


4. 配置蒸馏相关参数
^^^^^^^^^^^^
需要配置的参数包括教师模型实例；需要添加蒸馏的层，在教师网络和学生网络的 ``Embedding`` 层和每一个 ``Tranformer Block`` 层\
之间添加蒸馏损失，中间层的蒸馏损失使用默认的MSE损失函数；配置 ``lambda_distill`` 参数表示整体蒸馏损失的缩放比例。

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
本示例使用DynaBERT的策略进行超网络训练。

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
