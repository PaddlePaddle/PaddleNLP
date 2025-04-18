Using DynaBERT's Strategy to Compress BERT
============

This tutorial employs the training strategy from `DynaBERT-Dynamic BERT with Adaptive Width and Depth <https://arxiv.org/abs/2004.04037>`_. The original model is treated as the largest sub-model within a super-network, where a super-network refers to a network encompassing the entire search space. The original model contains multiple Transformer blocks of identical size. Before each training iteration, a sub-model to be trained in the current round is selected. Each sub-model consists of multiple Sub-Transformer blocks of the same size. Each Sub-Transformer block is derived by selecting different widths from the original Transformer block. A Transformer block contains one Multi-Head Attention and one Feed-Forward Network (FFN). The Sub-Transformer block is obtained as follows:

1. A ``Multi-Head Attention`` layer contains multiple heads. When selecting sub-models of different widths, the number of heads is proportionally reduced. For example: if the original model has 12 heads and the current sub-model's width is 75% of the original, the number of heads in all Transformer blocks becomes 9 during this training phase.

2. The parameter dimensions of the ``Linear`` layers in the ``Feed-Forward Network (FFN)`` are proportionally reduced. For example: if the original FFN layer has a hidden dimension of 3072 and the current sub-model's width is 75% of the original, the FFN's hidden dimension becomes 2304 in all Transformer blocks during this training phase.

Overall Principle
------------

1. **Reordering Parameters and Heads by Importance**: First, parameters and attention heads in the pre-trained model are reordered based on their importance, ensuring critical parameters/heads are prioritized to avoid being pruned during training. Parameter importance is calculated using gradient information from dev data, while head importance is determined by passing an all-ones mask and computing gradients for each head in Multi-Head Attention layers.

2. **Knowledge Distillation Framework**: The original pre-trained model serves as the teacher network. A super-network is defined as the student network, where its largest sub-network shares the same architecture as the teacher. Smaller sub-networks are derived by pruning parameters from the largest network, with all sub-networks sharing parameters during training.

3. **Initialization and Distillation Loss**: The super-network (student) is initialized using reordered parameters from the pre-trained model. Distillation losses are applied to the embedding layer, each transformer block, and the final logits output.

4. **Sub-network Training**: Before processing each batch, a sub-network configuration (currently focusing on width selection) is chosen. Only parameters involved in the current sub-network's computation are updated during backpropagation.

5. **Sub-network Selection**: After training, optimal sub-networks are selected based on both acceleration requirements and accuracy constraints.

.. image:: ../../../examples/model_compression/ofa/imgs/ofa_bert.jpg

.. centered:: Figure: OFA-BERT Architecture Overview
Model Compression with PaddleSlim
--------------------------------

In this example, we also need to train a task-specific BERT model, following the same method as the previous tutorial "Knowledge Distillation from BERT to Bi-LSTM". Here we focus on the model compression process.

1. Define Initial Network
^^^^^^^^^^^^^^^^^^^^^
Define the original BERT-base model and create a dictionary to store the original model parameters. After converting a regular model to a supernet, the original parameters become invalid due to changes in network operators. Therefore, we need to store the original parameters to initialize the supernet.

.. code-block::

    model = BertForSequenceClassification.from_pretrained('bert', num_classes=2)
    origin_weights = {}
    for name, param in model.named_parameters():
        origin_weights[name] = param

2. Build Supernet
^^^^^^^^^^^^^^^^
Define the search space and convert the regular network to a supernet based on this search space.

.. code-block::

    # Define search space
    sp_config = supernet(expand_ratio=[0.25, 0.5, 0.75, 1.0])
    # Convert model to supernet
    model = Convert(sp_config).convert(model)
    paddleslim.nas.ofa.utils.set_state_dict(model, origin_weights)

3. Define Teacher Network
^^^^^^^^^^^^^^^^^^^^^
Construct the teacher network.

.. code-block::

    teacher_model = BertForSequenceClassification.from_pretrained('bert', num_classes=2)

4. Configure Distillation Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Required configurations include:
- Teacher model instance
- Layers for distillation: add distillation loss between the teacher and student networks' `Embedding` layers and each `Transformer Block` layer (using default MSE loss)
- `lambda_distill` parameter to scale the overall distillation loss.

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

5. Define Once-For-All Model
^^^^^^^^^^^^^^^^^^^^^^^^^
Pass the regular model and distillation configurations to `OFA`.

（Note: The original text ends abruptly. The translation continues the final sentence to maintain grammatical completeness while following all specified formatting and technical requirements.)
Interfaces to automatically add the distillation process and convert the supernetwork training approach to the ``OFA`` training approach.

.. code-block::

    ofa_model = paddleslim.nas.ofa.OFA(model, distill_config=distill_config)


6. Compute Neuron and Head Importance and Reorder Parameters Accordingly
^^^^^^^^^^^^

.. code-block::

    head_importance, neuron_importance = utils.compute_neuron_head_importance(
        'sst-2',
        ofa_model.model,
        dev_data_loader,
        num_layers=model.bert.config['num_hidden_layers'],
        num_heads=model.bert.config['num_attention_heads'])
    reorder_neuron_head(ofa_model.model, head_importance, neuron_importance)


7. Set the Current Stage of OFA Training
^^^^^^^^^^^^

.. code-block::

    ofa_model.set_epoch(epoch)
    ofa_model.set_task('width')


8. Configure Network Settings and Start Training
^^^^^^^^^^^^
This example uses DynaBERT's strategy for supernetwork training.

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

Since calculating head importance requires gradient collection via masking, we need to apply a monkey patch to reimplement the ``forward`` method of the ``BERTModel`` class.
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