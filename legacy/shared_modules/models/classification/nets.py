"""
This module provide nets for text classification
"""

import paddle.fluid as fluid


def bow_net(data,
            seq_len,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            is_prediction=False):
    """
    Bow net
    """
    # embedding layer
    emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
    emb = fluid.layers.sequence_unpad(emb, length=seq_len)
    # bow layer
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    # full connect layer
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    # softmax layer
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    if is_prediction:
        return prediction
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, prediction


def cnn_net(data,
            seq_len,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            win_size=3,
            is_prediction=False):
    """
    Conv net
    """
    # embedding layer
    emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
    emb = fluid.layers.sequence_unpad(emb, length=seq_len)
    # convolution layer
    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=win_size,
        act="tanh",
        pool_type="max")

    # full connect layer
    fc_1 = fluid.layers.fc(input=[conv_3], size=hid_dim2)
    # softmax layer
    prediction = fluid.layers.fc(input=[fc_1], size=class_dim, act="softmax")
    if is_prediction:
        return prediction
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, prediction


def lstm_net(data,
             seq_len,
             label,
             dict_dim,
             emb_dim=128,
             hid_dim=128,
             hid_dim2=96,
             class_dim=2,
             emb_lr=30.0,
             is_prediction=False):
    """
    Lstm net
    """
    # embedding layer
    emb = fluid.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))
    emb = fluid.layers.sequence_unpad(emb, length=seq_len)
    # Lstm layer
    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)

    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)

    # max pooling layer
    lstm_max = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    lstm_max_tanh = fluid.layers.tanh(lstm_max)

    # full connect layer
    fc1 = fluid.layers.fc(input=lstm_max_tanh, size=hid_dim2, act='tanh')
    # softmax layer
    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')
    if is_prediction:
        return prediction
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, prediction


def bilstm_net(data,
               seq_len,
               label,
               dict_dim,
               emb_dim=128,
               hid_dim=128,
               hid_dim2=96,
               class_dim=2,
               emb_lr=30.0,
               is_prediction=False):
    """
    Bi-Lstm net
    """
    # embedding layer
    emb = fluid.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))

    emb = fluid.layers.sequence_unpad(emb, length=seq_len)

    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)
    rfc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)
    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)
    rlstm_h, c = fluid.layers.dynamic_lstm(
        input=rfc0, size=hid_dim * 4, is_reverse=True)
    # extract last layer
    lstm_last = fluid.layers.sequence_last_step(input=lstm_h)
    rlstm_last = fluid.layers.sequence_last_step(input=rlstm_h)
    lstm_last_tanh = fluid.layers.tanh(lstm_last)
    rlstm_last_tanh = fluid.layers.tanh(rlstm_last)

    # concat layer
    lstm_concat = fluid.layers.concat(input=[lstm_last, rlstm_last], axis=1)
    # full connect layer
    fc1 = fluid.layers.fc(input=lstm_concat, size=hid_dim2, act='tanh')
    # softmax layer
    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')
    if is_prediction:
        return prediction
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, prediction


def gru_net(data,
            seq_len,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            emb_lr=30.0,
            is_prediction=False):
    """
    gru net
    """
    emb = fluid.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))
    emb = fluid.layers.sequence_unpad(emb, length=seq_len)
    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 3)

    gru_h = fluid.layers.dynamic_gru(input=fc0, size=hid_dim, is_reverse=False)

    gru_max = fluid.layers.sequence_pool(input=gru_h, pool_type='max')
    gru_max_tanh = fluid.layers.tanh(gru_max)

    fc1 = fluid.layers.fc(input=gru_max_tanh, size=hid_dim2, act='tanh')

    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')
    if is_prediction:
        return prediction
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, prediction


def textcnn_net(data,
                seq_len,
                label,
                dict_dim,
                emb_dim=128,
                hid_dim=128,
                hid_dim2=96,
                class_dim=2,
                win_sizes=None,
                is_prediction=False):
    """
    Textcnn_net
    """
    if win_sizes is None:
        win_sizes = [1, 2, 3]

    # embedding layer
    emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
    emb = fluid.layers.sequence_unpad(emb, length=seq_len)
    # convolution layer
    convs = []
    for win_size in win_sizes:
        conv_h = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=hid_dim,
            filter_size=win_size,
            act="tanh",
            pool_type="max")
        convs.append(conv_h)
    convs_out = fluid.layers.concat(input=convs, axis=1)

    # full connect layer
    fc_1 = fluid.layers.fc(input=[convs_out], size=hid_dim2, act="tanh")
    # softmax layer
    prediction = fluid.layers.fc(input=[fc_1], size=class_dim, act="softmax")
    if is_prediction:
        return prediction

    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, prediction


def ernie_base_net(sentence_embeddings, labels, num_labels):
    """
    Ernie base net
    """
    cls_feats = fluid.layers.dropout(
        x=sentence_embeddings,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        input=cls_feats,
        size=num_labels,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))
    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)

    return ce_loss, probs


def ernie_bilstm_net(token_embeddings, labels, num_labels, hid_dim=128):
    """
    Ernie bilstm net
    """
    fc0 = fluid.layers.fc(input=token_embeddings, size=hid_dim * 4)
    rfc0 = fluid.layers.fc(input=token_embeddings, size=hid_dim * 4)
    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)
    rlstm_h, c = fluid.layers.dynamic_lstm(
        input=rfc0, size=hid_dim * 4, is_reverse=True)
    # extract last layer
    lstm_last = fluid.layers.sequence_last_step(input=lstm_h)
    rlstm_last = fluid.layers.sequence_last_step(input=rlstm_h)
    lstm_last_tanh = fluid.layers.tanh(lstm_last)
    rlstm_last_tanh = fluid.layers.tanh(rlstm_last)

    # concat layer
    lstm_concat = fluid.layers.concat(input=[lstm_last, rlstm_last], axis=1)
    # full connect layer
    logits = fluid.layers.fc(input=lstm_concat, size=num_labels, act='relu')

    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)

    return ce_loss, probs
