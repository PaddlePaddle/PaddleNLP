"""
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import paddle.fluid as fluid
import paddle

def textcnn_net_multi_label(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            win_sizes=None,
            is_infer=False,
            threshold=0.5,
            max_seq_len=100):
    """
    multi labels Textcnn_net 
    """
    init_bound = 0.1
    initializer = fluid.initializer.Uniform(low=-init_bound, high=init_bound)
    regularizer = fluid.regularizer.L2DecayRegularizer(
                                        regularization_coeff=1e-4)
    seg_param_attrs = fluid.ParamAttr(name="seg_weight",
                                  learning_rate=640.0,
                                  initializer=initializer,
                                  trainable=True)
    fc_param_attrs_1 = fluid.ParamAttr(name="fc_weight_1",
                                               learning_rate=1.0,
                                               regularizer=regularizer,
                                               initializer=initializer,
                                               trainable=True)
    fc_param_attrs_2 = fluid.ParamAttr(name="fc_weight_2",
                                               learning_rate=1.0,
                                               regularizer=regularizer,
                                               initializer=initializer,
                                               trainable=True)

    if win_sizes is None:
        win_sizes = [1, 2, 3]

    # embedding layer

    emb = fluid.embedding(input=data, size=[dict_dim, emb_dim], param_attr=seg_param_attrs)

    # convolution layer
    convs = []
    for cnt, win_size in enumerate(win_sizes):
        emb = fluid.layers.reshape(x=emb, shape=[-1, 1, max_seq_len, emb_dim], inplace=True)
        filter_size = (win_size, emb_dim)
        cnn_param_attrs = fluid.ParamAttr(name="cnn_weight" + str(cnt),
                                              learning_rate=1.0,
                                              regularizer=regularizer,
                                              initializer=initializer,
                                              trainable=True)
        conv_out = fluid.layers.conv2d(input=emb, num_filters=hid_dim, filter_size=filter_size, act="relu", \
                                    param_attr=cnn_param_attrs)
        pool_out = fluid.layers.pool2d(
                input=conv_out,
                pool_type='max',
                pool_stride=1,
                global_pooling=True)
        convs.append(pool_out)
    convs_out = fluid.layers.concat(input=convs, axis=1)

    # full connect layer
    fc_1 = fluid.layers.fc(input=[pool_out], size=hid_dim2, act=None, param_attr=fc_param_attrs_1)
    # sigmoid layer
    fc_2 = fluid.layers.fc(input=[fc_1], size=class_dim, act=None, param_attr=fc_param_attrs_2)
    prediction = fluid.layers.sigmoid(fc_2)
    if is_infer:
        return prediction

    cost = fluid.layers.sigmoid_cross_entropy_with_logits(x=fc_2, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    pred_label = fluid.layers.ceil(fluid.layers.thresholded_relu(prediction, threshold))
    return [avg_cost, prediction, pred_label, label]
