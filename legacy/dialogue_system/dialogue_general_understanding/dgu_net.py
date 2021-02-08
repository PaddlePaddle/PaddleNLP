# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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
"""Create model for dialogue task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid

from dgu.bert import BertModel
from dgu.utils.configure import JsonConfig


def create_net(is_training, model_input, num_labels, paradigm_inst, args):
    """create dialogue task model"""

    src_ids = model_input.src_ids
    pos_ids = model_input.pos_ids
    sent_ids = model_input.sent_ids
    input_mask = model_input.input_mask
    labels = model_input.labels

    assert isinstance(args.bert_config_path, str)

    bert_conf = JsonConfig(args.bert_config_path)
    bert = BertModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=bert_conf,
        use_fp16=False)

    params = {
        'num_labels': num_labels,
        'src_ids': src_ids,
        'pos_ids': pos_ids,
        'sent_ids': sent_ids,
        'input_mask': input_mask,
        'labels': labels,
        'is_training': is_training
    }

    results = paradigm_inst.paradigm(bert, params)
    return results
