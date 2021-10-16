# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import io

import paddle
import paddle.fluid.core as core
import paddle.nn as nn

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import in_dygraph_mode
from paddlenlp.ops import to_vocab_tensor

__all__ = ["FastTokenizer"]


def load_vocabulary(filepath):
    """
    load vocab
    """
    token_to_idx = {}
    with io.open(filepath, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.rstrip('\n')
            token_to_idx[token] = int(index)
    return token_to_idx


class FasterTokenizer(nn.Layer):
    def __init__(self, vocab_path):
        super(FasterTokenizer, self).__init__()
        vocab_dict = load_vocabulary(vocab_path)
        vocab_tensor = to_vocab_tensor(vocab_dict, "vocab")
        self.register_buffer("vocab", vocab_tensor, persistable=True)

    def forward(self,
                text,
                text_pair=None,
                max_seq_len=-1,
                is_split_into_words=False,
                pad_to_max_seq_len=False):
        if in_dygraph_mode():
            input_ids, seg_ids = core.ops.bert_tokenizer(
                self.vocab, text, text_pair, "max_seq_len", max_seq_len,
                "pad_to_max_seq_len", pad_to_max_seq_len, "is_split_into_words",
                is_split_into_words)
            return input_ids, seg_ids

        attrs = {
            "max_seq_len": max_seq_len,
            "pad_to_max_seq_len": pad_to_max_seq_len,
            "is_split_into_words": is_split_into_words,
        }
        helper = LayerHelper("bert_tokenizer")
        input_ids = helper.create_variable_for_type_inference(dtype="int64")
        seg_ids = helper.create_variable_for_type_inference(dtype="int64")
        if text_pair is None:
            helper.append_op(
                type='bert_tokenizer',
                inputs={'Vocab': self.vocab,
                        'Text': text},
                outputs={'InputIds': input_ids,
                         'SegmentIds': seg_ids},
                attrs=attrs)
        else:
            helper.append_op(
                type='tokenizer',
                inputs={
                    'Vocab': self.vocab,
                    'Text': text,
                    'TextPair': text_pair
                },
                outputs={'InputIds': input_ids,
                         'SegmentIds': seg_ids},
                attrs=attrs)
        return input_ids, seg_ids


if __name__ == "__main__":

    paddle.set_device("cpu")
    t = FastTokenizer(
        "/root/.paddlenlp/models/bert-base-chinese/bert-base-chinese-vocab.txt")
    text = [
        '6日晚上9点左右入住，同时在前台办理入住的还有另外两对。结果都被告知要出示结婚证，如果没有结婚证那么两个人的身份证上的地址必须相同，如果没有证身份证地址也不相同，就无法办理两个人入住一间房。当场六个人全部傻眼。无论怎么说出门都没有带结婚证的习惯而且订房的时候从来也没有要求出示结婚证，前台的那个服务员就是不松口。后来是有一位GG绕了好几圈找了个当地的熟人，然后给那个服务员打了个电话，保证半夜警察如果来查房的时候一切责任自负后，才算开始给我们办理。折腾到10点才住进4楼最角落的海景大床房，结果厕所的抽水马桶是坏的，服务员来弄了半天也没弄好，说要叫维修人员，吓得我们连忙说算了算了，这大晚上又坐了长途车，谁有精力再去对付维修人员啊。安置好，下楼，大堂里有潮汐时刻表，还挺管用。 绕到旁边的餐厅的那一头下到沙滩。夜里海边风大，海水也冰冷，穿了夹克犹觉得冷嗖嗖的，本来还想明天下游泳的念头也被风吹散了。在旁边假日酒店(Holiday Inn)的后门，发现一家还在营业的酒店，厨师和店里的人坐在门口打麻将。我们去，点了辣炒花蛤（15/斤）、清蒸海螺（25/斤，正好6个）、皮皮虾（也叫琵琶虾，15/斤）、花生米、皮蛋，后来还加了一个葱爆海参，一瓶青岛纯生，吃的那叫一个美味，最后才123块，真是大顺！强烈推荐这家酒店，因为事实证明，这是我们此行中吃的最价廉物美的海鲜，而且还是在5星级的假日酒店后门的沙滩上！ 回来一夜无话，除了阳台上的门总是关不紧，我们不得不拿单人椅顶住阳台门防止隔壁的旅伴梦游翻墙而过，呵呵。但是早上醒来后发现门还是开着。。。。幸亏7点就被外面光光光的扔钢筋的噪音吵醒了。入住的时候被告知酒店正在装修，可能会有影响。当时以为只是内部的装修，而且不会在客人休息的清晨进行。没想到装修是指在我们的阳台紧外面搭建一个建筑！而且工人们才不管你是不是在睡觉，天亮就开始干活了。于是很郁闷地起来，一边安慰自己说就当叫醒服务吧。 不管怎么样，从阳台看出去的海，还可以，而且侧头就可以和住你隔壁的旅友轻声细语，呵呵。夜晚安静的时候可以听见海浪的哗哗声，和隔壁的电视机声。 从服务、档次、设施、水平来说，这个酒店不值得ctrip上面的365/晚的价格。'
    ]
    text_pair = ["测试测试", "测试测试"]

    input_ids, token_type_ids = t(text)
    input_ids = input_ids.numpy()[0]
    token_type_ids = token_type_ids.numpy()[0]
    import numpy as np
    test = BertTokenizer.from_pretrained("bert-base-chinese")
    encoded_inputs = test(text[0])
    test_input_ids = np.array(encoded_inputs["input_ids"])
    test_token_type_ids = np.array(encoded_inputs["token_type_ids"])

    print(input_ids)
    print(test_input_ids)

    print(np.array_equal(input_ids, test_input_ids))
    print(np.array_equal(token_type_ids, test_token_type_ids))

    print()
