#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Reader utils."""

import numpy as np


def mask(batch_tokens,
         vocab_size,
         bos_id=1,
         eos_id=2,
         mask_id=3,
         sent_b_starts=None,
         labels=None,
         is_unidirectional=False,
         use_latent=False,
         use_bow=False):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    batch_tokens = np.copy(batch_tokens)
    max_len = max(map(len, batch_tokens))
    mask_label = []
    mask_pos = []
    if labels is not None:
        label_pos = []

    if is_unidirectional:
        # unidirectional language model
        if use_latent:
            max_len += 1
            shift_len = 1
        else:
            shift_len = 0
        for sent_index, sent in enumerate(batch_tokens):
            sent_b_index = sent_b_starts[
                sent_index] if sent_b_starts is not None else 0
            need_cal = True
            if labels is not None:
                label_pos.append(sent_index * max_len + len(sent) - 1 +
                                 shift_len)
                if labels[sent_index] == 0:
                    need_cal = False
            mask_label.extend(sent[sent_b_index + 1:])
            mask_pos.extend([
                sent_index * max_len + i + shift_len
                for i in range(sent_b_index,
                               len(sent) - 1)
            ])
        mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
        return_list = [mask_label, mask_pos]

        # latent related (bow label and pos)
        if use_latent and use_bow:
            bow_label = []
            bow_pos = []
            for sent_index, sent in enumerate(batch_tokens):
                sent_b_index = sent_b_starts[
                    sent_index] if sent_b_starts is not None else 0

                def __filter__(tok_id):
                    # TODO: exclude [EOS] from bow loss
                    return True

                bow_pos.extend([
                    sent_index for i in range(sent_b_index + 1, len(sent))
                    if __filter__(sent[i])
                ])
                bow_label.extend([
                    sent[i] for i in range(sent_b_index + 1, len(sent))
                    if __filter__(sent[i])
                ])
            bow_label = np.array(bow_label).astype("int64").reshape([-1, 1])
            bow_pos = np.array(bow_pos).astype("int64").reshape([-1, 1])
            return_list += [bow_label, bow_pos]
    else:
        # bidirectional mask language model
        total_token_num = sum(map(len, batch_tokens))
        prob_mask = np.random.rand(total_token_num)
        # TODO: fix replace_ids, include [UNK]
        replace_ids = np.random.randint(3,
                                        high=vocab_size,
                                        size=total_token_num)
        prob_index = 0
        for sent_index, sent in enumerate(batch_tokens):
            # add pair label position
            if labels is not None:
                label_pos.append(sent_index * max_len)

            # add mask label and position
            for token_index, token in enumerate(sent):
                if token == eos_id or token == bos_id:
                    continue
                prob = prob_mask[prob_index + token_index]
                if prob > 0.15:
                    continue
                elif 0.03 < prob <= 0.15:
                    # mask
                    mask_label.append(sent[token_index])
                    sent[token_index] = mask_id
                    mask_pos.append(sent_index * max_len + token_index)
                elif 0.015 < prob <= 0.03:
                    # random replace
                    mask_label.append(sent[token_index])
                    sent[token_index] = replace_ids[prob_index + token_index]
                    mask_pos.append(sent_index * max_len + token_index)
                else:
                    # keep the original token
                    mask_label.append(sent[token_index])
                    mask_pos.append(sent_index * max_len + token_index)

            prob_index += len(sent)

        mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
        mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
        return_list = [batch_tokens, mask_label, mask_pos]

    if labels is not None:
        label_pos = np.array(label_pos).astype("int64").reshape([-1, 1])
        assert len(labels) == len(label_pos)
        return_list.append(label_pos)
    return return_list
