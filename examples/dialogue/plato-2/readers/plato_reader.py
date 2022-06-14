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
"""Plato Reader."""

import numpy as np

from readers.dialog_reader import DialogReader
from utils import pad_batch_data
from utils.masking import mask


class PlatoReader(DialogReader):
    """The implement of PlatoReader"""

    def __init__(self, args):
        super(PlatoReader, self).__init__(args)
        self.latent_type_size = args.latent_type_size
        self.use_bow = args.use_bow

    def _pad_batch_records(self, batch_records, is_infer):
        """
        Padding batch records and construct model's inputs.
        """
        batch = {}
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_type_ids = [record.type_ids for record in batch_records]
        batch_pos_ids = [record.pos_ids for record in batch_records]

        batch_tgt_start_idx = [record.tgt_start_idx for record in batch_records]

        batch_size = len(batch_token_ids)

        # padding
        batch["token_ids"] = pad_batch_data(batch_token_ids, pad_id=self.pad_id)
        batch["type_ids"] = pad_batch_data(batch_type_ids, pad_id=self.pad_id)
        batch["pos_ids"] = pad_batch_data(batch_pos_ids, pad_id=self.pad_id)

        batch["generation_mask"] = self._gen_self_attn_mask(
            batch_token_ids,
            batch_tgt_start_idx=batch_tgt_start_idx,
            is_unidirectional=True,
            shift_len=1)
        if not is_infer:
            batch["recognition_mask"] = self._gen_self_attn_mask(
                batch_token_ids, is_unidirectional=False, shift_len=1)

        if is_infer:
            tgt_ids = np.array([[[self.bos_id]]] * batch_size, dtype="int64")
            if self.continuous_position:
                tgt_pos = np.array(batch_tgt_start_idx, dtype="int64")
            else:
                tgt_pos = np.zeros_like(batch_tgt_start_idx, dtype="int64")
            tgt_pos = tgt_pos.reshape(-1, 1, 1)
            batch["init_score"] = np.zeros_like(
                tgt_ids, dtype="float32").reshape(-1, 1).tolist()
            batch["tgt_ids"] = tgt_ids.tolist()
            batch["tgt_pos"] = tgt_pos.tolist()
            batch["parent_idx"] = np.array(range(batch_size), dtype="int32")

            batch["tgt_generation_mask"] = batch[
                "generation_mask"][:, 0:1, :].astype("float32")
        else:
            mask_return_list = mask(batch_tokens=batch_token_ids,
                                    vocab_size=self.vocab_size,
                                    sent_b_starts=batch_tgt_start_idx,
                                    is_unidirectional=True,
                                    use_latent=True,
                                    use_bow=self.use_bow)
            batch["tgt_label"] = mask_return_list[0]
            batch["tgt_pos"] = mask_return_list[1]
            if self.use_bow:
                batch["bow_label"] = mask_return_list[2]
                batch["bow_pos"] = mask_return_list[3]

        batch_data_id = [record.data_id for record in batch_records]
        batch["data_id"] = np.array(batch_data_id).astype("int64").reshape(
            [-1, 1])
        return batch
