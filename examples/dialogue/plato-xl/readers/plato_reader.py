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

import paddle

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

    def _pad_batch_records(self, batch_records, dtype="float32"):
        """
        Padding batch records and construct model's inputs.
        """
        batch = {}
        batch_input_ids = [record.input_ids for record in batch_records]
        batch_type_ids = [record.type_ids for record in batch_records]
        batch_pos_ids = [record.pos_ids for record in batch_records]
        if self.use_role:
            batch_role_ids = [record.role_ids for record in batch_records]
        batch_tgt_start_idx = [record.tgt_start_idx for record in batch_records]

        batch_size = len(batch_input_ids)

        batch["seq_len"] = paddle.to_tensor(
            [len(record.input_ids) for record in batch_records], dtype="int32")
        batch["decoder_type_ids"] = paddle.to_tensor(
            [1] * batch_size, dtype="int32").reshape([-1, 1])

        # padding
        batch["input_ids"] = paddle.to_tensor(
            pad_batch_data(
                batch_input_ids, pad_id=self.pad_id))
        batch["type_ids"] = paddle.to_tensor(
            pad_batch_data(
                batch_type_ids, pad_id=self.pad_id))
        batch["pos_ids"] = paddle.to_tensor(
            pad_batch_data(
                batch_pos_ids, pad_id=self.pad_id))
        if self.use_role:
            batch["role_ids"] = paddle.to_tensor(
                pad_batch_data(
                    batch_role_ids, pad_id=self.pad_id))
            batch["decoder_role_ids"] = paddle.to_tensor(
                np.zeros_like(
                    batch_tgt_start_idx, dtype="int32").reshape([-1, 1]))

        batch["attention_mask"] = paddle.to_tensor(
            self._gen_self_attn_mask(
                batch_input_ids,
                batch_tgt_start_idx=batch_tgt_start_idx,
                is_unidirectional=True,
                shift_len=0,
                dtype=dtype))

        if self.position_style == "continuous":
            decoder_position_ids = np.array(batch_tgt_start_idx, dtype="int32")
        else:
            decoder_position_ids = np.zeros_like(
                batch_tgt_start_idx, dtype="int32")
        decoder_position_ids = decoder_position_ids.reshape(-1, 1)
        batch["decoder_position_ids"] = paddle.to_tensor(
            decoder_position_ids.tolist(), dtype="int32")

        batch_data_id = [record.data_id for record in batch_records]
        batch["data_id"] = np.array(batch_data_id).astype("int32").reshape(
            [-1, 1])
        return batch
