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
"""Utils."""

from itertools import chain
import numpy as np
import paddle


def repeat_array(array, times):
    """Repeate numpy array."""
    if isinstance(array, list):
        return list(chain(*([array] * times)))
    else:
        return np.concatenate([array] * times, axis=0)


def gen_inputs(inputs, latent_type_size):
    batch_size = len(inputs["data_id"])
    new_bsz = batch_size * latent_type_size
    inputs = {
        name: repeat_array(array, latent_type_size)
        for name, array in inputs.items()
    }
    # Add latent_id
    inputs["latent_id"] = np.array(
        [i for i in range(latent_type_size) for _ in range(batch_size)],
        dtype="int64").reshape([-1, 1])

    #print('\nplato_inputs:')
    for key in inputs:
        inputs[key] = paddle.to_tensor(inputs[key])
        if key in [
                'token_ids', 'type_ids', 'pos_ids', 'tgt_ids', 'tgt_pos',
                'data_id'
        ]:
            inputs[key] = paddle.squeeze(inputs[key], axis=-1)
        #print(key, inputs[key].shape, inputs[key].dtype)
    return inputs


def pad_batch_data(insts, pad_id=0):
    """Pad the instances to the max sequence length in batch. """
    max_len = max(map(len, insts))
    inst_data = np.array(
        [list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts])
    return inst_data.astype("int64").reshape([-1, max_len, 1])
