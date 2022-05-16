# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
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

import logging
import numbers
from typing import Iterable, List

import numpy as np
from paddlenlp.datasets import IterDataset, MapDataset

from pipelines.utils.common_utils import flatten_list

logger = logging.getLogger(__name__)


def convert_features_to_dataset(features):
    """
    Converts a list of feature dictionaries (one for each sample) into a Paddle Dataset.

    :param features: A list of dictionaries. Each dictionary corresponds to one sample. Its keys are the
                     names of the type of feature and the keys are the features themselves.
    :Return: a Paddle dataset and a list of tensor names.
    """
    # features can be an empty list in cases where down sampling occurs
    if len(features) == 0:
        return None, None
    tensor_names = list(features[0].keys())
    all_tensors = []
    for t_name in tensor_names:
        try:
            # Checking whether a non-integer will be silently converted to Paddle.long
            check = features[0][t_name]
            if isinstance(check, numbers.Number):
                base = check
            # extract a base variable from a nested lists or tuples
            elif isinstance(check, list):
                base = list(flatten_list(check))[0]
            # extract a base variable from numpy arrays
            else:
                base = check.ravel()[0]
            if not np.issubdtype(type(base), np.integer):
                logger.warning(
                    f"Problem during conversion to Paddle tensors:\n"
                    f"A non-integer value for feature '{t_name}' with a value of: "
                    f"'{base}' will be converted to a Paddle tensor of dtype long."
                )
        except:
            logger.debug(f"Could not determine type for feature '{t_name}'. "
                         "Converting now to a tensor of default type long.")

        # Convert all remaining python objects to Paddle long tensors
        cur_tensor = [sample[t_name] for sample in features]
        all_tensors.append(cur_tensor)

    # Todo(tianxin): When set to IterDataset, throw Exception with paddle.io.BatchSampler
    # all_tensors: List[List[all_token_ids], List[all_segment_ids]]
    # list(zip(*all_tensors)): List[([token_ids], [segment_ids]), ([token_ids], [segment_ids])]
    # For Question Answering: tensor_names: ['input_ids', 'padding_mask', 'segment_ids', 'passage_start_t', 'start_of_word', 'labels', 'id', 'seq_2_start_t', 'span_mask']
    dataset = MapDataset(list(zip(*all_tensors)))
    return dataset, tensor_names
