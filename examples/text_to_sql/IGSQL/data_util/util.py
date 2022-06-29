#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Contains various utility functions."""


def subsequence(first_sequence, second_sequence):
    """
    Returns whether the first sequence is a subsequence of the second sequence.

    Args:
        first_sequence (`list`): A sequence.
        second_sequence (`list`): Another sequence.

    Returns:
        `bool`: Whether first_sequence is a subsequence of second_sequence.
    """
    for startidx in range(len(second_sequence) - len(first_sequence) + 1):
        if second_sequence[startidx:startidx +
                           len(first_sequence)] == first_sequence:
            return True
    return False
