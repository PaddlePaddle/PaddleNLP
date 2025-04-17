# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Any, List, Optional


def calculate_ranking_scores(list_items: List[Any], boost_first_factor: Optional[int] = None) -> List[float]:
    """
    Assigns scores to items in a list based on their rank position and ensures that the scores add up to 1.
    :param list_items: The list of items to score.
    :param boost_first_factor: The factor to boost the score of the first item by.
    """
    n = len(list_items)
    scores = [0.0] * n

    # Compute the scores based on rank position
    for i, _ in enumerate(list_items):
        scores[i] = (n - i) / ((n * (n + 1)) / 2)

    # Apply the boost factor to the first item
    if boost_first_factor is not None and n > 0:
        scores[0] *= boost_first_factor

    # Normalize the scores so they add up to 1
    total_score = sum(scores)
    normalized_scores = [score / total_score for score in scores]

    return normalized_scores
