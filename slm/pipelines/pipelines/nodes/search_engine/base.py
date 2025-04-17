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

from abc import ABC, abstractmethod
from typing import List, Optional

from pipelines.nodes.search_engine.utils import calculate_ranking_scores
from pipelines.schema import Document


class SearchEngine(ABC):
    """
    Abstract base class for search engines providers.
    """

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Document]:
        """
        Search the search engine for the given query and return the results.
        :param query: The query to search for.
        :param kwargs: Additional parameters to pass to the search engine, such as top_k.
        :return: List of search results as documents.
        """

    def score_results(
        self, results: List[Document], has_answer_box: Optional[bool] = False, boost_factor: Optional[int] = 5
    ) -> List[Document]:
        """
        Assigns scores to search results based on their rank position and ensures that the scores add up to 1.
        """
        scores = calculate_ranking_scores(results, boost_first_factor=boost_factor if has_answer_box else None)
        for doc, score in zip(results, scores):
            doc.score = doc.meta["score"] = score
        return results
