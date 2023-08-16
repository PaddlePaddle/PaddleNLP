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

import logging
from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union

from pipelines.nodes.base import BaseComponent

logger = logging.getLogger(__name__)


class BaseCombineDocuments(BaseComponent):
    outgoing_edges = 1

    def __init__(self, input_key="contens"):
        self.input_key = input_key
        """
        :param input_key: the key values corresponding to document content
        """

    def prompt_length(self, docs: List[dict], **kwargs: Any) -> Optional[int]:
        return None

    @abstractmethod
    def combine_docs(self, docs: List[dict], **kwargs: Any) -> Tuple[str, dict]:
        """Combine documents into a single string.

        Args:
            docs: List[Document], the documents to combine
            **kwargs: Other parameters to use in combining documents, often
                other inputs to the prompt.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """

    def run(
        self,
        documents: Union[dict, List[dict]],
        **kwargs,
    ) -> Tuple[dict, str]:
        """
        :param documents: Documents used for multi document summary generation
        """
        # Other keys are assumed to be needed for LLM prediction
        output, _ = self.combine_docs(documents)
        return output, "output_1"
