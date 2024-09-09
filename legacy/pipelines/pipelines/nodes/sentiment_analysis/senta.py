# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import json
import logging
import os
from typing import List

from pipelines.nodes.base import BaseComponent

from paddlenlp import Taskflow

logger = logging.getLogger(__name__)


class UIESenta(BaseComponent):
    """
    Senta: sentiment analysis for user's comments based on Taskflow
    """

    outgoing_edges = 1

    def __init__(
        self,
        model,
        schema,
        task="sentiment_analysis",
        aspects=None,
        max_seq_len=512,
        batch_size=1,
        split_sentence=False,
        position_prob=0.5,
        lazy_load=False,
        num_workers=0,
        use_fast=False,
        **kwargs
    ):
        """
        Init UIESenta for Sentiment Analysis.
        :param model: the model name that you wanna use, you can choose it in [use-base, uie-medium, uie-micro, uie-mini, uie-nano].
        :param schema: the schema for extracting sentiment information with UIE.
        :param task: the task name, you should set to be `sentiment_analysis` for the current task.
        :param aspects: optional, a list of pre-given aspects, that is to say, Taskflow only perform sentiment analysis on these pre-given aspects if you input it.
        :param max_seq_len: The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
        :param batch_size: Number of samples the model receives in one batch for inference.
        :param split_sentence: If True, sentence-level split will be performed on the inputing examples.
        :param position_prob: Probability threshold for start/end index probabiliry.
        :param lazy_load: whether to using `MapDataset` or an `IterDataset` when performing inference with UIE. True for `IterDataset`. False for `MapDataset`.
        :num_workers: the number of subprocess to load data for dataloader, 0 for no subprocess used and loading data in main process. Default 0.
        :use_fast: whether to fast tokenizer for UIE.
        """

        self.set_config(
            model=model,
            schema=schema,
            task=task,
            aspects=aspects,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            split_sentence=split_sentence,
            position_prob=position_prob,
            lazy_load=lazy_load,
            num_workers=num_workers,
            use_fast=use_fast,
            **kwargs,
        )

        self.senta = Taskflow(**self.pipeline_config["params"])

    def _predict(self, examples: str):
        return self.senta(examples)

    def _save_json_file(self, save_path: str, results: List[dict]):
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for result in results:
                line = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(line)

    def run(self, examples: List[str], sr_save_path: str):
        # predict with taskflow
        results = self._predict(examples)
        # save the result of sentiment analysis
        if sr_save_path:
            self._save_json_file(sr_save_path, results)
            logger.info("The result of sentiment analysis has been saved to : {}".format(sr_save_path))
        outputs = {"sr_save_path": sr_save_path}
        return outputs, "output_1"
