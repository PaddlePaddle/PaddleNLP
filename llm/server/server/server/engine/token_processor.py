# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import threading
import time
import traceback
from collections import Counter
from datetime import datetime

import numpy as np
from paddlenlp_ops import get_output
from server.utils import datetime_diff, model_server_logger, monitor_logger


class TokenProcessor(object):
    """
    get Token/Score from Paddle inference engine
    """
    def __init__(self, cfg):
        import paddle
        paddle.device.set_device("cpu")
        self.cfg = cfg
        self.resource_manager = None
        # record all tokens for each request
        self.all_tokens = [[] for _ in range(self.cfg.max_batch_size)]

        self.tokens_counter = Counter()
        self.output_tokens = paddle.full(shape=[self.cfg.max_batch_size + 2, 1], fill_value=2, dtype="int64")
        self.worker = None

        self.record_time_interval = int(os.getenv("RECORD_TIME_INTERVAL", "600"))
        assert self.record_time_interval < 3600, "The RECORD_TIME_INTERVAL cannot exceed 3600."
        self.statics_start_time = time.time()
        self.number_of_tasks = 0
        self.number_of_input_tokens = 0
        self.number_of_output_tokens = 0

    def set_resource_manager(self, resource_manager):
        """
        set ResourceManager

        Args:
            resource_manager (ResourceManager)
        """
        assert self.resource_manager is None, "The resource manager is not None, cannot set again."
        self.resource_manager = resource_manager

    def run(self):
        """
        start thread to get tokens
        """
        assert self.resource_manager is not None, "The resource manager is None, cannot run."
        if self.worker is not None:
            raise Exception("Worker is already running!")

        self.worker = threading.Thread(target=self.process_sampling_results, args=())
        self.worker.daemon = True
        self.worker.start()

    def process_sampling_results(self):
        """
        read tokens from paddle inference engine and process
        """
        while True:
            try:
                rank_id = 0
                is_blocking = True
                get_output(self.output_tokens, rank_id, is_blocking)

                if self.output_tokens[0, 0] == -2:
                    continue
                self._process_batch_output()
            except Exception as e:
                model_server_logger.info("while get input_data error: {0} {1}".format(e, str(traceback.format_exc())))

    def postprocess(self, batch_result, exist_finished_task=False):
        """
        single post-processing function

        Args:
            batch_result (list): batch results
            exist_finished_task (bool): whether there is a finished task
        """
        result_dir = "./generate_token_results"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for result in batch_result:
            result_file = os.path.join(result_dir, result["req_id"])
            with open(result_file, "a") as f:
                f.write("{}\n".format(result))

    def _get_single_result(self, i, task_id, token_id, task):
        """
        processing single results

        Args:
            i (int): batch index
            task_id (str): task id
            token_id (int): token id
            task (dict): task information

        Returns:
            dict: result
        """
        inference_time_cost = time.time() - task["inference_start_time"]
        task["inference_time_cost"] = inference_time_cost
        task["tokens_all_num"] = len(self.all_tokens[i])
        task["inference_current_step_time"] = datetime.now()
        result = {
            "req_id": task_id,
            "is_end": 0,
            "token_ids": [token_id],
            "send_idx": self.tokens_counter[task_id],
            "inference_time_cost": inference_time_cost,
            "infer_seed": task["infer_seed"],
            "return_all_tokens": task.get("return_all_tokens", False),
        }

        # get benchmark msg
        if task.get("benchmark"):
            keys = ["preprocess_start_time", "preprocess_end_time", "schedule_start_time",
                    "inference_start_time", "inference_current_step_time"]
            for key in keys:
                if key in task:
                    result[key] = str(task[key])

        # fill some extra information
        if token_id in task["eos_token_ids"]:
            result["is_end"] = 1
            result["token_ids"] = []
            result["tokens_all_num"] = len(self.all_tokens[i]) + 1
            result["tokens_all_ids"] = self.all_tokens[i]

            info_dict = {}
            info_dict["req_id"] = task["req_id"]
            info_dict["input_token_num"] = len(task["input_ids"])
            info_dict["output_token_num"] = len(self.all_tokens[i])
            if hasattr(task, "preprocess_start_time") and hasattr(task, "preprocess_end_time"):
                info_dict["preprocess_cost_time"] = datetime_diff(task["preprocess_start_time"],
                                                                  task["preprocess_end_time"])
            if hasattr(task, "preprocess_end_time") and hasattr(task, "schedule_start_time"):
                info_dict["cache_waiting_cost_time"] = datetime_diff(task["preprocess_end_time"],
                                                                     task["schedule_start_time"])
            info_dict["inference_time_cost"] = task["inference_time_cost"]
            info_dict["version"] = "4.6"
            info_dict["timestamp"] = time.time()
            monitor_logger.info(f"{info_dict}")

        return result

    def _recycle_resources(self, task_id, index, task):
        """
        recycle resources
        """
        self.resource_manager.stop_flags[index] = True
        self.resource_manager.tasks_list[index] = None
        self.resource_manager._recycle_block_tables(task["block_tables"])
        if task_id in self.tokens_counter:
            del self.tokens_counter[task_id]
        self.all_tokens[index] = list()

    def _process_batch_output(self):
        """
        batch post-processing function
        """
        tokens = self.output_tokens.numpy()
        batch = self.output_tokens[1, 0]
        tokens = tokens[2:batch + 2]

        batch_result = list()
        exist_finished_task = False
        for i in range(batch):
            if self.resource_manager.stop_flags[i]:
                continue

            token_id = int(tokens[i, 0])
            if token_id < 0:
                continue

            task = self.resource_manager.tasks_list[i]

            task_id = task["req_id"]
            result = self._get_single_result(i, task_id, token_id, task)

            self.tokens_counter[task_id] += 1
            if token_id not in task["eos_token_ids"]:
                self.all_tokens[i].append(token_id)

            self.number_of_output_tokens += 1
            if token_id in task["eos_token_ids"]:
                self._recycle_resources(task_id, i, task)
                model_server_logger.info("req_id: {0} finished".format(task_id))
                model_server_logger.info(f"{self.resource_manager.info()}")
                exist_finished_task = True
            batch_result.append(result)

        self.postprocess(batch_result, exist_finished_task)


class WarmUpTokenProcessor(TokenProcessor):
    """
    Warmup Processor
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self._is_running = True
        self._is_blocking = True

    def postprocess(self, batch_result, exist_finished_task=False):
        pass

    def process_sampling_results(self):
        """
        get output from model and process it
        """
        while self._is_running:
            try:
                rank_id = 0
                get_output(self.output_tokens, rank_id, self._is_blocking)

                if self.output_tokens[0, 0] == -2:
                    continue
                self._process_batch_output()
            except Exception as e:
                model_server_logger.info("while get input_data error: {0} {1}".format(e, str(traceback.format_exc())))

    def stop(self):
        """
        stop warm up thread
        """
        self._is_running = False
        self.worker.join()
        model_server_logger.info("warm up thread stop")
        del self.worker
