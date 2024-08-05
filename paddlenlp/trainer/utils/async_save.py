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
import copy
import multiprocessing
import os

import paddle

from paddlenlp.utils.log import logger


class AsyncSaver:
    def __init__(self):
        self.context = multiprocessing.get_context("spawn")
        self.cpu_optimizer_state_dict = {}
        self._queue = self.context.Queue()
        self.trigger_event = self.context.Event()
        self.completed_event = self.context.Event()
        self.completed_event.set()
        self._save_process = None

    def _save_optimizer(
        self,
        queue,
        # obj,
        trigger_event,
        completed_event,
    ):
        while True:
            logger.info("===trigger_event wait")
            trigger_event.wait()
            trigger_event.clear()
            # try:
            obj, name_mapping, path, saved_signal_path, protocol = queue.get()
            # obj = queue.get_nowait()
            # if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "master_weights" and isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, paddle.Tensor):
                            vv.name = name_mapping["master_weights"][kk]
                else:
                    if k in name_mapping and isinstance(v, paddle.Tensor):
                        v.name = name_mapping[k]
            logger.info(f"====save {path}===")
            paddle.save(obj, path, protocol)
            # dump savd_siganl
            with open(saved_signal_path, mode="w+") as f:
                f.write("1")
                f.flush()
                os.fsync(f.fileno())

            # except Exception as e:
            # logger.warning(f"Error occurred during saving: {e}")
            logger.info("===completed_event set")
            completed_event.set()

    def run(self, optimizer_state_dict, path, saved_signal_path, protocol=4):
        logger.info("===completed_event wait")
        self.completed_event.wait()
        self.completed_event.clear()

        self.cpu_optimizer_state_dict.clear()
        name_mapping = {"master_weights": {}}
        for k, v in optimizer_state_dict.items():
            if k == "master_weights":
                self.cpu_optimizer_state_dict[k] = {}
                for kk, vv in v.items():
                    self.cpu_optimizer_state_dict[k][kk] = vv.pin_memory()
                    name_mapping[k][kk] = vv.name
            elif k == "LR_Scheduler":
                self.cpu_optimizer_state_dict[k] = copy.deepcopy(v)
            else:
                self.cpu_optimizer_state_dict[k] = v.pin_memory()
                name_mapping[k] = v.name
            paddle.device.synchronize()

        self._queue.put((self.cpu_optimizer_state_dict, name_mapping, path, saved_signal_path, protocol))

        if self._save_process is None:
            self._save_process = self.context.Process(
                target=self._save_optimizer,
                args=(
                    self._queue,
                    self.trigger_event,
                    self.completed_event,
                ),
            )
            self._save_process.start()
        logger.info("===trigger_event set")
        self.trigger_event.set()
