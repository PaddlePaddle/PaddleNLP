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

import multiprocessing
import os
import signal
import subprocess
import time
import uuid
import weakref
from datetime import datetime
from multiprocessing import shared_memory

import numpy as np
from server.engine.resource_manager import ResourceManager
from server.engine.task_queue_manager import (TaskQueueManager,
                                              launch_queue_service)
from server.engine.token_processor import TokenProcessor, WarmUpTokenProcessor
from server.utils import model_server_logger


class Engine(object):
    """
    Engine Class
    """
    def __init__(self, cfg, token_processor):
        self.cfg = cfg
        self.resource_manager = ResourceManager(self.cfg)
        self.token_processor = token_processor
        self.token_processor.set_resource_manager(self.resource_manager)
        self.is_started = False

        self._init_engine_flags()
        self._finalizer = weakref.finalize(self, self._exit_sub_services)

    def start(self):
        """
        initialize engine and start sub services
        """
        assert not self.is_started, "The engine is already started.!"
        start_time = time.time()
        self.queue_service = self._start_tasks_queue_service()
        self.tasks_queue = TaskQueueManager(mp_num=self.cfg.mp_num, port=self.cfg.infer_port)

        self.token_processor.tasks_queue = self.tasks_queue
        self.infer_proc = self._start_infer_service()
        model_server_logger.info("Waitting infer processes ready...")
        while not self._infer_processes_ready():
            time.sleep(1)
        self.is_started = True

        # start warmup
        if self.cfg.use_warmup:
            model_server_logger.info("Start warmup")
            self._set_warmup_token_processor()
            self.warmup()
            self._del_warmup_token_processor()
            model_server_logger.info("Warmup finish")

        # start TokenProcessor thread
        self.token_processor.run()
        model_server_logger.info("Infer processes are launched with {} seconds.".format(time.time() - start_time))

    def warmup(self):
        """
        construct test tasks and avoid out of memory problem in the infer process
        """
        # get eos_token_id
        from server.data.processor import DataProcessor
        eos_token_ids = DataProcessor().get_eos_tokens()

       # construct test tasks
        res_task = []
        for j in range(2 * self.cfg.max_batch_size):
            data = {
                "input_ids": [5],
                "req_id": j,
                "max_dec_len": self.cfg.dec_len_limit,
                "min_dec_len": int(self.cfg.dec_len_limit * 0.5) + 1,
                "eos_token_ids": eos_token_ids
            }
            res_task.append(data)
        for j in range(2 * self.cfg.max_prefill_batch):
            data = {
                "input_ids": [5] * self.cfg.seq_len_limit,
                "req_id": j + 2 * self.cfg.max_batch_size,
                "max_dec_len": 1,
                "min_dec_len": 1,
                "eos_token_ids": eos_token_ids
            }
            res_task.append(data)

        for x in res_task:
            while self.available_batch() == 0 or not self.insert_tasks([x]):
                time.sleep(0.0002)

        self.token_processor._is_blocking = False
        # wait for all tasks finished
        while not self.all_tasks_finished():
            time.sleep(1)

    def insert_tasks(self, tasks):
        """
        insert tasks to the engine

        Args:
            tasks: list of tasks

        Returns:
            return: True if success, False otherwise
        """
        if not isinstance(tasks, list):
            tasks = [tasks]

        for item in tasks:
            item["schedule_start_time"] = datetime.now()

        available_batch = np.sum(self.resource_manager.stop_flags)
        if len(tasks) > available_batch:
            model_server_logger.error("Inserting batch:{} exceeds the available batch:{}.".format(
                len(tasks), available_batch))
            model_server_logger.error("The exceeded part will be ignored!")
            tasks = tasks[:available_batch]

        for i in range(len(tasks)):
            req_id = tasks[i]["req_id"]
            input_token_num = len(tasks[i]["input_ids"])
            if input_token_num >= self.cfg.max_seq_len - 1:
                model_server_logger.warning(f"{req_id}: Input length:{input_token_num}, exceed the limits.")
                tasks[i]["input_ids"] = tasks[i]["input_ids"][:self.cfg.max_seq_len - 1]
            if "seq_len" in tasks[i] and "max_dec_len" not in tasks[i]:
                tasks[i]["max_dec_len"] = tasks[i]["seq_len"]

            # max_dec_len + input_token_num > MAX_SEQ_LEN
            if input_token_num + tasks[i]["max_dec_len"] > self.cfg.max_seq_len:
                tasks[i]["max_dec_len"] = self.cfg.max_seq_len - input_token_num
                model_server_logger.warning("Force max_dec_len to be {} for req_id={}.".format(
                    tasks[i]["max_dec_len"], tasks[i]["req_id"]))

            # min_dec_len + input_token_num > MAX_SEQ_LEN
            if input_token_num + tasks[i]["min_dec_len"] > self.cfg.max_seq_len:
                tasks[i]["min_dec_len"] = self.cfg.max_seq_len - input_token_num
                model_server_logger.warning("Force min_dec_len to be {} for req_id={}.".format(
                    tasks[i]["min_dec_len"], tasks[i]["req_id"]))

        tasks = self.resource_manager.allocate_resources_for_new_tasks(tasks)
        if not tasks:
            return False

        self.token_processor.number_of_tasks += len(tasks)
        for i in range(len(tasks)):
            self.token_processor.number_of_input_tokens += len(tasks[i]["input_ids"])

        req_ids = [t["req_id"] for t in tasks]
        model_server_logger.info(f"Tasks are sent to engine, req_ids={req_ids}")
        self.tasks_queue.put((tasks, self.resource_manager.real_bsz))
        return True

    def task_is_finished(self, index):
        """
        judge if the task is finished

        Args:
            index: task index

        Returns:
            return: True if finished, False otherwise
        """
        assert index < len(self.resource_manager.stop_flags)
        return self.resource_manager.stop_flags[index]

    def is_queue_empty(self):
        """
        judge if the queue is empty

        Returns:
            return: True if empty, False otherwise
        """
        return self.tasks_queue.empty()

    def is_resource_sufficient(self, input_token_num):
        """
        judge if the resource is sufficient

        Args:
            input_token_num: input token number

        Returns:
            return: True if sufficient, False otherwise
        """
        return self.resource_manager.is_resource_sufficient(input_token_num)

    def all_tasks_finished(self):
        """
        judge if all tasks are finished

        Returns:
            return: True if all finished, False otherwise
        """
        return np.sum(self.resource_manager.stop_flags) == len(self.resource_manager.stop_flags)

    def available_batch(self):
        """
        available batch size of the engine

        Returns:
            return: available batch size
        """
        return self.resource_manager.available_batch()

    def available_block_num(self):
        """
        available block number of the engine

        Returns:
            return: available block number
        """
        return self.resource_manager.availabel_block_num()

    def _set_warmup_token_processor(self):
        """
        set token_processor for warmup
        """
        self.token_processor_backup = self.token_processor
        self.token_processor = WarmUpTokenProcessor(self.cfg)
        self.token_processor.set_resource_manager(self.resource_manager)
        self.token_processor.tasks_queue = self.tasks_queue

        # start TokenProcessor thread
        self.token_processor.run()

    def _del_warmup_token_processor(self):
        """
        delete token_processor for warmup
        """
        self.token_processor.stop()
        del self.token_processor

        # reset token_processor
        self.token_processor = self.token_processor_backup
        del self.token_processor_backup

    def _infer_processes_ready(self):
        """
        judge if all infer processes are ready

        Returns:
            return: True if all ready, False otherwise
        """
        if np.sum(self.flag_ready_array) == self.cfg.mp_num:
            return True
        return False

    def _clear_engine_flags(self):
        """
        clear engine flags
        """
        try:
            self.shm_flag_ready.close()
            self.shm_flag_ready.unlink()
            self.shm_flag_has_block_step.close()
            self.shm_flag_has_block_step.unlink()
        except:
            pass

    def _init_engine_flags(self):
        """
        Initialize shared memory to indicate engine status
        """
        flag_array = np.zeros([self.cfg.mp_num], dtype=np.int32)
        try:
            tmp = shared_memory.SharedMemory(
                create=False, size=flag_array.nbytes, name=self.cfg.get_unique_name("shm_flag_infer_ready")
            )
            tmp.close()
            tmp.unlink()
        except:
            pass
        self.shm_flag_ready = shared_memory.SharedMemory(
            create=True, size=flag_array.nbytes, name=self.cfg.get_unique_name("shm_flag_infer_ready")
        )
        self.flag_ready_array = np.ndarray(
            flag_array.shape, dtype=flag_array.dtype, buffer=self.shm_flag_ready.buf
        )
        self.flag_ready_array[:] = 0

        # broadcast flag for engine
        broadcast_flag_array = np.zeros([1], dtype=np.int32)
        try:
            tmp = shared_memory.SharedMemory(
                create=False,
                size=broadcast_flag_array.nbytes,
                name=self.cfg.get_unique_name("shm_pd_infer_flag_broadcast"),
            )
            tmp.close()
            tmp.unlink()
        except:
            pass
        self.shm_flag_broadcast = shared_memory.SharedMemory(
            create=True, size=broadcast_flag_array.nbytes, name=self.cfg.get_unique_name("shm_pd_infer_flag_broadcast")
        )
        self.flag_broadcast_array = np.ndarray(
            broadcast_flag_array.shape,
            dtype=broadcast_flag_array.dtype,
            buffer=self.shm_flag_broadcast.buf,
        )
        self.flag_broadcast_array[0] = 0

        has_block_step_flag_array = np.zeros([1], dtype=np.int32)
        try:
            tmp = shared_memory.SharedMemory(
                create=False,
                size=has_block_step_flag_array.nbytes,
                name=self.cfg.get_unique_name("shm_flag_has_block_step"))
            tmp.close()
            tmp.unlink()
        except:
            pass
        self.shm_flag_has_block_step = shared_memory.SharedMemory(
            create=True,
            size=has_block_step_flag_array.nbytes,
            name=self.cfg.get_unique_name("shm_flag_has_block_step"))
        self.flag_has_block_step_array = np.ndarray(
            has_block_step_flag_array.shape,
            dtype=has_block_step_flag_array.dtype,
            buffer=self.shm_flag_has_block_step.buf)
        self.flag_has_block_step_array[:] = 0

    def _exit_sub_services(self):
        """
        exit sub services
        """
        if hasattr(self, "queue_service") and self.queue_service is not None:
            self.queue_service.terminate()
            self.queue_service.join()
        if hasattr(self, "infer_proc") and self.infer_proc is not None:
            os.killpg(self.infer_proc.pid, signal.SIGTERM)

    def _start_tasks_queue_service(self):
        """
        start tasks queue service

        Returns:
            p: process handle
        """
        p = multiprocessing.Process(target=launch_queue_service, args=(self.cfg.infer_port, self.cfg.mp_num))
        p.start()
        time.sleep(0.3)
        if p.is_alive():
            model_server_logger.info("start tasks queue service successfully")
        else:
            error_msg = "Failed to start tasks queue service, please check " \
                        "the log/task_queue_manager.log for details"
            model_server_logger.info(error_msg)
            raise Exception(error_msg)
        return p

    def _start_gpu_infer_service(self):
        """
        start gpu infer service

        Returns:
            p: process handle
        """
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.split(current_file_path)[0]
        pd_cmd = "python3 -m paddle.distributed.launch "
        py_script = os.path.join(current_dir_path, "infer.py")

        arguments = (f" --devices {self.cfg.device_ids} {py_script} --model_dir {self.cfg.model_dir}"
                    f" --max_batch_size {self.cfg.max_batch_size} --max_seq_len {self.cfg.max_seq_len}"
                    f" --max_dec_len {self.cfg.max_dec_len}"
                    f" --max_block_num {self.cfg.total_block_num} --block_size {self.cfg.block_size}"
                    f" --use_cache_kv_int8 {self.cfg.use_cache_kv_int8}"
                    f" --enc_dec_block_num {self.cfg.enc_dec_block_num}"
                    f" --block_ratio {self.cfg.block_ratio} --dtype {self.cfg.dtype}")
        pd_cmd = pd_cmd + arguments + " >log/launch_infer.log 2>&1"
        model_server_logger.info("Launch infer service command: {}".format(pd_cmd))
        p = subprocess.Popen(
            pd_cmd,
            shell=True,
            preexec_fn=os.setsid,
        )
        return p

    def _start_infer_service(self):
        """
        start infer service
        """
        return self._start_gpu_infer_service()
