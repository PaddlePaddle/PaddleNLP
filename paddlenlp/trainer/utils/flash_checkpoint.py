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

import atexit
import json

# import copy
import multiprocessing
import os
import time
from collections import OrderedDict
from enum import Enum

import paddle
import paddle.autograd as imperative_base
import paddle.distributed as dist
from paddle.base import core
from paddle.incubate.tensor.manipulation import (
    async_offload_with_offset,
    create_async_load,
)
from paddle.optimizer.fusion_utils import FusionStorageHelper

from paddlenlp.utils.env import (
    CONFIG_NAME,
    MODEL_META_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
)
from paddlenlp.utils.fault_tolerance import FC_DUMP_ERROR
from paddlenlp.utils.log import logger


class FCTaskType(Enum):
    """
    TaskType defines the type of tasks that can be executed by the FlashCheckpointWorker.
    """

    UPDATE = 0
    PREPARE = 1
    OFFLOAD = 2
    FINISH = 3


class FCWorkerStatus(Enum):
    IDLE = 0
    OFFLOADING = 1
    DUMPING = 2
    ERROR = 3


def get_fused_param_mappings(optimizer, manipulated_state_dict):
    param_mappings = {}
    ipc_meta_mappings = {}
    index = 0
    sharding_comm_buffers = optimizer._comm_buffer_list
    for buffer in sharding_comm_buffers:
        ipc_meta_mappings[str(index)] = buffer.param_buffer_ipc_meta
        for k, v in manipulated_state_dict.items():
            logger.info(
                f"check vname: {v.name}; buffer._sharding_param_grad_view: {buffer._sharding_param_grad_view.keys()}"
            )
            if v.name in buffer._sharding_param_grad_view:
                assert k not in param_mappings, f"{k} has already been mapped, which is unexpected."
                param_meta = {}
                param_meta["buffer_index"] = str(index)
                param_meta["shape"] = v.shape
                param_meta["name"] = v.name
                param_meta["start"] = buffer._sharding_param_grad_view[v.name]._index
                param_meta["end"] = param_meta["start"] + v._numel()
                param_mappings[k] = param_meta
        index += 1
    assert len(manipulated_state_dict) == len(
        param_mappings
    ), f"manipulated state dict is not fully covered in param mappings, manipulated_state_dict:{manipulated_state_dict.keys()}, param_mappings:{param_mappings.keys()}"
    return param_mappings, ipc_meta_mappings


class ParamFusionStorageHelper:
    def __init__(
        self,
        model_weights_metas,
        buffer_ipc_metas,
    ):
        self.async_loader = create_async_load()
        self.inited_buffers = {}
        self.all_param_numel = 0
        self.model_weights_metas = OrderedDict()
        self.current_offloaded_numel = 0
        self.reset_meta(
            model_weights_metas,
            buffer_ipc_metas,
        )
        self.tasks = []

    @imperative_base.no_grad()
    def reset_meta(
        self,
        model_weights_metas,
        buffer_ipc_metas,
    ):
        self.inited_buffers = {}
        self.all_param_numel = 0
        self.model_weights_metas = OrderedDict()
        if len(model_weights_metas) == 0:
            logger.info("No model states need to save in current worker")
            return

        for k, v in model_weights_metas.items():
            assert isinstance(v, dict), "model_weights_metas must be a dict"
            buffer_index = v["buffer_index"]
            if buffer_index not in self.inited_buffers.keys():
                buffer_tuple = self.init_buffer(buffer_ipc_metas[buffer_index])
                self.inited_buffers[buffer_index] = buffer_tuple
            v["start"] = int(v["start"])
            v["end"] = int(v["end"])
            v["logical_start"] = self.all_param_numel
            self.all_param_numel += v["end"] - v["start"]
            v["logical_end"] = self.all_param_numel
            self.model_weights_metas[k] = v

    def init_buffer(self, meta):
        cuda_buffer = paddle.to_tensor(paddle.base.core.LoDTensor._new_shared_cuda(meta))
        cpu_buffer = cuda_buffer.pin_memory()
        return (cuda_buffer, cpu_buffer)

    @imperative_base.no_grad()
    def sync_partial_param(self, numel_to_sync):
        assert (
            self.current_offloaded_numel + numel_to_sync <= self.all_param_numel
        ), f"numel_to_sync: {numel_to_sync}, current_offloaded_numel: {self.current_offloaded_numel}, all_param_numel: {self.all_param_numel}"
        next_offload_index = 0
        meta_keys_in_order = list(self.model_weights_metas.keys())
        for i, k in enumerate(meta_keys_in_order):
            if self.current_offloaded_numel >= self.model_weights_metas[k]["logical_end"]:
                continue
            next_offload_index = i
            break

        while numel_to_sync > 0:
            offloading_param_key = meta_keys_in_order[next_offload_index]
            offloading_param_meta = self.model_weights_metas[offloading_param_key]
            logical_offload_param_start = self.current_offloaded_numel
            logical_offload_param_end = min(
                offloading_param_meta["logical_end"], logical_offload_param_start + numel_to_sync
            )
            actual_offload_start = (
                logical_offload_param_start - offloading_param_meta["logical_start"]
            ) + offloading_param_meta["start"]
            actual_offload_end = (
                logical_offload_param_end - offloading_param_meta["logical_end"]
            ) + offloading_param_meta["end"]
            actual_offload_size = actual_offload_end - actual_offload_start
            current_param_buffer = self.inited_buffers[offloading_param_meta["buffer_index"]][0]
            current_param_cpu_buffer = self.inited_buffers[offloading_param_meta["buffer_index"]][1]
            task = async_offload_with_offset(
                src_tensor=current_param_buffer,
                dst_tensor=current_param_cpu_buffer,
                src_offset=actual_offload_start,
                dst_offset=actual_offload_start,
                offload_size=actual_offload_size,
                async_loader=self.async_loader,
            )
            self.tasks.append(task)
            self.current_offloaded_numel += actual_offload_size
            numel_to_sync -= actual_offload_size
            next_offload_index += 1

    def wait_all(self):
        if len(self.tasks) == 0:
            return
        last_task = self.tasks.pop(-1)
        while len(self.tasks) > 0:
            task = self.tasks.pop(0)
            task.cuda_wait()
        last_task.cpu_wait()
        self.current_offloaded_numel = 0

    def state_dict(self):
        state_dict = {}
        for k, v in self.model_weights_metas.items():
            state_dict[k] = self.restore_tensor_from_meta(v)
        return state_dict

    @imperative_base.no_grad()
    def restore_tensor_from_meta(self, tensor_meta):
        shape = tensor_meta["shape"]
        name = tensor_meta["name"]
        start = tensor_meta["start"]
        end = tensor_meta["end"]
        cpu_buffer = self.inited_buffers[tensor_meta["buffer_index"]][1]
        tensor = cpu_buffer._slice(start, end)
        tensor.get_tensor()._set_dims(shape)
        tensor.name = name
        return tensor


class FlashCheckpointManager:
    def __init__(self, worker_num, pipeline_hooks_capacity, capacity_usage):
        assert worker_num > 0, "worker_num must be greater than 0"
        assert capacity_usage <= 1.0, "capacity_usage must be less than or equal to 1.0"
        self.cache_version = 0
        self.worker_num = worker_num
        self.workers = []
        self.processes = []
        self.current_worker = None
        self.device_id = int(os.getenv("FLAGS_selected_gpus"))
        self.pipeline_hooks_steps = max(int(pipeline_hooks_capacity * capacity_usage), 1)
        logger.info(
            f"[FC manager] pipeline hooks capacity: {pipeline_hooks_capacity}; pipeline hooks steps for offloading: {self.pipeline_hooks_steps}"
        )
        self.current_pipeline_hook_step = 0
        ctx = multiprocessing.get_context("spawn")
        for i in range(worker_num):
            worker_task_queue = ctx.Queue()
            worker_status = ctx.Value("i", FCWorkerStatus.IDLE.value)
            worker_version = ctx.Value("i", 0)
            worker = FlashCheckpointWorker(
                i,
                self.device_id,
                dist.get_rank(),
                self.pipeline_hooks_steps,
                worker_task_queue,
                worker_status,
                worker_version,
            )
            p = ctx.Process(target=worker_loop, args=(worker,))
            p.start()
            self.workers.append(worker)
            self.processes.append(p)
        self.ready_to_save = False
        atexit.register(self.terminate_workers)

    def update_flash_workers(self, new_version, dynamic_objecs, static_objects):
        self.report_error_worker()
        self.cache_version = new_version
        assert self.current_worker is None, "[FC manager] current_worker must be None"
        task = (FCTaskType.UPDATE, [self.cache_version, dynamic_objecs, static_objects])
        logger.info(f"[FC manager] updating flash workers, verison: {self.cache_version}")
        for worker in self.workers:
            worker.task_queue.put(task)
        logger.info("[FC manager] waiting workers update done")
        for worker in self.workers:
            while worker.version.value != self.cache_version:
                logger.info(
                    f"[FC manager] waiting worker{worker.worker_id} update. worker version: {worker.version.value}, expected version: {self.cache_version}"
                )
                time.sleep(1)
            logger.info(
                f"[FC manager] worker{worker.worker_id} updated. worker version: {worker.version.value}, expected version: {self.cache_version}"
            )
        logger.info("[FC manager] update all flash workers done")
        self.ready_to_save = True

    def get_idle_worker_for_saving(self, save_infos, non_cached_objects):
        self.report_error_worker()
        assert self.current_worker is None, "[FC manager] current_worker must be None"
        found_worker = False
        while True:
            for worker in self.workers:
                if worker.status.value == FCWorkerStatus.IDLE.value:
                    self.current_worker = worker
                    found_worker = True
                    break
            if found_worker:
                break
            logger.info("[FC manager] Waiting for idle worker...")
            time.sleep(1)
        task = (FCTaskType.PREPARE, (save_infos, non_cached_objects))
        logger.info("[FC manager] before putting task for prepare")
        self.current_worker.task_queue.put(task)
        logger.info("[FC manager] after putting task for prepare")

    def sync_offload_status(self):
        self.report_error_worker()
        assert self.current_worker is not None, "[FC manager] current_worker must not be None"
        while True:
            logger.info("[FC manager] Waiting current worker offloading done.")
            if self.current_worker.status.value == FCWorkerStatus.OFFLOADING.value:
                time.sleep(1)
            else:
                logger.info("[FC manager] Current worker offloading done")
                break
        self.current_pipeline_hook_step = 0
        self.current_worker = None

    def report_error_worker(self):
        for worker in self.workers:
            if worker.status.value == FCWorkerStatus.ERROR.value:
                logger.error(f"[FC manager] Worker{worker.worker_id} encountered error.")
                raise RuntimeError(f"{FC_DUMP_ERROR}")

    def flash_checkpoint_pipeline_hook(self, hook_id):
        if self.current_worker is None:
            return
        if self.current_pipeline_hook_step == self.pipeline_hooks_steps:
            return
        if not self.ready_to_save:
            return
        task = (FCTaskType.OFFLOAD, None)
        self.current_worker.task_queue.put(task)
        self.current_pipeline_hook_step += 1

    def finalize(self):
        # clean up if the final step need to save
        if self.current_worker is not None:
            logger.info("[FC manager] clean up last step saving")
            # trigger offload
            for i in range(self.pipeline_hooks_steps):
                self.flash_checkpoint_pipeline_hook(i)
            self.sync_offload_status()
        self.ready_to_save = False
        self.terminate_workers()

    def terminate_workers(self):
        for worker in self.workers:
            task = (FCTaskType.FINISH, None)
            worker.task_queue.put(task)
        for p in self.processes:
            p.join()


def worker_loop(worker):
    worker.run()


class FlashCheckpointWorker:
    def __init__(self, worker_id, device_id, global_rank, offload_chunks, task_queue, status, version):
        super().__init__()
        self.worker_id = worker_id
        self.device_id = device_id
        self.global_rank = global_rank
        self.offload_chunks = offload_chunks
        self.task_queue = task_queue
        self.status = status
        self.version = version

        # for dynamic objects saving
        self.optimizer_fusion_storage_helper = None
        self.param_fusion_storage_helper = None
        self.all_numel = 0
        self.chunk_size_in_numel = 0
        self.offloaded_numels = 0
        self.optimizer_states_name_path = None
        self.model_states_name_path = None

        # for static objects saving
        self.model_config_content = None
        self.training_args_content = None
        self.model_meta_content = None
        self.user_file_list = None

        # for non cached objects saving
        # TODO(@gexiao): remove lr scheduler saves
        self.lr_scheduler = None
        self.trainer_state = None

        # for dumping
        self.flash_save_dir = None
        self.persistent_save_dir = None

    def process_update_task(self, updates):
        """
        sync operation, main process should wait
        """
        version, dynamic_objecs, static_objects = updates
        optimizer_states_meta = dynamic_objecs["optimizer_states_meta"]
        model_states_meta = dynamic_objecs["model_states_meta"]
        self.optimizer_states_name_path = dynamic_objecs["optimizer_states_name_path"]
        self.model_states_name_path = dynamic_objecs["model_states_name_path"]
        self.build_fusion_storage_helper(optimizer_states_meta, model_states_meta)

        self.model_config_content = static_objects["model_config"]
        self.training_args_content = static_objects["training_args"]
        self.model_meta_content = static_objects["model_meta"]
        self.user_file_list = static_objects["user_file"]

        self.manage_offload_chunk()

        self.version.value = version

    def process_prepare_task(self, prepares):
        save_infos, non_cached_objects = prepares
        self.offloaded_numels = 0
        self.status.value = FCWorkerStatus.OFFLOADING.value
        self.flash_save_dir, self.persistent_save_dir = save_infos
        self.lr_scheduler, self.trainer_state = non_cached_objects

    def process_offload_task(self):
        actual_offload_size = (
            min(self.offloaded_numels + self.chunk_size_in_numel, self.all_numel) - self.offloaded_numels
        )
        # Scene1: offload optimizer only
        if self.offloaded_numels + actual_offload_size <= self.optimizer_fusion_storage_helper.buffer_length:
            self.optimizer_fusion_storage_helper.sync_partial_param(
                start=self.offloaded_numels, end=self.offloaded_numels + actual_offload_size
            )
        # Scene2: offload optimizer and param
        elif self.offloaded_numels < self.optimizer_fusion_storage_helper.buffer_length:
            self.optimizer_fusion_storage_helper.sync_partial_param(
                start=self.offloaded_numels, end=self.optimizer_fusion_storage_helper.buffer_length
            )
            self.param_fusion_storage_helper.sync_partial_param(
                numel_to_sync=(
                    actual_offload_size - (self.optimizer_fusion_storage_helper.buffer_length - self.offloaded_numels)
                )
            )
        # Scene3: offload param only
        else:
            self.param_fusion_storage_helper.sync_partial_param(numel_to_sync=actual_offload_size)
        self.offloaded_numels += actual_offload_size

        # wait tasks done and change status to DUMPING at the last chunk
        if self.offloaded_numels == self.all_numel:
            self.optimizer_fusion_storage_helper.wait_all()
            self.param_fusion_storage_helper.wait_all()
            self.status.value = FCWorkerStatus.DUMPING.value

        # continue to process dumping task at the last chunk
        if self.offloaded_numels == self.all_numel:
            need_report_error = self.process_dump_task()
            self.offloaded_numels = 0
            if need_report_error:
                self.status.value = FCWorkerStatus.ERROR.value
            else:
                self.status.value = FCWorkerStatus.IDLE.value

    def process_dump_task(self):
        """
        dump saved objects to either flash device or persistent device
        Notice:
        1. If dumping to flash device failed, the process will move on for other task
        2. If dumping to persistent device failed, the process will change status to fail, and the main process will raise Error.
        """
        need_report_error = False
        if self.flash_save_dir:
            try:
                self.process_dump_task_impl(self.flash_save_dir)
                logger.info(f"[FC worker{self.worker_id}] Dumping to flash device done: {self.flash_save_dir}")
            except Exception as e:
                logger.error(f"[FC worker{self.worker_id}] Failed to dump to flash device: {e}")
        if self.persistent_save_dir:
            try:
                self.process_dump_task_impl(self.persistent_save_dir)
                logger.info(
                    f"[FC worker{self.worker_id}] Dumping to persistent device done: {self.persistent_save_dir}"
                )
            except Exception as e:
                logger.error(f"[FC worker{self.worker_id}] Failed to dump to persistent device: {e}")
                need_report_error = True
        return need_report_error

    def process_dump_task_impl(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # Step1: save static objects
        if self.device_id == 0:
            # Step1.1: save model config
            json_file_path = os.path.join(output_dir, CONFIG_NAME)
            with open(json_file_path, "w", encoding="utf-8") as writer:
                writer.write(self.model_config_content)

            # Step1.2: save training args
            args_file_path = os.path.join(output_dir, TRAINING_ARGS_NAME)
            paddle.save(self.training_args_content, args_file_path)

            # Step1.3: save model meta
            model_meta_path = os.path.join(output_dir, MODEL_META_NAME)
            with open(model_meta_path, "w") as f:
                json.dump(self.model_meta_content, f)

            # Step1.4: save user files
            for (file_name, file_content) in self.user_file_list:
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "w") as f:
                    f.write(file_content)

        # Step2: save dynamic objects
        # Step2.1: save model states
        model_states_name_path = os.path.join(output_dir, self.model_states_name_path)
        paddle.save(self.param_fusion_storage_helper.state_dict(), model_states_name_path)

        # Step2.2: save optimizer states
        optimizer_state_name_path = os.path.join(output_dir, self.optimizer_states_name_path)
        paddle.save(self.optimizer_fusion_storage_helper.state_dict(), optimizer_state_name_path)

        # Step2.3: save LR Scheduler (To be removed)
        lr_state_name_path = os.path.join(output_dir, SCHEDULER_NAME)
        if self.device_id == 0:
            paddle.save(self.lr_scheduler, lr_state_name_path)

        # Step2.4: save TrainerState
        trainer_state_name_path = os.path.join(output_dir, TRAINER_STATE_NAME)
        if self.device_id == 0:
            self.trainer_state.save_to_json(trainer_state_name_path)

        # Step3: dump save signals
        saved_signal_path = os.path.join(output_dir, f"saved_signal_{self.global_rank}")
        with open(saved_signal_path, mode="w+") as f:
            f.write("1")

    def run(self):
        core.set_cuda_current_device_id(self.device_id)
        paddle.set_device(f"gpu:{self.device_id}")
        logger.info(f"[FC worker{self.worker_id}] Worker{self.worker_id} started.")
        while True:
            task = self.task_queue.get()
            task_type, task_body = task
            if task_type == FCTaskType.FINISH:
                logger.info(f"[FC worker{self.worker_id}] Flash checkpoint worker{self.worker_id} exit")
                break
            elif task_type == FCTaskType.UPDATE:
                self.process_update_task(task_body)
            elif task_type == FCTaskType.PREPARE:
                self.process_prepare_task(task_body)
            elif task_type == FCTaskType.OFFLOAD:
                self.process_offload_task()
            else:
                raise ValueError(f"[FC worker{self.worker_id}] Unknown task type: {task_type}")

    def build_fusion_storage_helper(self, optimizer_states_meta, model_states_meta):
        (
            accumulators_meta,
            master_weights_meta,
            merged_model_params_meta,
            buffer_ipc_meta,
        ) = optimizer_states_meta
        if self.optimizer_fusion_storage_helper is None:
            self.optimizer_fusion_storage_helper = FusionStorageHelper(
                accumulators_meta,
                master_weights_meta,
                merged_model_params_meta,
                buffer_ipc_meta,
            )
        else:
            self.optimizer_fusion_storage_helper.reset_meta(
                accumulators_meta,
                master_weights_meta,
                merged_model_params_meta,
                buffer_ipc_meta,
            )
        model_param_mappings, model_ipc_meta_mappings = model_states_meta
        if self.param_fusion_storage_helper is None:
            self.param_fusion_storage_helper = ParamFusionStorageHelper(model_param_mappings, model_ipc_meta_mappings)
        else:
            self.param_fusion_storage_helper.reset_meta(model_param_mappings, model_ipc_meta_mappings)

    def manage_offload_chunk(self):
        # TODO(@gexiao): more precise slice for different dtype
        optimizer_offload_numel = self.optimizer_fusion_storage_helper.buffer_length
        param_offload_numel = self.param_fusion_storage_helper.all_param_numel
        self.all_numel = optimizer_offload_numel + param_offload_numel
        self.chunk_size_in_numel = (self.all_numel - 1) // self.offload_chunks + 1
        logger.info(
            f"[FC worker{self.worker_id}] All numel: {self.all_numel}, Offload chunks: {self.offload_chunks}, Chunk size: {self.chunk_size_in_numel}]"
        )
