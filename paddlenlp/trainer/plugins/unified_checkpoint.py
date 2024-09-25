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

import copy
import gc
import json
import multiprocessing
import os
import sys
import time
from multiprocessing import shared_memory

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from tqdm.auto import tqdm

from paddlenlp.peft import LoRAModel, PrefixModelForCausalLM
from paddlenlp.trainer.argparser import strtobool
from paddlenlp.trainer.trainer_utils import ExplicitEnum
from paddlenlp.trainer.utils.helper import distributed_file, distributed_isfile
from paddlenlp.transformers.model_utils import (
    PretrainedModel,
    _add_variant,
    _load_state_dict_into_model,
    faster_set_state_dict,
    get_parameter_dtype,
    load_state_dict,
    unwrap_model,
)
from paddlenlp.transformers.utils import (
    device_guard,
    dtype_byte_size,
    get_checkpoint_shard_files,
    is_safetensors_available,
)
from paddlenlp.utils.distributed import distributed_allgather, distributed_gather
from paddlenlp.utils.env import (
    LORA_WEIGHTS_NAME,
    PADDLE_MASTER_WEIGHTS_INDEX_NAME,
    PADDLE_MASTER_WEIGHTS_NAME,
    PADDLE_OPTIMIZER_INDEX_NAME,
    PADDLE_OPTIMIZER_NAME,
    PADDLE_PEFT_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    PAST_KEY_VALUES_FILE_NAME,
    PREFIX_WEIGHTS_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_MASTER_WEIGHTS_NAME,
    SAFE_OPTIMIZER_INDEX_NAME,
    SAFE_OPTIMIZER_NAME,
    SAFE_PEFT_WEIGHTS_INDEX_NAME,
    SAFE_PEFT_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)
from paddlenlp.utils.log import logger
from paddlenlp.utils.nested import nested_copy, nested_copy_place
from paddlenlp.utils.tools import get_env_device

if is_safetensors_available():
    from safetensors.numpy import save_file as safe_save_file

    if sys.platform.startswith("win"):
        from safetensors import safe_open
        from safetensors.numpy import load_file
    else:
        from paddlenlp.utils.safetensors import fast_safe_open as safe_open
        from paddlenlp.utils.safetensors import fast_load_file as load_file

from .shared_memory_utils import (
    _read_state_dict_from_shm,
    _traverse_copy_to_shm,
    create_meta_dict,
)

FP32_MASTER = "fp32_master_0"
optimizer_scalar_name = [
    "beta1_pow_acc_0",
    "beta2_pow_acc_0",
]
optimizer_non_scaler_name = [
    "moment1_0",
    "moment2_0",
    "velocity_0",
]  # to be added


DEST_PLACE = paddle.CPUPlace()
if paddle.device.is_compiled_with_cuda():
    DEST_PLACE = paddle.CUDAPinnedPlace()


class UnifiedCheckpointOption(ExplicitEnum):
    """
    "- skip_save_model_weight: do not save model weights when the masters weight exist\n"
    "- master_weight_compatible: 1. if the master weights exist, only load when needed\n"
    "                            2. if master weights does not exist, convert model weights to master weights when needed\n"
    "- async_save: enable asynchronous saving checkpoints to disk\n"
    "- enable_all_options: enable all optimization configurations\n"
    """

    SKIP_SAVE_MODEL_WEIGHT = "skip_save_model_weight"
    MASTER_WEIGHT_COMPATIBLE = "master_weight_compatible"
    ASYNC_SAVE = "async_save"
    IGNORE_MERGE_OPTIMIZER = "ignore_merge_optimizer"


class UnifiedCheckpointHandler:
    def __init__(self, args):
        self.args = args
        self.global_rank = paddle.distributed.get_rank() if paddle.distributed.get_world_size() > 1 else -1

        # Mainly for asynchronous saving.
        self._shm_model_weight = None
        self._shm_master_weight = None
        self._shm_optimizer_weight = None
        self._meta_dict_model = None
        self._meta_dict_master_weight = None
        self._meta_dict_optim = None
        self._process_model_weight = None
        self._process_master_weight = None
        self._process_optimizer_weight = None
        self._lock = None
        self._shared_save_path = None
        self._shared_save_model_flag = None
        self._shared_save_master_weight_flag = None
        self._shared_save_optimizer_flag = None

        if "async_save" in self.args.unified_checkpoint_config:
            self._lock = multiprocessing.Lock()
            self._shared_save_model_path = multiprocessing.Array("c", 100000)
            self._shared_save_master_weight_path = multiprocessing.Array("c", 100000)
            self._shared_save_optimizer_path = multiprocessing.Array("c", 100000)
            self._shared_save_model_flag = multiprocessing.Array("i", 1)
            self._shared_save_master_weight_flag = multiprocessing.Array("i", 1)
            self._shared_save_optimizer_flag = multiprocessing.Array("i", 1)

    def _file_save_async_or_sync(self, state_dict, path, is_sync=True, state_dict_type="model_weight"):
        if is_sync:
            for k in list(state_dict.keys()):
                if isinstance(state_dict[k], paddle.Tensor):
                    state_dict[k] = state_dict.pop(k).cpu().numpy()
            safe_save_file(state_dict, path, metadata={"format": "np"})
        else:
            if state_dict_type == "model_weight":
                if self._shm_model_weight is None:
                    self._meta_dict_model, buffer_size = create_meta_dict(state_dict)
                    self._shm_model_weight = shared_memory.SharedMemory(create=True, size=buffer_size)
                shm_state_dict = self._shm_model_weight
                meta_dict = self._meta_dict_model
                shared_save_flag = self._shared_save_model_flag
                shared_save_path = self._shared_save_model_path
                if self._process_model_weight is None:
                    self._process_model_weight = multiprocessing.Process(
                        target=self._save_file_async_in_process,
                        args=(
                            meta_dict,
                            self._shm_model_weight.name,
                            self._shared_save_model_flag,
                            self._shared_save_model_path,
                            self._lock,
                            state_dict_type,
                            self.global_rank,
                        ),
                    )
                    self._process_model_weight.start()
            elif state_dict_type == "master_weight":
                if self._shm_master_weight is None:
                    self._meta_dict_master_weight, buffer_size = create_meta_dict(state_dict)
                    self._shm_master_weight = shared_memory.SharedMemory(create=True, size=buffer_size)
                shm_state_dict = self._shm_master_weight
                meta_dict = self._meta_dict_master_weight
                shared_save_flag = self._shared_save_master_weight_flag
                shared_save_path = self._shared_save_master_weight_path
                if self._process_master_weight is None:
                    self._process_master_weight = multiprocessing.Process(
                        target=self._save_file_async_in_process,
                        args=(
                            meta_dict,
                            self._shm_master_weight.name,
                            self._shared_save_master_weight_flag,
                            self._shared_save_master_weight_path,
                            self._lock,
                            "model_weight"
                            if "skip_save_model_weight" in self.args.unified_checkpoint_config
                            else state_dict_type,
                            self.global_rank,
                        ),
                    )
                    self._process_master_weight.start()
            elif state_dict_type == "optimizer_weight":
                if self._shm_optimizer_weight is None:
                    self._meta_dict_optim, buffer_size = create_meta_dict(state_dict)
                    self._shm_optimizer_weight = shared_memory.SharedMemory(create=True, size=buffer_size)
                shm_state_dict = self._shm_optimizer_weight
                meta_dict = self._meta_dict_optim
                shared_save_flag = self._shared_save_optimizer_flag
                shared_save_path = self._shared_save_optimizer_path
                if self._process_optimizer_weight is None:
                    self._process_optimizer_weight = multiprocessing.Process(
                        target=self._save_file_async_in_process,
                        args=(
                            meta_dict,
                            self._shm_optimizer_weight.name,
                            self._shared_save_optimizer_flag,
                            self._shared_save_optimizer_path,
                            self._lock,
                            state_dict_type,
                            self.global_rank,
                        ),
                    )
                    self._process_optimizer_weight.start()

            while True:  # wait until no process is saving.
                flag_value = shared_save_flag[0]
                if flag_value == 0:
                    break
                time.sleep(0.5)
                logger.info(f"Wait for the previous save process to finish saving {state_dict_type}")
            # only save model weight or save master weight, we enter this loop.
            self._reset_and_update(shared_save_path, path)
            _traverse_copy_to_shm(state_dict, meta_dict, shm_state_dict.buf)
            with self._lock:
                shared_save_flag[0] = 1

    def _save_file_async_in_process(
        self,
        meta_dict,
        shm_name,
        shared_save_flag,
        shared_save_path,
        lock,
        state_dict_type,
        global_rank,
    ):
        shm = shared_memory.SharedMemory(name=shm_name)
        while True:
            flag_value = shared_save_flag[0]  # if process uses `spawn`, cannot read this value.
            if flag_value == -1:  # stop process
                break
            if flag_value == 0:  # nothing to save
                continue
            if flag_value == 1:  # need to save
                path = shared_save_path[:].decode("utf-8").rstrip("\x00")
                logger.info(f"Start to async save {path}")
                state_dict = _read_state_dict_from_shm(meta_dict, shm)  # numpy array
                safe_save_file(state_dict, path, {"format": "np"})
                del state_dict
                saved_signal_path = os.path.join(os.path.dirname(path), f".{state_dict_type}.done.{global_rank}")
                paddle.save(global_rank, saved_signal_path)
                with lock:
                    shared_save_flag[0] = 0
            time.sleep(0.5)
        shm.close()

    def _reset_and_update(self, shared_array, new_value):
        # clear array
        for i in range(len(shared_array)):
            shared_array[i] = b"\0"
        # update array
        encoded_value = new_value.encode("utf-8")
        shared_array[: len(encoded_value)] = encoded_value

    def save_unified_checkpoint(self, model, optimizer, output_dir):
        """save unified checkpoint

        Args:
            model (PretrainedModel): model to save
            output_dir (str): save dir
            safe_serialization (bool, optional): use safetensors. Defaults to False.

        Raises:
            ValueError: if model is not an instance of `PretrainedModel` and the model cannot be saved
        """
        if isinstance(model, PretrainedModel):
            model_to_save = model
        elif isinstance(unwrap_model(model), PretrainedModel):
            model_to_save = unwrap_model(model)
        elif isinstance(model, PrefixModelForCausalLM) or isinstance(model, LoRAModel):
            model_to_save = model
        else:
            raise ValueError("Unified checkpoint only supports PretrainedModel, LoRAModel and PrefixModelForCausalLM!")

        # Under non distributed environment.
        if paddle.distributed.get_world_size() <= 1:
            self.save_single_card_checkpoint(model_to_save, output_dir)
            return

        skip_save_model_weight = False
        if UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value in self.args.unified_checkpoint_config:
            if is_need_master_weight(optimizer, is_fp16_or_bp16=(self.args.fp16 or self.args.bf16)):
                logger.info(
                    f"With {UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value}, skip the model checkpoint save."
                    " The master weight will be loaded as model weights for next resumption."
                )
                # not save model weight, load from master weight
                skip_save_model_weight = True

        save_directory = output_dir
        os.makedirs(save_directory, exist_ok=True)

        # save model weights
        if not skip_save_model_weight:
            state_dict, shard_file, sharded_index = unified_checkpoint_into_shards(
                self.args, model_to_save, safe_serialization=True
            )
            is_sync_save = True
            if "async_save" in self.args.unified_checkpoint_config:
                is_sync_save = False
            self._file_save_async_or_sync(
                state_dict,
                path=os.path.join(save_directory, shard_file),
                is_sync=is_sync_save,
                state_dict_type="model_weight",
            )
            if sharded_index is not None:
                if isinstance(model_to_save, LoRAModel) or isinstance(model_to_save, PrefixModelForCausalLM):
                    index_name = SAFE_PEFT_WEIGHTS_INDEX_NAME
                else:
                    index_name = SAFE_WEIGHTS_INDEX_NAME
                path = os.path.join(output_dir, index_name)

                if self.args.should_save:
                    with open(path, "w") as f:
                        json.dump(sharded_index, f, indent=4)

        if self.args.should_save:
            # Save prefix model past_key_values
            if isinstance(model_to_save, PrefixModelForCausalLM):
                save_prefix_past_key_value(model_to_save, save_directory)
                model_to_save.prefix_config.save_pretrained(save_directory)
            if isinstance(model_to_save, LoRAModel):
                model_to_save.lora_config.save_pretrained(save_directory)

        # save the config
        config_to_save = save_config(model_to_save)
        # Attach architecture to the config
        if isinstance(model_to_save, LoRAModel) or isinstance(model_to_save, PrefixModelForCausalLM):
            config_to_save.architectures = [model_to_save.model.__class__.__name__]
        else:
            config_to_save.architectures = [model_to_save.__class__.__name__]
        if self.args.should_save:
            config_to_save.save_pretrained(save_directory)
        paddle.device.cuda.empty_cache()

        if strtobool(os.getenv("FLAG_LLM_PDC", "False")) and self.args.should_save:
            world_size = paddle.distributed.get_world_size()
            save_info = {
                "world_size": world_size,
                "ignore_save_lr_and_optim": self.args.ignore_save_lr_and_optim,
                "skip_save_model_weight": "skip_save_model_weight" in self.args.unified_checkpoint_config,
            }
            paddle.save(save_info, os.path.join(save_directory, ".saving_info"))

    def load_unified_checkpoint(self, model, optimizer, resume_from_checkpoint: str):
        """Load potential model checkpoint

        Args:
            model (PretrainedModel): Your model to load
            resume_from_checkpoint (str): path of the checkpoint to load

        Returns:
            None
        """
        if paddle.distributed.get_world_size() <= 1:
            load_single_card_checkpoint(self.args, model, resume_from_checkpoint)
            return

        local_resume = check_unified_checkpoint(self.args, model, resume_from_checkpoint, safe_serialization=True)

        if not local_resume:
            logger.info("Begin to dynamically load unified checkpoint!")
            load_unified_checkpoint_dynamically(
                self.args, model, optimizer, resume_from_checkpoint, safe_serialization=True
            )
            return

        if self.args.dataset_rank == 0:
            load_unified_checkpoint_locally(self.args, model, resume_from_checkpoint, safe_serialization=True)

    def save_non_merge_optimizer(self, model, optimizer, output_dir):
        paddle.device.cuda.empty_cache()
        optim_state_dict = nested_copy(optimizer.state_dict())
        master_weights = None
        if "master_weights" in optim_state_dict.keys():
            master_weights = optim_state_dict["master_weights"]
            optim_state_dict.pop("master_weights")
        if "LR_Scheduler" in optim_state_dict.keys():
            optim_state_dict.pop("LR_Scheduler")

        # gather global master_weights status.
        global_master_weights = reduce_master_weights_status(master_weights is not None)
        if master_weights is None and global_master_weights:
            master_weights = {}

        # get optimizer param mappings
        static2struct_name_mappings = {}
        state_dict = get_expected_state_dict(model)
        for k, v in state_dict.items():
            static2struct_name_mappings[v.name] = k

        # rename optimizer param name
        for key in list(optim_state_dict.keys()):
            static_name, type_name = generate_base_static_name(key)
            new_name = static2struct_name_mappings[static_name] + "/" + type_name
            optim_state_dict[new_name] = optim_state_dict.pop(key)
        if master_weights is not None:
            for key in list(master_weights.keys()):
                master_weights[static2struct_name_mappings[key]] = master_weights.pop(key)

        optimizer_name = _add_variant(SAFE_OPTIMIZER_NAME, self.args.optimizer_name_suffix)
        master_weights_name = _add_variant(SAFE_MASTER_WEIGHTS_NAME, self.args.optimizer_name_suffix)

        is_sync_save = True
        if "async_save" in self.args.unified_checkpoint_config:
            is_sync_save = False
        self._file_save_async_or_sync(
            optim_state_dict,
            path=os.path.join(output_dir, optimizer_name),
            is_sync=is_sync_save,
            state_dict_type="optimizer_weight",
        )
        self._file_save_async_or_sync(
            master_weights,
            path=os.path.join(output_dir, master_weights_name),
            is_sync=is_sync_save,
            state_dict_type="master_weight",
        )

    def load_non_merge_optimizer(self, model, optimizer, resume_from_checkpoint):
        # init and get optimizer LR_Scheduler
        returned_optim_state_dict = nested_copy(optimizer.state_dict())

        optimizer_name = _add_variant(SAFE_OPTIMIZER_NAME, self.args.optimizer_name_suffix)
        master_weights_name = _add_variant(SAFE_MASTER_WEIGHTS_NAME, self.args.optimizer_name_suffix)
        optimizer_path = os.path.join(resume_from_checkpoint, optimizer_name)
        master_weights_path = os.path.join(resume_from_checkpoint, master_weights_name)
        has_master_weights = True if os.path.isfile(master_weights_path) else False

        model_state_dict = get_expected_state_dict(model)
        struct2static_name_mappings = {k: v.name for k, v in model_state_dict.items()}  # get optimizer param mappings
        optimizer_state_dict = load_file(optimizer_path)
        if has_master_weights:
            master_weights = load_file(master_weights_path)

        # rename and move to paddle.Tensor
        for key in list(optimizer_state_dict.keys()):
            key_name = key.split("/")
            static_name = struct2static_name_mappings[key_name[0]]
            if has_master_weights:
                key_name = "_".join([static_name, FP32_MASTER, key_name[1]])
            else:
                key_name = "_".join([static_name, key_name[1]])
            with device_guard():
                weight = paddle.Tensor(optimizer_state_dict.pop(key), zero_copy=True)
            weight = weight._copy_to(paddle.framework._current_expected_place(), False)
            returned_optim_state_dict[key_name] = weight
            returned_optim_state_dict[key_name].name = key_name

        if has_master_weights:
            returned_optim_state_dict["master_weights"] = {}
            for key in list(master_weights.keys()):
                static_name = struct2static_name_mappings[key]
                with device_guard():
                    weight = paddle.Tensor(master_weights.pop(key), zero_copy=True)
                weight = weight._copy_to(paddle.framework._current_expected_place(), False)
                returned_optim_state_dict["master_weights"][static_name] = weight
                returned_optim_state_dict["master_weights"][static_name].name = "_".join([static_name, FP32_MASTER])

        return returned_optim_state_dict

    def save_unified_optimizer(self, model, optimizer, output_dir):
        """save unified optimizer

        Args:
            model (PretrainedModel): model used to get key mapping.
            optimizer (Optimizer): optimizer to save
            output_dir (str): Save directory.

        """

        if "ignore_merge_optimizer" in self.args.unified_checkpoint_config:
            self.save_non_merge_optimizer(model, optimizer, output_dir)
            return

        if paddle.distributed.get_world_size() <= 1:
            self.save_single_card_optimizer(model, optimizer, output_dir)
            return

        # Split into naive optimizer params and master weights.
        results = unified_optimizer_into_shards(self.args, model, optimizer, safe_serialization=True)
        master_weight_state_dict = None
        if len(results) == 1:
            optim_state_dict, shard_optim_file, sharded_optim_index = results[0]
        else:
            optim_state_dict, shard_optim_file, sharded_optim_index = results[0]
            master_weight_state_dict, shard_master_weight_file, sharded_master_weight_index = results[1]

        paddle.device.cuda.empty_cache()

        save_directory = output_dir
        os.makedirs(save_directory, exist_ok=True)

        is_sync_save = True
        if "async_save" in self.args.unified_checkpoint_config:
            is_sync_save = False
        self._file_save_async_or_sync(
            optim_state_dict,
            path=os.path.join(save_directory, shard_optim_file),
            is_sync=is_sync_save,
            state_dict_type="optimizer_weight",
        )
        if master_weight_state_dict is not None:
            self._file_save_async_or_sync(
                master_weight_state_dict,
                path=os.path.join(save_directory, shard_master_weight_file),
                is_sync=is_sync_save,
                state_dict_type="master_weight",
            )

        if sharded_optim_index is not None:
            optimizer_index_name = SAFE_OPTIMIZER_INDEX_NAME
            path = os.path.join(output_dir, optimizer_index_name)
            if self.args.should_save:
                with open(path, "w") as f:
                    json.dump(sharded_optim_index, f, indent=4)

            master_weights_name = SAFE_MASTER_WEIGHTS_INDEX_NAME
            if UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value in self.args.unified_checkpoint_config:
                master_weights_name = SAFE_WEIGHTS_INDEX_NAME
            master_path = os.path.join(output_dir, master_weights_name)
            if master_weight_state_dict is not None:
                if self.args.should_save:
                    with open(master_path, "w") as f:
                        json.dump(sharded_master_weight_index, f, indent=4)

    def load_unified_optimizer(self, args, model, optimizer, resume_from_checkpoint):
        """Load potential model checkpoint

        Args:
            model (PretrainedModel): Your model to load
            resume_from_checkpoint (str): path of the checkpoint to load

        Returns:
            None
        """

        if paddle.distributed.get_world_size() <= 1:
            optim_state_dict = load_single_card_optimizer(self.args, model, optimizer, resume_from_checkpoint)
            return optim_state_dict

        has_merge_optimizer_safetensors = distributed_isfile(
            os.path.join(resume_from_checkpoint, SAFE_OPTIMIZER_INDEX_NAME)
        )
        # If not having merge optimizer, then load non-merge optimizer.
        if not has_merge_optimizer_safetensors:
            if self.args.data_parallel_rank == 0:
                returned_optim_state_dict = self.load_non_merge_optimizer(
                    model,
                    optimizer,
                    resume_from_checkpoint,
                )
                return returned_optim_state_dict
            else:
                return None

        local_resume = check_unified_optimizer(
            self.args, model, optimizer, resume_from_checkpoint, safe_serialization=True
        )
        if not local_resume:
            logger.info("Begin to dynamically load unified optimizer!")
            returned_optim_state_dict = load_unified_optimizer_dynamically(
                self.args, model, optimizer, resume_from_checkpoint, safe_serialization=True
            )
            return returned_optim_state_dict

        if self.args.data_parallel_rank == 0:
            returned_optim_state_dict = load_unified_optimizer_locally(
                self.args, model, optimizer, resume_from_checkpoint, safe_serialization=True
            )
            return returned_optim_state_dict
        return None

    def save_single_card_checkpoint(self, model_to_save, output_dir):
        """Save checkpoint for non-distributed environment."""

        state_dict = get_expected_state_dict(model_to_save)
        if isinstance(model_to_save, LoRAModel) or isinstance(model_to_save, PrefixModelForCausalLM):
            weight_filename = "peft_model-00001-of-00001.safetensors"
            index_filename = SAFE_PEFT_WEIGHTS_INDEX_NAME
        else:
            weight_filename = "model-00001-of-00001.safetensors"
            index_filename = SAFE_WEIGHTS_INDEX_NAME
        # get index json
        index_weight_file = {}
        total_size = 0
        for key, weight in state_dict.items():
            index_weight_file[key] = weight_filename
            total_size += weight.numel().item() * dtype_byte_size(weight.dtype)
        sharded_index_json = {}
        sharded_index_json["metadata"] = {"total_size": total_size}
        sharded_index_json["weight_map"] = index_weight_file
        if isinstance(model_to_save, LoRAModel):
            sharded_index_json["type"] = "lora"
        elif isinstance(model_to_save, PrefixModelForCausalLM):
            sharded_index_json["type"] = "ptuning"

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, index_filename)
        with open(path, "w") as f:
            json.dump(sharded_index_json, f, indent=4)

        # save checkpoint
        self._file_save_async_or_sync(
            state_dict, path=os.path.join(output_dir, weight_filename), is_sync=True, state_dict_type="model_weight"
        )

        if isinstance(model_to_save, PrefixModelForCausalLM):
            save_prefix_past_key_value(model_to_save, output_dir)
            model_to_save.prefix_config.save_pretrained(output_dir)
        if isinstance(model_to_save, LoRAModel):
            model_to_save.lora_config.save_pretrained(output_dir)

        config_to_save = save_config(model_to_save)
        config_to_save.architectures = [model_to_save.__class__.__name__]
        config_to_save.save_pretrained(output_dir)

    def save_single_card_optimizer(self, model, optimizer, output_dir):
        """ "Save optimizer for non-distributed environment."""
        # Split into optimizer params and master weights.
        optim_state_dict = nested_copy(optimizer.state_dict())
        master_weights = None
        if "master_weights" in optim_state_dict.keys():
            master_weights = optim_state_dict.pop("master_weights")
        if "LR_Scheduler" in optim_state_dict.keys():
            optim_state_dict.pop("LR_Scheduler")

        static2struct_name_mappings = {}
        state_dict = get_expected_state_dict(model)
        for k, v in state_dict.items():
            static2struct_name_mappings[v.name] = k

        # rename optimizer param
        for key in list(optim_state_dict.keys()):
            static_name, type_name = generate_base_static_name(key)
            new_name = static2struct_name_mappings[static_name] + "/" + type_name
            optim_state_dict[new_name] = optim_state_dict.pop(key)
        if master_weights is not None:
            for key in list(master_weights.keys()):
                master_weights[static2struct_name_mappings[key]] = master_weights.pop(key)

        # save index json
        index_optimizer_file, index_master_weight_file = {}, {}
        total_optim_size, total_master_weight_size = 0, 0
        for key, weight in optim_state_dict.items():
            index_optimizer_file[key] = "optimizer-00001-of-00001.safetensors"
            total_optim_size += weight.numel().item() * dtype_byte_size(weight.dtype)
        if master_weights is not None:
            for key, weight in master_weights.items():
                index_master_weight_file[key] = "master_weights-00001-of-00001.safetensors"
                total_master_weight_size += weight.numel().item() * dtype_byte_size(weight.dtype)
        path = os.path.join(output_dir, SAFE_OPTIMIZER_INDEX_NAME)
        master_path = os.path.join(output_dir, SAFE_MASTER_WEIGHTS_INDEX_NAME)
        with open(path, "w") as f:
            has_master_weights = master_weights is not None
            json.dump(
                {
                    "metadata": {"total_size": total_optim_size},
                    "weight_map": index_optimizer_file,
                    "master_weights": has_master_weights,
                },
                f,
                indent=4,
            )
        if master_weights is not None:
            with open(master_path, "w") as f:
                json.dump(
                    {"metadata": {"total_size": total_master_weight_size}, "weight_map": index_master_weight_file},
                    f,
                    indent=4,
                )

        # save optimizer state dict
        self._file_save_async_or_sync(
            optim_state_dict,
            path=os.path.join(output_dir, "optimizer-00001-of-00001.safetensors"),
            is_sync=True,
            state_dict_type="optimizer_weight",
        )
        if master_weights is not None:
            self._file_save_async_or_sync(
                master_weights,
                path=os.path.join(output_dir, "master_weights-00001-of-00001.safetensors"),
                is_sync=True,
                state_dict_type="master_weight",
            )

    def unlink_shared_memory(self):
        if not ("async_save" in self.args.unified_checkpoint_config):
            return

        if self._shared_save_model_flag is not None:
            while self._shared_save_model_flag[0] > 0:  # async process is saving
                time.sleep(0.5)
            self._shared_save_model_flag[0] = -1
        if self._shared_save_master_weight_flag is not None:
            while self._shared_save_master_weight_flag[0] > 0:
                time.sleep(0.5)
            self._shared_save_master_weight_flag[0] = -1
        if self._shared_save_optimizer_flag is not None:
            while self._shared_save_optimizer_flag[0] > 0:
                time.sleep(0.5)
            self._shared_save_optimizer_flag[0] = -1

        if self._shm_model_weight is not None:
            self._shm_model_weight.close()
            self._shm_model_weight.unlink()
            self._shm_model_weight = None
        if self._shm_master_weight is not None:
            self._shm_master_weight.close()
            self._shm_master_weight.unlink()
            self._shm_master_weight = None
        if self._shm_optimizer_weight is not None:
            self._shm_optimizer_weight.close()
            self._shm_optimizer_weight.unlink()
            self._shm_optimizer_weight = None

        dist.barrier()


def load_unified_checkpoint_locally(args, model, resume_from_checkpoint: str, safe_serialization=False):
    """
    Only dataset_rank == 0 can enter this function.
    """
    index_filename = select_model_weight_index(args, model, resume_from_checkpoint, safe_serialization, local=True)

    resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
        pretrained_model_name_or_path=resume_from_checkpoint,
        index_filename=os.path.join(resume_from_checkpoint, index_filename),
    )
    loaded_keys = sharded_metadata["all_checkpoint_keys"]

    model_state_dict = get_expected_state_dict(model)
    expected_keys = set(list(model_state_dict.keys()))
    missing_keys = expected_keys - set(loaded_keys)

    use_fast_set = True
    if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
        use_fast_set = False

    if len(missing_keys) > 0:
        raise ValueError(f"missing_keys: {missing_keys}")

    def _remove_unused_keys(
        state_dict,
        model_state_dict,
    ):
        unused_keys = set(state_dict.keys()) - set(model_state_dict.keys())
        for unused_key in unused_keys:
            del state_dict[unused_key]
        return unused_keys

    # This should always be a list but, just to be sure.
    if not isinstance(resolved_archive_file, list):
        resolved_archive_file = [resolved_archive_file]

    error_msgs = []

    if len(resolved_archive_file) > 1:
        resolved_archive_file = tqdm(resolved_archive_file, desc="Loading checkpoint shards")

    for shard_file in resolved_archive_file:
        # TODO: check if  no expected_keys in shard_file, then don't load it
        if expected_keys.isdisjoint(sharded_metadata["file_map"][os.path.split(shard_file)[-1]]):
            continue

        pre_tensor_parallel_split = False
        if shard_file.endswith(".safetensors") and model.config.tensor_parallel_degree > 1:
            pre_tensor_parallel_split = True
            assert loaded_keys is not None, "loaded_keys is not None."
            if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
                tp_actions = model._get_tensor_parallel_convert_actions(
                    set(loaded_keys), is_split=True, ignore_error=True
                )
            else:
                tp_actions = model.get_tensor_parallel_convert_actions(model.config, loaded_keys, ignore_error=True)
        # Here we use expected_keys to optimize weights loading for pipeline model. Only works for safetensors
        state_dict = load_state_dict(
            shard_file, tp_actions if pre_tensor_parallel_split else None, expected_keys, device="expected"
        )

        if not pre_tensor_parallel_split:
            # Since we load all keys but we only need one of pipeline stages
            _ = _remove_unused_keys(state_dict, model_state_dict)

        if model.config.tensor_parallel_degree > 1 and not pre_tensor_parallel_split:
            logger.info("Converting state_dict to Tensor Parallel Format")
            # ignore error for multi shard, since only parts of data
            state_dict = model.convert_tensor_parallel(
                None, model.config, state_dict=state_dict, ignore_error=len(resolved_archive_file) > 1
            )

        if use_fast_set:
            error_msgs += faster_set_state_dict(model, state_dict, strict_dtype=False)
        else:
            error_msgs += _load_state_dict_into_model(model, state_dict, "")

        # force memory release
        del state_dict
        # gc.collect()

    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        if " but the expected shape is" in error_msg:
            error_msg += (
                "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
            )
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")


def save_config(model_to_save):
    dtype = get_parameter_dtype(model_to_save)
    model_to_save.config.dtype = str(dtype).split(".")[1]
    config_to_save = copy.deepcopy(model_to_save.config)

    if config_to_save.tensor_parallel_degree > 1:
        # do we need to change?
        config_to_save.tensor_parallel_degree = 1

    return config_to_save


def unified_checkpoint_into_shards(
    args,
    model_to_save,
    safe_serialization=False,
):
    """Get state_dict and config to save

    Args:
        model_to_save (nn.Layer): model to, save
        safe_serialization (bool, optional): safe serialization using safetensors. Defaults to False.

    Returns:
        tuple: state_dict, config, shard_file: file name, sharded_index: map for weight to file name.
    """
    paddle.device.cuda.empty_cache()
    assert hasattr(model_to_save, "config")

    state_dict = get_expected_state_dict(model_to_save)
    all_filter_keys = filter_params(model_to_save, state_dict)

    config_to_save = copy.deepcopy(model_to_save.config)

    if config_to_save.tensor_parallel_degree > 1:
        if isinstance(model_to_save, LoRAModel) or isinstance(model_to_save, PrefixModelForCausalLM):
            tp_actions = model_to_save._get_tensor_parallel_convert_actions(
                all_filter_keys, is_split=False, ignore_error=True
            )
        else:
            tp_actions = model_to_save.get_tensor_parallel_convert_actions(
                model_to_save.config, state_dict.keys(), is_split=False, ignore_error=True
            )
        logger.info("Unified model tensor parallel weights in shards")
        state_dict = merge_tensor_parallel_with_shard(state_dict, tp_actions, all_filter_keys)

    # build index json file
    index_weight_file = {}
    total_size = 0
    if isinstance(model_to_save, LoRAModel):
        weights_name = SAFE_PEFT_WEIGHTS_NAME if safe_serialization else LORA_WEIGHTS_NAME
    elif isinstance(model_to_save, PrefixModelForCausalLM):
        weights_name = SAFE_PEFT_WEIGHTS_NAME if safe_serialization else PREFIX_WEIGHTS_NAME
    else:
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else PADDLE_WEIGHTS_NAME

    shard_file = get_sharded_file_name(args, weights_name)
    for key, weight in state_dict.items():
        index_weight_file[key] = shard_file
        total_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    index_file_list, total_size_list = gather_sharded_object(index_weight_file, total_size)
    sharded_index = get_sharded_index(
        index_file_list,
        total_size_list,
    )
    if sharded_index is not None:
        if isinstance(model_to_save, LoRAModel):
            sharded_index["type"] = "lora"
        elif isinstance(model_to_save, PrefixModelForCausalLM):
            sharded_index["type"] = "ptuning"

    paddle.device.cuda.empty_cache()

    return state_dict, shard_file, sharded_index


def load_unified_optimizer_locally(args, model, optimizer, resume_from_checkpoint, safe_serialization=False):
    # init and get optimizer LR_Scheduler
    returned_optim_state_dict = nested_copy(optimizer.state_dict())

    if not safe_serialization:
        index_filename, index_filename_master_weights = (
            PADDLE_OPTIMIZER_INDEX_NAME,
            PADDLE_MASTER_WEIGHTS_INDEX_NAME,
        )
    else:
        index_filename, index_filename_master_weights = SAFE_OPTIMIZER_INDEX_NAME, SAFE_MASTER_WEIGHTS_INDEX_NAME

    resolved_archive_file, sharded_metadata = get_optimizer_shard_files(
        optimizer_path=resume_from_checkpoint,
        index_filename=os.path.join(resume_from_checkpoint, index_filename),
    )
    has_master_weights = True if sharded_metadata["master_weights"] else False

    model_state_dict = get_expected_state_dict(model)
    model_keys = list(model_state_dict.keys())
    struct2static_name_mappings = {k: v.name for k, v in model_state_dict.items()}  # get optimizer param mappings

    expected_keys = get_expected_keys(sharded_metadata, model, optimizer)

    # This should always be a list but, just to be sure.
    if not isinstance(resolved_archive_file, list):
        resolved_archive_file = [resolved_archive_file]

    if len(resolved_archive_file) > 1:
        resolved_archive_file = tqdm(resolved_archive_file, desc="Loading optimizer shards")

    # update has_master_weights and index_filename_master_weights
    # 1. if the master weight exists, only has_master_weights is set True and loaded when needed
    # 2. if master weight does not exist, convert model weight to master weight when needed
    has_master_weights, index_filename_master_weights = update_master_weight_status(
        args, optimizer, has_master_weights, safe_serialization
    )

    if has_master_weights:
        returned_optim_state_dict["master_weights"] = {}

        resolved_archive_file_mw, sharded_metadata_mw = get_optimizer_shard_files(
            optimizer_path=resume_from_checkpoint,
            index_filename=os.path.join(resume_from_checkpoint, index_filename_master_weights),
        )

        expected_keys_mw = get_expected_keys(sharded_metadata_mw, model, optimizer)
        if not isinstance(resolved_archive_file_mw, list):
            resolved_archive_file_mw = [resolved_archive_file_mw]
        if len(resolved_archive_file_mw) > 1:
            resolved_archive_file_mw = tqdm(resolved_archive_file_mw, desc="Loading master weights shards")

    def load_resolved_archive_file(resolved_archive_file, sharded_metadata, expected_keys, is_master_weights=False):
        returned_state_dict = {}
        # load optimizer
        for shard_file in resolved_archive_file:
            # TODO: check if no expected_keys in shard_file, then don't load it
            if expected_keys.isdisjoint(sharded_metadata["file_map"][os.path.split(shard_file)[-1]]):
                continue

            if shard_file.endswith(".safetensors"):
                # assert model_keys is not None, "model_keys is None." TODO: correct the assert
                if model.config.tensor_parallel_degree > 1:
                    if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
                        tp_actions = model._get_tensor_parallel_convert_actions(
                            model_keys, is_split=True, ignore_error=True
                        )
                    else:
                        tp_actions = model.get_tensor_parallel_convert_actions(
                            model.config, model_keys, ignore_error=True
                        )
                    if not is_master_weights:
                        tp_actions = mapping_optimizer_tp_actions(tp_actions, expected_keys)

                    # Here we use expected_keys to optimize weights loading for pipeline model. Only works for safetensors
                    state_dict = load_state_dict(shard_file, tp_actions, expected_keys, device="expected")
                else:
                    # for pipeline model, we don't need to use tp_actions
                    state_dict = load_state_dict(shard_file, None, expected_keys, device="expected")

            returned_state_dict.update(state_dict)
            # force memory release
            del state_dict
            gc.collect()
        return returned_state_dict

    state_dict_optim = load_resolved_archive_file(resolved_archive_file, sharded_metadata, expected_keys)
    if has_master_weights:
        state_dict_master_weight = load_resolved_archive_file(
            resolved_archive_file_mw, sharded_metadata_mw, expected_keys_mw, is_master_weights=True
        )
    # rename optimizer param
    for key in list(state_dict_optim.keys()):
        key_name = key.split("/")
        static_name = struct2static_name_mappings[key_name[0]]
        if has_master_weights:
            key_name = "_".join([static_name, FP32_MASTER, key_name[1]])
        else:
            key_name = "_".join([static_name, key_name[1]])
        returned_optim_state_dict[key_name] = state_dict_optim.pop(key)
        returned_optim_state_dict[key_name].name = key_name

    if has_master_weights:
        for key in list(state_dict_master_weight.keys()):
            static_name = struct2static_name_mappings[key]
            returned_optim_state_dict["master_weights"][static_name] = state_dict_master_weight.pop(key)
            returned_optim_state_dict["master_weights"][static_name].name = "_".join([static_name, FP32_MASTER])

    return returned_optim_state_dict


def unified_optimizer_into_shards(
    args,
    model,
    optimizer,
    safe_serialization=False,
):
    """Get optimizer state dict and master weight state dict.

    Args:
        optimizer (Optimizer): optimizer to save.
        safe_serialization (bool, optional): safe serialization using safetensors. Defaults to False.
    """
    paddle.device.cuda.empty_cache()
    optim_state_dict = nested_copy(optimizer.state_dict())
    master_weights = None
    if "master_weights" in optim_state_dict.keys():
        master_weights = optim_state_dict["master_weights"]
        optim_state_dict.pop("master_weights")
    if "LR_Scheduler" in optim_state_dict.keys():
        optim_state_dict.pop("LR_Scheduler")

    # gather global master_weights status.
    global_master_weights = reduce_master_weights_status(master_weights is not None)
    if master_weights is None and global_master_weights:
        master_weights = {}

    # get optimizer param mappings
    static2struct_name_mappings = {}
    state_dict = get_expected_state_dict(model)
    for k, v in state_dict.items():
        static2struct_name_mappings[v.name] = k

    # rename optimizer param
    for key in list(optim_state_dict.keys()):
        static_name, type_name = generate_base_static_name(key)
        new_name = static2struct_name_mappings[static_name] + "/" + type_name
        optim_state_dict[new_name] = optim_state_dict.pop(key)
    if master_weights is not None:
        for key in list(master_weights.keys()):
            master_weights[static2struct_name_mappings[key]] = master_weights.pop(key)

    # filter optimizer param
    if master_weights is not None:
        filter_master_keys = filter_params(model, master_weights, is_optimizer=True)
    filter_optim_keys = filter_params(model, optim_state_dict, is_optimizer=True)

    tp_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
    tp_size = tp_group.nranks

    if tp_size > 1:
        # get tp_actions
        model_keys = []
        for key in optim_state_dict.keys():
            base_model_key = key.split("/")[0]
            if base_model_key not in model_keys:
                model_keys.append(base_model_key)
        if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
            tp_actions = model._get_tensor_parallel_convert_actions(model_keys, is_split=False, ignore_error=True)
        else:
            tp_actions = model.get_tensor_parallel_convert_actions(
                model.config, model_keys, is_split=False, ignore_error=True
            )
        logger.info("Unified optimizer tensor parallel in shards")
        optim_state_dict = merge_tensor_parallel_for_optimizer(
            optim_state_dict,
            tp_actions,
            filter_optim_keys,
        )
        paddle.device.cuda.empty_cache()

        if master_weights is not None:
            logger.info("Unified master weight tensor parallel in shards")
            master_weights = merge_tensor_parallel_for_optimizer(
                master_weights,
                tp_actions,
                filter_master_keys,
            )
            paddle.device.cuda.empty_cache()

    # build index json file
    index_optimizer_file, index_master_weight_file = {}, {}
    total_optim_size, total_master_weight_size = 0, 0
    optimizer_name = SAFE_OPTIMIZER_NAME if safe_serialization else PADDLE_OPTIMIZER_NAME
    master_weights_name = SAFE_MASTER_WEIGHTS_NAME if safe_serialization else PADDLE_MASTER_WEIGHTS_NAME
    if UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value in args.unified_checkpoint_config:
        master_weights_name = SAFE_WEIGHTS_NAME if safe_serialization else PADDLE_WEIGHTS_NAME
    shard_optimizer_file = get_sharded_file_name(args, optimizer_name, is_optimizer=True)
    shard_master_weight_file = get_sharded_file_name(args, master_weights_name, is_optimizer=True)

    for key, weight in optim_state_dict.items():
        index_optimizer_file[key] = shard_optimizer_file
        total_optim_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    if master_weights is not None:
        for key, weight in master_weights.items():
            index_master_weight_file[key] = shard_master_weight_file
            total_master_weight_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    index_optimizer_filelist, total_optim_size_list = gather_sharded_object(
        index_optimizer_file, total_optim_size, is_optimizer=True
    )
    sharded_optim_index = get_sharded_index(index_optimizer_filelist, total_optim_size_list)
    if master_weights is not None:
        index_master_weight_filelist, total_master_weight_size_list = gather_sharded_object(
            index_master_weight_file, total_master_weight_size, is_optimizer=True
        )
        sharded_master_weight_index = get_sharded_index(index_master_weight_filelist, total_master_weight_size_list)

    if sharded_optim_index is not None:
        if master_weights is not None:
            sharded_optim_index["master_weights"] = True
        else:
            sharded_optim_index["master_weights"] = False

    paddle.device.cuda.empty_cache()
    if master_weights is None:
        return [(optim_state_dict, shard_optimizer_file, sharded_optim_index)]
    else:
        return [
            (optim_state_dict, shard_optimizer_file, sharded_optim_index),
            (master_weights, shard_master_weight_file, sharded_master_weight_index),
        ]


def check_unified_checkpoint(args, model, resume_from_checkpoint, safe_serialization=False):
    index_filename = select_model_weight_index(args, model, resume_from_checkpoint, safe_serialization, local=False)
    index_filename = os.path.join(resume_from_checkpoint, index_filename)
    # Find index json file and distribute this file in global group.
    if distributed_isfile(index_filename):
        distributed_file(index_filename)
    else:
        raise Exception(
            f"Sorry, we can not find {index_filename}. This file should be appear at least on one machine."
        )

    with open(index_filename, "r") as f:
        index = json.loads(f.read())
    all_weight_filenames = sorted(set(index["weight_map"].values()))

    # Get existed weight file list on current machine.
    existed_filelist = []
    existed_files = []
    for filename in os.listdir(resume_from_checkpoint):
        if filename in all_weight_filenames:
            existed_files.append(filename)

    # Gather all the existed files in global group.
    dist.all_gather_object(existed_filelist, existed_files)
    flatten_existed_filelist = flatten_list(existed_filelist)
    diff_filelist = list(set(all_weight_filenames).difference(set(flatten_existed_filelist)))
    if len(diff_filelist) != 0:
        raise Exception(f"Sorry, the weight file list on the machines is not complete!, missing {diff_filelist}")

    # To decide whether to load the checkpoint locally, or need to dynamically send tensors across machines.
    local_resume = True
    if args.dataset_rank == 0:
        hcg = fleet.get_hybrid_communicate_group()
        tp_group = hcg.get_model_parallel_group()
        pp_group = hcg.get_pipe_parallel_group()

        need_files = set()
        state_dict = get_expected_state_dict(model)
        for key in state_dict.keys():
            filename = index["weight_map"][key]
            need_files.add(filename)
        diff_filelist = list(need_files.difference(set(existed_files)))
        num_diff = paddle.to_tensor([len(diff_filelist)])
        if tp_group.nranks > 1:
            dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=tp_group)
        if pp_group.nranks > 1:
            dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=pp_group)
        if num_diff.item() == 0:
            local_resume = True
        else:
            local_resume = False
    local_resume = paddle.to_tensor([local_resume])
    dist.all_reduce(local_resume, op=dist.ReduceOp.PROD)
    local_resume = local_resume.item()
    return local_resume


def check_unified_optimizer(args, model, optimizer, resume_from_checkpoint, safe_serialization=False):
    if not safe_serialization:
        index_filename, index_filename_master_weights = PADDLE_OPTIMIZER_INDEX_NAME, PADDLE_MASTER_WEIGHTS_INDEX_NAME
    else:
        index_filename, index_filename_master_weights = SAFE_OPTIMIZER_INDEX_NAME, SAFE_MASTER_WEIGHTS_INDEX_NAME
    index_filename = os.path.join(resume_from_checkpoint, index_filename)
    index_filename_master_weights = os.path.join(resume_from_checkpoint, index_filename_master_weights)

    # Find index json file and distribute the file in global group.
    if distributed_isfile(index_filename):
        distributed_file(index_filename)
    else:
        raise Exception(
            f"Sorry, we can not find {index_filename}. This file should be appear at least on one machine."
        )

    with open(index_filename, "r") as f:
        index = json.loads(f.read())
    all_optimizer_filenames = sorted(set(index["weight_map"].values()))

    has_master_weights = index["master_weights"]
    # update has_master_weights and index_filename_master_weights
    # 1. if the master weight exists, only has_master_weights is set True and loaded when needed
    # 2. if master weight does not exist, convert model weight to master weight when needed
    has_master_weights, index_filename_master_weights = update_master_weight_status(
        args, optimizer, has_master_weights, safe_serialization
    )
    if has_master_weights:
        index_filename_master_weights = os.path.join(resume_from_checkpoint, index_filename_master_weights)
        if distributed_isfile(index_filename_master_weights):
            distributed_file(index_filename_master_weights)
        else:
            raise Exception(
                f"Sorry, we can not find {index_filename_master_weights}. This file should be appear at least on one machine."
            )
        with open(index_filename_master_weights, "r") as f:
            index_mw = json.loads(f.read())
        all_mw_filenames = sorted(set(index_mw["weight_map"].values()))

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()
    sharding_rank = sharding_group.rank
    struct2static_name_mappings = {k: v.name for k, v in model.state_dict().items()}
    if sharding_group.nranks > 1:
        param2rank = optimizer._param2rank

    def check_complete(all_filenames):
        # Check whether the checkpoint files on machines are complete. If not complete, raise Exception.
        existed_filelist = []
        existed_files = []
        for filename in os.listdir(resume_from_checkpoint):
            if filename in all_filenames:
                existed_files.append(filename)

        dist.all_gather_object(existed_filelist, existed_files)
        flatten_existed_filelist = flatten_list(existed_filelist)
        diff_filelist = list(set(all_filenames).difference(set(flatten_existed_filelist)))
        if len(diff_filelist) != 0:
            raise Exception(
                f"Sorry, the optimizer file list on `data_parallel_rank==0` machines is not complete!, missing {diff_filelist}"
            )
        return existed_files

    def check_dynamic_load(args, weight_map, existed_files, is_master_weights=False, typename_set=None):
        # To decide whether to load the checkpoint locally, or need to dynamically distribute the checkpoint.
        local_resume = True
        if args.data_parallel_rank == 0:
            need_files = set()
            state_dict = get_expected_state_dict(model)
            for key in state_dict.keys():
                if sharding_group.nranks > 1:
                    static_name = struct2static_name_mappings.get(key, None)
                    param_rank = param2rank.get(static_name, None)
                    if param_rank != sharding_rank:
                        continue

                if not is_master_weights:
                    for type_name in typename_set:
                        type_key = key + "/" + type_name
                        filename = weight_map[type_key]
                        need_files.add(filename)
                else:
                    filename = weight_map[key]
                    need_files.add(filename)

            diff_filelist = list(need_files.difference(set(existed_files)))
            num_diff = paddle.to_tensor([len(diff_filelist)])
            if tp_group.nranks > 1:
                dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=tp_group)
            if pp_group.nranks > 1:
                dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=pp_group)
            if sharding_group.nranks > 1:
                dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=sharding_group)

            if num_diff.item() == 0:
                local_resume = True
            else:
                local_resume = False
        local_resume = paddle.to_tensor([local_resume])
        dist.all_reduce(local_resume, op=dist.ReduceOp.PROD)
        return local_resume.item()

    # check whether the optimizer checkpoint files are complete.
    existed_files = check_complete(all_optimizer_filenames)
    if has_master_weights:
        existed_files_mw = check_complete(all_mw_filenames)
    # get optimizer's param type name, like moment1_0.
    typename_set = set()
    for key in index["weight_map"].keys():
        _, typename = key.split("/")
        typename_set.add(typename)
    local_resume = check_dynamic_load(
        args, index["weight_map"], existed_files, is_master_weights=False, typename_set=typename_set
    )
    local_resume_rw = True
    if has_master_weights:
        local_resume_rw = check_dynamic_load(args, index_mw["weight_map"], existed_files_mw, is_master_weights=True)
    return local_resume & local_resume_rw


def save_prefix_past_key_value(model_to_save, save_directory):
    past_key_value = model_to_save.prefix_encoder(model_to_save.prefix_tokens.unsqueeze(0).expand([1, -1]))
    past_key_value = past_key_value.reshape(
        [
            model_to_save.prefix_config.num_prefix_tokens,
            2,
            model_to_save.prefix_config.num_hidden_layers,
            model_to_save.num_heads,
            model_to_save.head_dim,
        ]
    )
    past_key_value = paddle.transpose(past_key_value, perm=[2, 1, 3, 0, 4]).cpu().numpy()
    model_to_save.prefix_config.save_pretrained(save_directory)
    np.save(os.path.join(save_directory, PAST_KEY_VALUES_FILE_NAME), past_key_value)


def get_expected_state_dict(model_to_save):
    if isinstance(model_to_save, PretrainedModel):
        state_dict = model_to_save.state_dict()
        if (
            hasattr(model_to_save.config, "tie_word_embeddings")
            and model_to_save.config.tie_word_embeddings
            and hasattr(model_to_save, "_tied_weights_keys")
            and model_to_save._tied_weights_keys is not None
        ):
            for key in model_to_save._tied_weights_keys:
                if key in state_dict:
                    state_dict.pop(key)
    elif isinstance(model_to_save, LoRAModel):
        state_dict = model_to_save.get_trainable_state_dict()
    elif isinstance(model_to_save, PrefixModelForCausalLM):
        state_dict = model_to_save.prefix_encoder.state_dict()

    return state_dict


def create_dispatch_table(args, model, file_keyname_mappings, file_machine_mappings, resume_from_checkpoint):
    """Create dispatch table for dynamically loading state dict.

    Args:
        args
    """

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    tp_rank = tp_group.rank

    # Create tensor receive table, contains {"key0": [global_rank, tp_rank], "key1": [global_rank, tp_rank]}
    dispatch_list = []
    recv_table = {}
    if args.dataset_rank == 0:
        state_dict = get_expected_state_dict(model)
        for (k, v) in state_dict.items():
            if hasattr(v, "is_distributed") and v.is_distributed:
                recv_table[k] = [(dist.get_rank(), tp_rank)]
            else:
                recv_table[k] = [(dist.get_rank(), -1)]

    # Gather receive table in global group.
    dist.all_gather_object(dispatch_list, recv_table)
    recv_table = {}
    for dl in dispatch_list:
        for key, value in dl.items():
            if key not in recv_table:
                recv_table[key] = value
            else:
                recv_table[key] += value

    # Create send table, to decide which worker to send the key. Contains {"key0:" global_rank, "key1": global_rank, ...}
    send_table = create_send_table(file_keyname_mappings, file_machine_mappings)

    return send_table, recv_table


def create_optimizer_dispatch_table(
    args,
    model,
    optimizer,
    file_keyname_mappings,
    file_machine_mappings,
    resume_from_checkpoint,
    struct2static_name_mappings,
    is_master_weights=False,
    typename_set=None,
):
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()
    sharding_rank = sharding_group.rank
    if sharding_group.nranks > 1:
        param2rank = optimizer._param2rank
    tp_rank = tp_group.rank

    # Create receive table, contains {"param_key0": [global_rank, tp_rank], "param_key1": [global_rank, tp_rank]}
    dispatch_list = []
    recv_table = {}
    if args.data_parallel_rank == 0:
        state_dict = get_expected_state_dict(model)
        for (k, v) in state_dict.items():
            if sharding_group.nranks > 1:
                static_name = struct2static_name_mappings[k]
                param_rank = param2rank.get(static_name, None)
                if param_rank != sharding_rank:
                    continue
            if is_master_weights:
                if hasattr(v, "is_distributed") and v.is_distributed:
                    recv_table[k] = [(dist.get_rank(), tp_rank)]
                else:
                    recv_table[k] = [(dist.get_rank(), -1)]
            else:
                for typename in typename_set:
                    type_key = k + "/" + typename
                    if typename in optimizer_non_scaler_name:
                        if hasattr(v, "is_distributed") and v.is_distributed:
                            recv_table[type_key] = [(dist.get_rank(), tp_rank)]
                        else:
                            recv_table[type_key] = [(dist.get_rank(), -1)]
                    else:
                        recv_table[type_key] = [(dist.get_rank(), -1)]

    dist.all_gather_object(dispatch_list, recv_table)
    recv_table = {}
    for dl in dispatch_list:
        for k, v in dl.items():
            if k not in recv_table:
                recv_table[k] = v
            else:
                recv_table[k] += v

    # Create send table, to decide which worker to send the key. Contains {"param_key0:" 0, "param_key1": 1, ...}
    send_table = create_send_table(file_keyname_mappings, file_machine_mappings)
    return send_table, recv_table


def load_unified_checkpoint_dynamically(args, model, optimizer, resume_from_checkpoint, safe_serialization=False):
    index_filename = select_model_weight_index(args, model, resume_from_checkpoint, safe_serialization, local=False)
    index_filename = os.path.join(resume_from_checkpoint, index_filename)

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    # `file_keyname_mappings` indicates which keys each file contains. For example, {"model-00001-of-00002.safetensors": ["llama.embed_tokens.weight", "llama.layers.0.self_attn.q_proj.weight", ...]}
    # `file_machine_mappings` indicates the machine where the files appear. For example, {"model-00001-of-00002.safetensors": [machine_0, machine_1], "model-00002-of-00002.safetensors": [machine_0]}
    file_keyname_mappings, file_machine_mappings = get_file_mappings(index, resume_from_checkpoint)

    logger.debug("Creating dispatch table for unified checkpoint load ...")
    # Get send_table and recv_table. The send table indicates which workers are responsible for sending tensors, and the recv table indicates which workers should receive the tensors.
    send_table, recv_table = create_dispatch_table(
        args, model, file_keyname_mappings, file_machine_mappings, resume_from_checkpoint
    )

    # Get all the keys that are splited by tensor parallelism.
    all_tp_keys = set()
    for k, v in recv_table.items():
        if v[0][1] != -1:
            all_tp_keys.add(k)

    config_revise = copy.deepcopy(model.config)
    config_revise.tensor_parallel_rank = None
    if len(all_tp_keys) == 0:
        tp_actions = {}
    else:
        # Get corresponding tensor parallel actions.
        if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
            tp_actions = model._get_tensor_parallel_convert_actions(
                set(all_tp_keys), is_split=True, ignore_error=True, config=config_revise
            )
        else:
            tp_actions = model.get_tensor_parallel_convert_actions(config_revise, all_tp_keys, ignore_error=True)

    logger.debug("Distributed send recv for state dict load ...")
    # Distribute the checkpoint tensor dynamically, using the `send_table` and `recv_table` we create before.
    state_dict = distributed_send_recv(
        config_revise,
        get_expected_state_dict(model),
        tp_actions,
        send_table,
        recv_table,
        resume_from_checkpoint,
        file_keyname_mappings,
        file_machine_mappings,
    )
    dist.barrier()
    logger.debug("Setting state dict into model ...")
    error_msgs = _load_state_dict_into_model(model, state_dict, "")
    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        raise RuntimeError(f"Error(s) in loading dynamic state_dict for {model.__class__.__name__}:\n\t{error_msg}")


def load_unified_optimizer_dynamically(args, model, optimizer, resume_from_checkpoint, safe_serialization=False):
    optim_state_dict = nested_copy(optimizer.state_dict())
    if "master_weights" in optim_state_dict.keys():
        optim_state_dict.pop("master_weights")

    if safe_serialization:
        index_filename, index_filename_mw = SAFE_OPTIMIZER_INDEX_NAME, SAFE_MASTER_WEIGHTS_INDEX_NAME
    else:
        index_filename, index_filename_mw = PADDLE_OPTIMIZER_INDEX_NAME, PADDLE_MASTER_WEIGHTS_INDEX_NAME

    with open(os.path.join(resume_from_checkpoint, index_filename), "r") as f:
        index = json.loads(f.read())

    # `file_keyname_mappings` indicates which keys each file contains. For example, {"optimizer-00001-of-00002.safetensors": ["llama.embed_tokens.weight/moment1_0", "llama.layers.1.mlp.gate_proj.weight/moment1_0", ...]}
    # `file_machine_mappings` indicates the machine where the files appear. For example, {"optimizer-00001-of-00002.safetensors": [machine_0, machine_1], "optimizer-00002-of-00002.safetensors": [machine_0]}
    file_keyname_mappings, file_machine_mappings = get_file_mappings(index, resume_from_checkpoint)

    has_master_weights = index["master_weights"]
    # update has_master_weights and index_filename_master_weights
    # 1. if the master weights exists, only has_master_weights is set True and load master weights when needed
    # 2. if master weights does not exist, convert model weights to master weights when needed
    has_master_weights, index_filename_mw = update_master_weight_status(
        args, optimizer, has_master_weights, safe_serialization
    )

    if has_master_weights:
        with open(os.path.join(resume_from_checkpoint, index_filename_mw), "r") as f:
            index_mw = json.loads(f.read())
        file_keyname_mappings_mw, file_machine_mappings_mw = get_file_mappings(index_mw, resume_from_checkpoint)

    # Get optimizer param type name, like moment1_0, moment2_0, beta1_pow_acc_0.
    typename_set = set()
    for key in index["weight_map"].keys():
        _, typename = key.split("/")
        typename_set.add(typename)
    struct2static_name_mappings = {k: v.name for k, v in get_expected_state_dict(model).items()}
    static2struct_name_mappings = {v.name: k for k, v in get_expected_state_dict(model).items()}
    # Get send_table and recv_table. The send table indicates which workers are responsible for sending tensors, and the recv table indicates which workers should receive the tensors.
    send_table, recv_table = create_optimizer_dispatch_table(
        args,
        model,
        optimizer,
        file_keyname_mappings,
        file_machine_mappings,
        resume_from_checkpoint,
        struct2static_name_mappings,
        is_master_weights=False,
        typename_set=typename_set,
    )
    if has_master_weights:
        send_table_mw, recv_table_mw = create_optimizer_dispatch_table(
            args,
            model,
            optimizer,
            file_keyname_mappings_mw,
            file_machine_mappings_mw,
            resume_from_checkpoint,
            struct2static_name_mappings,
            is_master_weights=True,
        )

    # Initialize optimizer state dict.
    hcg = fleet.get_hybrid_communicate_group()
    sharding_group = hcg.get_sharding_parallel_group()
    if sharding_group.nranks > 1:
        param2rank = optimizer._param2rank
    optim_state_dict_mw = {}

    def check_optimizer_param(parameter):
        if sharding_group.nranks > 1:
            param_rank = param2rank.get(parameter.name, None)
            if param_rank != sharding_group.rank:
                return False
        if parameter.stop_gradient:
            return False
        return True

    optimizer_keys_with_shape = []
    if isinstance(optimizer._parameter_list[0], dict):
        for param_group in optimizer._parameter_list:
            # If parameter groups are set, there must be `params` key. This is guaranteed by the optimizer's initialization code.
            for parameter in param_group["params"]:
                if check_optimizer_param(parameter):
                    optimizer_keys_with_shape.append((parameter.name, parameter.shape))
    else:
        for parameter in optimizer._parameter_list:
            if check_optimizer_param(parameter):
                optimizer_keys_with_shape.append((parameter.name, parameter.shape))

    # see how to change
    for static_name, shape in optimizer_keys_with_shape:
        k = static2struct_name_mappings[static_name]
        for typename in typename_set:
            new_k = k + "/" + typename
            if typename in optimizer_scalar_name:
                optim_state_dict[new_k] = paddle.empty([1], dtype="float32")
            else:
                optim_state_dict[new_k] = paddle.empty(shape, dtype="float32")
        if has_master_weights:
            optim_state_dict_mw[k] = paddle.empty(shape, dtype="float32")

    # Get all the keys that are splited by tensor parallelism.
    all_tp_keys = set()
    for k, v in recv_table.items():
        structure_name, typename = k.split("/")
        if typename in optimizer_non_scaler_name:
            if v[0][1] != -1:
                all_tp_keys.add(structure_name)

    # Get corresponding tensor parallel actions.
    config_revise = copy.deepcopy(model.config)
    config_revise.tensor_parallel_rank = None
    if len(all_tp_keys) == 0:
        tp_actions = {}
    else:
        if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
            tp_actions = model._get_tensor_parallel_convert_actions(
                set(all_tp_keys), is_split=True, ignore_error=True, config=config_revise
            )
        else:
            tp_actions = model.get_tensor_parallel_convert_actions(config_revise, all_tp_keys, ignore_error=True)
    optimizer_keys = list(index["weight_map"].keys())
    optimizer_tp_actions = mapping_optimizer_tp_actions(tp_actions, optimizer_keys)
    if has_master_weights:
        optimizer_tp_actions.update(tp_actions)

    # Distribute the optimizer checkpoint dynamically, using the `send_table` and `recv_table` we create before.
    optim_state_dict = distributed_send_recv(
        config_revise,
        optim_state_dict,
        optimizer_tp_actions,
        send_table,
        recv_table,
        resume_from_checkpoint,
        file_keyname_mappings,
        file_machine_mappings,
    )
    dist.barrier()
    if has_master_weights:
        optim_state_dict_mw = distributed_send_recv(
            config_revise,
            optim_state_dict_mw,
            optimizer_tp_actions,
            send_table_mw,
            recv_table_mw,
            resume_from_checkpoint,
            file_keyname_mappings_mw,
            file_machine_mappings_mw,
        )
        dist.barrier()

    # Rename optimizer state dict.
    for key in list(optim_state_dict.keys()):
        if key == "LR_Scheduler":
            continue
        key_name = key.split("/")
        static_name = struct2static_name_mappings[key_name[0]]
        if has_master_weights:
            key_name = "_".join([static_name, FP32_MASTER, key_name[1]])
        else:
            key_name = "_".join([static_name, key_name[1]])
        optim_state_dict[key_name] = optim_state_dict.pop(key)
        optim_state_dict[key_name].name = key_name

    if has_master_weights:
        optim_state_dict["master_weights"] = {}
        for key in list(optim_state_dict_mw.keys()):
            static_name = struct2static_name_mappings[key]
            optim_state_dict["master_weights"][static_name] = optim_state_dict_mw.pop(key)
            optim_state_dict["master_weights"][static_name].name = "_".join([static_name, FP32_MASTER])

    if args.data_parallel_rank == 0:
        return optim_state_dict
    return None


def load_single_card_checkpoint(args, model, resume_from_checkpoint: str):
    if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
        index_filename = SAFE_PEFT_WEIGHTS_INDEX_NAME
    else:
        index_filename = SAFE_WEIGHTS_INDEX_NAME
    resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
        pretrained_model_name_or_path=resume_from_checkpoint,
        index_filename=os.path.join(resume_from_checkpoint, index_filename),
    )

    loaded_keys = sharded_metadata["all_checkpoint_keys"]
    model_state_dict = get_expected_state_dict(model)
    expected_keys = set(list(model_state_dict.keys()))
    missing_keys = expected_keys - set(loaded_keys)

    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys: {missing_keys}")

    state_dict = load_state_dict(resolved_archive_file[0], None, expected_keys)
    error_msgs = _load_state_dict_into_model(model, state_dict, "")
    del state_dict
    gc.collect()

    if error_msgs:
        raise RuntimeError(f"Error(s) in loading state dict for {model.__class__.__name__}:\n\t{error_msgs}")


def load_single_card_optimizer(args, model, optimizer, resume_from_checkpoint: str):
    returned_optim_state_dict = nested_copy(optimizer.state_dict())

    resolved_archive_file, sharded_metadata = get_optimizer_shard_files(
        optimizer_path=resume_from_checkpoint,
        index_filename=os.path.join(resume_from_checkpoint, SAFE_OPTIMIZER_INDEX_NAME),
    )
    has_master_weights = True if sharded_metadata["master_weights"] else False

    model_state_dict = get_expected_state_dict(model)
    struct2static_name_mappings = {k: v.name for k, v in model_state_dict.items()}
    expected_keys = sharded_metadata["all_optimizer_keys"]

    if has_master_weights:
        returned_optim_state_dict["master_weights"] = {}
        resolved_archive_file_mw, sharded_metadata_mw = get_optimizer_shard_files(
            optimizer_path=resume_from_checkpoint,
            index_filename=os.path.join(resume_from_checkpoint, SAFE_MASTER_WEIGHTS_INDEX_NAME),
        )
        expected_keys_mw = sharded_metadata_mw["all_optimizer_keys"]

    state_dict_optim = load_state_dict(resolved_archive_file[0], None, expected_keys)
    if has_master_weights:
        state_dict_optim_mw = load_state_dict(resolved_archive_file_mw[0], None, expected_keys_mw)

    for key in list(state_dict_optim.keys()):
        key_name = key.split("/")
        static_name = struct2static_name_mappings[key_name[0]]
        if has_master_weights:
            key_name = "_".join([static_name, FP32_MASTER, key_name[1]])
        else:
            key_name = "_".join([static_name, key_name[1]])
        returned_optim_state_dict[key_name] = state_dict_optim.pop(key)
        returned_optim_state_dict[key_name].name = key_name
    if has_master_weights:
        for key in list(state_dict_optim_mw.keys()):
            static_name = struct2static_name_mappings[key]
            returned_optim_state_dict["master_weights"][static_name] = state_dict_optim_mw.pop(key)
            returned_optim_state_dict["master_weights"][static_name].name = "_".join([static_name, FP32_MASTER])

    returned_optim_state_dict = nested_copy_place(
        returned_optim_state_dict,
        place=paddle.framework._current_expected_place(),
        blocking=True,
    )
    return returned_optim_state_dict


def get_file_mappings(index, resume_from_checkpoint):
    file_keyname_mappings = {}
    for k, v in index["weight_map"].items():
        if v not in file_keyname_mappings:
            file_keyname_mappings[v] = []
        file_keyname_mappings[v].append(k)
    for k in file_keyname_mappings.keys():
        file_keyname_mappings[k] = sorted(file_keyname_mappings[k])

    local_device_count = int(os.getenv("PADDLE_LOCAL_SIZE"))
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
    global_rank = dist.get_rank()
    file_machine_mappings = {}
    for filename in file_keyname_mappings.keys():
        if local_rank == 0 and os.path.exists(os.path.join(resume_from_checkpoint, filename)):
            file_machine_mappings[filename] = [global_rank // local_device_count]
    file_machine_list = []
    dist.all_gather_object(file_machine_list, file_machine_mappings)
    file_machine_mappings = {}
    for mappings in file_machine_list:
        for k, v in mappings.items():
            if k not in file_machine_mappings:
                file_machine_mappings[k] = v
            else:
                file_machine_mappings[k] += v
    return file_keyname_mappings, file_machine_mappings


def create_send_table(file_keyname_mappings, file_machine_mappings):
    send_table = {}
    global_rank = dist.get_rank()
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
    local_device_count = int(os.getenv("PADDLE_LOCAL_SIZE"))
    for filename, keys in file_keyname_mappings.items():
        machine = file_machine_mappings[filename][0]
        is_src = (global_rank // local_device_count) == machine
        for i, key in enumerate(keys):
            if is_src and local_rank == i % local_device_count:
                send_table[key] = global_rank
    dispatch_list = []
    dist.all_gather_object(dispatch_list, send_table)
    send_table = {}
    for dl in dispatch_list:
        send_table.update(dl)
    return send_table


def distributed_send_recv(
    config,
    state_dict,
    tp_actions,
    send_table,
    recv_table,
    resume_from_checkpoint,
    file_keyname_mappings,
    file_machine_mappings,
):

    local_device_count = int(os.getenv("PADDLE_LOCAL_SIZE"))
    global_rank = dist.get_rank()
    for filename in file_keyname_mappings.keys():
        machine = file_machine_mappings[filename][0]
        is_src = global_rank // local_device_count == machine
        if is_src:
            f = safe_open(os.path.join(resume_from_checkpoint, filename), framework="np")

        for key in file_keyname_mappings[filename]:
            recv_info = recv_table[key]
            recv_ranklist = [a for (a, b) in recv_info]
            if is_src and global_rank == send_table[key]:
                py_safe_slice_ = f.get_slice(key)
                # send
                if key in tp_actions:
                    weight = tp_actions[key](py_safe_slice_)
                    # copy weight to GPU
                    for j in range(len(weight)):
                        with device_guard():
                            weight[j] = paddle.Tensor(weight[j], zero_copy=True)
                        weight[j] = weight[j]._copy_to(paddle.framework._current_expected_place(), False)

                    for recv_rank, split_index in recv_info:
                        if recv_rank == global_rank:
                            state_dict[key] = weight[split_index]
                        else:
                            dist.stream.send(weight[split_index], dst=recv_rank)
                else:
                    # no need to tp split
                    weight = py_safe_slice_[:]
                    with device_guard():
                        weight = paddle.Tensor(weight, zero_copy=True)
                    weight = weight._copy_to(paddle.framework._current_expected_place(), False)
                    for recv_rank, _ in recv_info:
                        if recv_rank == global_rank:
                            state_dict[key] = weight
                        else:
                            dist.stream.send(weight, dst=recv_rank)

            if global_rank != send_table[key] and global_rank in recv_ranklist:
                dist.stream.recv(state_dict[key], src=send_table[key])

        if is_src:
            f.__exit__(None, None, None)

    return state_dict


def get_sharded_file_name(args, file_name, is_optimizer=False):
    if not is_optimizer:
        shard_file = file_name.replace(
            ".pdparams",
            f"-{args.logical_process_index + 1:05d}-of-{args.world_size//args.dataset_world_size:05d}.pdparams",
        )
        shard_file = shard_file.replace(
            ".safetensors",
            f"-{args.logical_process_index + 1:05d}-of-{args.world_size//args.dataset_world_size:05d}.safetensors",
        )
    else:
        hcg = fleet.get_hybrid_communicate_group()
        dp_group = hcg.get_data_parallel_group()
        shard_file = file_name.replace(
            ".pdparams", f"-{args.logical_process_index + 1:05d}-of-{args.world_size//dp_group.nranks:05d}.pdparams"
        )
        shard_file = shard_file.replace(
            ".safetensors",
            f"-{args.logical_process_index + 1:05d}-of-{args.world_size//dp_group.nranks:05d}.safetensors",
        )
        shard_file = shard_file.replace(
            ".pdopt", f"-{args.logical_process_index + 1:05d}-of-{args.world_size//dp_group.nranks:05d}.pdopt"
        )
    return shard_file


def get_sharded_index(
    index_file_list,
    total_size_list,
):
    # save index json file
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
    if local_rank == 0:
        sharded_index_json = {}

        sharded_index_json["metadata"] = {"total_size": sum(total_size_list)}

        weight_map = {}
        for i, index_file in enumerate(index_file_list):
            weight_map.update(index_file_list[i])

        sharded_index_json["weight_map"] = weight_map
        return sharded_index_json

    return None


def reduce_master_weights_status(has_master_weights=False):
    data = paddle.to_tensor([has_master_weights], dtype="int32")

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()

    if tp_group.nranks > 1:
        dist.all_reduce(data, op=dist.ReduceOp.SUM, group=tp_group)
    if pp_group.nranks > 1:
        dist.all_reduce(data, op=dist.ReduceOp.SUM, group=pp_group)
    if sharding_group.nranks > 1:
        dist.all_reduce(data, op=dist.ReduceOp.SUM, group=sharding_group)

    return data.item() > 0


def gather_sharded_object(index_file, total_size, is_optimizer=False):

    index_file_list, total_size_list = [], []

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()

    logger.info(
        f"Unified checkpoint: generating sharded_index json files for {'optimizer or master weight' if is_optimizer else 'model weight'}."
    )

    if tp_group.nranks > 1:
        dist.all_gather_object(index_file_list, index_file, tp_group)
        dist.all_gather_object(total_size_list, total_size, tp_group)
    if pp_group.nranks > 1:
        pp_index_file_list = []
        pp_total_size_list = []
        dist.all_gather_object(
            pp_index_file_list, index_file_list if len(index_file_list) > 0 else index_file, pp_group
        )
        dist.all_gather_object(
            pp_total_size_list, total_size_list if len(total_size_list) > 0 else total_size, pp_group
        )
        index_file_list = pp_index_file_list
        total_size_list = pp_total_size_list

    index_file_list = flatten_list(index_file_list)
    total_size_list = flatten_list(total_size_list)

    # for pure sharding
    if len(index_file_list) == 0 and len(total_size_list) == 0:
        index_file_list = [index_file]
        total_size_list = [total_size]
    if is_optimizer:
        sharding_group = hcg.get_sharding_parallel_group()
        if sharding_group.nranks > 1:
            sharding_index_file_list = []
            sharding_total_size_list = []
            dist.all_gather_object(sharding_index_file_list, index_file_list, sharding_group)
            dist.all_gather_object(sharding_total_size_list, total_size_list, sharding_group)
            index_file_list = flatten_list(sharding_index_file_list)
            total_size_list = flatten_list(sharding_total_size_list)

    return index_file_list, total_size_list


def generate_base_static_name(vname):
    # return base static name and specific type name, like [embedding_0.w_0, moment1_0]
    if FP32_MASTER in vname:
        vname = vname.split("_" + FP32_MASTER + "_")
        return vname[0], vname[1]
    else:
        vname = vname.split(".")
        a = vname[0] + "." + vname[1][:3]
        b = vname[1][4:]
        return a, b


def filter_params(model_to_save, state_dict, is_optimizer=False):
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()

    tp_size = tp_group.nranks
    tp_rank = tp_group.rank

    # for pure sharding or pure pp
    if tp_size <= 1:
        return [list(state_dict.keys())]

    filter_tensor_list = [[] for i in range(tp_size)]

    if tp_rank == 0:
        tensor_bytes_dict = {}
        model_state_dict = get_expected_state_dict(model_to_save)
        for (k, v) in state_dict.items():
            model_v = model_state_dict[k.split("/")[0]] if is_optimizer else v
            if hasattr(model_v, "is_distributed") and model_v.is_distributed:
                tensor_bytes_dict[k] = v.numel().item() * tp_size * dtype_byte_size(v.dtype)
            else:
                tensor_bytes_dict[k] = v.numel().item() * dtype_byte_size(v.dtype)

        filter_tensor_list = []
        current_block = []
        current_block_size = 0
        total_size = 0

        max_shard_size = (sum(tensor_bytes_dict.values()) + tp_size - 1) // tp_size

        for index, (key, weight_size) in enumerate(tensor_bytes_dict.items()):
            # If this weight is going to tip up over the maximal size, we split.
            # if current_block_size + weight_size > max_shard_size:
            if total_size + weight_size > max_shard_size * (len(filter_tensor_list) + 1) or (
                len(tensor_bytes_dict) - index < (tp_size - len(filter_tensor_list))
            ):
                # fix if the first param is large than max_shard_size
                if len(current_block) > 0:
                    filter_tensor_list.append(current_block)
                current_block = []
                current_block_size = 0

            current_block.append(key)
            current_block_size += weight_size
            total_size += weight_size

        filter_tensor_list.append(current_block)
        if len(filter_tensor_list) < tp_size:
            filter_tensor_list.extend([[] for i in range(tp_size - len(filter_tensor_list))])

    dist.broadcast_object_list(
        filter_tensor_list,
        src=hcg.get_model_parallel_group_src_rank(),
        group=tp_group,
    )

    return filter_tensor_list


def merge_large_tensor_parallel(tensor, tp_group, tp_action, dst_rank, is_dst):
    num_rows = tensor.shape[0]
    num_splits = 4
    parts = np.array_split(np.arange(num_rows), num_splits)
    splits = [len(part) for part in parts]
    split_parts = np.insert(np.cumsum(splits), 0, 0)
    split_tensors = []
    for i in range(num_splits):
        if get_env_device() == "xpu":
            ret = distributed_allgather(tensor[split_parts[i] : split_parts[i + 1], :], group=tp_group, offload=False)
        else:
            ret = distributed_gather(
                tensor[split_parts[i] : split_parts[i + 1], :], dst=dst_rank, group=tp_group, offload=False
            )
        # Copy to CPUPlace temporarily, may lower speed.
        if ret is not None:
            ret = [t.cpu() for t in ret]
        split_tensors.append(ret)
    concat_tensors = []
    if is_dst:
        for i in range(tp_group.nranks):
            tmp = []
            for j in range(num_splits):
                tmp.append(split_tensors[j][i])
            concat_tensors.append(paddle.concat(tmp))
        tensor = tp_action(concat_tensors)
    else:
        tensor = None
    return tensor


def merge_tensor_parallel_with_shard(state_dict, tp_actions, all_filter_keys):
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    tp_rank = tp_group.rank

    # filter actions for pipeline mode
    if hcg.get_pipe_parallel_group().nranks > 1:
        filter_keys = set([y for x in all_filter_keys for y in x])
        for key in list(tp_actions.keys()):
            if key not in filter_keys:
                tp_actions.pop(key)

    state_dict_to_save = {}
    max_key_len = max([len(_) for _ in all_filter_keys])
    for i in range(max_key_len):
        for j, filter_keys in enumerate(all_filter_keys):
            is_dst = tp_rank == j
            if i > len(filter_keys) - 1:
                continue
            key = filter_keys[i]
            tensor = state_dict[key]
            if key in tp_actions:
                # Get tensor size
                tensor_bytes = tensor.numel().item() * dtype_byte_size(tensor.dtype) * tp_group.nranks
                if tensor_bytes >= 5 * 1024 * 1024 * 1024:  # temporarily set 5GB as threshold
                    tensor = merge_large_tensor_parallel(tensor, tp_group, tp_actions[key], j, is_dst)
                else:
                    if get_env_device() == "xpu":
                        ret = distributed_allgather(tensor, group=tp_group, offload=False)
                    else:
                        ret = distributed_gather(tensor, dst=j, group=tp_group, offload=False)
                    action = tp_actions.pop(key)
                    tensor = action(ret) if is_dst else None
            else:
                if is_dst:
                    tensor = tensor._copy_to(DEST_PLACE, False) if tensor.place.is_cpu_place() else tensor
                else:
                    tensor = None

            if is_dst:
                state_dict_to_save[key] = tensor

    if len(tp_actions) > 0:
        for x in tp_actions.keys():
            logger.warning(f"key <{x}> need to merge tensor parallel but we can't find in model state.")

    return state_dict_to_save


def merge_tensor_parallel_for_optimizer(state_dict, tp_actions, all_filter_keys):
    # Core function for UC
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    tp_rank = tp_group.rank

    state_dict_to_save = {}
    max_key_len = max([len(_) for _ in all_filter_keys])
    for i in range(max_key_len):
        for j, filter_keys in enumerate(all_filter_keys):
            is_dst = tp_rank == j
            if i > len(filter_keys) - 1:
                continue
            # get base model key
            model_key = filter_keys[i].split("/")[0]
            tensor = state_dict[filter_keys[i]]
            if model_key in tp_actions:
                # for example: beta1, beta2
                if tensor.numel().item() == 1:
                    if is_dst:
                        tensor = tensor._copy_to(DEST_PLACE, False) if not tensor.place.is_cpu_place() else tensor
                    else:
                        tensor = None
                else:
                    # Get tensor size
                    tensor_bytes = tensor.numel().item() * dtype_byte_size(tensor.dtype) * tp_group.nranks
                    if tensor_bytes >= 5 * 1024 * 1024 * 1024:  # temporarily set 5GB as threshold
                        tensor = merge_large_tensor_parallel(tensor, tp_group, tp_actions[model_key], j, is_dst)
                    else:
                        if get_env_device() == "xpu":
                            ret = distributed_allgather(tensor, group=tp_group, offload=False)
                        else:
                            ret = distributed_gather(tensor, dst=j, group=tp_group, offload=False)
                        action = tp_actions[model_key]
                        tensor = action(ret) if is_dst else None
            else:
                if is_dst:
                    tensor = tensor._copy_to(DEST_PLACE, False) if not tensor.place.is_cpu_place() else tensor
                else:
                    tensor = None

            if is_dst:
                state_dict_to_save[filter_keys[i]] = tensor

    return state_dict_to_save


def get_optimizer_shard_files(optimizer_path, index_filename):
    """
    For a given model:
    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.
    For the description of each arg, see [`PretrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """

    import json

    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a optimizer index ({index_filename}) in {optimizer_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_optimizer_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()
    sharded_metadata["master_weights"] = index.get("master_weights", False)

    file_map = {file: set() for file in shard_filenames}
    for weight, file in index["weight_map"].items():
        file_map[file].add(weight)

    sharded_metadata["file_map"] = file_map

    # First, let's deal with local folder.
    # TODO: if optimizer_path is a folder, we should check if the optimizer is already cached or not.
    if os.path.isdir(optimizer_path):
        shard_filenames = [os.path.join(optimizer_path, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata


def get_expected_keys(sharded_metadata, model, optimizer):
    hcg = fleet.get_hybrid_communicate_group()
    sharding_group = hcg.get_sharding_parallel_group()
    sharding_rank = sharding_group.rank
    in_sharding_parallel_model = sharding_group.nranks > 1
    if in_sharding_parallel_model:
        params2rank = optimizer._param2rank

    struct2static_name_mappings = {k: v.name for k, v in get_expected_state_dict(model).items()}

    expected_keys = []
    for key in list(sharded_metadata["all_optimizer_keys"]):
        key_name = key.split("/")[0]
        static_name = struct2static_name_mappings.get(key_name, None)

        if in_sharding_parallel_model:
            params_rank = params2rank.get(static_name, None)
            if params_rank == sharding_rank:
                expected_keys.append(key)
        else:
            if static_name is not None:
                expected_keys.append(key)
    expected_keys = set(expected_keys)

    loaded_keys = sharded_metadata["all_optimizer_keys"]
    missing_keys = expected_keys - set(loaded_keys)
    if len(missing_keys) > 0:
        raise ValueError(f"optimizer missing weights keys: {missing_keys}")

    return expected_keys


def mapping_optimizer_tp_actions(tp_actions, optimizer_loaded_keys):
    """# convert param.name to
    param.key/moment1_0
    or param.key/beta1_XXX
    or param.key/beta2_XXX
    Args:
        tp_actions (dict): dictionay of tensor parallel actions {key: action}
        optimizer_loaded_keys (list or set): [param.key1/moment1_0, param.key2/beta1_XXX, param.key3/beta2_XXX]
    Returns:
        dict: new dictionay of tensor parallel actions {key: action}
    """
    new_actions = {}
    for key in optimizer_loaded_keys:
        key_base, typename = key.split("/")
        if typename in optimizer_non_scaler_name and key_base in tp_actions:
            new_actions[key] = tp_actions[key_base]
    return new_actions


def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def select_model_weight_index(args, model, resume_from_checkpoint, safe_serialization, local=True):
    """
    try select model weight index from model weight or master weight index.
    """

    # find model weight index file
    if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
        index_filename = SAFE_PEFT_WEIGHTS_INDEX_NAME if safe_serialization else PADDLE_PEFT_WEIGHTS_INDEX_NAME
    else:
        index_filename = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else PADDLE_WEIGHTS_INDEX_NAME

    index_filename_path = os.path.join(resume_from_checkpoint, index_filename)
    identify_func = os.path.isfile if local else distributed_isfile

    if identify_func(index_filename_path):
        return index_filename
    else:
        index_filename = PADDLE_MASTER_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_MASTER_WEIGHTS_INDEX_NAME
        index_filename_path = os.path.join(resume_from_checkpoint, index_filename)

        if identify_func(index_filename_path):
            return index_filename
        else:
            raise ValueError("Can't find a valid unified model or master weight checkpoint to load.")


def update_master_weight_status(args, optimizer, has_master_weight, safe_serialization):
    if is_need_master_weight(optimizer, is_fp16_or_bp16=(args.fp16 or args.bf16)):
        if not has_master_weight:
            if UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value in args.unified_checkpoint_config:
                index_filename_master_weights = (
                    PADDLE_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_WEIGHTS_INDEX_NAME
                )
                has_master_weight = True
                logger.warning(
                    "The unified checkpoint does not contain master weight, "
                    "the model weight will be loaded as master weight."
                )
            else:
                raise ValueError(
                    "Can't find a valid unified master weight checkpoint,"
                    f"add '{UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value}' into 'unified_checkpoint_config' to "
                    "load model checkpoint as master weight"
                )
        else:
            has_master_weight = True
            index_filename_master_weights = (
                PADDLE_MASTER_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_MASTER_WEIGHTS_INDEX_NAME
            )
            if UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value in args.unified_checkpoint_config:
                index_filename_master_weights = (
                    PADDLE_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_WEIGHTS_INDEX_NAME
                )
    else:
        has_master_weight = False
        index_filename_master_weights = None

    return has_master_weight, index_filename_master_weights


def unwrap_optimizer(optimizer):
    while hasattr(optimizer, "_inner_opt") or hasattr(optimizer, "_optim"):
        if hasattr(optimizer, "_inner_opt"):
            optimizer = optimizer._inner_opt
        if hasattr(optimizer, "_optim"):
            optimizer = optimizer._optim

    return optimizer


def is_need_master_weight(optimizer, is_fp16_or_bp16):
    optimizer = unwrap_optimizer(optimizer)
    if hasattr(optimizer, "_multi_precision"):
        return optimizer._multi_precision and is_fp16_or_bp16
    else:
        return False
