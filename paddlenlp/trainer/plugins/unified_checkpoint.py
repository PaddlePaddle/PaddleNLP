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

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from tqdm.auto import tqdm

try:
    from paddle.base import core
except:
    core = None

from paddlenlp.peft import LoRAModel, PrefixModelForCausalLM
from paddlenlp.trainer.argparser import strtobool
from paddlenlp.trainer.trainer_utils import ShardingOption
from paddlenlp.trainer.utils.helper import distributed_file, distributed_isfile
from paddlenlp.transformers.model_utils import (
    PretrainedModel,
    _add_variant,
    _load_state_dict_into_model,
    faster_set_state_dict,
    load_state_dict,
    unwrap_model,
)
from paddlenlp.transformers.utils import (
    device_guard,
    dtype_byte_size,
    get_checkpoint_shard_files,
    is_safetensors_available,
)
from paddlenlp.utils.env import (
    LORA_WEIGHTS_NAME,
    PADDLE_MASTER_WEIGHTS_INDEX_NAME,
    PADDLE_MASTER_WEIGHTS_NAME,
    PADDLE_OPTIMIZER_INDEX_NAME,
    PADDLE_OPTIMIZER_NAME,
    PADDLE_WEIGHTS_NAME,
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
from paddlenlp.utils.nested import flatten_list, nested_copy

if is_safetensors_available():
    from safetensors.numpy import save_file as safe_save_file

    if sys.platform.startswith("win"):
        from safetensors.numpy import load_file
    else:
        from paddlenlp.utils.safetensors import fast_load_file as load_file

from .shared_memory_utils import (
    _read_state_dict_from_shm,
    _traverse_copy_to_shm,
    create_meta_dict,
)
from .unified_checkpoint_dynamic import (
    load_unified_checkpoint_dynamically,
    load_unified_optimizer_dynamically,
)
from .unified_checkpoint_sharding_v2 import (
    gather_splited_param_for_optimizer,
    load_unified_optimizer_split_param,
)
from .unified_checkpoint_single_card import (
    load_single_card_checkpoint,
    load_single_card_optimizer,
    save_single_card_checkpoint,
    save_single_card_optimizer,
)
from .unified_checkpoint_utils import (
    FP32_MASTER,
    UnifiedCheckpointOption,
    filter_params,
    gather_sharded_object,
    generate_base_static_name,
    get_expected_keys,
    get_expected_state_dict,
    get_optimizer_shard_files,
    get_sharded_file_name,
    get_sharded_index,
    is_need_master_weight,
    mapping_optimizer_tp_actions,
    merge_tensor_parallel_for_optimizer,
    merge_tensor_parallel_with_shard,
    reduce_master_weights_status,
    rename_shard_file,
    save_config,
    save_prefix_past_key_value,
    select_model_weight_index,
    update_master_weight_status,
)


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
            save_single_card_checkpoint(model_to_save, output_dir)
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
            load_single_card_checkpoint(model, resume_from_checkpoint)
            return

        local_resume = check_unified_checkpoint(self.args, model, resume_from_checkpoint, safe_serialization=True)

        if not local_resume:
            logger.info("Begin to dynamically load unified checkpoint!")
            load_unified_checkpoint_dynamically(self.args, model, resume_from_checkpoint, safe_serialization=True)
            return

        if self.args.dataset_rank == 0 or self.args.use_expert_parallel:
            load_unified_checkpoint_locally(self.args, model, resume_from_checkpoint, safe_serialization=True)

    def save_non_merge_optimizer(self, model, optim_state_dict, master_weights, output_dir):
        paddle.device.cuda.empty_cache()

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

        no_sync_kname = []
        model_state_dict = get_expected_state_dict(model)
        for k, v in model_state_dict.items():
            if getattr(v, "no_sync", False):
                no_sync_kname.append(k)

        hcg = fleet.get_hybrid_communicate_group()
        dp_group = hcg.get_data_parallel_group()
        dp_rank = dp_group.rank if dp_group.nranks > 1 else 0
        if self.args.use_expert_parallel:
            for k in list(optim_state_dict.keys()):
                model_k = k.split("/")[0]
                if dp_rank > 0 and model_k not in no_sync_kname:
                    optim_state_dict.pop(k)
            if master_weights is not None:
                for k in list(master_weights.keys()):
                    model_k = k.split("/")[0]
                    if dp_rank > 0 and model_k not in no_sync_kname:
                        master_weights.pop(k)

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
                if model_state_dict[key_name[0]].dtype != core.VarDesc.VarType.FP32:
                    key_name = "_".join([static_name, FP32_MASTER, key_name[1]])
                else:
                    key_name = "_".join([static_name, key_name[1]])
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
        if paddle.distributed.get_world_size() <= 1:
            save_single_card_optimizer(model, optimizer, output_dir)
            return

        if (
            self.args.sharding_parallel_degree > 1
            and ShardingOption.SHARD_OP in self.args.sharding
            and "split_param" in self.args.sharding_parallel_config
        ):
            optim_state_dict, master_weights = gather_splited_param_for_optimizer(optimizer)
        else:
            optim_state_dict = nested_copy(optimizer.state_dict())
            master_weights = None
            if "master_weights" in optim_state_dict.keys():
                master_weights = optim_state_dict["master_weights"]
                optim_state_dict.pop("master_weights")
            if "LR_Scheduler" in optim_state_dict.keys():
                optim_state_dict.pop("LR_Scheduler")

        if "ignore_merge_optimizer" in self.args.unified_checkpoint_config:
            self.save_non_merge_optimizer(model, optim_state_dict, master_weights, output_dir)
            return

        # Split into naive optimizer params and master weights.
        results = unified_optimizer_into_shards(
            self.args, model, optim_state_dict, master_weights, safe_serialization=True
        )
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

    def load_unified_optimizer(self, model, optimizer, resume_from_checkpoint):
        """Load potential model checkpoint

        Args:
            model (PretrainedModel): Your model to load
            resume_from_checkpoint (str): path of the checkpoint to load

        Returns:
            None
        """

        if paddle.distributed.get_world_size() <= 1:
            optim_state_dict = load_single_card_optimizer(model, optimizer, resume_from_checkpoint)
            return optim_state_dict

        has_merge_optimizer_safetensors = distributed_isfile(
            os.path.join(resume_from_checkpoint, SAFE_OPTIMIZER_INDEX_NAME)
        )
        # If not having merge optimizer, then load non-merge optimizer.
        if not has_merge_optimizer_safetensors:
            if self.args.data_parallel_rank == 0 or self.args.use_expert_parallel:
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

        if self.args.data_parallel_rank == 0 or self.args.use_expert_parallel:
            returned_optim_state_dict = load_unified_optimizer_locally(
                self.args, model, optimizer, resume_from_checkpoint, safe_serialization=True
            )
            return returned_optim_state_dict
        return None

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
    Only dataset_rank == 0 or using expert parallel can enter this function.
    """
    index_filename = select_model_weight_index(model, resume_from_checkpoint, safe_serialization, local=True)

    resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
        pretrained_model_name_or_path=resume_from_checkpoint,
        index_filename=os.path.join(resume_from_checkpoint, index_filename),
    )
    loaded_keys = sharded_metadata["all_checkpoint_keys"]

    model_state_dict = get_expected_state_dict(model)
    # If using expert parallel, when dp_rank > 0, need to modify the expected_keys here.
    if not args.use_expert_parallel or (args.use_expert_parallel and args.data_parallel_rank == 0):
        expected_keys = set(list(model_state_dict.keys()))
    else:
        expected_keys = set()
        for key in model_state_dict.keys():
            if getattr(model_state_dict[key], "no_sync", False):
                expected_keys.add(key)
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
    # renumerize shard_file name for expert_parallel.
    if args.use_expert_parallel:
        shard_file = rename_shard_file(args, shard_file, weights_name)

    for key, weight in state_dict.items():
        index_weight_file[key] = shard_file
        total_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    index_file_list, total_size_list = gather_sharded_object(
        index_weight_file, total_size, use_expert_parallel=args.use_expert_parallel
    )
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
    # Special process with split param.
    if (
        args.sharding_parallel_degree > 1
        and ShardingOption.SHARD_OP in args.sharding
        and "split_param" in args.sharding_parallel_config
    ):
        returned_optim_state_dict = load_unified_optimizer_split_param(model, optimizer, resume_from_checkpoint)
        return returned_optim_state_dict

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

    expected_keys = get_expected_keys(args, sharded_metadata, model, optimizer)

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

        expected_keys_mw = get_expected_keys(args, sharded_metadata_mw, model, optimizer, is_master_weights=True)
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
            if model_state_dict[key_name[0]].dtype != core.VarDesc.VarType.FP32:
                key_name = "_".join([static_name, FP32_MASTER, key_name[1]])
            else:
                key_name = "_".join([static_name, key_name[1]])
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
    optim_state_dict,
    master_weights,
    safe_serialization=False,
):
    """Get optimizer state dict and master weight state dict.

    Args:
        optimizer (Optimizer): optimizer to save.
        safe_serialization (bool, optional): safe serialization using safetensors. Defaults to False.
    """
    paddle.device.cuda.empty_cache()

    # gather global master_weights status.
    global_master_weights = reduce_master_weights_status(master_weights is not None)
    if master_weights is None and global_master_weights:
        master_weights = {}

    # get optimizer param mappings
    static2struct_name_mappings = {}
    state_dict = get_expected_state_dict(model)
    fp32_weight = {}
    for k, v in state_dict.items():
        static2struct_name_mappings[v.name] = k
        if master_weights is not None and v.dtype == core.VarDesc.VarType.FP32:
            if args.dataset_rank > 0:  # deal with different dataset rank.
                continue
            fp32_weight[k] = v

    # rename optimizer param
    for key in list(optim_state_dict.keys()):
        static_name, type_name = generate_base_static_name(key)
        new_name = static2struct_name_mappings[static_name] + "/" + type_name
        optim_state_dict[new_name] = optim_state_dict.pop(key)
    if master_weights is not None:
        for key in list(master_weights.keys()):
            master_weights[static2struct_name_mappings[key]] = master_weights.pop(key)
        master_weights.update(fp32_weight)

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
            state_dict if args.use_expert_parallel else None,
        )
        paddle.device.cuda.empty_cache()

        if master_weights is not None:
            logger.info("Unified master weight tensor parallel in shards")
            master_weights = merge_tensor_parallel_for_optimizer(
                master_weights,
                tp_actions,
                filter_master_keys,
                state_dict if args.use_expert_parallel else None,
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
        index_optimizer_file,
        total_optim_size,
        is_optimizer=True,
        use_expert_parallel=args.use_expert_parallel,
    )
    sharded_optim_index = get_sharded_index(index_optimizer_filelist, total_optim_size_list)
    if master_weights is not None:
        index_master_weight_filelist, total_master_weight_size_list = gather_sharded_object(
            index_master_weight_file,
            total_master_weight_size,
            is_optimizer=True,
            use_expert_parallel=args.use_expert_parallel,
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
    index_filename = select_model_weight_index(model, resume_from_checkpoint, safe_serialization, local=False)
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
    if args.dataset_rank == 0 or args.use_expert_parallel:
        hcg = fleet.get_hybrid_communicate_group()
        tp_group = hcg.get_model_parallel_group()
        pp_group = hcg.get_pipe_parallel_group()
        dp_group = hcg.get_data_parallel_group()
        dp_rank = dp_group.rank if dp_group.nranks > 1 else 0

        need_files = set()
        state_dict = get_expected_state_dict(model)
        for key in state_dict.keys():
            filename = index["weight_map"][key]
            # When using expert parallel, there's no need to check tensors with `no_sync=False` when dp_rank > 0.
            if args.use_expert_parallel and dp_rank > 0 and not getattr(state_dict[key], "no_sync", False):
                continue
            need_files.add(filename)
        diff_filelist = list(need_files.difference(set(existed_files)))
        num_diff = paddle.to_tensor([len(diff_filelist)])
        if tp_group.nranks > 1:
            dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=tp_group)
        if pp_group.nranks > 1:
            dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=pp_group)
        if args.use_expert_parallel and dp_group.nranks > 1:
            dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=dp_group)
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
    dp_group = hcg.get_data_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()
    sharding_rank = sharding_group.rank
    dp_rank = dp_group.rank if dp_group.nranks > 1 else 0
    struct2static_name_mappings = {k: v.name for k, v in model.state_dict().items()}

    if (
        args.sharding_parallel_degree > 1
        and ShardingOption.SHARD_OP in args.sharding
        and "split_param" in args.sharding_parallel_config
    ):
        # We do not check optimizer files completion for split_param, since it is very complicated. Directly support local resume.
        logger.warning("We only support local resume for split_param mode, do not support dynamically loading.")
        return True

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
        if args.data_parallel_rank == 0 or args.use_expert_parallel:
            need_files = set()
            state_dict = get_expected_state_dict(model)

            for key in state_dict.keys():
                if sharding_group.nranks > 1:
                    static_name = struct2static_name_mappings.get(key, None)
                    param_rank = param2rank.get(static_name, None)
                    if param_rank != sharding_rank:
                        continue

                # When using expert parallel, there's no need to check tensors with `no_sync=False` when dp_rank > 0.
                if args.use_expert_parallel and dp_rank > 0 and not getattr(state_dict[key], "no_sync", False):
                    continue

                if is_master_weights and state_dict[key].dtype == core.VarDesc.VarType.FP32:
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
            if args.use_expert_parallel and dp_group.nranks > 1:
                dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=dp_group)

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
