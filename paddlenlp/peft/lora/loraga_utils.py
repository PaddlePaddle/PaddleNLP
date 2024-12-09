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

from typing import Optional

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler

from paddlenlp.peft.lora.lora_layers import (
    ColumnParallelLoRALinear,
    ColumnSequenceParallelLoRALinear,
    LoRALinear,
    RowParallelLoRALinear,
    RowSequenceParallelLoRALinear,
)
from paddlenlp.trainer.trainer_utils import IterableDatasetShard
from paddlenlp.transformers.model_utils import unwrap_model
from paddlenlp.utils.log import logger


def wrap_loraga_model(model, training_args):
    """Wrap Model with distributed strategies, support tp, dp, sharding"""

    from paddlenlp.trainer.trainer_utils import ShardingOption
    from paddlenlp.transformers.model_utils import PretrainedModel

    sharding = None
    if len(training_args.sharding) > 0:
        if training_args.local_rank == -1:
            raise ValueError("Using sharding only works in distributed training.")
        sharding = True

    in_pipeline_parallel_mode = training_args.pipeline_parallel_degree > 1
    in_sharding_parallel_mode = sharding is not None
    in_tensor_parallel_mode = training_args.tensor_parallel_degree > 1
    in_sep_parallel_mode = training_args.sep_parallel_degree > 1
    in_cp_parallel_mode = training_args.context_parallel_degree > 1

    # Multi-gpu training
    if training_args.world_size > 1 and (not training_args.use_hybrid_parallel):
        # MOE use DDP to broadcaset parameters.
        ddp_kwargs = {}
        if training_args.ddp_find_unused_parameters is not None:
            ddp_kwargs["find_unused_parameters"] = training_args.ddp_find_unused_parameters
        elif isinstance(model, PretrainedModel):
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
            ddp_kwargs["find_unused_parameters"] = not any(
                hasattr(m, "enable_recompute") and m.enable_recompute for m in model.sublayers(include_self=True)
            )
        else:
            ddp_kwargs["find_unused_parameters"] = True
        model = paddle.DataParallel(model, **ddp_kwargs)

    # No pipeline mode, sharding only
    if not in_pipeline_parallel_mode and in_sharding_parallel_mode:
        # Sharded DDP!
        if training_args.tensor_parallel_degree > 1:
            hcg = fleet.get_hybrid_communicate_group()
            assert (
                ShardingOption.SHARD_GRAD_OP in training_args.sharding
                or ShardingOption.SHARD_OP in training_args.sharding
            ), "Only support tensor parallel + sharding stage1/stage2 hybrid parallel now."
            model = paddle.distributed.fleet.meta_parallel.TensorParallel(model, hcg, strategy=None)
        if ShardingOption.SHARD_OP in training_args.sharding:
            model = fleet.distributed_model(model)

    if (
        not in_pipeline_parallel_mode
        and not in_sharding_parallel_mode
        and (in_tensor_parallel_mode or in_sep_parallel_mode or in_cp_parallel_mode)
    ):
        model = fleet.distributed_model(model)

    return model


def get_loraga_dataloader(train_dataset, data_collator, training_args):
    from paddlenlp.data import DistDataLoader

    def is_iterable_dataset(dataset):
        return isinstance(dataset, paddle.io.IterableDataset)

    def is_iterable_dataset_distributed(dataset):
        # For distributed dataloaer.
        is_iterable_dataset_tensor = paddle.to_tensor(is_iterable_dataset(dataset)).astype("int32").reshape([1])
        if dist.get_world_size() > 1:
            dist.all_reduce(is_iterable_dataset_tensor, op=dist.ReduceOp.MAX)
        if is_iterable_dataset_tensor.item() == 1:
            return True
        return False

    if training_args.distributed_dataloader:
        iterable_dataset = is_iterable_dataset_distributed(train_dataset)
    else:
        iterable_dataset = is_iterable_dataset(train_dataset)

    # if is_datasets_available() and train_dataset is not None and isinstance(train_dataset, datasets.Dataset):
    #     train_dataset = self._remove_unused_columns(train_dataset, description="training")
    _DataLoader = DistDataLoader if training_args.distributed_dataloader else DataLoader

    if iterable_dataset:  # For iterable dataset
        if training_args.dataset_world_size > 1 and train_dataset is not None:
            train_dataset = IterableDatasetShard(
                train_dataset,
                batch_size=training_args.per_device_train_batch_size,
                drop_last=training_args.dataloader_drop_last,
                num_processes=training_args.dataset_world_size,
                process_index=training_args.dataset_rank,
            )

        if training_args.distributed_dataloader:
            logger.info("Training using DistDataLoader.")
            additional_configs = {"is_iterable_dataset": True}
        else:
            additional_configs = {}
        return _DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=training_args.dataloader_num_workers,
            **additional_configs,
        )
    else:
        train_sampler = get_loraga_train_sampler(train_dataset, training_args)
        if training_args.distributed_dataloader:
            logger.info("Training using DistDataLoader.")
        return _DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=training_args.dataloader_num_workers,
        )


def get_loraga_train_sampler(train_dataset, training_args) -> Optional[paddle.io.Sampler]:
    if training_args.world_size <= 1:
        return paddle.io.BatchSampler(
            dataset=train_dataset,
            shuffle=True,
            batch_size=training_args.per_device_train_batch_size,
            drop_last=training_args.dataloader_drop_last,
        )

    return DistributedBatchSampler(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        num_replicas=training_args.dataset_world_size,
        rank=training_args.dataset_rank,
        drop_last=training_args.dataloader_drop_last,
    )


def estimate_gradient(model, train_ds, data_collator, training_args, loraga_init_iters=32, gradient_offload=False):
    """Estimate the gradient of the model on the given dataset"""
    gradient_dict = {}
    logger.info("Estimating gradient for LoraGA.")

    model = wrap_loraga_model(model, training_args)
    model.train()

    logger.info(f"Initialization iterations for LoraGA: {loraga_init_iters}")
    dataloader = get_loraga_dataloader(train_ds, data_collator, training_args)
    iters = 0

    with GradientOffloadHookContext(
        model=model,
        gradient_dict=gradient_dict,
        local_rank=training_args.local_rank,
        loraga_init_iters=loraga_init_iters,
        gradient_offload=gradient_offload,
    ):
        for batch in dataloader:
            iters += 1
            batch = {k: paddle.to_tensor(v) for k, v in batch.items()}

            # Pipeline parallel not supported currently
            loss, logits = model(**batch)
            loss.backward()

            if iters == loraga_init_iters:
                break

    return gradient_dict


def get_module_gradient(
    grad_name,
    base_model_prefix,
    gradient_dict,
    base_model_split_mappings,
    tp_degree,
    sharding_degree,
    dp_degree,
    local_rank,
):
    rank_suffix = "_" + str(local_rank)
    local_grad_name = ".".join(grad_name.split(".")[1:]) + ".weight" + rank_suffix
    gradient = gradient_dict.pop(local_grad_name).cuda()
    if tp_degree > 1:
        # remove prefix and suffix
        model_split_key = local_grad_name.split(base_model_prefix)[-1].rsplit(rank_suffix, 1)[0]
        if model_split_key in base_model_split_mappings:
            merge_func = base_model_split_mappings[model_split_key]
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            output_tensors = []
            dist.all_gather(output_tensors, gradient, group=model_parallel_group)

            output_tensors = [t if len(t.shape) > 0 else t.reshape_([-1]) for t in output_tensors]
            gradient = merge_func(output_tensors).cuda()

    # sharding
    if sharding_degree > 1:
        hcg = fleet.get_hybrid_communicate_group()
        sharding_parallel_group = hcg.get_sharding_parallel_group()
        if sharding_parallel_group.nranks > 1:

            dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=sharding_parallel_group)
            gradient /= sharding_parallel_group.nranks
    # dp
    if dp_degree > 1:
        hcg = fleet.get_hybrid_communicate_group()
        data_parallel_group = hcg.get_data_parallel_group()
        if data_parallel_group.nranks > 1:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=data_parallel_group)
            gradient /= data_parallel_group.nranks
    return gradient


def loraga_svd_reinit(model, gradient_dict, base_model_split_mappings, stable_gamma, training_args, **kwargs) -> None:
    """
    If Loraga has already been initialized, directly modify the base model weights.
    Otherwise, reinitialize and save the initialized model.

    Args:
        model (Any): The model to reinitialize.
        gradient_dict (Dict[str, Any]): Dictionary containing gradients.
        model_split_mappings (Any): Mappings for model tensor parallelism.
        stable_gamma (Any): Stable gamma parameter for Loraga.
        training_args (Any): Training arguments.
        **kwargs: Additional keyword arguments.
    """

    lora_split_mapping = None
    tensor_parallel_degree = training_args.tensor_parallel_degree
    in_tensor_parallel_mode = tensor_parallel_degree > 1

    base_model_prefix = unwrap_model(model).base_model_prefix + "."
    if in_tensor_parallel_mode:
        lora_split_mapping = model._get_tensor_parallel_mappings(model.config)
    loraga_init_dict = {}
    for name, module in model.named_sublayers():
        if isinstance(
            module,
            (
                LoRALinear,
                RowSequenceParallelLoRALinear,
                ColumnSequenceParallelLoRALinear,
                RowParallelLoRALinear,
                ColumnParallelLoRALinear,
            ),
        ):
            # gather gradient if in tensor parallel mode, average gradient if in data parallel mode
            module_gradient = get_module_gradient(
                name,
                base_model_prefix,
                gradient_dict,
                base_model_split_mappings,
                training_args.tensor_parallel_degree,
                training_args.sharding_parallel_degree,
                training_args.data_parallel_degree,
                training_args.local_rank,
            )
            # perform SVD to reinit base model weight and lora adapter weight
            loraga_svd_module(
                name,
                module,
                module_gradient,
                stable_gamma,
                loraga_init_dict,
                in_tensor_parallel_mode,
                lora_split_mapping,
                **kwargs,
            )
    model.loraga_init_dict = loraga_init_dict


def loraga_svd_module(
    name,
    module,
    grads,
    stable_gamma,
    loraga_init_dict,
    in_tensor_parallel_mode=False,
    lora_split_mapping=None,
    **kwargs
):
    with paddle.no_grad():
        lora_r = module.r

        loraA_name = ".".join(name.split(".")[1:]) + ".lora_A"
        loraB_name = ".".join(name.split(".")[1:]) + ".lora_B"

        U, S, V = paddle.linalg.svd_lowrank(grads.astype("float32"), q=4 * lora_r, niter=4)

        V = V.T
        # get new low rank adapter after SVD
        A = U[:, lora_r : 2 * lora_r]
        B = V[:lora_r, :]
        m, n = grads.shape  # m: feature_out, n: feature_in
        # If stable_gamma is not -1, scale the matrices A and B by the square root of the stable_gamma
        if stable_gamma != -1:
            A = A * m**0.25 / stable_gamma**0.5
            B = B * m**0.25 / stable_gamma**0.5
        else:
            A = A / module.scaling
            B = B / module.scaling

        if in_tensor_parallel_mode:
            # split lora adapter weight if in tensor parallel mode
            if module.lora_A.is_distributed and lora_split_mapping is not None:
                split_function = lora_split_mapping[loraA_name]
                A = paddle.to_tensor(split_function(A))
            if module.lora_B.is_distributed and lora_split_mapping is not None:
                split_function = lora_split_mapping[loraB_name]
                B = paddle.to_tensor(split_function(B))
        A = A.astype(module.lora_A.dtype)
        B = B.astype(module.lora_B.dtype)
        loraga_init_dict[loraA_name] = A
        loraga_init_dict[loraB_name] = B
        # reinit lora adapter weight
        module.lora_A.set_value(A)
        module.lora_B.set_value(B)

        offset = module.lora_A @ module.lora_B
        # reinit base model weight
        module.weight.data -= module.scaling * offset


def set_hook_enable(value=False):
    global ENABLE_HOOK
    ENABLE_HOOK = value


def get_hook_enable():
    global ENABLE_HOOK
    return ENABLE_HOOK


class GradientOffloadHookContext:
    def __init__(
        self,
        model,
        gradient_dict: dict,
        local_rank: int = 0,
        loraga_init_iters: int = 4,
        gradient_offload: bool = False,
        *args,
        **kwargs,
    ):
        """Offload gradient to cpu"""
        self.model = model
        self.gradient_dict = gradient_dict
        self.local_rank = local_rank
        self.loraga_init_iters = loraga_init_iters
        self.gradient_offload = gradient_offload

    def __enter__(self):
        set_hook_enable(True)
        self.register_gradient_hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_hook_enable(False)

    def register_gradient_hook(self):
        for grad_name, param in self.model.named_parameters():
            param._register_backward_hook(
                self.get_record_gradient_hook(self.model, self.gradient_dict, grad_name, param)
            )

    def get_record_gradient_hook(self, model, gradient_dict, grad_name, param):
        def record_gradient_hook(*_):
            if get_hook_enable():
                grad = param.grad
                local_grad_name = grad_name.split("_layers.")[-1] + "_" + str(self.local_rank)
                if not param.stop_gradient and grad is not None:
                    if local_grad_name not in gradient_dict:
                        if self.gradient_offload:
                            gradient_dict[local_grad_name] = (grad / self.loraga_init_iters).cpu()
                        else:
                            gradient_dict[local_grad_name] = grad.clone() / self.loraga_init_iters
                    else:
                        if self.gradient_offload:
                            new_grad = gradient_dict[local_grad_name].cuda() + grad / self.loraga_init_iters
                            gradient_dict[local_grad_name] = new_grad.cpu()
                        else:
                            gradient_dict[local_grad_name] += grad / self.loraga_init_iters
                param.clear_gradient(False)  # release gradient memory

        return record_gradient_hook
