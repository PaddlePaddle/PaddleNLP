import inspect
import os
import shutil
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)
from .trainer import Trainer
try:
    from paddle.distributed.fleet.utils.hybrid_parallel_util import (
        obtain_optimizer_parameters_list,
    )

    _obtain_optimizer_parameters_list = obtain_optimizer_parameters_list
except:
    try:
        from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
            _obtain_optimizer_parameters_list,
        )
    except:
        _obtain_optimizer_parameters_list = None

from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from paddle.io import DistributedBatchSampler
from tqdm.auto import tqdm

from ..data import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
)
from ..data.causal_dataset import get_datasets_weights_and_num_samples, build_train_valid_test_datasets
from ..peft import LoRAModel, PrefixModelForCausalLM
from ..transformers.model_utils import (
    PretrainedModel,
    _add_variant,
)
from ..transformers.tokenizer_utils import PretrainedTokenizer
from ..utils.batch_sampler import DistributedBatchSampler as NlpDistributedBatchSampler
from ..utils.env import (
    PADDLE_WEIGHTS_NAME,
)
from ..utils.log import logger
from .integrations import get_reporting_integration_callbacks
from .plugins.timer import get_timers, set_timers
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_utils import (  # set_hyrbid_parallel_seed,
    EvalPrediction,
    ShardingOption,
    TrainerMemoryTracker,
    PredictionOutput,
    TrainOutput,
    set_seed,
    speed_metrics,
)
from .training_args import TrainingArguments
from .utils import reshard as reshard_util
from .utils.helper import (  # nested_truncate,
    distributed_file,
    distributed_isfile,
)
from .utils.sharding_io import ShardingIO
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor
DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"

OPTIMIZER_NAME = "optimizer.pdopt"
SCHEDULER_NAME = "scheduler.pdparams"
SCALER_NAME = "scaler.pdparams"

try:
    from paddle.distributed.fleet.utils import mix_precision_utils
except:
    mix_precision_utils = None

local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
def is_dp_group_support_in_group_sharded_parallel():
    return "dp_group" in set(inspect.signature(paddle.distributed.sharding.group_sharded_parallel).parameters.keys())


__all__ = [
    "Mutil_dataset_PretrainingTrainer",
]

class Mutil_dataset_PretrainingTrainer(Trainer):
    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        data: List = None,
        data_args = None,
        train_val_test_num_samples: List = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)

        self.args = args
        self.is_in_train = False
        self.data = data
        self.train_val_test_num_samples = train_val_test_num_samples
        self.data_args = data_args
        self.train_dataset = None
        self.eval_dataset = None
        # self.do_grad_scaling = args.fp16

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # Seed must be set before instantiating the model when using model
        set_seed(args=self.args)

        if model is None:
            raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")

        if self.args.should_save or self.args.should_save_model_state:
            os.makedirs(self.args.output_dir, exist_ok=True)

        self.sharding = None
        if len(args.sharding) > 0:
            if args.local_rank == -1:
                raise ValueError("Using sharding only works in distributed training.")
            self.sharding = True

        # init parallel env
        if paddle.distributed.get_world_size() > 1:
            if self.args.use_hybrid_parallel:
                self.hcg = fleet.get_hybrid_communicate_group()
                self.dp_group = self.hcg.get_data_parallel_group()
                self.sharding_group = self.hcg.get_sharding_parallel_group()

        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)

        self.data_collator = data_collator if data_collator is not None else default_collator
        # self.train_dataset = train_dataset
        # self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        if not args.skip_profile_timer:
            set_timers()
        self.timers = get_timers()

        self.model_wrapped = model
        self.model = model
        self.criterion = criterion

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        # Label smoothing
        # if self.args.label_smoothing_factor != 0:
        #     self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        # else:
        self.label_smoother = None
        self.state = TrainerState()
        self.control = TrainerControl()
        self._signature_columns = None
        self.optimizer_grouped_parameters = None
        self.sharding_io = None
        if self.args.should_save_sharding_stage1_model or self.args.should_load_sharding_stage1_model:
            self.sharding_io = ShardingIO(self.args, self.model, self.optimizer)

        if self.sharding is not None and self.optimizer is not None:
            raise RuntimeError(
                "Passing `optimizers` is not allowed if sharding is enabled."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        if self.args.pipeline_parallel_degree > 1:
            from paddle.distributed.fleet.meta_parallel import PipelineLayer

            assert (isinstance(model, LoRAModel) and isinstance(model.model, PipelineLayer)) or isinstance(
                model, PipelineLayer
            ), "Only support pipeline parallel mode when model is PipelineLayer!!!"

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # if train_dataset is not None and not isinstance(train_dataset, collections.abc.Sized) and args.max_steps <= 0:
        #     raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")

        self.do_grad_scaling = False
        self.enable_autocast_context_manager = False

        if args.fp16 or args.bf16:
            logger.info("Using half precision")
            self.enable_autocast_context_manager = True
            self.do_grad_scaling = True if args.fp16 else False
            self.amp_dtype = "float16" if args.fp16 else "bfloat16"
            # fix for load saved fp16 or bf16 ckpt, decorate model first.
            if self.args.fp16_opt_level == "O2":
                if self.amp_dtype == "bfloat16":
                    # fix for paddlepaddle < 2.4.1, not support for bf16
                    paddle.amp.decorate(models=model, level=self.args.fp16_opt_level, dtype=self.amp_dtype)
                else:
                    paddle.amp.decorate(models=model, level=self.args.fp16_opt_level)
            # for pipeline mode and pure tensor parallel
            if self.args.pipeline_parallel_degree > 1 or (
                self.args.tensor_parallel_degree > 1 and self.sharding is None
            ):
                self.scaler = paddle.amp.GradScaler(init_loss_scaling=self.args.scale_loss)
                if self.args.amp_master_grad:
                    mix_precision_utils.MixPrecisionScaler(self.scaler)  # retun value has no use
                self.scaler = fleet.distributed_scaler(self.scaler)
            elif self.sharding is not None:
                self.scaler = paddle.amp.GradScaler(init_loss_scaling=self.args.scale_loss)
                if self.amp_dtype == "float16" or self.amp_dtype == "bfloat16":
                    if ShardingOption.SHARD_OP in self.args.sharding:
                        self.scaler = fleet.distributed_scaler(self.scaler)
                        if self.args.amp_master_grad:
                            mix_precision_utils.MixPrecisionScaler(self.scaler)  # retun value has no use
                    else:
                        # scaler for stage2 and stage3
                        from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import (
                            GroupShardedScaler,
                        )

                        if self.args.amp_master_grad:
                            assert (
                                ShardingOption.SHARD_GRAD_OP in self.args.sharding
                            ), "Main grad doesn't support sharding stage 3 for now."
                            mix_precision_utils.MixPrecisionScaler(self.scaler)  # return value has no use

                        self.scaler = GroupShardedScaler(self.scaler)
                else:
                    self.do_grad_scaling = False
                    self.use_cuda_amp = False
                    self.amp_dtype = None

            else:
                self.scaler = paddle.amp.GradScaler(init_loss_scaling=self.args.scale_loss)

        if args.recompute:

            def fn(layer):
                if hasattr(layer, "enable_recompute") and (
                    layer.enable_recompute is False or layer.enable_recompute == 0
                ):
                    layer.enable_recompute = True

            model.apply(fn)

        default_label_names = (
            ["start_positions", "end_positions"]
            if "QusetionAnswering" in type(self.model).__name__ or "UIE" in type(self.model).__name__
            else ["labels"]
        )
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names

        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)
        self.print_config()

        self._memory_tracker.stop_and_update_metrics()


    def get_dataset(self, datasets_train_valid_test_num_samples, total_batch_size, datasets_total_samples_dict):
        datasets_need_samples = [dataset[0] for dataset in datasets_train_valid_test_num_samples]
        datasets_steps = [max(1, samples // total_batch_size) for samples in datasets_need_samples]
        sum_steps = sum(datasets_steps)
        map_dataset = np.arange(sum_steps, dtype=np.int64)
        if sum_steps < self.args.max_steps:
            map_dataset = np.concatenate([map_dataset, np.arange(self.args.max_steps-sum_steps)])
        if datasets_total_samples_dict is None:
            return map_dataset
        else:
            datasets_total_samples = list(datasets_total_samples_dict.values())
            datasets_total_step = [max(1, dataset_total_samples // total_batch_size) for dataset_total_samples in datasets_total_samples]
            datasets_need_step = []
            for i, dataset_total_step in enumerate(datasets_total_step):
                datasets_need_step.append(min(dataset_total_step, datasets_steps[i]))
            offset = 0
            for index, dataset_need_step in enumerate(datasets_need_step):
                map_dataset[offset:offset+dataset_need_step] = [index] * len(map_dataset[offset:offset+dataset_need_step])
                offset += dataset_need_step
            if len(datasets_train_valid_test_num_samples) == len(datasets_need_step):
                left_steps = [i-j for i, j in zip(datasets_steps, datasets_need_step)]
                while sum(left_steps) > 0:
                    left_steps_2 = []
                    for i, left_step in enumerate(left_steps):
                        left_steps_2.append(min(left_step, datasets_total_step[i]))
                    for index, left_step_2 in enumerate(left_steps_2):
                        map_dataset[offset:offset+left_step_2] = [index] * len(map_dataset[offset:offset+left_step_2])
                        offset += left_step_2
                    left_steps = [i-j for i, j in zip(left_steps, left_steps_2)]
            else:
                map_dataset[offset] = index + 1
        return map_dataset
    
    def get_offset_step(self, step, datasets_train_valid_test_num_samples, total_batch_size, datasets_total_samples_dict):
        datasets_total_samples = list(datasets_total_samples_dict.values())
        datasets_need_samples = [dataset[0] for dataset in datasets_train_valid_test_num_samples]
        datasets_steps = [max(1, samples // total_batch_size) for samples in datasets_need_samples]
        sum_steps = sum(datasets_steps)
        offset_step = np.arange(sum_steps, dtype=np.int64)
        if sum_steps < self.args.max_steps:
            offset_step = np.concatenate([offset_step, np.zeros(self.args.max_steps-sum_steps, dtype=np.int64)])
        if len(datasets_total_samples) == 1:
            return offset_step[step]
        else:
            datasets_total_step = [max(1, dataset_total_samples // total_batch_size) for dataset_total_samples in datasets_total_samples]
            datasets_need_step = []
            for i, dataset_total_step in enumerate(datasets_total_step):
                datasets_need_step.append(min(dataset_total_step, datasets_steps[i]))
            offset = 0
            for dataset_need_step in datasets_need_step:
                offset_step[offset:offset+dataset_need_step] = np.arange(dataset_need_step)
                offset += dataset_need_step
            if len(datasets_train_valid_test_num_samples) == len(datasets_need_step):
                left_steps = [i-j for i, j in zip(datasets_steps, datasets_need_step)]
                while sum(left_steps) > 0:
                    left_steps_2 = []
                    for i, left_step in enumerate(left_steps):
                        left_steps_2.append(min(left_step, datasets_total_step[i]))
                    for left_step_2 in left_steps_2:
                        offset_step[offset:offset+left_step_2] = np.arange(left_step_2)
                        offset += left_step_2
                    left_steps = [i-j for i, j in zip(left_steps, left_steps_2)]
            else:
                offset_step[offset] = 0
        return offset_step[step]
        

    def pre_process(self, first=False):
        print('aaa')
        print(self.state.ready_dataset)
        if first:
            print('first')
            _, _, _ = build_train_valid_test_datasets(
                [self.data[1]], 
                self.data_args.data_impl,
                self.data_args.split,
                self.train_val_test_num_samples,
                self.data_args.max_seq_length,
                self.args.seed,
                self.data_args.skip_warmup,
                share_folder=self.data_args.share_folder,
                data_cache_path=self.data_args.data_cache,
                load_data=False
            )
            self.state.ready_dataset += 1
        else:
            print('not first')
            for i in range(self.state.ready_dataset, len(self.data)//2):
                _, _, _ = build_train_valid_test_datasets(
                    [self.data[2*i+1]], 
                    self.data_args.data_impl,
                    self.data_args.split,
                    self.train_val_test_num_samples,
                    self.data_args.max_seq_length,
                    self.args.seed,
                    self.data_args.skip_warmup,
                    share_folder=self.data_args.share_folder,
                    data_cache_path=self.data_args.data_cache,
                    load_data=False
                )
                self.state.ready_dataset += 1
        return self.state.ready_dataset
        

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
        """
        args = self.args
        self.is_in_train = True
        start_time = time.time()
        logger.info(f"Starting training from resume_from_checkpoint : {resume_from_checkpoint}")
        if local_rank == 0:  
            executor = ProcessPoolExecutor()
            logger.info(f"Starting prepare dataset : {self.data[1]}")
            prepare_future_0 = executor.submit(self.pre_process, True)
            print('1')
        # The resume_from_checkpoint could be None in some machine node.
        # Here we reset None to temp directory.
        if args.world_size > 1:
            is_resume_from_checkpoint = paddle.to_tensor([resume_from_checkpoint is not None])
            paddle.distributed.all_reduce(is_resume_from_checkpoint)
            is_resume_from_checkpoint = is_resume_from_checkpoint.item()
            if is_resume_from_checkpoint > 0 and is_resume_from_checkpoint < paddle.distributed.get_world_size():
                if resume_from_checkpoint is None:
                    resume_from_checkpoint = os.path.join(self.args.output_dir, "local_tempdir")
                    if os.path.exists(resume_from_checkpoint) and self.args.local_rank == 0:
                        shutil.rmtree(resume_from_checkpoint)
                    os.makedirs(resume_from_checkpoint, exist_ok=True)
                    logger.info(f"Reset resume_from_checkpoint to temp directory : {resume_from_checkpoint}")

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if not self.args.should_load_sharding_stage1_model:
            self._load_from_checkpoint(resume_from_checkpoint)

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.dataset_world_size
        output = get_datasets_weights_and_num_samples(self.data, self.train_val_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output
        train_num_samples, valid_num_samples, test_num_samples = map(sum, zip(*datasets_train_valid_test_num_samples))
        
        max_steps = args.max_steps
        # num_train_samples = train_num_samples
        # delay_optimizer_creation = (
        #     self.sharding is not None
        #     and ShardingOption.SHARD_OP in self.args.sharding
        # )
        delay_optimizer_creation = False

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()

        if self.args.should_load_sharding_stage1_model:
            model = self._wrap_model_and_load_sharded_checkpoint(resume_from_checkpoint)
        elif self.args.should_save_sharding_stage1_model:
            # In the non-sharded mode, should invoke _load_from_checkpoint before _wrap_model.
            # In this mode, the rank0 load all params and the _wrap_model implicitly broadcast params from rank0 to the other ranks.
            model = self._wrap_model(self.model_wrapped)
            if self.sharding_io is not None:
                assert delay_optimizer_creation is False, "delay_optimizer_creation should be False"
                # the self.optimizer should be wrapped and it is done in _wrap_model
                self.sharding_io.set_optimizer(self.optimizer)
            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model
            if delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            self._load_optimizer_and_scheduler(resume_from_checkpoint)
        else:
            model = self._wrap_model(self.model_wrapped)
            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model
            if delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            self._load_optimizer_and_scheduler(resume_from_checkpoint)

        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Total num train samples = {train_num_samples:,}")

        # per_device_trainable_numel = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
        # TODO: Temporary fix since Tensor.numel() not supported in distributed mode
        per_device_trainable_numel = sum(np.prod(p.shape) for p in model.parameters() if not p.stop_gradient)
        logger.info(f"  Number of trainable parameters = {per_device_trainable_numel:,} (per device)")
        if self.args.use_hybrid_parallel:
            # todo fix for pipeline_parallel_degree
            parts_num = max(self.args.tensor_parallel_degree, 1) * max(self.args.pipeline_parallel_degree, 1)
            if parts_num > 1:
                all_reduce_dtype = "int64"
                if paddle.get_device().split(":")[0] in ["npu", "xpu"]:
                    # TODO(duanyanhui): fix when NPU all_reduce supports int64
                    all_reduce_dtype = "float32"
                trainable_numel_tensor = paddle.to_tensor(per_device_trainable_numel, dtype=all_reduce_dtype)
                paddle.distributed.all_reduce(trainable_numel_tensor)
                trainable_numel = int(trainable_numel_tensor.item()) // self.args.dataset_world_size
                # the numel is roughly, because the tensor parallel still hold own bias or layer_norm weight without splited
                # so, the trainable numel is a little bigger than real.
                logger.info(f"  Number of trainable parameters = {trainable_numel:,} (all devices, roughly)")

        self._globalstep_last_start_time = time.time()

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and distributed_isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                distributed_file(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            )
            self._load_rng_state(resume_from_checkpoint)
            if self.args.world_size > 1:
                global_step_list = []
                paddle.distributed.all_gather(
                    global_step_list, paddle.to_tensor([self.state.global_step], dtype="int64")
                )
                assert (
                    paddle.sum(paddle.stack(global_step_list) - global_step_list[0]) == 0
                ), f"Error, get different globel step, please check! step list: {[x.item() for x in global_step_list]}"

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from global step {self.state.global_step}")

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler

        self.state.max_steps = int(max_steps)
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        tr_loss = paddle.to_tensor(0.0)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        if self.args.device == "npu" and self.args.flatten_param_grads:
            from .plugins.npu_plugin import npu_accelerate_plugin

            npu_accelerate_plugin(self.optimizer)

        front_dataset_id = -1
        flag = 1
        if local_rank == 0:
            wait([prepare_future_0])
            prepare_future_0.cancel()
        while self.state.global_step < max_steps:
            self.timers and self.timers("read-dataset").start()
            dataset_id_map = self.get_dataset(datasets_train_valid_test_num_samples, total_train_batch_size, self.state.dataset_samples)
            dataset_id = dataset_id_map[self.state.global_step]
            if front_dataset_id != dataset_id or self.train_dataset is None:
                if self.train_dataset is not None:
                    del train_dataset
                if self.eval_dataset is not None:
                    del val_dataset
                train_dataset, val_dataset, test_dataset = build_train_valid_test_datasets(
                    [self.data[dataset_id*2+1]], 
                    self.data_args.data_impl,
                    self.data_args.split,
                    self.train_val_test_num_samples,
                    self.data_args.max_seq_length,
                    self.args.seed,
                    self.data_args.skip_warmup,
                    share_folder=self.data_args.share_folder,
                    data_cache_path=self.data_args.data_cache,
                )
                self.train_dataset = train_dataset
                self.eval_dataset = val_dataset
                self.test_dataset = test_dataset
                skip_step = 0
                if self.state.dataset_samples is None:
                    self.state.dataset_samples = dict()
                    save_name = self.data[dataset_id*2+1]
                    self.state.dataset_samples[save_name] = len(train_dataset)
                else:
                    save_name = self.data[dataset_id*2+1]
                    self.state.dataset_samples[save_name] = len(train_dataset)
                train_dataloader = self.get_train_dataloader()
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, DistributedBatchSampler
                ):
                    train_dataloader.batch_sampler.set_epoch(self.state.num_train_epochs)
                self.callback_handler.train_dataloader = train_dataloader
                if flag == 1 and resume_from_checkpoint is not None and distributed_isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
                    skip_step = self.get_offset_step(self.state.global_step, datasets_train_valid_test_num_samples, total_train_batch_size, self.state.dataset_samples)
                    if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                        train_dataloader.batch_sampler, NlpDistributedBatchSampler
                    ):
                        consumed_samples = (
                            skip_step
                            * args.train_batch_size
                            * args.gradient_accumulation_steps
                            * args.dataset_world_size
                        )
                        train_dataloader.batch_sampler.set_epoch(consumed_samples=consumed_samples)
                        logger.info(f"Set DistributedBatchSampler consumed_samples to {consumed_samples}")
                        flag = 0
                
            else:
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, DistributedBatchSampler
                ):
                    train_dataloader.batch_sampler.set_epoch(self.state.num_train_epochs)
            if local_rank == 0:
                prepare_future_0 = executor.submit(self.pre_process, False)
            self.timers and self.timers("read-dataset").stop()
            front_dataset_id = dataset_id
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            self.timers and self.timers("read-data").start()
            consumed_samples = (
                skip_step
                * args.train_batch_size
                * args.gradient_accumulation_steps
                * args.dataset_world_size
            )
            for step, inputs in enumerate(train_dataloader):
                if flag == 1 and resume_from_checkpoint is not None and distributed_isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)) and isinstance(train_dataloader.batch_sampler, DistributedBatchSampler):
                    if skip_step > 0:
                        skip_step -= 1
                        continue
                    else:
                        logger.info(f"Set DistributedBatchSampler consumed_samples to {consumed_samples}")
                        flag = 0
                self.timers and self.timers("read-data").stop()
                os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
                self.callback_handler.on_load_data_end(args, self.state, self.control, inputs=inputs)

                if self.state.global_step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                    self.timers and self.timers("forward-backward").start()

                dp_enabled = (
                    self.args.data_parallel_degree > 1 if self.args.use_hybrid_parallel else args.local_rank != -1
                )
                forbidden_no_sync = False
                # stage2 and stage3 should not no_sync, because the is no DDP wrapper and no_sync API
                # hybrid_parallel (tp or pp or sharding stage 1) should not no_sync
                if self.args.use_hybrid_parallel:
                    forbidden_no_sync = True

                availiable_no_sync = dp_enabled and not forbidden_no_sync

                is_no_sync = (
                    ((self.state.global_step + 1) % args.gradient_accumulation_steps != 0)
                    and availiable_no_sync
                    and args._no_sync_in_gradient_accumulation
                ) or (args.recompute and availiable_no_sync)
                # sharding
                # stage1. the same as ddp
                # stage2. manualy collect gradient on dp group
                if is_no_sync:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                tr_loss += tr_loss_step

                if (self.state.global_step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    (self.state.global_step + 1) % args.gradient_accumulation_steps != 0
                    and (self.state.global_step + 1) == max_steps
                ):
                    self.timers and self.timers("forward-backward").stop()
                    # Maunally collect gradients when group_sharded_parallel can't accept dp_group
                    # Case 1: Use sharding stage 2/3 with dp
                    # Case 2: Use recompute and dp
                    # local_rank != -1 don't means dp in networks.
                    self.timers and self.timers("all-reduce").start()

                    if self.sharding and ShardingOption.SHARD_OP not in self.args.sharding:
                        if self.args.data_parallel_degree > 1 and not is_dp_group_support_in_group_sharded_parallel():
                            fused_allreduce_gradients(model.parameters(), fleet.get_hybrid_communicate_group())
                            if ShardingOption.FULL_SHARD in self.args.sharding:
                                # Why need sync on parm again ?
                                # TODO: fix this.
                                for p in model.parameters():
                                    if hasattr(p, "bw_storage"):
                                        assert p.grad is None, "This case shouldn't happen."
                                        p.bw_storage.scale_(1.0 / self.dp_group.nranks)
                                        paddle.distributed.all_reduce(p.bw_storage, group=self.dp_group)

                    # Case 2: Use recompute and dp / sharding stage1,
                    # manualy collect gradient for dp.
                    elif args.recompute and availiable_no_sync:
                        fused_allreduce_gradients(list(model.parameters()), None)

                    pipeline_parallel_config = set(args.pipeline_parallel_config.split(" "))
                    enable_delay_scale_loss = "enable_delay_scale_loss" in pipeline_parallel_config
                    enable_dp_comm_overlap = "enable_dp_comm_overlap" in pipeline_parallel_config

                    if isinstance(self.optimizer, HybridParallelOptimizer) and not self.do_grad_scaling:
                        parameters_list = _obtain_optimizer_parameters_list(self.optimizer._inner_opt)

                        if not enable_dp_comm_overlap:
                            if self.optimizer._sharding_enable:
                                assert reshard_util.is_sharding_opt(self.optimizer)
                                self.optimizer._inner_opt.reduce_gradients(list(parameters_list), self.optimizer._hcg)

                            if self.optimizer._dp_enable:
                                fused_allreduce_gradients(list(parameters_list), self.optimizer._hcg)
                    self.timers and self.timers("all-reduce").stop()
                    self.timers and self.timers("optimizer-step").start()

                    # pipeline parallel mode,  handle gradient merge here
                    if args.pipeline_parallel_degree > 1 and enable_delay_scale_loss:
                        for p in model._layers.parameters():
                            with paddle.no_grad():
                                if hasattr(p, "main_grad") and p.main_grad is not None:
                                    assert p.grad is None
                                    p.main_grad.scale_(1.0 / self.args.gradient_accumulation_steps)
                                elif p.grad is not None:
                                    p.grad.scale_(1.0 / self.args.gradient_accumulation_steps)

                    # Optimizer step
                    self.callback_handler.on_optimizer_begin(
                        args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                    )
                    optimizer_was_run = True
                    if self.do_grad_scaling:
                        scale_before = paddle.assign(self.scaler._scale)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler._scale
                        optimizer_was_run = not self.scaler._cache_founf_inf
                        if not optimizer_was_run:
                            scale_before_value = scale_before.cpu().numpy()
                            scale_after_value = scale_after.cpu().numpy()
                            logger.warning(
                                f"optimizer not run, scale_before: {scale_before_value[0]}, scale_after: {scale_after_value[0]}"
                            )
                    elif isinstance(self.optimizer, HybridParallelOptimizer):
                        self.optimizer._step(parameters_list)
                    else:
                        self.optimizer.step()

                    self.timers and self.timers("optimizer-step").stop()

                    if optimizer_was_run:
                        self.lr_scheduler.step()

                    self.optimizer.clear_grad()
                    self.callback_handler.on_optimizer_end(
                        args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                    )

                    self.state.global_step += 1
                    self.state.epoch = 0
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, 0, ignore_keys_for_eval, inputs=inputs)
                    self._print_timer()
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.state.global_step >= max_steps:
                    break
                if self.get_dataset(datasets_train_valid_test_num_samples, total_train_batch_size, self.state.dataset_samples)[self.state.global_step] != front_dataset_id:
                    break
                self.timers and self.timers("read-data").start()

            self.state.num_train_epochs += 1
            if local_rank == 0:
                prepare_future_0.cancel()

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, 0, ignore_keys_for_eval, inputs=inputs)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\nTraining completed. \n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, LoRAModel) or isinstance(self.model, PrefixModelForCausalLM):
                self._load_best_model_from_peft_checkpoint()
            else:
                weight_name = PADDLE_WEIGHTS_NAME
                best_model_path = os.path.join(
                    self.state.best_model_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix)
                )
                if os.path.exists(best_model_path):
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = paddle.load(best_model_path, return_numpy=True)
                    # If the model is on the GPU, it still works!
                    self._set_state_dict_in_model(state_dict)
                else:
                    logger.warning(
                        f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                        "on multiple nodes, you should activate `--save_on_each_node`."
                    )

        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=train_num_samples, num_steps=self.state.max_steps)

        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    def predict(
        self, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.
        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)
        <Tip>
        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.
        </Tip>
        Returns: *NamedTuple* A namedtuple with the following keys:
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        
        test_dataloader = self.get_test_dataloader(self.test_dataset)
        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            test_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
            prediction_loss_only=True if self.compute_metrics is None else None,
            metric_key_prefix=metric_key_prefix,
            max_eval_iters=self.args.max_evaluate_steps,
        )
        total_batch_size = self.args.per_device_eval_batch_size * self.args.dataset_world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)
