# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import lr
import numpy as np
import paddle
from dataset import GPTDataset, get_train_valid_test_split_

from paddlenlp.data import Stack, Tuple
from paddlenlp.ops import Topology
from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
    speed_metrics,
)
from paddlenlp.transformers import (
    GPTChineseTokenizer,
    GPTForPretraining,
    GPTPretrainingCriterion,
    GPTTokenizer,
)
from paddlenlp.utils.batch_sampler import DistributedBatchSampler
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "gpt": (GPTForPretraining, GPTTokenizer),
    "gpt-cn": (GPTForPretraining, GPTChineseTokenizer),
}


@dataclass
class ModelArguments(TrainingArguments):
    model_name_or_path: str = field(default="gpt2-en", metadata={"help": ""})
    max_seq_len: int = field(default=128, metadata={"help": "max sequence length"})
    input_dir: str = field(default="", metadata={"help": "input idr of dataset"})
    to_static: bool = field(default=False, metadata={"help": "whether use static pretraining mode."})
    min_lr: float = field(default=1e-5, metadata={"help": "The initial min learning rate for Adam."})
    split: str = field(default="949,50,1", metadata={"help": "Train/valid/test data split."})
    lr_decay_style: str = field(default="none", metadata={"help": "style of learning rate decay"})
    use_amp: bool = field(default=False, metadata={"help": "whether use amp mode."})

    scale_loss: float = field(
        default=32768, metadata={"help": "The value of scale_loss for fp16. This is only used for AMP training."}
    )

    # per_device_train_batch_size
    @property
    def micro_batch_size(self):
        return self.per_device_train_batch_size

    @property
    def eval_freq(self):
        return self.eval_steps


def set_seed(args):
    if args.device == "cpu":
        idx = 0
    else:
        idx = paddle.distributed.get_rank()
    random.seed(args.seed + idx)
    np.random.seed(args.seed + idx)
    paddle.seed(args.seed + idx)


def create_pretrained_dataset(
    args,
    input_path,
    local_rank,
    data_world_rank,
    data_world_size,
    eos_id,
    worker_init=None,
    max_seq_len=1024,
    places=None,
    data_holders=None,
    pipeline_mode=False,
):

    if local_rank == 0:
        try:
            from tool_helpers import helpers
        except:  # noqa: E722
            start_time = time.time()
            print("> compiling dataset index builder ...")
            from data_tools.dataset_utils import compile_helper

            compile_helper()
            print(
                ">>> done with dataset index builder. Compilation time: {:.3f} "
                "seconds".format(time.time() - start_time),
                flush=True,
            )
            import data_tools.helpers as helpers

    device_world_size = paddle.distributed.get_world_size()

    if device_world_size > 1 and local_rank != 0:
        while True:
            try:
                try:
                    from tool_helpers import helpers  # noqa: F811
                except:  # noqa: E722
                    import data_tools.helpers as helpers  # noqa: F401
                break
            except:  # noqa: E722
                print("> wait for helpers to be compiled!")
                time.sleep(1)

    logger.info(
        "The distributed run, total device num:{}, distinct dataflow num:{}.".format(
            device_world_size, data_world_size
        )
    )

    assert len(input_path) == 1, "GPT only support one dataset for now."

    input_prefix = input_path[0]

    if os.path.isfile(input_prefix + "_ids.npz"):
        logger.warning("You are using compatible dataset, please make new dataset as the readme!")
        process_data = np.load(input_prefix + "_ids.npz", mmap_mode="r+", allow_pickle=True)
        sample_ids = process_data["ids"]
        sample_lens = process_data["lens"].astype("int32")
    else:
        for suffix in ["_ids.npy", "_idx.npz"]:
            if not os.path.isfile(input_prefix + suffix):
                raise ValueError("File Not found, %s" % (input_prefix + suffix))

        sample_ids = np.load(input_prefix + "_ids.npy", mmap_mode="r", allow_pickle=True)
        # All documment ids, extend as 1-D array.

        process_data = np.load(input_prefix + "_idx.npz")
        # The len(sample_lens) num of docs
        # The sum(sample_lens) should equal len(sample_ids)
        sample_lens = process_data["lens"]

    splits = get_train_valid_test_split_(args.split, len(sample_lens))
    assert len(sample_lens) >= splits[-1], "The document nums should larger than max of splits, but %s < %s" % (
        len(sample_lens),
        splits[-1],
    )

    def build_dataset(index, name, num_samples):
        return GPTDataset(
            file_prefix=input_prefix,
            build_data_file=local_rank == 0,
            micro_batch_size=args.micro_batch_size,
            name="gpt_" + name,
            max_seq_len=max_seq_len,
            num_samples=num_samples,
            documents=np.arange(splits[index], splits[index + 1]),
            sample_ids=sample_ids,
            sample_lens=sample_lens,
            eos_id=eos_id,
            seed=args.seed,
        )

    # Note, data should be broardcast to all devices.
    # for train, valid, test, the distinct data num is data_world_size
    train_dataset = build_dataset(0, "train", args.micro_batch_size * args.max_steps * data_world_size)
    if pipeline_mode:
        valid_dataset, test_dataset = None, None
    else:
        valid_dataset = build_dataset(
            1,
            "valid",
            args.micro_batch_size * (args.max_steps // args.eval_freq + 1) * args.eval_iters * data_world_size,
        )
        test_dataset = build_dataset(2, "test", args.micro_batch_size * args.test_iters * data_world_size)

    return train_dataset, valid_dataset, test_dataset


class PretrainingTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataloader = getattr(self, "eval_dataloader", None)
        if eval_dataloader is None:
            eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            # must call data loader, otherwise, it will init many times, cause OOM error.
            self.eval_dataloader = eval_dataloader()

        start_time = time.time()
        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        eval_loop = self.evaluation_loop

        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if compute_metrics is None else None,
            ignore_keys=ignore_keys,
            # Only evaluate max_eval_iters
            max_eval_iters=self.args.eval_iters,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        return output.metrics

    def _get_eval_sampler(self, eval_dataset) -> Optional[paddle.io.Sampler]:
        return DistributedBatchSampler(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=self.args.dataloader_drop_last,
        )

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        return DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=self.args.dataloader_drop_last,
        )


def get_train_data_file(args):
    files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if (os.path.isfile(os.path.join(args.input_dir, f)) and str(f).endswith("_idx.npz"))
    ]
    files = [x.replace("_idx.npz", "") for x in files]
    if len(files) == 0:
        logger.warning(
            "Not found dataset with name of xxx_ids.npy and xxx_idx.npz! Try to found old compatible xxx_ids.npz file."
        )
    else:
        return files

    files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if (os.path.isfile(os.path.join(args.input_dir, f)) and str(f).endswith("_ids.npz"))
    ]

    files = [x.replace("_ids.npz", "") for x in files]
    return files


def get_device(device: str):
    if device == "cpu":
        return "cpu"
    if paddle.get_device() == "cpu":
        logger.warning("not detect gpu but receive GPU related params, we will run the model on cpu")
        return "cpu"
    return device


def do_train():
    model_args: ModelArguments = PdArgumentParser([ModelArguments]).parse_args_into_dataclasses()[0]
    model_args.eval_iters = 10
    model_args.test_iters = model_args.eval_iters * 10

    model_args.device = get_device(model_args.device)

    paddle.set_device(model_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    worker_index = paddle.distributed.get_rank()
    worker_num = paddle.distributed.get_world_size()
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
    set_seed(model_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(model_args.output_dir) and model_args.do_train and not model_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(model_args.output_dir)
        if last_checkpoint is not None and model_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Now, we only support data parallel in dygraph mode for now.
    topo = Topology(device_rank=worker_index, world_size=worker_num, dp_degree=worker_num)

    tokenizer = GPTTokenizer.from_pretrained(model_args.model_name_or_path)
    pretrained_models_list = list(GPTForPretraining.pretrained_init_configuration.keys())
    model = GPTForPretraining.from_pretrained(model_args.model_name_or_path)

    # Create the critrion for the gpt model
    criterion = GPTPretrainingCriterion()

    # decorate @to_static for benchmark, skip it by default.
    if model_args.to_static:
        specs = None
        model = paddle.jit.to_static(model, input_spec=specs)
        logger.info("Successfully to apply @to_static with specs: {}".format(specs))

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    lr_scheduler = None

    if model_args.lr_decay_style == "none":
        lr_scheduler = None
    elif model_args.lr_decay_style == "cosine":
        lr_scheduler = lr.CosineAnnealingWithWarmupDecay(
            max_lr=model_args.learning_rate, min_lr=model_args.min_lr, warmup_step=model_args.warmup_steps
        )

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler if lr_scheduler is not None else model_args.learning_rate,
        beta1=model_args.adam_beta1,
        beta2=model_args.adam_beta2,
        epsilon=model_args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=model_args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    if model_args.model_name_or_path not in pretrained_models_list:
        logger.info("Try to load checkpoint from %s " % model_args.model_name_or_path)
        opt_path = os.path.join(model_args.model_name_or_path, "model_state.pdopt")
        if os.path.exists(opt_path):
            opt_dict = paddle.load(opt_path)
            optimizer.set_state_dict(opt_dict)
        else:
            logger.warning("No optimizer checkpoint file found in %s." % opt_path)

    files = get_train_data_file(model_args)
    files.sort()
    num_files = len(files)
    for f_id in range(num_files):
        data_file = files[f_id]
        train_dataset, valid_dataset, test_dataset = create_pretrained_dataset(
            model_args,
            [data_file],
            local_rank=local_rank,
            data_world_size=topo.data_info.size,
            data_world_rank=topo.data_info.rank,
            max_seq_len=model_args.max_seq_len,
            eos_id=tokenizer.eos_token_id,
        )

        trainer = PretrainingTrainer(
            model=model,
            criterion=criterion,
            args=model_args,
            data_collator=Tuple(Stack(), Stack(), Stack(), Stack(), Stack()),
            train_dataset=train_dataset if model_args.do_train else None,
            eval_dataset=valid_dataset if model_args.do_eval else None,
            optimizers=(None, lr_scheduler),
            tokenizer=tokenizer,
        )

        checkpoint = None
        if model_args.resume_from_checkpoint is not None:
            checkpoint = model_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        # Training
        if model_args.do_train:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            trainer.save_model()
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        if model_args.do_predict:
            test_ret = trainer.predict(test_dataset)
            trainer.log_metrics("test", test_ret.metrics)


if __name__ == "__main__":
    do_train()
