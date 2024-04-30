""" Training DPO """

import os
import sys

from functools import partial
from paddlenlp.datasets import InTokensMapDataset, load_dataset

import paddle
from paddlenlp.trainer import (
    IntervalStrategy,
    PdArgumentParser,
    get_last_checkpoint,
    set_seed,
)
from paddlenlp.utils.log import logger

# isort: off
# fmt: off
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# isort: on
from dpo_trainer import DPOTrainer
from dpo_utils import DataArgument, DPOTrainingArguments, ModelArgument
from dpo_data import process_example, collate_fn

# fmt: on
#from dpo_estimate_training import dpo_estimate_training


def main():
    """main"""
    parser = PdArgumentParser((ModelArgument, DataArgument, DPOTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    if data_args.autotuner_benchmark:
        training_args.num_train_epochs = 1
        training_args.max_steps = 5
        training_args.do_train = True
        training_args.do_export = False
        training_args.do_predict = False
        training_args.do_eval = False
        training_args.overwrite_output_dir = True
        training_args.load_best_model_at_end = False
        training_args.report_to = []
        training_args.save_strategy = IntervalStrategy.NO
        training_args.evaluation_strategy = IntervalStrategy.NO
    if data_args.dpo_benchmark:
        training_args.max_steps = -1
        training_args.do_train = True
        training_args.do_export = False
        training_args.do_predict = False
        training_args.do_eval = False
        training_args.overwrite_output_dir = True
        training_args.load_best_model_at_end = False
        training_args.report_to = []
        training_args.save_strategy = IntervalStrategy.NO
        training_args.evaluation_strategy = IntervalStrategy.NO

    paddle.set_device(training_args.device)

    set_seed(training_args.seed)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: "
        f"{training_args.world_size}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set the dtype for loading model
    dtype = paddle.get_default_dtype()
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    logger.info("Start to load model ...")

    model_kwargs = dict(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        dtype=dtype,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        recompute_granularity=model_args.recompute_granularity,
        use_flash_attention=model_args.use_flash_attention,
        tensor_parallel_output=True,
    )
    if training_args.pipeline_parallel_degree > 1:
        raise ValueError(
            "DPO does not support pipeline parallelism yet."
        )

    model_class = AutoModelForCausalLM

    if not data_args.autotuner_benchmark and not data_args.dpo_benchmark:
        ref_model = model_class.from_pretrained(**model_kwargs)
        config = AutoConfig.from_pretrained(**model_kwargs)
        config.use_recompute = training_args.recompute
        model = model_class.from_config(config, dtype=dtype)
        model.set_state_dict(ref_model.state_dict())
    else:
        config = AutoConfig.from_pretrained(**model_kwargs)
        model = model_class.from_config(config, dtype=dtype)
        ref_model = model_class.from_config(config, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    logger.info("Loading model & tokenizer successfully !")

    logger.info("Start to create dataset ...")
        
    if not data_args.autotuner_benchmark and training_args.max_steps == -1:
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()
            pd_max_steps = paddle.to_tensor([training_args.max_steps])
            paddle.distributed.broadcast(pd_max_steps, src=0)
            training_args.max_steps = int(pd_max_steps.item())
        logger.info(
                f"Re-setting training_args.max_steps to {training_args.max_steps} ({training_args.num_train_epochs})"
            )
    if training_args.save_strategy == IntervalStrategy.EPOCH:
        training_args.save_strategy = IntervalStrategy.STEPS
        training_args.save_steps = int(training_args.max_steps / training_args.num_train_epochs)
    if training_args.evaluation_strategy == IntervalStrategy.EPOCH:
        training_args.evaluation_strategy = IntervalStrategy.STEPS
        training_args.eval_steps = int(training_args.max_steps / training_args.num_train_epochs)
    if training_args.logging_strategy == IntervalStrategy.EPOCH:
        training_args.logging_strategy = IntervalStrategy.STEPS
        training_args.logging_steps = int(training_args.max_steps / training_args.num_train_epochs)

    trans_func = partial(process_example, tokenizer=tokenizer, data_args=data_args)
    intoken_dataset = InTokensMapDataset
    if training_args.should_load_dataset:
        if data_args.dataset_name_or_path is None or not os.path.exists(os.path.join(data_args.dataset_name_or_path, "train.json")):
            raise ValueError(f"Please specific dataset name or path (got {data_args.dataset_name_or_path})")

        logger.info("Creating Zero Padding Data Stream. This may take a few minutes.")
        train_ds = load_dataset(
            "json",
            data_files=os.path.join(data_args.dataset_name_or_path, "train.json"),
        )[0]
        train_ds = (
            intoken_dataset(
                train_ds.map(trans_func),
                tokenizer=tokenizer,
                max_length=data_args.max_seq_length,
            )
            if train_ds is not None
            else None
        )

    if training_args.do_eval and training_args.should_load_dataset:
        if data_args.dataset_name_or_path is None or not os.path.exists(os.path.join(data_args.dataset_name_or_path, "dev.json")):
            raise ValueError(f"Please specific dataset name or path (got {data_args.dataset_name_or_path})")

        eval_ds = load_dataset(
            "json",
            data_files=os.path.join(data_args.dataset_name_or_path, "dev.json"),
        )[0]
        eval_ds = (
            intoken_dataset(
                eval_ds.map(trans_func),
                tokenizer=tokenizer,
                max_length=data_args.max_seq_length,
            )
            if eval_ds is not None
            else None
        )

    logger.info("Creating dataset successfully ...")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train and training_args.should_load_dataset else None,
        eval_dataset=eval_ds if training_args.do_eval and training_args.should_load_dataset else None,
        tokenizer=tokenizer,
        data_collator=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
        ),
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        if not data_args.autotuner_benchmark and not data_args.dpo_benchmark:
            trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)


if __name__ == "__main__":
    with paddle.amp.auto_cast(enable=False):
        main()
