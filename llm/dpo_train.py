""" Training DPO """

import os
import sys
import time


from functools import partial
from paddlenlp.datasets import InTokensIterableDataset, InTokensMapDataset, load_dataset

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
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForCausalLMPipe
# isort: on
from dpo_trainer import DPOTrainer
from dpo_utils import DataArgument, DPOTrainingArguments, ModelArgument
from dpo_data import process_example

# fmt: on
from paddlenlp.dataset.dpo import collate_fn, create_dataset
from dpo_estimate_training import dpo_estimate_training


def main():
    """main"""
    parser = PdArgumentParser((ModelArgument, DataArgument, DPOTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.fused_rotary:
        raise NotImplementedError("The fused_rotary dosen not support for DPO.")

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
        # if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 1:
        #     raise ValueError(
        #         f"Output directory ({training_args.output_dir}) already exists and is not empty. "
        #         "Use --overwrite_output_dir to overcome."
        #     )
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

    # fused_rotary and fused_softmax_mask only support for rocm.
    if not paddle.is_compiled_with_rocm():
        if model_args.fused_rotary:
            logger.warning("The fused_rotary flag is only available when using the ROCM version of paddlepaddle. ")
            model_args.fused_rotary = False
        if model_args.fused_softmax_mask:
            logger.warning(
                "The fused_softmax_mask flag is only available when using the ROCM version of paddlepaddle. "
            )
            model_args.fused_softmax_mask = False
    model_kwargs = dict(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        dtype=dtype,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        recompute=training_args.recompute,
        recompute_granularity=model_args.recompute_granularity,
        use_flash_attention=model_args.use_flash_attention,
        tensor_parallel_output=model_args.tensor_parallel_output,
        fused_linear=model_args.fused_linear,
        fused_rotary=model_args.fused_rotary,
        fused_softmax_mask=model_args.fused_softmax_mask,
        dpo=True,
        dpo_beta=training_args.dpo_beta,
        dpo_normalize_logps=training_args.dpo_normalize_logps,
        use_fast_ln=model_args.use_fast_ln,
        use_fast_ffn=model_args.use_fast_ffn,
    )
    if training_args.pipeline_parallel_degree > 1:
        model_class = AutoModelForCausalLMPipe
    else:
        model_class = AutoModelForCausalLM
    if not data_args.autotuner_benchmark and not data_args.dpo_benchmark:
        ref_model = model_class.from_pretrained(**model_kwargs)
        config = AutoConfig.from_pretrained(**model_kwargs)
        model = model_class._from_config(config, dtype=dtype)
        model.set_state_dict(ref_model.state_dict())
        # for DPO save
        model.config.dpo = False
    else:
        config = AutoConfig.from_pretrained(**model_kwargs)
        model = model_class._from_config(config, dtype=dtype)
        ref_model = model_class._from_config(config, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    logger.info("Loading model & tokenizer successfully !")

    logger.info("Start to create dataset ...")
    dataset_config = {
        "tokenizer": tokenizer,
        "max_seq_len": data_args.max_seq_length,
        "max_prompt_len": data_args.max_prompt_len,
        "random_seed": training_args.seed,
        "num_replicas": training_args.dataset_world_size,
        "rank": training_args.dataset_rank,
        "num_samples_each_epoch": data_args.num_samples_each_epoch,
        "greedy_intokens": data_args.greedy_intokens,
        "buffer_size": data_args.buffer_size,
    }
        
    if not data_args.autotuner_benchmark and training_args.max_steps == -1:
        if training_args.should_load_dataset and paddle.distributed.get_rank() == 0:
            # NOTE(gongenlei): not to feed train_dataset, or the data will be wrong in next training.
            training_args, res = dpo_estimate_training(tokenizer, data_args, training_args)

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
    if training_args.should_load_dataset:
        if data_args.train_task_config is None:
            raise ValueError(f"Please specific dataset name or path (got {data_args.train_task_config})")
        elif (
            os.path.exists(os.path.join(data_args.train_task_config, "train.json"))
        ):
            train_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "train.json"),
                lazy=data_args.lazy,
            )[0]
            train_dataset = trans_func(train_ds)

    if training_args.do_eval and training_args.should_load_dataset:
        if data_args.eval_task_config is None:
            raise ValueError(f"Please specific dataset name or path (got {data_args.eval_task_config})")
        elif (
            os.path.exists(os.path.join(data_args.eval_task_config, "eval.json"))
        ):
            eval_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "eval.json"),
                lazy=data_args.lazy,
            )[0]
            eval_dataset = trans_func(eval_ds)

    logger.info("Creating dataset successfully ...")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train and training_args.should_load_dataset else None,
        eval_dataset=eval_dataset if training_args.do_eval and training_args.should_load_dataset else None,
        tokenizer=tokenizer,
        data_collator=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_seq_len=data_args.max_seq_length,
        ),
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        if data_args.dpo_benchmark and training_args.should_load_dataset and paddle.distributed.get_rank() == 0:
            effective_tokens, total_tokens = res["effective_tokens"], res["total_tokens"]
            effective_tokens_per_second = effective_tokens / train_result.metrics["train_runtime"]
            logger.info("[timelog] {}: {:.2f} token/s ({}) ".format(
               "training speed", effective_tokens_per_second, time.strftime("%Y-%m-%d %H:%M:%S")))
            # logger.info("[timelog] {}: {} tokens ({}) ".format(
            #    "Effective_Tokens", effective_tokens, time.strftime("%Y-%m-%d %H:%M:%S")))
            # logger.info("[timelog] {}: {} tokens ({}) ".format(
            #    "Total_Tokens", total_tokens, time.strftime("%Y-%m-%d %H:%M:%S")))
            # logger.info("[timelog] {}: {:.2f} s ({}) ".format(
            #    "training running time", train_result.metrics["train_runtime"], time.strftime("%Y-%m-%d %H:%M:%S")))

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
