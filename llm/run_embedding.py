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
# import inspect
import os
import sys

import paddle
from utils.argument import EmbeddingArgument

from paddlenlp.data import DataCollatorForEmbedding
from paddlenlp.datasets import EmbeddingIterableDataset, load_dataset
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint, set_seed
from paddlenlp.trainer.trainer_callback import TrainerState
from paddlenlp.transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen2Config,
    Qwen2SentenceEmbedding,
)
from paddlenlp.transformers.configuration_utils import LlmMetaConfig
from paddlenlp.transformers.refined_recompute import update_refined_recompute
from paddlenlp.trl import DataConfig, EmbeddingTrainer, ModelConfig, SFTConfig
from paddlenlp.trl.llm_utils import compute_metrics, init_chat_template
from paddlenlp.utils.log import logger

# Fine-tune Environment Variables to support sharding stage1 overlap optimization.
os.environ["USE_CASUAL_MASK"] = "False"


def main():
    parser = PdArgumentParser((ModelConfig, DataConfig, SFTConfig, EmbeddingArgument))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, embedding_args = parser.parse_json_file_and_cmd_lines()
    else:
        model_args, data_args, training_args, embedding_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    set_seed(seed=training_args.seed)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    if training_args.pipeline_parallel_degree > 1:
        raise NotImplementedError("Cannot support pipeline parallel for Embedding training now.")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load model
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        elif training_args.bf16:
            dtype = "bfloat16"
        else:
            raise ValueError("Please specific dtype: --fp16 or --bf16")
    else:
        dtype = "float32"

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        from_aistudio=model_args.from_aistudio,
    )
    assert isinstance(model_config, Qwen2Config), "Now only qwen2 supported"

    LlmMetaConfig.set_llm_config(model_config, training_args)
    model_config.refined_recompute = update_refined_recompute(training_args.refined_recompute)
    model_config.use_fast_layer_norm = model_args.use_fast_layer_norm

    # Config for model using dropout, such as GPT.
    if hasattr(model_config, "hidden_dropout_prob"):
        model_config.hidden_dropout_prob = model_args.hidden_dropout_prob
    if hasattr(model_config, "attention_probs_dropout_prob"):
        model_config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob
    if hasattr(model_config, "ignore_index"):
        model_config.ignore_index = -100

    if model_args.fuse_attention_qkv is not None:
        model_config.fuse_attention_qkv = model_args.fuse_attention_qkv
    if model_args.fuse_attention_ffn is not None:
        model_config.fuse_attention_ffn = model_args.fuse_attention_ffn

    model_config.seq_length = data_args.max_length
    model_config.embedding_negatives_cross_device = embedding_args.embedding_negatives_cross_device
    logger.info(f"Final model config: {model_config}")

    model_class = Qwen2SentenceEmbedding

    if model_args.continue_training and not training_args.autotuner_benchmark:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            from_aistudio=model_args.from_aistudio,
        )
    else:
        model = model_class.from_config(model_config, dtype=dtype)

    if model_args.flash_mask and (not data_args.zero_padding or not model.config.use_flash_attention):
        logger.warning("`flash_mask` must use with zero padding and flash attention.")
        data_args.zero_padding = True
        model.config.use_flash_attention = True

    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, from_aistudio=model_args.from_aistudio)

    # init chat_template for tokenizer
    init_chat_template(tokenizer, model_args.model_name_or_path, data_args.chat_template)

    # if using chat_template, data_args.eval_with_do_generation must be false
    if tokenizer.chat_template is not None:
        data_args.eval_with_do_generation = False

    if training_args.do_eval:
        logger.warning("Warning: 'do_eval' is set to True, but will be set to False for Embedding training currently.")
        training_args.do_eval = False
        training_args.evaluation_strategy = "no"

    if data_args.dataset_name_or_path is None:
        raise ValueError(f"Please specific dataset name or path (got {data_args.dataset_name_or_path})")
    elif os.path.exists(os.path.join(data_args.dataset_name_or_path, "train.json")) or os.path.exists(
        os.path.join(data_args.dataset_name_or_path, "dev.json")
    ):
        if training_args.do_train:
            train_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "train.json"),
                lazy=data_args.lazy,
            )[0]
        else:
            train_ds = None
        if training_args.do_eval:
            dev_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "dev.json"),
                lazy=data_args.lazy,
            )[0]
        else:
            dev_ds = None

    elif os.path.exists(os.path.join(data_args.dataset_name_or_path, "train")) or os.path.exists(
        os.path.join(data_args.dataset_name_or_path, "dev")
    ):
        import glob

        if training_args.do_train:
            train_ds = load_dataset(
                "json",
                data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "train", "*.json")),
                lazy=data_args.lazy,
            )[0]
        else:
            train_ds = None
        if training_args.do_eval:
            dev_ds = load_dataset(
                "json",
                data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "dev", "*.json")),
                lazy=data_args.lazy,
            )[0]
        else:
            dev_ds = None

    else:
        if training_args.do_train:
            train_ds = load_dataset(data_args.dataset_name_or_path, splits=["train"])[0]
        else:
            train_ds = None
        if training_args.do_eval:
            dev_ds = load_dataset(data_args.dataset_name_or_path, splits=["dev"])[0]
        else:
            dev_ds = None

    # TODO(ZHUI & sijunhe): Temporary implementation. Generalize this logic and move to Trainer later.
    if training_args.resume_from_checkpoint is not None and data_args.lazy:
        logger.info(
            f"Loading from '{training_args.resume_from_checkpoint}' with `lazy=True`, manually skipping dataset and setting `ignore_data_skip` to True."
        )
        training_args.ignore_data_skip = True
        state = TrainerState.load_from_json(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json"))
        if state.trial_params is not None and "zero_padding_global_step" in state.trial_params:
            consumed_samples = state.trial_params["zero_padding_global_step"]
        else:
            consumed_samples = (
                state.global_step
                * training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.dataset_world_size
            )
        logger.info(
            f"Skipping the first {consumed_samples} samples to warmup the dataset from checkpoint '{training_args.resume_from_checkpoint}'."
        )
        train_ds = train_ds.skip(consumed_samples)

    if train_ds is not None:
        train_ds = EmbeddingIterableDataset(
            train_ds,
            tokenizer,
            max_query_len=embedding_args.max_query_len,
            max_passage_len=embedding_args.max_passage_len,
            group_size=embedding_args.group_size,
            query_template=embedding_args.query_template,
            passage_template=embedding_args.passage_template,
        )

    if dev_ds is not None:
        dev_ds = EmbeddingIterableDataset(
            dev_ds,
            tokenizer,
            max_query_len=embedding_args.max_query_len,
            max_passage_len=embedding_args.max_passage_len,
            group_size=embedding_args.group_size,
            query_template=embedding_args.query_template,
            passage_template=embedding_args.passage_template,
        )

    # Create trainer
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = True

    data_collator_fn = DataCollatorForEmbedding(
        tokenizer=tokenizer,
        max_query_len=embedding_args.max_query_len,
        padding=padding,
        max_passage_len=embedding_args.max_passage_len,
        return_tensors="np",
        return_attention_mask=not model_args.flash_mask,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
    )
    trainer = EmbeddingTrainer(
        model=model,
        model_args=embedding_args,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator_fn,
    )
    trainable_parameters = [p for p in model.parameters() if not p.stop_gradient]
    trainer.set_optimizer_grouped_parameters(trainable_parameters)

    # Train
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation dev set
    if training_args.do_eval:
        logger.info("*** Evaluate result after train ***")
        eval_result = trainer.evaluate(dev_ds)
        trainer.log_metrics("eval", eval_result)


if __name__ == "__main__":
    main()
