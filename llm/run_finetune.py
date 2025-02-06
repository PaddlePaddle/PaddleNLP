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
# import inspect
import json
import logging
import os
import sys
from functools import partial

import paddle
from utils.argument import GenerateArgument, ReftArgument
from utils.data import convert_example_for_reft, get_convert_example

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import (
    ZeroPaddingIterableDataset,
    ZeroPaddingMapDataset,
    load_dataset,
)
from paddlenlp.metrics import BLEU, Rouge1, Rouge2, RougeL
from paddlenlp.peft import (
    LoKrConfig,
    LoKrModel,
    LoRAConfig,
    LoRAModel,
    PrefixConfig,
    PrefixModelForCausalLM,
    VeRAConfig,
    VeRAModel,
)
from paddlenlp.peft.reft import (
    ReFTConfig,
    ReftDataCollator,
    ReFTModel,
    intervention_mapping,
)
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint, set_seed
from paddlenlp.trainer.trainer_callback import TrainerState
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
    AutoTokenizer,
    Llama3Tokenizer,
    LlamaForCausalLM,
    LlamaForCausalLMPipe,
    LlamaTokenizer,
    Qwen2ForCausalLM,
    Qwen2ForCausalLMPipe,
)
from paddlenlp.transformers.configuration_utils import LlmMetaConfig
from paddlenlp.trl import DataConfig, ModelConfig, SFTConfig, SFTTrainer
from paddlenlp.trl.llm_utils import (
    ZeroPaddingIterDatasetCallback,
    compute_metrics,
    get_lora_target_modules,
    get_prefix_tuning_params,
    init_chat_template,
)
from paddlenlp.utils.log import logger
from paddlenlp.utils.tools import get_env_device

# Fine-tune Environment Variables to support sharding stage1 overlap optimization.
os.environ["USE_CASUAL_MASK"] = "False"

flash_mask_support_list = [LlamaForCausalLM, LlamaForCausalLMPipe, Qwen2ForCausalLM, Qwen2ForCausalLMPipe]


def paddlenlp_verison_check():
    import paddlenlp
    from paddlenlp.utils.tools import compare_version

    if not compare_version(paddlenlp.__version__, "3.0.0.b2"):
        raise ValueError(
            "This scripts require paddlenlp >= 3.0.0b3, please reinstall: pip install paddlenlp >= 3.0.0b3 "
        )


def main():
    paddlenlp_verison_check()
    parser = PdArgumentParser((GenerateArgument, ModelConfig, ReftArgument, DataConfig, SFTConfig))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        gen_args, model_args, reft_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    else:
        gen_args, model_args, reft_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(gen_args, "Generation")

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    set_seed(seed=training_args.seed)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if get_env_device() == "xpu" and training_args.gradient_accumulation_steps > 1:
        try:
            from paddle_xpu.layers.nn.linear import LinearConfig  # noqa: F401

            LinearConfig.enable_accumulate_steps_opt()
            LinearConfig.set_accumulate_steps(training_args.gradient_accumulation_steps)
        except ImportError:
            # It's OK, not use accumulate_steps optimization
            pass

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
    quantization_config = dict(
        weight_quantize_algo=model_args.weight_quantize_algo,
        weight_blocksize=model_args.weight_blocksize,
        weight_double_quant=model_args.weight_double_quant,
        weight_double_quant_block_size=model_args.weight_double_quant_block_size,
    )

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        from_aistudio=model_args.from_aistudio,
        quantization_config=quantization_config,
    )

    if "Qwen2Moe" in str(model_config.architectures) and training_args.data_parallel_degree > 1:
        training_args.use_expert_parallel = True

    LlmMetaConfig.set_llm_config(model_config, training_args)
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

    # Config for model useing long sequence strategy
    if model_args.use_long_sequence_strategies:
        data_args.scaled_max_length = int(data_args.max_length * model_args.rope_scaling_factor)
        model_config.use_long_sequence_strategies = True
        model_config.long_sequence_strategy_type = model_args.strategy_type
        model_config.long_sequence_strategy_name = model_args.strategy_name
        model_config.rope_scaling_factor = model_args.rope_scaling_factor
        model_config.long_sequence_init_args = {
            "dim": int(model_config.hidden_size / model_config.num_attention_heads),
            "max_position_embeddings": data_args.scaled_max_length,  # extended context window
            "base": model_config.rope_theta,
            "scaling_factor": model_args.rope_scaling_factor,
        }
        if model_args.strategy_name == "YaRNScalingRotaryEmbedding":
            model_config.long_sequence_init_args["original_max_position_embeddings"] = data_args.max_length

    logger.info(f"Final model config: {model_config}")

    model_class = AutoModelForCausalLM
    if training_args.pipeline_parallel_degree > 1:
        if data_args.eval_with_do_generation and training_args.do_eval:
            raise ValueError("Plese set eval_with_do_generation to false in pipeline parallel mode.")

        model_class = AutoModelForCausalLMPipe

    if model_args.continue_training and not training_args.autotuner_benchmark:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            from_aistudio=model_args.from_aistudio,
        )
    else:
        # NOTE(gongenlei): new add autotuner_benchmark
        model = model_class.from_config(model_config, dtype=dtype)

    if model_args.flash_mask and (not data_args.zero_padding or not model.config.use_flash_attention):
        logger.warning("`flash_mask` must use with zero padding and flash attention.")
        data_args.zero_padding = True
        model.config.use_flash_attention = True

    if model_args.flash_mask and not any(isinstance(model, cls) for cls in flash_mask_support_list):
        raise NotImplementedError(f"{model.__class__} not support flash mask.")

    if training_args.do_train and model_args.neftune:
        # Inspired by https://github.com/neelsjain/NEFTune
        if hasattr(model, "get_input_embeddings"):

            def neft_post_hook(module, input, output):
                if module.training:
                    mag_norm = model_args.neftune_noise_alpha / paddle.sqrt(
                        paddle.to_tensor(output.shape[0] * output.shape[1], dtype="float32")
                    )
                    output = output + paddle.uniform(
                        shape=output.shape, dtype=output.dtype, min=-mag_norm, max=mag_norm
                    )
                return output

            neft_post_hook_handle = model.get_input_embeddings().register_forward_post_hook(neft_post_hook)
        else:
            raise NotImplementedError("Only support neftune for model with get_input_embeddings")

    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, from_aistudio=model_args.from_aistudio)
    reft_layers = None
    if model_args.reft:
        # reft requires padding side right
        tokenizer.padding_side = "right"
        layers = reft_args.layers
        if reft_args.layers != "all":
            layers = [int(l) for l in layers.split(";")]
        else:
            layers = [l for l in range(model_config.num_hidden_layers)]
        reft_layers = layers
        logging.info("Using ReFT with layers: ", reft_layers)
    # init chat_template for tokenizer
    init_chat_template(tokenizer, model_args.model_name_or_path, data_args.chat_template)

    # if using chat_template, data_args.eval_with_do_generation must be false
    if tokenizer.chat_template is not None:
        data_args.eval_with_do_generation = False

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, Llama3Tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_ds, dev_ds, test_ds = create_dataset(data_args, training_args)

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

    if training_args.pipeline_parallel_degree > 1:
        from utils.data import convert_example_common

        trans_func = partial(convert_example_common, tokenizer=tokenizer, data_args=data_args)
    elif model_args.reft:
        trans_func = partial(
            convert_example_for_reft,
            tokenizer=tokenizer,
            data_args=data_args,
            positions=reft_args.position,
            num_interventions=len(reft_layers),
        )
    else:
        trans_func = partial(get_convert_example(model), tokenizer=tokenizer, data_args=data_args)

    eval_zero_padding = data_args.zero_padding
    if data_args.zero_padding and data_args.eval_with_do_generation:
        logger.warning(
            "`zero_padding` conflicts with `eval_with_do_generation`. Setting zero_padding to False for the eval_dataset."
        )
        eval_zero_padding = False

    train_ds, dev_ds, test_ds = trans_dataset_to_ids(
        train_ds, dev_ds, test_ds, model_args, data_args, trans_func, eval_zero_padding
    )

    if data_args.zero_padding:
        if data_args.lazy:
            intoken_dataset = ZeroPaddingIterableDataset
        else:
            intoken_dataset = ZeroPaddingMapDataset
        logger.info("Creating Zero Padding Data Stream. This may take a few minutes.")
        if train_ds is not None:
            train_ds = intoken_dataset(
                train_ds,
                tokenizer=tokenizer,
                max_length=data_args.max_length,
                greedy_zero_padding=data_args.greedy_zero_padding,
            )
        if eval_zero_padding and dev_ds is not None:
            dev_ds = intoken_dataset(dev_ds, tokenizer=tokenizer, max_length=data_args.max_length)
        if eval_zero_padding and test_ds is not None:
            test_ds = intoken_dataset(test_ds, tokenizer=tokenizer, max_length=data_args.max_length)

    model = create_peft_model(model_args, reft_args, training_args, dtype, model_config, model, reft_layers)

    def compute_metrics_do_generation(eval_preds):
        rouge1 = Rouge1()
        rouge2 = Rouge2()
        rougel = RougeL()
        bleu4 = BLEU(n_size=4)

        predictions = [x[x != -100].tolist() for x in eval_preds.predictions]
        references = [x[x != -100].tolist() for x in eval_preds.label_ids]

        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        references = tokenizer.batch_decode(references, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if data_args.save_generation_output:
            with open(os.path.join(training_args.output_dir, "generated_output.json"), "w", encoding="utf-8") as f:
                for pred, ref in zip(predictions, references):
                    out = {"output": pred, "tgt": ref}
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

        # for pred in predictions:
        rouge1_score = rouge1.score(predictions, references)
        rouge2_score = rouge2.score(predictions, references)
        for pred, ref in zip(predictions, references):
            rougel.add_inst(pred, [ref])
            bleu4.add_inst(pred, [ref])
        return {
            "rouge1": rouge1_score,
            "rouge2": rouge2_score,
            "rougel": rougel.score(),
            "bleu4": bleu4.score(),
        }

    # Create trainer

    if (
        training_args.pipeline_parallel_degree > 1
        or training_args.sequence_parallel
        or training_args.autotuner_benchmark
        or data_args.zero_padding
        or data_args.pad_to_max_length
    ):
        # NOTE(gongenlei): new add autotuner_benchmark
        max_length = data_args.max_length
        padding = "max_length"
    else:
        max_length = None
        padding = True

    if training_args.pipeline_parallel_degree > 1:
        metrics = None
    elif data_args.eval_with_do_generation:
        metrics = compute_metrics_do_generation
    else:
        metrics = compute_metrics

    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        max_length=max_length,
        padding=padding,
        max_label_length=max_length,
        return_tensors="np",
        return_attention_mask=not model_args.flash_mask,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics,
        data_collator=data_collator_fn if not model_args.reft else ReftDataCollator(data_collator=data_collator_fn),
        do_generation=data_args.eval_with_do_generation,
        callbacks=[ZeroPaddingIterDatasetCallback()] if isinstance(train_ds, ZeroPaddingIterableDataset) else None,
        gen_args=gen_args,
        data_args=data_args,
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
        if model_args.neftune:
            neft_post_hook_handle.remove()
        if training_args.benchmark:
            total_effective_tokens = (
                sum([len(i["input_ids"]) for i in trainer.train_dataset]) * train_result.metrics["progress_or_epoch"]
            )
            effective_tokens_per_second = total_effective_tokens / train_result.metrics["train_runtime"]
            logger.info(f"Effective_Tokens_per_second: {effective_tokens_per_second} ")
            logger.info("Benchmark done.")
        else:
            if model_args.save_to_aistudio:
                save_to_aistudio(model_args, training_args, trainer)

            if not training_args.autotuner_benchmark:
                trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
                trainer.log_metrics("train", train_result.metrics)
                trainer.save_metrics("train", train_result.metrics)
                trainer.save_state()

    # Evaluation test set
    if training_args.do_predict:
        eval_result = trainer.predict(test_ds).metrics
        trainer.log_metrics("test", eval_result)

    # Evaluation dev set
    if training_args.do_eval:
        logger.info("*** Evaluate result after train ***")
        eval_result = trainer.evaluate(dev_ds)
        trainer.log_metrics("eval", eval_result)


def save_to_aistudio(model_args, training_args, trainer):
    kwargs = {}
    if model_args.aistudio_token is not None:
        kwargs["token"] = model_args.aistudio_token
        # PEFT Model only save PEFT parameters, if pretrained model obtains from aistudio
    if model_args.from_aistudio and (model_args.lora or model_args.prefix_tuning):
        kwargs["base_model"] = model_args.model_name_or_path
    else:
        trainer.tokenizer.save_to_aistudio(
            repo_id=model_args.aistudio_repo_id,
            private=model_args.aistudio_repo_private,
            license=model_args.aistudio_repo_license,
            exist_ok=True,
            **kwargs,
        )
    trainer.model.save_to_aistudio(
        repo_id=model_args.aistudio_repo_id,
        private=model_args.aistudio_repo_private,
        license=model_args.aistudio_repo_license,
        merge_tensor_parallel=training_args.tensor_parallel_degree > 1,
        exist_ok=True,
        **kwargs,
    )


def create_peft_model(model_args, reft_args, training_args, dtype, model_config, model, reft_layers):
    if model_args.prefix_tuning:
        if training_args.pipeline_parallel_degree > 1:
            raise NotImplementedError("Prefix tuning is not implemented for pipeline parallelism.")

        prefix_tuning_params = get_prefix_tuning_params(model)
        prefix_config = PrefixConfig(
            num_prefix_tokens=model_args.num_prefix_tokens,
            num_attention_heads=prefix_tuning_params["num_attention_heads"],
            num_hidden_layers=prefix_tuning_params["num_hidden_layers"],
            hidden_size=prefix_tuning_params["hidden_size"],
            multi_query_group_num=prefix_tuning_params["multi_query_group_num"],
            dtype=dtype,
        )
        if model_args.prefix_path is None:
            model = PrefixModelForCausalLM(
                model=model,
                prefix_config=prefix_config,
                postprocess_past_key_value=prefix_tuning_params["postprocess_past_key_value"],
            )
        else:
            model = PrefixModelForCausalLM.from_pretrained(
                model=model,
                prefix_path=model_args.prefix_path,
                postprocess_past_key_value=prefix_tuning_params["postprocess_past_key_value"],
            )
        model.print_trainable_parameters()

    if model_args.lora:
        if training_args.sharding_parallel_degree > 1:
            assert (
                "enable_stage1_overlap" not in training_args.sharding_parallel_config
            ), "Currently not support enabling sharding_stage1_overlap in lora mode."
        if model_args.lora_path is None:
            target_modules = get_lora_target_modules(model)
            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=2 * model_args.lora_rank if not model_args.rslora else 4,
                rslora=model_args.rslora,
                lora_plus_scale=model_args.lora_plus_scale,
                pissa=model_args.pissa,
                merge_weights=False,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                dtype=dtype,
                base_model_name_or_path=model_args.model_name_or_path,
                use_quick_lora=model_args.use_quick_lora,
                lora_use_mixer=model_args.lora_use_mixer,
                use_mora=model_args.use_mora,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)

        model.print_trainable_parameters()

    if model_args.lokr:
        if model_args.lokr_path is None:
            target_modules = get_lora_target_modules(model)
            lokr_config = LoKrConfig(
                target_modules=target_modules,
                lokr_dim=model_args.lokr_dim,
                dtype=dtype,
                base_model_name_or_path=model_args.model_name_or_path,
            )
            model = LoKrModel(model, lokr_config)
        else:
            model = LoKrModel.from_pretrained(model=model, lokr_path=model_args.lokr_path)

    if model_args.reft:
        intervention_dtype = dtype
        intervention_params = {
            "embed_dim": model_config.hidden_size,
            "low_rank_dimension": reft_args.rank,
            "dropout": reft_args.dropout,
            "dtype": intervention_dtype,
            "act_fn": reft_args.act_fn,
            "device": "gpu",
            "add_bias": reft_args.add_bias,
        }
        representations = [
            {
                "layer": l,
                "component": "block_output",
                "low_rank_dimension": reft_args.rank,
                "intervention": intervention_mapping[reft_args.intervention_type](**intervention_params),
            }
            for l in reft_layers
        ]
        reft_config = ReFTConfig(
            representations=representations, intervention_params=intervention_params, position=reft_args.position
        )
        # get reft model
        model = ReFTModel(reft_config, model)
        # disable origianl model gradients
        model.disable_model_gradients()
        model.print_trainable_parameters()

    if model_args.vera:
        target_modules = get_lora_target_modules(model)
        vera_config = VeRAConfig(
            target_modules=target_modules,
            r=model_args.vera_rank,
            vera_alpha=model_args.vera_rank,
            dtype=dtype,
            base_model_name_or_path=model_args.model_name_or_path,
            pissa_init=True,
        )
        model = VeRAModel(model, vera_config)
        model.mark_only_vera_as_trainable(notfreezeB=True)
        model.print_trainable_parameters()

    return model


def trans_dataset_to_ids(train_ds, dev_ds, test_ds, model_args, data_args, trans_func, eval_zero_padding):
    if train_ds is not None:
        train_ds = train_ds.map(
            partial(trans_func, is_test=False, zero_padding=data_args.zero_padding, flash_mask=model_args.flash_mask)
        )
    if dev_ds is not None:
        dev_ds = dev_ds.map(
            partial(
                trans_func,
                is_test=data_args.eval_with_do_generation,
                zero_padding=eval_zero_padding,
                flash_mask=model_args.flash_mask,
            )
        )
    if test_ds is not None:
        test_ds = test_ds.map(partial(trans_func, is_test=data_args.eval_with_do_generation))

    return train_ds, dev_ds, test_ds


def create_dataset(data_args, training_args):
    if data_args.dataset_name_or_path is None:
        raise ValueError(f"Please specific dataset name or path (got {data_args.dataset_name_or_path})")

    train_ds = None
    dev_ds = None
    test_ds = None
    if os.path.exists(os.path.join(data_args.dataset_name_or_path, "train.json")) or os.path.exists(
        os.path.join(data_args.dataset_name_or_path, "dev.json")
    ):
        if training_args.do_train:
            train_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "train.json"),
                lazy=data_args.lazy,
            )[0]
        if training_args.do_eval:
            dev_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "dev.json"),
                lazy=data_args.lazy,
            )[0]
        if training_args.do_predict:
            test_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "test.json"),
                lazy=data_args.lazy,
            )[0]

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
        if training_args.do_eval:
            dev_ds = load_dataset(
                "json",
                data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "dev", "*.json")),
                lazy=data_args.lazy,
            )[0]
        if training_args.do_predict:
            test_ds = load_dataset(
                "json",
                data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "test", "*.json")),
                lazy=data_args.lazy,
            )[0]
    else:
        if training_args.do_train:
            train_ds = load_dataset(data_args.dataset_name_or_path, splits=["train"])[0]

        if training_args.do_eval:
            dev_ds = load_dataset(data_args.dataset_name_or_path, splits=["dev"])[0]

        if training_args.do_predict:
            test_ds = load_dataset(data_args.dataset_name_or_path, splits=["test"])[0]

    return train_ds, dev_ds, test_ds


if __name__ == "__main__":
    main()
