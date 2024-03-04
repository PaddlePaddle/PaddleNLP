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
import json
import os
import sys
from functools import partial

import paddle
from argument import (
    DataArgument,
    GenerateArgument,
    ModelArgument,
    QuantArgument,
    TrainingArguments,
)
from data import get_convert_example
from utils import (
    CausalLMTrainer,
    InTokensIterDatasetCallback,
    compute_metrics,
    get_lora_target_modules,
    get_prefix_tuning_params,
    init_chat_template,
)

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import InTokensIterableDataset, InTokensMapDataset, load_dataset
from paddlenlp.metrics import BLEU, Rouge1, Rouge2, RougeL
from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint
from paddlenlp.trainer.trainer_callback import TrainerState
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)
from paddlenlp.utils.log import logger


def read_local_dataset(path):
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            yield json.loads(line.strip())


def main():
    # Arguments
    parser = PdArgumentParser((GenerateArgument, QuantArgument, ModelArgument, DataArgument, TrainingArguments))
    # Support format as "args.json --arg1 value1 --arg2 value2.”
    # In case of conflict, command line arguments take precedence.
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        gen_args, quant_args, model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    else:
        gen_args, quant_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(quant_args, "Quant")
    training_args.print_config(gen_args, "Generation")

    if sum([quant_args.do_ptq, quant_args.do_qat, quant_args.do_gptq, training_args.do_train]) > 1:
        raise ValueError(
            "--do_train, --do_ptq, --do_gptq and --do_qat cannot work at the same time. Please choose only one at a time"
        )

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
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
    if training_args.pipeline_parallel_degree > 1:
        if data_args.eval_with_do_generation and training_args.do_eval:
            raise ValueError("Plese set eval_with_do_generation to false in pipeline parallel mode.")
        from paddlenlp.transformers import AutoModelForCausalLMPipe

        if not training_args.autotuner_benchmark:
            model = AutoModelForCausalLMPipe.from_pretrained(
                model_args.model_name_or_path,
                tensor_parallel_output=False,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                tensor_parallel_rank=training_args.tensor_parallel_rank,
                use_flash_attention=model_args.use_flash_attention,
                dtype=dtype,
                from_aistudio=model_args.from_aistudio,
                quantization_config=quantization_config,
            )
        else:
            # NOTE(gongenlei): new add autotuner_benchmark
            model_config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                tensor_parallel_output=False,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                tensor_parallel_rank=training_args.tensor_parallel_rank,
                dtype=dtype,
                from_aistudio=model_args.from_aistudio,
                quantization_config=quantization_config,
            )
            model = AutoModelForCausalLMPipe.from_config(model_config, dtype=dtype)
    else:
        model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            tensor_parallel_output=False,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
            dtype=dtype,
            from_aistudio=model_args.from_aistudio,
            quantization_config=quantization_config,
        )
        if hasattr(model_config, "use_flash_attention"):
            model_config.use_flash_attention = model_args.use_flash_attention

        if not training_args.autotuner_benchmark:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=model_config,
                from_aistudio=model_args.from_aistudio,
            )
        else:
            # NOTE(gongenlei): new add autotuner_benchmark
            model = AutoModelForCausalLM.from_config(model_config, dtype=dtype)
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
    # init chat_template for tokenizer
    init_chat_template(tokenizer, model_args.model_name_or_path, data_args.chat_template)

    # if using chat_template, data_args.eval_with_do_generation must be false
    if tokenizer.chat_template is not None:
        data_args.eval_with_do_generation = False

    if isinstance(tokenizer, LlamaTokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.dataset_name_or_path is None:
        raise ValueError(f"Please specific dataset name or path (got {data_args.dataset_name_or_path})")
    elif (
        os.path.exists(os.path.join(data_args.dataset_name_or_path, "train.json"))
        or os.path.exists(os.path.join(data_args.dataset_name_or_path, "dev.json"))
        or os.path.exists(os.path.join(data_args.dataset_name_or_path, "quant.json"))
    ):
        if training_args.do_train or quant_args.do_qat:
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
        if quant_args.do_ptq or quant_args.do_gptq:
            if os.path.exists(os.path.join(data_args.dataset_name_or_path, "quant.json")):
                ptq_ds = load_dataset(
                    "json",
                    data_files=os.path.join(data_args.dataset_name_or_path, "quant.json"),
                    lazy=data_args.lazy,
                )[0]
            elif os.path.exists(os.path.join(data_args.dataset_name_or_path, "train.json")):
                ptq_ds = load_dataset(
                    "json",
                    data_files=os.path.join(data_args.dataset_name_or_path, "train.json"),
                    lazy=data_args.lazy,
                )[0]
                logger.info(
                    f"Not found quant.json in {data_args.dataset_name_or_path}. Set train dataset as PTQ calibration dataset."
                )
            else:
                raise ValueError(
                    f"Quant strategy requires quant.json or train.json in {data_args.dataset_name_or_path}"
                )
        else:
            ptq_ds = None
    elif (
        os.path.exists(os.path.join(data_args.dataset_name_or_path, "train"))
        or os.path.exists(os.path.join(data_args.dataset_name_or_path, "dev"))
        or os.path.exists(os.path.join(data_args.dataset_name_or_path, "quant"))
    ):
        import glob

        if training_args.do_train or quant_args.do_qat:
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
        if quant_args.do_ptq or quant_args.do_gptq:
            if os.path.exists(os.path.join(data_args.dataset_name_or_path, "quant")):
                ptq_ds = load_dataset(
                    "json",
                    data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "quant", "*.json")),
                    lazy=data_args.lazy,
                )[0]
            elif os.path.exists(os.path.join(data_args.dataset_name_or_path, "train")):
                ptq_ds = load_dataset(
                    "json",
                    data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "train", "*.json")),
                    lazy=data_args.lazy,
                )[0]
                logger.info(
                    f"Not found quant.json in {data_args.dataset_name_or_path}. Set train dataset as PTQ calibration dataset."
                )
            else:
                raise ValueError(f"Quant strategy requires quant or train folder in {data_args.dataset_name_or_path}")
        else:
            ptq_ds = None
    else:
        if training_args.do_train or quant_args.do_qat:
            train_ds = load_dataset(data_args.dataset_name_or_path, splits=["train"])[0]
        else:
            train_ds = None
        if training_args.do_eval:
            dev_ds = load_dataset(data_args.dataset_name_or_path, splits=["dev"])[0]
        else:
            dev_ds = None
        if quant_args.do_ptq or quant_args.do_gptq:
            ptq_ds = load_dataset(data_args.dataset_name_or_path, splits=["train"])[0]
            logger.info("Set train dataset as PTQ calibration dataset.")
        else:
            ptq_ds = None
    # TODO(ZHUI & sijunhe): Temporary implementation. Generalize this logic and move to Trainer later.
    if training_args.resume_from_checkpoint is not None and data_args.lazy:
        logger.info(
            f"Loading from '{training_args.resume_from_checkpoint}' with `lazy=True`, manually skipping dataset and setting `ignore_data_skip` to True."
        )
        training_args.ignore_data_skip = True
        state = TrainerState.load_from_json(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json"))
        if state.trial_params is not None and "intokens_global_step" in state.trial_params:
            consumed_samples = state.trial_params["intokens_global_step"]
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
        from data import convert_example_common

        trans_func = partial(convert_example_common, tokenizer=tokenizer, data_args=data_args)
    else:
        trans_func = partial(get_convert_example(model), tokenizer=tokenizer, data_args=data_args)

    if data_args.zero_padding:
        if (
            model.base_model_prefix not in ["llama", "bloom", "chatglm", "chatglm_v2", "qwen"]
            and training_args.pipeline_parallel_degree < 1
        ):
            raise NotImplementedError(
                "Zero Padding data stream is only implemented for LLaMA, Bloom, ChatGLM and QWen so far."
            )
    train_ds = (
        train_ds.map(partial(trans_func, is_test=False, intokens=data_args.zero_padding))
        if train_ds is not None
        else None
    )
    ptq_ds = (
        ptq_ds.map(partial(trans_func, is_test=False, intokens=data_args.zero_padding)) if ptq_ds is not None else None
    )
    eval_intokens = data_args.zero_padding
    if data_args.zero_padding and data_args.eval_with_do_generation:
        logger.warning(
            "`zero_padding` conflicts with `eval_with_do_generation`. Setting zero_padding to False for the eval_dataset."
        )
        eval_intokens = False
    dev_ds = (
        dev_ds.map(partial(trans_func, is_test=data_args.eval_with_do_generation, intokens=eval_intokens))
        if dev_ds is not None
        else None
    )
    if data_args.zero_padding:
        if data_args.lazy:
            intoken_dataset = InTokensIterableDataset
        else:
            intoken_dataset = InTokensMapDataset
        logger.info("Creating Zero Padding Data Stream. This may take a few minutes.")
        train_ds = (
            intoken_dataset(
                train_ds,
                tokenizer=tokenizer,
                max_length=data_args.max_length,
            )
            if train_ds is not None
            else None
        )
        ptq_ds = (
            intoken_dataset(
                ptq_ds,
                tokenizer=tokenizer,
                max_length=data_args.max_length,
            )
            if ptq_ds is not None
            else None
        )

        if eval_intokens:
            dev_ds = (
                intoken_dataset(
                    dev_ds,
                    tokenizer=tokenizer,
                    max_length=data_args.max_length,
                )
                if dev_ds is not None
                else None
            )

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
        if model_args.lora_path is None:
            target_modules = get_lora_target_modules(model)
            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=2 * model_args.lora_rank,
                merge_weights=False,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                dtype=dtype,
                do_qat=quant_args.do_qat,
                base_model_name_or_path=model_args.model_name_or_path,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)
        model.print_trainable_parameters()

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
    max_length = (
        data_args.max_length
        if training_args.pipeline_parallel_degree > 1 or training_args.autotuner_benchmark
        else None
    )  # NOTE(gongenlei): new add autotuner_benchmark
    padding = (
        "max_length" if training_args.pipeline_parallel_degree > 1 or training_args.autotuner_benchmark else True
    )  # NOTE(gongenlei): new add autotuner_benchmark
    if training_args.pipeline_parallel_degree > 1:
        metrics = None
    elif data_args.eval_with_do_generation:
        metrics = compute_metrics_do_generation
    else:
        metrics = compute_metrics

    trainer = CausalLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            max_label_length=max_length,
            return_tensors="np",
        ),
        do_generation=data_args.eval_with_do_generation,
        callbacks=[InTokensIterDatasetCallback()] if isinstance(train_ds, InTokensIterableDataset) else None,
        gen_args=gen_args,
        data_args=data_args,
    )

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
                sum([len(i["input_ids"]) for i in trainer.train_dataset]) * training_args.num_train_epochs
            )
            effective_tokens_per_second = total_effective_tokens / train_result.metrics["train_runtime"]
            logger.info(f"Effective_Tokens_per_second: {effective_tokens_per_second} ")
            logger.info("Benchmark done.")
        else:
            if model_args.save_to_aistudio:
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

            if not training_args.autotuner_benchmark:
                trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
                trainer.log_metrics("train", train_result.metrics)
                trainer.save_metrics("train", train_result.metrics)
                trainer.save_state()

    # QAT
    if quant_args.do_qat:
        from quant import create_qat_model

        trainer.model = create_qat_model(quant_args, trainer.model, dtype)
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.log_metrics("qat", train_result.metrics)
        trainer.save_metrics("qat", train_result.metrics)
        trainer.save_state()

    # PTQ
    if quant_args.do_ptq:
        if isinstance(model, LoRAModel):
            raise NotImplementedError(
                "PTQ strategy not supported for LoRA model. Please merge lora parameters to pretrain model first."
            )
        from quant import (
            apply_autoclip,
            apply_ptq,
            apply_shift,
            apply_smooth,
            get_ptq_model_config,
        )

        trainer.model.eval()
        trainer.model.config.quantization_config.quant_type = quant_args.quant_type
        trainer.model.config.quantization_config.smooth = quant_args.smooth
        trainer.model.config.quantization_config.shift = quant_args.shift
        trainer.model.config.quantization_config.shift_smooth_all_linears = (
            quant_args.smooth_all_linears or quant_args.shift_all_linears
        )
        ptq_dataloader = trainer.get_ptq_dataloader(ptq_ds)
        if quant_args.shift or quant_args.smooth:
            ptq_model_config = get_ptq_model_config(trainer.model)

        if quant_args.shift:
            apply_shift(quant_args, trainer, ptq_dataloader, ptq_model_config)

        if quant_args.smooth:
            apply_smooth(quant_args, trainer, ptq_dataloader, ptq_model_config)

        if quant_args.auto_clip:
            apply_autoclip(quant_args, trainer, ptq_dataloader)

        apply_ptq(quant_args, trainer, ptq_dataloader)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)

    if quant_args.do_gptq:
        if isinstance(model, LoRAModel):
            raise NotImplementedError(
                "PTQ strategy not supported for LoRA model. Please merge lora parameters to pretrain model first."
            )
        from quant import apply_gptq

        ptq_dataloader = trainer.get_ptq_dataloader(ptq_ds)
        apply_gptq(quant_args, trainer, ptq_dataloader)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)

    # Evaluation dev set
    if training_args.do_eval:
        eval_result = trainer.evaluate(dev_ds)
        trainer.log_metrics("eval", eval_result)

    # Evaluation test set
    if training_args.do_predict:
        test_ds = load_dataset(
            "json",
            data_files=os.path.join(data_args.dataset_name_or_path, "test.json"),
            lazy=data_args.lazy,
        )[0]

        test_ds = test_ds.map(partial(trans_func, is_test=data_args.eval_with_do_generation))
        if eval_intokens:
            test_ds = intoken_dataset(
                test_ds,
                tokenizer=tokenizer,
                max_length=data_args.max_length,
            )
        eval_result = trainer.predict(test_ds).metrics
        trainer.log_metrics("test", eval_result)


if __name__ == "__main__":
    main()
