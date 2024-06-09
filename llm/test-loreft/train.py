import os
import json
import argparse
from tqdm import tqdm, trange
from paddlenlp.transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

from paddlenlp.data import (
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
)

from paddlenlp.trainer import TrainingArguments
from paddlenlp.trainer.trainer_utils import get_linear_schedule_with_warmup, set_seed

from transformers.trainer_utils import EvalPrediction


import datetime
import json
import math
import numpy as np


from task_config import task_config

from dataset import LoReftSupervisedDataset

from compute_metrics import compute_metrics

import paddle
import paddlenlp.reft.pareft as pareft
from paddlenlp.reft.pareft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM,
    LoreftIntervention,
    ReftDataCollator,
)

device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"

classification_tasks = {"glue"}
residual_stream_component_mapping = {
    "robertaformaskedlm": "roberta.encoder.layer[%s].output"
}
dtype_mapping = {
    "float32": paddle.float32,
    "float16": paddle.float16,
    "bfloat16": paddle.bfloat16,
    "float8": "float8",
}
intervention_mapping = {
    "LoreftIntervention": LoreftIntervention,
}


def finetune(
    act_fn: str,
    add_bias: bool,
    model: str,
    layers: str,
    rank: int,
    position: str,
    epochs: int,
    seed: int,
    intervention_type: str,
    max_n_train_example: int,
    max_n_eval_example: int,
    is_wandb: bool,
    wandb_name: str,
    gradient_accumulation_steps: int,
    batch_size: int,
    output_dir: str,
    task: str,
    lr: float,
    schedule: str,
    data_dir: str,
    train_dataset: str,
    eval_dataset: str,
    save_model: bool,
    eval_batch_size: int,
    warmup_ratio: float,
    weight_decay: float,
    dropout: float,
    test_split: str,
    train_on_inputs: bool,
    max_length: int,
    use_normalized_template: bool,
    allow_cls_grad: bool,
    metric_for_best_model: str,
    dtype: str,
    logging_steps: int,
    wandb_dir: str,
    wandb_proj: str,
    share_weights: bool,
    greedy_decoding: bool,
    temperature: float,
    top_p: float,
    top_k: float,
    args,
):
    """
    Generic Representation Finetuning.
    """
    assert task in {
        "boolq",
        "commonsense",
        "math",
        "alpaca",
        "instruct",
        "ultrafeedback",
        "glue",
        "gsm8k",
        "ultrafeedback_pair",
    }

    dtype = dtype_mapping[dtype]

    # store/log run details
    print(
        f"task: {task}, model: {model}, intervention_type: {intervention_type}, "
        f"layers: {layers}, rank: {rank}, "
        f"position: {position}, epoch: {epochs}, train_on_inputs: {train_on_inputs}, "
        f"max_length: {max_length}, allow_cls_grad: {allow_cls_grad}"
    )

    # everything is guarded by a single seed
    set_seed(seed)

    model_name = model
    model_str = model.split("/")[-1]
    train_dataset_str = train_dataset
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    if train_dataset is not None:
        run_name = f"{model_str}.{task}.{train_dataset_str}.{test_split}.{now}"
    else:
        run_name = f"{model_str}.{task}.{now}"

    print(f"run_name: {run_name}")

    # 干预的层
    if layers != "all":
        layers = [int(l) for l in layers.split(";")]
    else:
        temp_config = AutoConfig.from_pretrained(model)
        layers = [l for l in range(temp_config.num_hidden_layers)]

    # position str takes the following formats:
    # f1 -> first token; f2 -> first two tokens.
    # f1+l1 -> first and last tokens; f2+l2 -> first and last two tokens.
    # fn or ln shares the same intervention.
    if "+" in position and not share_weights:
        layers += layers

    print(f"layers: {layers}")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False

    # # paddle
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # load dataset splits
    assert task in task_config, f"Unrecognized task: {task}"
    train_datasets = (
        task_config[task]["train_datasets"]
        if train_dataset is None
        else [train_dataset]
    )
    if task == "glue":
        eval_datasets = [train_dataset]
    else:
        eval_datasets = (
            task_config[task]["eval_datasets"]
            if eval_dataset is None
            else [eval_dataset]
        )

    print(f"train_datasets: {train_datasets}")
    print(f"eval_datasets: {eval_datasets}")

    # ReftDataset = LoReftGLUEDataset if task == "glue" else LoReftSupervisedDataset
    ReftDataset = LoReftSupervisedDataset
    train_dataset = ReftDataset(
        task,
        (
            train_datasets[0]
            if task == "glue" or task == "ultrafeedback_pair"
            else (
                os.path.join(data_dir, train_datasets[0])
                if data_dir is not None
                else train_datasets[0]
            )
        ),
        tokenizer,
        data_split="train",
        seed=seed,
        max_n_example=max_n_train_example,
        **{
            "num_interventions": len(layers),
            "position": position,
            "share_weights": share_weights,
            "test_split": test_split,
        },
    )
    trigger_tokens = train_dataset.trigger_tokens
    num_labels = train_dataset.num_labels

    all_eval_datasets = {}
    for eval_dataset in eval_datasets:
        test_splits = test_split.split(";")
        all_eval_datasets[eval_dataset] = {}
        for split in test_splits:
            raw_eval = ReftDataset(
                task,
                (
                    eval_dataset
                    if task == "glue"
                    else os.path.join(data_dir, eval_dataset)
                ),
                tokenizer,
                data_split=split,
                seed=seed,
                max_n_example=max_n_eval_example,
                **{
                    "num_interventions": len(layers),
                    "position": position,
                    "share_weights": share_weights,
                },
            )
            all_eval_datasets[eval_dataset][split] = [raw_eval, raw_eval.raw_dataset]
    eval_datasets = all_eval_datasets


    # load model based on task type.
    if task in classification_tasks:
        config = AutoConfig.from_pretrained(
            model,
            num_labels=num_labels,
            finetuning_task=train_dataset_str,
            # load_in_8bit=True if dtype == "float8" else False,
            # device_map=device,
        )
        # full precision loading since usually for small models
        model = AutoModelForSequenceClassification.from_pretrained(
            model,
            config=config,  # just providing the label
            torch_dtype=dtype if dtype != "float8" else None,
            load_in_8bit=True if dtype == "float8" else False,
            device_map=device,
        )
    else:
        print(
            f"dtype is:",
            {dtype},
        )
        model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=dtype if dtype != "float8" else None,  # save memory
            # load_in_8bit=True if dtype == "float8" else False,
            # device_map=device,
        )
        config = model.config
    if need_resize:
        print("resizing token embeddings ...", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    intervention_type = intervention_mapping[intervention_type]

    # select collator based on the type
    if task in classification_tasks:
        data_collator_fn = DataCollatorWithPadding(
            tokenizer=tokenizer, padding="longest"
        )
    else:
        data_collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
        )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)

    # intervention config based on model type
    intervention_dtype = paddle.bfloat16 if isinstance(dtype, str) else dtype
    model_arch = model.config.architectures[0].lower()
    if model_arch in residual_stream_component_mapping:
        representations = [
            {
                "component": residual_stream_component_mapping[model_arch] % l,
                "low_rank_dimension": rank,
                "intervention": intervention_type(
                    embed_dim=config.hidden_size,
                    low_rank_dimension=rank,
                    dropout=dropout,
                    dtype=intervention_dtype,
                    act_fn=act_fn,
                    device=device,
                    add_bias=add_bias,
                ),
            }
            for l in layers
        ]
        task_type = TaskType.SEQ_CLS
    else:
        representations = [
            {
                "layer": l,
                "component": "block_output",
                "low_rank_dimension": rank,
                "intervention": intervention_type(
                    embed_dim=config.hidden_size,
                    low_rank_dimension=rank,
                    dropout=dropout,
                    dtype=intervention_dtype,
                    act_fn=act_fn,
                    device=device,
                    add_bias=add_bias,
                ),
            }
            for l in layers
        ]
        task_type = TaskType.CAUSAL_LM

    reft_config = ReftConfig(representations=representations)
    reft_model = get_reft_model(
        model, reft_config, set_device=not isinstance(dtype, str)
    )
    reft_model.print_trainable_parameters()

    # for GLUE tasks, we enable gradients on the classifier head.
    # the parameter will be counted as well.
    if task == "glue" and allow_cls_grad:
        for param in reft_model.model.classifier.parameters():
            # reft_model with HF trainer will automatically pick up these params to optimize
            param.requires_grad = True

    # train enables dropout but no grads.
    # this line might not be necessary since HF trainer enables this by default.
    reft_model.model.train()
    n_params = reft_model.count_parameters(include_model=False)



    # # training args
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{run_name}",
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch" if task == "glue" else "no",
        save_strategy="epoch" if task == "glue" else "no",
        metric_for_best_model=metric_for_best_model if task == "glue" else None,
        load_best_model_at_end=True if task == "glue" else False,
        logging_strategy="steps",
        save_total_limit=1,  
        logging_steps=logging_steps,
        lr_scheduler_type=schedule,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        optim="adamw",
        weight_decay=weight_decay,
        # use_cpu=False if device == "cuda" else True,
        # use_cpu=False,
        seed=seed,
        # until HF supports ReFT, this remains False! :)
        remove_unused_columns=False,
    )


    trainer_class = ReftTrainerForCausalLM
    trainer = trainer_class(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
    )

    trainer.train()

    print("train end and start eval")

    # dump config
    args_dict = vars(args)
    args_dict["n_params"] = int(n_params)
    json_file_name = f"{output_dir}/{run_name}/args.json"
    with open(json_file_name, "w") as json_file:
        json.dump(args_dict, json_file, indent=4)

    # save model
    if save_model:
        reft_model.save(f"{output_dir}/{run_name}")

    # ensure everything is in eval mode
    reft_model.model.eval()
    for k, v in reft_model.interventions.items():
        _ = v[0].eval()

    print({"n_params": n_params})
    # do eval
    eval_results = {}
    for dataset_name in eval_datasets:
        # split evalset into chunks
        print(f"Evaluating on {dataset_name}")
        for split, (eval_dataset, data_items) in eval_datasets[dataset_name].items():

            generations, stats = compute_metrics(
                task,
                dataset_name,
                reft_model,
                tokenizer,
                eval_dataset,
                data_items,
                trigger_tokens,
                run_name,
                eval_batch_size,
                data_collator if task in classification_tasks else None,
                split,
                greedy_decoding,
                temperature,
                top_p,
                top_k,
            )

            # log
            eval_results.update(stats)
            # if is_wandb:
            #     wandb.log(stats)
            generations = stats if generations is None else generations
            result_json_file_name = (
                f"{output_dir}/{run_name}/{dataset_name}_{split}_outputs.json"
            )
            with open(result_json_file_name, "w") as json_file:
                json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{output_dir}/{run_name}/eval_results.json"
    eval_results["n_params"] = int(n_params)
    with open(result_json_file_name, "w") as json_file:
        json.dump(eval_results, json_file, indent=4)

    print(f"Training results can be found in {output_dir}/{run_name}")


def main():
    parser = argparse.ArgumentParser(
        description="A simple script that takes different arguments."
    )

    parser.add_argument("-task", "--task", type=str, default="commonsense")
    parser.add_argument(
        "-data_dir",
        "--data_dir",
        type=str,
        default="/home/ldn/baidu/pyreft/paddle-version/loreft/datasets",
    )
    parser.add_argument("-train_dataset", "--train_dataset", type=str, default=None)
    parser.add_argument("-eval_dataset", "--eval_dataset", type=str, default=None)
    parser.add_argument(
        "-model",
        "--model",
        type=str,
        help="meta-llama/Llama-2-7b",
        default="/home/ldn/.paddlenlp/models/meta-llama/Llama-2-7b",
    )
    parser.add_argument("-seed", "--seed", type=int, help="42", default=42)
    parser.add_argument("-l", "--layers", type=str, help="2;10;18;26", default="all")
    parser.add_argument("-r", "--rank", type=int, help=8, default=8)
    parser.add_argument("-p", "--position", type=str, help="f1+l1", default="f7+l7")
    parser.add_argument("-e", "--epochs", type=int, help="1", default=1)
    parser.add_argument("-is_wandb", "--is_wandb", action="store_true")
    parser.add_argument("-wandb_name", "--wandb_name", type=str, default="reft")
    parser.add_argument(
        "-save_model", "--save_model", action="store_true", default=True
    )
    parser.add_argument(
        "-max_n_train_example", "--max_n_train_example", type=int, default=100
    )
    parser.add_argument(
        "-max_n_eval_example", "--max_n_eval_example", type=int, default=20
    )
    parser.add_argument(
        "-type",
        "--intervention_type",
        type=str,
        help="LoreftIntervention",
        default="LoreftIntervention",
    )
    parser.add_argument(
        "-gradient_accumulation_steps",
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument("-batch_size", "--batch_size", type=int, default=8)
    parser.add_argument("-eval_batch_size", "--eval_batch_size", type=int, default=4)
    parser.add_argument(
        "-output_dir", "--output_dir", type=str, default="./commonsense-results"
    )
    parser.add_argument("-lr", "--lr", type=float, default=1e-4)
    parser.add_argument("-schedule", "--schedule", type=str, default="linear")
    parser.add_argument("-wu", "--warmup_ratio", type=float, default=0.01)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.00)
    parser.add_argument("-dropout", "--dropout", type=float, default=0.00)
    parser.add_argument("-act_fn", "--act_fn", type=str, default=None)
    parser.add_argument("-add_bias", "--add_bias", action="store_true")
    parser.add_argument("-test_split", "--test_split", type=str, default="test")
    parser.add_argument("-train_on_inputs", "--train_on_inputs", action="store_true")
    parser.add_argument("-max_length", "--max_length", type=int, help=512, default=512)
    parser.add_argument(
        "-nt", "--use_normalized_template", action="store_true", default=True
    )
    parser.add_argument("-allow_cls_grad", "--allow_cls_grad", action="store_true")
    parser.add_argument(
        "-metric_for_best_model",
        "--metric_for_best_model",
        type=str,
        default="accuracy",
    )
    parser.add_argument(
        "-dtype",
        "--dtype",
        type=str,
        default="bfloat16" if device == "gpu" else "float32",
    )
    parser.add_argument(
        "-logging_steps", "--logging_steps", type=int, help=1, default=1
    )
    parser.add_argument("-wandb_dir", "--wandb_dir", type=str, default="wandb")
    parser.add_argument("-wandb_proj", "--wandb_proj", type=str, default="MyReFT")
    parser.add_argument("-sw", "--share_weights", action="store_true", default=True)
    parser.add_argument("-gd", "--greedy_decoding", action="store_true", default=True)

    # decoding params
    parser.add_argument("-t", "--temperature", type=float, default=None)
    parser.add_argument("-top_p", "--top_p", type=float, default=None)
    parser.add_argument("-top_k", "--top_k", type=float, default=None)

    args = parser.parse_args()

    finetune(**vars(args), args=args)


if __name__ == "__main__":
    main()
