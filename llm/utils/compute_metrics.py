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

import re

import paddle

# from torch.utils.data import DataLoader
from paddle.io import DataLoader, Dataset
from tqdm import tqdm

import paddlenlp.reft.pavenv as pv
from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.reft.pareft import ReftDataCollator
from paddlenlp.transformers import AutoTokenizer

# from datasets import Dataset


device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"


# from templates import *

task_config = {
    "commonsense": {
        "train_datasets": ["commonsense_170k"],
        "eval_datasets": [
            "boolq",
            "piqa",
            "social_i_qa",
            "hellaswag",
            "winogrande",
            "ARC-Easy",
            "ARC-Challenge",
            "openbookqa",
        ],
        "task_prompt_template": "%s\n",
        "trigger_tokens": "the correct answer is ",
        "generation_args": {
            # align with https://github.com/AGI-Edgerunners/LLM-Adapters
            True: {
                "max_new_tokens": 32,
                "do_sample": False,
            },
            False: {
                "max_new_tokens": 32,
                "temperature": 0.1,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            },
        },
    }
}


def is_float(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def extract_answer_number(sentence: str) -> float:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py
    """
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float("inf")
            print("ValueError occurred:", e)
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py

    Note that it becomes ambiguous whether to extract the
    first letter or the last letter. Either way may lead
    to inaccurately assess the model performance.

    We choose to follow the LLM-Adaptor repo, but leave this note
    for future research to explore the impact of this.
    """
    sentence_ = sentence.strip()
    pred_answers = re.findall(r"A|B|C|D|E", sentence_)
    if pred_answers:
        if not pred_answers:
            return ""
        return pred_answers[0]
    else:
        return ""


def extract_output(pred, trigger=""):
    if not trigger:
        return pred
    # for causallm only, use special trigger to detect new tokens.
    # if cannot find trigger --> generation is too long; default to empty generation
    start = pred.find(trigger)
    if start < 0:
        return ""
    output = pred[start + len(trigger) :].lstrip()  # left strip any whitespaces
    return output


def make_data_collator(tokenizer, model) -> ReftDataCollator:
    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        max_length=2048,
    )
    return ReftDataCollator(data_collator=data_collator_fn)


def make_dataloader(
    dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool
) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)


def compute_metrics_reft(
    task: str,
    dataset_name: str,
    intervenable: pv.IntervenableModel,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    data_items: list,
    trigger_tokens: str,
    run_name: str,
    batch_size: int = 4,
    data_collator=None,
    split=None,
    greedy_decoding=False,
    temperature=None,
    top_p=None,
    top_k=None,
):
    # switch the tokenizer mode first for generation tasks
    if task != "glue":
        tokenizer.padding_side = "left"  # switch padding side for collator
        num_beams = 4 if task in ["commonsense", "math"] and not greedy_decoding else 1

    data_collator = data_collator if data_collator is not None else make_data_collator(tokenizer, intervenable.model)
    eval_dataloader = make_dataloader(eval_dataset, batch_size, data_collator, shuffle=False)
    correct_count = 0
    total_count = 0
    generations = []
    eval_iterator = tqdm(eval_dataloader, position=0, leave=True)
    # all_preds = []
    # all_labels = []

    if "Meta-Llama-3-8B-Instruct" in tokenizer.name_or_path:  # pretty bad workaround for llama-3, forgive me
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        trigger_tokens = "assistant\n\n"

    with paddle.no_grad():
        for step, inputs in enumerate(eval_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, paddle.Tensor):
                    inputs[k] = v.to(device)

            # [layers, batch_size, positions]
            # intervention_locations = inputs["intervention_locations"].permute(1, 0, 2)
            intervention_locations = paddle.transpose(inputs["intervention_locations"], perm=[1, 0, 2])

            # get left padding count, [batch_size], and add to locations
            left_padding = (inputs["input_ids"] == tokenizer.bos_token_id).nonzero(as_tuple=True)[1]
            if left_padding.numel() > 0:
                left_padding = left_padding.reshape([1, -1, 1]).to(device)  # [1, batch_size, 1]
                intervention_locations += left_padding
                intervention_locations -= 1  # offset for the sink padding
            else:
                print("Warning: No BOS token found, skipping left padding adjustment.")

            # repeat each batch by num_beams times in intervention locations
            # -> [layers, batch_size * num_beams, positions]
            intervention_locations = intervention_locations.repeat_interleave(num_beams, axis=1).tolist()

            # set generation args depending on task
            generation_args = {
                "base": {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
                "unit_locations": {"sources->base": (None, intervention_locations)},
                "intervene_on_prompt": True,
                "eos_token_id": tokenizer.eos_token_id,
                "early_stopping": True,
            }
            if "generation_args" in task_config[task]:
                generation_args.update(task_config[task]["generation_args"][greedy_decoding])
            if "Meta-Llama-3-8B-Instruct" in tokenizer.name_or_path:  # pretty bad workaround for llama-3, forgive me
                generation_args["eos_token_id"] = terminators

            # override generation args if necessary
            if temperature is not None:
                generation_args["temperature"] = temperature
            if top_p is not None:
                generation_args["top_p"] = top_p
            if top_k is not None:
                generation_args["top_k"] = top_k

            # generate with intervention on prompt
            _, steered_response = intervenable.generate(**generation_args)

            # detokenize in batch

            actual_preds = tokenizer.batch_decode(steered_response[0], skip_special_tokens=True)
            # print("actual preds: ", actual_preds)

            for id, pred in zip(inputs["id"].tolist(), actual_preds):
                example = data_items[id]
                try:
                    raw_generation = extract_output(pred, trigger_tokens)
                except:
                    print("get not split based on trigger tokens: ", raw_generation)
                    raw_generation = "WRONG"

                # check if generation is correct
                if task == "commonsense":
                    answer = example["answer"]
                    generation = raw_generation[:]
                    if generation.strip() == answer.strip():
                        correct_count += 1
                elif task == "math":
                    answer = example["answer"]
                    answer = answer.strip()
                    if not is_float(answer):  # assuming this is from AQuA:
                        generation = extract_answer_letter(raw_generation)
                        if generation.strip() == answer.strip():
                            correct_count += 1
                    else:
                        generation = extract_answer_number(raw_generation)
                        if abs(float(answer) - generation) <= 0.001:
                            correct_count += 1
                elif task == "gsm8k":
                    answer = example["answer"].split("####")[-1].strip()
                    generation = extract_answer_number(raw_generation)
                    if abs(float(extract_answer_number(answer)) - generation) <= 0.001:
                        correct_count += 1

                # log
                total_count += 1
                if task not in [
                    "alpaca",
                    "instruct",
                    "ultrafeedback",
                    "ultrafeedback_pair",
                ]:
                    metric_str = round(correct_count / total_count, 3)
                    eval_iterator.set_postfix({"em": metric_str})
                    instruction = example["question"] if task == "gsm8k" else example["instruction"]
                    generations += [
                        {
                            "instruction": instruction,
                            "raw_generation": raw_generation,
                            "generation": generation,
                            "answer": answer,
                        }
                    ]
                else:
                    generations += [
                        {
                            "instruction": example["instruction"],
                            "output": raw_generation,
                            "dataset": dataset_name,
                            "generator": run_name,
                        }
                    ]

    # compute metrics
    if task in ["alpaca", "instruct", "ultrafeedback", "ultrafeedback_pair"]:
        return generations, {}
    else:
        return generations, {f"eval/{dataset_name}": correct_count / total_count}
