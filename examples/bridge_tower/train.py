# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import functools
import json
import os
from dataclasses import dataclass, field

import paddle
from paddle.metric import Accuracy
from src.modules import BTTransformer
from src.modules.meter_utils import set_schedule
from trainer_util import BridgeTowerPreTrainTrainer, BridgeTowerTrainer, MyCallback
from utils import collate_fn, get_dataset, get_pretrained_dataset

from paddlenlp.data import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from paddlenlp.trainer import PdArgumentParser, TrainingArguments

os.environ["NCCL_DEBUG"] = "INFO"


@dataclass
class DataArguments:
    test_only: bool = field(default=False, metadata={"help": "Whether to evaluate model on public test datasets."})
    data_root: str = field(
        default="/root/paddlejob/workspace/env_run/output/dataset/fine-tune",
        metadata={"help": "Whether to evaluate model on public test datasets."},
    )


@dataclass
class ModelArguments:
    export_type: str = field(default="paddle", metadata={"help": "The type to export. Support `paddle` and `onnx`."})
    dropout: float = field(default=0.1, metadata={"help": "The dropout used for pretrained model."})
    lr_end: float = field(default=0.1, metadata={"help": "The dropout used for pretrained model."})
    power: int = field(default=1, metadata={"help": "Save checkpoint every X updates steps."})
    config_name: str = field(default="configs/config.json", metadata={"help": "training config"})
    checkpoint_path: str = field(default="", metadata={"help": "checkpoint path"})
    batch_size: int = field(default=4096, metadata={"help": "Total batch size for training."})
    num_nodes: int = field(default=1, metadata={"help": "The number of nodes for training."})
    num_gpus: int = field(default=8, metadata={"help": "The number of gpus per device."})


def get_config(file_path):
    with open(file_path) as f:
        result = json.load(f)
    return result


def main():

    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    _config = get_config(model_args.config_name)
    _config["data_root"] = data_args.data_root
    _config["batch_size"] = model_args.batch_size
    _config["per_gpu_batchsize"] = training_args.per_device_train_batch_size
    _config["num_nodes"] = model_args.num_nodes
    _config["num_gpus"] = model_args.num_gpus
    _config["load_path"] = model_args.checkpoint_path

    model = BTTransformer(_config)

    optimizer_grouped_parameters = set_schedule(model, _config)

    raw_state_dict = model.state_dict()
    if os.path.exists(model_args.checkpoint_path):
        state_dict = paddle.load(model_args.checkpoint_path)
        with open("converted_torch_param.txt", "w") as f:
            for k, v in state_dict.items():
                f.write(k + "\n")
        f_loaded = open("completed_keys.txt", "w")
        with open("imcomplete_keys.txt", "w") as f:
            for k, v in raw_state_dict.items():
                if k not in state_dict:
                    f.write(k + "\n")
                else:
                    f_loaded.write(k + "\n")
        f_loaded.close()
        model.load_dict(state_dict)
        print("Loading parameters from {}".format(model_args.checkpoint_path))

    if _config["group_name"] == "mlm_itm":
        tokenizer, train_dataset, eval_dataset = get_pretrained_dataset(_config)
    else:
        tokenizer, train_dataset, eval_dataset = get_dataset(_config)
    collator = DataCollatorForWholeWordMask if _config["whole_word_masking"] else DataCollatorForLanguageModeling

    mlm_collator = collator(tokenizer=tokenizer, mlm=True, mlm_probability=_config["mlm_prob"])
    my_collate = functools.partial(
        collate_fn,
        mlm_collator=mlm_collator,
    )
    grad_steps = max(
        _config["batch_size"] // (_config["per_gpu_batchsize"] * _config["num_gpus"] * _config["num_nodes"]), 1
    )
    training_args.gradient_accumulation_steps = grad_steps
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    # Define the metrics of tasks.
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        preds = paddle.to_tensor(preds)
        label = paddle.to_tensor(p.label_ids)

        metric = Accuracy()
        metric.reset()
        result = metric.compute(preds, label)
        metric.update(result)
        accu = metric.accumulate()
        metric.reset()
        return {"accuracy": accu}

    if _config["group_name"] == "mlm_itm":
        trainer = BridgeTowerPreTrainTrainer(
            model=model,
            criterion=None,
            args=training_args,
            data_collator=my_collate,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[MyCallback],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = BridgeTowerTrainer(
            model=model,
            criterion=None,
            args=training_args,
            data_collator=my_collate,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[MyCallback],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    trainer.set_optimizer_grouped_parameters(optimizer_grouped_parameters)
    if data_args.test_only:
        train_result = trainer.evaluate()
        print(train_result)
    else:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()
