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
from dataclasses import dataclass, field

import paddle
from paddle.metric import Accuracy
from src.modules import BTTransformer
from src.modules.meter_utils import set_schedule
from trainer_util import BridgeTowerPreTrainTrainer, BridgeTowerTrainer
from utils import collate_fn, get_dataset, get_pretrained_dataset

from paddlenlp.data import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from paddlenlp.trainer import PdArgumentParser, TrainingArguments


@dataclass
class DataArguments:
    test_only: bool = field(default=False, metadata={"help": "Whether to evaluate model on public test datasets."})


@dataclass
class ModelArguments:
    export_type: str = field(default="paddle", metadata={"help": "The type to export. Support `paddle` and `onnx`."})
    dropout: float = field(default=0.1, metadata={"help": "The dropout used for pretrained model."})
    lr_end: float = field(default=0.1, metadata={"help": "The dropout used for pretrained model."})
    power: int = field(default=1, metadata={"help": "Save checkpoint every X updates steps."})
    config_name: str = field(default="configs/config.json", metadata={"help": "training config"})


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

    _config = get_config(model_args.config_name)

    _config["num_workers"] = 0
    # _config["batch_size"]=2

    # _config['num_gpus']=1
    # _config['num_workers']=1
    # _config['per_gpu_batchsize']=8
    # _config['per_gpu_eval_batchsize']=8

    model = BTTransformer(_config)

    optimizer_grouped_parameters = set_schedule(model, _config)

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
    # print(training_args)
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    # Define the metrics of tasks.
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        preds = paddle.to_tensor(preds)
        label = paddle.to_tensor(p.label_ids)

        # probs = F.softmax(preds, axis=-1)
        metric = Accuracy()
        metric.reset()
        result = metric.compute(preds, label)
        metric.update(result)
        accu = metric.accumulate()
        metric.reset()
        return {"accuracy": accu}

    # criterion = nn.BCEWithLogitsLoss()
    if _config["group_name"] == "mlm_itm":
        trainer = BridgeTowerPreTrainTrainer(
            model=model,
            criterion=None,
            args=training_args,
            data_collator=my_collate,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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
