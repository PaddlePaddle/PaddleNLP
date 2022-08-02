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

from dataclasses import dataclass, field
import os
import paddle
from paddle.metric import Accuracy
from paddlenlp.utils.log import logger
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM, export_model
from paddlenlp.trainer import PdArgumentParser, get_scheduler
from paddlenlp.prompt import (AutoTemplate, SoftVerbalizer, MLMTokenizerWrapper,
                              PromptTuningArguments, PromptTrainer,
                              PromptModelForClassification, FewShotSampler)
from utils import load_local_dataset


@dataclass
class DataArguments:
    data_dir: str = field(
        metadata={
            "help":
            "The dataset dictionary includes train.txt, dev.txt and label.txt files."
        })
    prompt: str = field(metadata={"help": "The input prompt for tuning."})
    soft_encoder: str = field(
        default=None,
        metadata={
            "help": "The encoder type of soft template, `lstm`, `mlp` or None."
        })
    verbalizer: str = field(
        default=None, metadata={"help": "The mapping from labels to words."})
    train_sample_per_label: int = field(
        default=None,
        metadata={"help": "Number of examples sampled per label for training."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="ernie-3.0-base-zh",
        metadata={
            "help": "The build-in pretrained model or the path to local model."
        })
    export_type: str = field(
        default='paddle',
        metadata={"help": "The type to export. Support `paddle` and `onnx`."})


def main():
    # Parse the arguments.
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Define the template for preprocess and the verbalizer for postprocess.
    template = AutoTemplate.create_from(data_args.prompt,
                                        tokenizer,
                                        model=model,
                                        prompt_encoder=data_args.soft_encoder)
    logger.info("Using template: {}".format(template.template))

    label_file = os.path.join(data_args.data_dir, "label.txt")
    verbalizer = SoftVerbalizer.from_file(tokenizer, model, label_file)
    logger.info("Using verbalizer: {}".format(data_args.verbalizer))

    # Load the few-shot datasets.
    train_ds, dev_ds = load_local_dataset(data_path=data_args.data_dir,
                                          splits=["train", "dev"],
                                          label_list=verbalizer.labels_to_ids)
    if data_args.train_sample_per_label is not None:
        sampler = FewShotSampler(
            num_sample_per_label=data_args.train_sample_per_label)
        train_ds = sampler.sample_datasets(train_ds, seed=training_args.seed)

    # Define the criterion.
    criterion = paddle.nn.CrossEntropyLoss()

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForClassification(
        model,
        template,
        verbalizer,
        freeze_plm=training_args.freeze_plm,
        freeze_dropout=training_args.freeze_dropout)

    # Only update the prompt-related parameters to reduce memory cost.
    if training_args.max_steps > 0:
        num_training_steps = training_args.max_steps
    else:
        _train_batch_size = training_args.per_device_train_batch_size
        _num_train_epochs = training_args.num_train_epochs
        num_update_per_epoch = len(train_ds) // _train_batch_size
        num_update_per_epoch //= training_args.gradient_accumulation_steps
        num_update_per_epoch = max(num_update_per_epoch, 1)
        num_training_steps = num_update_per_epoch * _num_train_epochs
    if training_args.warmup_steps > 0:
        num_warmup_steps = training_args.warmup_steps
    else:
        num_warmup_steps = int(training_args.warmup_ratio * num_training_steps)

    lr_scheduler = get_scheduler(training_args.lr_scheduler_type,
                                 training_args.ppt_learning_rate,
                                 num_warmup_steps, num_training_steps)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=[{
            'params': prompt_model.verbalizer.non_head_parameters()
        }, {
            'params': [p for p in prompt_model.verbalizer.head_parameters()] +
            [p for p in prompt_model.template.parameters()],
            'learning_rate':
            training_args.ppt_learning_rate / training_args.learning_rate
        }])

    # Define the metric function.
    def compute_metrics(eval_preds):
        metric = Accuracy()
        correct = metric.compute(paddle.to_tensor(eval_preds.predictions),
                                 paddle.to_tensor(eval_preds.label_ids))
        metric.update(correct)
        acc = metric.accumulate()
        return {'accuracy': acc}

    trainer = PromptTrainer(model=prompt_model,
                            tokenizer=tokenizer,
                            args=training_args,
                            criterion=criterion,
                            train_dataset=train_ds,
                            eval_dataset=dev_ds,
                            optimizers=[optimizer, lr_scheduler],
                            compute_metrics=compute_metrics)

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_export:
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            InputSpec(shape=[None, None], dtype="float32")  # soft_token_ids
        ]
        export_path = os.path.join(training_args.output_dir, 'export')
        os.makedirs(export_path, exist_ok=True)
        export_model(prompt_model, input_spec, export_path,
                     model_args.export_type)


if __name__ == '__main__':
    main()
