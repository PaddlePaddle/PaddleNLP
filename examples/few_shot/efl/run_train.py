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

import os
import time
from dataclasses import dataclass, field
from functools import partial

import paddle
from data import load_fewclue_dataset
from paddle.metric import Accuracy
from paddle.static import InputSpec
from utils import load_prompt_arguments, save_fewclue_prediction, save_pseudo_data

from paddlenlp.prompt import (
    ManualTemplate,
    ManualVerbalizer,
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
)
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger


# yapf: disable
@dataclass
class DataArguments:
    task_name: str = field(default="eprstmt", metadata={"help": "The task name in FewCLUE."})
    split_id: str = field(default="0", metadata={"help": "The split id of datasets, including 0, 1, 2, 3, 4, few_all."})
    prompt_path: str = field(default="prompt/eprstmt.json", metadata={"help": "Path to the defined prompts."})
    prompt_index: int = field(default=0, metadata={"help": "The index of defined prompt for training."})
    pseudo_data_path: str = field(default=None, metadata={"help": "Path to data with pseudo labels."})
    do_label: bool = field(default=False, metadata={"help": "Whether to label unsupervised data in unlabeled datasets"})
    do_test: bool = field(default=False, metadata={"help": "Whether to evaluate model on public test datasets."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-1.0-large-zh-cw", metadata={"help": "Build-in pretrained model name or the path to local model."})
    export_type: str = field(default='paddle', metadata={"help": "The type to export. Support `paddle` and `onnx`."})
    dropout: float = field(default=0.1, metadata={"help": "The dropout used for pretrained model."})
# yapf: enable


def main():
    # Parse the arguments.
    parser = PdArgumentParser((ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args = load_prompt_arguments(data_args)
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2,
        hidden_dropout_prob=model_args.dropout,
        attention_probs_dropout_prob=model_args.dropout,
    )

    # Define template for preprocess and verbalizer for postprocess.
    template = ManualTemplate(data_args.prompt, tokenizer, training_args.max_seq_length)
    logger.info("Using template: {}".format(template.prompt))

    verbalizer = ManualVerbalizer(data_args.label_words, tokenizer)
    ids_to_labels = {idx: label for idx, label in enumerate(verbalizer.labels)}
    logger.info("Using verbalizer: {}".format(data_args.label_words))

    # Load datasets.
    train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds = load_fewclue_dataset(data_args, verbalizer=verbalizer)

    # Define the criterion.
    criterion = paddle.nn.CrossEntropyLoss()

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForSequenceClassification(
        model, template, None, freeze_plm=training_args.freeze_plm, freeze_dropout=training_args.freeze_dropout
    )

    # Define the metric function.
    def compute_metrics(eval_preds, num_labels):
        metric = Accuracy()
        preds = paddle.to_tensor(eval_preds.predictions)
        preds = paddle.nn.functional.softmax(preds, axis=1)[:, 1]
        preds = preds.reshape([-1, num_labels])
        labels = paddle.to_tensor(eval_preds.label_ids)
        labels = paddle.argmax(labels.reshape([-1, num_labels]), axis=1)
        correct = metric.compute(preds, labels)
        metric.update(correct)
        acc = metric.accumulate()
        return {"accuracy": acc}

    # Initialize the trainer.
    compute_metrics = partial(compute_metrics, num_labels=len(verbalizer.labels))
    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=None,
        compute_metrics=compute_metrics,
    )

    # Traininig.
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    time_stamp = time.strftime("%m%d-%H-%M-%S", time.localtime())

    # Test.
    if data_args.do_test and public_test_ds is not None:
        test_ret = trainer.predict(public_test_ds)
        trainer.log_metrics("test", test_ret.metrics)

    # Predict.
    if training_args.do_predict and test_ds is not None:
        pred_ret = trainer.predict(test_ds)
        logger.info("Prediction done.")
        predict_path = os.path.join(training_args.output_dir, "fewclue_submit_examples_" + time_stamp)
        save_fewclue_prediction(predict_path, data_args.task_name, pred_ret, verbalizer, ids_to_labels)

    # Label unsupervised data.
    if data_args.do_label and unlabeled_ds is not None:
        label_ret = trainer.predict(unlabeled_ds)
        logger.info("Labeling done.")
        pseudo_path = os.path.join(training_args.output_dir, "pseudo_data_" + time_stamp + ".txt")
        save_pseudo_data(pseudo_path, data_args.task_name, label_ret, verbalizer, ids_to_labels)

    # Export static model.
    if training_args.do_export:
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64"),  # input_ids,
            InputSpec(shape=[None, None], dtype="int64"),  # token_type_ids
            InputSpec(shape=[None, None], dtype="int64"),  # position_ids
            InputSpec(shape=[None, None, None, None], dtype="float32"),  # attention_mask
        ]
        export_path = os.path.join(training_args.output_dir, "export")
        trainer.export_model(export_path, input_spec=input_spec, export_type=model_args.export_type)


if __name__ == "__main__":
    main()
