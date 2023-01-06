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

import paddle
from paddle.metric import Accuracy
from paddle.static import InputSpec
from utils import MetricReport, UTCLoss, read_local_dataset

from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import (
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
    UTCTemplate,
)
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import UTC, AutoTokenizer, export_model


@dataclass
class DataArguments:
    dataset_path: str = field(
        metadata={"help": "Local dataset directory including train.txt, dev.txt and label.txt (optional)."}
    )
    train_file: str = field(default="train.txt", metadata={"help": "Train dataset file name."})
    dev_file: str = field(default="dev.txt", metadata={"help": "Dev dataset file name."})
    threshold: float = field(default=0.5, metadata={"help": "The threshold to produce predictions."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="utc-large", metadata={"help": "The build-in pretrained UTC model name or path to its checkpoints."}
    )
    export_type: str = field(default="paddle", metadata={"help": "The type to export. Support `paddle` and `onnx`."})
    export_model_dir: str = field(default="checkpoints/model_best", metadata={"help": "The export model path."})


def main():
    # Parse the arguments.
    parser = PdArgumentParser((ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = UTC.from_pretrained(model_args.model_name_or_path)

    # Define template for preprocess and verbalizer for postprocess.
    template = UTCTemplate(tokenizer, training_args.max_seq_length)

    # Load and preprocess dataset.
    train_ds = load_dataset(
        read_local_dataset,
        data_path=data_args.dataset_path,
        data_file=data_args.train_file,
        lazy=False,
    )
    dev_ds = load_dataset(
        read_local_dataset,
        data_path=data_args.dataset_path,
        data_file=data_args.dev_file,
        lazy=False,
    )

    # Define the criterion.
    criterion = UTCLoss()

    # Initialize the prompt model.
    prompt_model = PromptModelForSequenceClassification(
        model, template, None, freeze_plm=training_args.freeze_plm, freeze_dropout=training_args.freeze_dropout
    )

    # Define the metric function.
    def compute_metrics(eval_preds):
        labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
        preds = paddle.to_tensor(eval_preds.predictions)

        metric_f1 = MetricReport(threshold=data_args.threshold)
        preds_f1 = paddle.nn.functional.sigmoid(preds)
        preds_f1 = preds_f1[labels != -100]
        labels_f1 = labels[labels != -100]
        metric_f1.update(preds_f1, labels_f1)
        micro_f1, macro_f1 = metric_f1.accumulate()

        metric_acc = Accuracy()
        labels_acc = paddle.argmax(labels, axis=1)
        correct = metric_acc.compute(preds, labels_acc)
        metric_acc.update(correct)
        acc = metric_acc.accumulate()
        return {"acc": acc, "micro_f1": micro_f1, "macro_f1": macro_f1}

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

    # Training.
    if training_args.do_train:
        train_results = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_results.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Export.
    if training_args.do_export:
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            InputSpec(shape=[None, None, None, None], dtype="float32", name="attention_mask"),
            InputSpec(shape=[None, None], dtype="int64", name="omask_positions"),
            InputSpec(shape=[None], dtype="int64", name="cls_positions"),
        ]
        export_model(trainer.pretrained_model, input_spec, model_args.export_model_dir, model_args.export_type)


if __name__ == "__main__":
    main()
