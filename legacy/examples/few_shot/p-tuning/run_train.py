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
    MaskedLMVerbalizer,
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
    SoftTemplate,
)
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import AutoModelForMaskedLM, AutoTokenizer
from paddlenlp.utils.log import logger


# yapf: disable
@dataclass
class DataArguments:
    task_name: str = field(default="eprstmt", metadata={"help": "The task name in FewCLUE."})
    split_id: str = field(default="0", metadata={"help": "The split id of datasets, including 0, 1, 2, 3, 4, few_all."})
    prompt_path: str = field(default="prompt/eprstmt.json", metadata={"help": "Path to the defined prompts."})
    prompt_index: int = field(default=0, metadata={"help": "The index of defined prompt for training."})
    augment_type: str = field(default=None, metadata={"help": "The strategy used for data augmentation, including `swap`, `delete`, `insert`, `subsitute`."})
    num_augment: str = field(default=5, metadata={"help": "Number of augmented data per example, which works when `augment_type` is set."})
    word_augment_percent: str = field(default=0.1, metadata={"help": "Percentage of augmented words in sequences, used for `swap`, `delete`, `insert`, `subsitute`."})
    augment_method: str = field(default="mlm", metadata={"help": "Strategy used for `insert` and `subsitute`."})
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
    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        hidden_dropout_prob=model_args.dropout,
        attention_probs_dropout_prob=model_args.dropout,
    )

    # Define template for preprocess and verbalizer for postprocess.
    template = SoftTemplate(data_args.prompt, tokenizer, training_args.max_seq_length, model.get_input_embeddings())
    logger.info("Using template: {}".format(template.prompt))

    verbalizer = MaskedLMVerbalizer(data_args.label_words, tokenizer)
    labels_to_ids = verbalizer.labels_to_ids
    ids_to_labels = {idx: label for label, idx in labels_to_ids.items()}
    logger.info("Using verbalizer: {}".format(data_args.label_words))

    # Load datasets.
    data_ds, label_list = load_fewclue_dataset(data_args, verbalizer=verbalizer, example_keys=template.example_keys)
    train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds = data_ds
    dev_labels, test_labels = label_list

    # Define the criterion.
    criterion = paddle.nn.CrossEntropyLoss()

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForSequenceClassification(
        model, template, verbalizer, freeze_plm=training_args.freeze_plm, freeze_dropout=training_args.freeze_dropout
    )

    # Define the metric function.
    def compute_metrics(eval_preds, labels, verbalizer):
        metric = Accuracy()
        predictions = paddle.to_tensor(eval_preds.predictions)
        predictions = verbalizer.aggregate_multiple_mask(predictions)
        correct = metric.compute(predictions, paddle.to_tensor(labels))
        metric.update(correct)
        acc = metric.accumulate()
        return {"accuracy": acc}

    # Initialize the trainer.
    dev_compute_metrics = partial(compute_metrics, labels=dev_labels, verbalizer=verbalizer)
    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=None,
        compute_metrics=dev_compute_metrics,
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
        test_compute_metrics = partial(compute_metrics, labels=test_labels, verbalizer=verbalizer)
        trainer.compute_metrics = test_compute_metrics
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
        template = prompt_model.template
        template_keywords = template.extract_template_keywords(template.prompt)
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64"),  # input_ids,
            InputSpec(shape=[None, None], dtype="int64"),  # token_type_ids
            InputSpec(shape=[None, None], dtype="int64"),  # position_ids
            InputSpec(shape=[None, None, None, None], dtype="float32"),  # attention_mask
            InputSpec(shape=[None], dtype="int64"),  # masked_positions
            InputSpec(shape=[None, None], dtype="int64"),  # soft_token_ids
        ]
        if "encoder" in template_keywords:
            input_spec.append(InputSpec(shape=[None, None], dtype="int64"))  # encoder_ids
        export_path = os.path.join(training_args.output_dir, "export")
        trainer.export_model(export_path, input_spec=input_spec, export_type=model_args.export_type)


if __name__ == "__main__":
    main()
