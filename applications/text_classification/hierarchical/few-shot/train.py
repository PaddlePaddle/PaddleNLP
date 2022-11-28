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
import sys
from collections import defaultdict

import paddle
import paddle.nn.functional as F
from paddle.static import InputSpec
from paddlenlp.utils.log import logger
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM
from paddlenlp.trainer import PdArgumentParser, EarlyStoppingCallback
from paddlenlp.prompt import (
    AutoTemplate,
    SoftVerbalizer,
    PromptTuningArguments,
    PromptTrainer,
    PromptModelForSequenceClassification,
)

from utils import load_local_dataset

sys.path.append("../")
from metric import MetricReport


# yapf: disable
@dataclass
class DataArguments:
    data_dir: str = field(default="./data", metadata={"help": "The dataset dictionary includes train.txt, dev.txt, test.txt, label.txt and data.txt (optional) files."})
    prompt: str = field(default=None, metadata={"help": "The input prompt for tuning."})

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-3.0-base-zh", metadata={"help": "The build-in pretrained model or the path to local model."})
    export_type: str = field(default='paddle', metadata={"help": "The type to export. Support `paddle` and `onnx`."})
# yapf: enable


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
                                        training_args.max_seq_length,
                                        model=model)
    logger.info("Using template: {}".format(template.prompt))

    label_file = os.path.join(data_args.data_dir, "label.txt")
    with open(label_file, "r", encoding="utf-8") as fp:
        label_words = defaultdict(list)
        for line in fp:
            data = line.strip().split("==")
            word = data[1] if len(data) > 1 else data[0].split("##")[-1]
            label_words[data[0]].append(word)
    verbalizer = SoftVerbalizer(label_words, tokenizer, model)

    # Load the few-shot datasets.
    train_ds, dev_ds, test_ds = load_local_dataset(
        data_path=data_args.data_dir,
        splits=["train", "dev", "test"],
        label_list=verbalizer.labels_to_ids)

    # Define the criterion.
    criterion = paddle.nn.BCEWithLogitsLoss()

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForSequenceClassification(
        model,
        template,
        verbalizer,
        freeze_plm=training_args.freeze_plm,
        freeze_dropout=training_args.freeze_dropout)

    # Define the metric function.
    def compute_metrics(eval_preds):
        metric = MetricReport()
        preds = F.sigmoid(paddle.to_tensor(eval_preds.predictions))
        metric.update(preds, paddle.to_tensor(eval_preds.label_ids))
        micro_f1_score, macro_f1_score = metric.accumulate()
        return {
            "micro_f1_score": micro_f1_score,
            "macro_f1_score": macro_f1_score
        }

    # Deine the early-stopping callback.
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=4,
                              early_stopping_threshold=0.)
    ]

    # Initialize the trainer.
    trainer = PromptTrainer(model=prompt_model,
                            tokenizer=tokenizer,
                            args=training_args,
                            criterion=criterion,
                            train_dataset=train_ds,
                            eval_dataset=dev_ds,
                            callbacks=callbacks,
                            compute_metrics=compute_metrics)

    # Training.
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Prediction.
    if training_args.do_predict:
        test_ret = trainer.predict(test_ds)
        trainer.log_metrics("test", test_ret.metrics)

    # Export static model.
    if training_args.do_export:
        template = prompt_model.template
        template_keywords = template.extract_template_keywords(template.prompt)
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64"),  # input_ids,
            InputSpec(shape=[None, None], dtype="int64"),  # token_type_ids
            InputSpec(shape=[None, None], dtype="int64"),  # position_ids
            InputSpec(shape=[None, None, None, None],
                      dtype="float32")  # attention_mask
        ]
        if "mask" in template_keywords:
            input_spec.append(InputSpec(shape=[None],
                                        dtype="int64"))  # masked_positions
        if "soft" in template_keywords:
            input_spec.append(InputSpec(shape=[None, None],
                                        dtype="int64"))  # soft_token_ids
        if "encoder" in template_keywords:
            input_spec.append(InputSpec(shape=[None, None],
                                        dtype="int64"))  # encoder_ids
        export_path = os.path.join(training_args.output_dir, 'export')
        trainer.export_model(export_path,
                             input_spec=input_spec,
                             export_type=model_args.export_type)


if __name__ == '__main__':
    main()
