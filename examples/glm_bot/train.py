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
import os

from utils import GLMTrainer, read_data

from paddlenlp.data import DefaultDataCollator
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import Rouge1, Rouge2, RougeL
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from paddlenlp.transformers import AutoModelForConditionalGeneration, AutoTokenizer


def main():
    parser = PdArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()[0]
    training_args.print_config()

    filenames = os.listdir("/ssd2/hesijun/workspace/paddle_bot/answer")
    train_files, dev_files = [], []
    for filename in filenames:
        filenum = int(filename.split(".")[0])
        if filenum < 10:
            train_files.append(filename)
        elif filenum >= 10 and filenum < 12:
            dev_files.append(filename)

    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-large-chinese")
    model = AutoModelForConditionalGeneration.from_pretrained(
        "THUDM/glm-large-chinese", output_predict=True, parallel_output=True, load_state_as_np=True
    )

    def preprocess(example):
        input_text = "问题：" + example["source"] + " 回答： [gMASK]"
        result = tokenizer(text=input_text, max_length=32, truncation=True, padding="max_length", return_tensors="pd")
        inputs = tokenizer.build_inputs_for_generation(
            result, targets=example["target"], max_gen_length=256, padding=True
        )
        for input_name in inputs.keys():
            inputs[input_name] = inputs[input_name].squeeze(0)
        return inputs

    train_ds = load_dataset(read_data, filenames=train_files, lazy=False)
    dev_ds = load_dataset(read_data, filenames=dev_files, lazy=False)
    processed_train = train_ds.map(preprocess, lazy=False)
    processed_dev = dev_ds.map(preprocess, lazy=False)

    collate_fn = DefaultDataCollator()

    def compute_metrics(eval_preds):
        rouge1 = Rouge1()
        rouge2 = Rouge2()
        rougel = RougeL()
        predictions = [x[x != -100] for x in eval_preds.predictions]
        references = [x[x != -100] for x in eval_preds.label_ids]

        rouge1_score = rouge1.score(predictions, references)
        rouge2_score = rouge2.score(predictions, references)
        for pred, ref in zip(predictions, references):
            rougel.add_inst(pred, [ref])
        return {
            "rouge1": rouge1_score,
            "rouge2": rouge2_score,
            "rougel": rougel.score(),
        }

    trainer = GLMTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_dev,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        do_generation=False,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
